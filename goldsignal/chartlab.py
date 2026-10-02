"""看圖工具組 - the common chart-reading tools, drawn automatically on the dashboard chart.

Every tool looks at the bars of the chart's timeframe and returns
  rel   0..1  how useful it is *right now* (e.g. price testing a level, a fresh crossover,
              a divergence, a pattern breakout); tools scoring >= AUTO_MIN pop up by themselves
  why   short reason it popped up
  text  plain-language reading and what to watch
  draw  what to draw: lines, horizontal levels, markers, or an oscillator pane

These are reference tools only. The trade signal is the back-tested strategy (green / red
arrows and the trade ticket); nothing in this module creates or changes a signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .bars import resample

log = logging.getLogger("goldsignal.chartlab")
NY = "America/New_York"

#        key        name           group   default-on
TOOLS = [("sr", "支撐壓力", "main", True),
         ("ma", "均線", "main", True),
         ("trend", "趨勢線／通道", "main", False),
         ("fib", "費波納奇", "main", False),
         ("pattern", "型態", "main", False),
         ("bb", "布林通道", "main", False),
         ("round", "整數關卡", "main", False),
         ("pivot", "樞軸點", "main", False),
         ("session", "亞洲盤區間", "main", False),
         ("vwap", "VWAP", "main", False),
         ("ichimoku", "一目均衡表", "main", False),
         ("sar", "拋物線 SAR", "main", False),
         ("candle", "K 線型態", "main", False),
         ("volume", "成交量", "pane", False),
         ("rsi", "RSI", "pane", False),
         ("macd", "MACD", "pane", False),
         ("kd", "KD", "pane", False)]
TOOL_NAME = {k: n for k, n, _, _ in TOOLS}
DEFAULT_ON = {k for k, _, _, on in TOOLS if on}
AUTO_MIN = 0.7          # relevance needed to pop up by itself
AUTO_MAX = 3            # at most this many tools pop up at once
INTRADAY = ("5m", "15m", "1h")
CFG = {  # pivot half-window, bars scanned for levels, Fibonacci window, pattern window
    "5m": dict(k=6, look=600, fib=288, tri=150),
    "15m": dict(k=5, look=500, fib=192, tri=120),
    "1h": dict(k=4, look=500, fib=120, tri=100),
    "4h": dict(k=3, look=400, fib=90, tri=80),
    "1D": dict(k=3, look=300, fib=120, tri=80),
}
FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
TF_LABEL = {"5m": "5 分 K", "15m": "15 分 K", "1h": "1 小時 K", "4h": "4 小時 K", "1D": "日 K"}


# ----------------------------------------------------------------------------- context

@dataclass
class Ctx:
    bars: pd.DataFrame
    tf: str
    t: list
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    n: int
    atr: float
    price: float
    fmt: object
    bias: int = 0
    pos_dir: int = 0
    stop: float | None = None
    ph: list = field(default_factory=list)
    pl: list = field(default_factory=list)
    regime: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    def pts(self, arr, full: bool = False, digits: int = 5) -> list:
        """[[time, value]] - full=True keeps every bar (None for gaps) so panes stay aligned."""
        a = np.asarray(arr, float)
        if full:
            return [[ti, (round(float(x), digits) if np.isfinite(x) else None)] for ti, x in zip(self.t, a)]
        return [[ti, round(float(x), digits)] for ti, x in zip(self.t, a) if np.isfinite(x)]

    def seg(self, i0: int, v0: float, i1: int, v1: float) -> list:
        return [[self.t[i0], float(v0)], [self.t[i1], float(v1)]]


def pivots(h: np.ndarray, l: np.ndarray, k: int) -> tuple[list[int], list[int]]:
    """Confirmed swing highs/lows: the extreme of a window of k bars on each side."""
    ph, pl = [], []
    for i in range(k, len(h) - k):
        if h[i - k:i + k + 1].argmax() == k:
            ph.append(i)
        if l[i - k:i + k + 1].argmin() == k:
            pl.append(i)
    return ph, pl


def _line(idx, vals):
    idx, vals = np.asarray(idx, float), np.asarray(vals, float)
    if len(idx) == 2:
        s = (vals[1] - vals[0]) / (idx[1] - idx[0])
        return s, vals[0] - s * idx[0], 0.0
    s, b = np.polyfit(idx, vals, 1)
    return s, b, float(np.max(np.abs(np.polyval([s, b], idx) - vals)))


def _crossed(a: np.ndarray, b: np.ndarray, lookback: int) -> int:
    """+1 if a crossed above b within the last `lookback` bars, -1 if below, 0 otherwise."""
    for i in range(len(a) - 1, max(len(a) - 1 - lookback, 0), -1):
        if not (np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(a[i - 1]) and np.isfinite(b[i - 1])):
            continue
        if a[i] > b[i] and a[i - 1] <= b[i - 1]:
            return 1
        if a[i] < b[i] and a[i - 1] >= b[i - 1]:
            return -1
    return 0


def _regime(cx: Ctx) -> dict:
    b = cx.bars
    adx = float(ind.adx(b, 14)["adx"].iloc[-1])
    e20 = float(ind.ema(b["close"], 20).iloc[-1])
    e50 = float(ind.ema(b["close"], 50).iloc[-1])
    bb = ind.bollinger(b["close"], 20, 2.0)
    bw = ((bb["upper"] - bb["lower"]) / bb["mid"]).iloc[-200:].dropna()
    pct = float((bw <= bw.iloc[-1]).mean()) if len(bw) > 20 else 0.5
    c = cx.c[-1]
    if adx >= 25 and c > e50 and e20 > e50:
        key, label = "up", "上升趨勢"
    elif adx >= 25 and c < e50 and e20 < e50:
        key, label = "down", "下降趨勢"
    elif pct <= 0.15 and adx < 25:
        key, label = "squeeze", "波動收斂、可能快要變盤"
    elif adx < 20:
        key, label = "range", "盤整（沒有明顯方向）"
    else:
        key, label = "mixed", "方向不明確"
    suits = {"up": ["ma", "trend", "fib", "sar", "ichimoku"], "down": ["ma", "trend", "fib", "sar", "ichimoku"],
             "range": ["sr", "bb", "kd", "rsi", "pivot"], "squeeze": ["bb", "pattern", "session", "volume"],
             "mixed": ["sr", "ma", "pattern"]}[key]
    return {"key": key, "label": label, "adx": adx, "bb_pct": pct, "suits": suits,
            "text": f"行情判斷（{TF_LABEL.get(cx.tf, cx.tf)}）：{label}（ADX {adx:.0f}）。這種行情比較適合看："
                    + "、".join(TOOL_NAME[k] for k in suits) + "。"}


def _trending(cx: Ctx, d: int | None = None) -> bool:
    k = cx.regime.get("key")
    return k in ("up", "down") if d is None else k == ("up" if d > 0 else "down")


# ----------------------------------------------------------------------------- tools

def t_sr(cx: Ctx) -> dict:
    cfg = CFG.get(cx.tf, CFG["15m"])
    extra = []
    if cx.tf in INTRADAY:
        d1 = resample(cx.bars, "1D")
        if len(d1) >= 2:
            extra = [("昨日高點", float(d1["high"].iloc[-2])), ("昨日低點", float(d1["low"].iloc[-2]))]
    lv = _levels(cx.h, cx.l, cx.ph, cx.pl, max(0, cx.n - cfg["look"]), cx.price, cx.atr, extra)
    cx.cache["levels"] = lv
    if not lv:
        return {"rel": 0.0}
    f = cx.fmt
    res = [x for x in lv if x["kind"] == "resistance"]
    sup = [x for x in lv if x["kind"] == "support"]
    parts = []
    if res:
        parts.append("上方壓力 " + "、".join(f"{f(x['price'])}（{x['label']}）" for x in res))
    if sup:
        parts.append("下方支撐 " + "、".join(f"{f(x['price'])}（{x['label']}）" for x in sup))
    txt = "；".join(parts) + "。"
    near = min(lv, key=lambda x: x["dist_atr"])
    ranging = cx.regime.get("key") == "range"
    rel, why = (0.6 if ranging else 0.4), ("盤整時支撐壓力最好用" if ranging else None)
    if near["dist_atr"] <= 0.4:
        rel = 0.85
        if near["kind"] == "resistance":
            why = f"價格正在測試壓力 {f(near['price'])}"
            txt += f"價格正在測試壓力 {f(near['price'])}：收盤站上去通常會續漲，碰到就掉頭則容易拉回。"
        else:
            why = f"價格正在測試支撐 {f(near['price'])}"
            txt += f"價格正在測試支撐 {f(near['price'])}：守住通常會反彈，收盤跌破則容易再往下。"
    if cx.pos_dir > 0 and res:
        txt += f"持多單的話，接近 {f(res[0]['price'])} 要留意是否過不去。"
    elif cx.pos_dir < 0 and sup:
        txt += f"持空單的話，接近 {f(sup[0]['price'])} 要留意是否跌不破。"
    draw = {"levels": [{"price": x["price"], "title": ("壓力 " if x["kind"] == "resistance" else "支撐 ") + x["label"],
                        "color": "lvl-r" if x["kind"] == "resistance" else "lvl-s", "style": 1} for x in lv]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def _levels(h, l, ph, pl, start, price, atr, extra) -> list[dict]:
    pts = sorted([(h[i], i) for i in ph if i >= start] + [(l[i], i) for i in pl if i >= start])
    tol = 0.35 * atr
    clusters: list[dict] = []
    for p, i in pts:
        if clusters and p - clusters[-1]["prices"][0] <= 2 * tol:   # a level is at most 0.7 ATR wide
            clusters[-1]["prices"].append(p)
            clusters[-1]["idx"].append(i)
        else:
            clusters.append({"prices": [p], "idx": [i]})
    cands = [{"price": float(np.mean(c["prices"])), "touches": len(c["prices"]), "names": []} for c in clusters]
    for name, p in extra:  # yesterday's high / low
        near = [c for c in cands if abs(c["price"] - p) <= tol]
        if near:
            near[0]["names"].append(name)
        else:
            cands.append({"price": float(p), "touches": 1, "names": [name]})
    out = []
    for side in ("resistance", "support"):
        pool = [c for c in cands if (c["price"] > price + 0.1 * atr if side == "resistance" else c["price"] < price - 0.1 * atr)]
        strong = [c for c in pool if c["touches"] >= 2 or c["names"]]
        for c in sorted(strong or pool, key=lambda c: abs(c["price"] - price))[:2]:
            if c["names"]:
                label = "、".join(c["names"])
            elif c["touches"] >= 2:
                label = f"碰過 {c['touches']} 次"
            else:
                label = "前高" if side == "resistance" else "前低"
            out.append({"kind": side, "price": c["price"], "touches": c["touches"], "label": label,
                        "dist_atr": abs(c["price"] - price) / atr})
    return out


def t_ma(cx: Ctx) -> dict:
    close = cx.bars["close"]
    e20, e50, e200 = (ind.ema(close, n).to_numpy() for n in (20, 50, 200))
    f, c = cx.fmt, cx.c[-1]
    draw = {"lines": [{"points": cx.pts(e200), "color": "ma200", "width": 2, "title": "EMA200"}], "markers": []}
    rel, why = 0.4, None
    txt = ""
    if np.isfinite(e50[-1]):
        if e20[-1] > e50[-1] and c > e20[-1]:
            txt = f"多頭排列（價格 > EMA20 {f(e20[-1])} > EMA50 {f(e50[-1])}），回檔到均線附近常有支撐。"
        elif e20[-1] < e50[-1] and c < e20[-1]:
            txt = f"空頭排列（價格 < EMA20 {f(e20[-1])} < EMA50 {f(e50[-1])}），反彈到均線附近常有壓力。"
        else:
            txt = f"均線糾結（EMA20 {f(e20[-1])}、EMA50 {f(e50[-1])}），短線方向不明。"
    if np.isfinite(e200[-1]):
        txt += f"EMA200 在 {f(e200[-1])}，價格在它{'之上' if c > e200[-1] else '之下'}，長線{'偏多' if c > e200[-1] else '偏空'}。"
    x = _crossed(e20, e50, 4)
    if x:
        rel = 0.85
        why = "EMA20 剛" + ("上穿 EMA50（黃金交叉）" if x > 0 else "下穿 EMA50（死亡交叉）")
        txt = why + "，短線趨勢" + ("轉強。" if x > 0 else "轉弱。") + txt
    elif _crossed(cx.c, e200, 3):
        rel = 0.8
        why = "價格剛" + ("站上" if cx.c[-1] > e200[-1] else "跌破") + " EMA200"
    elif _trending(cx) and np.isfinite(e50[-1]) and min(abs(c - e20[-1]), abs(c - e50[-1])) <= 0.3 * cx.atr:
        rel, why = 0.75, "趨勢中價格回到均線附近"
    elif _trending(cx):
        rel = 0.65
    for i in range(max(51, cx.n - 120), cx.n):  # recent crossovers on the chart
        if np.isfinite(e50[i - 1]) and (e20[i] - e50[i]) * (e20[i - 1] - e50[i - 1]) < 0:
            up = e20[i] > e50[i]
            draw["markers"].append({"time": cx.t[i], "position": "belowBar" if up else "aboveBar", "shape": "circle",
                                    "color": "ma200", "text": "黃金交叉" if up else "死亡交叉"})
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def t_trend(cx: Ctx) -> dict:
    rk = cx.regime.get("key")
    d = 1 if rk == "up" else -1 if rk == "down" else (1 if cx.bias > 0 else -1 if cx.bias < 0 else 0)
    if d == 0:
        d = 1 if cx.c[-1] >= cx.c[max(0, cx.n - 50)] else -1
    piv = cx.pl if d > 0 else cx.ph
    src = cx.l if d > 0 else cx.h
    cand = [i for i in piv if i >= cx.n - CFG.get(cx.tf, CFG["15m"])["tri"] * 2][-3:]
    if len(cand) < 2:
        return {"rel": 0.0}
    s, b, resid = _line(cand, src[cand])
    if (d > 0 and s <= 0) or (d < 0 and s >= 0) or resid > 0.5 * cx.atr:
        return {"rel": 0.0}
    line = lambda i: s * i + b  # noqa: E731
    i0 = cand[0]
    body = np.arange(i0, max(i0 + 1, cx.n - 3))
    viol = (cx.c[body] < line(body) - 0.5 * cx.atr) if d > 0 else (cx.c[body] > line(body) + 0.5 * cx.atr)
    if viol.mean() > 0.05:
        return {"rel": 0.0}
    opp = cx.h if d > 0 else cx.l             # channel line through the most extreme opposite point
    rng = np.arange(i0, cx.n)
    off = opp[rng] - line(rng)
    k = int(np.argmax(off) if d > 0 else np.argmin(off))
    chan = lambda i: line(i) + off[k]  # noqa: E731
    now = line(cx.n - 1)
    f = cx.fmt
    name = "上升趨勢線" if d > 0 else "下降趨勢線"
    brk = any(((cx.c[i] < line(i) - 0.1 * cx.atr) if d > 0 else (cx.c[i] > line(i) + 0.1 * cx.atr)) for i in range(cx.n - 3, cx.n))
    dist = abs(cx.c[-1] - now) / cx.atr
    if brk:
        rel, why = 0.85, ("剛跌破" if d > 0 else "剛突破") + name
        txt = f"收盤{'跌破' if d > 0 else '突破'}{name}（約 {f(now)}），原本的{'上漲' if d > 0 else '下跌'}節奏被打破，要小心反轉。"
        if cx.pos_dir == d:
            txt += "你的部位和原趨勢同方向，可考慮先減碼或收緊停損。"
    elif dist <= 0.5:
        rel, why = 0.75, f"價格回到{name}附近"
        txt = f"價格回到{name}（約 {f(now)}）附近，{'守住常是買點、跌破則轉弱' if d > 0 else '壓住常是賣點、突破則轉強'}。"
    else:
        rel, why = (0.55 if _trending(cx, d) else 0.4), None
        txt = f"{name}在 {f(now)}、通道另一邊在 {f(chan(cx.n - 1))}；價格在通道內就是趨勢延續。"
    draw = {"lines": [{"points": cx.seg(i0, line(i0), cx.n - 1, now), "color": "trend", "width": 2, "title": ""},
                      {"points": cx.seg(i0, chan(i0), cx.n - 1, chan(cx.n - 1)), "color": "trend", "width": 1, "style": 2, "title": ""}]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def t_fib(cx: Ctx) -> dict:
    w = CFG.get(cx.tf, CFG["15m"])["fib"]
    a = max(0, cx.n - w)
    hi_i = a + int(np.argmax(cx.h[a:]))
    lo_i = a + int(np.argmin(cx.l[a:]))
    hi, lo = float(cx.h[hi_i]), float(cx.l[lo_i])
    if hi - lo < 3 * cx.atr:
        return {"rel": 0.0}
    up = lo_i < hi_i
    s_i = lo_i if up else hi_i
    rng = hi - lo
    lv = {r: (hi - r * rng) if up else (lo + r * rng) for r in FIB_RATIOS}
    ext = {r: (lo + r * rng) if up else (hi - r * rng) for r in (1.272, 1.618)}
    r = (hi - cx.price) / rng if up else (cx.price - lo) / rng
    f = cx.fmt
    leg, back = ("上漲", "回檔") if up else ("下跌", "反彈")
    d = 1 if up else -1
    if r < 0.15:
        where = f"價格還在{leg}波段的{'高' if up else '低'}檔，還沒有明顯{back}"
    elif r > 1.0:
        where = f"已經完全吃掉這段{leg}，原本的走勢失效"
    else:
        where = f"目前{back}了約 {r * 100:.0f}%"
    txt = (f"最近一段{leg} {f(lo if up else hi)} → {f(hi if up else lo)}，{where}。常見的{back}區在 38.2%–61.8%"
           f"（{f(lv[0.382])}–{f(lv[0.618])}）；超過 78.6%（{f(lv[0.786])}）通常代表這段{leg}結束。")
    rel, why = 0.45, None
    near = min((0.382, 0.5, 0.618), key=lambda q: abs(cx.price - lv[q]))
    if _trending(cx, d) and 0.3 <= r <= 0.7:
        rel, why = 0.85, f"趨勢中的{back}，到了 {near * 100:.1f}% 附近"
        txt += f"這是順著大方向的{back}，{back}守在 61.8% 以內、再{'創新高' if up else '創新低'}的機率比較高。"
    elif abs(cx.price - lv[near]) <= 0.25 * cx.atr and 0.2 < r < 0.85:
        rel, why = 0.75, f"價格正好在 {near * 100:.1f}% 回撤位"
    elif _trending(cx, d) and r < 0.12:
        rel, why = 0.7, f"價格接近{leg}前{'高' if up else '低'}"
        txt += f"如果{'創新高' if up else '創新低'}，延伸目標可參考 127.2%（{f(ext[1.272])}）、161.8%（{f(ext[1.618])}）。"
    if cx.pos_dir == d:
        txt += f"你的部位和這段{leg}同方向：{back}超過 61.8%（{f(lv[0.618])}）可以考慮先減碼或平倉" + \
               (f"，系統停損在 {f(cx.stop)}" if cx.stop is not None else "") + "。"
    elif cx.pos_dir == -d and r < 0.382:
        txt += f"你的部位方向和這段{leg}相反，{back}不到 38.2%（{f(lv[0.382])}）就再{'創新高' if up else '創新低'}的話要特別小心。"
    lines = []
    for q, p in lv.items():
        key = q in (0.382, 0.5, 0.618)
        lines.append({"points": cx.seg(s_i, p, cx.n - 1, p), "color": "fib", "width": 1, "style": 2 if key else 3,
                      "title": f"{q * 100:.1f}%" if (key or q == 0.786) else ""})
    for q, p in ext.items():
        lines.append({"points": cx.seg(s_i, p, cx.n - 1, p), "color": "fib", "width": 1, "style": 1, "title": f"延伸 {q * 100:.1f}%"})
    return {"rel": rel, "why": why, "text": txt, "draw": {"lines": lines}}


def t_bb(cx: Ctx) -> dict:
    bb = ind.bollinger(cx.bars["close"], 20, 2.0)
    up_, mid, lo_ = (bb[k].to_numpy() for k in ("upper", "mid", "lower"))
    if not np.isfinite(mid[-1]):
        return {"rel": 0.0}
    f = cx.fmt
    pct = cx.regime.get("bb_pct", 0.5)
    rel, why = (0.6 if cx.regime.get("key") == "range" else 0.4), None
    txt = f"上軌 {f(up_[-1])}、中軌 {f(mid[-1])}、下軌 {f(lo_[-1])}。"
    out_up = any(cx.c[i] > up_[i] for i in range(cx.n - 2, cx.n))
    out_dn = any(cx.c[i] < lo_[i] for i in range(cx.n - 2, cx.n))
    bw = (up_ - lo_) / mid
    recent_bw = bw[-200:][np.isfinite(bw[-200:])]
    was_tight = len(recent_bw) > 30 and np.nanmin(bw[-10:]) <= np.quantile(recent_bw, 0.15)
    if (out_up or out_dn) and was_tight:
        rel, why = 0.85, "布林通道收窄後" + ("向上突破" if out_up else "向下跌破")
        txt += f"通道收窄一段時間後收盤衝出{'上' if out_up else '下'}軌，常是新一段{'漲' if out_up else '跌'}勢的開始；回到通道內則是假突破。"
    elif pct <= 0.12:
        rel, why = 0.85, "布林通道收窄到近期最窄"
        txt += "通道收得很窄，常是一段大波動的前兆：收盤突破上軌偏多、跌破下軌偏空，方向等突破再確認。"
    elif out_up or out_dn:
        rel, why = (0.72 if cx.regime.get("key") == "range" else 0.6), "收盤衝出布林" + ("上軌" if out_up else "下軌")
        if _trending(cx, 1 if out_up else -1):
            txt += f"沿著{'上' if out_up else '下'}軌走代表趨勢很強，不建議逆勢{'做空' if out_up else '做多'}。"
        else:
            txt += f"盤整中衝出{'上' if out_up else '下'}軌常會被拉回中軌（{f(mid[-1])}）附近。"
    elif cx.regime.get("key") == "range":
        txt += "盤整時價格常在上下軌之間來回，靠近上軌偏向壓力、靠近下軌偏向支撐。"
    draw = {"lines": [{"points": cx.pts(up_), "color": "bb", "width": 1, "title": ""},
                      {"points": cx.pts(mid), "color": "bb", "width": 1, "style": 2, "title": ""},
                      {"points": cx.pts(lo_), "color": "bb", "width": 1, "title": ""}]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def t_round(cx: Ctx) -> dict:
    p = cx.price
    major = 10 ** (np.floor(np.log10(p)) - 1)        # gold ~4,000 -> 100 ; silver ~60 -> 1
    minor = major / 2
    base = np.floor(p / minor) * minor
    levels = [base - minor, base, base + minor, base + 2 * minor]
    f = cx.fmt
    near = min(levels, key=lambda x: abs(x - p))
    is_major = abs(near / major - round(near / major)) < 1e-9
    dist = abs(near - p) / cx.atr
    rel, why = 0.35, None
    if dist <= (0.4 if is_major else 0.25):
        rel, why = (0.78 if is_major else 0.7), f"價格接近整數關卡 {f(near)}"
    txt = f"附近的整數關卡：{'、'.join(f(x) for x in levels)}。整數價位常有很多掛單，第一次碰到容易震盪，站穩或跌破後常會加速。"
    draw = {"levels": [{"price": float(x), "title": "整數 " + f(x), "color": "round", "style": 3} for x in levels]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def _sessions(cx: Ctx) -> np.ndarray:
    if "session" not in cx.cache:
        ny = cx.bars.index.tz_convert(NY)
        shifted = ny.tz_localize(None) + pd.Timedelta(hours=7)
        cx.cache["session"] = np.asarray((shifted.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1))
        cx.cache["ny_h"] = np.asarray(ny.hour + ny.minute / 60.0, float)
    return cx.cache["session"]


def t_pivot(cx: Ctx) -> dict:
    if cx.tf not in INTRADAY:
        return {"rel": 0.0}
    d1 = resample(cx.bars, "1D")
    if len(d1) < 2:
        return {"rel": 0.0}
    H, L, C = (float(d1[k].iloc[-2]) for k in ("high", "low", "close"))
    P = (H + L + C) / 3
    lv = {"P": P, "R1": 2 * P - L, "S1": 2 * P - H, "R2": P + (H - L), "S2": P - (H - L)}
    f = cx.fmt
    name, val = min(lv.items(), key=lambda kv: abs(kv[1] - cx.price))
    rel, why = (0.55 if cx.regime.get("key") == "range" else 0.4), None
    if abs(val - cx.price) <= 0.3 * cx.atr:
        rel, why = 0.72, f"價格在今日樞軸 {name}（{f(val)}）附近"
    side = "之上，今天偏多看待" if cx.price > P else "之下，今天偏空看待"
    txt = (f"今日樞軸 P {f(P)}（多空分界），價格在 P {side}。上方 R1 {f(lv['R1'])}、R2 {f(lv['R2'])}；"
           f"下方 S1 {f(lv['S1'])}、S2 {f(lv['S2'])}。盤整日價格常在 S1–R1 之間來回，衝過 R1／S1 才算有方向。")
    draw = {"levels": [{"price": v, "title": k, "color": "pivot", "style": 2} for k, v in lv.items()]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def t_session(cx: Ctx) -> dict:
    if cx.tf not in ("5m", "15m"):
        return {"rel": 0.0}
    sess = _sessions(cx)
    nyh = cx.cache["ny_h"]
    idx = np.flatnonzero((sess == sess[-1]) & ((nyh >= 19) | (nyh < 3)))
    if len(idx) < 6:
        return {"rel": 0.0}
    top, bot = float(cx.h[idx].max()), float(cx.l[idx].min())
    f = cx.fmt
    now_h = nyh[-1]
    rel, why = 0.35, None
    in_box = bot <= cx.c[-1] <= top
    txt = f"今天亞洲盤區間 {f(bot)}–{f(top)}（{(top - bot) / cx.atr:.1f} 倍 ATR）。"
    if 3 <= now_h < 12:
        brk = 0
        for i in range(cx.n - 3, cx.n):
            if cx.c[i] > top + 0.1 * cx.atr and cx.c[i - 1] <= top + 0.1 * cx.atr:
                brk = 1
            elif cx.c[i] < bot - 0.1 * cx.atr and cx.c[i - 1] >= bot - 0.1 * cx.atr:
                brk = -1
        if brk:
            rel, why = 0.8, "倫敦／紐約盤" + ("突破" if brk > 0 else "跌破") + "亞洲盤區間"
            txt += f"歐美盤{'向上突破' if brk > 0 else '向下跌破'}亞洲盤區間，常會順勢走一段；回到區間內則是假突破。"
        elif in_box and min(top - cx.c[-1], cx.c[-1] - bot) <= 0.3 * cx.atr:
            rel, why = 0.7, "價格貼近亞洲盤區間邊緣"
            txt += "價格貼近區間邊緣，留意歐美盤是否突破。"
        else:
            txt += "歐美盤常會測試這個區間的上緣或下緣。"
    i0 = int(idx[0])
    draw = {"lines": [{"points": cx.seg(i0, top, cx.n - 1, top), "color": "session", "width": 1, "title": "亞洲高"},
                      {"points": cx.seg(i0, bot, cx.n - 1, bot), "color": "session", "width": 1, "title": "亞洲低"}]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def t_vwap(cx: Ctx) -> dict:
    if cx.tf not in ("5m", "15m"):
        return {"rel": 0.0}
    vw = ind.session_vwap(cx.bars, _sessions(cx)).to_numpy()
    if not np.isfinite(vw[-1]):
        return {"rel": 0.0}
    f = cx.fmt
    above = cx.c[-1] > vw[-1]
    rel, why = 0.4, None
    x = _crossed(cx.c, vw, 2)
    nyh = cx.cache["ny_h"][-1]
    if abs(cx.c[-1] - vw[-1]) <= 0.2 * cx.atr and 3 <= nyh < 16 and _trending(cx, 1 if above else -1):
        rel, why = 0.72, "趨勢中價格拉回 VWAP"
    elif x:
        rel, why = 0.6, "價格剛" + ("站上" if x > 0 else "跌破") + " VWAP"
    txt = (f"今日 VWAP（成交量加權平均價）{f(vw[-1])}，價格在它{'之上，今天買方占優' if above else '之下，今天賣方占優'}；"
           f"拉回 VWAP 不破常是短線{'支撐' if above else '壓力'}，穿越 VWAP 代表短線多空換手。")
    return {"rel": rel, "why": why, "text": txt, "draw": {"lines": [{"points": cx.pts(vw), "color": "vwap", "width": 2, "title": "VWAP"}]}}


def t_ichimoku(cx: Ctx) -> dict:
    ich = ind.ichimoku(cx.bars)
    tk, kj, sa, sb = (ich[k].to_numpy() for k in ("tenkan", "kijun", "span_a", "span_b"))
    if not (np.isfinite(sa[-1]) and np.isfinite(sb[-1])):
        return {"rel": 0.0}
    f = cx.fmt
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    c = cx.c[-1]
    pos = "雲帶之上（多方）" if c > top[-1] else "雲帶之下（空方）" if c < bot[-1] else "雲帶之中（方向不明）"
    rel, why = (0.55 if _trending(cx) else 0.4), None
    out = 0
    for i in range(cx.n - 3, cx.n):
        if cx.c[i] > top[i] and cx.c[i - 1] <= top[i - 1]:
            out = 1
        elif cx.c[i] < bot[i] and cx.c[i - 1] >= bot[i - 1]:
            out = -1
    tkx = _crossed(tk, kj, 3)
    if out:
        rel, why = 0.8, "價格剛" + ("突破雲帶上緣" if out > 0 else "跌破雲帶下緣")
    elif tkx:
        rel, why = 0.62, "轉換線" + ("上穿" if tkx > 0 else "下穿") + "基準線"
    txt = (f"價格在{pos}；雲帶 {f(bot[-1])}–{f(top[-1])}，轉換線 {f(tk[-1])}、基準線 {f(kj[-1])}。"
           "價格在雲上、轉換線在基準線上方是多方格局，反之是空方；雲帶本身常是支撐或壓力。")
    draw = {"lines": [{"points": cx.pts(tk), "color": "ich-t", "width": 1, "title": ""},
                      {"points": cx.pts(kj), "color": "ich-k", "width": 1, "title": ""},
                      {"points": cx.pts(sa), "color": "ich-a", "width": 1, "title": "雲A"},
                      {"points": cx.pts(sb), "color": "ich-b", "width": 1, "title": "雲B"}]}
    return {"rel": rel, "why": why, "text": txt, "draw": draw}


def t_sar(cx: Ctx) -> dict:
    ps = ind.parabolic_sar(cx.bars)
    sar, dr = ps["sar"].to_numpy(), ps["direction"].to_numpy()
    if not np.isfinite(sar[-1]):
        return {"rel": 0.0}
    f = cx.fmt
    flip = next((i for i in range(cx.n - 1, cx.n - 4, -1) if dr[i] != dr[i - 1]), None)
    rel, why = 0.4, None
    if flip is not None:
        rel = 0.75 if _trending(cx, int(dr[-1])) else 0.55
        why = "SAR 翻到價格" + ("下方（短線轉多）" if dr[-1] > 0 else "上方（短線轉空）")
    txt = (f"SAR 在價格{'下方 → 短線偏多' if dr[-1] > 0 else '上方 → 短線偏空'}，目前 {f(sar[-1])}。"
           "SAR 會跟著價格移動，可以當短線的移動停損參考；翻到另一邊代表短線轉向。")
    return {"rel": rel, "why": why, "text": txt,
            "draw": {"lines": [{"points": cx.pts(sar), "color": "sar", "width": 1, "dots": True, "title": ""}]}}


def _candle_pattern(cx: Ctx, i: int) -> tuple[str, int] | None:
    o, h, l, c, a = cx.o, cx.h, cx.l, cx.c, cx.atr
    body = abs(c[i] - o[i])
    rng = h[i] - l[i]
    if rng <= 0:
        return None
    up_w, dn_w = h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]
    pb = abs(c[i - 1] - o[i - 1])
    if c[i] > o[i] and c[i - 1] < o[i - 1] and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body >= 0.3 * a and body > pb:
        return "多頭吞噬", 1
    if c[i] < o[i] and c[i - 1] > o[i - 1] and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body >= 0.3 * a and body > pb:
        return "空頭吞噬", -1
    if i >= 2:
        b2 = abs(c[i - 2] - o[i - 2])
        mid2 = (o[i - 2] + c[i - 2]) / 2
        if c[i - 2] < o[i - 2] and b2 >= 0.6 * a and pb <= 0.3 * a and c[i] > o[i] and c[i] > mid2:
            return "晨星", 1
        if c[i - 2] > o[i - 2] and b2 >= 0.6 * a and pb <= 0.3 * a and c[i] < o[i] and c[i] < mid2:
            return "暮星", -1
    if rng >= 0.8 * a and dn_w >= 2 * max(body, 0.05 * a) and up_w <= 0.35 * rng:
        return "錘子線", 1
    if rng >= 0.8 * a and up_w >= 2 * max(body, 0.05 * a) and dn_w <= 0.35 * rng:
        return "流星線", -1
    if rng >= 0.6 * a and body <= 0.1 * rng:
        return "十字線", 0
    return None


def t_candle(cx: Ctx) -> dict:
    marks, latest = [], None
    for i in range(max(2, cx.n - 80), cx.n):
        p = _candle_pattern(cx, i)
        if p:
            name, d = p
            marks.append({"time": cx.t[i], "position": "belowBar" if d > 0 else "aboveBar",
                          "shape": "square" if d == 0 else ("arrowUp" if d > 0 else "arrowDown"), "color": "candle", "text": name})
            if i >= cx.n - 2:
                latest = (name, d, i)
    rel, why, txt = 0.3, None, "最近幾根 K 棒沒有明顯的反轉型態。"
    if latest:
        name, d, i = latest
        levels = cx.cache.get("levels", [])
        at = [x for x in levels if abs(x["price"] - (cx.l[i] if d >= 0 else cx.h[i])) <= 0.3 * cx.atr]
        meaning = {1: "下方有買盤撐住、可能止跌", -1: "上方有賣壓、可能止漲", 0: "多空拉鋸、可能變盤"}[d]
        txt = f"剛出現「{name}」（{meaning}）。"
        if at and d != 0 and (d > 0) == (at[0]["kind"] == "support"):
            where = "支撐" if at[0]["kind"] == "support" else "壓力"
            rel, why = 0.78, f"在{where} {cx.fmt(at[0]['price'])} 出現{name}"
            txt += f"而且剛好出現在{where} {cx.fmt(at[0]['price'])}，可信度比較高。"
        else:
            rel = 0.5
            txt += "但不在重要價位上，單看 K 線型態參考性有限。"
    return {"rel": rel, "why": why, "text": txt, "draw": {"markers": marks}}


def t_volume(cx: Ctx) -> dict:
    v = cx.v
    if not np.isfinite(v).any() or np.nansum(v) <= 0:
        return {"rel": 0.0}
    avg = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    rel, why = 0.35, None
    ratio = v[-1] / avg[-1] if np.isfinite(avg[-1]) and avg[-1] > 0 else 1.0
    if ratio >= 2.0 and abs(cx.c[-1] - cx.o[-1]) >= 0.8 * cx.atr:
        rel, why = 0.75, "爆量長 K"
        txt = f"最新一根成交量是平常的 {ratio:.1f} 倍、而且是長{'紅' if cx.c[-1] > cx.o[-1] else '黑'} K：有大量資金進出，突破比較可信。"
    else:
        txt = f"最新一根成交量約是 20 根平均的 {ratio:.1f} 倍。突破時有放量比較可信，縮量突破容易是假突破（這裡的成交量是報價筆數，僅供參考）。"
    colors = ["up" if cx.c[i] >= cx.o[i] else "down" for i in range(cx.n)]
    pane = {"title": "成交量", "series": [{"type": "hist", "points": cx.pts(v, full=True, digits=1), "colors": colors},
                                         {"type": "line", "points": cx.pts(avg, full=True, digits=1), "color": "pane2"}], "levels": []}
    return {"rel": rel, "why": why, "text": txt, "draw": {"pane": pane}}


def _divergence(cx: Ctx, osc: np.ndarray, gap: float) -> tuple[int, int, int] | None:
    """(+1 bullish / -1 bearish, pivot1, pivot2) when price and the oscillator disagree at the last two swings."""
    ph, pl = pivots(cx.h, cx.l, 3)
    for kind, piv, src in ((-1, ph, cx.h), (1, pl, cx.l)):
        rec = [i for i in piv if i >= cx.n - 60]
        if len(rec) < 2:
            continue
        i1, i2 = rec[-2], rec[-1]
        if i2 < cx.n - 12 or i2 - i1 < 5 or not (np.isfinite(osc[i1]) and np.isfinite(osc[i2])):
            continue
        if kind < 0 and src[i2] > src[i1] and osc[i2] < osc[i1] - gap:
            return -1, i1, i2
        if kind > 0 and src[i2] < src[i1] and osc[i2] > osc[i1] + gap:
            return 1, i1, i2
    return None


def _div_draw(cx: Ctx, dv, osc) -> tuple[list, list, str]:
    d, i1, i2 = dv
    src = cx.l if d > 0 else cx.h
    word = "底背離（價格創新低、指標沒有）→ 跌勢可能減弱" if d > 0 else "頂背離（價格創新高、指標沒有）→ 漲勢可能減弱"
    main = [{"points": cx.seg(i1, src[i1], i2, src[i2]), "color": "div", "width": 2, "title": ""}]
    pane = [{"points": cx.seg(i1, osc[i1], i2, osc[i2]), "color": "div", "width": 2}]
    return main, pane, word


def t_rsi(cx: Ctx) -> dict:
    r = ind.rsi(cx.bars["close"], 14).to_numpy()
    if not np.isfinite(r[-1]):
        return {"rel": 0.0}
    rel, why = 0.35, None
    txt = f"RSI = {r[-1]:.0f}。"
    main_lines, pane_lines = [], []
    dv = _divergence(cx, r, 3.0)
    if dv:
        main_lines, pane_lines, word = _div_draw(cx, dv, r)
        rel, why = 0.85, "RSI " + word.split("（")[0]
        txt += f"出現{word}，{'持空單要小心反彈' if dv[0] > 0 else '持多單要小心拉回'}。"
    elif r[-1] >= 70 or r[-1] <= 30:
        hot = r[-1] >= 70
        rel = 0.78 if cx.regime.get("key") == "range" else 0.62
        why = "RSI " + ("超買（≥70）" if hot else "超賣（≤30）")
        txt += (f"{'超買' if hot else '超賣'}。盤整行情容易{'回落' if hot else '反彈'}；"
                f"但強勢趨勢中 RSI 可以一直{'高' if hot else '低'}檔鈍化，不要只因為{'超買就做空' if hot else '超賣就做多'}。")
    else:
        txt += "在 30–70 之間，沒有過熱或過冷。"
    pane = {"title": "RSI(14)", "series": [{"type": "line", "points": cx.pts(r, full=True, digits=1), "color": "rsi"}],
            "levels": [70, 50, 30], "lines": pane_lines}
    return {"rel": rel, "why": why, "text": txt, "draw": {"pane": pane, "lines": main_lines}}


def t_macd(cx: Ctx) -> dict:
    m = ind.macd(cx.bars["close"])
    line, sig, hist = (m[k].to_numpy() for k in ("macd", "signal", "hist"))
    if not np.isfinite(sig[-1]):
        return {"rel": 0.0}
    rel, why = 0.35, None
    x = _crossed(line, sig, 3)
    z = _crossed(line, np.zeros(cx.n), 3)
    txt = f"MACD {'在零軸之上（多方）' if line[-1] > 0 else '在零軸之下（空方）'}，柱狀體{'變長' if abs(hist[-1]) > abs(hist[-2]) else '縮短'}。"
    main_lines, pane_lines = [], []
    dv = _divergence(cx, line, 0.1 * cx.atr)
    if dv:
        main_lines, pane_lines, word = _div_draw(cx, dv, line)
        rel, why = 0.85, "MACD " + word.split("（")[0]
        txt += f"出現{word}。"
    elif x:
        rel, why = 0.62, "MACD " + ("黃金交叉" if x > 0 else "死亡交叉")
        txt += f"MACD 線剛{'上穿' if x > 0 else '下穿'}訊號線，短線動能轉{'強' if x > 0 else '弱'}" + \
               ("（在零軸上方的黃金交叉更可信）。" if x > 0 and line[-1] > 0 else "（在零軸下方的死亡交叉更可信）。" if x < 0 and line[-1] < 0 else "。")
    elif z:
        rel, why = 0.6, "MACD 穿越零軸"
    colors = ["up" if hv >= 0 else "down" for hv in np.nan_to_num(hist)]
    pane = {"title": "MACD(12,26,9)", "series": [{"type": "hist", "points": cx.pts(hist, full=True), "colors": colors},
                                                 {"type": "line", "points": cx.pts(line, full=True), "color": "macd"},
                                                 {"type": "line", "points": cx.pts(sig, full=True), "color": "pane2"}],
            "levels": [0], "lines": pane_lines}
    return {"rel": rel, "why": why, "text": txt, "draw": {"pane": pane, "lines": main_lines}}


def t_kd(cx: Ctx) -> dict:
    kd = ind.stoch_kd(cx.bars, 9)
    k, d = kd["k"].to_numpy(), kd["d"].to_numpy()
    if not np.isfinite(k[-1]):
        return {"rel": 0.0}
    rel, why = 0.35, None
    x = _crossed(k, d, 2)
    txt = f"K = {k[-1]:.0f}、D = {d[-1]:.0f}。"
    if x > 0 and min(k[-1], d[-1]) <= 25:
        rel, why = 0.8, "KD 低檔黃金交叉"
        txt += "K 在 20 附近由下往上穿過 D（低檔黃金交叉），短線可能反彈。"
    elif x < 0 and max(k[-1], d[-1]) >= 75:
        rel, why = 0.8, "KD 高檔死亡交叉"
        txt += "K 在 80 附近由上往下穿過 D（高檔死亡交叉），短線可能回落。"
    elif k[-1] >= 80 or k[-1] <= 20:
        rel = 0.6
        txt += f"在{'高' if k[-1] >= 80 else '低'}檔區；強勢行情 KD 會鈍化，要等交叉再看。"
    else:
        txt += "在中間區域，沒有明確訊號。"
    pane = {"title": "KD(9,3,3)", "series": [{"type": "line", "points": cx.pts(k, full=True, digits=1), "color": "kd-k"},
                                             {"type": "line", "points": cx.pts(d, full=True, digits=1), "color": "kd-d"}],
            "levels": [80, 50, 20]}
    return {"rel": rel, "why": why, "text": txt, "draw": {"pane": pane}}


# ----------------------------------------------------------------------------- patterns

def _p_triangle(cx: Ctx) -> dict | None:
    cfg = CFG.get(cx.tf, CFG["15m"])
    ph, pl = pivots(cx.h, cx.l, max(2, cfg["k"] - 2))
    a, n, atr = cx.n - cfg["tri"], cx.n, cx.atr
    his = [i for i in ph if i >= a][-3:]
    los = [i for i in pl if i >= a][-3:]
    if len(his) < 2 or len(los) < 2:
        return None
    sh, bh, rh = _line(his, cx.h[his])
    sl, bl, rl = _line(los, cx.l[los])
    if max(rh, rl) > 0.6 * atr:
        return None
    start = min(his[0], los[0])
    span = (n - 1) - start
    if span < 15:
        return None
    dh, dl = sh * span / atr, sl * span / atr          # total drift of each line, in ATRs
    if dh <= -0.5 and dl >= 0.5:
        kind, name = "sym", "收斂三角形"
    elif abs(dh) < 0.5 and dl >= 0.8:
        kind, name = "asc", "上升三角形"
    elif dh <= -0.8 and abs(dl) < 0.5:
        kind, name = "desc", "下降三角形"
    else:
        return None
    up = lambda i: sh * i + bh  # noqa: E731
    lo = lambda i: sl * i + bl  # noqa: E731
    if up(start) - lo(start) < 1.5 * atr or up(n - 1) - lo(n - 1) < 0.2 * atr:
        return None
    body = np.arange(start, max(start + 1, n - 3))
    inside = (cx.c[body] <= up(body) + 0.4 * atr) & (cx.c[body] >= lo(body) - 0.4 * atr)
    if inside.mean() < 0.9:
        return None
    brk, bi = None, None
    for i in range(max(start + 1, n - 3), n):
        if cx.c[i] > up(i) + 0.1 * atr and cx.c[i - 1] <= up(i - 1) + 0.1 * atr:
            brk, bi = 1, i
        elif cx.c[i] < lo(i) - 0.1 * atr and cx.c[i - 1] >= lo(i - 1) - 0.1 * atr:
            brk, bi = -1, i
    height = up(start) - lo(start)
    hint = {"asc": "上升三角形較常往上突破", "desc": "下降三角形較常往下跌破", "sym": "收斂三角形方向未定"}[kind]
    if brk:
        txt = f"{'向上突破' if brk > 0 else '向下跌破'}{name}，{'偏多' if brk > 0 else '偏空'}；量測目標約 {cx.fmt(cx.c[bi] + brk * height)}（三角形開口高度）。"
    else:
        txt = (f"正在形成{name}：上緣約 {cx.fmt(up(n - 1))}、下緣約 {cx.fmt(lo(n - 1))}，越收越窄；"
               f"收盤突破上緣偏多、跌破下緣偏空（{hint}）。")
    return {"name": name, "type": kind, "brk": brk, "bi": bi, "end": n - 1, "text": txt,
            "upper_now": up(n - 1), "lower_now": lo(n - 1),
            "lines": [{"points": cx.seg(his[0], up(his[0]), n - 1, up(n - 1)), "color": "pattern", "width": 2, "title": ""},
                      {"points": cx.seg(los[0], lo(los[0]), n - 1, lo(n - 1)), "color": "pattern", "width": 2, "title": ""}]}


def _p_double(cx: Ctx) -> dict | None:
    w = int(CFG.get(cx.tf, CFG["15m"])["tri"] * 1.5)
    atr, n = cx.atr, cx.n
    best = None
    for d in (-1, 1):                       # -1 = double top (M), +1 = double bottom (W)
        piv = cx.ph if d < 0 else cx.pl
        src = cx.h if d < 0 else cx.l
        rec = [i for i in piv if i >= n - w]
        if len(rec) < 2:
            continue
        i1, i2 = rec[-2], rec[-1]
        if i2 - i1 < 8 or abs(src[i1] - src[i2]) > 0.6 * atr or i2 < n - 40:
            continue
        mid = slice(i1, i2 + 1)
        neck = float(cx.l[mid].min()) if d < 0 else float(cx.h[mid].max())
        depth = (min(src[i1], src[i2]) - neck) if d < 0 else (neck - max(src[i1], src[i2]))
        if depth < 2.5 * atr:
            continue
        if i2 + 1 < n and ((d < 0 and cx.h[i2 + 1:].max() > max(src[i1], src[i2]) + 0.3 * atr) or
                           (d > 0 and cx.l[i2 + 1:].min() < min(src[i1], src[i2]) - 0.3 * atr)):
            continue
        brk, bi = None, None
        for i in range(max(i2 + 1, n - 3), n):
            if d < 0 and cx.c[i] < neck - 0.1 * atr and cx.c[i - 1] >= neck - 0.1 * atr:
                brk, bi = -1, i
            if d > 0 and cx.c[i] > neck + 0.1 * atr and cx.c[i - 1] <= neck + 0.1 * atr:
                brk, bi = 1, i
        name = "M 頭（雙頂）" if d < 0 else "W 底（雙底）"
        if brk:
            txt = f"{name}確認：收盤{'跌破' if d < 0 else '突破'}頸線 {cx.fmt(neck)}，{'偏空' if d < 0 else '偏多'}；量測目標約 {cx.fmt(neck + d * depth)}。"
        else:
            txt = (f"可能形成{name}：兩次在 {cx.fmt(src[i2])} 附近{'過不去' if d < 0 else '跌不破'}，頸線 {cx.fmt(neck)}；"
                   f"收盤{'跌破' if d < 0 else '突破'}頸線才算確認，在那之前只是可能。")
        pat = {"name": name, "brk": brk, "bi": bi, "end": i2, "text": txt,
               "lines": [{"points": cx.seg(i1, src[i1], i2, src[i2]), "color": "pattern", "width": 2, "title": ""},
                         {"points": cx.seg(i1, neck, n - 1, neck), "color": "pattern", "width": 1, "style": 2, "title": "頸線"}]}
        if best is None or pat["end"] > best["end"]:
            best = pat
    return best


def _p_hs(cx: Ctx) -> dict | None:
    w = int(CFG.get(cx.tf, CFG["15m"])["tri"] * 2)
    atr, n = cx.atr, cx.n
    for d in (-1, 1):                       # -1 = head & shoulders top, +1 = inverse
        piv = cx.ph if d < 0 else cx.pl
        src = cx.h if d < 0 else cx.l
        rec = [i for i in piv if i >= n - w]
        if len(rec) < 3:
            continue
        i1, i2, i3 = rec[-3:]
        s1, hd, s3 = src[i1], src[i2], src[i3]
        if i3 < n - 30 or abs(s1 - s3) > 1.0 * atr:
            continue
        if (d < 0 and hd < max(s1, s3) + 0.8 * atr) or (d > 0 and hd > min(s1, s3) - 0.8 * atr):
            continue
        t1 = i1 + int(np.argmin(cx.l[i1:i2 + 1]) if d < 0 else np.argmax(cx.h[i1:i2 + 1]))
        t2 = i2 + int(np.argmin(cx.l[i2:i3 + 1]) if d < 0 else np.argmax(cx.h[i2:i3 + 1]))
        if t2 == t1:
            continue
        y1 = cx.l[t1] if d < 0 else cx.h[t1]
        y2 = cx.l[t2] if d < 0 else cx.h[t2]
        s = (y2 - y1) / (t2 - t1)
        neck = lambda i: y1 + s * (i - t1)  # noqa: E731
        if i3 + 1 < n and ((d < 0 and cx.h[i3 + 1:].max() > s3 + 0.5 * atr) or (d > 0 and cx.l[i3 + 1:].min() < s3 - 0.5 * atr)):
            continue
        brk, bi = None, None
        for i in range(max(i3 + 1, n - 3), n):
            if d < 0 and cx.c[i] < neck(i) - 0.1 * atr and cx.c[i - 1] >= neck(i - 1) - 0.1 * atr:
                brk, bi = -1, i
            if d > 0 and cx.c[i] > neck(i) + 0.1 * atr and cx.c[i - 1] <= neck(i - 1) + 0.1 * atr:
                brk, bi = 1, i
        name = "頭肩頂" if d < 0 else "頭肩底"
        height = abs(hd - neck(i2))
        if brk:
            txt = f"{name}確認：收盤{'跌破' if d < 0 else '突破'}頸線（約 {cx.fmt(neck(n - 1))}），{'偏空' if d < 0 else '偏多'}；量測目標約 {cx.fmt(neck(n - 1) + d * height)}。"
        else:
            txt = f"可能形成{name}，頸線約 {cx.fmt(neck(n - 1))}；收盤{'跌破' if d < 0 else '突破'}頸線才算確認。"
        return {"name": name, "brk": brk, "bi": bi, "end": i3, "text": txt,
                "lines": [{"points": [[cx.t[i1], float(s1)], [cx.t[i2], float(hd)], [cx.t[i3], float(s3)]], "color": "pattern", "width": 2, "title": ""},
                          {"points": cx.seg(t1, y1, n - 1, neck(n - 1)), "color": "pattern", "width": 1, "style": 2, "title": "頸線"}]}
    return None


def _p_box(cx: Ctx) -> dict | None:
    atr, n = cx.atr, cx.n
    w = CFG.get(cx.tf, CFG["15m"])["tri"]
    for N in range(w, 23, -4):
        a, b = n - 3 - N, n - 3
        if a < 0:
            continue
        top, bot = float(cx.h[a:b].max()), float(cx.l[a:b].min())
        hgt = top - bot
        if not (1.5 * atr <= hgt <= 4.0 * atr):
            continue
        tops = [i for i in cx.ph if a <= i < b and cx.h[i] >= top - 0.4 * atr]
        bots = [i for i in cx.pl if a <= i < b and cx.l[i] <= bot + 0.4 * atr]
        if len(tops) < 2 or len(bots) < 2:
            continue
        brk, bi = None, None
        for i in range(b, n):
            if cx.c[i] > top + 0.1 * atr and cx.c[i - 1] <= top + 0.1 * atr:
                brk, bi = 1, i
            elif cx.c[i] < bot - 0.1 * atr and cx.c[i - 1] >= bot - 0.1 * atr:
                brk, bi = -1, i
        if brk:
            txt = f"{'向上突破' if brk > 0 else '向下跌破'}箱型整理（{cx.fmt(bot)}–{cx.fmt(top)}），量測目標約 {cx.fmt((top if brk > 0 else bot) + brk * hgt)}。"
        else:
            txt = f"箱型整理中：{cx.fmt(bot)}–{cx.fmt(top)}（上緣碰 {len(tops)} 次、下緣碰 {len(bots)} 次）；收盤突破上緣偏多、跌破下緣偏空，箱內來回操作風險高。"
        return {"name": "箱型整理", "brk": brk, "bi": bi, "end": b, "text": txt,
                "lines": [{"points": cx.seg(a, top, n - 1, top), "color": "pattern", "width": 1, "title": "箱頂"},
                          {"points": cx.seg(a, bot, n - 1, bot), "color": "pattern", "width": 1, "title": "箱底"}]}
    return None


def _p_flag(cx: Ctx) -> dict | None:
    atr, n = cx.atr, cx.n
    seg = slice(max(0, n - 40), n - 5)
    if seg.start >= seg.stop:
        return None
    for d in (1, -1):
        p = seg.start + int(np.argmax(cx.h[seg]) if d > 0 else np.argmin(cx.l[seg]))
        base = cx.l[max(0, p - 12):p + 1].min() if d > 0 else cx.h[max(0, p - 12):p + 1].max()
        pole = (cx.h[p] - base) if d > 0 else (base - cx.l[p])
        if pole < 4 * atr or n - 1 - p < 5:
            continue
        cons = slice(p + 1, n - 1)
        if d > 0:
            retr, edge, other = cx.h[p] - cx.l[cons].min(), cx.h[cons].max(), cx.l[cons].min()
        else:
            retr, edge, other = cx.h[cons].max() - cx.l[p], cx.l[cons].min(), cx.h[cons].max()
        if retr > 0.5 * pole or retr < 0.15 * pole:
            continue
        brk = (cx.c[-1] > edge + 0.1 * atr) if d > 0 else (cx.c[-1] < edge - 0.1 * atr)
        name = "多頭旗形" if d > 0 else "空頭旗形"
        if brk:
            txt = f"{name}突破：急{'漲' if d > 0 else '跌'}後小幅整理，現在{'突破' if d > 0 else '跌破'}整理區，常會再走一段（量測目標約 {cx.fmt(cx.c[-1] + d * pole)}）。"
        else:
            txt = f"可能是{name}：先急{'漲' if d > 0 else '跌'} {pole / atr:.1f} 倍 ATR，再小幅{'回檔' if d > 0 else '反彈'}整理；收盤{'突破' if d > 0 else '跌破'} {cx.fmt(edge)} 偏{'多' if d > 0 else '空'}。"
        return {"name": name, "brk": d if brk else None, "bi": n - 1 if brk else None, "end": n - 2, "text": txt,
                "lines": [{"points": cx.seg(p + 1, edge, n - 1, edge), "color": "pattern", "width": 1, "title": "旗頂" if d > 0 else "旗底"},
                          {"points": cx.seg(p + 1, other, n - 1, other), "color": "pattern", "width": 1, "style": 2, "title": ""}]}
    return None


def t_pattern(cx: Ctx) -> dict:
    found = []
    for fn in (_p_hs, _p_double, _p_triangle, _p_flag, _p_box):
        try:
            p = fn(cx)
        except Exception as e:  # a detector must never break the chart
            log.debug("pattern %s failed: %s", fn.__name__, e)
            p = None
        if p:
            found.append(p)
    if not found:
        return {"rel": 0.0}
    found.sort(key=lambda p: (p["brk"] is not None, p["end"]), reverse=True)
    pick = found[:2]
    main = pick[0]
    cx.cache["patterns"] = pick
    forming_rel = {"多頭旗形": 0.55, "空頭旗形": 0.55, "箱型整理": 0.7, "M 頭（雙頂）": 0.72, "W 底（雙底）": 0.72,
                   "頭肩頂": 0.76, "頭肩底": 0.76}
    if main["brk"]:
        rel, why = 0.9, f"{main['name']}{'突破' if main['brk'] > 0 else '跌破'}"
    else:
        rel, why = forming_rel.get(main["name"], 0.75), f"正在形成{main['name']}"
    texts, lines, marks = [], [], []
    for p in pick:
        t = p["text"]
        if p["brk"]:
            if (p["brk"] > 0 and cx.bias > 0) or (p["brk"] < 0 and cx.bias < 0):
                t += "和日線方向一致，比較可信。"
            elif cx.bias:
                t += "但和日線方向相反，小心假突破。"
            if cx.pos_dir and (p["brk"] > 0) != (cx.pos_dir > 0):
                t += "⚠ 和你的部位方向相反，確認停損有設好。"
            marks.append({"time": cx.t[p["bi"]], "position": "belowBar" if p["brk"] > 0 else "aboveBar", "shape": "circle",
                          "color": "pattern", "text": ("突破" if p["brk"] > 0 else "跌破") + p["name"].split("（")[0]})
        texts.append(t)
        lines += p["lines"]
    return {"rel": rel, "why": why, "text": "".join(texts), "draw": {"lines": lines, "markers": marks}}


REGISTRY = {"sr": t_sr, "ma": t_ma, "trend": t_trend, "fib": t_fib, "pattern": t_pattern, "bb": t_bb,
            "round": t_round, "pivot": t_pivot, "session": t_session, "vwap": t_vwap, "ichimoku": t_ichimoku,
            "sar": t_sar, "candle": t_candle, "volume": t_volume, "rsi": t_rsi, "macd": t_macd, "kd": t_kd}


# ----------------------------------------------------------------------------- entry point

def build(bars: pd.DataFrame, tf: str, price: float | None, core: dict, fmt, inst_name: str,
          pos_dir: int = 0, stop: float | None = None, show: set | None = None, auto: bool = True) -> dict:
    """Run every tool, decide which ones pop up, and return what the dashboard draws and says."""
    head = _headline(core, fmt, inst_name)
    note = ("綠色▲／紅色▼箭頭與交易單是回測驗證過的系統訊號；其他看圖工具（標 ★ 的是依目前行情自動跳出的）"
            "只當輔助參考，不會單獨產生訊號。")
    meta = [{"key": k, "name": nm, "group": g, "rel": 0.0, "why": None, "auto": False, "visible": False} for k, nm, g, _ in TOOLS]
    empty = {"headline": head, "regime": None, "tools": meta, "draw": {}, "lines": [], "note": note}
    if len(bars) < 60:
        return empty
    h, l, c, o = (bars[k].to_numpy(float) for k in ("high", "low", "close", "open"))
    v = bars["volume"].to_numpy(float) if "volume" in bars else np.full(len(bars), np.nan)
    atr = float(ind.atr(bars, 14).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return empty
    t = ((bars.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).tolist()
    ph, pl = pivots(h, l, CFG.get(tf, CFG["15m"])["k"])
    cx = Ctx(bars=bars, tf=tf, t=t, o=o, h=h, l=l, c=c, v=v, n=len(bars), atr=atr,
             price=float(c[-1]) if price is None else float(price), fmt=fmt,
             bias=int((core or {}).get("bias") or 0), pos_dir=pos_dir, stop=stop, ph=ph, pl=pl)
    cx.regime = _regime(cx)
    res = {}
    for key, fn in REGISTRY.items():  # sr runs first: candle patterns look at its levels
        try:
            res[key] = fn(cx) or {"rel": 0.0}
        except Exception as e:
            log.warning("chart tool %s failed: %s", key, e)
            res[key] = {"rel": 0.0}
    ranked = sorted((k for k in res if res[k].get("rel", 0) >= AUTO_MIN and res[k].get("why")),
                    key=lambda k: -res[k]["rel"])[:AUTO_MAX]
    chosen = set(DEFAULT_ON if show is None else show)
    visible = {k for k in chosen | (set(ranked) if auto else set()) if res.get(k, {}).get("draw")}
    for m in meta:
        r = res.get(m["key"], {})
        m.update(rel=round(float(r.get("rel", 0.0)), 2), why=r.get("why"), auto=m["key"] in ranked and auto,
                 visible=m["key"] in visible)
    order = [k for k in ranked if k in visible] + [k for k, _, _, _ in TOOLS if k in visible and k not in ranked]
    lines = [{"tool": k, "name": TOOL_NAME[k], "auto": auto and k in ranked, "why": res[k].get("why"), "text": res[k]["text"]}
             for k in order if res[k].get("text")]
    return {"headline": head, "regime": cx.regime, "tools": meta, "draw": {k: res[k]["draw"] for k in visible},
            "lines": lines, "note": note}


def _headline(core: dict, fmt, name: str) -> dict:
    if not core:
        return {"dir": 0, "kind": "wait", "title": "資料準備中", "text": ""}
    st = core.get("state")
    if st in ("signal_long", "signal_short"):
        ns = core["new_signal"]
        d = ns["dir"]
        return {"dir": d, "kind": "signal", "title": f"現在可以{'做多' if d > 0 else '做空'}",
                "text": f"{name}{'做多' if d > 0 else '做空'}訊號成立：參考價 {fmt(ns['ref_price'])}，停損 {fmt(ns['stop'])}。"
                        f"價格超過 {fmt(ns['max_chase'])} 就不要追。"}
    if st in ("long", "short"):
        a = core["active"]
        d = a["dir"]
        return {"dir": d, "kind": "hold", "title": f"系統持有{'多' if d > 0 else '空'}單",
                "text": f"停損（移動）在 {fmt(a['stop'])}，碰到就出場；不要加碼、不要把停損拉遠。"}
    if core.get("cooldown_until"):
        return {"dir": 0, "kind": "wait", "title": "剛出場，冷卻中", "text": "1 小時內不出新訊號。"}
    bias = core.get("bias", 0)
    side = "long" if bias > 0 else "short" if bias < 0 else None
    setup = (core.get("setup") or {}).get(side) if side else None
    if setup and setup.get("ready"):
        d = 1 if side == "long" else -1
        return {"dir": d, "kind": "ready", "title": f"偏{'多' if d > 0 else '空'}，等{'突破' if d > 0 else '跌破'}",
                "text": f"大方向一致向{'上' if d > 0 else '下'}；15 分 K 收盤{'站上' if d > 0 else '跌破'} {fmt(setup['trigger'])} 才{'做多' if d > 0 else '做空'}，在那之前先不要進場。"}
    if setup and setup.get("missing") == ["squeeze"]:
        return {"dir": 0, "kind": "wait", "title": "觀望（波動還沒收斂）",
                "text": f"日線偏{'多' if bias > 0 else '空'}、趨勢條件都到齊，但最近 4 小時 15 分 K 的布林通道沒有收窄過；"
                        "回測顯示這種「波動已經放大後才追的突破」平均幾乎不賺，所以系統先不進場。"}
    reason = "日線沒有明確方向（盤整）" if bias == 0 else f"日線偏{'多' if bias > 0 else '空'}，但其他條件還沒到齊"
    return {"dir": 0, "kind": "wait", "title": "觀望", "text": f"{reason}，現在不要進場。"}
