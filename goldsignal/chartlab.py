"""Chart tools drawn on the dashboard (reference only - NOT part of the back-tested signal):

* support / resistance  - clusters of swing highs/lows that price touched several times,
                          plus yesterday's high/low on intraday charts
* Fibonacci retracement - on the biggest move of the recent window (auto-drawn)
* triangles             - converging trendlines through recent swing highs and lows
                          (symmetrical / ascending / descending) and their breakouts

`analyze` works on any timeframe's OHLC bars (UTC index); `describe` turns the result and
the strategy state into plain-language lines for the "現在的情況" panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .bars import resample

CFG = {  # pivot half-window, bars scanned for levels, Fibonacci window, triangle window
    "5m": dict(k=6, look=600, fib=288, tri=150),
    "15m": dict(k=5, look=500, fib=192, tri=120),
    "1h": dict(k=4, look=500, fib=120, tri=100),
    "4h": dict(k=3, look=400, fib=90, tri=80),
    "1D": dict(k=3, look=300, fib=120, tri=80),
}
FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
TRI_NAMES = {"sym": "收斂三角形", "asc": "上升三角形", "desc": "下降三角形"}


def pivots(h: np.ndarray, l: np.ndarray, k: int) -> tuple[list[int], list[int]]:
    """Confirmed swing highs/lows: the extreme of a window of k bars on each side."""
    ph, pl = [], []
    for i in range(k, len(h) - k):
        wh, wl = h[i - k:i + k + 1], l[i - k:i + k + 1]
        if wh.argmax() == k:
            ph.append(i)
        if wl.argmin() == k:
            pl.append(i)
    return ph, pl


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
    cands = [{"price": float(np.mean(c["prices"])), "touches": len(c["prices"]), "last_i": max(c["idx"]), "names": []}
             for c in clusters]
    for name, p in extra:  # yesterday's high / low
        near = [c for c in cands if abs(c["price"] - p) <= tol]
        if near:
            near[0]["names"].append(name)
        else:
            cands.append({"price": float(p), "touches": 1, "last_i": -1, "names": [name]})
    out = []
    for side in ("resistance", "support"):
        pool = [c for c in cands if (c["price"] > price + 0.1 * atr if side == "resistance" else c["price"] < price - 0.1 * atr)]
        strong = [c for c in pool if c["touches"] >= 2 or c["names"]]
        pick = sorted(strong or pool, key=lambda c: abs(c["price"] - price))[:2]
        for c in pick:
            if c["names"]:
                label = "、".join(c["names"])
            elif c["touches"] >= 2:
                label = f"碰過 {c['touches']} 次"
            else:
                label = "前高" if side == "resistance" else "前低"
            out.append({"kind": side, "price": c["price"], "touches": c["touches"], "label": label,
                        "dist_atr": abs(c["price"] - price) / atr})
    return out


def _fib(h, l, n, window, atr, price) -> dict | None:
    a = max(0, n - window)
    hi_i = a + int(np.argmax(h[a:]))
    lo_i = a + int(np.argmin(l[a:]))
    hi, lo = float(h[hi_i]), float(l[lo_i])
    if hi - lo < 3 * atr:
        return None
    up = lo_i < hi_i                       # the move went from the low to the high
    start_i, end_i = (lo_i, hi_i) if up else (hi_i, lo_i)
    rng = hi - lo
    lv = [{"ratio": r, "price": (hi - r * rng) if up else (lo + r * rng)} for r in FIB_RATIOS]
    retr = (hi - price) / rng if up else (price - lo) / rng
    return {"dir": 1 if up else -1, "start_i": start_i, "end_i": end_i, "high": hi, "low": lo,
            "levels": lv, "retrace": float(retr)}


def _line(idx, vals):
    if len(idx) == 2:
        s = (vals[1] - vals[0]) / (idx[1] - idx[0])
        return s, vals[0] - s * idx[0], 0.0
    s, b = np.polyfit(idx, vals, 1)
    return s, b, float(np.max(np.abs(np.polyval([s, b], idx) - vals)))


def _triangle(h, l, c, n, k, window, atr) -> dict | None:
    ph, pl = pivots(h, l, max(2, k - 2))
    a = n - window
    his = [i for i in ph if i >= a][-3:]
    los = [i for i in pl if i >= a][-3:]
    if len(his) < 2 or len(los) < 2:
        return None
    sh, bh, rh = _line(np.array(his, float), h[his])
    sl, bl, rl = _line(np.array(los, float), l[los])
    if max(rh, rl) > 0.6 * atr:
        return None
    start = min(his[0], los[0])
    span = (n - 1) - start
    if span < 15:
        return None
    dh, dl = sh * span / atr, sl * span / atr          # total drift of each line, in ATRs
    if dh <= -0.5 and dl >= 0.5:
        kind = "sym"
    elif abs(dh) < 0.5 and dl >= 0.8:
        kind = "asc"
    elif dh <= -0.8 and abs(dl) < 0.5:
        kind = "desc"
    else:
        return None
    up = lambda i: sh * i + bh  # noqa: E731
    lo = lambda i: sl * i + bl  # noqa: E731
    if up(start) - lo(start) < 1.5 * atr or up(n - 1) - lo(n - 1) < 0.2 * atr:
        return None
    body = np.arange(start, max(start + 1, n - 3))
    inside = (c[body] <= up(body) + 0.4 * atr) & (c[body] >= lo(body) - 0.4 * atr)
    if inside.mean() < 0.9:
        return None
    brk, brk_i = None, None
    for i in range(max(start + 1, n - 3), n):
        if c[i] > up(i) + 0.1 * atr and c[i - 1] <= up(i - 1) + 0.1 * atr:
            brk, brk_i = "up", i
        elif c[i] < lo(i) - 0.1 * atr and c[i - 1] >= lo(i - 1) - 0.1 * atr:
            brk, brk_i = "down", i
    return {"type": kind, "name": TRI_NAMES[kind], "start_i": start,
            "upper": [(his[0], up(his[0])), (n - 1, up(n - 1))], "lower": [(los[0], lo(los[0])), (n - 1, lo(n - 1))],
            "upper_now": up(n - 1), "lower_now": lo(n - 1), "breakout": brk, "break_i": brk_i}


def analyze(bars: pd.DataFrame, tf: str, price: float | None = None) -> dict:
    """Support/resistance, Fibonacci and triangle for these bars (raw prices)."""
    cfg = CFG.get(tf, CFG["15m"])
    if len(bars) < 60:
        return {}
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(bars)
    atr = float(ind.atr(bars, 14).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return {}
    price = float(c[-1]) if price is None else float(price)
    ph, pl = pivots(h, l, cfg["k"])
    extra = []
    if tf in ("5m", "15m", "1h"):
        d1 = resample(bars, "1D")
        if len(d1) >= 2:
            extra = [("昨日高點", float(d1["high"].iloc[-2])), ("昨日低點", float(d1["low"].iloc[-2]))]
    return {"atr": atr, "price": price, "n": n,
            "levels": _levels(h, l, ph, pl, max(0, n - cfg["look"]), price, atr, extra),
            "fib": _fib(h, l, n, cfg["fib"], atr, price),
            "triangle": _triangle(h, l, c, n, cfg["k"], cfg["tri"], atr)}


# ---------------------------------------------------------------------------------------- text

def describe(an: dict, core: dict, fmt, tf_label: str, inst_name: str, position_dir: int = 0,
             stop: float | None = None, off: float = 0.0) -> dict:
    """Plain-language summary. `fmt` formats a raw price for display (adds the price offset)."""
    head = _headline(core, fmt, inst_name)
    lines = []
    bias = core.get("bias", 0) if core else 0
    if an:
        atr = an["atr"]
        res = [x for x in an["levels"] if x["kind"] == "resistance"]
        sup = [x for x in an["levels"] if x["kind"] == "support"]
        if res or sup:
            parts = []
            if res:
                parts.append("上方壓力 " + "、".join(f"{fmt(x['price'])}（{x['label']}）" for x in res))
            if sup:
                parts.append("下方支撐 " + "、".join(f"{fmt(x['price'])}（{x['label']}）" for x in sup))
            txt = "；".join(parts) + "。"
            near = min(an["levels"], key=lambda x: x["dist_atr"])
            if near["dist_atr"] <= 0.5:
                if near["kind"] == "resistance":
                    txt += f"價格正在測試壓力 {fmt(near['price'])}：收盤站上去通常會續漲，碰到就掉頭則容易拉回。"
                else:
                    txt += f"價格正在測試支撐 {fmt(near['price'])}：守住通常會反彈，收盤跌破則容易再往下。"
            if position_dir > 0 and res:
                txt += f"持多單的話，接近 {fmt(res[0]['price'])} 可留意是否過不去。"
            elif position_dir < 0 and sup:
                txt += f"持空單的話，接近 {fmt(sup[0]['price'])} 可留意是否跌不破。"
            lines.append({"tool": "sr", "text": txt})
        fb = an.get("fib")
        if fb:
            lv = {round(x["ratio"], 3): x["price"] for x in fb["levels"]}
            leg = "上漲" if fb["dir"] > 0 else "下跌"
            back = "回檔" if fb["dir"] > 0 else "反彈"
            r = fb["retrace"]
            if r < 0.15:
                where = f"價格還在{leg}波段的高檔附近，還沒有明顯{back}"
            elif r > 1.0:
                where = f"已經完全吃掉這段{leg}，原本的走勢失效"
            else:
                where = f"目前{back}了約 {r * 100:.0f}%"
            txt = (f"費波納奇：最近一段{leg} {fmt(fb['low'] if fb['dir'] > 0 else fb['high'])} → "
                   f"{fmt(fb['high'] if fb['dir'] > 0 else fb['low'])}，{where}。常見的{back}區在 38.2%–61.8%"
                   f"（{fmt(lv[0.382])}–{fmt(lv[0.618])}）；超過 78.6%（{fmt(lv[0.786])}）通常代表這段{leg}結束。")
            if position_dir == fb["dir"]:
                txt += (f"你的部位和這段{leg}同方向：{back}超過 61.8%（{fmt(lv[0.618])}）可以考慮先減碼或平倉"
                        + (f"，系統停損在 {fmt(stop)}" if stop is not None else "") + "。")
            elif position_dir == -fb["dir"] and r < 0.382:
                txt += f"你的部位方向和這段{leg}相反，要是{back}不到 38.2%（{fmt(lv[0.382])}）就再創新{'高' if fb['dir'] > 0 else '低'}，要特別小心。"
            lines.append({"tool": "fib", "text": txt})
        tri = an.get("triangle")
        if tri:
            name = tri["name"]
            u, d = fmt(tri["upper_now"]), fmt(tri["lower_now"])
            if tri["breakout"] == "up":
                agree = "，和日線偏多的大方向一致，比較可信" if bias > 0 else ("，但日線偏空，小心假突破" if bias < 0 else "")
                txt = f"剛剛收盤突破{name}上緣（約 {u}）→ 偏多{agree}。"
            elif tri["breakout"] == "down":
                agree = "，和日線偏空的大方向一致，比較可信" if bias < 0 else ("，但日線偏多，小心假跌破" if bias > 0 else "")
                txt = f"剛剛收盤跌破{name}下緣（約 {d}）→ 偏空{agree}。"
            else:
                hint = {"asc": "上升三角形通常較常往上突破", "desc": "下降三角形通常較常往下跌破", "sym": "收斂三角形方向未定"}[tri["type"]]
                txt = (f"正在形成{name}：上緣約 {u}、下緣約 {d}，越收越窄。{tf_label}收盤突破上緣偏多、跌破下緣偏空"
                       f"（{hint}；和日線方向一致時較可信）。")
            if position_dir and tri["breakout"] and ((tri["breakout"] == "up") != (position_dir > 0)):
                txt += "⚠ 形態方向和你的部位相反，確認停損有設好。"
            lines.append({"tool": "tri", "text": txt})
    return {"headline": head, "lines": lines,
            "note": "綠色／紅色箭頭與交易單是回測驗證過的系統訊號；支撐壓力、費波納奇、三角形是常見的看圖工具，只當輔助參考，不會單獨產生訊號。"}


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
    reason = "日線沒有明確方向（盤整）" if bias == 0 else f"日線偏{'多' if bias > 0 else '空'}，但其他條件還沒到齊"
    return {"dir": 0, "kind": "wait", "title": "觀望", "text": f"{reason}，現在不要進場。"}
