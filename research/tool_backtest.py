"""Back-test the chart-reading tools (docs/RESEARCH.md section 9).

Part A  every tool as a stand-alone signal (15m / 1h / 4h, two exit styles)
Part B  every tool as a filter on the existing system's trades

Protocol (fixed before looking at the results):
  dev  2005-2016   pick candidates: n >= 100 (4h: 50), mean R t-stat >= 2.0, PF >= 1.15
  val  2017-2020   candidates must stay positive
  test 2021-2025/3 and stay positive again
Filters must improve the average R in all three periods while keeping >= 60 % of the trades.
Costs: spread 0.012 % + slippage 0.005 % per fill + Mitrade overnight fees.

    python research/tool_backtest.py            # both parts, gold 2005-2025/3 (Kaggle 15m file)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import chartlab as C  # noqa: E402
from goldsignal import indicators as ind  # noqa: E402
from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import ExitRules, simulate  # noqa: E402
from goldsignal.bars import align_htf, resample  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import SWAP_LONG_PCT, SWAP_SHORT_PCT, Clock, simulate_plan  # noqa: E402

PERIODS = {"dev": ("2005-01-01", "2016-12-31"), "val": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2025-03-21")}
SPREAD, SLIP = 0.00012, 0.00005
NY = "America/New_York"
OUT = ROOT / "results"
TF_LEN = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}


def load(path: str | None = None) -> pd.DataFrame:
    pq = Path(path) if path else ROOT / "data" / "cache_xau15.parquet"
    if pq.exists():
        return pd.read_parquet(pq).loc[:"2025-03-21"]
    from goldsignal.bars import load_kaggle_15m
    return load_kaggle_15m(ROOT / "data" / "raw" / "XAU_15m_data.csv").loc[:"2025-03-21"]


# =============================================================================== features

def cross_up(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_dn(a, b):
    return cross_up(-np.asarray(a, float), -np.asarray(b, float))


def features(df: pd.DataFrame, tf: str) -> dict:
    c = df["close"]
    F = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}
    F["atr"] = ind.atr(df, 14).to_numpy()
    F["adx"] = ind.adx(df, 14)["adx"].to_numpy()
    for n in (20, 50, 200):
        F[f"e{n}"] = ind.ema(c, n).to_numpy()
    bb = ind.bollinger(c, 20, 2.0)
    F["bb_up"], F["bb_mid"], F["bb_lo"] = (bb[k].to_numpy() for k in ("upper", "mid", "lower"))
    bw = (bb["upper"] - bb["lower"]) / bb["mid"]
    F["bw_pct"] = bw.rolling(200, min_periods=50).rank(pct=True).to_numpy()
    F["rsi"] = ind.rsi(c, 14).to_numpy()
    m = ind.macd(c)
    F["macd"], F["macd_sig"], F["macd_hist"] = (m[k].to_numpy() for k in ("macd", "signal", "hist"))
    kd = ind.stoch_kd(df, 9)
    F["k"], F["d"] = kd["k"].to_numpy(), kd["d"].to_numpy()
    ps = ind.parabolic_sar(df)
    F["sar_dir"] = ps["direction"].to_numpy()
    ich = ind.ichimoku(df)
    F["tenkan"], F["kijun"] = ich["tenkan"].to_numpy(), ich["kijun"].to_numpy()
    F["cloud_top"] = np.fmax(ich["span_a"].to_numpy(), ich["span_b"].to_numpy())
    F["cloud_bot"] = np.fmin(ich["span_a"].to_numpy(), ich["span_b"].to_numpy())
    F["vol_avg"] = df["volume"].rolling(20, min_periods=10).mean().to_numpy()
    clock = Clock.build(df.index)
    F["session"] = clock.session
    ny = df.index.tz_convert(NY)
    F["ny_h"] = np.asarray(ny.hour + ny.minute / 60.0, float)            # bar OPEN hour (NY)
    F["ny_close_h"] = (F["ny_h"] + TF_LEN.get(tf, pd.Timedelta(hours=24)).total_seconds() / 3600) % 24
    # yesterday's (previous trading day) high / low / close for pivots
    d1 = resample(df, "1D")
    sess_of_d1 = Clock.build(d1.index).session
    prev = pd.DataFrame({"H": d1["high"].to_numpy(), "L": d1["low"].to_numpy(), "C": d1["close"].to_numpy()},
                        index=sess_of_d1).shift(1)
    mp = prev.reindex(F["session"])
    P = (mp["H"] + mp["L"] + mp["C"]).to_numpy() / 3
    F["P"], F["R1"], F["S1"] = P, 2 * P - mp["L"].to_numpy(), 2 * P - mp["H"].to_numpy()
    if tf in ("5m", "15m"):
        F["vwap"] = ind.session_vwap(df, F["session"]).to_numpy()
        # Asian range so far in each trading day (19:00-03:00 NY), known from 03:00 on
        asian = (F["ny_h"] >= 19) | (F["ny_h"] < 3)
        hi = pd.Series(np.where(asian, F["high"], np.nan)).groupby(F["session"]).cummax().to_numpy()
        lo = pd.Series(np.where(asian, F["low"], np.nan)).groupby(F["session"]).cummin().to_numpy()
        hi = pd.Series(hi).groupby(F["session"]).ffill().to_numpy()
        lo = pd.Series(lo).groupby(F["session"]).ffill().to_numpy()
        F["asia_hi"], F["asia_lo"], F["asian"] = hi, lo, asian
    return F


# =============================================================================== Part A: signals

def _sig(long_mask, short_mask) -> np.ndarray:
    s = np.zeros(len(long_mask), np.int64)
    s[np.asarray(long_mask, bool)] = 1
    s[np.asarray(short_mask, bool) & (s == 0)] = -1
    return s


def indicator_events(F: dict, tf: str) -> dict:
    o, h, l, c = F["open"], F["high"], F["low"], F["close"]
    atr = F["atr"]
    E = {}
    E["均線黃金/死亡交叉"] = _sig(cross_up(F["e20"], F["e50"]), cross_dn(F["e20"], F["e50"]))
    E["價格穿越EMA200"] = _sig(cross_up(c, F["e200"]), cross_dn(c, F["e200"]))
    squeeze = pd.Series(F["bw_pct"]).shift(1).rolling(10, min_periods=1).min().to_numpy() <= 0.15
    E["布林收窄後突破"] = _sig(cross_up(c, F["bb_up"]) & squeeze, cross_dn(c, F["bb_lo"]) & squeeze)
    rng = F["adx"] < 20
    E["布林盤整回歸"] = _sig(cross_up(c, F["bb_lo"]) & rng, cross_dn(c, F["bb_up"]) & rng)
    E["RSI 30/70 回歸"] = _sig(cross_up(F["rsi"], np.full(len(c), 30.0)), cross_dn(F["rsi"], np.full(len(c), 70.0)))
    E["MACD 交叉"] = _sig(cross_up(F["macd"], F["macd_sig"]), cross_dn(F["macd"], F["macd_sig"]))
    E["KD 高低檔交叉"] = _sig(cross_up(F["k"], F["d"]) & (np.fmin(F["k"], F["d"]) <= 25),
                          cross_dn(F["k"], F["d"]) & (np.fmax(F["k"], F["d"]) >= 75))
    sd = F["sar_dir"]
    flip_up = np.r_[False, (sd[1:] > 0) & (sd[:-1] < 0)]
    flip_dn = np.r_[False, (sd[1:] < 0) & (sd[:-1] > 0)]
    E["SAR 翻轉"] = _sig(flip_up, flip_dn)
    E["雲帶突破"] = _sig(cross_up(c, F["cloud_top"]), cross_dn(c, F["cloud_bot"]))
    E["轉換線/基準線交叉(雲外)"] = _sig(cross_up(F["tenkan"], F["kijun"]) & (c > F["cloud_top"]),
                                cross_dn(F["tenkan"], F["kijun"]) & (c < F["cloud_bot"]))
    vr = F["volume"] / F["vol_avg"]
    big = np.abs(c - o) >= 0.8 * atr
    E["爆量長K"] = _sig((vr >= 2) & big & (c > o), (vr >= 2) & big & (c < o))
    if tf in ("15m", "1h"):
        E["穿越樞軸P"] = _sig(cross_up(c, F["P"]), cross_dn(c, F["P"]))
        E["突破R1/跌破S1"] = _sig(cross_up(c, F["R1"]), cross_dn(c, F["S1"]))
    if tf == "15m":
        euro = (F["ny_close_h"] > 3) & (F["ny_close_h"] <= 11)
        up = cross_up(c, F["asia_hi"] + 0.1 * atr) & euro & ~F["asian"]
        dn = cross_dn(c, F["asia_lo"] - 0.1 * atr) & euro & ~F["asian"]
        # only the first breakout of each side per day
        sess = F["session"]
        first_up = up & (pd.Series(up.astype(int)).groupby(sess).cumsum().to_numpy() == 1)
        first_dn = dn & (pd.Series(dn.astype(int)).groupby(sess).cumsum().to_numpy() == 1)
        E["亞洲盤區間突破"] = _sig(first_up, first_dn)
        E["穿越VWAP(歐美盤)"] = _sig(cross_up(c, F["vwap"]) & euro, cross_dn(c, F["vwap"]) & euro)
    major = 10 ** (np.floor(np.log10(np.maximum(c, 1e-9))) - 1)
    lvl_up = np.floor(np.r_[c[0], c[:-1]] / major) * major + major      # next round level above the previous close
    lvl_dn = np.ceil(np.r_[c[0], c[:-1]] / major) * major - major
    E["突破整數關卡"] = _sig(c > lvl_up + 0.1 * atr, c < lvl_dn - 0.1 * atr)
    # candlestick patterns (no level filter)
    body = np.abs(c - o)
    rngb = h - l
    up_w, dn_w = h - np.maximum(o, c), np.minimum(o, c) - l
    po, pc = np.r_[o[0], o[:-1]], np.r_[c[0], c[:-1]]
    bull_eng = (c > o) & (pc < po) & (o <= pc) & (c >= po) & (body >= 0.3 * atr) & (body > np.abs(pc - po))
    bear_eng = (c < o) & (pc > po) & (o >= pc) & (c <= po) & (body >= 0.3 * atr) & (body > np.abs(pc - po))
    hammer = (rngb >= 0.8 * atr) & (dn_w >= 2 * np.maximum(body, 0.05 * atr)) & (up_w <= 0.35 * rngb)
    star = (rngb >= 0.8 * atr) & (up_w >= 2 * np.maximum(body, 0.05 * atr)) & (dn_w <= 0.35 * rngb)
    E["吞噬/錘子/流星(不看位置)"] = _sig(bull_eng | hammer, bear_eng | star)
    for k_, s in E.items():
        E[k_] = np.asarray(s, np.int64)
    return E


def rolling_events(df: pd.DataFrame, tf: str, window: int = 400, progress: bool = True) -> dict:
    """Events that need the chartlab detectors (patterns, S/R, trendlines, Fibonacci, divergences).
    Each bar only sees the `window` bars up to and including itself (no look-ahead)."""
    n = len(df)
    names = ["三角形突破", "M頭/W底頸線突破", "頭肩頸線突破", "箱型突破", "旗形突破", "壓力突破/支撐跌破", "支撐/壓力反彈",
             "趨勢線反彈", "趨勢線跌破/突破", "費波納奇回撤反彈", "RSI背離", "MACD背離", "K線型態@支撐壓力"]
    E = {k: np.zeros(n, np.int64) for k in names}
    t0 = time.time()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    atr_all = ind.atr(df, 14).to_numpy()
    rsi_all = ind.rsi(df["close"], 14).to_numpy()
    macd_all = ind.macd(df["close"])["macd"].to_numpy()
    e20, e50 = ind.ema(df["close"], 20).to_numpy(), ind.ema(df["close"], 50).to_numpy()
    adx_all = ind.adx(df, 14)["adx"].to_numpy()
    k = C.CFG[tf]["k"]
    tlist = list(range(window))
    fmt = lambda x: ""  # noqa: E731
    for i in range(window, n):
        a = i - window + 1
        atr = atr_all[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sl = slice(a, i + 1)
        hh, ll, cc, oo = h[sl], l[sl], c[sl], o[sl]
        ph, pl = C.pivots(hh, ll, k)
        cx = C.Ctx(bars=None, tf=tf, t=tlist, o=oo, h=hh, l=ll, c=cc, v=v[sl], n=window, atr=atr, price=cc[-1], fmt=fmt,
                   ph=ph, pl=pl)
        up_reg = adx_all[i] >= 25 and cc[-1] > e50[i] and e20[i] > e50[i]
        dn_reg = adx_all[i] >= 25 and cc[-1] < e50[i] and e20[i] < e50[i]
        cx.regime = {"key": "up" if up_reg else "down" if dn_reg else "range" if adx_all[i] < 20 else "mixed"}
        last = window - 1
        # ---- patterns: count only when the breakout happens on THIS bar
        for name, fn in (("三角形突破", C._p_triangle), ("M頭/W底頸線突破", C._p_double), ("頭肩頸線突破", C._p_hs),
                         ("箱型突破", C._p_box), ("旗形突破", C._p_flag)):
            try:
                p = fn(cx)
            except Exception:
                p = None
            if p and p.get("brk") and p.get("bi") == last:
                E[name][i] = p["brk"]
        # ---- support / resistance (levels from bars before this one)
        lv = C._levels(hh[:-1], ll[:-1], [x for x in ph if x < last], [x for x in pl if x < last], 0, cc[-2], atr, [])
        for x in lv:
            if x["touches"] < 2:
                continue
            if x["kind"] == "resistance" and cc[-1] > x["price"] + 0.1 * atr and cc[-2] <= x["price"] + 0.1 * atr:
                E["壓力突破/支撐跌破"][i] = 1
            if x["kind"] == "support" and cc[-1] < x["price"] - 0.1 * atr and cc[-2] >= x["price"] - 0.1 * atr:
                E["壓力突破/支撐跌破"][i] = -1
            if x["kind"] == "support" and ll[-1] <= x["price"] + 0.2 * atr and cc[-1] > x["price"] and cc[-1] > oo[-1]:
                E["支撐/壓力反彈"][i] = 1
            if x["kind"] == "resistance" and hh[-1] >= x["price"] - 0.2 * atr and cc[-1] < x["price"] and cc[-1] < oo[-1]:
                E["支撐/壓力反彈"][i] = -1
        # ---- trendline (in the regime direction)
        d = 1 if up_reg else -1 if dn_reg else 0
        if d:
            piv = pl if d > 0 else ph
            src = ll if d > 0 else hh
            cand = [x for x in piv if x < last][-3:]
            if len(cand) >= 2:
                s, b, resid = C._line(cand, src[cand])
                if ((d > 0 and s > 0) or (d < 0 and s < 0)) and resid <= 0.5 * atr:
                    lt = s * last + b
                    if d > 0 and ll[-1] <= lt + 0.3 * atr and cc[-1] > lt and cc[-1] > oo[-1]:
                        E["趨勢線反彈"][i] = 1
                    if d < 0 and hh[-1] >= lt - 0.3 * atr and cc[-1] < lt and cc[-1] < oo[-1]:
                        E["趨勢線反彈"][i] = -1
                    lt_prev = s * (last - 1) + b
                    if d > 0 and cc[-1] < lt - 0.1 * atr and cc[-2] >= lt_prev - 0.1 * atr:
                        E["趨勢線跌破/突破"][i] = -1
                    if d < 0 and cc[-1] > lt + 0.1 * atr and cc[-2] <= lt_prev + 0.1 * atr:
                        E["趨勢線跌破/突破"][i] = 1
            # ---- Fibonacci pullback bounce in the trend direction
            w = C.CFG[tf]["fib"]
            hi_i = window - w + int(np.argmax(hh[-w:]))
            lo_i = window - w + int(np.argmin(ll[-w:]))
            hi_, lo_ = hh[hi_i], ll[lo_i]
            if hi_ - lo_ >= 3 * atr:
                upleg = lo_i < hi_i
                r = (hi_ - cc[-1]) / (hi_ - lo_) if upleg else (cc[-1] - lo_) / (hi_ - lo_)
                if 0.382 <= r <= 0.618:
                    if d > 0 and upleg and cc[-1] > oo[-1] and cc[-1] > hh[-2]:
                        E["費波納奇回撤反彈"][i] = 1
                    if d < 0 and not upleg and cc[-1] < oo[-1] and cc[-1] < ll[-2]:
                        E["費波納奇回撤反彈"][i] = -1
        # ---- divergences (second swing must be the most recent confirmed pivot, just confirmed)
        for name, osc, gap in (("RSI背離", rsi_all[sl], 3.0), ("MACD背離", macd_all[sl], 0.1 * atr)):
            dv = C._divergence(cx, osc, gap)
            if dv and dv[2] == last - 3:          # pivot confirmed exactly now (k=3 bars after it)
                E[name][i] = dv[0]
        # ---- candle pattern at a 2-touch level
        cp = C._candle_pattern(cx, last)
        if cp and cp[1] != 0:
            for x in lv:
                if x["touches"] >= 2 and abs(x["price"] - (ll[-1] if cp[1] > 0 else hh[-1])) <= 0.3 * atr and \
                        (cp[1] > 0) == (x["kind"] == "support"):
                    E["K線型態@支撐壓力"][i] = cp[1]
                    break
        if progress and i % 50000 == 0:
            print(f"    {tf} rolling {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    return E


SCHEMES = {
    "跟蹤": dict(stop=2.0, rules=ExitRules(tp_r=0.0, trail_mult=5.0, cooldown=1)),
    "固定1.5R": dict(stop=1.5, rules=ExitRules(tp_r=1.5, trail_mult=0.0, max_bars=48, cooldown=1)),
}


def backtest(df: pd.DataFrame, F: dict, sig: np.ndarray, scheme: str) -> pd.DataFrame:
    sc = SCHEMES[scheme]
    c, atr = F["close"], F["atr"]
    px = F["open"]
    tr = simulate(df, sig, c - sc["stop"] * atr, c + sc["stop"] * atr, sc["rules"], px * SPREAD, px * SLIP, atr=atr)
    if tr.empty:
        return tr
    tr = tr[tr["reason"] != "end"].copy()
    nights = F["session"][tr["exit_i"].to_numpy()] - F["session"][np.minimum(tr["sig_i"].to_numpy() + 1, len(c) - 1)]
    rate = np.where(tr["dir"] > 0, SWAP_LONG_PCT, SWAP_SHORT_PCT)
    tr["R"] = tr["R"] - nights * rate * tr["entry"] / tr["risk"]
    return tr


def stats(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"n": 0}
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq)) if len(r) else 0.0
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return {"n": int(len(r)), "win%": round(100 * float((r > 0).mean()), 1), "avgR": round(float(r.mean()), 3),
            "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd and sd > 0 else 0.0,
            "PF": round(float(wins / losses), 2) if losses > 0 else 99.0, "maxDD": round(dd, 1)}


def by_period(tr: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        out[name] = stats(tr.loc[m, "R"].to_numpy())
    out["all"] = stats(tr["R"].to_numpy())
    return out


def tool_events(df15: pd.DataFrame, tf: str, rolling: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """All tool events for one time frame; the slow rolling detectors are cached in data/cache/."""
    df = df15 if tf == "15m" else resample(df15, tf)
    F = features(df, tf)
    E = indicator_events(F, tf)
    if rolling:
        cache = ROOT / "data" / "cache" / f"tool_events_{tf}.pkl"
        key = (len(df), str(df.index[0]), str(df.index[-1]))
        R = None
        if cache.exists():
            got = pd.read_pickle(cache)
            if got.get("key") == key:
                R = got["events"]
        if R is None:
            t0 = time.time()
            R = rolling_events(df, tf)
            print(f"  {tf}: rolling detectors {time.time() - t0:.0f}s", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle({"key": key, "events": R}, cache)
        E.update(R)
    return df, F, E


def part_a(df15: pd.DataFrame, tfs=("15m", "1h", "4h"), rolling_tfs=("1h", "4h", "15m")) -> pd.DataFrame:
    rows = []
    for tf in tfs:
        df, F, E = tool_events(df15, tf, tf in rolling_tfs)
        for name, sig in E.items():
            for scheme in SCHEMES:
                tr = backtest(df, F, sig, scheme)
                if tr.empty:
                    continue
                res = by_period(tr)
                rows.append({"tool": name, "tf": tf, "exit": scheme, **{f"{p}_{k}": v for p, d in res.items() for k, v in d.items()}})
        print(f"  {tf}: {len(E)} tools done", flush=True)
    return pd.DataFrame(rows)


def select(res: pd.DataFrame) -> pd.DataFrame:
    min_n = np.where(res["tf"] == "4h", 50, 100)
    cand = res[(res["dev_n"] >= min_n) & (res["dev_t"] >= 2.0) & (res["dev_PF"] >= 1.15)].copy()
    cand["passed"] = (cand["val_avgR"] > 0) & (cand["test_avgR"] > 0)
    return cand


# =============================================================================== Part B: the system + tools

def system_setup(df15: pd.DataFrame, symbol: str = "XAUUSD") -> dict:
    """The live system (day mode) exactly as scripts/make_backtest_summary.py runs it."""
    inst = INSTRUMENTS[symbol]
    p = inst.strategy_params(True, squeeze=False)      # the system BEFORE any tool filter
    f = S.compute_features(df15)
    t = S.rule_table(df15, f, p, None)
    sig = S.signals_from_rules(t)
    px = df15["open"].to_numpy(float)
    return {"p": p, "f": f, "sig": np.asarray(sig, np.int64), "atr1": f["h1_atr"].to_numpy(float),
            "sp": px * inst.spread_pct, "sl": px * inst.slip_pct, "clock": Clock.build(df15.index), "plan": p.plan(swap=True)}


def run_system(df15: pd.DataFrame, SY: dict, sig=None, exit_sig=None, target_px=None, start="2005-01-01") -> pd.DataFrame:
    tr = simulate_plan(df15, SY["sig"] if sig is None else sig, SY["atr1"], SY["plan"], SY["sp"], SY["sl"], SY["clock"],
                       exit_sig=exit_sig, target_px=target_px)
    return tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp(start, tz="UTC"))].reset_index(drop=True)


def last_closed_1h(df15: pd.DataFrame, h1: pd.DataFrame) -> np.ndarray:
    """For every 15m bar: index of the last 1h bar that has closed by the 15m bar's close (-1 = none)."""
    return h1.index.searchsorted(df15.index + pd.Timedelta(minutes=15) - pd.Timedelta(hours=1), side="right") - 1


def h1_context(h1: pd.DataFrame, F1: dict, e: int, price: float, window: int = 400):
    """chartlab context on the 1h bars that end at index e (inclusive)."""
    sl = slice(e - window + 1, e + 1)
    o1, hh1, ll1, cc1, v1 = (h1[k].to_numpy(float)[sl] for k in ("open", "high", "low", "close", "volume"))
    ph, pl = C.pivots(hh1, ll1, C.CFG["1h"]["k"])
    cx = C.Ctx(bars=None, tf="1h", t=list(range(window)), o=o1, h=hh1, l=ll1, c=cc1, v=v1, n=window,
               atr=float(F1["atr"][e]), price=price, fmt=lambda x: "", ph=ph, pl=pl)
    cx.regime = {"key": "mixed"}
    return cx, sl


def fib_leg(cx) -> tuple[float, float, bool] | None:
    w = C.CFG["1h"]["fib"]
    hi_i, lo_i = cx.n - w + int(np.argmax(cx.h[-w:])), cx.n - w + int(np.argmin(cx.l[-w:]))
    hi_, lo_ = float(cx.h[hi_i]), float(cx.l[lo_i])
    if hi_ - lo_ < 3 * cx.atr:
        return None
    return hi_, lo_, lo_i < hi_i


def filter_features(df15: pd.DataFrame, i: np.ndarray, d: np.ndarray, F15: dict, h1: pd.DataFrame, F1: dict) -> pd.DataFrame:
    """Tool states at each system signal bar i (15m close) with direction d; only data known at that moment."""
    c = F15["close"][i]
    cols = ["close", "rsi", "macd_hist", "k", "sar_dir", "cloud_top", "cloud_bot", "e200", "atr", "adx", "bw_pct"]
    h1f = pd.DataFrame({k: F1[k] for k in cols}, index=h1.index)
    h1f["adx5"] = h1f["adx"].shift(5)
    a1 = align_htf(df15.index, pd.Timedelta(minutes=15), h1f, pd.Timedelta(hours=1)).iloc[i]
    X = pd.DataFrame({"i": i, "dir": d})
    up = d > 0
    X["RSI15未過熱"] = np.where(up, F15["rsi"][i] < 70, F15["rsi"][i] > 30)
    X["RSI1h未過熱"] = np.where(up, a1["rsi"].to_numpy() < 70, a1["rsi"].to_numpy() > 30)
    X["1h RSI同向(>50/<50)"] = np.where(up, a1["rsi"].to_numpy() > 50, a1["rsi"].to_numpy() < 50)
    X["KD15未過熱"] = np.where(up, F15["k"][i] < 80, F15["k"][i] > 20)
    X["收在布林通道內"] = np.where(up, c <= F15["bb_up"][i], c >= F15["bb_lo"][i])
    X["15m布林收窄後"] = pd.Series(F15["bw_pct"]).shift(1).rolling(16, min_periods=1).min().to_numpy()[i] <= 0.3
    X["1h布林收窄後"] = a1["bw_pct"].to_numpy() <= 0.3
    X["1h雲帶同向"] = np.where(up, a1["close"].to_numpy() > a1["cloud_top"].to_numpy(), a1["close"].to_numpy() < a1["cloud_bot"].to_numpy())
    X["VWAP同向"] = np.where(up, c > F15["vwap"][i], c < F15["vwap"][i])
    X["1h MACD同向"] = np.where(up, a1["macd_hist"].to_numpy() > 0, a1["macd_hist"].to_numpy() < 0)
    X["1h SAR同向"] = np.sign(a1["sar_dir"].to_numpy()) == d
    X["1h EMA200同向"] = np.where(up, a1["close"].to_numpy() > a1["e200"].to_numpy(), a1["close"].to_numpy() < a1["e200"].to_numpy())
    X["樞軸P同向"] = np.where(up, c > F15["P"][i], c < F15["P"][i])
    X["已突破R1/S1"] = np.where(up, c > F15["R1"][i], c < F15["S1"][i])
    X["在亞洲盤區間外(同向)"] = np.where(up, c > F15["asia_hi"][i], c < F15["asia_lo"][i])
    X["1h ADX上升"] = a1["adx"].to_numpy() > a1["adx5"].to_numpy()
    X["訊號K放量(≥1.5倍)"] = F15["volume"][i] / F15["vol_avg"][i] >= 1.5
    major = 10 ** (np.floor(np.log10(c)) - 1)
    nxt = np.where(up, np.ceil(c / major) * major, np.floor(c / major) * major)
    X["整數關卡還有空間"] = np.abs(nxt - c) >= 0.5 * a1["atr"].to_numpy()
    # 1h chart-reading tools at the last completed 1h bar
    end_1h = last_closed_1h(df15, h1)[i]
    room, div_against, pat_with, pat_against, fib_hi = [], [], [], [], []
    rsi1 = F1["rsi"]
    for j, e in enumerate(end_1h):
        if e < 400:
            room.append(np.nan); div_against.append(False); pat_with.append(False); pat_against.append(False); fib_hi.append(False)
            continue
        cx, sl = h1_context(h1, F1, e, c[j])
        lv = C._levels(cx.h, cx.l, cx.ph, cx.pl, 0, c[j], cx.atr, [])
        opp = [x for x in lv if x["touches"] >= 2 and x["kind"] == ("resistance" if d[j] > 0 else "support")]
        room.append(min((x["dist_atr"] for x in opp), default=np.inf))
        dv = C._divergence(cx, rsi1[sl], 3.0)
        div_against.append(bool(dv and dv[0] == -d[j]))
        pats = []
        for fn in (C._p_triangle, C._p_double, C._p_hs, C._p_box, C._p_flag):
            try:
                pt = fn(cx)
            except Exception:
                pt = None
            if pt:
                pats.append(pt)
        pat_with.append(any(pt.get("brk") == d[j] for pt in pats))
        pat_against.append(any(pt.get("brk") == -d[j] for pt in pats))
        leg = fib_leg(cx)
        if leg:
            hi_, lo_, upleg = leg
            r = (hi_ - c[j]) / (hi_ - lo_) if upleg else (c[j] - lo_) / (hi_ - lo_)
            fib_hi.append(bool(upleg == (d[j] > 0) and r <= 0.236))
        else:
            fib_hi.append(False)
    X["1h前方壓力≥1.5ATR"] = np.asarray(room) >= 1.5
    X["1h無反向背離"] = ~np.asarray(div_against)
    X["1h型態同向突破"] = np.asarray(pat_with)
    X["1h無反向型態"] = ~np.asarray(pat_against)
    X["費波納奇在波段高檔(延續)"] = np.asarray(fib_hi)
    agree = X[["1h雲帶同向", "VWAP同向", "1h MACD同向", "1h SAR同向", "1h EMA200同向", "樞軸P同向"]].sum(axis=1)
    for kk in (4, 5, 6):
        X[f"同向指標≥{kk}/6"] = agree >= kk
    return X


def period_stats(tr: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        r = tr.loc[m, "R"].to_numpy()
        st = stats(r)
        st["totR"] = round(float(r.sum()), 1)
        out[name] = st
    r = tr["R"].to_numpy()
    out["all"] = {**stats(r), "totR": round(float(r.sum()), 1)}
    return out


def compare(name: str, tr: pd.DataFrame, base: pd.DataFrame, kind: str) -> dict:
    ps, pb = period_stats(tr), period_stats(base)
    row = {"name": name, "kind": kind, "n": ps["all"]["n"], "keep%": round(100 * ps["all"]["n"] / max(pb["all"]["n"], 1))}
    better = True
    for p in list(PERIODS) + ["all"]:
        row[f"{p}_avgR"] = ps[p].get("avgR", np.nan)
        row[f"{p}_base"] = pb[p].get("avgR", np.nan)
        row[f"{p}_totR"] = ps[p].get("totR", 0.0)
        row[f"{p}_DD"] = ps[p].get("maxDD", np.nan)
        if p != "all":
            better &= ps[p].get("n", 0) > 0 and ps[p]["avgR"] > pb[p]["avgR"]
    row["all_base_totR"] = pb["all"]["totR"]
    row["all_base_DD"] = pb["all"]["maxDD"]
    row["passed"] = bool(better and (kind != "filter" or row["keep%"] >= 60))
    return row


def exit_events(df15: pd.DataFrame, h1: pd.DataFrame, F1: dict, E1: dict, F15: dict) -> dict:
    """Chart-tool exit rules -> per-15m-bar exit arrays (-1 close longs / +1 close shorts) at the 15m bar
    that closes together with (or right after) the 1h bar that produced the event."""
    k_of_j = last_closed_1h(df15, h1)
    first_j = np.searchsorted(k_of_j, np.arange(len(h1)), side="left")
    n15 = len(df15)

    def to15(ev1: np.ndarray) -> np.ndarray:
        out = np.zeros(n15, np.int64)
        k = np.flatnonzero(ev1)
        j = first_j[k]
        ok = j < n15
        out[j[ok]] = ev1[k[ok]]
        return out

    X = {}
    X["1h 趨勢線被反向突破"] = to15(E1["趨勢線跌破/突破"])
    pats = np.zeros(len(h1), np.int64)
    for k_ in ("三角形突破", "M頭/W底頸線突破", "頭肩頸線突破", "箱型突破", "旗形突破"):
        pats = np.where(pats == 0, E1[k_], pats)
    X["1h 反向型態突破"] = to15(pats)
    X["1h 反向背離(RSI/MACD)"] = to15(np.where(E1["RSI背離"] != 0, E1["RSI背離"], E1["MACD背離"]))
    X["1h SAR 翻轉"] = to15(E1["SAR 翻轉"])
    X["1h MACD 反向交叉"] = to15(E1["MACD 交叉"])
    X["1h 均線反向交叉"] = to15(E1["均線黃金/死亡交叉"])
    X["1h KD 高低檔反向交叉"] = to15(E1["KD 高低檔交叉"])
    X["1h RSI 超買/超賣回落"] = to15(E1["RSI 30/70 回歸"])
    X["1h 跌破支撐/突破壓力"] = to15(E1["壓力突破/支撐跌破"])
    X["1h K線反轉型態@支撐壓力"] = to15(E1["K線型態@支撐壓力"])
    X["15m 穿越VWAP"] = _sig(cross_up(F15["close"], F15["vwap"]), cross_dn(F15["close"], F15["vwap"]))
    # Fibonacci: the leg in the position's direction has retraced more than 61.8 % (state, checked each 1h close)
    h, l, c, atr = F1["high"], F1["low"], F1["close"], F1["atr"]
    w = C.CFG["1h"]["fib"]
    hi = pd.Series(h).rolling(w).max().to_numpy()
    lo = pd.Series(l).rolling(w).min().to_numpy()
    hi_i = pd.Series(h).rolling(w).apply(np.argmax, raw=True).to_numpy()
    lo_i = pd.Series(l).rolling(w).apply(np.argmin, raw=True).to_numpy()
    rng = hi - lo
    ok = rng >= 3 * atr
    upleg = lo_i < hi_i
    r_up = (hi - c) / rng
    r_dn = (c - lo) / rng
    fib = np.where(ok & upleg & (r_up > 0.618), -1, np.where(ok & ~upleg & (r_dn > 0.618), 1, 0))
    X["1h 費波納奇回撤>61.8%"] = to15(fib.astype(np.int64))
    return X


def fib_targets(df15: pd.DataFrame, i: np.ndarray, d: np.ndarray, h1: pd.DataFrame, F1: dict, F15: dict) -> dict:
    """Price targets at each signal bar: Fibonacci extensions of the 1h leg and the next 1h S/R level."""
    end_1h = last_closed_1h(df15, h1)[i]
    c = F15["close"][i]
    out = {k: np.full(len(i), np.nan) for k in ("費波納奇延伸127.2%", "費波納奇延伸161.8%", "前方1h壓力/支撐")}
    for j, e in enumerate(end_1h):
        if e < 400:
            continue
        cx, _ = h1_context(h1, F1, e, c[j])
        leg = fib_leg(cx)
        if leg:
            hi_, lo_, upleg = leg
            if upleg and d[j] > 0:
                out["費波納奇延伸127.2%"][j] = lo_ + 1.272 * (hi_ - lo_)
                out["費波納奇延伸161.8%"][j] = lo_ + 1.618 * (hi_ - lo_)
            elif not upleg and d[j] < 0:
                out["費波納奇延伸127.2%"][j] = hi_ - 1.272 * (hi_ - lo_)
                out["費波納奇延伸161.8%"][j] = hi_ - 1.618 * (hi_ - lo_)
        lv = C._levels(cx.h, cx.l, cx.ph, cx.pl, 0, c[j], cx.atr, [])
        opp = [x for x in lv if x["touches"] >= 2 and x["kind"] == ("resistance" if d[j] > 0 else "support")]
        if opp:
            x = min(opp, key=lambda x: x["dist_atr"])
            out["前方1h壓力/支撐"][j] = x["price"] - d[j] * 0.1 * cx.atr
    return out


def part_b(df15: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    SY = system_setup(df15)
    base = run_system(df15, SY)
    print(f"  system: {len(base)} trades, avgR {base['R'].mean():.3f}", flush=True)
    F15 = features(df15, "15m")
    h1, F1, E1 = tool_events(df15, "1h", True)
    i = np.flatnonzero(SY["sig"])
    d = SY["sig"][i]
    t0 = time.time()
    X = filter_features(df15, i, d, F15, h1, F1)
    print(f"  filters at {len(i)} signal bars {time.time() - t0:.0f}s", flush=True)
    rows = [compare("（基準）系統", base, base, "base")]
    for col in X.columns:
        if col in ("i", "dir"):
            continue
        keep = X[col].to_numpy(bool)
        sig = np.zeros_like(SY["sig"])
        sig[i[keep]] = d[keep]
        rows.append(compare(col, run_system(df15, SY, sig=sig), base, "filter"))
    for name, ex in exit_events(df15, h1, F1, E1, F15).items():
        rows.append(compare(name, run_system(df15, SY, exit_sig=ex), base, "exit"))
    for name, tgt in fib_targets(df15, i, d, h1, F1, F15).items():
        full = np.full(len(df15), np.nan)
        full[i] = tgt
        rows.append(compare(name, run_system(df15, SY, target_px=full), base, "target"))
    return pd.DataFrame(rows), {"X": X, "SY": SY}


# =============================================================================== Part C: is the winner real?

def squeeze_mask(F15: dict, lookback: int = 16, q: float = 0.3) -> np.ndarray:
    """15m Bollinger band width was in its lowest q of the last 200 bars at some point in the
    `lookback` bars before the signal bar (the 'squeeze' that the breakout comes out of)."""
    return pd.Series(F15["bw_pct"]).shift(1).rolling(lookback, min_periods=1).min().to_numpy() <= q


def part_c(df15: pd.DataFrame, n_perm: int = 1000, seed: int = 7) -> dict:
    SY = system_setup(df15)
    base = run_system(df15, SY)
    F15 = features(df15, "15m")
    i = np.flatnonzero(SY["sig"])
    d = SY["sig"][i]
    out = {"grid": [], "perm": {}, "by_year": {}, "by_dir": {}}

    def with_keep(keep):
        sig = np.zeros_like(SY["sig"])
        sig[i[keep]] = d[keep]
        return run_system(df15, SY, sig=sig)

    for lb in (8, 16, 24, 32):
        for q in (0.2, 0.3, 0.4):
            keep = squeeze_mask(F15, lb, q)[i]
            row = compare(f"lookback {lb} bars, q {q}", with_keep(keep), base, "filter")
            row.update(lookback=lb, q=q)
            out["grid"].append(row)
    keep = squeeze_mask(F15)[i]
    tr = with_keep(keep)
    ref = period_stats(tr)
    # permutation test: drop the same share of signal bars at random
    rng = np.random.default_rng(seed)
    frac = keep.mean()
    sims = {p: [] for p in list(PERIODS) + ["all"]}
    for _ in range(n_perm):
        k = rng.random(len(i)) < frac
        ps = period_stats(with_keep(k))
        for p in sims:
            sims[p].append(ps[p].get("avgR", np.nan))
    for p in sims:
        a = np.asarray(sims[p], float)
        out["perm"][p] = {"observed": ref[p]["avgR"], "random_mean": round(float(np.nanmean(a)), 3),
                          "random_p95": round(float(np.nanpercentile(a, 95)), 3),
                          "p_value": round(float(np.mean(a >= ref[p]["avgR"])), 4)}
    # kept vs dropped trades of the original system (no re-simulation)
    kept_sig = set(i[keep].tolist())
    in_keep = base["sig_i"].isin(kept_sig)
    for name, m in (("kept", in_keep), ("dropped", ~in_keep)):
        r = base.loc[m, "R"].to_numpy()
        out[f"base_{name}"] = stats(r)
    out["by_year"] = {
        "system": base.groupby(base["entry_time"].dt.year)["R"].agg(["count", "sum", "mean"]).round(3).reset_index().values.tolist(),
        "filtered": tr.groupby(tr["entry_time"].dt.year)["R"].agg(["count", "sum", "mean"]).round(3).reset_index().values.tolist()}
    for dname, dv in (("long", 1), ("short", -1)):
        out["by_dir"][dname] = {"system": stats(base.loc[base["dir"] == dv, "R"].to_numpy()),
                                "filtered": stats(tr.loc[tr["dir"] == dv, "R"].to_numpy())}
    return out


# =============================================================================== Part S: summary for the app

# chartlab tool key -> (stand-alone signals, filters, exits/targets) in the result tables
TOOL_KEYS = {
    "sr": (["壓力突破/支撐跌破", "支撐/壓力反彈"], [], ["1h 跌破支撐/突破壓力", "前方1h壓力/支撐"]),
    "ma": (["均線黃金/死亡交叉", "價格穿越EMA200"], ["1h EMA200同向"], ["1h 均線反向交叉"]),
    "trend": (["趨勢線反彈", "趨勢線跌破/突破"], [], ["1h 趨勢線被反向突破"]),
    "fib": (["費波納奇回撤反彈"], ["費波納奇在波段高檔(延續)"], ["1h 費波納奇回撤>61.8%", "費波納奇延伸127.2%", "費波納奇延伸161.8%"]),
    "pattern": (["三角形突破", "M頭/W底頸線突破", "頭肩頸線突破", "箱型突破", "旗形突破"], ["1h型態同向突破", "1h無反向型態"],
                ["1h 反向型態突破"]),
    "bb": (["布林收窄後突破", "布林盤整回歸"], ["15m布林收窄後", "1h布林收窄後", "收在布林通道內"], []),
    "round": (["突破整數關卡"], ["整數關卡還有空間"], []),
    "pivot": (["穿越樞軸P", "突破R1/跌破S1"], ["樞軸P同向", "已突破R1/S1"], []),
    "session": (["亞洲盤區間突破"], ["在亞洲盤區間外(同向)"], []),
    "vwap": (["穿越VWAP(歐美盤)"], ["VWAP同向"], ["15m 穿越VWAP"]),
    "ichimoku": (["雲帶突破", "轉換線/基準線交叉(雲外)"], ["1h雲帶同向"], []),
    "sar": (["SAR 翻轉"], ["1h SAR同向"], ["1h SAR 翻轉"]),
    "candle": (["吞噬/錘子/流星(不看位置)", "K線型態@支撐壓力"], [], ["1h K線反轉型態@支撐壓力"]),
    "volume": (["爆量長K"], ["訊號K放量(≥1.5倍)"], []),
    "rsi": (["RSI 30/70 回歸", "RSI背離"], ["RSI15未過熱", "RSI1h未過熱", "1h RSI同向(>50/<50)", "1h無反向背離"],
            ["1h RSI 超買/超賣回落", "1h 反向背離(RSI/MACD)"]),
    "macd": (["MACD 交叉", "MACD背離"], ["1h MACD同向"], ["1h MACD 反向交叉"]),
    "kd": (["KD 高低檔交叉"], ["KD15未過熱"], ["1h KD 高低檔反向交叉"]),
}
TF_WORD = {"15m": "15 分", "1h": "1 小時", "4h": "4 小時"}


def _r(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def part_s() -> dict:
    """results/XAUUSD_tool_backtest.json - what the dashboard shows about the chart tools."""
    A = pd.read_csv(OUT / "XAUUSD_tool_signals.csv")
    B = pd.read_csv(OUT / "XAUUSD_tool_filters.csv")
    sq = json.loads((OUT / "XAUUSD_squeeze_filter_check.json").read_text(encoding="utf-8"))
    duka = json.loads((OUT / "XAUUSD_tool_dukascopy.json").read_text(encoding="utf-8"))
    rv = json.loads((OUT / "XAUUSD_recent_validation.json").read_text(encoding="utf-8"))
    sel = select(A)
    passed = sel[sel["passed"]]
    base = B[B["kind"] == "base"].iloc[0]
    key_of = {nm: k for k, (sig, fil, ex) in TOOL_KEYS.items() for nm in sig + fil + ex}
    signals = []
    for name, g in A.groupby("tool", sort=False):
        best = g.sort_values("dev_t", ascending=False).iloc[0]
        ok = passed[passed["tool"] == name]
        row = ok.iloc[0] if len(ok) else best
        cand = bool(((g["dev_t"] >= 2.0) & (g["dev_PF"] >= 1.15) & (g["dev_n"] >= np.where(g["tf"] == "4h", 50, 100))).any())
        verdict = "pass" if len(ok) else ("faded" if cand else "none")
        signals.append({"name": name, "key": key_of.get(name), "tf": row["tf"], "exit": row["exit"], "n": int(row["all_n"]),
                        "dev": _r(row["dev_avgR"]), "val": _r(row["val_avgR"]), "test": _r(row["test_avgR"]),
                        "dev_t": _r(row["dev_t"], 2), "all": _r(row["all_avgR"]), "dd": _r(row["all_maxDD"], 1), "verdict": verdict})
    rows = {}
    for kind in ("filter", "exit", "target"):
        rows[kind] = [{"name": r["name"], "key": key_of.get(r["name"]), "keep": int(r["keep%"]), "n": int(r["n"]),
                       "dev": _r(r["dev_avgR"]), "val": _r(r["val_avgR"]), "test": _r(r["test_avgR"]), "all": _r(r["all_avgR"]),
                       "totR": _r(r["all_totR"], 1), "dd": _r(r["all_DD"], 1), "passed": bool(r["passed"])}
                      for _, r in B[B["kind"] == kind].iterrows()]
    sysd = {"n": int(base["n"]), "dev": _r(base["dev_avgR"]), "val": _r(base["val_avgR"]), "test": _r(base["test_avgR"]),
            "all": _r(base["all_avgR"]), "totR": _r(base["all_totR"], 1), "dd": _r(base["all_DD"], 1)}
    win = next(f for f in rows["filter"] if f["name"] == "15m布林收窄後")
    notes = {}
    for key, (sig, fil, ex) in TOOL_KEYS.items():
        parts = []
        mine = [x for x in signals if x["name"] in sig]
        good = [x for x in mine if x["verdict"] == "pass"]
        if good:
            x = good[0]
            parts.append(f"單獨進出場：{TF_WORD[x['tf']]}「{x['name']}」勉強通過檢驗，但平均只有 {x['all']:+.2f}R"
                         f"（2017–2020 {x['val']:+.2f}R），不如系統。")
        elif mine:
            x = max(mine, key=lambda z: z["dev_t"] or -9)
            if x["verdict"] == "faded":
                parts.append(f"單獨進出場：2005–2016 平均 {x['dev']:+.2f}R 看似有效，但 2017–2020 {x['val']:+.2f}R、"
                             f"2021–2025 {x['test']:+.2f}R，之後就失效。")
            else:
                parts.append(f"單獨進出場：沒有優勢（最好的一組每筆平均 {x['all']:+.2f}R）。")
        for f in rows["filter"]:
            if f["name"] in fil and f["passed"]:
                parts.append(f"當系統濾網：「{f['name']}」通過檢驗，已加進黃金系統（每筆 {sysd['all']:+.2f}R → {f['all']:+.2f}R、"
                             f"最大回撤 {sysd['dd']:.1f}R → {f['dd']:.1f}R）。")
        if fil and not any(f["name"] in fil and f["passed"] for f in rows["filter"]):
            parts.append("當系統濾網：沒有幫助。")
        exs = [e for e in rows["exit"] + rows["target"] if e["name"] in ex]
        if exs:
            changed = [e for e in exs if e["n"] != sysd["n"] or abs((e["all"] or 0) - sysd["all"]) > 0.005]
            if changed:
                lo, hi = min(e["all"] for e in changed), max(e["all"] for e in changed)
                if hi >= sysd["all"] - 0.02:
                    parts.append(f"持單時拿它出場：整體差不多（{hi:+.2f}R），但不是每段期間都比較好，不需要特別用。")
                else:
                    rng = f"{lo:+.2f}R" if abs(hi - lo) < 0.005 else f"{lo:+.2f}～{hi:+.2f}R"
                    parts.append(f"持單時拿它提早出場或停利：系統每筆 {sysd['all']:+.2f}R 會變成 {rng}，交給移動停損就好。")
            else:
                parts.append("持單時拿它出場：幾乎不會比系統停損先觸發，結果跟原本一樣。")
        notes[key] = "回測（黃金 2005–2025/3）：" + "".join(parts)
    rs = rv.get("summary", {})
    rn = rv.get("summary_nofilter", {})
    return {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "symbol": "XAUUSD", "data": "Kaggle 15m 2005–2025/3（扣點差 0.012%、滑價 0.005%/次、隔夜費）",
        "counts": {"signals": int(A["tool"].nunique()), "combos": int(len(A.groupby(["tool", "tf"]))), "tests": int(len(A)),
                   "candidates": int(len(sel)), "passed": int(len(passed))},
        "system": sysd, "signals": signals, "filters": rows["filter"], "exits": rows["exit"] + rows["target"],
        "squeeze": {
            "kaggle": {"system": sysd, "filter": win, "perm_p": {k: v["p_value"] for k, v in sq["perm"].items()},
                       "kept": sq["base_kept"], "dropped": sq["base_dropped"],
                       "grid": [{"lookback": g["lookback"], "q": g["q"], "keep": g["keep%"], "all": g["all_avgR"], "dd": g["all_DD"],
                                 "passed": g["passed"]} for g in sq["grid"]]},
            "dukascopy": duka["squeeze"], "periods": duka["periods"],
            "recent": {"window": rv.get("window"), "filter": {k: rs.get(k) for k in ("n", "avgR", "totR", "maxDD_R")},
                       "system": {k: rn.get(k) for k in ("n", "avgR", "totR", "maxDD_R")}},
        },
        "ichimoku_dukascopy": duka["ichimoku_tk_4h_trailing"],
        "notes": notes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ab", help="e = event caches only, a = signals, b = filters/exits, c = squeeze checks, "
                                                     "s = summary json for the app")
    ap.add_argument("--tfs", default="15m,1h,4h")
    ap.add_argument("--no-15m-rolling", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    df15 = load()
    OUT.mkdir(exist_ok=True)
    tfs = tuple(args.tfs.split(","))
    rolling_tfs = ("1h", "4h") if args.no_15m_rolling else ("1h", "4h", "15m")
    if "e" in args.part:
        for tf in tfs:
            tool_events(df15, tf, tf in rolling_tfs)
    if "a" in args.part:
        res = part_a(df15, tfs, rolling_tfs)
        res.to_csv(OUT / "XAUUSD_tool_signals.csv", index=False)
        show = ["tool", "tf", "exit", "dev_n", "dev_avgR", "dev_t", "dev_PF", "val_n", "val_avgR", "test_n", "test_avgR", "all_avgR", "all_maxDD"]
        print(res.sort_values("dev_t", ascending=False)[show].head(40).to_string(index=False))
        sel = select(res)
        print("\ncandidates (dev t>=2, PF>=1.15):")
        print(sel[show + ["passed"]].to_string(index=False))
    if "s" in args.part:
        out = part_s()
        (OUT / "XAUUSD_tool_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps({k: out[k] for k in ("counts", "system")}, ensure_ascii=False))
        for k, v in out["notes"].items():
            print(k, v)
    if "c" in args.part:
        pc = part_c(df15)
        (OUT / "XAUUSD_squeeze_filter_check.json").write_text(json.dumps(pc, ensure_ascii=False, indent=1, default=float))
        g = pd.DataFrame(pc["grid"])
        print(g[["name", "keep%", "dev_avgR", "val_avgR", "test_avgR", "all_avgR", "all_totR", "all_DD", "passed"]].to_string(index=False))
        print(json.dumps({k: pc[k] for k in ("perm", "base_kept", "base_dropped", "by_dir")}, ensure_ascii=False, indent=1))
    if "b" in args.part:
        fb, extra = part_b(df15)
        extra["X"].to_pickle(ROOT / "data" / "cache" / "tool_filter_features.pkl")
        fb.to_csv(OUT / "XAUUSD_tool_filters.csv", index=False)
        show = ["name", "kind", "n", "keep%"] + [f"{p}_{k}" for p in ("dev", "val", "test") for k in ("avgR", "base")] + \
               ["all_avgR", "all_totR", "all_DD", "all_base_totR", "all_base_DD", "passed"]
        print(fb[show].to_string(index=False))


if __name__ == "__main__":
    main()
