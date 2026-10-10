"""High-frequency ("積極模式") strategy research (docs/RESEARCH.md section 10).

The user asked for 5-10 trades a week instead of the ~0.5 of the trend-breakout system,
and suggested channel / grid trading. This script tests several families of intraday
strategies on gold 15m bars 2005-2025/3, all with the same conventions:

* signals on bar CLOSE, market entries at the NEXT bar open (limit entries fill at the limit
  price, never on the bar that placed them); stop checked before target inside a bar;
* everything is closed at 16:45 New York (no overnight fees), no new entries 15:30-18:00 NY;
* costs: spread 0.012 % + slippage 0.005 % per market fill (limit take-profits: no slippage).

Families (parameters fixed before looking at the results):
  F1 順勢回檔   buy dips / sell rallies in the daily trend (1h RSI(2), Bollinger, RSI(14))
  F2 區間網格   fade the edges of a channel (1h Bollinger or 24h high/low) when ADX is low
  F3 加速突破   the system's breakout idea with 8h / 12h channels and no quality filters
  F4 開盤突破   London / New York opening-range breakouts
  F5 亞洲時段   hold gold during the Asian session (the 'gold rises in Asia' pattern)

Protocol: a variant is a candidate if in 2005-2016 it trades >= 1 time a week, its mean R
has t >= 2.5 and PF >= 1.10; it passes if 2017-2020 and 2021-2025/3 are also positive.
Passing pieces are combined (one position at a time) into the active algorithm.

    python research/active_strategies.py               # all families -> results/XAUUSD_active_variants.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import indicators as ind  # noqa: E402
from goldsignal.backtest import njit  # noqa: E402  (numba if installed)
from goldsignal.bars import resample  # noqa: E402
from goldsignal.plans import Clock  # noqa: E402

PERIODS = {"dev": ("2005-01-01", "2016-12-31"), "val": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2025-03-21")}
SPREAD, SLIP = 0.00012, 0.00005
NY = "America/New_York"
OUT = ROOT / "results"
EOD_H = 16.75                 # everything is closed at the 16:45 NY bar close
NO_ENTRY = (15.5, 18.0)       # no new entries between these NY hours (bar close)


def load(path: str | None = None) -> pd.DataFrame:
    pq = Path(path) if path else ROOT / "data" / "cache_xau15.parquet"
    if pq.exists():
        return pd.read_parquet(pq).loc[:"2025-03-21"]
    from goldsignal.bars import load_kaggle_15m
    return load_kaggle_15m(ROOT / "data" / "raw" / "XAU_15m_data.csv").loc[:"2025-03-21"]


# =============================================================================== simulator

EXIT_NAMES = {0: "stop", 1: "target", 2: "eod", 3: "time", 4: "rule", 6: "end"}


@njit(cache=True)
def _sim(o, h, l, c, spread, slip, sig, stop_px, tgt_px, lim_px, lim_bars, mode, exit_l, exit_s,
         trail, atr, max_bars, force_exit, no_entry, cooldown):
    """One position at a time. Per-signal stop / target / limit price / exit mode / trail / max bars."""
    n = len(o)
    out = np.zeros((n // 2 + 1, 10))   # sig_i, entry_i, exit_i, dir, entry, exit, risk, R, reason, mfeR
    k = 0
    i = 0
    next_ok = 0
    while i < n - 1:
        d = sig[i]
        if d == 0 or i < next_ok or no_entry[i] or force_exit[i]:
            i += 1
            continue
        stop = stop_px[i]
        # ---- entry
        j0 = -1
        entry = np.nan
        if np.isnan(lim_px[i]):                       # market at the next open
            j0 = i + 1
            entry = o[j0] + spread[j0] + slip[j0] if d > 0 else o[j0] - slip[j0]
        else:                                          # resting limit order
            lim = lim_px[i]
            for jj in range(i + 1, min(n, i + 1 + lim_bars)):
                if jj > i + 1 and (force_exit[jj - 1] or no_entry[jj - 1]):
                    break                                   # pending orders are cancelled at the day's end
                if d > 0 and l[jj] + spread[jj] <= lim:
                    j0 = jj
                    entry = min(lim, o[jj] + spread[jj])
                    break
                if d < 0 and h[jj] >= lim:
                    j0 = jj
                    entry = max(lim, o[jj])
                    break
        if j0 < 0:
            i += 1
            continue
        risk = entry - stop if d > 0 else stop - entry
        if not (risk > 0):
            i += 1
            continue
        tgt = tgt_px[i]
        if not np.isnan(tgt) and ((d > 0 and tgt <= entry) or (d < 0 and tgt >= entry)):
            tgt = np.nan
        limit_fill = not np.isnan(lim_px[i])
        best = entry
        exit_px = np.nan
        reason = 6
        j = j0
        while j < n:
            sp = spread[j]
            if d > 0:
                if j > j0 and o[j] <= stop:
                    exit_px = o[j] - slip[j]
                    reason = 0
                    break
                if l[j] <= stop:
                    exit_px = stop - slip[j]
                    reason = 0
                    break
                if not np.isnan(tgt) and not (limit_fill and j == j0) and h[j] >= tgt:
                    exit_px = tgt
                    reason = 1
                    break
                if h[j] > best:
                    best = h[j]
            else:
                if j > j0 and o[j] + sp >= stop:
                    exit_px = o[j] + sp + slip[j]
                    reason = 0
                    break
                if h[j] + sp >= stop:
                    exit_px = stop + slip[j]
                    reason = 0
                    break
                if not np.isnan(tgt) and not (limit_fill and j == j0) and l[j] + sp <= tgt:
                    exit_px = tgt
                    reason = 1
                    break
                if l[j] + sp < best:
                    best = l[j] + sp
            held = j - j0 + 1
            rule = (d > 0 and exit_l[mode[i], j]) or (d < 0 and exit_s[mode[i], j])
            if force_exit[j] or (max_bars[i] > 0 and held >= max_bars[i]) or rule:
                exit_px = c[j] - slip[j] if d > 0 else c[j] + sp + slip[j]
                reason = 2 if force_exit[j] else (3 if (max_bars[i] > 0 and held >= max_bars[i]) else 4)
                break
            if trail[i] > 0 and not np.isnan(atr[j]):
                ts = best - d * trail[i] * atr[j]
                if (d > 0 and ts > stop) or (d < 0 and ts < stop):
                    stop = ts
            j += 1
        if j >= n:
            j = n - 1
            exit_px = c[j] - slip[j] if d > 0 else c[j] + spread[j] + slip[j]
            reason = 6
        r = (exit_px - entry) / risk if d > 0 else (entry - exit_px) / risk
        out[k, 0] = i
        out[k, 1] = j0
        out[k, 2] = j
        out[k, 3] = d
        out[k, 4] = entry
        out[k, 5] = exit_px
        out[k, 6] = risk
        out[k, 7] = r
        out[k, 8] = reason
        out[k, 9] = (best - entry) / risk if d > 0 else (entry - best) / risk
        k += 1
        next_ok = j + cooldown + 1
        i = j + 1
        if k >= out.shape[0]:
            break
    return out[:k]


@dataclass
class Orders:
    """Signals plus per-signal trade parameters (arrays over the 15m bars)."""
    sig: np.ndarray
    stop: np.ndarray
    tgt: np.ndarray
    lim: np.ndarray
    trail: np.ndarray
    max_bars: np.ndarray
    mode: np.ndarray
    lim_bars: int = 4
    name: str = ""
    tag: np.ndarray | None = None      # which strategy produced each signal (combined runs)
    meta: dict = field(default_factory=dict)


def empty_orders(n: int, name: str = "") -> Orders:
    return Orders(sig=np.zeros(n, np.int64), stop=np.full(n, np.nan), tgt=np.full(n, np.nan), lim=np.full(n, np.nan),
                  trail=np.zeros(n), max_bars=np.zeros(n, np.int64), mode=np.zeros(n, np.int64), name=name,
                  tag=np.full(n, -1, np.int64))


def simulate(F: dict, od: Orders, cost_mult: float = 1.0, cooldown: int = 0) -> pd.DataFrame:
    px = F["open"]
    sp = px * SPREAD * cost_mult
    sl = px * SLIP * cost_mult
    res = _sim(F["open"], F["high"], F["low"], F["close"], sp, sl, od.sig.astype(np.int64), od.stop.astype(float),
               od.tgt.astype(float), od.lim.astype(float), int(od.lim_bars), od.mode.astype(np.int64), F["exit_l"], F["exit_s"],
               od.trail.astype(float), F["atr1"], od.max_bars.astype(np.int64), F["eod"], F["no_entry"], int(cooldown))
    cols = ["sig_i", "entry_i", "exit_i", "dir", "entry", "exit", "risk", "R", "reason", "mfeR"]
    t = pd.DataFrame(res, columns=cols)
    if t.empty:
        return t
    for col in ("sig_i", "entry_i", "exit_i", "dir"):
        t[col] = t[col].astype(int)
    t["reason"] = t["reason"].astype(int).map(EXIT_NAMES)
    idx = F["index"]
    t["entry_time"] = idx[t["entry_i"].to_numpy()]
    t["exit_time"] = idx[t["exit_i"].to_numpy()]
    t["bars"] = t["exit_i"] - t["entry_i"] + 1
    if od.tag is not None:
        t["tag"] = od.tag[t["sig_i"].to_numpy()]
    return t[t["reason"] != "end"].reset_index(drop=True)


# =============================================================================== features

def _rank_pct(s: pd.Series, n: int = 200) -> pd.Series:
    return s.rolling(n, min_periods=50).rank(pct=True)


def _trend(close, e20, e50) -> np.ndarray:
    return np.where((close > e20) & (e20 > e50), 1, np.where((close < e20) & (e20 < e50), -1, 0))


def features(df: pd.DataFrame) -> dict:
    """15m arrays plus completed-1h / 4h / daily values as known at each 15m bar close."""
    F = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    F["index"] = df.index
    n = len(df)
    F["atr15"] = ind.atr(df, 14).to_numpy()
    c15 = df["close"]
    F["rsi2_15"] = ind.rsi(c15, 2).to_numpy()
    # ---- 1h
    h1 = resample(df, "1h")
    hf = pd.DataFrame(index=h1.index)
    hf["close"], hf["high"], hf["low"], hf["open"] = h1["close"], h1["high"], h1["low"], h1["open"]
    hf["e20"], hf["e50"] = ind.ema(h1["close"], 20), ind.ema(h1["close"], 50)
    hf["atr"] = ind.atr(h1, 14)
    hf["adx"] = ind.adx(h1, 14)["adx"]
    hf["rsi2"], hf["rsi14"] = ind.rsi(h1["close"], 2), ind.rsi(h1["close"], 14)
    bb = ind.bollinger(h1["close"], 20, 2.0)
    hf["bb_up"], hf["bb_mid"], hf["bb_lo"] = bb["upper"], bb["mid"], bb["lower"]
    hf["bw_pct"] = _rank_pct((bb["upper"] - bb["lower"]) / bb["mid"])
    hf["don_hi"] = h1["high"].rolling(24, min_periods=24).max().shift(1)     # previous 24 hours, not this bar
    hf["don_lo"] = h1["low"].rolling(24, min_periods=24).min().shift(1)
    k1 = h1.index.searchsorted(df.index + pd.Timedelta(minutes=15) - pd.Timedelta(hours=1), side="right") - 1
    ok1 = k1 >= 0
    for col in hf.columns:
        v = np.full(n, np.nan)
        v[ok1] = hf[col].to_numpy(float)[k1[ok1]]
        F["h1_" + col] = v
    F["h1_new"] = np.r_[False, k1[1:] != k1[:-1]] & ok1        # a 1h bar has just closed at this 15m close
    F["atr1"] = F["h1_atr"]
    # ---- 4h and daily trend
    h4 = resample(h1, "4h")
    t4 = pd.Series(_trend(h4["close"], ind.ema(h4["close"], 20), ind.ema(h4["close"], 50)), index=h4.index)
    k4 = h4.index.searchsorted(df.index + pd.Timedelta(minutes=15) - pd.Timedelta(hours=4), side="right") - 1
    F["trend4"] = np.where(k4 >= 0, t4.to_numpy()[np.maximum(k4, 0)], 0)
    d1 = resample(h1, "1D")
    de20, de50 = ind.ema(d1["close"], 20), ind.ema(d1["close"], 50)
    td = pd.Series(_trend(d1["close"], de20, de50), index=d1.index)
    kd = d1.index.searchsorted(df.index + pd.Timedelta(minutes=15) - pd.Timedelta(days=1), side="right") - 1
    okd = kd >= 0
    F["trend_d"] = np.where(okd, td.to_numpy()[np.maximum(kd, 0)], 0)
    above = np.sign(d1["close"] - de50).fillna(0).to_numpy()
    F["side_d50"] = np.where(okd, above[np.maximum(kd, 0)], 0)          # daily close above / below EMA50
    # ---- clock
    clock = Clock.build(df.index)
    F["session"] = clock.session
    F["ny_close_h"] = clock.ny_close_h
    F["dow"] = clock.sess_dow
    F["eod"] = clock.eod_mask(EOD_H)
    a, b = NO_ENTRY
    F["no_entry"] = (F["ny_close_h"] >= a) & (F["ny_close_h"] < b)
    # exit-condition modes (rows): 0 none, 1 1h RSI(2) back above 70 / below 30
    F["exit_l"] = np.zeros((2, n), np.bool_)
    F["exit_s"] = np.zeros((2, n), np.bool_)
    F["exit_l"][1] = F["h1_new"] & (F["h1_rsi2"] > 70)
    F["exit_s"][1] = F["h1_new"] & (F["h1_rsi2"] < 30)
    return F


# =============================================================================== strategies

def f1_pullback(F: dict, trigger: str, trend: str, exit_: str) -> Orders:
    n = len(F["close"])
    od = empty_orders(n, f"F1 順勢回檔 {trigger}/{trend}/{exit_}")
    t = F["trend_d"] if trend == "d" else np.where(F["trend_d"] == F["trend4"], F["trend_d"], 0)
    new = F["h1_new"]
    if trigger == "rsi2":
        lo, hi = F["h1_rsi2"] < 10, F["h1_rsi2"] > 90
    elif trigger == "bb":
        lo, hi = F["h1_close"] < F["h1_bb_lo"], F["h1_close"] > F["h1_bb_up"]
    else:  # rsi14
        lo, hi = F["h1_rsi14"] < 40, F["h1_rsi14"] > 60
    sig = np.where(new & (t > 0) & lo, 1, np.where(new & (t < 0) & hi, -1, 0))
    c, a = F["close"], F["atr1"]
    if exit_ == "mid":
        od.stop = c - sig * 2.0 * a
        od.tgt = F["h1_bb_mid"].copy()
        od.max_bars[:] = 32
    elif exit_ == "1R":
        od.stop = c - sig * 1.5 * a
        od.tgt = c + sig * 1.5 * a          # about +1R from the next open
    else:  # rsi
        od.stop = c - sig * 2.0 * a
        od.mode[:] = 1
        od.max_bars[:] = 48
    od.sig = sig.astype(np.int64)
    return od


def f2_range(F: dict, regime: str, channel: str, entry: str, exit_: str, direction: str) -> Orders:
    n = len(F["close"])
    od = empty_orders(n, f"F2 區間網格 {regime}/{channel}/{entry}/{exit_}/{direction}")
    reg = F["h1_adx"] < 20
    if regime == "adx20bw":
        reg &= F["h1_bw_pct"] < 0.5
    if channel == "bb":
        up, lo, mid = F["h1_bb_up"], F["h1_bb_lo"], F["h1_bb_mid"]
    else:
        up, lo = F["h1_don_hi"], F["h1_don_lo"]
        mid = (up + lo) / 2
    a = F["atr1"]
    width_ok = (up - lo) >= 1.5 * a                       # a channel worth trading
    new = F["h1_new"] & reg & width_ok
    allow_l = np.ones(n, bool) if direction == "both" else F["side_d50"] >= 0
    allow_s = np.ones(n, bool) if direction == "both" else F["side_d50"] <= 0
    c, hh, ll = F["h1_close"], F["h1_high"], F["h1_low"]
    if entry == "touch":   # the last 1h bar poked through the edge and closed back inside
        long_ = new & allow_l & (ll <= lo + 0.1 * a) & (c > lo) & (c < mid)
        short = new & allow_s & (hh >= up - 0.1 * a) & (c < up) & (c > mid)
        sig = np.where(long_, 1, np.where(short, -1, 0))
        edge = np.where(sig > 0, lo, up)
    else:                  # resting limit order at the nearer edge for the next hour
        inside = (c > lo + 0.25 * a) & (c < up - 0.25 * a)
        near_lo = (c - lo) <= (up - c)
        sig = np.where(new & inside & near_lo & allow_l, 1, np.where(new & inside & ~near_lo & allow_s, -1, 0))
        edge = np.where(sig > 0, lo, up)
        od.lim = np.where(sig != 0, edge, np.nan)
        od.lim_bars = 4
    od.stop = edge - sig * 1.0 * a
    od.tgt = mid.copy() if exit_ == "mid" else np.where(sig > 0, up - 0.1 * a, lo + 0.1 * a)
    od.max_bars[:] = 32
    od.sig = sig.astype(np.int64)
    return od


def f3_breakout(F: dict, hours: int, trend: str, exit_: str) -> Orders:
    n = len(F["close"])
    od = empty_orders(n, f"F3 加速突破 {hours}h/{trend}/{exit_}")
    h, l, c = (pd.Series(F[k]) for k in ("high", "low", "close"))
    m = hours * 4
    hi = h.rolling(m, min_periods=m).max().shift(1).to_numpy()
    lo = l.rolling(m, min_periods=m).min().shift(1).to_numpy()
    cc, cp = F["close"], np.r_[np.nan, F["close"][:-1]]
    rng = np.maximum(F["high"] - F["low"], 1e-9)
    up = (cc > hi) & (cp <= np.r_[np.nan, hi[:-1]]) & ((cc - F["low"]) / rng >= 0.7)
    dn = (cc < lo) & (cp >= np.r_[np.nan, lo[:-1]]) & ((F["high"] - cc) / rng >= 0.7)
    t = F["trend_d"] if trend == "d" else np.where(F["trend_d"] == F["trend4"], F["trend_d"], 0)
    sig = np.where(up & (t > 0), 1, np.where(dn & (t < 0), -1, 0))
    a = F["atr1"]
    od.stop = cc - sig * 1.5 * a
    if exit_ == "trail":
        od.trail[:] = 3.0
    else:
        od.tgt = cc + sig * 1.5 * 1.5 * a
    od.sig = sig.astype(np.int64)
    return od


def f4_orb(F: dict, session: str, trend: str) -> Orders:
    n = len(F["close"])
    od = empty_orders(n, f"F4 開盤突破 {session}/{trend}")
    hcl = F["ny_close_h"]
    if session == "london":
        in_rng = (hcl > 2.0) & (hcl <= 3.0)
        window = (hcl > 3.0) & (hcl <= 11.0)
    else:
        in_rng = (hcl > 8.0) & (hcl <= 9.5)
        window = (hcl > 9.5) & (hcl <= 14.0)
    sess = F["session"]
    hi = pd.Series(np.where(in_rng, F["high"], np.nan)).groupby(sess).transform(lambda s: s.cummax().ffill()).to_numpy()
    lo = pd.Series(np.where(in_rng, F["low"], np.nan)).groupby(sess).transform(lambda s: s.cummin().ffill()).to_numpy()
    c = F["close"]
    up = window & (c > hi)
    dn = window & (c < lo)
    first_up = up & (pd.Series(up.astype(int)).groupby(sess).cumsum().to_numpy() == 1)
    first_dn = dn & (pd.Series(dn.astype(int)).groupby(sess).cumsum().to_numpy() == 1)
    t = F["trend_d"] if trend == "d" else np.ones(n)
    allow_s = (t < 0) if trend == "d" else np.ones(n, bool)
    sig = np.where(first_up & (t > 0), 1, np.where(first_dn & allow_s, -1, 0))
    a = F["atr1"]
    od.stop = np.where(sig > 0, np.minimum(lo, c - 0.5 * a), np.maximum(hi, c + 0.5 * a))   # at least 0.5 ATR(1h)
    risk = np.abs(c - od.stop)
    od.tgt = c + sig * 1.5 * risk
    od.sig = sig.astype(np.int64)
    return od


def f5_asia(F: dict, variant: str) -> Orders:
    n = len(F["close"])
    od = empty_orders(n, f"F5 亞洲時段 {variant}")
    sess = F["session"]
    first = np.r_[True, sess[1:] != sess[:-1]]          # first bar of each trading day (around 18:00 NY)
    first &= (F["ny_close_h"] > 18.0) & (F["ny_close_h"] <= 19.0)
    td = F["trend_d"]
    if variant == "long":
        sig = np.where(first, 1, 0)
    elif variant == "long_not_down":
        sig = np.where(first & (td >= 0), 1, 0)
    else:
        sig = np.where(first & (td > 0), 1, np.where(first & (td < 0), -1, 0))
    a = F["atr1"]
    od.stop = F["close"] - sig * 2.0 * a
    od.mode[:] = 2                                        # exit at the 02:00 NY bar close
    od.sig = sig.astype(np.int64)
    return od


def f6_rangebreak(F: dict, channel: str, exit_: str) -> Orders:
    """The mirror of F2: when the market is ranging (1h ADX < 20) trade the BREAK of the channel."""
    n = len(F["close"])
    od = empty_orders(n, f"F6 區間突破 {channel}/{exit_}")
    if channel == "bb":
        up, lo = F["h1_bb_up"], F["h1_bb_lo"]
    else:
        up, lo = F["h1_don_hi"], F["h1_don_lo"]
    a = F["atr1"]
    new = F["h1_new"] & (F["h1_adx"] < 20)
    c = F["h1_close"]
    prev = np.r_[np.nan, c[:-1]]
    sig = np.where(new & (c > up + 0.1 * a), 1, np.where(new & (c < lo - 0.1 * a), -1, 0))
    cc = F["close"]
    od.stop = cc - sig * 1.5 * a
    if exit_ == "trail":
        od.trail[:] = 3.0
    else:
        od.tgt = cc + sig * 1.5 * 1.5 * a
    od.sig = sig.astype(np.int64)
    del prev
    return od


def f9_prevday(F: dict, trend: str, exit_: str) -> Orders:
    """Break of the previous trading day's high / low in the direction of the daily trend."""
    n = len(F["close"])
    od = empty_orders(n, f"F9 昨日高低突破 {trend}/{exit_}")
    sess = F["session"]
    day_hi = pd.Series(F["high"]).groupby(sess).max()
    day_lo = pd.Series(F["low"]).groupby(sess).min()
    prev_hi = pd.Series(sess).map(day_hi.shift(1)).to_numpy()
    prev_lo = pd.Series(sess).map(day_lo.shift(1)).to_numpy()
    c = F["close"]
    up = c > prev_hi
    dn = c < prev_lo
    first_up = up & (pd.Series(up.astype(int)).groupby(sess).cumsum().to_numpy() == 1)
    first_dn = dn & (pd.Series(dn.astype(int)).groupby(sess).cumsum().to_numpy() == 1)
    t = F["trend_d"] if trend == "d" else np.where(F["trend_d"] == F["trend4"], F["trend_d"], 0)
    sig = np.where(first_up & (t > 0), 1, np.where(first_dn & (t < 0), -1, 0))
    a = F["atr1"]
    od.stop = c - sig * 1.5 * a
    if exit_ == "trail":
        od.trail[:] = 3.0
    else:
        od.tgt = c + sig * 1.5 * 1.5 * a
    od.sig = sig.astype(np.int64)
    return od


def variants(F: dict) -> list[Orders]:
    out = []
    for trig in ("rsi2", "bb", "rsi14"):
        for tr in ("d", "d4"):
            for ex in ("mid", "1R", "rsi"):
                out.append(f1_pullback(F, trig, tr, ex))
    for reg in ("adx20", "adx20bw"):
        for ch in ("bb", "don"):
            for en in ("touch", "limit"):
                for ex in ("mid", "far"):
                    for dr in ("both", "trend"):
                        out.append(f2_range(F, reg, ch, en, ex, dr))
    for hrs in (8, 12):
        for tr in ("d", "d4"):
            for ex in ("trail", "1.5R"):
                out.append(f3_breakout(F, hrs, tr, ex))
    for se in ("london", "ny"):
        for tr in ("d", "none"):
            out.append(f4_orb(F, se, tr))
    for v in ("long", "long_not_down", "trend"):
        out.append(f5_asia(F, v))
    for ch in ("bb", "don"):
        for ex in ("trail", "1.5R"):
            out.append(f6_rangebreak(F, ch, ex))
    for tr in ("d", "d4"):
        for ex in ("trail", "1.5R"):
            out.append(f9_prevday(F, tr, ex))
    return out


def add_asia_exit(F: dict) -> None:
    """exit mode 2: close at the 02:00 NY bar close (London open)."""
    n = len(F["close"])
    m = np.zeros(n, np.bool_)
    h = F["ny_close_h"]
    m[(h > 1.99) & (h < 2.01)] = True
    el = np.zeros((3, n), np.bool_)
    es = np.zeros((3, n), np.bool_)
    el[:2], es[:2] = F["exit_l"], F["exit_s"]
    el[2], es[2] = m, m
    F["exit_l"], F["exit_s"] = el, es


# =============================================================================== evaluation

def stats(tr: pd.DataFrame, a: str, b: str) -> dict:
    m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
    r = tr.loc[m, "R"].to_numpy()
    weeks = (pd.Timestamp(b) - pd.Timestamp(a)).days / 7.0
    if len(r) < 2:
        return {"n": int(len(r)), "per_week": round(len(r) / weeks, 2)}
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq))
    sd = r.std(ddof=1)
    w, lo = r[r > 0].sum(), -r[r <= 0].sum()
    return {"n": int(len(r)), "per_week": round(len(r) / weeks, 2), "win%": round(100 * float((r > 0).mean()), 1),
            "avgR": round(float(r.mean()), 4), "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else 0.0,
            "PF": round(float(w / lo), 3) if lo > 0 else 99.0, "maxDD": round(dd, 1), "totR": round(float(r.sum()), 1),
            "R_per_year": round(float(r.sum()) / (weeks / 52.18), 1)}


def evaluate(tr: pd.DataFrame) -> dict:
    out = {p: stats(tr, a, b) for p, (a, b) in PERIODS.items()}
    out["all"] = stats(tr, PERIODS["dev"][0], PERIODS["test"][1])
    return out


def flat(name: str, ev: dict, extra: dict | None = None) -> dict:
    row = {"name": name, **(extra or {})}
    for p, d in ev.items():
        for k, v in d.items():
            row[f"{p}_{k}"] = v
    return row


def select(df: pd.DataFrame) -> pd.Series:
    cand = (df["dev_per_week"] >= 1.0) & (df["dev_t"] >= 2.5) & (df["dev_PF"] >= 1.10)
    return cand & (df["val_avgR"] > 0) & (df["test_avgR"] > 0)


def system_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """How far can the validated trend-breakout system itself be pushed? Shorter breakout channels,
    no squeeze filter, and 'every signal is its own trade' (adding a position on each new signal)."""
    from dataclasses import replace
    from goldsignal import strategy as S
    from goldsignal.instruments import INSTRUMENTS
    from goldsignal.plans import simulate_plan
    inst = INSTRUMENTS["XAUUSD"]
    px = df["open"].to_numpy(float)
    sp, sl = px * inst.spread_pct, px * inst.slip_pct
    clock = Clock.build(df.index)
    rows = []
    for hours in (24, 16, 12, 8):
        for squeeze in (True, False):
            p = replace(inst.strategy_params(True, squeeze=squeeze), breakout_hours=hours)
            f = S.compute_features(df, params=p)
            sig = S.signals_from_rules(S.rule_table(df, f, p))
            atr1 = f["h1_atr"].to_numpy(float)
            plan = p.plan(swap=True)
            one = simulate_plan(df, sig, atr1, plan, sp, sl, clock)
            name = f"系統 {hours}h{'＋收窄' if squeeze else ''}"
            for label, tr in ((name, one), (name + "（每個訊號都進場）", None)):
                if tr is None:   # every signal its own trade (pyramiding): simulate each signal alone
                    parts = []
                    for i in np.flatnonzero(sig):
                        s1 = np.zeros_like(sig)
                        s1[i] = sig[i]
                        t1 = simulate_plan(df, s1, atr1, plan, sp, sl, clock)
                        if not t1.empty:
                            parts.append(t1)
                    tr = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
                tr = tr[tr["reason"] != "end"]
                ev = evaluate(tr)
                conc = 0
                if len(tr):
                    ev_t = np.r_[tr["entry_i"].to_numpy() if "entry_i" in tr else tr["sig_i"].to_numpy() + 1, tr["exit_i"].to_numpy() + 1]
                    ev_d = np.r_[np.ones(len(tr)), -np.ones(len(tr))]
                    order = np.lexsort((ev_d, ev_t))
                    conc = int(np.max(np.cumsum(ev_d[order])))
                rows.append(flat(label, ev, {"max_open": conc}))
                print(f"  {label}: {ev['all'].get('per_week')}/wk avgR {ev['all'].get('avgR')}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="a")
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    t0 = time.time()
    df = load()
    F = features(df)
    add_asia_exit(F)
    print(f"features {time.time() - t0:.0f}s, {len(df)} bars", flush=True)
    OUT.mkdir(exist_ok=True)
    if "a" in args.part:
        rows = []
        for od in variants(F):
            tr = simulate(F, od)
            ev = evaluate(tr)
            ev15 = evaluate(simulate(F, od, cost_mult=1.5))
            rows.append(flat(od.name, ev, {"all_avgR_cost1.5": ev15["all"].get("avgR")}))
        res = pd.DataFrame(rows)
        res["candidate"] = (res["dev_per_week"] >= 1.0) & (res["dev_t"] >= 2.5) & (res["dev_PF"] >= 1.10)
        res["passed"] = select(res)
        res.to_csv(OUT / "XAUUSD_active_variants.csv", index=False)
        show = ["name", "dev_per_week", "dev_avgR", "dev_t", "dev_PF", "val_avgR", "val_t", "test_avgR", "test_t",
                "all_per_week", "all_avgR", "all_R_per_year", "all_maxDD", "all_avgR_cost1.5", "candidate", "passed"]
        print(res[show].to_string(index=False))
        print(f"done {time.time() - t0:.0f}s")
    if "b" in args.part:
        fr = system_frontier(df)
        fr.to_csv(OUT / "XAUUSD_active_system_frontier.csv", index=False)
        show = ["name", "max_open", "dev_per_week", "dev_avgR", "val_avgR", "test_avgR", "all_per_week", "all_avgR", "all_t",
                "all_R_per_year", "all_maxDD"]
        print(fr[show].to_string(index=False))


if __name__ == "__main__":
    main()
