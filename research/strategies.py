"""Candidate strategy families for research. Each returns (sig, stop_long, stop_short)
as numpy arrays aligned to the 15m base bars; sig is +1/-1/0 evaluated at bar close."""
from __future__ import annotations

import numpy as np
import pandas as pd


def entry_window(f: pd.DataFrame, start_ny: float, end_ny: float, friday_cut: float = 12.0) -> np.ndarray:
    """True if the bar CLOSES inside [start_ny, end_ny) New York time (Mon-Fri)."""
    close_h = (f["ny_h"].to_numpy() + 0.25) % 24
    dow = f["dow"].to_numpy()
    ok = (close_h > start_ny) & (close_h <= end_ny) & (dow <= 4)
    ok &= ~((dow == 4) & (close_h > friday_cut))
    return ok


def session_exit(f: pd.DataFrame, exit_ny: float = 16.75) -> np.ndarray:
    """Force flat at the close of the bar that ends at `exit_ny` New York time."""
    close_h = (f["ny_h"].to_numpy() + 0.25) % 24
    return np.isclose(close_h, exit_ny)


def _clip_stops(entry_ref, stop_l, stop_s, atr, min_atr, max_atr):
    risk_l = entry_ref - stop_l
    risk_s = stop_s - entry_ref
    stop_l = np.where(risk_l < min_atr * atr, entry_ref - min_atr * atr, stop_l)
    stop_s = np.where(risk_s < min_atr * atr, entry_ref + min_atr * atr, stop_s)
    bad_l = (entry_ref - stop_l) > max_atr * atr
    bad_s = (stop_s - entry_ref) > max_atr * atr
    return stop_l, stop_s, bad_l, bad_s


def trend_dir(f: pd.DataFrame, tf: str = "h1", mode: str = "ema") -> np.ndarray:
    """+1 / -1 / 0 trend on a higher timeframe (completed bars only)."""
    if mode == "ema":
        e20, e50 = f[f"{tf}_ema20"].to_numpy(), f[f"{tf}_ema50"].to_numpy()
        cl, sl = f[f"{tf}_close"].to_numpy(), f[f"{tf}_ema50_slope"].to_numpy()
        up = (e20 > e50) & (cl > e50) & (sl > 0)
        dn = (e20 < e50) & (cl < e50) & (sl < 0)
    elif mode == "st":
        d = f[f"{tf}_st_dir"].to_numpy()
        up, dn = d > 0, d < 0
    elif mode == "ema200":
        cl, e200, e50 = f[f"{tf}_close"].to_numpy(), f[f"{tf}_ema200"].to_numpy(), f[f"{tf}_ema50"].to_numpy()
        up = (cl > e200) & (e50 > e200)
        dn = (cl < e200) & (e50 < e200)
    else:
        raise ValueError(mode)
    return np.where(up, 1, np.where(dn, -1, 0))


# ---------------------------------------------------------------- family A
def trend_pullback(df, f, tf="h1", trend_mode="ema", adx_min=20.0, lookback=6,
                   rsi_gate=50.0, need_h4=False, window=(3.0, 12.0), stop_buf=0.25,
                   min_atr=1.0, max_atr=3.0):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = f["atr"].to_numpy()
    e20 = f["ema20"].to_numpy()
    rsi = f["rsi"].to_numpy()
    tdir = trend_dir(f, tf, trend_mode)
    if need_h4:
        t4 = trend_dir(f, "h4", "ema")
        tdir = np.where(tdir == t4, tdir, 0)
    strong = f[f"{tf}_adx"].to_numpy() >= adx_min
    lo_n = pd.Series(l).rolling(lookback).min().to_numpy()
    hi_n = pd.Series(h).rolling(lookback).max().to_numpy()
    touched_dn = pd.Series(l <= e20).rolling(lookback).max().to_numpy() > 0  # pullback to ema20 (long)
    touched_up = pd.Series(h >= e20).rolling(lookback).max().to_numpy() > 0
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    long_trig = touched_dn & (c > e20) & (c > prev_h) & (c > o) & (rsi > rsi_gate)
    short_trig = touched_up & (c < e20) & (c < prev_l) & (c < o) & (rsi < 100 - rsi_gate)
    win = entry_window(f, *window)
    stop_l = lo_n - stop_buf * atr
    stop_s = hi_n + stop_buf * atr
    stop_l, stop_s, bad_l, bad_s = _clip_stops(c, stop_l, stop_s, atr, min_atr, max_atr)
    sig = np.where(win & strong & (tdir > 0) & long_trig & ~bad_l, 1,
                   np.where(win & strong & (tdir < 0) & short_trig & ~bad_s, -1, 0))
    return sig, stop_l, stop_s


# ---------------------------------------------------------------- family B
def donchian_breakout(df, f, n=24, tf="h1", trend_mode="ema", adx_min=0.0, window=(3.0, 12.0),
                      stop_atr=2.0, use_trend=True, er_min=0.0):
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    atr = f["atr"].to_numpy()
    up = pd.Series(h).rolling(n).max().shift(1).to_numpy()
    dn = pd.Series(l).rolling(n).min().shift(1).to_numpy()
    prev_c = np.roll(c, 1)
    long_trig = (c > up) & (prev_c <= up)
    short_trig = (c < dn) & (prev_c >= dn)
    ok = entry_window(f, *window)
    if adx_min > 0:
        ok &= f[f"{tf}_adx"].to_numpy() >= adx_min
    if er_min > 0:
        ok &= f[f"{tf}_er"].to_numpy() >= er_min
    if use_trend:
        tdir = trend_dir(f, tf, trend_mode)
    else:
        tdir = np.where(long_trig, 1, np.where(short_trig, -1, 0))
    stop_l = c - stop_atr * atr
    stop_s = c + stop_atr * atr
    sig = np.where(ok & (tdir > 0) & long_trig, 1, np.where(ok & (tdir < 0) & short_trig, -1, 0))
    return sig, stop_l, stop_s


# ---------------------------------------------------------------- family C
def asian_range_breakout(df, f, range_ldn=(0.0, 7.0), trade_ldn=(7.0, 11.0), buf_atr=0.1,
                         min_w=0.3, max_w=1.5, stop_mode="mid", use_trend=False, tf="h4"):
    """Range = London-time [range_ldn) of the current trading day; first close beyond the
    range during trade_ldn triggers. Width measured in daily ATRs."""
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    atr = f["atr"].to_numpy()
    datr = f["d1_atr"].to_numpy()
    ldn_h = f["ldn_h"].to_numpy()
    date_ldn = df.index.tz_convert("Europe/London").normalize()
    in_rng = (ldn_h >= range_ldn[0]) & (ldn_h < range_ldn[1])
    tmp = pd.DataFrame({"d": date_ldn, "h": np.where(in_rng, h, np.nan), "l": np.where(in_rng, l, np.nan)})
    g = tmp.groupby("d")
    # cumulative within the day so the range is only known once its bars have passed
    rh = g["h"].cummax().to_numpy()
    rl = g["l"].cummin().to_numpy()
    rh = pd.Series(rh).groupby(tmp["d"].to_numpy()).ffill().to_numpy()
    rl = pd.Series(rl).groupby(tmp["d"].to_numpy()).ffill().to_numpy()
    close_ldn = (ldn_h + 0.25)
    in_trade = (close_ldn > trade_ldn[0]) & (close_ldn <= trade_ldn[1]) & (f["dow"].to_numpy() <= 4)
    width = rh - rl
    wok = (width >= min_w * datr) & (width <= max_w * datr)
    long_trig = c > rh + buf_atr * atr
    short_trig = c < rl - buf_atr * atr
    # only the first breakout of the day
    first = np.zeros(len(c), dtype=bool)
    fired = {}
    dvals = tmp["d"].to_numpy()
    for i in np.flatnonzero(in_trade & wok & (long_trig | short_trig)):
        if dvals[i] not in fired:
            fired[dvals[i]] = True
            first[i] = True
    if use_trend:
        tdir = trend_dir(f, tf, "ema")
        long_ok, short_ok = tdir > 0, tdir < 0
    else:
        long_ok = short_ok = np.ones(len(c), dtype=bool)
    mid = (rh + rl) / 2
    if stop_mode == "mid":
        stop_l, stop_s = mid, mid
    else:  # opposite side
        stop_l, stop_s = rl, rh
    sig = np.where(first & long_trig & long_ok, 1, np.where(first & short_trig & short_ok, -1, 0))
    return sig, stop_l, stop_s


# ---------------------------------------------------------------- family E
def htf_trend_flip(df, f, tf="h1", mode="st", window=(3.0, 12.0), stop_atr=2.0):
    """Enter when the higher-timeframe trend state turns (Supertrend flip or EMA regime change)."""
    c = df["close"].to_numpy()
    atr = f["atr"].to_numpy()
    tdir = trend_dir(f, tf, mode)
    prev = np.roll(tdir, 1)
    flip_up = (tdir > 0) & (prev <= 0)
    flip_dn = (tdir < 0) & (prev >= 0)
    ok = entry_window(f, *window)
    stop_l = c - stop_atr * atr
    stop_s = c + stop_atr * atr
    sig = np.where(ok & flip_up, 1, np.where(ok & flip_dn, -1, 0))
    return sig, stop_l, stop_s
