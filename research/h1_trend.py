"""Trend-following evaluated directly on 1h bars (signals at 1h close, next-bar entry)."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldsignal import indicators as ind
from goldsignal.bars import resample, align_htf
from goldsignal.backtest import ExitRules, simulate, metrics
from research.common import PERIODS, SPREAD_PCT, SLIP_PCT
from research.run_grid import load_cached, fmt

df15, _ = load_cached()
h = resample(df15, "1h")
d = resample(df15, "1D")
H1 = pd.Timedelta(hours=1); D1 = pd.Timedelta(days=1)
c = h["close"]
atr = ind.atr(h, 14)
ny = h.index.tz_convert("America/New_York")
close_ny = (ny.hour + 1) % 24
dow = ny.dayofweek
spread = h["open"].to_numpy() * SPREAD_PCT
slip = h["open"].to_numpy() * SLIP_PCT
dfeat = pd.DataFrame({"ema20": ind.ema(d["close"], 20), "ema50": ind.ema(d["close"], 50), "close": d["close"]}, index=d.index)
dal = align_htf(h.index, H1, dfeat, D1)
d_trend = np.where((dal["close"] > dal["ema20"]) & (dal["ema20"] > dal["ema50"]), 1, np.where((dal["close"] < dal["ema20"]) & (dal["ema20"] < dal["ema50"]), -1, 0))

def run(sig, sl, ss, rules, label):
    tr = simulate(h, sig, sl, ss, rules, spread, slip, atr=atr.to_numpy())
    rows = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
    print(f"{label}\n    {fmt(rows)}")
    return tr

win_all = (dow <= 4) & ~((dow == 4) & (close_ny >= 12))
for n, stop_k, trail, dfilt in itertools.product([20, 40, 80], [2.0, 3.0], [3.0, 4.5], [False, True]):
    up = h["high"].rolling(n).max().shift(1); dn = h["low"].rolling(n).min().shift(1)
    lt = (c > up) & (c.shift(1) <= up.shift(1)); st = (c < dn) & (c.shift(1) >= dn.shift(1))
    ok = win_all.copy()
    sig = np.where(ok & lt & ((d_trend > 0) | (not dfilt)), 1, np.where(ok & st & ((d_trend < 0) | (not dfilt)), -1, 0))
    sl = (c - stop_k * atr).to_numpy(); ss = (c + stop_k * atr).to_numpy()
    run(sig, sl, ss, ExitRules(tp_r=0, trail_mult=trail, cooldown=2), f"H1-donchian n={n} stop={stop_k} trail={trail} dtrend={dfilt}")
for fast, slow, trail in itertools.product([10, 20], [50, 100], [3.0, 4.5]):
    ef, es = ind.ema(c, fast), ind.ema(c, slow)
    up = (ef > es) & (ef.shift(1) <= es.shift(1)); dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    sig = np.where(win_all & up, 1, np.where(win_all & dn, -1, 0))
    sl = (c - 3 * atr).to_numpy(); ss = (c + 3 * atr).to_numpy()
    run(sig, sl, ss, ExitRules(tp_r=0, trail_mult=trail, cooldown=2, opp_exit=True, min_hold=4), f"H1-emaX {fast}/{slow} trail={trail}")
