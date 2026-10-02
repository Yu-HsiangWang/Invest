"""Does the same breakout + multi-timeframe filter idea work on 15m bars (shorter holds)?"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldsignal import indicators as ind
from goldsignal.bars import resample, align_htf
from goldsignal.backtest import ExitRules, simulate, metrics
from research.common import PERIODS, costs
from research.run_grid import load_cached, fmt
from research.strategies import entry_window, session_exit

df, f = load_cached()
c = df["close"]; h = df["high"]; l = df["low"]; o = df["open"]
atr = f["atr"].to_numpy()
d = resample(df, "1D")
D = align_htf(df.index, pd.Timedelta(minutes=15), pd.DataFrame({"close": d["close"], "ema20": ind.ema(d["close"], 20), "ema50": ind.ema(d["close"], 50), "atr": ind.atr(d, 14)}), pd.Timedelta(days=1))
dtr = np.where((D.close > D.ema20) & (D.ema20 > D.ema50), 1, np.where((D.close < D.ema20) & (D.ema20 < D.ema50), -1, 0))
h4t = np.where((f.h4_close > f.h4_ema20) & (f.h4_ema20 > f.h4_ema50), 1, np.where((f.h4_close < f.h4_ema20) & (f.h4_ema20 < f.h4_ema50), -1, 0))
h1t = np.where((f.h1_close > f.h1_ema20) & (f.h1_ema20 > f.h1_ema50), 1, np.where((f.h1_close < f.h1_ema20) & (f.h1_ema20 < f.h1_ema50), -1, 0))
ext = ((D.close - D.ema20) / D.atr).to_numpy()
rng = (h - l).to_numpy(); rng = np.where(rng > 0, rng, np.nan)
cpl = ((c - l).to_numpy() / rng); cps = ((h - c).to_numpy() / rng)
adx1 = f["h1_adx"].to_numpy()

def sigs(n, need_h1=True, window=None, cpos=0.7, adx_max=30, ext_max=1.5):
    up = h.rolling(n).max().shift(1); dn = l.rolling(n).min().shift(1)
    lt = ((c > up) & (c.shift(1) <= up.shift(1))).to_numpy(); st = ((c < dn) & (c.shift(1) >= dn.shift(1))).to_numpy()
    ok = (f["dow"].to_numpy() <= 4) & ~((f["dow"].to_numpy() == 4) & (((f["ny_h"].to_numpy() + .25) % 24) > 12))
    if window: ok &= entry_window(f, *window)
    ok &= ~(adx1 > adx_max)
    L = ok & lt & (dtr > 0) & (h4t > 0) & (ext <= ext_max) & (cpl >= cpos)
    S = ok & st & (dtr < 0) & (h4t < 0) & (-ext <= ext_max) & (cps >= cpos)
    if need_h1:
        L &= h1t > 0; S &= h1t < 0
    return np.where(L, 1, np.where(S, -1, 0))

sp, sl_ = costs(df)
force = session_exit(f, 16.0)
if len(sys.argv) == 1:
  for n, sk, tm, mode in itertools.product([48, 96], [3.0, 5.0], [6.0, 9.0], ["swing", "intraday"]):
      sig = sigs(n, window=(3.0, 12.0) if mode == "intraday" else None)
      stl = (c - sk * f["atr"]).to_numpy(); sts = (c + sk * f["atr"]).to_numpy()
      tr = simulate(df, sig, stl, sts, ExitRules(tp_r=0, trail_mult=tm, cooldown=4), sp, sl_, atr=atr, force_exit=force if mode == "intraday" else None)
      rows = {}
      for name, (a, b) in PERIODS.items():
          m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
          rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
      print(f"m15 {mode} n={n} stop={sk} trail={tm} avg_hours={tr['bars'].mean()/4:.1f}\n    {fmt(rows)}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "grid2":
    print("########## grid2 (swing only)")
    for n, sk, tm, h1f in itertools.product([64, 96, 128], [3.0, 4.0], [7.5, 9.0, 10.5], [True, False]):
        sig = sigs(n, need_h1=h1f)
        stl = (c - sk * f["atr"]).to_numpy(); sts = (c + sk * f["atr"]).to_numpy()
        tr = simulate(df, sig, stl, sts, ExitRules(tp_r=0, trail_mult=tm, cooldown=4), sp, sl_, atr=atr)
        rows = {}
        for name, (a, b) in PERIODS.items():
            m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
            rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
        print(f"g2 n={n} stop={sk} trail={tm} h1={h1f} hrs={tr['bars'].mean()/4:.1f}\n    {fmt(rows)}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "hybrid":
    print("########## hybrid: 15m entries, stops/trail in 1h-ATR units")
    atr1 = f["h1_atr"].to_numpy()
    for n, sk, tm, h1f in itertools.product([64, 96, 128], [1.5, 2.0, 3.0], [3.5, 4.5, 5.5], [True]):
        sig = sigs(n, need_h1=h1f)
        stl = c.to_numpy() - sk * atr1; sts = c.to_numpy() + sk * atr1
        tr = simulate(df, sig, stl, sts, ExitRules(tp_r=0, trail_mult=tm, cooldown=4), sp, sl_, atr=atr1)
        rows = {}
        for name, (a, b) in PERIODS.items():
            m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
            rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
        print(f"hy n={n} stop={sk}xATR1h trail={tm}xATR1h hrs={tr['bars'].mean()/4:.1f}\n    {fmt(rows)}")
