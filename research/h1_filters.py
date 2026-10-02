"""Re-simulate the core system with the quality filters found in trade_quality.py."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.h1_refine import *  # noqa

h4trend = np.where((H4F["close"] > H4F["ema20"]) & (H4F["ema20"] > H4F["ema50"]), 1,
                   np.where((H4F["close"] < H4F["ema20"]) & (H4F["ema20"] < H4F["ema50"]), -1, 0))
datr = align_htf(h.index, H1, pd.DataFrame({"atr": ind.atr(d, 14)}), D1)["atr"].to_numpy()
d_ext_long = (D["close"].to_numpy() - D["ema20"].to_numpy()) / datr
rng = (h["high"] - h["low"]).to_numpy()
cpos_long = (h["close"] - h["low"]).to_numpy() / np.where(rng > 0, rng, np.nan)
cpos_short = (h["high"] - h["close"]).to_numpy() / np.where(rng > 0, rng, np.nan)

def filtered(n, use_h4=True, adx_max=30.0, ext_max=1.5, cpos_min=0.7):
    base = signals(n, "ema20/50")
    keep = np.ones(len(base), dtype=bool)
    L = base > 0; S = base < 0
    if use_h4:
        keep &= np.where(L, h4trend > 0, np.where(S, h4trend < 0, True))
    if adx_max:
        keep &= ~(adx1 > adx_max)
    if ext_max:
        keep &= np.where(L, d_ext_long <= ext_max, np.where(S, -d_ext_long <= ext_max, True))
    if cpos_min:
        keep &= np.where(L, cpos_long >= cpos_min, np.where(S, cpos_short >= cpos_min, True))
    return np.where(keep, base, 0)

if __name__ == "__main__":
    R = ExitRules(tp_r=0, trail_mult=4.5, cooldown=2)
    print("### filters added one at a time (n=32)")
    run(signals(32, "ema20/50"), 3.0, R, "core")
    run(filtered(32, True, 0, 0, 0), 3.0, R, "+h4 agree")
    run(filtered(32, False, 30, 0, 0), 3.0, R, "+adx<=30")
    run(filtered(32, False, 0, 1.5, 0), 3.0, R, "+ext<=1.5")
    run(filtered(32, False, 0, 0, 0.7), 3.0, R, "+closepos>=0.7")
    run(filtered(32, True, 30, 1.5, 0), 3.0, R, "+h4 +adx +ext")
    run(filtered(32, True, 30, 1.5, 0.7), 3.0, R, "ALL filters")
    print("### ALL filters, neighbourhood")
    for n, sk, tm in itertools.product([24, 32, 40], [2.5, 3.0, 3.5], [3.5, 4.5, 5.5]):
        run(filtered(n), sk, ExitRules(tp_r=0, trail_mult=tm, cooldown=2), f"ALL n={n} stop={sk} trail={tm}")
