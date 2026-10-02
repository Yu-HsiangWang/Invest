"""Robustness checks for the filtered 1h breakout system."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.h1_filters import *  # noqa
from goldsignal.backtest import by_year

SWAP_PCT = 0.00015  # assumed overnight funding per night, both directions (conservative)

def nights(tr):
    a = tr["entry_time"].dt.tz_convert("America/New_York"); b = tr["exit_time"].dt.tz_convert("America/New_York")
    # count 17:00 NY rollovers crossed
    a_day = (a + pd.Timedelta(hours=7)).dt.normalize(); b_day = (b + pd.Timedelta(hours=7)).dt.normalize()
    return ((b_day - a_day).dt.days).clip(lower=0)

def with_swap(tr):
    t = tr.copy()
    t["nights"] = nights(t)
    t["R"] = t["R"] - t["nights"] * SWAP_PCT * t["entry"] / t["risk"]
    return t

def summarize(tr, label):
    rows = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        rows[name] = metrics(tr[m], (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
    print(f"{label}\n    {fmt(rows)}")

if __name__ == "__main__":
    for n in (24, 32):
        sig = filtered(n)
        R = ExitRules(tp_r=0, trail_mult=4.5, cooldown=2)
        tr, _ = run(sig, 3.0, R, f"== n={n} base costs", show=False)
        summarize(tr, f"== n={n} base costs")
        trs = with_swap(tr)
        summarize(trs, f"   + swap {SWAP_PCT*100:.3f}%/night (avg nights {trs['nights'].mean():.2f})")
        tr2, _ = run(sig, 3.0, R, "", cost_mult=2.0, show=False)
        summarize(with_swap(tr2), "   2x spread+slip + swap")
        tr3, _ = run(sig, 3.0, R, "", cost_mult=3.0, show=False)
        summarize(with_swap(tr3), "   3x spread+slip + swap")
        print("   long only:", metrics(trs[trs.dir > 0]))
        print("   short only:", metrics(trs[trs.dir < 0]))
        print("   exit reasons:", trs["reason"].value_counts().to_dict())
        print("   holding hours: median", trs["bars"].median(), " p75", trs["bars"].quantile(0.75), " p90", trs["bars"].quantile(0.9))
        print(by_year(trs).to_string())
