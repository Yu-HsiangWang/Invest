"""Grid evaluation of candidate strategy families across IS / VAL / TEST periods."""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goldsignal.backtest import ExitRules, metrics, simulate  # noqa: E402
from research import strategies as S  # noqa: E402
from research.common import PERIODS, costs  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def load_cached():
    df = pd.read_parquet(DATA / "cache_xau15.parquet")
    f = pd.read_parquet(DATA / "cache_feat15.parquet")
    return df, f


def evaluate(df, f, sig, sl, ss, rules, cost_mult=1.0, force=None):
    spread, slip = costs(df, cost_mult)
    tr = simulate(df, sig, sl, ss, rules, spread, slip, atr=f["atr"].to_numpy(), force_exit=force)
    rows = {}
    for name, (a, b) in PERIODS.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)) if not tr.empty else []
        sub = tr[m] if not tr.empty else tr
        yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        rows[name] = metrics(sub, yrs)
    return tr, rows


def fmt(rows):
    parts = []
    for name, m in rows.items():
        if m.get("n", 0) == 0:
            parts.append(f"{name.split()[0]}: n=0")
            continue
        parts.append(f"{name.split()[0]}: n={m['n']:4d} w={m['win%']:4.1f}% avgR={m['avgR']:+.3f} PF={m['PF']:.2f} DD={m['maxDD_R']:5.1f}")
    return " | ".join(parts)


def grid(fn, base_kwargs, grid_kwargs, rules_list, df, f, force=None, label=""):
    keys = list(grid_kwargs)
    out = []
    for vals in itertools.product(*[grid_kwargs[k] for k in keys]):
        kw = dict(base_kwargs, **dict(zip(keys, vals)))
        sig, sl, ss = fn(df, f, **kw)
        for rname, rules in rules_list:
            tr, rows = evaluate(df, f, sig, sl, ss, rules, force=force)
            desc = f"{label} {dict(zip(keys, vals))} {rname}"
            out.append((desc, rows, tr))
            print(f"{desc}\n    {fmt(rows)}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    df, f = load_cached()
    force = S.session_exit(f, 16.75)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    R = [
        ("tp2", ExitRules(tp_r=2.0, cooldown=4)),
        ("tp1.5_be1", ExitRules(tp_r=1.5, be_r=1.0, be_lock_r=0.1, cooldown=4)),
        ("trail3", ExitRules(tp_r=0, trail_mult=3.0, cooldown=4)),
    ]
    if which in ("A", "all"):
        grid(S.trend_pullback, dict(window=(3.0, 12.0)),
             {"tf": ["h1", "h4"], "adx_min": [0, 20, 25], "lookback": [4, 8]}, R, df, f, force, "A-pullback")
    if which in ("B", "all"):
        grid(S.donchian_breakout, dict(window=(3.0, 12.0)),
             {"n": [16, 32, 48], "tf": ["h1", "h4"], "stop_atr": [1.5, 2.5]}, R, df, f, force, "B-donchian")
    if which in ("C", "all"):
        grid(S.asian_range_breakout, dict(),
             {"stop_mode": ["mid", "opp"], "use_trend": [False, True], "max_w": [1.0, 2.0]}, R, df, f, force, "C-asian")
    if which in ("E", "all"):
        grid(S.htf_trend_flip, dict(window=(3.0, 12.0)),
             {"tf": ["h1", "h4"], "mode": ["st", "ema"], "stop_atr": [2.0, 3.0]}, R, df, f, force, "E-flip")
    print(f"done in {time.time()-t0:.0f}s")
