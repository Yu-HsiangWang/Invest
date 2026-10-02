"""Silver (XAG/USD) research: apply the unchanged gold rules, then a small grid.

Downloads Dukascopy 1-minute data (cached under data/cache; first run ~3,300 requests)
and reproduces the numbers in docs/RESEARCH.md section 7:

    python research/silver_check.py              # silver, 2013-2026
    python research/silver_check.py --symbol XAUUSD   # same checks on gold (cross-vendor)
"""
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import metrics  # noqa: E402
from goldsignal.datafeed import Dukascopy, to_m15  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402

PER = {"2013-19": ("2013-07-01", "2019-12-31"), "2020-22": ("2020-01-01", "2022-12-31"), "2023-26": ("2023-01-01", "2026-12-31")}


def load(sym: str, start: date, end: date | None) -> pd.DataFrame:
    cache = ROOT / "data" / f"dukascopy_{sym}_15m_full.csv.gz"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    feed = Dukascopy(ROOT / "data" / "cache" / "dukascopy")
    m1 = feed.minutes(INSTRUMENTS[sym].code, start, end,
                      progress=lambda k, n: print(f"  {k}/{n}", end="\r", flush=True) if k % 50 == 0 or k == n else None)
    m15 = to_m15(m1)
    m15.to_csv(cache)
    print()
    return m15


def evaluate(df: pd.DataFrame, feats: dict, cost_mult: float = 1.0, spread=0.0006, slip=0.0001, **kw) -> dict:
    p = S.StrategyParams(**kw)
    if p.breakout_hours not in feats:
        feats[p.breakout_hours] = S.compute_features(df, params=p)
    px = df["open"].to_numpy(float)
    tr, _, _ = S.run(df, feats[p.breakout_hours], p, px * spread * cost_mult, px * slip * cost_mult)
    tr = tr[tr["reason"] != "end"]
    out = {}
    for name, (a, b) in PER.items():
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] <= pd.Timestamp(b, tz="UTC"))
        out[name] = metrics(tr[m])
    return out, tr


def line(label, res):
    cells = [f"{k}: n={v.get('n', 0):3d} avgR={v.get('avgR', float('nan')):+.3f} PF={v.get('PF', float('nan')):.2f} DD={v.get('maxDD_R', float('nan')):.1f}"
             for k, v in res.items()]
    print(f"{label:28s} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAGUSD", choices=list(INSTRUMENTS))
    ap.add_argument("--start", default="2013-01-01")
    args = ap.parse_args()
    inst = INSTRUMENTS[args.symbol]
    df = load(inst.key, date.fromisoformat(args.start), None)
    print(f"{inst.symbol}: {len(df)} bars {df.index[0]} .. {df.index[-1]}")
    feats: dict = {}
    sp, sl = inst.spread_pct, inst.slip_pct
    print("\n== unchanged rules")
    for mult in (1.0, 1.5, 2.0):
        res, tr = evaluate(df, feats, mult, sp, sl)
        line(f"costs x{mult}", res)
        if mult == 1.0:
            base_tr = tr
    print("\n== by direction")
    for name, (a, b) in PER.items():
        x = base_tr[(base_tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (base_tr["entry_time"] <= pd.Timestamp(b, tz="UTC"))]
        print(f"{name}: long {metrics(x[x['dir'] > 0]).get('avgR')}  short {metrics(x[x['dir'] < 0]).get('avgR')}")
    print("\n== by year")
    print(base_tr.groupby(base_tr["entry_time"].dt.year)["R"].agg(["size", "sum", "mean"]).round(2).to_string())
    print("\n== parameter grid (breakout hours / stop ATR / trail ATR)")
    for bh, sk, tm in itertools.product((16, 24, 32), (2.0, 2.5, 3.0), (4.0, 5.0, 6.5)):
        res, _ = evaluate(df, feats, 1.0, sp, sl, breakout_hours=bh, stop_atr_h1=sk, trail_atr_h1=tm)
        line(f"{bh}h stop{sk} trail{tm}", res)
    print("\n== filter variants")
    for label, kw in {"no 1h filter": {"use_h1": False}, "no 4h filter": {"use_h4": False}, "close_pos 0.6": {"close_pos_min": 0.6},
                      "close_pos 0.8": {"close_pos_min": 0.8}, "adx<=25": {"adx_max": 25.0}, "adx<=35": {"adx_max": 35.0},
                      "ext<=1.0": {"ext_max": 1.0}, "ext<=2.0": {"ext_max": 2.0}}.items():
        res, _ = evaluate(df, feats, 1.0, sp, sl, **kw)
        line(label, res)


if __name__ == "__main__":
    main()
