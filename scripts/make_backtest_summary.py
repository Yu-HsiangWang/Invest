"""Re-create results/XAUUSD_backtest_summary.json (shown in the dashboard) from the
2004-2025 15-minute history.

The history file is the public 'XAU_15m_data.csv' (Kaggle dataset by
novandraanugrah, also mirrored in github.com/BaseMax/XAUUSD-LSTM). Put it at
data/raw/XAU_15m_data.csv, then run:

    python scripts/make_backtest_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import by_year, metrics  # noqa: E402
from goldsignal.bars import load_kaggle_15m  # noqa: E402

SPREAD_PCT, SLIP_PCT = 0.00012, 0.00005
PERIODS = [
    ("開發期 2005–2016", "2005-01-01", "2016-12-31"),
    ("驗證期 2017–2020", "2017-01-01", "2020-12-31"),
    ("測試期 2021–2025/3", "2021-01-01", "2025-03-21"),
]


def main():
    src = ROOT / "data" / "raw" / "XAU_15m_data.csv"
    df = load_kaggle_15m(src).loc[:"2025-03-21"]
    px = df["open"].to_numpy(float)
    trades, f, t = S.run(df, spread=px * SPREAD_PCT, slip=px * SLIP_PCT)
    tr = trades[(trades["reason"] != "end") & (trades["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC"))]
    out_periods = []
    for label, a, b in PERIODS:
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        out_periods.append({"label": label, **metrics(tr[m], yrs)})
    allm = metrics(tr, (tr["entry_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 365.25)
    eq = tr[["exit_time", "R"]].copy()
    eq["cum"] = eq["R"].cumsum()
    step = max(1, len(eq) // 400)
    equity = [[ts.isoformat(), round(float(v), 2)] for ts, v in zip(eq["exit_time"].iloc[::step], eq["cum"].iloc[::step])]
    yr = by_year(tr).reset_index().rename(columns={"entry_time": "year"})
    res = {
        "caption": f"2005–2025/3 共 {allm['n']} 筆（約每月 {allm['n'] / ((tr['entry_time'].iloc[-1] - tr['entry_time'].iloc[0]).days / 30.4):.1f} 筆），累積 {allm['totR']:+.0f}R；已扣點差 0.012% 與每次成交滑價 0.005%。",
        "params": S.StrategyParams().to_dict(),
        "costs": {"spread_pct": SPREAD_PCT, "slip_pct_per_fill": SLIP_PCT},
        "all": allm,
        "periods": out_periods,
        "equity": equity,
        "by_year": yr.to_dict("records"),
        "long": metrics(tr[tr["dir"] > 0]),
        "short": metrics(tr[tr["dir"] < 0]),
        "hold_hours": {"median": float(tr["bars"].median() / 4), "p75": float(tr["bars"].quantile(.75) / 4), "p90": float(tr["bars"].quantile(.9) / 4)},
    }
    out = ROOT / "results" / "XAUUSD_backtest_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: res[k] for k in ("caption", "all", "periods", "long", "short", "hold_hours")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
