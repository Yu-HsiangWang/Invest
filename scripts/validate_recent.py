"""Download fresh XAU/USD data from Dukascopy and re-run the exact live strategy on it.

Usage:
    python scripts/validate_recent.py --symbol XAUUSD --start 2025-04-01
    python scripts/validate_recent.py --symbol XAGUSD --start 2025-04-01

Outputs a summary in the terminal, results/<SYMBOL>_recent_validation.json and the
15m bars at data/dukascopy_<SYMBOL>_15m.csv.gz (also used by the replay mode).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import by_year, metrics  # noqa: E402
from goldsignal.datafeed import Dukascopy, to_m15  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD", choices=list(INSTRUMENTS))
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--warmup-days", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    inst = INSTRUMENTS[args.symbol]
    out_path = args.out or str(ROOT / "results" / f"{inst.key}_recent_validation.json")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else datetime.now(timezone.utc).date()
    fetch_from = start - timedelta(days=args.warmup_days)

    feed = Dukascopy(ROOT / "data" / "cache" / "dukascopy")

    def progress(k, total):
        if k % 25 == 0 or k == total:
            print(f"  downloaded {k}/{total} days", flush=True)

    print(f"Downloading {inst.symbol} 1-minute bars {fetch_from} .. {end} from Dukascopy ...")
    m1 = feed.minutes(inst.code, fetch_from, end, progress=progress)
    m15 = to_m15(m1)
    out_csv = ROOT / "data" / f"dukascopy_{inst.key}_15m.csv.gz"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m15.to_csv(out_csv)
    print(f"15m bars: {len(m15)}  ({m15.index[0]} .. {m15.index[-1]})  saved -> {out_csv}")

    px = m15["open"].to_numpy(float)
    trades, f, t = S.run(m15, params=inst.params, spread=px * inst.spread_pct, slip=px * inst.slip_pct)
    tr = trades[(trades["entry_time"] >= pd.Timestamp(start, tz="UTC"))
                & (trades["entry_time"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
    closed = tr[tr["reason"] != "end"]
    years = max((end - start).days / 365.25, 1e-9)
    summary = metrics(closed, years)
    print("\n=== Out-of-sample result (closed trades) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not closed.empty:
        print(by_year(closed).to_string())
        monthly = closed.groupby(closed["entry_time"].dt.strftime("%Y-%m"))["R"].agg(["size", "sum"]).round(2)
        print(monthly.to_string())
    res = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": [start.isoformat(), end.isoformat()],
        "symbol": inst.key,
        "data_source": "Dukascopy",
        "params": inst.params.to_dict(),
        "costs": {"spread_pct": inst.spread_pct, "slip_pct_per_fill": inst.slip_pct},
        "summary": summary,
        "trades": [
            {k: (v.isoformat() if isinstance(v, pd.Timestamp) else (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer,)) else v)))
             for k, v in row.items()}
            for row in tr[["entry_time", "exit_time", "dir", "entry", "exit", "init_stop", "R", "reason"]].to_dict("records")
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
