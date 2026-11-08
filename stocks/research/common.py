"""Shared helpers for the research scripts: load normalized daily bars, period splits, output paths."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "daily"          # <SYMBOL>.csv : date,open,high,low,close,volume[,close_raw]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Pre-registered period split (see docs/RESEARCH.md section 0)
PERIODS = {"IS": ("1990-01-01", "2012-12-31"), "VAL": ("2013-01-01", "2018-12-31"), "TEST": ("2019-01-01", "2030-12-31")}


def period_of(idx: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series("IS", index=idx)
    out[idx >= "2013-01-01"] = "VAL"
    out[idx >= "2019-01-01"] = "TEST"
    return out


def load_daily(symbol: str, data_dir: Path = DATA) -> pd.DataFrame | None:
    p = data_dir / f"{symbol}.csv.gz"
    if not p.exists():
        p = data_dir / f"{symbol}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        return None
    df = df.dropna(subset=["close"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    # guard against zero / missing opens in old data
    bad = (df["open"] <= 0) | df["open"].isna()
    df.loc[bad, "open"] = df.loc[bad, "close"]
    return df


def available(symbols: list[str], data_dir: Path = DATA) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        df = load_daily(s, data_dir)
        if df is not None and len(df) > 300:
            out[s] = df
    return out


def fmt_pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.{nd}f}%"
