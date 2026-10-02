"""Bar utilities: loading historical files, resampling, and look-ahead-safe
alignment of higher-timeframe (HTF) data onto a lower timeframe.

Conventions
-----------
* Index = bar OPEN time, timezone-aware UTC.
* A bar with open time t and length L is "complete" at t + L.
* The gold trading day rolls at 17:00 New York time (daily break 17:00-18:00 NY).
  Many MetaTrader brokers therefore run their server clock at New York + 7h, so
  that midnight server time == the 17:00 NY rollover. We use that same clock for
  daily/4h bucketing so our bars match what most retail platforms display.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NY = "America/New_York"
SERVER_OFFSET = pd.Timedelta(hours=7)  # broker server clock = NY + 7h

OHLC_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def server_clock_to_utc(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert naive broker server timestamps (NY+7h) to UTC."""
    idx = pd.DatetimeIndex(ts) - SERVER_OFFSET
    return idx.tz_localize(NY, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def to_server_clock(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC -> naive server clock (NY + 7h)."""
    return idx.tz_convert(NY).tz_localize(None) + SERVER_OFFSET


def load_kaggle_15m(path: str | Path) -> pd.DataFrame:
    """Load the widely used 'XAU_15m_data.csv' (semicolon separated, broker server time)."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["date"], format="%Y.%m.%d %H:%M")
    df.index = server_clock_to_utc(ts)
    df = df[~df.index.isna()].drop(columns=["date"])
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated()].sort_index()


def load_ejtrader_15m(path: str | Path, scale: float = 100.0) -> pd.DataFrame:
    """Load ejtraderLabs CSV (prices stored as integers * 100, broker server time)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["date"])
    df.index = server_clock_to_utc(ts)
    df = df[~df.index.isna()].drop(columns=["date"]).rename(columns={"tick_volume": "volume"})
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float) / scale
    return df[["open", "high", "low", "close", "volume"]][~df.index.duplicated()].sort_index()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC bars. Buckets are formed on the broker server clock (NY+7h),
    so '1D' == the 17:00 NY trading day and '4h' buckets start at 17:00/21:00/... NY.
    Returned index is the bucket open time in UTC."""
    if df.empty:
        return df.copy()
    server = to_server_clock(df.index)
    tmp = df.copy()
    tmp.index = server
    cols = {k: v for k, v in OHLC_AGG.items() if k in tmp.columns}
    out = tmp.resample(rule, label="left", closed="left").agg(cols).dropna(subset=["open"])
    # back to UTC
    out.index = (out.index - SERVER_OFFSET).tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    out = out[~out.index.isna()]
    return out


def bar_length(rule: str) -> pd.Timedelta:
    return pd.Timedelta(pd.tseries.frequencies.to_offset(rule).nanos, unit="ns") if rule[-1] not in "D" else pd.Timedelta(days=int(rule[:-1] or 1))


def align_htf(base_index: pd.DatetimeIndex, base_len: pd.Timedelta, htf: pd.DataFrame, htf_len: pd.Timedelta) -> pd.DataFrame:
    """Project completed higher-timeframe rows onto the base timeframe.

    For a base bar that closes at T = open + base_len, we may only use HTF bars whose
    close time (htf_open + htf_len) <= T. This avoids the classic HTF look-ahead bug.
    """
    avail = htf.copy()
    avail.index = avail.index + htf_len  # time at which this HTF row becomes known
    base_close = base_index + base_len
    out = avail.reindex(avail.index.union(base_close)).ffill().reindex(base_close)
    out.index = base_index
    return out
