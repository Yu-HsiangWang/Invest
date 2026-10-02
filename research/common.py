"""Shared helpers for the research notebooks/scripts (not used by the live app)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import indicators as ind  # noqa: E402
from goldsignal.bars import align_htf, load_kaggle_15m, resample  # noqa: E402

M15 = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)
H4 = pd.Timedelta(hours=4)
D1 = pd.Timedelta(days=1)

SPREAD_PCT = 0.00012   # 0.012 % of price ~ $0.50 at $4,200 (Mitrade-like retail spread)
SLIP_PCT = 0.00005     # extra slippage per market fill for manual execution

PERIODS = {
    "IS 2005-2016": ("2005-01-01", "2016-12-31"),
    "VAL 2017-2020": ("2017-01-01", "2020-12-31"),
    "TEST 2021-2025Q1": ("2021-01-01", "2025-03-21"),
}


def load_base(path: str | Path = ROOT / "data/raw/XAU_15m_data.csv") -> pd.DataFrame:
    df = load_kaggle_15m(path)
    df = df.loc[:"2025-03-21"]  # data after this date has multi-week gaps
    return df


def htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 15m features plus look-ahead-safe 1h / 4h / daily features."""
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    f["atr"] = ind.atr(df, 14)
    f["ema20"] = ind.ema(c, 20)
    f["ema50"] = ind.ema(c, 50)
    f["ema200"] = ind.ema(c, 200)
    f["rsi"] = ind.rsi(c, 14)
    a = ind.adx(df, 14)
    f["adx"] = a["adx"]
    f["pdi"], f["mdi"] = a["plus_di"], a["minus_di"]
    m = ind.macd(c)
    f["macd_h"] = m["hist"]
    f["er20"] = ind.efficiency_ratio(c, 20)

    for name, rule, ln in [("h1", "1h", H1), ("h4", "4h", H4)]:
        hb = resample(df, rule)
        hf = pd.DataFrame(index=hb.index)
        hc = hb["close"]
        hf["close"] = hc
        hf["ema20"] = ind.ema(hc, 20)
        hf["ema50"] = ind.ema(hc, 50)
        hf["ema100"] = ind.ema(hc, 100)
        hf["ema200"] = ind.ema(hc, 200)
        hf["atr"] = ind.atr(hb, 14)
        ad = ind.adx(hb, 14)
        hf["adx"] = ad["adx"]
        hf["pdi"], hf["mdi"] = ad["plus_di"], ad["minus_di"]
        hf["rsi"] = ind.rsi(hc, 14)
        hf["er"] = ind.efficiency_ratio(hc, 20)
        st = ind.supertrend(hb, 10, 3.0)
        hf["st_dir"] = st["direction"]
        hf["st_line"] = st["line"]
        hf["macd_h"] = ind.macd(hc)["hist"]
        hf["ema50_slope"] = ind.slope(hf["ema50"], 5)
        al = align_htf(df.index, M15, hf, ln)
        for col in al.columns:
            f[f"{name}_{col}"] = al[col]

    db = resample(df, "1D")
    dfeat = pd.DataFrame(index=db.index)
    dfeat["atr"] = ind.atr(db, 14)
    dfeat["ema20"] = ind.ema(db["close"], 20)
    dfeat["close"] = db["close"]
    dfeat["high"] = db["high"]
    dfeat["low"] = db["low"]
    al = align_htf(df.index, M15, dfeat, D1)
    for col in al.columns:
        f[f"d1_{col}"] = al[col]

    ny = df.index.tz_convert("America/New_York")
    ldn = df.index.tz_convert("Europe/London")
    f["ny_h"] = ny.hour + ny.minute / 60.0
    f["ldn_h"] = ldn.hour + ldn.minute / 60.0
    f["utc_h"] = df.index.hour + df.index.minute / 60.0
    f["dow"] = ny.dayofweek  # NY calendar day of week
    # trading-day id: the session that ends at 17:00 NY on this calendar date
    f["tday"] = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    return f


def costs(df: pd.DataFrame, mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar spread and slippage in price units (proportional to price)."""
    px = df["open"].to_numpy(float)
    return px * SPREAD_PCT * mult, px * SLIP_PCT * mult


def period_slices(index: pd.DatetimeIndex):
    for name, (a, b) in PERIODS.items():
        mask = (index >= pd.Timestamp(a, tz="UTC")) & (index <= pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        yield name, mask
