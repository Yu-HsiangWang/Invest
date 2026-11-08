"""Buy-zone / sell-zone constructions (all computed from data up to each bar, no look-ahead).

Three families are implemented; the research scripts decide which ones are worth using and with
which parameters, and the dashboard shows the chosen ones.

1. Regression channel ("樂活五線譜"): log-linear trend over a long window with +-1 / +-2 sigma bands.
2. Moving-average deviation ("年線乖離"): distance from the 200-day SMA, ranked against the stock's
   own trailing history, so "cheap" and "expensive" adapt to each stock's volatility.
3. Valuation band ("本益比河流圖"): price = EPS(TTM) x P/E quantiles of the stock's own history.
   Needs an EPS or P/E history (TWSE BWIBBU for Taiwan, SEC companyfacts for the US).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import pct_rank, rolling_log_regression, sma

ZONE_LABELS = {-2: "很便宜", -1: "便宜", 0: "合理", 1: "偏貴", 2: "很貴"}


@dataclass
class ChannelParams:
    window: int = 876          # ~3.5 years of trading days (the classic 五線譜 default)
    k1: float = 1.0
    k2: float = 2.0


def channel_bands(close: pd.Series, p: ChannelParams = ChannelParams()) -> pd.DataFrame:
    reg = rolling_log_regression(close, p.window)
    logc = np.log(reg["center"])
    out = pd.DataFrame(index=close.index)
    out["center"] = reg["center"]
    out["slope_annual"] = reg["slope"] * 252.0
    out["zscore"] = reg["zscore"]
    for k, name in ((p.k2, "2"), (p.k1, "1")):
        out[f"up{name}"] = np.exp(logc + k * reg["sigma"])
        out[f"dn{name}"] = np.exp(logc - k * reg["sigma"])
    return out


def channel_zone(z: pd.Series, p: ChannelParams = ChannelParams()) -> pd.Series:
    """-2 very cheap .. +2 very expensive from the channel z-score."""
    zone = pd.Series(0, index=z.index, dtype=float)
    zone[z <= -p.k2] = -2
    zone[(z > -p.k2) & (z <= -p.k1)] = -1
    zone[(z >= p.k1) & (z < p.k2)] = 1
    zone[z >= p.k2] = 2
    zone[z.isna()] = np.nan
    return zone


@dataclass
class DeviationParams:
    ma: int = 200
    rank_window: int = 1260    # 5 years of history for the percentile
    cheap_pct: float = 0.15    # below this percentile of own deviation history = buy zone
    very_cheap_pct: float = 0.05
    rich_pct: float = 0.85
    very_rich_pct: float = 0.95


def deviation_bands(close: pd.Series, p: DeviationParams = DeviationParams()) -> pd.DataFrame:
    ma = sma(close, p.ma)
    dev = close / ma - 1.0
    rank = pct_rank(dev, p.rank_window, min_periods=max(252, p.rank_window // 4))
    out = pd.DataFrame({"ma": ma, "dev": dev, "dev_rank": rank}, index=close.index)
    # price levels of the percentile thresholds (so the UI can draw them)
    for name, q in (("dn2", p.very_cheap_pct), ("dn1", p.cheap_pct), ("up1", p.rich_pct), ("up2", p.very_rich_pct)):
        qdev = dev.rolling(p.rank_window, min_periods=max(252, p.rank_window // 4)).quantile(q)
        out[name] = ma * (1.0 + qdev)
    return out


def deviation_zone(rank: pd.Series, p: DeviationParams = DeviationParams()) -> pd.Series:
    zone = pd.Series(0, index=rank.index, dtype=float)
    zone[rank <= p.very_cheap_pct] = -2
    zone[(rank > p.very_cheap_pct) & (rank <= p.cheap_pct)] = -1
    zone[(rank >= p.rich_pct) & (rank < p.very_rich_pct)] = 1
    zone[rank >= p.very_rich_pct] = 2
    zone[rank.isna()] = np.nan
    return zone


@dataclass
class PEParams:
    window: int = 1260           # P/E history used for the quantiles (5 years)
    q_very_cheap: float = 0.10
    q_cheap: float = 0.25
    q_rich: float = 0.75
    q_very_rich: float = 0.90
    min_pe: float = 3.0          # ignore nonsense P/E (negative / tiny earnings)
    max_pe: float = 150.0


def pe_bands(close: pd.Series, pe: pd.Series, p: PEParams = PEParams()) -> pd.DataFrame:
    """P/E river: price bands = EPS(TTM) x P/E quantiles. `pe` is the daily trailing P/E series."""
    pe = pe.reindex(close.index).ffill()
    pe_clean = pe.where((pe >= p.min_pe) & (pe <= p.max_pe))
    eps = close / pe_clean
    out = pd.DataFrame({"pe": pe_clean, "eps_ttm": eps}, index=close.index)
    mp = max(252, p.window // 4)
    for name, q in (("dn2", p.q_very_cheap), ("dn1", p.q_cheap), ("mid", 0.5), ("up1", p.q_rich), ("up2", p.q_very_rich)):
        out[f"pe_{name}"] = pe_clean.rolling(p.window, min_periods=mp).quantile(q)
        out[name] = out[f"pe_{name}"] * eps
    out["pe_rank"] = pct_rank(pe_clean, p.window, min_periods=mp)
    return out


def pe_zone(bands: pd.DataFrame, p: PEParams = PEParams()) -> pd.Series:
    r = bands["pe_rank"]
    zone = pd.Series(0, index=r.index, dtype=float)
    zone[r <= p.q_very_cheap] = -2
    zone[(r > p.q_very_cheap) & (r <= p.q_cheap)] = -1
    zone[(r >= p.q_rich) & (r < p.q_very_rich)] = 1
    zone[r >= p.q_very_rich] = 2
    zone[r.isna()] = np.nan
    return zone


def trend_state(close: pd.Series, ma: int = 200, slope_days: int = 21) -> pd.Series:
    """+1 uptrend (close above a rising MA), -1 downtrend (below a falling MA), 0 otherwise."""
    m = sma(close, ma)
    rising = m > m.shift(slope_days)
    st = pd.Series(0, index=close.index, dtype=float)
    st[(close > m) & rising] = 1
    st[(close < m) & ~rising] = -1
    st[m.isna()] = np.nan
    return st
