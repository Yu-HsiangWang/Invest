"""Direction ("偏多 / 中性 / 偏空") for the next few months, and the historical "probability cloud".

Design (docs/EVIDENCE.md, design implications A1/C10):
* The direction is a TREND STATE, decided by an equal-weight ensemble of several look-backs rather
  than one magic number (the GEM study showed single-specification results are mostly luck).
* The market/sector trend (QQQ / SOX / TAIEX / 0050) is checked separately; a stock in an uptrend
  inside a market downtrend is downgraded to 中性.
* Instead of promising a point forecast, `forward_cloud` reports what happened historically after
  the same state: share of positive 3/6/12-month outcomes and the 10-90% range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import momentum, sma

LOOKBACKS = (100, 150, 200, 250)


def trend_votes(close: pd.Series) -> pd.DataFrame:
    """Each column is a vote in {-1, +1} (NaN while warming up)."""
    v = pd.DataFrame(index=close.index)
    for n in LOOKBACKS:
        m = sma(close, n)
        v[f"above_{n}"] = np.sign(close - m).where(m.notna())
    m200 = sma(close, 200)
    v["slope_200"] = np.sign(m200 - m200.shift(21)).where(m200.notna() & m200.shift(21).notna())
    mom = momentum(close, 252, 21)
    v["mom_12_1"] = np.sign(mom).where(mom.notna())
    r6 = close / close.shift(126) - 1.0
    v["ret_6m"] = np.sign(r6).where(r6.notna())
    return v.replace(0, 1)


def trend_score(close: pd.Series) -> pd.Series:
    """Mean of the votes, in [-1, 1]; NaN until all votes are available."""
    v = trend_votes(close)
    return v.mean(axis=1).where(v.notna().all(axis=1))


def score_to_state(score: pd.Series, up: float = 0.5, down: float = -0.5) -> pd.Series:
    """+1 / 0 / -1 from the ensemble score with a neutral band (reduces whipsaw)."""
    st = pd.Series(0.0, index=score.index)
    st[score >= up] = 1
    st[score <= down] = -1
    st[score.isna()] = np.nan
    return st


def direction_label(stock_state: float, market_state: float | None) -> tuple[str, str]:
    """Return (label, reason)."""
    if stock_state is None or np.isnan(stock_state):
        return "資料不足", "歷史不足一年，還算不出趨勢"
    if stock_state > 0:
        if market_state is not None and not np.isnan(market_state) and market_state < 0:
            return "中性", "個股趨勢向上，但大盤趨勢向下：歷史上這種背離多半撐不久，先當中性"
        return "偏多", "多數長期趨勢指標向上（價格在 100–250 日均線之上、年線上揚、12 個月動能為正）"
    if stock_state < 0:
        if market_state is not None and not np.isnan(market_state) and market_state > 0:
            return "偏空", "個股趨勢向下（雖然大盤向上）：相對弱勢，等它重新站回長期均線再說"
        return "偏空", "多數長期趨勢指標向下（價格在長期均線之下、年線下彎、12 個月動能為負）"
    return "中性", "趨勢指標多空各半，方向還不明"


def forward_cloud(close: pd.Series, state: pd.Series, current_state: float, horizons=(63, 126, 252), min_n: int = 60) -> dict:
    """Historical outcomes after days with the same state (uses this instrument's own history only)."""
    out = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        d = pd.DataFrame({"fwd": fwd, "st": state}).dropna()
        all_ = d["fwd"]
        g = d[d["st"] == current_state]["fwd"]
        if len(all_) < min_n:
            continue
        rec = {"n_all": int(len(all_)), "hit_all": float((all_ > 0).mean()), "median_all": float(all_.median())}
        if len(g) >= min_n:
            rec.update({"n": int(len(g)), "n_eff": round(len(g) / h, 1), "hit": float((g > 0).mean()), "median": float(g.median()),
                        "p10": float(g.quantile(0.1)), "p90": float(g.quantile(0.9)), "mean": float(g.mean())})
        out[str(h)] = rec
    return out


def relative_strength(close: pd.Series, bench: pd.Series, window: int = 126) -> pd.Series:
    """Stock return minus benchmark return over `window` days (aligned on dates)."""
    b = bench.reindex(close.index).ffill()
    return (close / close.shift(window) - 1.0) - (b / b.shift(window) - 1.0)
