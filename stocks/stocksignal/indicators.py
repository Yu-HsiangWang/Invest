"""Daily-bar indicators used by the zone / regime logic and by the research scripts.

Every function only looks backwards: the value at row t is computed from rows <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def pct_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile (0..1) of the latest value inside its trailing window (inclusive)."""
    mp = min_periods or window

    def _rank(a: np.ndarray) -> float:
        x = a[-1]
        if np.isnan(x):
            return np.nan
        a = a[~np.isnan(a)]
        return (a <= x).mean() if len(a) else np.nan

    return s.rolling(window, min_periods=mp).apply(_rank, raw=True)


def drawdown_from_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Negative number: how far the close is below its trailing `window`-day high."""
    hi = close.rolling(window, min_periods=1).max()
    return close / hi - 1.0


def momentum(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Classic 12-1 momentum when lookback=252, skip=21 (return from t-252 to t-21)."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def rolling_log_regression(close: pd.Series, window: int) -> pd.DataFrame:
    """Rolling OLS of log(price) on time, fitted at the LAST point of each window.

    Returns columns: center (fitted log price at t, exponentiated), sigma (std of log residuals
    inside the window), slope (log return per bar), zscore (current log residual / sigma).
    This is the "樂活五線譜" construction, computed without look-ahead.
    """
    y = np.log(close.to_numpy(dtype=float))
    n = len(y)
    out = pd.DataFrame(index=close.index, columns=["center", "sigma", "slope", "zscore"], dtype=float)
    if n < window:
        return out
    x = np.arange(window, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    win = sliding_window_view(y, window)            # shape (n-window+1, window)
    ym = win.mean(axis=1)
    sxy = ((x - xm)[None, :] * (win - ym[:, None])).sum(axis=1)
    slope = sxy / sxx
    intercept = ym - slope * xm
    fitted_last = intercept + slope * (window - 1)
    fit_all = intercept[:, None] + slope[:, None] * x[None, :]
    resid = win - fit_all
    sigma = np.sqrt((resid ** 2).mean(axis=1))
    z = (win[:, -1] - fitted_last) / np.where(sigma > 0, sigma, np.nan)
    idx = slice(window - 1, n)
    out.iloc[idx, 0] = np.exp(fitted_last)
    out.iloc[idx, 1] = sigma
    out.iloc[idx, 2] = slope
    out.iloc[idx, 3] = z
    return out


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from t to t+horizon (NaN near the end). Used only in research, never live."""
    return close.shift(-horizon) / close - 1.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def cagr(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (len(equity) - 1) / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())
