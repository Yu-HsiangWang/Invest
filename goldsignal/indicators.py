"""Technical indicators used by both the live signal engine and the backtester.

All functions take pandas Series (or DataFrames with open/high/low/close columns)
and return Series aligned to the input index. Nothing here looks ahead: the value
at bar t only uses bars <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average (used by ATR, RSI, ADX)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = rma(delta.clip(lower=0.0), n)
    down = rma((-delta).clip(lower=0.0), n)
    rs = up / down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(down.notna())


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Returns DataFrame with columns adx, plus_di, minus_di."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_n = rma(true_range(df), n)
    plus_di = 100.0 * rma(pd.Series(plus_dm, index=df.index), n) / tr_n
    minus_di = 100.0 * rma(pd.Series(minus_dm, index=df.index), n) / tr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame({"adx": rma(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / sum(|bar moves|). 1 = straight line trend, ~0 = chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.replace(0.0, np.nan)


def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Channel of the previous n bars (excludes the current bar, so a close above `upper` is a breakout)."""
    upper = df["high"].rolling(n, min_periods=n).max().shift(1)
    lower = df["low"].rolling(n, min_periods=n).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Classic Supertrend. direction = +1 (up) / -1 (down)."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()
    upper_basic = hl2 + mult * a
    lower_basic = hl2 - mult * a
    m = len(df)
    upper = np.full(m, np.nan)
    lower = np.full(m, np.nan)
    direction = np.zeros(m)
    line = np.full(m, np.nan)
    for i in range(m):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(upper[i - 1]):
            upper[i], lower[i] = upper_basic[i], lower_basic[i]
            direction[i] = 1.0
        else:
            upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
            lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
            if direction[i - 1] == 1.0:
                direction[i] = -1.0 if close[i] < lower[i] else 1.0
            else:
                direction[i] = 1.0 if close[i] > upper[i] else -1.0
        line[i] = lower[i] if direction[i] == 1.0 else upper[i]
    return pd.DataFrame({"line": line, "direction": direction}, index=df.index)


def slope(s: pd.Series, n: int) -> pd.Series:
    """Change over n bars (simple, robust slope proxy)."""
    return s - s.shift(n)


def swing_low(df: pd.DataFrame, n: int) -> pd.Series:
    return df["low"].rolling(n, min_periods=n).min()


def swing_high(df: pd.DataFrame, n: int) -> pd.Series:
    return df["high"].rolling(n, min_periods=n).max()


def stoch_kd(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """KD (Taiwan-style stochastic, 9,3,3): RSV over n bars, K and D smoothed by 1/3."""
    lo = df["low"].rolling(n, min_periods=n).min()
    hi = df["high"].rolling(n, min_periods=n).max()
    rsv = (100.0 * (df["close"] - lo) / (hi - lo).replace(0.0, np.nan)).to_numpy()
    k = np.full(len(df), np.nan)
    d = np.full(len(df), np.nan)
    pk = pd_ = 50.0
    for i, r in enumerate(rsv):
        if np.isnan(r):
            continue
        pk = pk * 2.0 / 3.0 + r / 3.0
        pd_ = pd_ * 2.0 / 3.0 + pk / 3.0
        k[i], d[i] = pk, pd_
    return pd.DataFrame({"k": k, "d": d}, index=df.index)


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    """Wilder's Parabolic SAR. direction = +1 (SAR below price, uptrend) / -1."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    m = len(df)
    sar = np.full(m, np.nan)
    direction = np.zeros(m)
    if m < 3:
        return pd.DataFrame({"sar": sar, "direction": direction}, index=df.index)
    up = h[1] >= h[0]
    ep = h[1] if up else l[1]
    s = l[0] if up else h[0]
    af = step
    for i in range(1, m):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            s = max(s, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        sar[i] = s
        direction[i] = 1.0 if up else -1.0
    return pd.DataFrame({"sar": sar, "direction": direction}, index=df.index)


def ichimoku(df: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52) -> pd.DataFrame:
    """Tenkan, Kijun and the cloud that applies to each bar (spans computed `k` bars earlier)."""
    mid = lambda n: (df["high"].rolling(n, min_periods=n).max() + df["low"].rolling(n, min_periods=n).min()) / 2.0  # noqa: E731
    tenkan, kijun = mid(t), mid(k)
    span_a = ((tenkan + kijun) / 2.0).shift(k)
    span_b = mid(s).shift(k)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b})


def session_vwap(df: pd.DataFrame, session_id: np.ndarray) -> pd.Series:
    """Volume-weighted average price restarting every trading day (tick volume)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0).clip(lower=0.0)
    vol = vol.where(vol > 0, 1.0)  # bars without volume still count once
    g = pd.Series(session_id, index=df.index)
    pv = (tp * vol).groupby(g).cumsum()
    vv = vol.groupby(g).cumsum()
    return pv / vv
