"""Core maths: no look-ahead, regression channel matches polyfit, simulator costs, zone labelling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stocksignal import zones as Z  # noqa: E402
from stocksignal.backtest import TW_STOCK, US_STOCK, conditional_forward, dca, metrics, simulate  # noqa: E402
from stocksignal.indicators import rolling_log_regression, sma  # noqa: E402
from stocksignal.regime import score_to_state, trend_score  # noqa: E402


def _series(n=2000, seed=1, drift=0.0004):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.015, n)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-01", periods=n))
    df = pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0})
    return df


def test_regression_channel_matches_polyfit():
    df = _series()
    reg = rolling_log_regression(df["close"], 500)
    y = np.log(df["close"].to_numpy()[-500:])
    x = np.arange(500)
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    assert np.isclose(reg["center"].iloc[-1], np.exp(a + b * 499))
    assert np.isclose(reg["sigma"].iloc[-1], np.sqrt(((y - fit) ** 2).mean()))
    assert np.isclose(reg["zscore"].iloc[-1], (y[-1] - fit[-1]) / reg["sigma"].iloc[-1])


def test_no_lookahead_in_zones_and_trend():
    df = _series(1500)
    full = Z.channel_bands(df["close"], Z.ChannelParams(window=400))
    cut = Z.channel_bands(df["close"].iloc[:1200], Z.ChannelParams(window=400))
    pd.testing.assert_series_equal(full["zscore"].iloc[:1200], cut["zscore"], check_names=False)
    dv_full = Z.deviation_bands(df["close"], Z.DeviationParams(rank_window=500))
    dv_cut = Z.deviation_bands(df["close"].iloc[:1200], Z.DeviationParams(rank_window=500))
    pd.testing.assert_series_equal(dv_full["dev_rank"].iloc[:1200], dv_cut["dev_rank"], check_names=False)
    ts_full = trend_score(df["close"])
    ts_cut = trend_score(df["close"].iloc[:1200])
    pd.testing.assert_series_equal(ts_full.iloc[:1200], ts_cut, check_names=False)


def test_zone_labels():
    z = pd.Series([-2.5, -1.5, 0.0, 1.5, 2.5, np.nan])
    out = Z.channel_zone(z)
    assert out.tolist()[:5] == [-2, -1, 0, 1, 2]
    assert np.isnan(out.iloc[5])
    r = pd.Series([0.01, 0.10, 0.5, 0.90, 0.99])
    assert Z.deviation_zone(r).tolist() == [-2, -1, 0, 1, 2]


def test_simulator_buy_and_hold_and_costs():
    df = _series(600)
    res = simulate(df, pd.Series(1.0, index=df.index), US_STOCK)
    # first trade executes at the open of day 2 with slippage, then tracks the close
    expected = df["close"].iloc[-1] / df["open"].iloc[1] / (1 + US_STOCK.buy())
    assert np.isclose(res["equity"].iloc[-1], expected, rtol=1e-9)
    # a round trip in Taiwan costs about 0.57% (fees both ways + 0.3% tax)
    tgt = pd.Series(0.0, index=df.index)
    tgt.iloc[10] = 1.0
    tgt.iloc[11] = 0.0
    res = simulate(df, tgt, TW_STOCK, exec_at="open")
    px_in, px_out = df["open"].iloc[11], df["open"].iloc[12]
    gross = px_out / px_in
    net = gross * (1 - TW_STOCK.sell()) / (1 + TW_STOCK.buy())
    assert np.isclose(res["equity"].iloc[-1], net, rtol=1e-9)
    m = metrics(res)
    assert 0 < m["time_in_market"] < 0.01


def test_dca_tilt_runs():
    df = _series(800)
    w = pd.Series(1.0, index=df.index)
    out = dca(df, w, US_STOCK)
    assert out["invested"].iloc[-1] > 30
    assert out["equity"].iloc[-1] > 0


def test_conditional_forward_shape():
    df = _series(1200)
    st = score_to_state(trend_score(df["close"]))
    t = conditional_forward(df["close"], st, horizons=(63,))
    assert set(t.columns) >= {"state", "mean", "hit", "n"}
    assert (t["state"] == "all").any()


def test_sma_warmup():
    s = pd.Series(np.arange(10, dtype=float))
    m = sma(s, 5)
    assert m.isna().sum() == 4 and m.iloc[-1] == 7.0
