"""Core correctness tests:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import ExitRules, simulate  # noqa: E402
from goldsignal.bars import align_htf, resample  # noqa: E402
from goldsignal.datafeed import decode_candles  # noqa: E402
from goldsignal.news import parse_feed, score_headline  # noqa: E402


def synthetic_m15(n=9000, seed=7):
    """Random walk with regime-switching drift, 15m bars, weekdays only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 00:00", periods=n * 2, freq="15min", tz="UTC")
    ny = idx.tz_convert("America/New_York")
    keep = ~((ny.dayofweek == 5) | ((ny.dayofweek == 6) & (ny.hour < 18)) | ((ny.dayofweek == 4) & (ny.hour >= 17)) | (ny.hour == 17))
    idx = idx[keep][:n]
    drift = np.repeat(rng.choice([-0.0004, 0.0, 0.0004], size=n // 400 + 1), 400)[:n]
    ret = drift + rng.normal(0, 0.0015, n)
    close = 2000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.001, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx)


def test_decode_candles_roundtrip():
    js = {"timestamp": 1790726400000, "multiplier": 0.001, "open": 4184.555, "high": 4187.135, "low": 4184.275,
          "close": 4186.895, "shift": 60000, "times": [0, 1, 2], "opens": [0, 100, -50], "highs": [0, 50, 0],
          "lows": [0, 10, -20], "closes": [0, -100, 30], "volumes": [1, 2, 3]}
    df = decode_candles(js)
    assert list(df.index.minute) == [0, 1, 3]  # gap of 2 minutes kept as a gap
    assert df["open"].tolist() == pytest.approx([4184.555, 4184.655, 4184.605])
    assert df["close"].tolist() == pytest.approx([4186.895, 4186.795, 4186.825])


def test_htf_alignment_has_no_lookahead():
    m15 = synthetic_m15(800)
    h1 = resample(m15, "1h")
    al = align_htf(m15.index, pd.Timedelta(minutes=15), h1[["close"]], pd.Timedelta(hours=1))
    for ts, row in al.dropna().iterrows():
        bar_close = ts + pd.Timedelta(minutes=15)
        known = h1[h1.index + pd.Timedelta(hours=1) <= bar_close]
        assert row["close"] == known["close"].iloc[-1]


@pytest.mark.parametrize("squeeze", [0, 16])
def test_signals_do_not_change_when_future_data_is_added(squeeze):
    """The rule table at bar t must be identical whether or not bars after t exist
    (also with the 布林收窄濾網, whose band-width ranks use a rolling window)."""
    from dataclasses import replace
    p = replace(S.StrategyParams(), squeeze_bars=squeeze, use_h4=False, adx_max=100.0)
    m15 = synthetic_m15(9000)
    full = S.rule_table(m15, S.compute_features(m15), p)
    assert (full["long_signal"] | full["short_signal"]).sum() > 0
    for cut in (6000, 7337, 8999):
        part = m15.iloc[:cut]
        t = S.rule_table(part, S.compute_features(part), p)
        cols = ["long_signal", "short_signal", "long_setup", "short_setup", "squeeze_ok"]
        assert (t[cols].values == full[cols].iloc[:cut].values).all(), f"look-ahead detected at cut={cut}"


def test_simulator_stop_and_costs():
    idx = pd.date_range("2025-01-06 14:00", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 100, 99, 98, 98], "high": [100.5] * 6, "low": [99.5, 99.5, 99.5, 97.0, 97.5, 97.5],
                       "close": [100, 100, 99.5, 98, 98, 98]}, index=idx, dtype=float)
    sig = np.array([1, 0, 0, 0, 0, 0])
    stop_long = np.full(6, 98.5)
    tr = simulate(df, sig, stop_long, stop_long, ExitRules(tp_r=0, cooldown=0), spread=np.full(6, 0.1), slip=0.0)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["entry"] == pytest.approx(100.1)       # buy at ask
    assert t["reason"] == "stop"
    assert t["exit"] == pytest.approx(98.5)
    assert t["R"] == pytest.approx(-1.0)


def test_trailing_stop_only_tightens():
    idx = pd.date_range("2025-01-06 14:00", periods=8, freq="15min", tz="UTC")
    close = np.array([100, 101, 103, 105, 105, 107, 105.5, 99], float)
    df = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close}, index=idx)
    sig = np.zeros(8, int); sig[0] = 1
    atr = np.full(8, 1.0)
    tr = simulate(df, sig, np.full(8, 95.0), np.full(8, 0.0), ExitRules(tp_r=0, trail_mult=2.0, cooldown=0),
                  spread=np.zeros(8), slip=0.0, atr=atr)
    t = tr.iloc[0]
    assert t["reason"] == "trail"
    assert t["exit"] == pytest.approx(107.3 - 2.0)  # best high 107.3 minus 2 ATR, never loosened


def test_news_scoring_and_rss_parsing():
    s, rel, _ = score_headline("Gold climbs to record high as dollar weakens")
    assert s > 0.5 and rel == 1.0
    s, rel, _ = score_headline("Gold slips as Treasury yields jump")
    assert s < -0.5
    assert score_headline("New smartphone launched")[1] == 0.0
    s, rel, _ = score_headline("Silver jumps 3% as industrial demand improves", "silver")
    assert s > 0.5 and rel == 1.0
    assert score_headline("Silver jumps 3%", "gold")[1] == 0.0
    xml = """<rss><channel><item><title>Gold rises</title><link>https://x/1</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_feed(xml, "test")
    assert items[0]["title"] == "Gold rises" and items[0]["time"].year == 2026
