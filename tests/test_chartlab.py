"""Chart-reading tools: patterns, Fibonacci/levels, indicators and the pop-up logic."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldsignal import chartlab, indicators as ind  # noqa: E402
from tests.test_core import synthetic_m15  # noqa: E402

FMT = lambda x: f"{x:.2f}"  # noqa: E731
CORE = {"state": "flat", "bias": 1, "setup": {}}


def _zigzag(points, n_per_leg=8, start="2026-09-01 13:00"):
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        closes += list(np.linspace(a, b, n_per_leg, endpoint=False))
    closes.append(points[-1])
    c = np.array(closes)
    idx = pd.date_range(start, periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1.0}, index=idx)


def _tool(out, key):
    return next(m for m in out["tools"] if m["key"] == key)


def test_triangle_forming_then_breakout():
    pre = list(np.linspace(80, 100, 6)) + [95, 100, 95, 100, 95, 100, 96]
    tri = [100, 88, 98, 90, 96, 92, 94]               # lower highs, higher lows
    bars = _zigzag(pre + tri, 8)
    out = chartlab.build(bars, "1h", None, CORE, FMT, "黃金", show={"pattern"}, auto=False)
    txt = " ".join(x["text"] for x in out["lines"] if x["tool"] == "pattern")
    assert "收斂三角形" in txt and "pattern" in out["draw"]
    ext = _zigzag([94, 96.5, 98.5], 1).iloc[1:]
    ext.index = bars.index[-1] + pd.Timedelta(hours=1) * np.arange(1, len(ext) + 1)
    out2 = chartlab.build(pd.concat([bars, ext]), "1h", None, CORE, FMT, "黃金", show=set(), auto=True)
    pat = _tool(out2, "pattern")
    assert pat["auto"] and pat["visible"] and "突破" in pat["why"]       # a breakout pops up by itself
    assert any(m["text"].startswith("突破") for m in out2["draw"]["pattern"]["markers"])


def test_double_top_detected():
    bars = _zigzag([90, 100, 94, 104, 96, 104.2, 98.5], 10)
    out = chartlab.build(bars, "1h", None, CORE, FMT, "黃金", show={"pattern"}, auto=False)
    txt = " ".join(x["text"] for x in out["lines"] if x["tool"] == "pattern")
    assert "M 頭" in txt


def test_fib_and_levels_exist():
    bars = _zigzag([100, 130, 118, 126, 119, 125, 120], 20)
    out = chartlab.build(bars, "1h", None, CORE, FMT, "黃金", show={"sr", "fib"}, auto=False)
    assert out["draw"]["sr"]["levels"] and any("回檔" in x["text"] for x in out["lines"] if x["tool"] == "fib")
    assert set(out["draw"]) == {"sr", "fib"}                             # nothing pops up when auto is off


def test_indicators_ranges_and_divergence():
    df = synthetic_m15(3000)
    kd = ind.stoch_kd(df).dropna()
    assert kd["k"].between(0, 100).all() and kd["d"].between(0, 100).all()
    sar = ind.parabolic_sar(df)
    up = sar["direction"] > 0
    assert (sar.loc[up, "sar"] <= df.loc[up, "high"] + 1e-9).all()
    bars = _zigzag([90, 100, 95, 100, 96, 104, 99, 110, 104, 111, 108.5], 12)
    out = chartlab.build(bars, "1h", None, CORE, FMT, "黃金", show={"rsi", "kd", "macd"}, auto=False)
    assert all(k in out["draw"] and out["draw"][k]["pane"]["series"][0]["points"] for k in ("rsi", "kd", "macd"))
    # bearish divergence: the second swing high is higher, the oscillator's is lower
    b = bars.iloc[:115]                                   # the last swing high is recent
    cx = chartlab.Ctx(bars=b, tf="1h", t=list(range(len(b))), o=b["open"].to_numpy(), h=b["high"].to_numpy(),
                      l=b["low"].to_numpy(), c=b["close"].to_numpy(), v=b["volume"].to_numpy(), n=len(b),
                      atr=1.0, price=float(b["close"].iloc[-1]), fmt=FMT)
    ph, _ = chartlab.pivots(cx.h, cx.l, 3)
    osc = np.full(cx.n, 50.0)
    osc[ph[-2]], osc[ph[-1]] = 80.0, 60.0
    dv = chartlab._divergence(cx, osc, 3.0)
    assert dv is not None and dv[0] == -1


def test_every_tool_runs_on_real_like_data():
    df = synthetic_m15(4000)
    for tf in ("15m", "1h"):
        bars = df if tf == "15m" else df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        out = chartlab.build(bars.iloc[-700:], tf, None, CORE, FMT, "黃金", show={k for k, *_ in chartlab.TOOLS}, auto=True)
        assert out["regime"]["label"] and len(out["tools"]) == len(chartlab.TOOLS)
        assert sum(m["auto"] for m in out["tools"]) <= chartlab.AUTO_MAX
