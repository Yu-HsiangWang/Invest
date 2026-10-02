"""Chart reference tools: a clean synthetic triangle and its breakout are found."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldsignal import chartlab  # noqa: E402


def _zigzag(points, n_per_leg=8):
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        closes += list(np.linspace(a, b, n_per_leg, endpoint=False))
    closes.append(points[-1])
    c = np.array(closes)
    idx = pd.date_range("2026-09-01 13:00", periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1.0}, index=idx)


def test_symmetrical_triangle_and_breakout():
    pre = list(np.linspace(80, 100, 6)) + [95, 100, 95, 100, 95, 100, 96]
    tri = [100, 88, 98, 90, 96, 92, 94]            # lower highs, higher lows
    bars = _zigzag(pre + tri, 8)
    an = chartlab.analyze(bars, "1h")
    assert an["triangle"] and an["triangle"]["type"] == "sym" and an["triangle"]["breakout"] is None
    ext = _zigzag([94, 96.5, 98.5], 1).iloc[1:]
    ext.index = bars.index[-1] + pd.Timedelta(hours=1) * np.arange(1, len(ext) + 1)
    up = pd.concat([bars, ext])
    an2 = chartlab.analyze(up, "1h")
    assert an2["triangle"] and an2["triangle"]["breakout"] == "up"
    txt = chartlab.describe(an2, {"state": "flat", "bias": 1, "setup": {}}, lambda x: f"{x:.2f}", "1 小時 K", "黃金")
    assert any("突破" in x["text"] for x in txt["lines"])


def test_fib_and_levels_exist():
    bars = _zigzag([100, 130, 118, 126, 119, 125, 120], 20)
    an = chartlab.analyze(bars, "1h")
    assert an["fib"]["dir"] == 1 and 0 < an["fib"]["retrace"] < 1
    assert any(x["kind"] == "support" for x in an["levels"]) and any(x["kind"] == "resistance" for x in an["levels"])
