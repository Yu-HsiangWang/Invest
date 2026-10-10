"""Smoke test of the LIVE code path with the network replaced by a fake Dukascopy API
built from historical bars (no internet needed)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import datafeed  # noqa: E402
from goldsignal.engine import Hub, LiveSource  # noqa: E402
from tests.test_core import synthetic_m15  # noqa: E402


def encode(df: pd.DataFrame, shift_ms: int, mult: float = 0.001) -> dict:
    if df.empty:
        return {"timestamp": 0, "multiplier": mult, "open": 0, "high": 0, "low": 0, "close": 0, "shift": shift_ms,
                "times": [], "opens": [], "highs": [], "lows": [], "closes": [], "volumes": []}
    ts = (df.index.view("int64") // 10**6) if df.index.dtype.str.endswith("[ns, UTC]") else np.array([int(x.timestamp() * 1000) for x in df.index])
    units = {k: np.round(df[k].to_numpy() / mult).astype(np.int64) for k in ("open", "high", "low", "close")}
    base_t = int(ts[0])
    times = np.diff(np.concatenate([[base_t], ts])) // shift_ms
    js = {"timestamp": base_t, "multiplier": mult, "shift": shift_ms, "times": times.tolist(),
          "volumes": [1.0] * len(df)}
    for k, arr in (("open", "opens"), ("high", "highs"), ("low", "lows"), ("close", "closes")):
        u = units[k]
        js[k] = float(u[0] * mult)
        js[arr] = np.diff(np.concatenate([[u[0]], u])).tolist()
    return js


def minutes_from_15m(m15: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, r in m15.iterrows():
        path = [r.open, r.high, r.low, r.close] if r.close >= r.open else [r.open, r.low, r.high, r.close]
        pts = np.interp(np.linspace(0, 3, 16), [0, 1, 2, 3], path)
        for k in range(15):
            a, b = pts[k], pts[k + 1]
            rows.append((t + pd.Timedelta(minutes=k), a, max(a, b), min(a, b), b))
    out = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"]).set_index("time")
    return out


@pytest.fixture()
def fake_api(monkeypatch):
    now = datetime.now(timezone.utc)
    m15 = synthetic_m15(70 * 70)
    shift = (pd.Timestamp(now) - m15.index[-1]).floor("7D")
    m15.index = m15.index + shift
    m1 = minutes_from_15m(m15)

    class Resp:
        def __init__(self, js, code=200):
            self._js, self.status_code, self.text = js, code, ""

        def json(self):
            return self._js

        def raise_for_status(self):
            pass

    def fake_get(self, url, timeout=None, **kw):
        if "swissquote" in url:
            last = float(m1["close"].iloc[-1])
            return Resp([{"topo": {"platform": "x"}, "spreadProfilePrices": [{"bid": last, "ask": last + 0.4}], "ts": int(now.timestamp() * 1000)}])
        if "jetta.dukascopy.com" not in url:
            raise ConnectionError("blocked in test")
        if "/instruments" in url:
            return Resp({})
        parts = url.split("?")[0].split("/")
        kind = parts[parts.index("candles") + 1]
        frm = int(url.split("from=")[1]) if "from=" in url else None
        src = m1 if kind == "minute" else m1.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if frm is not None:
            start = pd.Timestamp(frm, unit="ms", tz="UTC")
            end = start + (pd.Timedelta(days=1) if kind == "minute" else pd.Timedelta(days=31))
        else:
            y, m = int(parts[-3 if kind == "minute" else -2]), int(parts[-2 if kind == "minute" else -1])
            d = int(parts[-1]) if kind == "minute" else 1
            start = pd.Timestamp(year=y, month=m, day=d, tz="UTC")
            end = start + pd.Timedelta(days=1) if kind == "minute" else (start + pd.offsets.MonthBegin(1))
        sub = src[(src.index >= start) & (src.index < end)]
        return Resp(encode(sub, 60000 if kind == "minute" else 3600000))

    monkeypatch.setattr(datafeed.requests.Session, "get", fake_get)
    import goldsignal.news as news
    monkeypatch.setattr(news.requests.Session, "get", fake_get)
    return m15


@pytest.mark.parametrize("sym", ["XAUUSD", "XAGUSD"])
def test_live_engine_end_to_end(fake_api, tmp_path, sym):
    hub = Hub(tmp_path, symbols=[sym])
    eng = hub.engines[sym]
    assert isinstance(eng.src, LiveSource)
    eng.src.bootstrap()
    assert len(eng.src.m1) > 5000
    eng._tick()
    snap = hub.state(sym)
    assert snap["bar_time"] is not None and snap["state"] in ("flat", "long", "short", "signal_long", "signal_short")
    assert snap["price"]["bid"] > 0 and len(snap["mtf"]) == 4
    assert snap["instrument"]["key"] == sym and snap["instruments"][0]["key"] == sym
    assert snap["day"]["on"] is True and snap["sizing"]["balance"] == 2000.0          # 當日平倉模式 is the default
    assert len(snap["lots_table"]) >= 5 and "eod" in snap
    keys = [c["key"] for c in snap["setup"]["long"]["checks"]]
    assert ("squeeze" in keys) == (sym == "XAUUSD")                                   # 布林收窄濾網: gold only
    hub.update_settings({"instruments": {sym: {"squeeze_filter": sym != "XAUUSD"}}})
    assert eng.params.squeeze_bars == (0 if sym == "XAUUSD" else 16)
    # 積極模式: 12-hour breakouts, no squeeze filter, several positions allowed
    hub.update_settings({"active_mode": True})
    assert eng.active_mode and eng.params.breakout_hours == 12 and eng.params.squeeze_bars == 0
    eng._tick()
    snap = hub.state(sym)
    assert snap["active_mode"] is True and snap["max_open"] == 2 and isinstance(snap["positions"], list)
    assert snap["cooldown_until"] is None
    assert eng.candles("15m", 300)["ta"]["headline"]["title"]
    hub.update_settings({"active_mode": False})
    assert eng.params.breakout_hours == 24
    # switching modes rebuilds the positions: no false "stopped out" alert, one info alert instead
    pos = [{"entry_time": "2026-09-01T00:00:00Z", "dir": 1, "entry": 100.0, "stop": 99.0},
           {"entry_time": "2026-09-01T01:00:00Z", "dir": 1, "entry": 101.0, "stop": 99.0}]
    n0 = len(hub.alerts)
    eng._emit_transition_alerts({"active_mode": True, "params": {"breakout_hours": 12}, "positions": pos, "active": pos[-1]},
                                {"active_mode": False, "params": {"breakout_hours": 24}, "positions": [], "history": []})
    assert [a["kind"] for a in list(hub.alerts)[n0:]] == ["info"]
    # two positions whose trailing stops moved to the same price -> a single stop-move alert
    n0 = len(hub.alerts)
    moved = [dict(x, stop=99.5) for x in pos]
    eng._emit_transition_alerts({"active_mode": True, "params": {}, "positions": pos, "active": pos[-1]},
                                {"active_mode": True, "params": {}, "positions": moved, "active": moved[-1], "history": [], "atr_h1": 1.0})
    assert [a["kind"] for a in list(hub.alerts)[n0:]] == ["stop_move"]
    for tf in ("5m", "15m", "1h", "4h", "1D"):
        c = eng.candles(tf, 300)
        assert "ta" in c and c["ta"]["headline"]["title"]
        assert len(c["bars"]) > 10
        times = [b["time"] for b in c["bars"]]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert all(v is not None for b in c["bars"] for v in b.values())
