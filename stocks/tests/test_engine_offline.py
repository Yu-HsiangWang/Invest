"""Offline end-to-end: the Hub analyses the bundled history and the API serves it without network."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stocksignal.engine import Hub  # noqa: E402
from stocksignal.server import clean, create_app  # noqa: E402


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    h = Hub(ROOT, offline=True)
    h.settings.watch_tw = ["2330.TW", "0050.TW", "0056.TW"]
    h.settings.watch_us = ["QQQ", "NVDA", "AAPL"]
    h.refresh_prices()
    h.refresh_feeds()
    return h


def test_overview_and_verdicts(hub):
    ov = hub.overview("TW")
    assert len(ov["cards"]) == 3
    for c in ov["cards"]:
        assert c["verdict"] in ("便宜區", "偏便宜", "合理區", "略貴", "偏貴區", "資料不足")
        assert c["price"] > 0
    assert "0050.TW" in ov["regime"] and "^TWII" in ov["regime"]


def test_instrument_payload_is_json_safe(hub):
    d = hub.instrument("2330.TW")
    assert d["zones"] and d["zones"][0]["method"] == "channel"
    assert set(d["zones"][0]["levels"]) == {"dn2", "dn1", "center", "up1", "up2"}
    assert d["zones"][0]["levels"]["dn2"] < d["zones"][0]["levels"]["up2"]
    assert "pe" in {z["method"] for z in d["zones"]}          # bundled TWSE valuation history
    import json
    json.dumps(clean(d), allow_nan=False)


def test_chart_timestamps(hub):
    ch = hub.chart("QQQ", "1M", 200)
    assert ch["bars"] and ch["bars"][-1]["time"] > 1_700_000_000
    assert "ch_dn2" in ch["lines"]


def test_congress_offline_seed(hub):
    d = hub.instrument("NVDA")
    cg = d["feeds"].get("congress")
    assert cg and cg["buys"] >= 0
    mv = hub.market_view("US")
    assert isinstance(mv["congress_recent"], list)


def test_api_routes(hub):
    from fastapi.testclient import TestClient
    app = create_app(hub)
    with TestClient(app) as c:
        assert c.get("/api/overview?market=US").status_code == 200
        assert c.get("/api/instrument?sym=QQQ").json()["symbol"] == "QQQ"
        assert c.get("/api/market?market=TW").status_code == 200
        assert c.get("/api/research").status_code == 200
        assert c.get("/").status_code == 200
        bad = c.get("/api/instrument?sym=../x")
        assert bad.status_code == 422
