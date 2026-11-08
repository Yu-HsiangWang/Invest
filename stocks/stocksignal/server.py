"""FastAPI web server: serves the dashboard and a small JSON API."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Hub


def clean(o):
    """Make any analysis payload JSON-safe (NaN -> null, numpy / pandas scalars -> python, dates -> iso)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and (math.isnan(o) or math.isinf(o))) or (isinstance(o, np.floating) and (np.isnan(o) or np.isinf(o))) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return None if pd.isna(o) else o.isoformat()
    if o is pd.NaT or o is pd.NA:
        return None
    try:
        if pd.isna(o) is True:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(o, "item"):
        return clean(o.item())
    return o


class R(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False).encode("utf-8")

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
MKT = "^(TW|US)$"
SYM = r"^[\^A-Za-z0-9.\-]{1,12}$"


def create_app(hub: Hub) -> FastAPI:
    app = FastAPI(title="Stock zones", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _start():
        hub.start()

    @app.on_event("shutdown")
    def _stop():
        hub.stop()

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/overview")
    def overview(market: str = Query("TW", pattern=MKT)):
        return R(hub.overview(market))

    @app.get("/api/instrument")
    def instrument(sym: str = Query(..., pattern=SYM)):
        return R(hub.instrument(sym.upper()))

    @app.get("/api/chart")
    def chart(sym: str = Query(..., pattern=SYM), tf: str = Query("1D", pattern="^(1D|1W|1M)$"), limit: int = Query(1500, ge=100, le=8000)):
        return R(hub.chart(sym.upper(), tf, limit))

    @app.get("/api/market")
    def market(market: str = Query("TW", pattern=MKT)):
        return R(hub.market_view(market))

    @app.get("/api/research")
    def research():
        return R(hub.research or {})

    @app.get("/api/settings")
    def get_settings():
        return hub.settings.public()

    @app.post("/api/settings")
    def post_settings(data: dict = Body(...)):
        for k in hub.settings.SECRET_KEYS:
            if data.get(k) == "********":
                data.pop(k)
        return hub.update_settings(data)

    @app.post("/api/watchlist")
    def watchlist(data: dict = Body(...)):
        market = str(data.get("market", "TW")).upper()
        if market not in ("TW", "US"):
            return {"ok": False, "error": "market"}
        if data.get("action") == "remove":
            return hub.remove_symbol(market, str(data.get("symbol", "")).upper())
        return hub.add_symbol(market, str(data.get("symbol", "")), str(data.get("name", "")), str(data.get("kind", "")))

    @app.post("/api/holding")
    def holding(data: dict = Body(...)):
        sym = str(data.get("symbol", "")).upper()
        if not sym:
            return {"ok": False}
        if data.get("shares"):
            hub.settings.my_holdings[sym] = {"shares": float(data["shares"]), "cost": float(data.get("cost") or 0)}
        else:
            hub.settings.my_holdings.pop(sym, None)
        hub.settings.save(ROOT / "settings.json")
        return {"ok": True}

    @app.post("/api/refresh")
    def refresh():
        import threading
        threading.Thread(target=hub._refresh_all, daemon=True).start()
        return {"ok": True}

    @app.post("/api/test_alert")
    def test_alert():
        hub.alert("TEST", "test", "🔔 測試通知", "如果你看到這則通知，提醒功能就正常。", "signal")
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
