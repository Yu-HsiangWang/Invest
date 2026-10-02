"""FastAPI web server: serves the dashboard and a small JSON API."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Hub
from .instruments import INSTRUMENTS

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SYM = "^(XAUUSD|XAGUSD)$"


def create_app(hub: Hub) -> FastAPI:
    app = FastAPI(title="Precious metals signal", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _start():
        hub.start()

    @app.on_event("shutdown")
    def _stop():
        hub.stop()

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/state")
    def state(sym: str = Query("XAUUSD", pattern=SYM)):
        return JSONResponse(hub.state(sym))

    @app.get("/api/candles")
    def candles(sym: str = Query("XAUUSD", pattern=SYM), tf: str = Query("15m", pattern="^(5m|15m|1h|4h|1D)$"),
                limit: int = Query(600, ge=50, le=3000), show: str | None = Query(None, max_length=300),
                auto: int = Query(1, ge=0, le=1)):
        e = hub.engines.get(sym)
        sel = None if show is None else {x for x in show.split(",") if x}
        return JSONResponse(e.candles(tf, limit, sel, bool(auto)) if e else {"bars": []})

    @app.get("/api/backtest")
    def backtest(sym: str = Query("XAUUSD", pattern=SYM)):
        out = {}
        for kind in ("backtest_summary", "recent_validation", "money"):
            p = ROOT / "results" / f"{sym}_{kind}.json"
            if p.exists():
                out[kind] = json.loads(p.read_text(encoding="utf-8"))
        return JSONResponse(out)

    @app.get("/api/settings")
    def get_settings():
        return hub.settings.public()

    @app.post("/api/settings")
    def post_settings(data: dict = Body(...)):
        for k in hub.settings.SECRET_KEYS:
            if data.get(k) == "********":
                data.pop(k)
        return hub.update_settings(data)

    @app.post("/api/my_position")
    def my_position(sym: str = Query("XAUUSD", pattern=SYM), data: dict = Body(default={})):
        hub.set_my_position(sym, data if data.get("entry") else None)
        return {"ok": True}

    @app.post("/api/replay")
    def replay(data: dict = Body(...)):
        hub.replay_control(data)
        return {"ok": True}

    @app.post("/api/test_alert")
    def test_alert():
        hub.alert(INSTRUMENTS["XAUUSD"], "test", "🔔 測試通知", "如果你看到這則通知並聽到聲音，提醒功能就正常。", "signal")
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
