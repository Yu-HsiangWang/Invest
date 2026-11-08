"""User settings stored in settings.json next to the app (edited from the web UI)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .universe import TW_DEFAULT, US_DEFAULT


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


@dataclass
class Settings:
    markets: list = field(default_factory=lambda: ["TW", "US"])
    watch_tw: list = field(default_factory=lambda: [i.symbol for i in TW_DEFAULT])
    watch_us: list = field(default_factory=lambda: [i.symbol for i in US_DEFAULT])
    custom_names: dict = field(default_factory=dict)      # symbol -> {"name":..., "market":..., "kind":...} for user-added tickers
    defense_mode: bool = False                             # 防守模式: ETF / index 10-month SMA filter suggestions
    color_convention: str = "tw"                           # "tw" 紅漲綠跌 / "intl"
    sound: bool = True
    desktop_notify: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    finmind_token: str = ""                                # optional, raises the FinMind rate limit
    sec_user_agent: str = "stocksignal-dashboard/0.1 (personal research tool)"   # SEC asks for a contact in the UA
    price_refresh_hours: float = 6.0
    feeds_refresh_minutes: float = 45.0
    alert_zone_changes: bool = True
    alert_congress_cluster: bool = True
    alert_insider_buys: bool = True
    alert_trend_flips: bool = True
    my_holdings: dict = field(default_factory=dict)        # symbol -> {"shares": float, "cost": float}

    SECRET_KEYS = ("telegram_bot_token", "discord_webhook_url", "finmind_token")

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        p = Path(path)
        s = cls()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                s.update(raw)
            except Exception:  # noqa: BLE001
                pass
        s.save(p)
        return s

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def update(self, data: dict) -> None:
        known = {f.name: f for f in fields(self)}
        for k, v in data.items():
            if k not in known or k == "SECRET_KEYS":
                continue
            cur = getattr(self, k)
            if isinstance(cur, bool):
                setattr(self, k, _to_bool(v))
            elif isinstance(cur, float):
                try:
                    setattr(self, k, float(v))
                except (TypeError, ValueError):
                    pass
            elif isinstance(cur, list):
                if isinstance(v, str):
                    v = [x.strip() for x in v.replace("\n", ",").split(",") if x.strip()]
                if isinstance(v, list):
                    setattr(self, k, [str(x).strip().upper() if k.startswith("watch") else x for x in v])
            elif isinstance(cur, dict):
                if isinstance(v, dict):
                    setattr(self, k, v)
            else:
                setattr(self, k, "" if v is None else str(v))

    def public(self) -> dict:
        d = asdict(self)
        for k in self.SECRET_KEYS:
            if d.get(k):
                d[k] = "********"
        return d
