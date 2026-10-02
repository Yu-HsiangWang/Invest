"""User settings stored in settings.json next to the app (edited from the web UI)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .instruments import INSTRUMENTS


def _default_instruments() -> dict:
    return {k: {"oz_per_lot": ins.default_oz_per_lot, "price_offset": 0.0, "risk_pct": ins.default_risk_pct,
                "min_lot": 0.01, "confirmed": False, "squeeze_filter": ins.squeeze_default} for k, ins in INSTRUMENTS.items()}


def _to_bool(v) -> bool:
    """JSON true/false from the UI, but also "false"/"0"/"off" from a hand-edited file."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


@dataclass
class Settings:
    account_balance: float = 2000.0      # USD
    day_mode: bool = True                # 當日平倉模式: 16:45 NY check + no late entries
    max_lots: float = 0.3                # cap on one position (lots)
    leverage: float = 100.0              # broker leverage (only affects margin)
    color_convention: str = "tw"         # "tw" = 紅漲綠跌 (Taiwan), "intl" = 綠漲紅跌
    sound: bool = True
    desktop_notify: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    blackout_before_min: int = 30
    blackout_after_min: int = 30
    fomc_after_min: int = 60
    show_medium_impact: bool = False
    # per instrument: oz_per_lot, price_offset, risk_pct, min_lot, confirmed, squeeze_filter (布林收窄濾網)
    instruments: dict = field(default_factory=_default_instruments)
    # optional real positions, per instrument: {"dir": 1/-1, "entry": float, "time": iso, "lots": float}
    my_positions: dict = field(default_factory=dict)

    SECRET_KEYS = ("telegram_bot_token", "discord_webhook_url")

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        p = Path(path)
        s = cls()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                known = {f.name for f in fields(cls)}
                for k, v in raw.items():
                    if k in known:
                        setattr(s, k, v)
                # migrate the single-instrument settings of the first version
                gold = s.instruments.setdefault("XAUUSD", {})
                for key in ("oz_per_lot", "price_offset", "risk_pct", "min_lot"):
                    if key in raw and key not in raw.get("instruments", {}).get("XAUUSD", {}):
                        gold[key] = raw[key]
                if raw.get("my_position") and "XAUUSD" not in s.my_positions:
                    s.my_positions["XAUUSD"] = raw["my_position"]
            except Exception:
                pass
        defaults = _default_instruments()
        for k, d in defaults.items():
            cur = s.instruments.setdefault(k, {})
            for kk, vv in d.items():
                cur.setdefault(kk, vv)
        s.save(p)
        return s

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def update(self, data: dict) -> None:
        for f in fields(self):
            if f.name not in data or f.name == "my_positions":
                continue
            v = data[f.name]
            if f.name == "instruments":
                if not isinstance(v, dict):
                    continue
                for key, vals in v.items():
                    if key not in INSTRUMENTS or not isinstance(vals, dict):
                        continue
                    tgt = self.instruments.setdefault(key, {})
                    for kk, vv in vals.items():
                        try:
                            if kk in ("confirmed", "squeeze_filter"):
                                tgt[kk] = _to_bool(vv)
                            elif kk in ("oz_per_lot", "price_offset", "risk_pct", "min_lot"):
                                tgt[kk] = float(vv)
                        except (TypeError, ValueError):
                            pass
                continue
            cur = getattr(self, f.name)
            try:
                if isinstance(cur, bool):
                    v = _to_bool(v)
                elif isinstance(cur, int):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
                elif isinstance(cur, str):
                    v = str(v)
                else:
                    continue
            except (TypeError, ValueError):
                continue
            setattr(self, f.name, v)

    def inst(self, key: str) -> dict:
        return self.instruments.get(key) or _default_instruments()[key]

    def public(self) -> dict:
        d = asdict(self)
        for k in self.SECRET_KEYS:
            if d.get(k):
                d[k] = "********"
        return d
