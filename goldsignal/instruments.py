"""Tradable instruments. Both metals use exactly the same rules (StrategyParams
defaults); only costs, display precision and contract defaults differ."""
from __future__ import annotations

from dataclasses import dataclass, field

from .strategy import StrategyParams


@dataclass(frozen=True)
class Instrument:
    key: str                  # internal id, e.g. "XAUUSD"
    code: str                 # Dukascopy instrument code
    name: str                 # 中文名稱
    symbol: str               # display symbol
    seal: str                 # one-character mark used in the UI
    metal: str                # "gold" / "silver" (news relevance)
    decimals: int
    spread_pct: float         # assumed broker spread (fraction of price)
    slip_pct: float           # assumed slippage per market fill
    default_oz_per_lot: float
    default_risk_pct: float
    swissquote: str           # Swissquote public quote path
    intermarket: dict = field(default_factory=dict)
    note: str = ""
    params: StrategyParams = field(default_factory=StrategyParams)


INSTRUMENTS: dict[str, Instrument] = {
    "XAUUSD": Instrument(
        key="XAUUSD", code="XAU-USD", name="黃金", symbol="XAU/USD", seal="金", metal="gold", decimals=2,
        spread_pct=0.00012, slip_pct=0.00005, default_oz_per_lot=100.0, default_risk_pct=1.0, swissquote="XAU/USD",
        intermarket={"DOLLAR.IDX-USD": "美元指數 DXY", "XAG-USD": "白銀 XAG", "USA500.IDX-USD": "標普500", "USTBOND.TR-USD": "美國長債"},
    ),
    "XAGUSD": Instrument(
        key="XAGUSD", code="XAG-USD", name="白銀", symbol="XAG/USD", seal="銀", metal="silver", decimals=3,
        spread_pct=0.0006, slip_pct=0.0001, default_oz_per_lot=5000.0, default_risk_pct=0.5, swissquote="XAG/USD",
        intermarket={"DOLLAR.IDX-USD": "美元指數 DXY", "XAU-USD": "黃金 XAU", "COPPER.CMD-USD": "銅", "USA500.IDX-USD": "標普500"},
        note="白銀點差約為黃金的 4 倍、波動更劇烈，勝率約三成；建議每筆風險減半（約 0.5%）。",
    ),
}
