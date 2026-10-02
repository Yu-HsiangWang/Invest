"""Tradable instruments. Both metals use the same entry rules (StrategyParams defaults) except
the 布林收窄濾網, which only gold uses; costs, display precision, contract defaults and the
day-mode settings differ."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .strategy import SQUEEZE_BARS, StrategyParams


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
    # 當日平倉模式 (docs/RESEARCH.md section 8): last NY hour for new entries, and the profit (in R)
    # a trade needs at the 16:45 NY check to be kept overnight
    day_cutoff_ny: float = 12.0
    eod_keep_r: float = 0.25
    # 布林收窄濾網 (docs/RESEARCH.md section 10) on by default? Gold yes (same total profit with a much
    # smaller drawdown 2005-2026); silver's results were mixed, so it keeps the plain rules.
    squeeze_default: bool = False

    def strategy_params(self, day_mode: bool, squeeze: bool | None = None) -> StrategyParams:
        """The rules the app trades. squeeze=None -> the instrument default; True/False forces the filter."""
        on = self.squeeze_default if squeeze is None else bool(squeeze)
        p = replace(self.params, squeeze_bars=SQUEEZE_BARS if on else 0)
        if not day_mode:
            return p
        return replace(p, day_cutoff_ny=self.day_cutoff_ny, eod_keep_r=self.eod_keep_r)


INSTRUMENTS: dict[str, Instrument] = {
    "XAUUSD": Instrument(
        key="XAUUSD", code="XAU-USD", name="黃金", symbol="XAU/USD", seal="金", metal="gold", decimals=2,
        spread_pct=0.00012, slip_pct=0.00005, default_oz_per_lot=100.0, default_risk_pct=2.0, swissquote="XAU/USD",
        intermarket={"DOLLAR.IDX-USD": "美元指數 DXY", "XAG-USD": "白銀 XAG", "USA500.IDX-USD": "標普500", "USTBOND.TR-USD": "美國長債"},
        day_cutoff_ny=12.0, eod_keep_r=0.25, squeeze_default=True,
    ),
    "XAGUSD": Instrument(
        key="XAGUSD", code="XAG-USD", name="白銀", symbol="XAG/USD", seal="銀", metal="silver", decimals=3,
        spread_pct=0.0006, slip_pct=0.0001, default_oz_per_lot=5000.0, default_risk_pct=1.0, swissquote="XAG/USD",
        intermarket={"DOLLAR.IDX-USD": "美元指數 DXY", "XAU-USD": "黃金 XAU", "COPPER.CMD-USD": "銅", "USA500.IDX-USD": "標普500"},
        note="白銀 0.01 手 = 50 盎司，價格動 $1 就賺賠 $50；點差約黃金的 4 倍、勝率約三成。本金 $5,000 以下建議先只做黃金。",
        day_cutoff_ny=13.75, eod_keep_r=0.0,
    ),
}
