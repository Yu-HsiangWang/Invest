"""Watch-lists: Taiwan tech (stocks + ETFs) and US tech (stocks + ETFs).

The user can add/remove tickers in settings; these are the defaults. Yahoo Finance symbols:
Taiwan listed = <code>.TW, OTC (櫃買) = <code>.TWO.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    symbol: str          # internal id (Yahoo symbol)
    name: str            # Chinese display name
    market: str          # "TW" or "US"
    kind: str            # "stock" or "etf"
    sector: str = ""
    code: str = ""       # Taiwan stock code (for TWSE APIs), or US ticker
    benchmark: str = ""  # relative-strength benchmark
    tags: tuple = field(default_factory=tuple)

    @property
    def is_tw(self) -> bool:
        return self.market == "TW"


def _tw(code: str, name: str, sector: str, kind: str = "stock", otc: bool = False, bench: str = "0050.TW", tags=()) -> Instrument:
    sym = f"{code}.TWO" if otc else f"{code}.TW"
    return Instrument(symbol=sym, name=name, market="TW", kind=kind, sector=sector, code=code, benchmark=bench, tags=tuple(tags))


def _us(ticker: str, name: str, sector: str, kind: str = "stock", bench: str = "QQQ", tags=()) -> Instrument:
    return Instrument(symbol=ticker, name=name, market="US", kind=kind, sector=sector, code=ticker, benchmark=bench, tags=tuple(tags))


TW_DEFAULT: list[Instrument] = [
    # ETFs
    _tw("0050", "元大台灣50", "大盤", "etf", bench="^TWII", tags=("市值型",)),
    _tw("006208", "富邦台50", "大盤", "etf", bench="^TWII", tags=("市值型",)),
    _tw("0056", "元大高股息", "高股息", "etf", bench="^TWII", tags=("高股息",)),
    _tw("00878", "國泰永續高股息", "高股息", "etf", bench="^TWII", tags=("高股息",)),
    _tw("00919", "群益台灣精選高息", "高股息", "etf", bench="^TWII", tags=("高股息",)),
    _tw("00881", "國泰台灣5G+", "科技", "etf", tags=("科技",)),
    _tw("00891", "中信關鍵半導體", "半導體", "etf", tags=("半導體",)),
    _tw("00830", "國泰費城半導體", "美國半導體", "etf", bench="SOXX", tags=("海外",)),
    _tw("00662", "富邦NASDAQ", "美國科技", "etf", bench="QQQ", tags=("海外",)),
    _tw("00757", "統一FANG+", "美國科技", "etf", bench="QQQ", tags=("海外",)),
    # Semiconductors
    _tw("2330", "台積電", "半導體"),
    _tw("2454", "聯發科", "半導體"),
    _tw("2303", "聯電", "半導體"),
    _tw("3711", "日月光投控", "半導體"),
    _tw("3034", "聯詠", "半導體"),
    _tw("2379", "瑞昱", "半導體"),
    _tw("3443", "創意", "半導體"),
    _tw("3661", "世芯-KY", "半導體"),
    _tw("2408", "南亞科", "半導體"),
    _tw("3529", "力旺", "半導體", otc=True),
    _tw("5274", "信驊", "半導體", otc=True),
    # AI servers / hardware
    _tw("2317", "鴻海", "電子組裝"),
    _tw("2382", "廣達", "AI伺服器"),
    _tw("3231", "緯創", "AI伺服器"),
    _tw("6669", "緯穎", "AI伺服器"),
    _tw("2356", "英業達", "AI伺服器"),
    _tw("2308", "台達電", "電源／散熱"),
    _tw("3017", "奇鋐", "散熱"),
    _tw("2383", "台光電", "PCB／CCL"),
    _tw("3037", "欣興", "PCB"),
    _tw("2345", "智邦", "網通"),
    _tw("2357", "華碩", "PC"),
    _tw("2376", "技嘉", "PC／伺服器"),
    _tw("2395", "研華", "工業電腦"),
    _tw("2301", "光寶科", "電源"),
    _tw("2412", "中華電", "電信", bench="^TWII"),
]

US_DEFAULT: list[Instrument] = [
    # ETFs
    _us("QQQ", "Nasdaq 100", "科技", "etf", bench="SPY"),
    _us("SMH", "半導體 ETF", "半導體", "etf"),
    _us("SOXX", "費城半導體 ETF", "半導體", "etf"),
    _us("XLK", "SPDR 科技", "科技", "etf"),
    _us("VGT", "Vanguard 科技", "科技", "etf"),
    _us("SPY", "S&P 500", "大盤", "etf", bench="SPY"),
    # Mega caps
    _us("AAPL", "Apple", "消費電子"),
    _us("MSFT", "Microsoft", "軟體／雲"),
    _us("NVDA", "NVIDIA", "半導體"),
    _us("GOOGL", "Alphabet", "網路／雲"),
    _us("AMZN", "Amazon", "電商／雲"),
    _us("META", "Meta", "網路"),
    _us("TSLA", "Tesla", "電動車／AI"),
    _us("AVGO", "Broadcom", "半導體"),
    _us("TSM", "台積電 ADR", "半導體"),
    _us("ORCL", "Oracle", "軟體／雲"),
    _us("NFLX", "Netflix", "串流"),
    # Semis & equipment
    _us("AMD", "AMD", "半導體"),
    _us("ASML", "ASML", "半導體設備"),
    _us("AMAT", "Applied Materials", "半導體設備"),
    _us("LRCX", "Lam Research", "半導體設備"),
    _us("KLAC", "KLA", "半導體設備"),
    _us("QCOM", "Qualcomm", "半導體"),
    _us("MU", "Micron", "記憶體"),
    _us("INTC", "Intel", "半導體"),
    _us("TXN", "Texas Instruments", "半導體"),
    _us("ARM", "Arm", "半導體 IP"),
    _us("MRVL", "Marvell", "半導體"),
    # Software / AI
    _us("CRM", "Salesforce", "軟體"),
    _us("ADBE", "Adobe", "軟體"),
    _us("NOW", "ServiceNow", "軟體"),
    _us("PLTR", "Palantir", "軟體／AI"),
    _us("SNOW", "Snowflake", "軟體／資料"),
    _us("CRWD", "CrowdStrike", "資安"),
    _us("PANW", "Palo Alto Networks", "資安"),
    _us("UBER", "Uber", "平台"),
    _us("SHOP", "Shopify", "電商軟體"),
]

DEFAULT_UNIVERSE: dict[str, Instrument] = {i.symbol: i for i in TW_DEFAULT + US_DEFAULT}


def by_market(market: str, universe: dict[str, Instrument] | None = None) -> list[Instrument]:
    u = universe or DEFAULT_UNIVERSE
    return [i for i in u.values() if i.market == market]
