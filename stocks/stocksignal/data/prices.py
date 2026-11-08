"""Daily OHLCV (dividend/split adjusted) from Yahoo Finance's chart API, cached as CSV.

Yahoo symbols: Taiwan listed "2330.TW", OTC "5274.TWO", TAIEX "^TWII"; US tickers as is.
The chart endpoint needs no API key; we send a browser User-Agent and fall back to the
`yfinance` package (if installed) and then to Stooq's daily CSV when Yahoo is unavailable.

Adjusted prices: Yahoo's `adjclose` is dividend- and split-adjusted. We scale O/H/L/C by
adjclose/close so the whole bar is adjusted consistently (total-return series), and keep the
raw close in `close_raw` for display and for P/E bands.
"""
from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("stocksignal.prices")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
COLS = ["open", "high", "low", "close", "volume", "close_raw"]


class PriceStore:
    def __init__(self, cache_dir: str | Path, max_age_hours: float = 6.0):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json,text/plain,*/*"})
        self.status: dict[str, str] = {}

    # ------------------------------------------------------------------ public
    def daily(self, symbol: str, years: int = 25, refresh: bool = True) -> pd.DataFrame:
        """Adjusted daily bars (index = date, tz-naive). Empty frame if nothing is available."""
        p = self.dir / f"{_safe(symbol)}.csv"
        cached = self._read(p)
        fresh = cached is not None and (datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)) < self.max_age
        if cached is not None and (fresh or not refresh):
            return cached
        df = None
        for fetch in (self._yahoo, self._yfinance, self._stooq):
            try:
                df = fetch(symbol, years)
            except Exception as e:                 # noqa: BLE001 - every source may fail
                self.status[symbol] = f"{fetch.__name__}: {type(e).__name__}: {e}"[:160]
                df = None
            if df is not None and len(df) > 50:
                self.status[symbol] = f"ok ({fetch.__name__[1:]}, {df.index[-1].date()})"
                break
        if df is None or len(df) <= 50:
            if cached is not None:
                p.touch()
                return cached
            return pd.DataFrame(columns=COLS)
        if cached is not None and len(cached) > len(df):
            # a short fetch (e.g. Stooq limits) must not shrink the history
            df = pd.concat([cached[cached.index < df.index[0]], df])
        df.to_csv(p)
        return df

    # ----------------------------------------------------------------- sources
    def _yahoo(self, symbol: str, years: int) -> pd.DataFrame | None:
        end = int(time.time())
        start = end - int(years * 365.25 * 86400)
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = self.s.get(url, params={"period1": start, "period2": end, "interval": "1d", "events": "div,splits",
                                    "includeAdjustedClose": "true"}, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        js = r.json()
        res = (js.get("chart") or {}).get("result") or []
        if not res:
            return None
        return parse_yahoo_chart(res[0])

    def _yfinance(self, symbol: str, years: int) -> pd.DataFrame | None:
        try:
            import yfinance as yf  # optional dependency
        except ImportError:
            return None
        h = yf.Ticker(symbol).history(period=f"{years}y", interval="1d", auto_adjust=False, actions=False)
        if h is None or h.empty:
            return None
        h = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
                              "Adj Close": "adjclose"})
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return _adjust(h)

    def _stooq(self, symbol: str, years: int) -> pd.DataFrame | None:
        code = _stooq_symbol(symbol)
        if not code:
            return None
        r = self.s.get(f"https://stooq.com/q/d/l/?s={code}&i=d", timeout=20)
        r.raise_for_status()
        if "Date,Open" not in r.text[:100]:
            return None
        h = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
        h = h.rename(columns=str.lower)
        h["close_raw"] = h["close"]            # Stooq data is split-adjusted but not dividend-adjusted
        h["volume"] = h.get("volume", 0)
        return h[COLS].dropna(subset=["close"])

    # ------------------------------------------------------------------- utils
    @staticmethod
    def _read(p: Path) -> pd.DataFrame | None:
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df[COLS] if len(df) else None
        except Exception:  # noqa: BLE001
            return None


def parse_yahoo_chart(res: dict) -> pd.DataFrame | None:
    ts = res.get("timestamp")
    q = (res.get("indicators") or {}).get("quote") or [{}]
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not ts or not q[0]:
        return None
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    h = pd.DataFrame({"open": q[0].get("open"), "high": q[0].get("high"), "low": q[0].get("low"),
                      "close": q[0].get("close"), "volume": q[0].get("volume")}, index=idx)
    h["adjclose"] = adj if adj is not None else h["close"]
    h = h[~h.index.duplicated(keep="last")]
    return _adjust(h)


def _adjust(h: pd.DataFrame) -> pd.DataFrame:
    h = h.dropna(subset=["close"]).copy()
    h["close_raw"] = h["close"]
    factor = (h["adjclose"] / h["close"]).replace([float("inf")], 1.0).fillna(1.0)
    for c in ("open", "high", "low", "close"):
        h[c] = h[c] * factor
    h["volume"] = h["volume"].fillna(0)
    h.index.name = "date"
    return h[COLS].sort_index()


def _stooq_symbol(symbol: str) -> str | None:
    if symbol.startswith("^"):
        return None
    if symbol.endswith(".TW") or symbol.endswith(".TWO"):
        return None                                   # Stooq has little Taiwan coverage
    return f"{symbol.lower()}.us"


def _safe(symbol: str) -> str:
    return symbol.replace("^", "_idx_").replace("/", "_")
