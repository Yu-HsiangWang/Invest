"""Free market data feeds (no API key needed).

Primary : Dukascopy public JSON data API (the same backend the dukascopy-node
          library uses). Spot XAU/USD bid candles, 1-minute resolution, updated
          every few seconds, history back to 2003.
Quotes  : Swissquote public bid/ask quotes (real-time snapshot, used for the
          live price/spread display; falls back to the last Dukascopy candle).

Completed days/months are cached on disk, so after the first start only the
current day is downloaded again.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .bars import resample

log = logging.getLogger("goldsignal.datafeed")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

GOLD = "XAU-USD"
SILVER = "XAG-USD"


def decode_candles(js: dict) -> pd.DataFrame:
    """Decode Dukascopy's delta-encoded candle JSON into an OHLCV DataFrame (UTC index)."""
    times = js.get("times") or []
    if not times:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    mult = float(js["multiplier"])
    shift = int(js["shift"])
    t = js["timestamp"] + np.cumsum(np.asarray(times, dtype=np.int64)) * shift
    base = {k: round(float(js[k]) / mult) for k in ("open", "high", "low", "close")}
    cols = {}
    for k, arr in (("open", "opens"), ("high", "highs"), ("low", "lows"), ("close", "closes")):
        units = base[k] + np.cumsum(np.asarray(js[arr], dtype=np.int64))
        cols[k] = np.round(units * mult, 6)
    cols["volume"] = np.asarray(js.get("volumes", [0] * len(times)), dtype=float)
    idx = pd.to_datetime(t, unit="ms", utc=True)
    df = pd.DataFrame(cols, index=idx)
    df.index.name = "time"
    return df[~df.index.duplicated(keep="last")]


class Dukascopy:
    BASE = "https://jetta.dukascopy.com/v1"

    def __init__(self, cache_dir: str | Path, timeout: float = 20.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json"})

    # ------------------------------------------------------------------ http
    def _get_json(self, url: str, retries: int = 3) -> dict:
        last = None
        for attempt in range(retries):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 404):
                    break
            except Exception as e:  # network hiccup
                last = repr(e)
            time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"Dukascopy request failed: {url} -> {last}")

    def _cache_path(self, *parts: str) -> Path:
        p = self.cache_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _cached_json(self, path: Path, url: str) -> dict:
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                path.unlink(missing_ok=True)
        js = self._get_json(url)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(js, fh)
        return js

    def check(self, code: str = "XAU-USD") -> None:
        """Fail fast when the data API is unreachable (one quick request)."""
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        self._get_json(f"{self.BASE}/candles/minute/{code}/BID?from={int(start.timestamp() * 1000)}", retries=1)

    # --------------------------------------------------------------- minutes
    def minute_day(self, code: str, day: date, side: str = "BID") -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        if now < start:
            return decode_candles({})
        if now < end + timedelta(minutes=2):  # still active
            url = f"{self.BASE}/candles/minute/{code}/{side}?from={int(start.timestamp() * 1000)}"
            return decode_candles(self._get_json(url))
        url = f"{self.BASE}/candles/minute/{code}/{side}/{day.year}/{day.month}/{day.day}"
        path = self._cache_path(code, side, "m1", f"{day.isoformat()}.json.gz")
        try:
            return decode_candles(self._cached_json(path, url))
        except RuntimeError:
            if now - end > timedelta(hours=3):
                raise
            # just after midnight the completed-day file may not exist yet
            url = f"{self.BASE}/candles/minute/{code}/{side}?from={int(start.timestamp() * 1000)}"
            return decode_candles(self._get_json(url))

    def minutes(self, code: str, start: date, end: date | None = None, side: str = "BID",
                progress=None) -> pd.DataFrame:
        end = end or datetime.now(timezone.utc).date()
        frames = []
        d = start
        total = (end - start).days + 1
        k = 0
        while d <= end:
            if d.weekday() != 5:  # gold does not trade on Saturdays (UTC)
                try:
                    frames.append(self.minute_day(code, d, side))
                except RuntimeError as e:
                    log.warning("skip %s %s: %s", code, d, e)
            k += 1
            if progress:
                progress(k, total)
            d += timedelta(days=1)
        frames = [f for f in frames if not f.empty]
        if not frames:
            return decode_candles({})
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")]

    # ----------------------------------------------------------------- hours
    def hour_month(self, code: str, year: int, month: int, side: str = "BID") -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
        if now < end + timedelta(minutes=2):
            url = f"{self.BASE}/candles/hour/{code}/{side}?from={int(start.timestamp() * 1000)}"
            return decode_candles(self._get_json(url))
        url = f"{self.BASE}/candles/hour/{code}/{side}/{year}/{month}"
        path = self._cache_path(code, side, "h1", f"{year:04d}-{month:02d}.json.gz")
        return decode_candles(self._cached_json(path, url))

    def hours(self, code: str, months: int = 12, side: str = "BID") -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        y, m = now.year, now.month
        frames = []
        for _ in range(months):
            try:
                frames.append(self.hour_month(code, y, m, side))
            except RuntimeError as e:
                log.warning("skip hours %s %04d-%02d: %s", code, y, m, e)
            y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        frames = [f for f in frames if not f.empty]
        if not frames:
            return decode_candles({})
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")]


def to_m15(m1: pd.DataFrame, drop_incomplete: bool = True, now: datetime | None = None) -> pd.DataFrame:
    """1-minute -> 15-minute bars. The bar in progress is dropped unless drop_incomplete=False."""
    if m1.empty:
        return m1.copy()
    out = resample(m1, "15min")
    if drop_incomplete:
        now = now or datetime.now(timezone.utc)
        out = out[out.index + pd.Timedelta(minutes=15) <= pd.Timestamp(now)]
    return out


class SwissquoteQuote:
    BASE = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/"

    def __init__(self, symbol: str = "XAU/USD", timeout: float = 8.0):
        self.URL = self.BASE + symbol
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.timeout = timeout

    def get(self) -> dict | None:
        """Returns {'bid','ask','spread','ts'} or None."""
        try:
            r = self.s.get(self.URL, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            best = None
            for platform in data:
                for prof in platform.get("spreadProfilePrices", []):
                    bid, ask = float(prof["bid"]), float(prof["ask"])
                    if best is None or (ask - bid) < (best["ask"] - best["bid"]):
                        best = {"bid": bid, "ask": ask, "ts": int(platform.get("ts", 0))}
            if best:
                best["spread"] = round(best["ask"] - best["bid"], 3)
                best["source"] = "Swissquote"
            return best
        except Exception as e:
            log.debug("swissquote quote failed: %s", e)
            return None
