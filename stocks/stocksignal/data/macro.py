"""Macro context: FRED series (no key, CSV), FOMC dates, earnings dates (yfinance), Taiwan revenue deadline."""
from __future__ import annotations

import io
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("stocksignal.macro")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
FRED_SERIES = {"DGS10": "美國 10 年期公債殖利率", "DGS2": "美國 2 年期公債殖利率", "T10Y2Y": "10 年 − 2 年利差",
               "FEDFUNDS": "聯邦資金利率", "CPIAUCSL": "美國 CPI", "UNRATE": "美國失業率", "VIXCLS": "VIX", "DTWEXBGS": "美元指數（廣義）"}


class Macro:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.status: dict[str, str] = {}

    def fred(self, series: str, max_age_h: float = 12.0) -> pd.Series:
        cache = self.dir / f"fred_{series}.csv"
        if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
            return self._read(cache)
        try:
            r = self.s.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=20)
            r.raise_for_status()
            cache.write_text(r.text, encoding="utf-8")
            self.status[series] = "ok"
        except Exception as e:  # noqa: BLE001
            self.status[series] = f"error: {type(e).__name__}"
        return self._read(cache) if cache.exists() else pd.Series(dtype=float)

    @staticmethod
    def _read(p: Path) -> pd.Series:
        try:
            df = pd.read_csv(p)
            dcol = "observation_date" if "observation_date" in df.columns else df.columns[0]
            s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df[dcol]))
            return s.dropna()
        except Exception:  # noqa: BLE001
            return pd.Series(dtype=float)

    def snapshot(self) -> list[dict]:
        out = []
        for sid, name in FRED_SERIES.items():
            s = self.fred(sid)
            if s.empty:
                continue
            last = s.iloc[-1]
            prev_m = s[s.index <= s.index[-1] - pd.Timedelta(days=30)]
            chg = float(last - prev_m.iloc[-1]) if len(prev_m) else None
            out.append({"id": sid, "name": name, "value": float(last), "date": s.index[-1].strftime("%Y-%m-%d"), "chg_1m": chg})
        return out

    def fomc_dates(self, max_age_h: float = 24 * 7) -> list[date]:
        """Scheduled FOMC meetings (second day) from the Fed's calendar page; cached a week."""
        cache = self.dir / "fomc.txt"
        if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
            return [date.fromisoformat(x) for x in cache.read_text().split() if x]
        try:
            html = self.s.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", timeout=20).text
            dates = parse_fomc(html)
            cache.write_text("\n".join(d.isoformat() for d in dates))
            self.status["fomc"] = f"ok ({len(dates)})"
            return dates
        except Exception as e:  # noqa: BLE001
            self.status["fomc"] = f"error: {type(e).__name__}"
            return [date.fromisoformat(x) for x in cache.read_text().split()] if cache.exists() else []


def parse_fomc(html: str) -> list[date]:
    """The calendar page lists panels per year with month names and day ranges like '28-29' or '16-17*'."""
    out = []
    for ym in re.finditer(r'<div class="panel panel-default">\s*<div class="panel-heading">\s*<h4>\s*(\d{4}) FOMC Meetings', html):
        year = int(ym.group(1))
        block = html[ym.end(): ym.end() + 20000]
        nxt = block.find("FOMC Meetings</h4>")
        if nxt > 0:
            block = block[:nxt]
        for m in re.finditer(r'<div class="fomc-meeting__month[^"]*">\s*<strong>([A-Za-z/]+)</strong>\s*</div>\s*<div class="fomc-meeting__date[^"]*">\s*([0-9]+)(?:-([0-9]+))?', block):
            month_txt = m.group(1).split("/")[-1][:3]
            try:
                month = datetime.strptime(month_txt, "%b").month
            except ValueError:
                continue
            day = int(m.group(3) or m.group(2))
            try:
                out.append(date(year, month, day))
            except ValueError:
                continue
    return sorted(set(out))


def tw_revenue_deadline(today: date | None = None) -> date:
    """Taiwan listed companies publish last month's revenue by the 10th (weekend -> next weekday)."""
    today = today or (datetime.now(timezone.utc) + timedelta(hours=8)).date()
    d = date(today.year, today.month, 10)
    if today > d:
        y, m = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        d = date(y, m, 10)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def earnings_dates(symbol: str) -> list[str]:
    """Next earnings dates via yfinance (optional dependency); [] when unavailable."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = t.get_earnings_dates(limit=8)
        if df is None or df.empty:
            return []
        now = pd.Timestamp.now(tz=df.index.tz) if getattr(df.index, "tz", None) is not None else pd.Timestamp.now()
        return [d.strftime("%Y-%m-%d") for d in df.index if d >= now - pd.Timedelta(days=1)][:3]
    except Exception:  # noqa: BLE001
        return []
