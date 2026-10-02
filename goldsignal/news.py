"""News headlines + economic calendar for gold.

How news is used by the signal engine (deliberately conservative):
  * High-impact US events (CPI, NFP, FOMC, ...) -> no NEW signals from 30 min before
    to 30 min after (60 min after FOMC). Research: gold reprices within ~1-5 minutes of
    US releases and there is no reliable post-release drift, so trading into them only
    adds slippage/whipsaw risk.
  * Headline sentiment is shown as context and used to WARN when news flow is strongly
    against an open position. It never creates a trade by itself (headline sentiment was
    found to be mostly contemporaneous, i.e. not predictive, in the literature).
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("goldsignal.news")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

FEEDS = [
    {"name": "FXStreet", "url": "https://www.fxstreet.com/rss/news"},
    {"name": "investingLive", "url": "https://investinglive.com/feed/"},
    {"name": "Google News", "url": "https://news.google.com/rss/search?q=gold+price+OR+XAUUSD+OR+bullion+when:1d&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Google News", "url": "https://news.google.com/rss/search?q=%22silver+price%22+OR+XAGUSD+when:1d&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Google News", "url": "https://news.google.com/rss/search?q=%22Federal+Reserve%22+OR+%22Treasury+yields%22+OR+%22dollar+index%22+when:1d&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Yahoo Finance", "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US"},
]

CALENDAR_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

# ------------------------------------------------------------------ relevance
METAL_WORDS = {
    "gold": r"\b(gold|xau|bullion|precious metals?|xauusd|spot gold|comex)\b",
    "silver": r"\b(silver|xag|xagusd|spot silver|precious metals?)\b",
}
METAL_NOUN = {"gold": r"(gold|xau/?usd|bullion)", "silver": r"(silver|xag/?usd)"}
MACRO_WORDS = (r"\b(fed|fomc|powell|federal reserve|rate cut|rate hike|interest rates?|yields?|treasur(y|ies)|"
               r"dollar|dxy|greenback|inflation|cpi|pce|ppi|payrolls?|nfp|jobs report|jobless|unemployment|gdp|"
               r"recession|tariffs?|sanctions?|war|missile|attack|ceasefire|geopolitic\w*|middle east|iran|israel|"
               r"russia|ukraine|china|taiwan|central banks?|ecb|boj|pboc|safe[- ]haven|risk[- ]off|risk[- ]on)\b")

# ------------------------------------------------------------------ sentiment for GOLD (+ bullish / - bearish)
LEXICON: list[tuple[str, float]] = [
    # direct price action of the metal itself ({M} is replaced by the metal noun)
    (r"{M}\w*\s+(\w+\s+){0,3}(rises?|rising|gains?|climbs?|jumps?|surges?|soars?|rall(y|ies)|rebounds?|advances?|extends? gains|edges? (up|higher)|firms?)", 1.0),
    (r"{M}\w*\s+(\w+\s+){0,3}(falls?|falling|drops?|slips?|slides?|declines?|tumbles?|plunges?|sinks?|retreats?|eases?|edges? (down|lower)|loses?)", -1.0),
    (r"(record|all[- ]time) high", 0.8),
    (r"(profit[- ]taking|sell[- ]off|selloff)", -0.6),
    # monetary policy / rates
    (r"rate cuts?|cuts? (interest )?rates|dovish|easing", 0.8),
    (r"rate hikes?|hikes? (interest )?rates|hawkish|tightening", -0.8),
    (r"yields? (fall|falls|drop|drops|slip|slips|decline|declines|ease|eases|retreat)", 0.7),
    (r"yields? (rise|rises|jump|jumps|climb|climbs|surge|surges|spike|spikes|soar)", -0.7),
    # dollar
    (r"(dollar|greenback|dxy)\w*\s+(\w+\s+){0,2}(falls?|drops?|slips?|weakens?|slides?|declines?|tumbles?|retreats?)", 0.7),
    (r"(dollar|greenback|dxy)\w*\s+(\w+\s+){0,2}(rises?|gains?|strengthens?|jumps?|climbs?|rall(y|ies)|firms?|surges?)", -0.7),
    (r"weak(er)? dollar", 0.6), (r"strong(er)? dollar", -0.6),
    # risk / geopolitics
    (r"safe[- ]haven", 0.6), (r"risk[- ]off", 0.5), (r"risk[- ]on", -0.4),
    (r"\b(war|missile|attack|escalat\w+|invasion|conflict|tensions?)\b", 0.5),
    (r"\b(ceasefire|peace (deal|talks)|truce|de-?escalat\w+)\b", -0.5),
    (r"central banks? (buy|buying|purchases?|add)", 0.6),
    (r"etf (inflows?)", 0.5), (r"etf (outflows?)", -0.5),
    (r"recession|slowdown|crisis|turmoil|uncertainty", 0.4),
    # data surprises (generic)
    (r"(payrolls?|jobs|employment)\w*\s+(\w+\s+){0,3}(miss|misses|disappoints?|weaker|slows?|falls?)", 0.6),
    (r"(payrolls?|jobs|employment)\w*\s+(\w+\s+){0,3}(beat|beats|stronger|surges?|jumps?|tops?)", -0.6),
    (r"(inflation|cpi|pce)\w*\s+(\w+\s+){0,3}(hotter|accelerat\w+|jumps?|surges?|above (expectations|forecast))", -0.4),
    (r"(inflation|cpi|pce)\w*\s+(\w+\s+){0,3}(cooler|cools|slows?|eases?|below (expectations|forecast))", 0.5),
]
_LEX = {m: [(re.compile(p.replace("{M}", noun), re.I), w) for p, w in LEXICON] for m, noun in METAL_NOUN.items()}
_METAL = {m: re.compile(rx, re.I) for m, rx in METAL_WORDS.items()}
_MACRO = re.compile(MACRO_WORDS, re.I)
_TAG = re.compile(r"<[^>]+>")


def score_headline(text: str, metal: str = "gold") -> tuple[float, float, list[str]]:
    """Return (sentiment for the metal in [-1,1], relevance 0..1, matched cues)."""
    t = text or ""
    rel = 1.0 if _METAL[metal].search(t) else (0.5 if _MACRO.search(t) else 0.0)
    total, cues = 0.0, []
    for rx, w in _LEX[metal]:
        m = rx.search(t)
        if m:
            total += w
            cues.append(m.group(0)[:40])
    return float(np.tanh(total)), rel, cues


def _clean(s: str | None) -> str:
    return html.unescape(_TAG.sub("", s or "")).strip()


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)
    except Exception:
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def parse_feed(xml_text: str, source: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    ns_atom = "{http://www.w3.org/2005/Atom}"
    for it in root.iter("item"):
        title = _clean(it.findtext("title"))
        link = (it.findtext("link") or "").strip()
        when = _parse_date(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"))
        src = it.find("source")
        items.append({"title": title, "link": link, "time": when, "source": (src.text.strip() if src is not None and src.text else source)})
    for it in root.iter(f"{ns_atom}entry"):
        title = _clean(it.findtext(f"{ns_atom}title"))
        link_el = it.find(f"{ns_atom}link")
        link = link_el.get("href") if link_el is not None else ""
        when = _parse_date(it.findtext(f"{ns_atom}updated") or it.findtext(f"{ns_atom}published"))
        items.append({"title": title, "link": link, "time": when, "source": source})
    return items


class NewsMonitor:
    def __init__(self, cache_dir: str | Path, feeds: list[dict] | None = None, max_age_h: float = 24.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feeds = feeds or FEEDS
        self.max_age_h = max_age_h
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"})
        self.items: list[dict] = []
        self.feed_status: dict[str, str] = {}
        self.calendar: list[dict] = []
        self.calendar_fetched: float = 0.0

    # -------------------------------------------------------------- headlines
    def refresh_news(self) -> list[dict]:
        seen, out = set(), []
        now = datetime.now(timezone.utc)
        for fd in self.feeds:
            try:
                r = self.s.get(fd["url"], timeout=12)
                r.raise_for_status()
                items = parse_feed(r.text, fd["name"])
                self.feed_status[fd["url"]] = f"ok ({len(items)})"
            except Exception as e:
                self.feed_status[fd["url"]] = f"error: {type(e).__name__}"
                continue
            for it in items:
                key = re.sub(r"\W+", "", it["title"].lower())[:90]
                if not it["title"] or key in seen:
                    continue
                if it["time"] and (now - it["time"]) > timedelta(hours=self.max_age_h):
                    continue
                seen.add(key)
                scores = {}
                for metal in METAL_WORDS:
                    sent, rel, cues = score_headline(it["title"], metal)
                    scores[metal] = {"sentiment": round(sent, 2), "relevance": rel, "cues": cues}
                if max(v["relevance"] for v in scores.values()) <= 0:
                    continue
                it["scores"] = scores
                out.append(it)
        out.sort(key=lambda x: x["time"] or now, reverse=True)
        self.items = out[:150]
        return self.items

    def items_for(self, metal: str = "gold", limit: int = 40) -> list[dict]:
        out = []
        for it in self.items:
            sc = it.get("scores", {}).get(metal)
            if not sc or sc["relevance"] <= 0:
                continue
            out.append({"title": it["title"], "link": it["link"], "time": it["time"], "source": it["source"], **sc})
            if len(out) >= limit:
                break
        return out

    def aggregate(self, metal: str = "gold", hours: float = 12.0, half_life_h: float = 4.0) -> dict:
        now = datetime.now(timezone.utc)
        num = den = 0.0
        n_bull = n_bear = 0
        for it in self.items_for(metal, limit=1000):
            if not it.get("time"):
                continue
            age = (now - it["time"]).total_seconds() / 3600.0
            if age > hours or age < -1:
                continue
            w = it["relevance"] * math.pow(0.5, max(age, 0) / half_life_h)
            num += w * it["sentiment"]
            den += w
            n_bull += it["sentiment"] > 0.2
            n_bear += it["sentiment"] < -0.2
        score = num / den if den > 0 else 0.0
        label = "偏多" if score > 0.15 else ("偏空" if score < -0.15 else "中性")
        return {"score": round(score, 3), "label": label, "bullish": n_bull, "bearish": n_bear, "weight": round(den, 2)}

    # --------------------------------------------------------------- calendar
    def refresh_calendar(self, force: bool = False) -> list[dict]:
        """ForexFactory weekly JSON. Cached to disk; refreshed at most every 2 hours."""
        cache = self.cache_dir / "calendar.json"
        if not force and time.time() - self.calendar_fetched < 7200 and self.calendar:
            return self.calendar
        events = []
        ok = False
        for url in CALENDAR_URLS:
            try:
                r = self.s.get(url, timeout=12)
                if r.status_code == 200:
                    events.extend(r.json())
                    ok = True
            except Exception as e:
                log.debug("calendar %s failed: %s", url, e)
        if ok:
            cache.write_text(json.dumps(events))
            self.calendar_fetched = time.time()
        elif cache.exists():
            events = json.loads(cache.read_text())
        parsed = []
        for ev in events:
            try:
                t = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
            except Exception:
                continue
            parsed.append({
                "time": t, "title": ev.get("title", ""), "country": ev.get("country", ""),
                "impact": ev.get("impact", ""), "forecast": ev.get("forecast", ""), "previous": ev.get("previous", ""),
            })
        parsed.sort(key=lambda e: e["time"])
        # de-duplicate (this week / next week overlap)
        uniq, seen = [], set()
        for e in parsed:
            k = (e["time"], e["title"], e["country"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        self.calendar = uniq
        return uniq

    @staticmethod
    def is_fomc(title: str) -> bool:
        return bool(re.search(r"FOMC|Federal Funds Rate|Fed Chair|Powell", title, re.I))

    def blackout_windows(self, before_min: int = 30, after_min: int = 30, fomc_after_min: int = 60,
                         countries=("USD",), impacts=("High",)) -> list[tuple[datetime, datetime, str]]:
        out = []
        for e in self.calendar:
            if e["country"] in countries and e["impact"] in impacts:
                after = fomc_after_min if self.is_fomc(e["title"]) else after_min
                out.append((e["time"] - timedelta(minutes=before_min), e["time"] + timedelta(minutes=after), e["title"]))
        return out

    def blackout_mask(self, index: pd.DatetimeIndex, **kw) -> np.ndarray:
        """True for 15m bars whose CLOSE falls inside a blackout window."""
        mask = np.zeros(len(index), dtype=bool)
        closes = index + pd.Timedelta(minutes=15)
        for a, b, _ in self.blackout_windows(**kw):
            mask |= (closes >= pd.Timestamp(a)) & (closes <= pd.Timestamp(b))
        return mask
