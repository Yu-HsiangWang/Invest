"""News headlines for the watch-lists (RSS / JSON feeds, English + Traditional Chinese).

How news is used: shown as context, tagged 利多 / 利空 / 中性 per headline, aggregated per ticker
and per market over the last few days. It never produces a buy/sell signal on its own - at
monthly horizons headline sentiment is mostly coincident with price, not predictive.
"""
from __future__ import annotations

import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import requests

log = logging.getLogger("stocksignal.news")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
_TAG = re.compile(r"<[^>]+>")

# ----------------------------------------------------------------------------- feeds
# {name, url, market, kind: rss|cnyes|yahoo_ticker|gnews_query}
MARKET_FEEDS = {
    "US": [
        {"name": "CNBC Technology", "url": "https://www.cnbc.com/id/19854910/device/rss/rss.html", "kind": "rss"},
        {"name": "CNBC Markets", "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html", "kind": "rss"},
        {"name": "Google News", "url": "https://news.google.com/rss/search?q=(Nasdaq+OR+%22tech+stocks%22+OR+semiconductor+OR+Nvidia+OR+%22AI+chips%22)+when:2d&hl=en-US&gl=US&ceid=US:en", "kind": "rss"},
        {"name": "Google News", "url": "https://news.google.com/rss/search?q=(%22Federal+Reserve%22+OR+%22rate+cut%22+OR+inflation+OR+tariffs)+stocks+when:2d&hl=en-US&gl=US&ceid=US:en", "kind": "rss"},
        {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex", "kind": "rss"},
    ],
    "TW": [
        {"name": "鉅亨網 台股", "url": "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30", "kind": "cnyes"},
        {"name": "鉅亨網 美股", "url": "https://api.cnyes.com/media/api/v1/newslist/category/us_stock?limit=30", "kind": "cnyes"},
        {"name": "Google 新聞", "url": "https://news.google.com/rss/search?q=(台股+OR+台積電+OR+半導體+OR+AI伺服器+OR+外資)+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", "kind": "rss"},
        {"name": "Google 新聞", "url": "https://news.google.com/rss/search?q=(聯準會+OR+降息+OR+關稅+OR+美股)+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", "kind": "rss"},
        {"name": "中央社 財經", "url": "https://feeds.feedburner.com/rsscna/finance", "kind": "rss"},
    ],
}


def ticker_feed(symbol: str, name: str, market: str) -> list[dict]:
    """Per-ticker feeds (Yahoo per-symbol RSS for US; Google News search for both)."""
    if market == "US":
        return [
            {"name": "Yahoo Finance", "url": f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US", "kind": "rss"},
            {"name": "Google News", "url": "https://news.google.com/rss/search?q=" + quote_plus(f'"{name}" OR {symbol} stock when:7d') + "&hl=en-US&gl=US&ceid=US:en", "kind": "rss"},
        ]
    code = symbol.split(".")[0]
    return [
        {"name": "Google 新聞", "url": "https://news.google.com/rss/search?q=" + quote_plus(f"{name} OR {code} when:7d") + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", "kind": "rss"},
    ]


# ------------------------------------------------------------------------ sentiment
# (regex, weight). Positive = bullish for the stock / market mentioned.
LEX_EN: list[tuple[str, float]] = [
    (r"\b(surge|soar|jump|rall(y|ies)|record high|all-time high|beat(s)? (estimates|expectations)|raises? (guidance|forecast|outlook)|upgrade[sd]?|outperform|strong demand|buyback|dividend (hike|increase))\b", 1.0),
    (r"\b(plunge|tumble|slump|sink|sell-?off|miss(es|ed)? (estimates|expectations)|cuts? (guidance|forecast|outlook)|downgrade[sd]?|underperform|weak demand|layoffs?|recall|probe|antitrust|lawsuit|ban|export (controls?|curbs?))\b", -1.0),
    (r"\b(rate cut|dovish|easing|soft landing|cooling inflation)\b", 0.6),
    (r"\b(rate hike|hawkish|tightening|hot(ter)? inflation|recession|tariffs?|trade war)\b", -0.6),
    (r"\b(gains?|rises?|climbs?|advances?|rebounds?|higher)\b", 0.4),
    (r"\b(falls?|drops?|slides?|declines?|lower|losses?)\b", -0.4),
]
LEX_ZH: list[tuple[str, float]] = [
    (r"(大漲|飆漲|漲停|創新高|歷史新高|法說會?利多|優於預期|上修|調升目標價|買超|加碼|擴產|接單|急單|回補|超預期|報喜|營收創|年增)", 1.0),
    (r"(大跌|重挫|跌停|崩跌|殺盤|低於預期|下修|調降目標價|賣超|減碼|砍單|庫存調整|裁員|反壟斷|罰款|出口管制|禁令|衰退|關稅)", -1.0),
    (r"(降息|寬鬆|降溫|軟著陸)", 0.6),
    (r"(升息|緊縮|通膨升溫|鷹派)", -0.6),
    (r"(上漲|走高|反彈|收紅|翻紅|勁揚)", 0.4),
    (r"(下跌|走低|收黑|翻黑|回檔|拉回|重跌)", -0.4),
]
_LEX = [(re.compile(p, re.I), w) for p, w in LEX_EN] + [(re.compile(p), w) for p, w in LEX_ZH]


def score_headline(text: str) -> tuple[float, list[str]]:
    total, cues = 0.0, []
    for rx, w in _LEX:
        m = rx.search(text or "")
        if m:
            total += w
            cues.append(m.group(0)[:30])
    import math
    return float(math.tanh(total)), cues


def label(score: float) -> str:
    return "利多" if score > 0.2 else ("利空" if score < -0.2 else "中性")


# ----------------------------------------------------------------------------- parse
def _clean(s: str | None) -> str:
    return html.unescape(_TAG.sub("", s or "")).strip()


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)
    except Exception:  # noqa: BLE001
        try:
            d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def parse_rss(xml_text: str, source: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    atom = "{http://www.w3.org/2005/Atom}"
    for it in root.iter("item"):
        src = it.find("source")
        items.append({"title": _clean(it.findtext("title")), "link": (it.findtext("link") or "").strip(),
                      "time": _parse_date(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date")),
                      "source": (src.text.strip() if src is not None and src.text else source)})
    for it in root.iter(f"{atom}entry"):
        link_el = it.find(f"{atom}link")
        items.append({"title": _clean(it.findtext(f"{atom}title")), "link": link_el.get("href") if link_el is not None else "",
                      "time": _parse_date(it.findtext(f"{atom}updated") or it.findtext(f"{atom}published")), "source": source})
    return items


def parse_cnyes(js: dict, source: str) -> list[dict]:
    items = []
    data = (js.get("items") or {}).get("data") or []
    for d in data:
        nid = d.get("newsId")
        t = d.get("publishAt")
        items.append({"title": _clean(d.get("title")), "link": f"https://news.cnyes.com/news/id/{nid}" if nid else "",
                      "time": datetime.fromtimestamp(t, timezone.utc) if t else None, "source": source,
                      "stocks": [s.get("key") for s in (d.get("market") or []) if isinstance(s, dict)]})
    return items


class NewsMonitor:
    def __init__(self, cache_dir: str | Path, max_age_h: float = 72.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_h = max_age_h
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/rss+xml, application/json, application/xml, text/xml, */*"})
        self.items: dict[str, list[dict]] = {"TW": [], "US": []}
        self.ticker_items: dict[str, list[dict]] = {}
        self.feed_status: dict[str, str] = {}

    def fetch(self, feed: dict) -> list[dict]:
        try:
            r = self.s.get(feed["url"], timeout=12)
            r.raise_for_status()
            if feed.get("kind") == "cnyes":
                items = parse_cnyes(r.json(), feed["name"])
            else:
                items = parse_rss(r.text, feed["name"])
            self.feed_status[feed["url"]] = f"ok ({len(items)})"
            return items
        except Exception as e:  # noqa: BLE001
            self.feed_status[feed["url"]] = f"error: {type(e).__name__}"
            return []

    def _dedupe(self, items: list[dict]) -> list[dict]:
        seen, out = set(), []
        now = datetime.now(timezone.utc)
        for it in items:
            key = re.sub(r"\W+", "", (it.get("title") or "").lower())[:80]
            if not key or key in seen:
                continue
            if it.get("time") and (now - it["time"]) > timedelta(hours=self.max_age_h):
                continue
            seen.add(key)
            sc, cues = score_headline(it["title"])
            it["sentiment"] = round(sc, 2)
            it["label"] = label(sc)
            it["cues"] = cues
            out.append(it)
        out.sort(key=lambda x: x.get("time") or now, reverse=True)
        return out

    def refresh_market(self, market: str) -> list[dict]:
        items = []
        for fd in MARKET_FEEDS.get(market, []):
            items.extend(self.fetch(fd))
        self.items[market] = self._dedupe(items)[:120]
        self._save()
        return self.items[market]

    def refresh_ticker(self, symbol: str, name: str, market: str) -> list[dict]:
        items = []
        for fd in ticker_feed(symbol, name, market):
            items.extend(self.fetch(fd))
        code = symbol.split(".")[0]
        # also pick market-feed items that mention the ticker
        rx = re.compile(re.escape(name) + ("|" + re.escape(code) if market == "TW" or len(code) >= 3 else ""), re.I)
        for it in self.items.get(market, []):
            if rx.search(it.get("title", "")) or code in (it.get("stocks") or []):
                items.append(dict(it))
        self.ticker_items[symbol] = self._dedupe(items)[:40]
        return self.ticker_items[symbol]

    def aggregate(self, items: list[dict], days: float = 3.0) -> dict:
        now = datetime.now(timezone.utc)
        n_bull = n_bear = 0
        tot = 0.0
        n = 0
        for it in items:
            if it.get("time") and (now - it["time"]) > timedelta(days=days):
                continue
            n += 1
            tot += it.get("sentiment", 0.0)
            n_bull += it.get("sentiment", 0) > 0.2
            n_bear += it.get("sentiment", 0) < -0.2
        score = tot / n if n else 0.0
        return {"score": round(score, 2), "label": "偏多" if score > 0.15 else ("偏空" if score < -0.15 else "中性"),
                "bullish": n_bull, "bearish": n_bear, "n": n}

    def _save(self) -> None:
        try:
            ser = {m: [{**it, "time": it["time"].isoformat() if it.get("time") else None} for it in v] for m, v in self.items.items()}
            (self.cache_dir / "news.json").write_text(json.dumps(ser, ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
