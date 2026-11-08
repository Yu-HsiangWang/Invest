"""SEC EDGAR (free, no key; a descriptive User-Agent is mandatory).

* companyfacts  -> quarterly diluted EPS history -> trailing-12-month EPS -> P/E history for the
  本益比河流圖 (P/E river) of US stocks.
* EFTS full-text search -> recent Form 4 filings for a company -> the Form 4 XML -> open-market
  purchases (code P) and sales (code S) by insiders.
* OpenInsider HTML screener as a fallback for insider trades (one page per ticker).
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("stocksignal.sec")
UA = "stocksignal-dashboard/0.1 (personal research tool; contact: user@example.com)"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/"


class SEC:
    def __init__(self, cache_dir: str | Path, user_agent: str = UA, min_interval_s: float = 0.15):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Accept": "application/json,*/*"})
        self.min_interval = min_interval_s
        self._last = 0.0
        self.status: dict[str, str] = {}
        self._tickers: dict[str, int] | None = None

    def _get(self, url: str, **kw) -> requests.Response:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.s.get(url, timeout=30, **kw)
        self._last = time.time()
        r.raise_for_status()
        return r

    # ------------------------------------------------------------------- CIK map
    def cik(self, ticker: str) -> int | None:
        if self._tickers is None:
            cache = self.dir / "company_tickers.json"
            try:
                if not cache.exists() or (time.time() - cache.stat().st_mtime) > 7 * 86400:
                    cache.write_bytes(self._get(TICKERS_URL).content)
                js = pd.read_json(cache, orient="index")
                self._tickers = {str(r["ticker"]).upper(): int(r["cik_str"]) for _, r in js.iterrows()}
            except Exception as e:  # noqa: BLE001
                self.status["tickers"] = f"error: {type(e).__name__}"
                self._tickers = {}
        return self._tickers.get(ticker.upper().replace("-", "."))

    # ---------------------------------------------------------------- EPS / P/E
    def eps_quarterly(self, ticker: str) -> pd.DataFrame:
        """Quarterly diluted EPS (fy, fp, end, filed, eps). Cached 24 h."""
        cik = self.cik(ticker)
        if not cik:
            return pd.DataFrame()
        cache = self.dir / f"facts_{cik}.json"
        try:
            if not cache.exists() or (time.time() - cache.stat().st_mtime) > 86400:
                cache.write_bytes(self._get(FACTS_URL.format(cik=cik)).content)
            import json
            facts = json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            self.status[f"facts:{ticker}"] = f"error: {type(e).__name__}"
            return pd.DataFrame()
        gaap = (facts.get("facts") or {}).get("us-gaap") or {}
        rows = []
        for tag in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
            units = (gaap.get(tag) or {}).get("units") or {}
            for u, vals in units.items():
                for v in vals:
                    if v.get("fp") in ("Q1", "Q2", "Q3", "Q4", "FY") and v.get("form") in ("10-Q", "10-K", "10-K/A", "10-Q/A", "20-F", "40-F"):
                        rows.append({"tag": tag, "fy": v.get("fy"), "fp": v.get("fp"), "start": v.get("start"), "end": v.get("end"),
                                     "filed": v.get("filed"), "eps": v.get("val"), "form": v.get("form")})
            if rows:
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for c in ("start", "end", "filed"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df["days"] = (df["end"] - df["start"]).dt.days
        q = df[(df["days"] > 80) & (df["days"] < 100)].copy()            # true quarterly periods
        annual = df[(df["days"] > 350) & (df["days"] < 380)].copy()
        # derive Q4 = FY - (Q1+Q2+Q3) when companies only report the annual figure in the 10-K
        for _, a in annual.iterrows():
            fy_q = q[(q["end"] > a["start"]) & (q["end"] <= a["end"])]
            if len(fy_q) == 3 and not ((q["end"] == a["end"]).any()):
                q = pd.concat([q, pd.DataFrame([{"tag": a["tag"], "fy": a["fy"], "fp": "Q4", "start": fy_q["end"].max(), "end": a["end"],
                                                 "filed": a["filed"], "eps": a["eps"] - fy_q["eps"].sum(), "form": a["form"], "days": 91}])])
        q = q.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")      # first filing of each quarter
        return q[["fy", "fp", "start", "end", "filed", "eps"]].reset_index(drop=True)

    def eps_ttm_series(self, ticker: str) -> pd.Series:
        """TTM EPS indexed by FILING date (so the P/E history uses only what was public at the time)."""
        q = self.eps_quarterly(ticker)
        if len(q) < 4:
            return pd.Series(dtype=float)
        ttm = q["eps"].rolling(4).sum()
        s = pd.Series(ttm.to_numpy(), index=q["filed"]).dropna()
        return s[~s.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------------ Form 4
    def form4_recent(self, ticker: str, days: int = 180) -> pd.DataFrame:
        """Open-market insider purchases/sales from recent Form 4 filings (EFTS search + XML parse)."""
        cik = self.cik(ticker)
        if not cik:
            return pd.DataFrame()
        cache = self.dir / f"form4_{cik}.csv"
        if cache.exists() and (time.time() - cache.stat().st_mtime) < 12 * 3600:
            return pd.read_csv(cache, parse_dates=["filed", "date"]) if cache.stat().st_size > 10 else pd.DataFrame()
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)
        rows = []
        try:
            r = self._get(EFTS_URL, params={"q": f"{cik}", "ciks": f"{cik:010d}", "forms": "4", "dateRange": "custom",
                                            "startdt": start.isoformat(), "enddt": end.isoformat()})
            hits = ((r.json().get("hits") or {}).get("hits") or [])[:60]
            for h in hits:
                src = h.get("_source") or {}
                adsh = src.get("adsh")
                if not adsh:
                    continue
                rows.extend(self._parse_form4(cik, adsh, src.get("file_date")))
            df = pd.DataFrame(rows)
            if len(df):
                df = df.sort_values("filed", ascending=False)
            df.to_csv(cache, index=False)
            self.status[f"form4:{ticker}"] = f"ok ({len(df)})"
            return df
        except Exception as e:  # noqa: BLE001
            self.status[f"form4:{ticker}"] = f"error: {type(e).__name__}"
            return pd.read_csv(cache, parse_dates=["filed", "date"]) if cache.exists() and cache.stat().st_size > 10 else pd.DataFrame()

    def _parse_form4(self, cik: int, adsh: str, file_date: str | None) -> list[dict]:
        folder = ARCHIVE.format(cik=cik, adsh=adsh.replace("-", ""))
        idx = self._get(folder).text
        xmls = re.findall(r'href="([^"]+\.xml)"', idx)
        xmls = [x for x in xmls if "primary_doc" not in x.lower() or True]
        for x in xmls:
            url = x if x.startswith("http") else ("https://www.sec.gov" + x if x.startswith("/") else folder + x)
            try:
                txt = self._get(url).text
            except Exception:  # noqa: BLE001
                continue
            if "<ownershipDocument" in txt:
                return parse_form4_xml(txt, filed=file_date)
        return []

    # --------------------------------------------------------------- OpenInsider
    def openinsider(self, ticker: str, days: int = 365) -> pd.DataFrame:
        url = f"http://openinsider.com/screener?s={ticker}&fd={days}&td=0&xp=1&xs=1&cnt=200&page=1"
        try:
            r = self._get(url, headers={"Accept": "text/html"})
            tables = pd.read_html(r.text, attrs={"class": "tinytable"})
            if not tables:
                return pd.DataFrame()
            t = tables[0]
            t.columns = [re.sub(r"\W+", "_", str(c)).strip("_").lower() for c in t.columns]
            return t
        except Exception as e:  # noqa: BLE001
            self.status[f"openinsider:{ticker}"] = f"error: {type(e).__name__}"
            return pd.DataFrame()


def parse_form4_xml(txt: str, filed: str | None = None) -> list[dict]:
    """Non-derivative open-market transactions (codes P / S) from a Form 4 ownershipDocument."""
    out = []
    try:
        root = ET.fromstring(txt.encode("utf-8") if isinstance(txt, str) else txt)
    except ET.ParseError:
        return out
    def tx(el, path):
        n = el.find(path)
        return (n.text or "").strip() if n is not None else ""
    owner = tx(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
    rel = root.find(".//reportingOwner/reportingOwnerRelationship")
    role = ""
    if rel is not None:
        if tx(rel, "isDirector") in ("1", "true"):
            role = "Director"
        if tx(rel, "isOfficer") in ("1", "true"):
            role = tx(rel, "officerTitle") or "Officer"
        if tx(rel, "isTenPercentOwner") in ("1", "true"):
            role = (role + " / " if role else "") + "10% owner"
    issuer = tx(root, ".//issuer/issuerTradingSymbol")
    for t in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        code = tx(t, "transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        shares = _f(tx(t, "transactionAmounts/transactionShares/value"))
        price = _f(tx(t, "transactionAmounts/transactionPricePerShare/value"))
        out.append({"filed": pd.to_datetime(filed) if filed else pd.NaT, "date": pd.to_datetime(tx(t, "transactionDate/value"), errors="coerce"),
                    "ticker": issuer, "owner": owner, "role": role, "code": code, "type": "買進" if code == "P" else "賣出",
                    "shares": shares, "price": price, "value": (shares or 0) * (price or 0),
                    "after": _f(tx(t, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"))})
    return out


def _f(x: str) -> float | None:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None
