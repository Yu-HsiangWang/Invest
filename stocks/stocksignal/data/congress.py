"""US politicians' stock-trade disclosures (STOCK Act PTRs).

Source: the open dataset of kadoa-org/congress-trading-monitor (MIT), which parses the official
House Clerk, Senate eFD and OGE filings every day and publishes the latest 5,000 transactions as
static JSON on GitHub. We read that file (one ~4 MB download, cached for 12 h) and keep a local
history so the dashboard can show 1-2 years of trades per ticker. The official PDF of every
filing is linked (doc_url).

What the dashboard does with it (see docs/RESEARCH.md): information feed + a "cluster" flag when
several different members bought the same stock within a month. It is NOT a buy signal by itself -
the evidence that following disclosed trades (45-day lag) beats the market is weak.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("stocksignal.congress")
URL_LATEST = "https://raw.githubusercontent.com/kadoa-org/congress-trading-monitor/main/public/data/trades.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) stocksignal-dashboard"
KEEP = ["id", "source_id", "filer_id", "filer_name", "chamber", "party", "state", "branch", "transaction_date",
        "filing_date", "owner", "ticker", "asset_name", "asset_type", "transaction_type", "amount_range_low",
        "amount_range_high", "amount_range_label", "days_to_file", "is_late", "doc_url"]


class CongressFeed:
    def __init__(self, cache_dir: str | Path, seed_csv: str | Path | None = None, max_age_h: float = 12.0):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history = self.dir / "congress_trades.csv.gz"
        self.seed_csv = Path(seed_csv) if seed_csv else None
        self.max_age = timedelta(hours=max_age_h)
        self.status = "not loaded"
        self.df: pd.DataFrame = pd.DataFrame(columns=KEEP)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})

    # ------------------------------------------------------------------ loading
    def load(self, refresh: bool = True) -> pd.DataFrame:
        if self.history.exists():
            self.df = pd.read_csv(self.history, compression="gzip")
        elif self.seed_csv and self.seed_csv.exists():
            self.df = pd.read_csv(self.seed_csv, compression="gzip" if self.seed_csv.suffix == ".gz" else None)
        self.df = self._clean(self.df)
        fresh = self.history.exists() and (time.time() - self.history.stat().st_mtime) < self.max_age.total_seconds()
        if refresh and not fresh:
            try:
                r = self.s.get(URL_LATEST, timeout=60)
                r.raise_for_status()
                latest = self._clean(pd.DataFrame(r.json()))
                n0 = len(self.df)
                self.df = pd.concat([self.df, latest]).drop_duplicates(subset=["id"], keep="last")
                self.df = self._clean(self.df)
                self.df.to_csv(self.history, index=False, compression="gzip")
                self.status = f"ok ({len(self.df) - n0:+d} new, latest filing {self.df['filing_date'].max().date()})"
            except Exception as e:  # noqa: BLE001
                self.status = f"error: {type(e).__name__}"
                if self.history.exists():
                    self.history.touch()
        else:
            self.status = f"cached (latest filing {self.df['filing_date'].max().date()})" if len(self.df) else "empty"
        return self.df

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=KEEP)
        for k in KEEP:
            if k not in df.columns:
                df[k] = None
        df = df[KEEP].copy()
        for k in ("transaction_date", "filing_date"):
            df[k] = pd.to_datetime(df[k], errors="coerce")
        df["ticker"] = df["ticker"].astype("string").str.upper().str.strip()
        df = df.dropna(subset=["filing_date"])
        return df.sort_values(["filing_date", "transaction_date"]).reset_index(drop=True)

    # ------------------------------------------------------------------ queries
    def for_ticker(self, ticker: str, days: int = 365) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        since = pd.Timestamp.now(tz=None).normalize() - pd.Timedelta(days=days)
        d = self.df[(self.df["ticker"] == ticker.upper()) & (self.df["filing_date"] >= since)]
        return d.sort_values("filing_date", ascending=False)

    def summary(self, ticker: str, days: int = 90) -> dict:
        d = self.for_ticker(ticker, days)
        if d.empty:
            return {"ticker": ticker, "days": days, "buys": 0, "sells": 0, "buyers": 0, "sellers": 0, "cluster": False,
                    "net_mid": 0.0, "latest": None}
        buys = d[d["transaction_type"].str.startswith("Purchase", na=False)]
        sells = d[d["transaction_type"].str.startswith("Sale", na=False)]
        mid = lambda x: ((x["amount_range_low"].fillna(0) + x["amount_range_high"].fillna(x["amount_range_low"]).fillna(0)) / 2).sum()  # noqa: E731
        buyers = buys["filer_id"].nunique()
        # cluster = 3+ different members bought within any 30-day window of filing dates
        cluster = False
        if buyers >= 3:
            fd = buys.drop_duplicates("filer_id")["filing_date"].sort_values().to_numpy()
            for i in range(len(fd) - 2):
                if (fd[i + 2] - fd[i]) <= pd.Timedelta(days=30):
                    cluster = True
                    break
        latest = d.iloc[0]
        return {"ticker": ticker, "days": days, "buys": int(len(buys)), "sells": int(len(sells)), "buyers": int(buyers),
                "sellers": int(sells["filer_id"].nunique()), "cluster": bool(cluster), "net_mid": float(mid(buys) - mid(sells)),
                "latest": {"who": latest["filer_name"], "party": latest["party"], "chamber": latest["chamber"],
                           "type": latest["transaction_type"], "amount": latest["amount_range_label"],
                           "traded": _d(latest["transaction_date"]), "filed": _d(latest["filing_date"]), "doc": latest["doc_url"]}}

    def recent(self, tickers: list[str] | None = None, days: int = 30, limit: int = 60) -> list[dict]:
        if self.df.empty:
            return []
        since = pd.Timestamp.now(tz=None).normalize() - pd.Timedelta(days=days)
        d = self.df[self.df["filing_date"] >= since]
        if tickers:
            d = d[d["ticker"].isin([t.upper() for t in tickers])]
        d = d.sort_values("filing_date", ascending=False).head(limit)
        return [{"who": r.filer_name, "party": r.party, "chamber": r.chamber, "ticker": r.ticker, "asset": r.asset_name,
                 "type": r.transaction_type, "amount": r.amount_range_label, "traded": _d(r.transaction_date),
                 "filed": _d(r.filing_date), "late": bool(r.is_late) if pd.notna(r.is_late) else False, "doc": r.doc_url}
                for r in d.itertuples()]

    def market_flow(self, weeks: int = 12) -> list[dict]:
        """Weekly counts of purchases vs sales (all tickers) - a crude 'politicians buying or selling?' gauge."""
        if self.df.empty:
            return []
        since = pd.Timestamp.now(tz=None).normalize() - pd.Timedelta(weeks=weeks)
        d = self.df[(self.df["filing_date"] >= since) & self.df["ticker"].notna()].copy()
        d["week"] = d["filing_date"].dt.to_period("W").dt.start_time
        g = d.groupby("week").agg(buys=("transaction_type", lambda s: int(s.str.startswith("Purchase").sum())),
                                  sells=("transaction_type", lambda s: int(s.str.startswith("Sale").sum())),
                                  members=("filer_id", "nunique"))
        return [{"week": _d(w), **{k: int(v) for k, v in row.items()}} for w, row in g.iterrows()]


def _d(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d")
