"""Taiwan Stock Exchange open data (no key): daily snapshots that we append to local history.

OpenAPI (https://openapi.twse.com.tw/v1/...) returns the LAST trading day only, so every call
is appended to a CSV under the cache dir; after a few weeks the dashboard has its own history
for P/E, margin balances, insider filings and monthly revenue. For long P/E history the
research scripts use FinMind / TWSE legacy JSON on the user's PC (see scripts/).

Endpoints verified 2026-11-08 (see docs/DATA_SOURCES.md):
  /exchangeReport/STOCK_DAY_ALL   daily OHLCV, all listed stocks (Date in ROC calendar)
  /exchangeReport/BWIBBU_ALL      P/E, dividend yield, P/B per stock
  /exchangeReport/MI_MARGN        margin trading balances
  /opendata/t187ap05_L            monthly revenue (listed companies)
  /opendata/t187ap12_L            insider pre-announced share transfers (內部人轉讓事前申報)
  /opendata/t187ap11_L            directors' / supervisors' holdings
  /opendata/t187ap04_L            material information announcements
  /fund/MI_QFIIS_cat              foreign ownership by industry
  /holidaySchedule/holidaySchedule
Access from cloud IPs is blocked by the exchange's WAF; on a home PC it works with a browser UA.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("stocksignal.twse")
BASE = "https://openapi.twse.com.tw/v1"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
TPEX_BASE = "https://www.tpex.org.tw/openapi/v1"


def roc_to_date(s: str) -> pd.Timestamp | None:
    """'1151106' -> 2026-11-06 ; '115/11/06' -> same."""
    if s is None:
        return None
    s = str(s).strip().replace("/", "")
    if not s.isdigit() or len(s) < 6:
        return None
    y, m, d = int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:])
    try:
        return pd.Timestamp(year=y, month=m, day=d)
    except ValueError:
        return None


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").replace("+", "").strip())
    except (TypeError, ValueError):
        return float("nan")


class TWSEOpenData:
    def __init__(self, cache_dir: str | Path, min_interval_s: float = 1.5):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json", "Referer": "https://www.twse.com.tw/"})
        self.min_interval = min_interval_s
        self._last = 0.0
        self.status: dict[str, str] = {}

    # ------------------------------------------------------------------ fetch
    def get(self, path: str, base: str = BASE, max_age_h: float = 6.0) -> list[dict]:
        """GET one endpoint, cached to <name>.json for max_age_h."""
        name = path.strip("/").replace("/", "_")
        cache = self.dir / f"{name}.json"
        if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
            try:
                return pd.read_json(cache).to_dict(orient="records")
            except ValueError:
                pass
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.get(f"{base}/{path.strip('/')}", timeout=20)
            self._last = time.time()
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError("unexpected payload")
            pd.DataFrame(data).to_json(cache, orient="records", force_ascii=False)
            self.status[path] = f"ok ({len(data)})"
            return data
        except Exception as e:  # noqa: BLE001
            self.status[path] = f"error: {type(e).__name__}"
            if cache.exists():
                try:
                    return pd.read_json(cache).to_dict(orient="records")
                except ValueError:
                    return []
            return []

    # ------------------------------------------------------------- snapshots
    def daily_quotes(self) -> pd.DataFrame:
        rows = self.get("exchangeReport/STOCK_DAY_ALL")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "date": df["Date"].map(roc_to_date), "code": df["Code"].astype(str), "name": df["Name"],
            "open": df["OpeningPrice"].map(_num), "high": df["HighestPrice"].map(_num), "low": df["LowestPrice"].map(_num),
            "close": df["ClosingPrice"].map(_num), "volume": df["TradeVolume"].map(_num), "value": df["TradeValue"].map(_num),
            "change": df["Change"].map(_num)})
        return out.dropna(subset=["date", "close"])

    def valuation(self) -> pd.DataFrame:
        """P/E, dividend yield (%), P/B for every listed stock; appended to history/valuation.csv."""
        rows = self.get("exchangeReport/BWIBBU_ALL")
        if not rows:
            return self._history("valuation")
        df = pd.DataFrame(rows)
        out = pd.DataFrame({"date": df["Date"].map(roc_to_date), "code": df["Code"].astype(str), "name": df["Name"],
                            "pe": df["PEratio"].map(_num), "yield": df["DividendYield"].map(_num), "pb": df["PBratio"].map(_num)})
        out = out.dropna(subset=["date"])
        self._append("valuation", out, keys=["date", "code"])
        return self._history("valuation")

    def margin(self) -> pd.DataFrame:
        rows = self.get("exchangeReport/MI_MARGN")
        if not rows:
            return self._history("margin")
        df = pd.DataFrame(rows)
        cols = {c: c for c in df.columns}
        # column names are Chinese; pick by substring to survive small renames
        def pick(sub: str):
            for c in df.columns:
                if sub in c:
                    return c
            return None
        code_c, name_c = pick("代號"), pick("名稱")
        bal_c = next((c for c in df.columns if "融資" in c and "今日餘額" in c), None)
        prev_c = next((c for c in df.columns if "融資" in c and "前日餘額" in c), None)
        short_c = next((c for c in df.columns if "融券" in c and "今日餘額" in c), None)
        if not (code_c and bal_c):
            return self._history("margin")
        today = pd.Timestamp(datetime.now(timezone.utc) + timedelta(hours=8)).normalize()
        out = pd.DataFrame({"date": today, "code": df[code_c].astype(str).str.strip(), "name": df[name_c] if name_c else "",
                            "margin_balance": df[bal_c].map(_num), "margin_prev": df[prev_c].map(_num) if prev_c else float("nan"),
                            "short_balance": df[short_c].map(_num) if short_c else float("nan")})
        self._append("margin", out, keys=["date", "code"])
        return self._history("margin")

    def monthly_revenue(self) -> pd.DataFrame:
        rows = self.get("opendata/t187ap05_L", max_age_h=12)
        if not rows:
            return self._history("revenue")
        df = pd.DataFrame(rows)
        def pick(*subs):
            for c in df.columns:
                if all(s in c for s in subs):
                    return c
            return None
        c_code, c_name, c_ym = pick("公司代號"), pick("公司名稱"), pick("資料年月")
        c_rev, c_prev, c_ly = pick("營業收入", "當月營收"), pick("營業收入", "上月營收"), pick("營業收入", "去年當月營收")
        c_mom, c_yoy = pick("上月比較", "增減"), pick("去年同月", "增減")
        c_cum_yoy = pick("累計營業收入", "前期比較", "增減") or pick("累計", "增減")
        c_out = pick("出表日期")
        if not (c_code and c_ym and c_rev):
            return self._history("revenue")
        ym = df[c_ym].astype(str).str.strip()
        year = ym.str[:-2].astype(int) + 1911
        month = ym.str[-2:].astype(int)
        out = pd.DataFrame({"code": df[c_code].astype(str).str.strip(), "name": df[c_name] if c_name else "",
                            "year": year, "month": month, "revenue": df[c_rev].map(_num),
                            "revenue_prev_month": df[c_prev].map(_num) if c_prev else float("nan"),
                            "revenue_last_year": df[c_ly].map(_num) if c_ly else float("nan"),
                            "mom_pct": df[c_mom].map(_num) if c_mom else float("nan"),
                            "yoy_pct": df[c_yoy].map(_num) if c_yoy else float("nan"),
                            "cum_yoy_pct": df[c_cum_yoy].map(_num) if c_cum_yoy else float("nan"),
                            "published": df[c_out].map(roc_to_date) if c_out else pd.NaT})
        out["period"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
        self._append("revenue", out, keys=["code", "period"])
        return self._history("revenue")

    def insider_transfers(self) -> pd.DataFrame:
        """內部人（董監、經理人、大股東）轉讓事前申報 - pre-announced sales/transfers by insiders."""
        rows = self.get("opendata/t187ap12_L", max_age_h=12)
        if not rows:
            return self._history("insider_transfers")
        df = pd.DataFrame(rows)
        def pick(*subs):
            for c in df.columns:
                if all(s in c for s in subs):
                    return c
            return None
        c_code, c_name, c_role, c_who = pick("公司代號"), pick("公司名稱"), pick("申報人身分"), pick("姓名")
        c_shares = pick("預定轉讓總股數") or pick("轉讓股數")
        c_hold = pick("目前持有股數", "自有") or pick("目前持有股數")
        c_after = pick("預定轉讓後持股") or pick("轉讓後持股")
        c_method = pick("轉讓方式")
        c_out = pick("出表日期")
        if not (c_code and c_who):
            return self._history("insider_transfers")
        out = pd.DataFrame({"published": df[c_out].map(roc_to_date) if c_out else pd.NaT,
                            "code": df[c_code].astype(str).str.strip(), "name": df[c_name] if c_name else "",
                            "role": df[c_role] if c_role else "", "who": df[c_who],
                            "method": df[c_method] if c_method else "",
                            "shares": df[c_shares].map(_num) if c_shares else float("nan"),
                            "holding": df[c_hold].map(_num) if c_hold else float("nan"),
                            "holding_after": df[c_after].map(_num) if c_after else float("nan")})
        self._append("insider_transfers", out, keys=["published", "code", "who", "shares"])
        return self._history("insider_transfers")

    def material_info(self) -> pd.DataFrame:
        rows = self.get("opendata/t187ap04_L", max_age_h=3)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        def pick(*subs):
            for c in df.columns:
                if all(s in c for s in subs):
                    return c
            return None
        c_code, c_name, c_date, c_subj = pick("公司代號"), pick("公司名稱"), pick("發言日期"), pick("主旨")
        if not (c_code and c_subj):
            return pd.DataFrame()
        return pd.DataFrame({"date": df[c_date].map(roc_to_date) if c_date else pd.NaT, "code": df[c_code].astype(str).str.strip(),
                             "name": df[c_name] if c_name else "", "subject": df[c_subj]})

    def holidays(self) -> list[pd.Timestamp]:
        rows = self.get("holidaySchedule/holidaySchedule", max_age_h=24 * 7)
        out = []
        for r in rows:
            d = roc_to_date(r.get("Date") or r.get("日期"))
            if d is not None:
                out.append(d)
        return out

    # ---------------------------------------------------------------- history
    def _hist_path(self, name: str) -> Path:
        return self.dir / f"history_{name}.csv"

    def _history(self, name: str) -> pd.DataFrame:
        p = self._hist_path(name)
        if not p.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(p, dtype={"code": str})
            for c in ("date", "period", "published"):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            return df
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _append(self, name: str, new: pd.DataFrame, keys: list[str]) -> None:
        old = self._history(name)
        df = pd.concat([old, new]) if len(old) else new
        df = df.drop_duplicates(subset=keys, keep="last")
        df.to_csv(self._hist_path(name), index=False)


# ------------------------------------------------------------- legacy JSON (www.twse.com.tw)
class TWSELegacy:
    """The exchange's older JSON endpoints (work from a home IP with a browser UA; one request per day).

    T86 = 三大法人買賣超 for every listed stock on one day. We fetch the last N trading days once,
    then one new day per refresh, and keep history/t86.csv so per-stock series grow over time.
    """
    BASE = "https://www.twse.com.tw/rwd/zh"

    def __init__(self, cache_dir: str | Path, min_interval_s: float = 3.5):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Referer": "https://www.twse.com.tw/zh/trading/foreign/t86.html",
                               "Accept-Language": "zh-TW,zh;q=0.9"})
        self.min_interval = min_interval_s
        self._last = 0.0
        self.status: dict[str, str] = {}

    def _get(self, path: str, params: dict) -> dict | None:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.get(f"{self.BASE}/{path}", params=params, timeout=20)
            self._last = time.time()
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, dict) else None
        except Exception as e:  # noqa: BLE001
            self.status[path] = f"error: {type(e).__name__}"
            return None

    def t86_day(self, day: pd.Timestamp) -> pd.DataFrame:
        js = self._get("fund/T86", {"date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"})
        if not js or js.get("stat") != "OK" or not js.get("data"):
            return pd.DataFrame()
        fields = js.get("fields") or []
        df = pd.DataFrame(js["data"], columns=fields[: len(js["data"][0])])
        def col(*subs):
            for c in df.columns:
                if all(x in c for x in subs):
                    return c
            return None
        c_code = col("證券代號")
        c_f = next((c for c in df.columns if ("外陸資買賣超" in c or "外資買賣超" in c) and not c.startswith("外資自營商")), None)
        c_fd = col("外資自營商買賣超")
        c_t = col("投信買賣超")
        c_d = next((c for c in df.columns if c.strip() == "自營商買賣超股數"), None)
        c_tot = col("三大法人買賣超")
        if not (c_code and c_tot and c_f):
            return pd.DataFrame()
        out = pd.DataFrame({"date": day, "code": df[c_code].astype(str).str.strip(),
                            "foreign": df[c_f].map(_num) + (df[c_fd].map(_num).fillna(0) if c_fd else 0),
                            "trust": df[c_t].map(_num) if c_t else float("nan"),
                            "dealer": df[c_d].map(_num) if c_d else float("nan"),
                            "total": df[c_tot].map(_num)})
        self.status["fund/T86"] = f"ok {day.date()} ({len(out)})"
        return out

    def t86_history(self) -> pd.DataFrame:
        p = self.dir / "history_t86.csv"
        if not p.exists():
            return pd.DataFrame(columns=["date", "code", "foreign", "trust", "dealer", "total"])
        df = pd.read_csv(p, dtype={"code": str})
        df["date"] = pd.to_datetime(df["date"])
        return df

    def t86_update(self, backfill_days: int = 40, max_new: int = 12) -> pd.DataFrame:
        """Append missing trading days (at most `max_new` requests per call, newest first)."""
        hist = self.t86_history()
        have = set(hist["date"].dt.normalize()) if len(hist) else set()
        today = pd.Timestamp(datetime.now(timezone.utc) + timedelta(hours=8)).normalize()
        days = [d for d in pd.bdate_range(today - pd.Timedelta(days=backfill_days), today) if d not in have][::-1]
        new = []
        for d in days[:max_new]:
            df = self.t86_day(d)
            if len(df):
                new.append(df)
            elif d == today:
                continue                      # not published yet (before ~17:00 Taipei)
        if new:
            hist = pd.concat([hist] + new).drop_duplicates(subset=["date", "code"], keep="last").sort_values(["date", "code"])
            hist.to_csv(self.dir / "history_t86.csv", index=False)
        return hist

    def institutional(self, code: str, days: int = 120) -> pd.DataFrame:
        hist = self.t86_history()
        if not len(hist):
            return pd.DataFrame()
        h = hist[(hist["code"] == code) & (hist["date"] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=days))]
        return h.set_index("date")[["foreign", "trust", "dealer", "total"]].sort_index()


# ----------------------------------------------------------------------- FinMind
class FinMind:
    """Historical series for Taiwan (free tier: ~300 req/h anonymous, 600 with a free token)."""
    URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, cache_dir: str | Path, token: str = ""):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.status: dict[str, str] = {}

    def dataset(self, dataset: str, stock_id: str, start: str, end: str | None = None, max_age_h: float = 12.0) -> pd.DataFrame:
        cache = self.dir / f"{dataset}_{stock_id}.csv"
        if cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_h * 3600:
            return pd.read_csv(cache, parse_dates=["date"]) if cache.stat().st_size > 10 else pd.DataFrame()
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start}
        if end:
            params["end_date"] = end
        if self.token:
            params["token"] = self.token
        try:
            r = self.s.get(self.URL, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            df = pd.DataFrame(js.get("data") or [])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df.to_csv(cache, index=False)
            self.status[f"{dataset}:{stock_id}"] = f"ok ({len(df)})"
            return df
        except Exception as e:  # noqa: BLE001
            self.status[f"{dataset}:{stock_id}"] = f"error: {type(e).__name__}"
            if cache.exists() and cache.stat().st_size > 10:
                return pd.read_csv(cache, parse_dates=["date"])
            return pd.DataFrame()

    def institutional(self, stock_id: str, start: str) -> pd.DataFrame:
        """三大法人買賣超: columns date, name (Foreign_Investor, Investment_Trust, Dealer_*), buy, sell."""
        df = self.dataset("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start)
        if df.empty:
            return df
        df["net"] = df["buy"] - df["sell"]
        piv = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
        piv["foreign"] = piv.get("Foreign_Investor", 0) + piv.get("Foreign_Dealer_Self", 0)
        piv["trust"] = piv.get("Investment_Trust", 0)
        piv["dealer"] = piv.get("Dealer_self", 0) + piv.get("Dealer_Hedging", 0)
        piv["total"] = piv["foreign"] + piv["trust"] + piv["dealer"]
        return piv[["foreign", "trust", "dealer", "total"]]

    def per(self, stock_id: str, start: str) -> pd.DataFrame:
        """Daily P/E, P/B, dividend yield history (TaiwanStockPER)."""
        df = self.dataset("TaiwanStockPER", stock_id, start, max_age_h=24)
        if df.empty:
            return df
        return df.rename(columns={"PER": "pe", "PBR": "pb", "dividend_yield": "yield"}).set_index("date")

    def monthly_revenue(self, stock_id: str, start: str) -> pd.DataFrame:
        return self.dataset("TaiwanStockMonthRevenue", stock_id, start, max_age_h=24)
