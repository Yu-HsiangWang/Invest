"""Hub: keeps prices / feeds fresh, runs the analysis for every watch-list instrument, raises alerts."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .analysis import analyze, chart_payload
from .backtest import TW_ETF, TW_STOCK, US_STOCK, Costs
from .data.congress import CongressFeed
from .data.macro import Macro, earnings_dates, tw_revenue_deadline
from .data.news import NewsMonitor
from .data.prices import PriceStore
from .data.sec import SEC
from .data.twse import FinMind, TWSELegacy, TWSEOpenData
from .settings import Settings
from .universe import DEFAULT_UNIVERSE, Instrument

log = logging.getLogger("stocksignal.engine")
BENCH = {"TW": "^TWII", "US": "SPY"}
SECTOR_INDEX = {"TW": ["0050.TW", "^TWII"], "US": ["QQQ", "SOXX", "SPY"]}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Hub:
    def __init__(self, root: Path, offline: bool = False, markets: list[str] | None = None):
        self.root = Path(root)
        self.offline = offline
        self.settings = Settings.load(self.root / "settings.json")
        if markets:
            self.settings.markets = markets
        cache = self.root / "data" / "cache"
        self.prices = PriceStore(cache / "prices", max_age_hours=self.settings.price_refresh_hours)
        self.news = NewsMonitor(cache / "news")
        self.congress = CongressFeed(cache / "congress", seed_csv=self.root / "data" / "congress_seed.csv.gz")
        self.twse = TWSEOpenData(cache / "twse")
        self.twse_legacy = TWSELegacy(cache / "twse")
        self.finmind = FinMind(cache / "finmind", token=self.settings.finmind_token)
        self.sec = SEC(cache / "sec", user_agent=self.settings.sec_user_agent or "stocksignal-dashboard/0.1")
        self.macro = Macro(cache / "macro")
        self.research = self._load_json(self.root / "results" / "research_summary.json") or {}
        self.analyses: dict[str, dict] = {}
        self.frames: dict[str, pd.DataFrame] = {}
        self.pe_series: dict[str, pd.Series] = {}
        self.ticker_feeds: dict[str, dict] = {}
        self.alerts: list[dict] = []
        self.status: dict = {"prices": "尚未更新", "feeds": "尚未更新", "last_price_refresh": None, "last_feed_refresh": None}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_prices = 0.0
        self._last_feeds = 0.0
        self._prev_state: dict[str, dict] = {}
        self.alert_log = self.root / "data" / "cache" / "alerts.json"
        try:
            self.alerts = json.loads(self.alert_log.read_text(encoding="utf-8"))[-200:] if self.alert_log.exists() else []
        except Exception:  # noqa: BLE001
            self.alerts = []

    # ----------------------------------------------------------------- universe
    def universe(self) -> dict[str, Instrument]:
        out: dict[str, Instrument] = {}
        want = {"TW": self.settings.watch_tw, "US": self.settings.watch_us}
        for m in self.settings.markets:
            for sym in want.get(m, []):
                sym = sym.upper()
                if sym in DEFAULT_UNIVERSE:
                    out[sym] = DEFAULT_UNIVERSE[sym]
                else:
                    meta = self.settings.custom_names.get(sym, {})
                    code = sym.split(".")[0]
                    kind = meta.get("kind") or ("etf" if (m == "TW" and code.startswith("00")) else "stock")
                    out[sym] = Instrument(symbol=sym, name=meta.get("name", sym), market=m, kind=kind, sector=meta.get("sector", ""),
                                          code=code, benchmark=BENCH[m])
        return out

    def costs_for(self, inst: Instrument) -> Costs:
        if inst.market == "TW":
            return TW_ETF if inst.kind == "etf" else TW_STOCK
        return US_STOCK

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="hub")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            self.refresh_prices()
        except Exception as e:  # noqa: BLE001
            log.exception("price refresh failed: %s", e)
        try:
            self.refresh_feeds()
        except Exception as e:  # noqa: BLE001
            log.exception("feed refresh failed: %s", e)
        while not self._stop.is_set():
            now = time.time()
            try:
                if now - self._last_prices > self.settings.price_refresh_hours * 3600:
                    self.refresh_prices()
                if now - self._last_feeds > self.settings.feeds_refresh_minutes * 60:
                    self.refresh_feeds()
            except Exception as e:  # noqa: BLE001
                log.exception("refresh loop: %s", e)
            self._stop.wait(60)

    # -------------------------------------------------------------------- prices
    def _bundled(self, sym: str) -> pd.DataFrame | None:
        p = self.root / "data" / "daily" / f"{sym}.csv.gz"
        if not p.exists():
            p = self.root / "data" / "daily" / f"{sym}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if "close_raw" not in df.columns:
            df["close_raw"] = df["close"]
        return df[["open", "high", "low", "close", "volume", "close_raw"]]

    def _frame(self, sym: str) -> pd.DataFrame | None:
        """Live (Yahoo) history when reachable, else the bundled research history; live data wins where it overlaps."""
        live = None if self.offline else self.prices.daily(sym, years=27)
        bundled = self._bundled(sym)
        if live is None or len(live) < 300:
            if bundled is None:
                return None
            self.prices.status.setdefault(sym, "bundled history only")
            return bundled
        if bundled is not None and bundled.index[0] < live.index[0]:
            # extend backwards with the bundled history, scaled to the live series on the first overlap day
            common = live.index.intersection(bundled.index)
            if len(common):
                k = float(live.loc[common[0], "close"] / bundled.loc[common[0], "close"])
                early = bundled[bundled.index < common[0]].copy()
                for c in ("open", "high", "low", "close"):
                    early[c] = early[c] * k
                live = pd.concat([early, live])
        return live

    def _pe(self, sym: str, inst: Instrument, df: pd.DataFrame) -> pd.Series | None:
        parts = []
        bundled = self.root / "data" / "daily" / f"{sym}.valuation.csv.gz"
        if not bundled.exists():
            bundled = self.root / "data" / "daily" / f"{sym}.valuation.csv"
        if bundled.exists():
            v = pd.read_csv(bundled, index_col=0, parse_dates=True)
            if "pe" in v.columns:
                parts.append(v["pe"])
        if not self.offline and inst.market == "TW":
            hist = self.twse._history("valuation")
            if len(hist):
                h = hist[hist["code"] == inst.code].set_index("date")["pe"]
                if len(h):
                    parts.append(h)
            fm = self.finmind.per(inst.code, start=str((df.index[-1] - pd.Timedelta(days=365 * 3)).date()))
            if len(fm) and "pe" in fm.columns:
                parts.append(fm["pe"])
        if not self.offline and inst.market == "US" and inst.kind == "stock":
            eps = self.sec.eps_ttm_series(inst.code)
            if len(eps):
                eps_d = eps.reindex(df.index.union(eps.index)).ffill().reindex(df.index)
                raw = df["close_raw"].where(df["close_raw"].notna(), df["close"])
                pe = (raw / eps_d).where(eps_d > 0)
                parts.append(pe)
        if not parts:
            return None
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = pd.to_numeric(s, errors="coerce")
        return s.dropna()

    def refresh_prices(self):
        uni = self.universe()
        bench_syms = set(BENCH[m] for m in self.settings.markets) | {s for m in self.settings.markets for s in SECTOR_INDEX[m]}
        frames = {}
        for sym in list(uni) + sorted(bench_syms - set(uni)):
            try:
                df = self._frame(sym)
            except Exception as e:  # noqa: BLE001
                log.warning("price %s: %s", sym, e)
                df = self._bundled(sym)
            if df is not None and len(df) > 60:
                frames[sym] = df
        pooled = {m: (self.research.get("markets", {}).get(m, {}).get("pooled") or {}) for m in ("TW", "US")}
        new_analyses = {}
        pes = {}
        for sym, inst in uni.items():
            df = frames.get(sym)
            if df is None:
                continue
            try:
                pe = self._pe(sym, inst, df)
                pes[sym] = pe
                bench = frames.get(BENCH[inst.market])
                a = analyze(df, pe=pe, bench=bench["close"] if bench is not None else None, kind=inst.kind, market=inst.market,
                            costs=self.costs_for(inst), pooled=_pooled_for(pooled[inst.market]))
                a.update({"symbol": sym, "name": inst.name, "market": inst.market, "kind": inst.kind, "sector": inst.sector, "code": inst.code,
                          "source": self.prices.status.get(sym, "bundled")})
                new_analyses[sym] = a
            except Exception as e:  # noqa: BLE001
                log.exception("analysis %s failed: %s", sym, e)
        with self._lock:
            self.frames = frames
            self.analyses = new_analyses
            self.pe_series = pes
            self._last_prices = time.time()
            self.status["prices"] = f"{len(new_analyses)} 檔已更新"
            self.status["last_price_refresh"] = _utcnow().isoformat()
        self._zone_alerts()

    # --------------------------------------------------------------------- feeds
    def refresh_feeds(self):
        if self.offline:
            try:
                self.congress.load(refresh=False)
            except Exception as e:  # noqa: BLE001
                log.warning("congress seed: %s", e)
            for sym, inst in self.universe().items():
                if inst.market == "US":
                    self.ticker_feeds[sym] = {"symbol": sym, "congress": self.congress.summary(inst.code, 90),
                                              "congress_trades": self.congress.recent([inst.code], days=365, limit=30), "news": [], "news_agg": {}}
            self._last_feeds = time.time()
            self.status["feeds"] = "離線模式（只有國會交易的內建資料）"
            return
        uni = self.universe()
        for m in self.settings.markets:
            try:
                self.news.refresh_market(m)
            except Exception as e:  # noqa: BLE001
                log.warning("news %s: %s", m, e)
        try:
            self.congress.load()
        except Exception as e:  # noqa: BLE001
            log.warning("congress: %s", e)
        if "TW" in self.settings.markets:
            for fn in (self.twse.valuation, self.twse.margin, self.twse.monthly_revenue, self.twse.insider_transfers, self.twse_legacy.t86_update):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    log.warning("twse %s: %s", fn.__name__, e)
        try:
            self.macro.snapshot()
            self.macro.fomc_dates()
        except Exception as e:  # noqa: BLE001
            log.warning("macro: %s", e)
        # per-ticker feeds (cheap ones every cycle; SEC / FinMind are cached ~12-24 h inside the clients)
        for sym, inst in uni.items():
            try:
                self.ticker_feeds[sym] = self._ticker_feed(sym, inst)
            except Exception as e:  # noqa: BLE001
                log.warning("feed %s: %s", sym, e)
        self._last_feeds = time.time()
        self.status["feeds"] = "已更新"
        self.status["last_feed_refresh"] = _utcnow().isoformat()
        self._feed_alerts()

    def _ticker_feed(self, sym: str, inst: Instrument) -> dict:
        out: dict = {"symbol": sym, "updated": _utcnow().isoformat()}
        out["news"] = self.news.refresh_ticker(sym, inst.name, inst.market)
        out["news_agg"] = self.news.aggregate(out["news"])
        if inst.market == "US":
            out["congress"] = self.congress.summary(inst.code, 90)
            out["congress_trades"] = self.congress.recent([inst.code], days=365, limit=30)
            if inst.kind == "stock":
                f4 = self.sec.form4_recent(inst.code, days=180)
                out["insiders"] = _records(f4.head(40)) if len(f4) else []
                out["insider_summary"] = _insider_summary(f4)
            out["earnings"] = earnings_dates(sym)
        else:
            start = str((datetime.now() - timedelta(days=120)).date())
            inst_flow = self.twse_legacy.institutional(inst.code, 120)
            fm_flow = self.finmind.institutional(inst.code, start)
            if len(fm_flow):
                inst_flow = fm_flow.combine_first(inst_flow) if len(inst_flow) else fm_flow
            out["institutional"] = [{"date": str(d.date()), **{k: float(v) for k, v in r.items()}} for d, r in inst_flow.tail(30).iterrows()] if len(inst_flow) else []
            if len(inst_flow):
                last5 = inst_flow.tail(5).sum()
                last20 = inst_flow.tail(20).sum()
                out["institutional_summary"] = {"foreign_5d": float(last5["foreign"]), "foreign_20d": float(last20["foreign"]),
                                                "trust_20d": float(last20["trust"]), "total_20d": float(last20["total"]), "unit": "股"}
            rev = self.twse._history("revenue")
            if len(rev):
                r = rev[rev["code"] == inst.code].sort_values("period").tail(13)
                out["revenue"] = [{"period": p.strftime("%Y-%m"), "revenue": float(x.revenue), "yoy": None if pd.isna(x.yoy_pct) else float(x.yoy_pct),
                                   "mom": None if pd.isna(x.mom_pct) else float(x.mom_pct), "cum_yoy": None if pd.isna(x.cum_yoy_pct) else float(x.cum_yoy_pct)}
                                  for p, x in zip(r["period"], r.itertuples())]
            fmrev = self.finmind.monthly_revenue(inst.code, start=str((datetime.now() - timedelta(days=800)).date()))
            if len(fmrev) and "revenue" in fmrev.columns:
                fmrev = fmrev.sort_values("date").tail(26)
                yoy = fmrev["revenue"].pct_change(12) * 100
                out["revenue_history"] = [{"period": f"{int(r.revenue_year)}-{int(r.revenue_month):02d}", "revenue": float(r.revenue),
                                           "yoy": None if pd.isna(y) else round(float(y), 1)} for r, y in zip(fmrev.itertuples(), yoy)]
            val = self.twse._history("valuation")
            if len(val):
                v = val[val["code"] == inst.code].sort_values("date").tail(1)
                if len(v):
                    out["valuation_today"] = {"pe": float(v.pe.iloc[0]) if pd.notna(v.pe.iloc[0]) else None, "yield": float(v["yield"].iloc[0]) if pd.notna(v["yield"].iloc[0]) else None,
                                              "pb": float(v.pb.iloc[0]) if pd.notna(v.pb.iloc[0]) else None, "date": str(v.date.iloc[0].date())}
            mg = self.twse._history("margin")
            if len(mg):
                g = mg[mg["code"] == inst.code].sort_values("date").tail(20)
                out["margin"] = [{"date": str(d.date()), "balance": float(b)} for d, b in zip(g["date"], g["margin_balance"])]
            ins = self.twse._history("insider_transfers")
            if len(ins):
                g = ins[ins["code"] == inst.code].sort_values("published", ascending=False).head(20)
                out["insiders"] = [{"published": str(p.date()) if pd.notna(p) else None, "who": w, "role": r, "method": m, "shares": None if pd.isna(s) else float(s),
                                    "holding": None if pd.isna(h) else float(h)} for p, w, r, m, s, h in zip(g["published"], g["who"], g["role"], g["method"], g["shares"], g["holding"])]
        return out

    # -------------------------------------------------------------------- alerts
    def _zone_alerts(self):
        if not self.settings.alert_zone_changes:
            return
        for sym, a in self.analyses.items():
            prev = self._prev_state.get(sym)
            cur = {"zone_min": a.get("zone_min"), "zone_max": a.get("zone_max"), "trend_state": a.get("trend", {}).get("state"),
                   "market_up": (a.get("market_trend") or {}).get("faber_up")}
            if prev:
                if cur["zone_min"] is not None and cur["zone_min"] <= -2 and (prev["zone_min"] is None or prev["zone_min"] > -2):
                    self.alert(sym, "zone", f"{a['name']} 進入「很便宜」區", f"現價 {a['price']}。{a['verdict']['title']}。", "signal")
                if cur["zone_max"] is not None and cur["zone_max"] >= 2 and (prev["zone_max"] is None or prev["zone_max"] < 2):
                    self.alert(sym, "zone", f"{a['name']} 進入「很貴」區", "不追高；要用錢優先從這裡減碼。", "info")
                if self.settings.defense_mode and self.settings.alert_trend_flips and a.get("kind") == "etf" and cur["market_up"] is not None and prev.get("market_up") is not None and cur["market_up"] != prev["market_up"]:
                    self.alert(sym, "trend", f"大盤 10 月均線{'翻多' if cur['market_up'] else '翻空'}", "防守模式：ETF 建議" + ("恢復滿倉" if cur["market_up"] else "減碼至空手或半倉"), "signal")
            self._prev_state[sym] = cur

    def _feed_alerts(self):
        for sym, fd in self.ticker_feeds.items():
            a = self.analyses.get(sym)
            if not a:
                continue
            key = f"{sym}:feeds"
            prev = self._prev_state.get(key, {})
            cg = fd.get("congress") or {}
            cur = {"cluster": bool(cg.get("cluster")), "insider_buys": (fd.get("insider_summary") or {}).get("buys", 0)}
            if self.settings.alert_congress_cluster and cur["cluster"] and not prev.get("cluster"):
                self.alert(sym, "congress", f"{a['name']}：3 位以上國會議員近 30 天都買進", f"近 90 天買 {cg.get('buys')} 筆／賣 {cg.get('sells')} 筆。純資訊，不是訊號。", "info")
            if self.settings.alert_insider_buys and cur["insider_buys"] and cur["insider_buys"] > prev.get("insider_buys", 0):
                self.alert(sym, "insider", f"{a['name']}：內部人公開市場買進", "Form 4 顯示新的內部人買進（加分提示）。", "info")
            self._prev_state[key] = cur

    def alert(self, sym: str, kind: str, title: str, body: str, level: str = "info"):
        rec = {"time": _utcnow().isoformat(), "symbol": sym, "kind": kind, "title": title, "body": body, "level": level}
        with self._lock:
            self.alerts.append(rec)
            self.alerts = self.alerts[-200:]
            try:
                self.alert_log.parent.mkdir(parents=True, exist_ok=True)
                self.alert_log.write_text(json.dumps(self.alerts, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        s = self.settings
        text = f"{title}\n{body}"
        if s.telegram_bot_token and s.telegram_chat_id:
            threading.Thread(target=_post, args=(f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage", {"chat_id": s.telegram_chat_id, "text": text}), daemon=True).start()
        if s.discord_webhook_url:
            threading.Thread(target=_post, args=(s.discord_webhook_url, {"content": f"**{title}**\n{body}"}), daemon=True).start()

    # ------------------------------------------------------------------- queries
    def overview(self, market: str) -> dict:
        with self._lock:
            cards = []
            for sym, a in self.analyses.items():
                if a.get("market") != market:
                    continue
                fd = self.ticker_feeds.get(sym, {})
                badges = []
                if (fd.get("congress") or {}).get("cluster"):
                    badges.append("國會集群買進")
                if (fd.get("insider_summary") or {}).get("buys"):
                    badges.append("內部人買進")
                rev = fd.get("revenue_history") or fd.get("revenue") or []
                if rev and rev[-1].get("yoy") is not None:
                    badges.append(f"月營收 YoY {rev[-1]['yoy']:+.0f}%")
                cards.append({"symbol": sym, "name": a["name"], "kind": a["kind"], "sector": a.get("sector", ""), "price": a["price"], "chg_1d": a["chg_1d"],
                              "ret": a["ret"], "dd_52w": a["dd_from_high_52w"], "zone_min": a["zone_min"], "zone_max": a["zone_max"],
                              "zones": [{"method": z["method"], "zone": z["zone"], "label": z["label"]} for z in a["zones"]],
                              "verdict": a["verdict"]["stamp"], "verdict_cls": a["verdict"]["cls"], "verdict_title": a["verdict"]["title"],
                              "trend": a["trend"]["label"], "trend_state": a["trend"]["state"], "badges": badges, "last": a["last"], "source": a.get("source")})
            order = {"buy": 0, "buy-soft": 1, "hold": 2, "hold-warn": 3, "neutral": 4}
            cards.sort(key=lambda c: (order.get(c["verdict_cls"], 9), c["zone_min"] if c["zone_min"] is not None else 9))
            return {"market": market, "cards": cards, "status": self.status, "regime": self.market_regime(market), "generated": _utcnow().isoformat()}

    def market_regime(self, market: str) -> dict:
        out = {}
        for sym in SECTOR_INDEX[market]:
            df = self.frames.get(sym)
            if df is None or len(df) < 300:
                continue
            c = df["close"]
            me = c.index.to_period("M")
            is_me = np.r_[me[1:] != me[:-1], True]
            mclose = c[is_me]
            sma10 = mclose.rolling(10).mean()
            s200 = c.rolling(200).mean()
            out[sym] = {"name": DEFAULT_UNIVERSE[sym].name if sym in DEFAULT_UNIVERSE else ("加權指數" if sym == "^TWII" else sym),
                        "last": round(float(c.iloc[-1]), 2), "date": str(c.index[-1].date()),
                        "faber_up": bool(mclose.iloc[-2] > sma10.iloc[-2]) if len(mclose) > 10 and pd.notna(sma10.iloc[-2]) else None,
                        "live_above_10m": bool(c.iloc[-1] > sma10.iloc[-1]) if pd.notna(sma10.iloc[-1]) else None,
                        "above_200": bool(c.iloc[-1] > s200.iloc[-1]) if pd.notna(s200.iloc[-1]) else None,
                        "dd_52w": round(float(c.iloc[-1] / c.tail(252).max() - 1), 4), "ret_1y": round(float(c.iloc[-1] / c.iloc[-253] - 1), 4) if len(c) > 253 else None}
        return out

    def instrument(self, sym: str) -> dict:
        with self._lock:
            a = self.analyses.get(sym)
            if not a:
                return {"error": "unknown symbol"}
            out = dict(a)
            out["feeds"] = self.ticker_feeds.get(sym, {})
            out["holding"] = self.settings.my_holdings.get(sym)
            return out

    def chart(self, sym: str, tf: str = "1D", limit: int = 1500) -> dict:
        with self._lock:
            df = self.frames.get(sym)
            if df is None:
                return {"bars": [], "lines": {}}
            return chart_payload(df, self.pe_series.get(sym), tf, limit)

    def market_view(self, market: str) -> dict:
        with self._lock:
            out = {"market": market, "regime": self.market_regime(market), "news": self.news.items.get(market, [])[:60],
                   "news_agg": self.news.aggregate(self.news.items.get(market, [])), "macro": self.macro.snapshot() if not self.offline else [],
                   "alerts": self.alerts[-40:][::-1], "status": self.status, "feed_status": {**self.news.feed_status, **self.prices.status},
                   "congress_status": self.congress.status}
            today = datetime.now().date()
            fomc = [d for d in (self.macro.fomc_dates() if not self.offline else []) if d >= today][:3]
            out["calendar"] = {"fomc": [d.isoformat() for d in fomc], "tw_revenue_deadline": tw_revenue_deadline().isoformat()}
            if market == "US":
                tickers = [i.code for i in self.universe().values() if i.market == "US"]
                out["congress_recent"] = self.congress.recent(tickers, days=45, limit=60)
                out["congress_flow"] = self.congress.market_flow(12)
                out["congress_all_recent"] = self.congress.recent(None, days=14, limit=40)
            else:
                mi = self.twse.material_info() if not self.offline else pd.DataFrame()
                codes = {i.code for i in self.universe().values() if i.market == "TW"}
                if len(mi):
                    mi = mi[mi["code"].isin(codes)].sort_values("date", ascending=False).head(40)
                    out["material_info"] = [{"date": str(d.date()) if pd.notna(d) else None, "code": c, "name": n, "subject": s}
                                            for d, c, n, s in zip(mi["date"], mi["code"], mi["name"], mi["subject"])]
                ins = self.twse._history("insider_transfers")
                if len(ins):
                    g = ins[ins["code"].isin(codes)].sort_values("published", ascending=False).head(30)
                    out["insider_transfers"] = [{"published": str(p.date()) if pd.notna(p) else None, "code": c, "name": n, "who": w, "role": r, "shares": None if pd.isna(s) else float(s)}
                                                for p, c, n, w, r, s in zip(g["published"], g["code"], g["name"], g["who"], g["role"], g["shares"])]
            return out

    # ------------------------------------------------------------------ settings
    def update_settings(self, data: dict) -> dict:
        with self._lock:
            before = (list(self.settings.watch_tw), list(self.settings.watch_us), list(self.settings.markets))
            self.settings.update(data)
            self.settings.save(self.root / "settings.json")
            self.finmind.token = self.settings.finmind_token
            self.prices.max_age = timedelta(hours=self.settings.price_refresh_hours)
        if before != (list(self.settings.watch_tw), list(self.settings.watch_us), list(self.settings.markets)):
            threading.Thread(target=self._refresh_all, daemon=True).start()
        return self.settings.public()

    def _refresh_all(self):
        try:
            self.refresh_prices()
            self.refresh_feeds()
        except Exception as e:  # noqa: BLE001
            log.exception("refresh after settings: %s", e)

    def add_symbol(self, market: str, symbol: str, name: str = "", kind: str = "") -> dict:
        symbol = symbol.strip().upper()
        if market == "TW" and "." not in symbol:
            symbol = f"{symbol}.TW"
        lst = self.settings.watch_tw if market == "TW" else self.settings.watch_us
        if symbol not in lst:
            lst.append(symbol)
        if name or kind:
            self.settings.custom_names[symbol] = {"name": name or symbol, "kind": kind or "stock"}
        self.settings.save(self.root / "settings.json")
        threading.Thread(target=self._refresh_all, daemon=True).start()
        return {"ok": True, "symbol": symbol}

    def remove_symbol(self, market: str, symbol: str) -> dict:
        lst = self.settings.watch_tw if market == "TW" else self.settings.watch_us
        if symbol in lst:
            lst.remove(symbol)
        self.settings.save(self.root / "settings.json")
        with self._lock:
            self.analyses.pop(symbol, None)
        return {"ok": True}

    @staticmethod
    def _load_json(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except Exception:  # noqa: BLE001
            return None


def _pooled_for(pooled: dict) -> dict:
    """{method: {zone: {...}}} for the 12-month horizon (used by the verdict text)."""
    out = {}
    for method in ("channel", "deviation", "pe"):
        tbl = (pooled.get(method) or {}).get("252") or {}
        if tbl:
            out[method] = tbl
    return out


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict(orient="records"):
        rec = {}
        for k, v in r.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                rec[k] = None if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")
            elif isinstance(v, float) and np.isnan(v):
                rec[k] = None
            else:
                rec[k] = v
        out.append(rec)
    return out


def _insider_summary(f4: pd.DataFrame) -> dict:
    if f4 is None or len(f4) == 0:
        return {"buys": 0, "sells": 0, "buy_value": 0.0, "sell_value": 0.0, "buyers": 0}
    b = f4[f4["code"] == "P"]
    s = f4[f4["code"] == "S"]
    return {"buys": int(len(b)), "sells": int(len(s)), "buy_value": float(b["value"].fillna(0).sum()), "sell_value": float(s["value"].fillna(0).sum()),
            "buyers": int(b["owner"].nunique())}


def _post(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:  # noqa: BLE001
        log.warning("notify failed: %s", e)
