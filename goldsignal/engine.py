"""Live engine: keeps market data fresh, runs the strategy on every new 15m bar and
builds the snapshot the web UI displays. One Engine per instrument (gold, silver);
a Hub owns the shared settings, news/calendar monitor and alert stream.

A replay source lets the whole app run offline on historical data
(practice / weekends / testing)."""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import chartlab
from . import indicators as ind
from . import strategy as S
from .bars import resample
from .datafeed import Dukascopy, SwissquoteQuote, to_m15
from .instruments import INSTRUMENTS, Instrument
from .news import NewsMonitor
from .plans import eod_decision, lots_table, size_position
from .settings import Settings

log = logging.getLogger("goldsignal.engine")

REPLAY_DAYS = 45          # history replayed to reconstruct the current trade state
H1_MONTHS = 12            # hourly history for daily/4h indicator warm-up
NY = "America/New_York"


def _iso(ts) -> str | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _f(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)


def _trend_val(close, e20, e50) -> int:
    try:
        if close > e20 > e50:
            return 1
        if close < e20 < e50:
            return -1
    except TypeError:
        pass
    return 0


REASON_WORD = {"stop": "停損", "trail": "移動停損", "eod": "收盤前平倉", "target": "分批止盈", "end": "-"}


def _hm(h: float | None) -> str | None:
    """12.0 -> '12:00', 16.75 -> '16:45' (New York clock)."""
    if h is None:
        return None
    m = int(round(float(h) * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


def market_status(now: datetime) -> dict:
    ny = pd.Timestamp(now).tz_convert(NY)
    wd, h = ny.dayofweek, ny.hour + ny.minute / 60
    closed = (wd == 5) or (wd == 6 and h < 18) or (wd == 4 and h >= 17) or (17 <= h < 18 and wd < 5)
    if closed:
        session = "休市"
    elif h >= 18 or h < 3:
        session = "亞洲盤"
    elif h < 8:
        session = "倫敦盤"
    elif h < 12:
        session = "倫敦/紐約重疊"
    else:
        session = "紐約盤"
    return {"open": not closed, "session": session, "ny_time": ny.strftime("%a %H:%M")}


# =============================================================== data sources
class LiveSource:
    name = "live"

    def __init__(self, cache_dir: Path, inst: Instrument):
        self.inst = inst
        self.feed = Dukascopy(cache_dir / "dukascopy")
        self.quote_src = SwissquoteQuote(inst.swissquote)
        self.m1 = pd.DataFrame()
        self.h1_hist = pd.DataFrame()
        self.inter: dict[str, pd.DataFrame] = {}
        self.last_quote: dict | None = None
        self.progress = {"done": 0, "total": 0, "stage": ""}

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def bootstrap(self):
        self.feed.check(self.inst.code)
        today = self.now().date()
        self.progress = {"done": 0, "total": H1_MONTHS + REPLAY_DAYS + 1, "stage": "下載歷史資料"}
        self.h1_hist = self.feed.hours(self.inst.code, H1_MONTHS)
        self.progress["done"] = H1_MONTHS

        def prog(k, total):
            self.progress["done"] = H1_MONTHS + k

        self.m1 = self.feed.minutes(self.inst.code, today - timedelta(days=REPLAY_DAYS), today, progress=prog)
        if len(self.m1) < 5000:
            raise RuntimeError("downloaded too little history")
        self.progress["stage"] = "完成"

    def update(self):
        """Refresh today's (and, right after midnight UTC, yesterday's) minute bars."""
        now = self.now()
        days = [now.date()]
        if now.hour == 0 and now.minute < 10:
            days.insert(0, now.date() - timedelta(days=1))
        frames = [f for f in (self.feed.minute_day(self.inst.code, d) for d in days) if not f.empty]
        if frames:
            m1 = pd.concat([self.m1] + frames)
            m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
            self.m1 = m1[m1.index >= pd.Timestamp(now - timedelta(days=REPLAY_DAYS + 2))]

    def quote(self) -> dict | None:
        q = self.quote_src.get()
        if q:
            self.last_quote = q
        elif not self.m1.empty:  # fall back to the last Dukascopy candle
            last = float(self.m1["close"].iloc[-1])
            sp = round(last * self.inst.spread_pct, self.inst.decimals)
            self.last_quote = {"bid": last, "ask": last + sp, "spread": sp, "source": "Dukascopy",
                               "ts": int(self.m1.index[-1].timestamp() * 1000) + 60000}
        return self.last_quote

    def update_intermarket(self):
        today = self.now().date()
        for code in self.inst.intermarket:
            try:
                frames = [self.feed.minute_day(code, today - timedelta(days=k)) for k in (3, 2, 1, 0)]
                frames = [f for f in frames if not f.empty]
                if frames:
                    self.inter[code] = pd.concat(frames).sort_index()
            except Exception as e:
                log.debug("intermarket %s: %s", code, e)


class ReplaySource:
    """Plays historical 15m bars forward. `speed` = seconds of wall time per 15m bar."""
    name = "replay"

    def __init__(self, csv_path: Path, inst: Instrument, start: str | None = None, speed: float = 2.0):
        self.inst = inst
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        self.all = df[["open", "high", "low", "close"]].astype(float)
        self.all["volume"] = 0.0
        if start:
            pos = int(self.all.index.searchsorted(pd.Timestamp(start, tz="UTC")))
        else:
            pos = len(self.all) - 2000
        self.cursor = max(pos, min(3000, len(self.all) - 1))
        self.speed = speed
        self.paused = False
        self._last_step = time.time()
        self.h1_hist = pd.DataFrame()
        self.m1 = pd.DataFrame()
        self.inter = {}
        self.last_quote = None
        self.progress = {"done": 1, "total": 1, "stage": "回放模式"}

    def now(self) -> datetime:
        return (self.all.index[self.cursor - 1] + pd.Timedelta(minutes=15)).to_pydatetime()

    def bootstrap(self):
        pass

    def update(self):
        if self.paused:
            self._last_step = time.time()
            return
        steps = int((time.time() - self._last_step) / max(self.speed, 0.05))
        if steps > 0:
            self.cursor = min(self.cursor + steps, len(self.all))
            self._last_step = time.time()

    def m15(self) -> pd.DataFrame:
        lo = max(0, self.cursor - (REPLAY_DAYS + 220) * 96)
        return self.all.iloc[lo:self.cursor]

    def quote(self):
        last = self.all.iloc[self.cursor - 1]
        sp = round(float(last["close"]) * self.inst.spread_pct, self.inst.decimals)
        self.last_quote = {"bid": float(last["close"]), "ask": float(last["close"]) + sp, "spread": sp,
                           "ts": int(self.now().timestamp() * 1000), "source": "回放"}
        return self.last_quote

    def update_intermarket(self):
        pass

    def jump(self, when: str):
        pos = int(self.all.index.searchsorted(pd.Timestamp(when, tz="UTC")))
        self.cursor = max(min(3000, len(self.all) - 1), min(pos, len(self.all)))


# =============================================================== engine (one instrument)
class Engine:
    def __init__(self, hub: "Hub", inst: Instrument, replay_csv: Path | None = None,
                 replay_start: str | None = None, replay_speed: float = 2.0):
        self.hub = hub
        self.inst = inst
        self.src = ReplaySource(replay_csv, inst, replay_start, replay_speed) if replay_csv else LiveSource(hub.cache, inst)
        self.lock = threading.RLock()
        self.core: dict = {}
        self.m15: pd.DataFrame = pd.DataFrame()
        self.feat: pd.DataFrame = pd.DataFrame()
        self.trades: pd.DataFrame = pd.DataFrame()
        self.status = {"ok": False, "error": None, "last_update": None, "phase": "starting"}
        self._last_bar = None
        self._last_data = 0.0
        self._last_boundary = None
        self._last_inter = 0.0
        self._sent: set = set()
        self._stop = threading.Event()

    @property
    def settings(self) -> Settings:
        return self.hub.settings

    @property
    def params(self) -> S.StrategyParams:
        """Strategy parameters, including the day-mode rules when 當日平倉模式 is on."""
        return self.inst.strategy_params(bool(self.settings.day_mode))

    @property
    def iset(self) -> dict:
        return self.hub.settings.inst(self.inst.key)

    @property
    def news(self) -> NewsMonitor:
        return self.hub.news

    # ------------------------------------------------------------ lifecycle
    def start(self):
        threading.Thread(target=self._run, daemon=True, name=f"engine-{self.inst.key}").start()

    def stop(self):
        self._stop.set()

    def _run(self):
        booted = False
        while not self._stop.is_set():
            t0 = time.time()
            if not booted:
                try:
                    self.status["phase"] = "bootstrap"
                    self.src.bootstrap()
                    booted = True
                    self.status.update(phase="running", error=None)
                except Exception as e:
                    log.warning("%s bootstrap failed: %s", self.inst.key, e)
                    self.status.update(ok=False, phase="error",
                                       error="無法下載行情資料（Dukascopy）。請確認網路連線；30 秒後自動重試。")
                    self._stop.wait(30)
                    continue
            try:
                self._tick()
                self.status.update(ok=True, error=None, last_update=_iso(datetime.now(timezone.utc)))
            except Exception as e:
                log.warning("%s tick failed: %s", self.inst.key, e)
                self.status.update(ok=False, error=f"更新行情失敗：{str(e)[:160]}（會自動重試）")
            wait = 1.0 if isinstance(self.src, ReplaySource) else 10.0
            self._stop.wait(max(0.5, wait - (time.time() - t0)))

    def _boundary_due(self) -> bool:
        """Fetch right after each 15-minute boundary so new bars are processed promptly."""
        now = datetime.now(timezone.utc)
        b = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        if (now - b).total_seconds() >= 22 and self._last_boundary != b:
            self._last_boundary = b
            return True
        return False

    def _tick(self):
        src = self.src
        now = time.time()
        if isinstance(src, ReplaySource) or now - self._last_data >= 20 or self._boundary_due():
            src.update()
            self._last_data = now
        src.quote()
        if isinstance(src, LiveSource) and now - self._last_inter > 60:
            self._last_inter = now
            threading.Thread(target=src.update_intermarket, daemon=True).start()
        # a 15m bar counts as closed 20 s after its end, so its last 1-minute candle has arrived
        m15 = src.m15() if isinstance(src, ReplaySource) else to_m15(src.m1, now=src.now() - timedelta(seconds=20))
        if m15.empty or len(m15) < 200:
            return
        lb = m15.iloc[-1]
        key = (m15.index[-1], float(lb["high"]), float(lb["low"]), float(lb["close"]))
        if key != self._last_bar:
            self._recompute(m15)
            self._last_bar = key
        self._live_checks()

    # ------------------------------------------------------------ strategy
    def _blackout(self, index: pd.DatetimeIndex) -> np.ndarray:
        if isinstance(self.src, ReplaySource):  # the calendar only covers the current weeks
            return np.zeros(len(index), dtype=bool)
        s = self.settings
        return self.news.blackout_mask(index, before_min=s.blackout_before_min, after_min=s.blackout_after_min,
                                       fomc_after_min=s.fomc_after_min)

    def _recompute(self, m15: pd.DataFrame):
        p, inst = self.params, self.inst
        h1_hist = getattr(self.src, "h1_hist", pd.DataFrame())
        feat = S.compute_features(m15, h1_hist if not h1_hist.empty else None, p)
        px = m15["open"].to_numpy(float)
        blackout = self._blackout(m15.index)
        trades, feat, rules = S.run(m15, feat, p, px * inst.spread_pct, px * inst.slip_pct, blackout, swap=True)
        prev = self.core
        core = self._build_core(m15, feat, rules, trades, blackout)
        with self.lock:
            self.m15, self.feat, self.trades, self.core = m15, feat, trades, core
        self._emit_transition_alerts(prev, core)

    def _build_core(self, m15, f, t, trades, blackout) -> dict:
        p, d = self.params, self.inst.decimals
        n = len(m15)
        last = m15.iloc[-1]
        fl = f.iloc[-1]
        tl = t.iloc[-1]
        bar_close_time = m15.index[-1] + pd.Timedelta(minutes=15)
        atr1 = float(fl["h1_atr"])
        state = "flat"
        active = None
        if not trades.empty:
            lt = trades.iloc[-1]
            if lt["reason"] == "end" and int(lt["exit_i"]) == n - 1:
                state = "long" if lt["dir"] > 0 else "short"
                held_bars = n - 1 - int(lt["sig_i"])
                active = {
                    "dir": int(lt["dir"]), "signal_time": _iso(lt["signal_time"]), "entry_time": _iso(lt["entry_time"]),
                    "entry": _f(lt["entry"], d), "init_stop": _f(lt["init_stop"], d), "stop": _f(lt["stop_last"], d),
                    "risk": _f(lt["risk"], d), "best_R": _f(lt["mfeR"]), "bars_held": held_bars,
                    "hours_held": round(held_bars / 4, 1),
                }
        closed = trades[(trades["reason"] != "end") | (trades["exit_i"] < n - 1)] if not trades.empty else trades
        cooldown_until = None
        if state == "flat" and not closed.empty:
            ex_i = int(closed.iloc[-1]["exit_i"])
            if n - 1 < ex_i + p.cooldown_bars + 1:
                cooldown_until = _iso(m15.index[min(ex_i + p.cooldown_bars + 1, n - 1)] + pd.Timedelta(minutes=15))

        new_signal = None
        if state == "flat" and cooldown_until is None and (tl["long_signal"] or tl["short_signal"]):
            dr = 1 if tl["long_signal"] else -1
            c = float(last["close"])
            stop = c - dr * p.stop_atr_h1 * atr1
            new_signal = {"dir": dr, "time": _iso(bar_close_time), "ref_price": _f(c, d), "stop": _f(stop, d),
                          "risk": _f(abs(c - stop), d), "valid_minutes": 30, "max_chase": _f(c + dr * 0.5 * atr1, d)}
            state = "signal_long" if dr > 0 else "signal_short"

        fmt = f"{{:.{d}f}}"
        if p.day_mode:
            timing_label = f"交易時段允許（美東 {_hm(p.day_cutoff_ny)} 後到收盤不開新倉）"
            timing_detail = "當日平倉模式：留時間給 16:45 收盤前檢查；週五 12:00 後也不進場"
        else:
            timing_label, timing_detail = "交易時段允許（週五午後不進場）", ""

        def checks(side: str) -> list[dict]:
            sgn = 1 if side == "long" else -1
            word = "多" if side == "long" else "空"
            out = []
            for key, label, pre in (("d1", "日線", "d1"), ("h4", "4小時", "h4"), ("h1", "1小時", "h1")):
                if key == "h4" and not p.use_h4 or key == "h1" and not p.use_h1:
                    continue
                out.append({"key": key, "label": f"{label}趨勢偏{word}", "ok": bool(tl[f"{side}_{key}"]),
                            "detail": f"收 {fmt.format(fl[pre + '_close'])} / EMA20 {fmt.format(fl[pre + '_ema20'])} / EMA50 {fmt.format(fl[pre + '_ema50'])}"})
            out += [
                {"key": "adx", "label": f"未過熱（1小時 ADX ≤ {p.adx_max:g}）", "ok": bool(tl["adx_ok"]),
                 "detail": f"ADX = {fl['h1_adx']:.1f}"},
                {"key": "ext", "label": f"未偏離日均線太遠（≤ {p.ext_max:g} 日ATR）", "ok": bool(tl[f"{side}_ext"]),
                 "detail": f"偏離 = {sgn * fl['d1_ext']:.2f} 倍日ATR"},
                {"key": "break", "label": f"15分K 收盤{'突破' if side == 'long' else '跌破'} {p.breakout_hours} 小時{'高' if side == 'long' else '低'}點",
                 "ok": bool(tl[f"{side}_break"]),
                 "detail": f"最新收盤 {fmt.format(last['close'])}，需{'高於' if side == 'long' else '低於'} {fmt.format(fl['chan_hi' if side == 'long' else 'chan_lo'])}"},
                {"key": "close", "label": f"突破K棒收在強勢位置（≥ {p.close_pos_min * 100:.0f}%）", "ok": bool(tl[f"{side}_close"]),
                 "detail": f"收盤位置 {100 * (fl['cpos_long'] if side == 'long' else fl['cpos_short']):.0f}%"},
                {"key": "timing", "label": timing_label, "ok": bool(tl["timing_ok"]), "detail": timing_detail},
                {"key": "news", "label": "無重大數據公布前後禁區", "ok": bool(tl["news_ok"]), "detail": ""},
            ]
            return out

        nb = p.breakout_hours * 4
        hi_n = float(m15["high"].iloc[-nb:].max())
        lo_n = float(m15["low"].iloc[-nb:].min())
        setup = {}
        for side in ("long", "short"):
            ck = checks(side)
            trend_ok = all(c["ok"] for c in ck if c["key"] not in ("break", "close"))
            setup[side] = {"checks": ck, "ready": trend_ok, "trigger": _f(hi_n if side == "long" else lo_n, d),
                           "distance": _f((hi_n - float(last["close"])) if side == "long" else (float(last["close"]) - lo_n), d)}

        mtf = []
        for tf, pre in (("1D", "d1"), ("4H", "h4"), ("1H", "h1")):
            mtf.append({"tf": tf, "trend": _trend_val(fl[f"{pre}_close"], fl[f"{pre}_ema20"], fl[f"{pre}_ema50"]),
                        "close": _f(fl[f"{pre}_close"], d), "ema20": _f(fl[f"{pre}_ema20"], d), "ema50": _f(fl[f"{pre}_ema50"], d),
                        "rsi": _f(fl.get(f"{pre}_rsi"), 1), "adx": _f(fl.get(f"{pre}_adx"), 1), "atr": _f(fl.get(f"{pre}_atr"), d)})
        mtf.append({"tf": "15m", "trend": _trend_val(last["close"], fl["m15_ema20"], fl["m15_ema50"]),
                    "close": _f(last["close"], d), "ema20": _f(fl["m15_ema20"], d), "ema50": _f(fl["m15_ema50"], d),
                    "rsi": _f(fl["m15_rsi"], 1), "adx": None, "atr": _f(fl["m15_atr"], d)})

        hist = []
        if not trades.empty:
            for _, r in trades.tail(12).iloc[::-1].iterrows():
                is_open = r["reason"] == "end" and int(r["exit_i"]) == n - 1
                hist.append({"dir": int(r["dir"]), "entry_time": _iso(r["entry_time"]),
                             "exit_time": None if is_open else _iso(r["exit_time"] + pd.Timedelta(minutes=15)),
                             "entry": _f(r["entry"], d), "exit": None if is_open else _f(r["exit"], d), "R": None if is_open else _f(r["R"]),
                             "reason": "持倉中" if is_open else REASON_WORD.get(r["reason"], r["reason"])})
        return {
            "bar_time": _iso(m15.index[-1]), "bar_close_time": _iso(bar_close_time),
            "state": state, "active": active, "new_signal": new_signal, "cooldown_until": cooldown_until,
            "bias": int(fl["trend_d1"]), "setup": setup, "mtf": mtf, "history": hist,
            "atr_h1": _f(atr1, d + 1), "last_close": _f(last["close"], d),
            "blackout_now": bool(blackout[-1]) if len(blackout) else False,
            "params": p.to_dict(),
            "day": {"on": p.day_mode, "cutoff": _hm(p.day_cutoff_ny) if p.day_mode else None,
                    "eod_time": _hm(p.eod_time_ny), "keep_r": p.eod_keep_r},
        }

    # ------------------------------------------------------------ alerts
    def _alert(self, kind: str, title: str, body: str, level: str = "info"):
        self.hub.alert(self.inst, kind, title, body, level)

    def _emit_transition_alerts(self, prev: dict, core: dict):
        if not prev:
            return
        name = self.inst.name
        ns = core.get("new_signal")
        if ns and not prev.get("new_signal"):
            word = "做多" if ns["dir"] > 0 else "做空"
            self._alert("signal", f"🔔 {name}{word}訊號", f"參考價 {ns['ref_price']}，停損 {ns['stop']}（風險 {ns['risk']}）。"
                        f"請在 {ns['valid_minutes']} 分鐘內、價格未超過 {ns['max_chase']} 前進場，否則放棄。", "signal")
        pa, ca = prev.get("active"), core.get("active")
        if pa and not ca:
            last = core["history"][0] if core.get("history") else None
            r = last["R"] if last else None
            res = f"，結果 {r:+.2f}R" if r is not None else ""
            if last and last.get("reason") == REASON_WORD["eod"]:
                keep = core.get("day", {}).get("keep_r") or 0
                why = f"獲利未達 {keep:+.2f}R" if keep > 0 else "還在虧損"
                self._alert("exit", f"⏰ {name}收盤前平倉", f"16:45 檢查時{'多單' if pa['dir'] > 0 else '空單'}{why}，"
                            f"系統平倉{res}。請在 17:00 前到 Mitrade 平倉。", "exit")
            else:
                self._alert("exit", f"🏁 {name}系統出場", f"{'多單' if pa['dir'] > 0 else '空單'}已觸及停損/移動停損出場{res}", "exit")
        if pa and ca and pa.get("stop") and ca.get("stop") and abs(ca["stop"] - pa["stop"]) >= 0.25 * (core.get("atr_h1") or 1):
            self._alert("stop_move", f"↕ {name}移動停損更新", f"新停損 {ca['stop']}（原 {pa['stop']}）", "info")

    def _live_checks(self):
        core = self.core
        q = self.src.last_quote
        if not core or not q:
            return
        a = core.get("active")
        if a and a.get("stop"):
            hit = (a["dir"] > 0 and q["bid"] <= a["stop"]) or (a["dir"] < 0 and q["ask"] >= a["stop"])
            key = ("stop_hit", a["entry_time"])
            if hit and key not in self._sent:
                self._sent.add(key)
                self._alert("stop_hit", f"⛔ {self.inst.name}觸及停損", f"即時價已觸及停損 {a['stop']}，系統視為出場。", "exit")
        self._eod_heads_up()
        if isinstance(self.src, ReplaySource):
            return
        now = datetime.now(timezone.utc)
        for ev in self.news.calendar:
            if ev["country"] == "USD" and ev["impact"] == "High":
                mins = (ev["time"] - now).total_seconds() / 60
                key = ("event", ev["time"].isoformat(), ev["title"])
                if 0 < mins <= 30 and key not in self._sent and (a or self.inst.key == "XAUUSD"):
                    self._sent.add(key)
                    extra = f"你的{self.inst.name}有持倉，可考慮確認停損或先減碼。" if a else "公布前後 30 分鐘不會產生新訊號。"
                    self._alert("event", f"📅 {int(mins)} 分鐘後：{ev['title']}", extra, "warn")

    def _eod_heads_up(self):
        """Day mode: one reminder per day between 16:25 and 16:45 New York if a position is open."""
        p = self.params
        if not p.day_mode:
            return
        ny = pd.Timestamp(self.now()).tz_convert(NY)
        mins = ny.hour * 60 + ny.minute
        if ny.dayofweek >= 5 or not (16 * 60 + 25 <= mins < 16 * 60 + 45):
            return
        key = ("eod", ny.date().isoformat())
        if key in self._sent:
            return
        view = self._eod_view(self.price())
        parts = []
        for who, label in (("system", "系統"), ("mine", "你的")):
            v = view.get(who)
            if not v:
                continue
            word = "多單" if v["dir"] > 0 else "空單"
            if v["keep"]:
                parts.append(f"{label}{word}目前 {v['R_now']:+.2f}R：可以留過夜，停損 {v['stop']}。")
            else:
                need = f"仍未到 {v['keep_r']:+.2f}R（{v['target_price']}）" if v["keep_r"] > 0 else f"還在虧損（沒回到 {v['target_price']}）"
                parts.append(f"{label}{word}目前 {v['R_now']:+.2f}R：16:45 若{need}就平倉。")
        if parts:
            self._sent.add(key)
            self._alert("eod", f"⏰ {self.inst.name} 16:45 收盤前檢查", " ".join(parts), "warn")

    def _eod_view(self, price: dict | None) -> dict:
        """Day mode: what the 16:45 New York check says about the open position(s)."""
        p, d = self.params, self.inst.decimals
        if not p.day_mode:
            return {}
        now_ny = pd.Timestamp(self.now()).tz_convert(NY)
        check = now_ny.normalize() + pd.Timedelta(hours=16, minutes=45)
        if now_ny >= check + pd.Timedelta(minutes=15):
            check += pd.Timedelta(days=1)
        while check.dayofweek >= 5:
            check += pd.Timedelta(days=1)
        out = {"time_ny": _hm(p.eod_time_ny), "check_time": _iso(check), "keep_r": p.eod_keep_r,
               "minutes": int((check - now_ny).total_seconds() // 60)}
        off = self.iset["price_offset"]
        a = self.core.get("active")
        if a and price and a.get("risk"):
            mark = (price["bid"] if a["dir"] > 0 else price["ask"]) - off
            v = eod_decision(a["dir"], a["entry"], a["risk"], mark, p.eod_keep_r)
            locked = (a["dir"] > 0 and a["stop"] >= a["entry"]) or (a["dir"] < 0 and a["stop"] <= a["entry"])
            v.update(dir=a["dir"], keep=v["keep"] or locked, locked=locked, stop=_f(a["stop"] + off, d),
                     target_price=_f(v["target_price"] + off, d))
            out["system"] = v
        mp = self.settings.my_positions.get(self.inst.key) or {}
        if mp.get("entry") and mp.get("dir") and not self.m15.empty and price:
            m = self._my_position(mp, price)
            if m.get("risk"):
                dr = m["dir"]
                mark = (price["bid"] if dr > 0 else price["ask"])
                v = eod_decision(dr, m["entry"], m["risk"], mark, p.eod_keep_r)
                locked = (dr > 0 and m["stop"] >= m["entry"]) or (dr < 0 and m["stop"] <= m["entry"])
                v.update(dir=dr, keep=v["keep"] or locked, locked=locked, stop=m["stop"], target_price=_f(v["target_price"], d))
                out["mine"] = v
        return out

    # ------------------------------------------------------------ views
    def now(self) -> datetime:
        return self.src.now() if isinstance(self.src, ReplaySource) else datetime.now(timezone.utc)

    def price(self) -> dict | None:
        q = self.src.last_quote
        if not q:
            return None
        off, d = self.iset["price_offset"], self.inst.decimals
        return {"bid": _f(q["bid"] + off, d), "ask": _f(q["ask"] + off, d), "spread": _f(q.get("spread"), d + 1),
                "source": q.get("source"),
                "time": _iso(pd.Timestamp(q["ts"], unit="ms", tz="UTC")) if q.get("ts") else None}

    def brief(self) -> dict:
        """Small summary used by the instrument tabs."""
        core = self.core
        p = self.price()
        return {"key": self.inst.key, "name": self.inst.name, "symbol": self.inst.symbol, "seal": self.inst.seal,
                "decimals": self.inst.decimals, "bid": p["bid"] if p else None, "state": core.get("state"),
                "bias": core.get("bias"), "ready_long": core.get("setup", {}).get("long", {}).get("ready"),
                "ready_short": core.get("setup", {}).get("short", {}).get("ready"), "error": self.status.get("error")}

    def snapshot(self) -> dict:
        with self.lock:
            core = dict(self.core)
        s, iset, inst, d = self.settings, self.iset, self.inst, self.inst.decimals
        now = self.now()
        price = self.price()
        out = {"now": _iso(now), "mode": self.src.name, "market": market_status(now), "status": dict(self.status),
               "progress": getattr(self.src, "progress", None), "price": price,
               "instrument": {"key": inst.key, "name": inst.name, "symbol": inst.symbol, "seal": inst.seal,
                              "decimals": d, "note": inst.note, "metal": inst.metal},
               "inst_settings": iset}
        out.update(core)
        if not self.m15.empty and price:
            ny_now = pd.Timestamp(now).tz_convert(NY)
            day_start = (ny_now - pd.Timedelta(hours=17)).normalize() + pd.Timedelta(hours=17)
            prev = self.m15[self.m15.index < day_start.tz_convert("UTC")]
            if not prev.empty:
                pc = float(prev["close"].iloc[-1]) + iset["price_offset"]
                out["day_change"] = {"abs": _f(price["bid"] - pc, d), "pct": _f(100 * (price["bid"] - pc) / pc, 2), "prev_close": _f(pc, d)}
        a = core.get("active")
        if a and price and a.get("risk"):
            mark = price["bid"] - iset["price_offset"] if a["dir"] > 0 else price["ask"] - iset["price_offset"]
            out["active"] = dict(a, R_now=_f((mark - a["entry"]) / a["risk"] * a["dir"]), mark=_f(mark + iset["price_offset"], d))
        atr1 = core.get("atr_h1")
        if atr1:
            spread = float(price["spread"]) if price and price.get("spread") else 0.0
            ns = core.get("new_signal")
            stop_dist = abs(ns["ref_price"] - ns["stop"]) if ns else self.params.stop_atr_h1 * atr1
            px = price["bid"] if price else (core.get("last_close") or 0) + iset["price_offset"]
            oz, min_lot = iset["oz_per_lot"], iset["min_lot"] or 0.01
            sz = size_position(s.account_balance, iset["risk_pct"], stop_dist, px, oz, min_lot, s.max_lots, s.leverage, spread)
            sz.update(stop_dist=_f(stop_dist + spread, d), risk_pct=iset["risk_pct"], oz_per_lot=oz, min_lot=min_lot,
                      confirmed=iset.get("confirmed", False), balance=s.account_balance, max_lots=s.max_lots, leverage=s.leverage)
            n2 = sum(1 for x in sz.get("legs", []) if x > 0)
            if n2 and ns:
                sz["scale_out"] = {"lots": round(n2 * min_lot, 2), "r": 2.0,
                                   "price": _f(ns["ref_price"] + ns["dir"] * 2.0 * stop_dist + iset["price_offset"], d)}
            out["sizing"] = sz
            out["lots_table"] = lots_table(s.account_balance, stop_dist, px, oz, spread, s.leverage)
        eod = self._eod_view(price)
        if eod:
            out["eod"] = eod
        mp = s.my_positions.get(inst.key) or {}
        if mp.get("entry") and mp.get("dir") and not self.m15.empty and atr1:
            out["my_position"] = self._my_position(mp, price)
        out["news"] = {"items": [dict(it, time=_iso(it["time"])) for it in self.news.items_for(inst.metal, 40)],
                       "aggregate": self.news.aggregate(inst.metal), "feeds": self.news.feed_status}
        out["calendar"] = self._calendar_view(now)
        out["intermarket"] = self._intermarket_view()
        if isinstance(self.src, ReplaySource):
            out["replay"] = {"speed": self.src.speed, "paused": self.src.paused, "cursor_time": _iso(self.src.now()),
                             "start": _iso(self.src.all.index[0]), "end": _iso(self.src.all.index[-1])}
        return out

    def _my_position(self, mp: dict, price: dict | None) -> dict:
        d, off = self.inst.decimals, self.iset["price_offset"]
        dr = int(mp["dir"])
        entry = float(mp["entry"]) - off
        t0 = pd.Timestamp(mp["time"]).tz_convert("UTC") if mp.get("time") else self.m15.index[-1]
        bars = self.m15[self.m15.index >= t0.floor("15min")]
        atr1 = self.feat["h1_atr"].reindex(bars.index).to_numpy() if not bars.empty else np.array([])
        init_risk = self.params.stop_atr_h1 * float(self.core.get("atr_h1") or 0)
        stop = entry - dr * init_risk
        best = entry
        for (_, row), a in zip(bars.iterrows(), atr1):
            best = max(best, row["high"]) if dr > 0 else min(best, row["low"])
            if not np.isnan(a):
                ts_ = best - dr * self.params.trail_atr_h1 * a
                stop = max(stop, ts_) if dr > 0 else min(stop, ts_)
        mark = ((price["bid"] if dr > 0 else price["ask"]) - off) if price else None
        pnl = (mark - entry) * dr if mark is not None else None
        lots = float(mp.get("lots") or 0)
        oz = lots * self.iset["oz_per_lot"]
        return {"dir": dr, "entry": _f(entry + off, d), "stop": _f(stop + off, d), "risk": _f(init_risk, d + 1),
                "R_now": _f(pnl / init_risk) if pnl is not None and init_risk > 0 else None,
                "pnl_usd": _f(pnl * oz) if pnl is not None else None, "lots": lots, "time": mp.get("time"),
                "risk_now_usd": _f(max(0.0, dr * (entry - stop)) * oz),
                "locked_usd": _f(max(0.0, dr * (stop - entry)) * oz)}

    def _calendar_view(self, now: datetime) -> dict:
        evs = []
        show = ("High", "Medium") if self.settings.show_medium_impact else ("High",)
        for e in self.news.calendar:
            if e["impact"] in show and e["country"] in ("USD", "CNY", "EUR") and e["time"] >= now - timedelta(hours=6):
                evs.append({"time": _iso(e["time"]), "title": e["title"], "country": e["country"], "impact": e["impact"],
                            "forecast": e["forecast"], "previous": e["previous"],
                            "minutes": round((e["time"] - now).total_seconds() / 60)})
        s = self.settings
        windows = self.news.blackout_windows(s.blackout_before_min, s.blackout_after_min, s.fomc_after_min)
        active = [w for w in windows if w[0] <= now <= w[1]] if not isinstance(self.src, ReplaySource) else []
        nxt = next((e for e in evs if e["minutes"] > 0 and e["impact"] == "High" and e["country"] == "USD"), None)
        return {"events": evs[:25], "blackout_now": bool(active), "blackout_title": active[0][2] if active else None, "next_high": nxt}

    def _intermarket_view(self) -> list[dict]:
        out = []
        inter = getattr(self.src, "inter", {})
        for code, name in self.inst.intermarket.items():
            df = inter.get(code)
            if df is None or df.empty:
                continue
            last = float(df["close"].iloc[-1])
            ny = df.index.tz_convert(NY)
            day_start = (ny[-1] - pd.Timedelta(hours=17)).normalize() + pd.Timedelta(hours=17)
            prev = df[ny < day_start]
            chg = 100 * (last / float(prev["close"].iloc[-1]) - 1) if not prev.empty else None
            h1 = resample(df, "1h")
            e20 = ind.ema(h1["close"], 20).iloc[-1] if len(h1) >= 20 else np.nan
            e50 = ind.ema(h1["close"], 50).iloc[-1] if len(h1) >= 50 else np.nan
            tr = _trend_val(h1["close"].iloc[-1], e20, e50) if len(h1) >= 50 else 0
            out.append({"code": code, "name": name, "last": _f(last, 3), "chg_pct": _f(chg, 2), "trend_1h": int(tr),
                        "spark": h1["close"].iloc[-48:].round(4).tolist()})
        return out

    def candles(self, tf: str = "15m", limit: int = 600) -> dict:
        with self.lock:
            m15, feat, trades, core = self.m15, self.feat, self.trades, self.core
        if m15.empty:
            return {"bars": []}
        d = self.inst.decimals
        h1_hist = getattr(self.src, "h1_hist", pd.DataFrame())
        live_m1 = self.src.m1 if isinstance(self.src, LiveSource) and not self.src.m1.empty else None
        partial = None
        if live_m1 is not None:
            tail = live_m1[live_m1.index >= m15.index[-1] + pd.Timedelta(minutes=15)]
            if not tail.empty:
                partial = tail
        recent = m15 if partial is None else pd.concat([m15, resample(partial, "15min")])
        recent = recent[~recent.index.duplicated(keep="last")].sort_index()
        if tf == "5m":
            if live_m1 is None:
                return {"tf": tf, "decimals": d, "bars": [], "note": "回放模式只有 15 分 K 以上的資料，5 分 K 只在即時模式顯示。"}
            bars = resample(live_m1[live_m1.index >= live_m1.index[-1] - pd.Timedelta(days=6)], "5min")
        elif tf == "15m":
            bars = recent
        else:
            base_h1 = resample(recent, "1h")
            if not h1_hist.empty:
                older = h1_hist[h1_hist.index < base_h1.index[0]]
                base_h1 = pd.concat([older[base_h1.columns.intersection(older.columns)], base_h1])
            bars = base_h1 if tf == "1h" else resample(base_h1, tf)
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        off = self.iset["price_offset"]
        e20 = ind.ema(bars["close"], 20)
        e50 = ind.ema(bars["close"], 50)
        bars = bars.iloc[-limit:]
        t = ((bars.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).tolist()
        out = {"tf": tf, "decimals": d,
               "bars": [{"time": int(ts), "open": _f(o + off, d), "high": _f(h + off, d), "low": _f(l + off, d), "close": _f(c + off, d)}
                        for ts, o, h, l, c in zip(t, bars["open"], bars["high"], bars["low"], bars["close"])],
               "ema20": [{"time": int(ts), "value": _f(v + off, d)} for ts, v in zip(t, e20.reindex(bars.index)) if not math.isnan(v)],
               "ema50": [{"time": int(ts), "value": _f(v + off, d)} for ts, v in zip(t, e50.reindex(bars.index)) if not math.isnan(v)]}
        if tf == "15m" and not feat.empty:
            ch = feat[["chan_hi", "chan_lo"]].reindex(bars.index)
            out["chan_hi"] = [{"time": int(ts), "value": _f(v + off, d)} for ts, v in zip(t, ch["chan_hi"]) if not math.isnan(v)]
            out["chan_lo"] = [{"time": int(ts), "value": _f(v + off, d)} for ts, v in zip(t, ch["chan_lo"]) if not math.isnan(v)]
        # ---- system trades: green arrow = long, red arrow = short
        marks = []
        t0 = int(bars.index[0].timestamp())
        if not trades.empty:
            for _, r in trades.iterrows():
                et = int(_floor_to_bar(pd.Timestamp(r["entry_time"]), bars.index).timestamp())
                if et >= t0:
                    marks.append({"time": et, "position": "belowBar" if r["dir"] > 0 else "aboveBar",
                                  "shape": "arrowUp" if r["dir"] > 0 else "arrowDown", "kind": "long" if r["dir"] > 0 else "short",
                                  "text": ("做多 " if r["dir"] > 0 else "做空 ") + f"{r['entry'] + off:.{d}f}"})
                is_open = r["reason"] == "end" and int(r["exit_i"]) == len(m15) - 1
                if not is_open:
                    xt = int(_floor_to_bar(pd.Timestamp(r["exit_time"]), bars.index).timestamp())
                    if xt >= t0:
                        marks.append({"time": xt, "position": "aboveBar" if r["dir"] > 0 else "belowBar",
                                      "shape": "circle", "kind": "exit", "text": f"出場 {r['R']:+.1f}R"})
        ns = core.get("new_signal") if core else None
        if ns:
            marks.append({"time": t[-1], "position": "belowBar" if ns["dir"] > 0 else "aboveBar",
                          "shape": "arrowUp" if ns["dir"] > 0 else "arrowDown", "kind": "long" if ns["dir"] > 0 else "short",
                          "text": "現在做多" if ns["dir"] > 0 else "現在做空"})
        # ---- reference tools
        price = self.price()
        mark_px = (price["bid"] - off) if price else float(bars["close"].iloc[-1])
        an = chartlab.analyze(bars, tf, mark_px)
        fmtp = lambda x: f"{x + off:,.{d}f}"  # noqa: E731
        ov = {"levels": [], "fib": None, "triangle": None}
        if an:
            ov["levels"] = [{"price": _f(x["price"] + off, d), "kind": x["kind"],
                             "title": ("壓力 " if x["kind"] == "resistance" else "支撐 ") + x["label"]} for x in an["levels"]]
            fb = an.get("fib")
            if fb:
                ov["fib"] = {"t0": t[fb["start_i"]], "t1": t[-1], "dir": fb["dir"],
                             "levels": [{"ratio": x["ratio"], "price": _f(x["price"] + off, d)} for x in fb["levels"]]}
            tri = an.get("triangle")
            if tri:
                ov["triangle"] = {"name": tri["name"], "breakout": tri["breakout"],
                                  "upper": [{"time": t[i], "value": _f(v + off, d)} for i, v in tri["upper"]],
                                  "lower": [{"time": t[i], "value": _f(v + off, d)} for i, v in tri["lower"]]}
                if tri["breakout"]:
                    up_ = tri["breakout"] == "up"
                    marks.append({"time": t[tri["break_i"]], "position": "belowBar" if up_ else "aboveBar", "shape": "circle",
                                  "kind": "pattern", "text": "突破三角形" if up_ else "跌破三角形"})
        out["overlays"] = ov
        out["markers"] = sorted(marks, key=lambda m: m["time"])
        a = core.get("active") if core else None
        if a:
            out["lines"] = [{"price": _f(a["entry"] + off, d), "title": "進場", "kind": "entry"},
                            {"price": _f(a["stop"] + off, d), "title": "停損", "kind": "stop", "dir": a["dir"]}]
        elif ns:
            out["lines"] = [{"price": _f(ns["ref_price"] + off, d), "title": "訊號價", "kind": "entry"},
                            {"price": _f(ns["stop"] + off, d), "title": "停損", "kind": "stop", "dir": ns["dir"]}]
        elif core and not core.get("cooldown_until"):
            bias = core.get("bias", 0)
            side = "long" if bias > 0 else "short" if bias < 0 else None
            st = (core.get("setup") or {}).get(side) if side else None
            if st and st.get("ready") and st.get("trigger") is not None:
                out["lines"] = [{"price": _f(st["trigger"] + off, d), "title": "站上做多" if bias > 0 else "跌破做空",
                                 "kind": "trigger", "dir": bias}]
        mp = self.settings.my_positions.get(self.inst.key) or {}
        pos_dir = int(a["dir"]) if a else int(mp.get("dir") or 0)
        stop = a["stop"] if a else None
        tf_label = {"5m": "5 分 K", "15m": "15 分 K", "1h": "1 小時 K", "4h": "4 小時 K", "1D": "日 K"}.get(tf, tf)
        out["analysis"] = chartlab.describe(an, core, fmtp, tf_label, self.inst.name, pos_dir, stop)
        return out


def _floor_to_bar(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Map a timestamp to the open time of the bar (in `index`) that contains it."""
    pos = index.searchsorted(ts, side="right") - 1
    return index[max(pos, 0)]


# =============================================================== hub
class Hub:
    def __init__(self, root: Path, symbols: list[str] | None = None, replay_csvs: dict | None = None,
                 replay_start: str | None = None, replay_speed: float = 2.0):
        self.root = Path(root)
        self.cache = self.root / "data" / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.root / "settings.json"
        self.settings = Settings.load(self.settings_path)
        self.news = NewsMonitor(self.cache)
        self.alerts: deque = deque(maxlen=300)
        self._alert_id = 0
        self._alert_lock = threading.Lock()
        self._stop = threading.Event()
        syms = symbols or list(INSTRUMENTS)
        self.engines: dict[str, Engine] = {}
        for k in syms:
            csv = (replay_csvs or {}).get(k)
            self.engines[k] = Engine(self, INSTRUMENTS[k], replay_csv=csv, replay_start=replay_start, replay_speed=replay_speed)
        self.replay = bool(replay_csvs)

    def start(self):
        for e in self.engines.values():
            e.start()
        threading.Thread(target=self._news_loop, daemon=True, name="news").start()

    def stop(self):
        self._stop.set()
        for e in self.engines.values():
            e.stop()

    def _news_loop(self):
        while not self._stop.is_set():
            try:
                self.news.refresh_calendar()
                self.news.refresh_news()
            except Exception as e:
                log.warning("news refresh failed: %s", e)
            self._stop.wait(300)

    def alert(self, inst: Instrument, kind: str, title: str, body: str, level: str = "info"):
        with self._alert_lock:
            self._alert_id += 1
            a = {"id": self._alert_id, "time": _iso(datetime.now(timezone.utc)), "sym": inst.key if inst else None,
                 "kind": kind, "title": title, "body": body, "level": level}
            self.alerts.append(a)
        s = self.settings
        if kind in ("signal", "exit", "stop_hit", "eod", "test"):
            text = f"{title}\n{body}"
            if s.telegram_bot_token and s.telegram_chat_id:
                threading.Thread(target=self._post, args=(f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage",
                                                          {"chat_id": s.telegram_chat_id, "text": text}), daemon=True).start()
            if s.discord_webhook_url:
                threading.Thread(target=self._post, args=(s.discord_webhook_url, {"content": f"**{title}**\n{body}"}), daemon=True).start()

    @staticmethod
    def _post(url: str, payload: dict):
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            log.warning("notification failed: %s", e)

    def state(self, sym: str) -> dict:
        e = self.engines.get(sym) or next(iter(self.engines.values()))
        out = e.snapshot()
        out["instruments"] = [x.brief() for x in self.engines.values()]
        out["alerts"] = list(self.alerts)[-40:]
        out["settings"] = self.settings.public()
        return out

    def update_settings(self, data: dict) -> dict:
        self.settings.update(data)
        self.settings.save(self.settings_path)
        for e in self.engines.values():  # e.g. day mode toggled: recompute on the next tick
            e._last_bar = None
        return self.settings.public()

    def set_my_position(self, sym: str, data: dict | None) -> None:
        if data:
            self.settings.my_positions[sym] = data
        else:
            self.settings.my_positions.pop(sym, None)
        self.settings.save(self.settings_path)

    def replay_control(self, data: dict) -> None:
        for e in self.engines.values():
            if not isinstance(e.src, ReplaySource):
                continue
            if "speed" in data:
                e.src.speed = max(0.05, float(data["speed"]))
            if "paused" in data:
                e.src.paused = bool(data["paused"])
            if "jump" in data:
                e.src.jump(data["jump"])
                e._last_bar = None
