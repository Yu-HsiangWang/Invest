"""More trades, but at least as efficient as the original rules (docs/RESEARCH.md section 11) -
the study behind 高效積極模式 (goldsignal/active.py).

The first 積極模式 (12-hour breakouts, every signal its own position) doubled the number of
trades but halved the profit per trade and tripled the drawdown. This script looks for ways
to trade more that do NOT dilute the edge. Efficiency = profit per year / max drawdown (both in
R), the same yardstick for every variant.

Ideas, fixed before running:
  * pyramiding: open a second (third) position only while the open ones are already safe -
    their trailing stop has reached the entry price ("locked") or they are >= +1R in profit;
    the first position still needs the full original rules;
  * a shorter channel / no squeeze filter only for those add-on positions;
  * the 16-hour channel with the squeeze filter (the most efficient faster variant earlier);
  * then (second run) its neighbourhood: 14-20 hour channels and the squeeze settings
    (robustness()), and full-size squeeze signals plus reduced-size other signals (tiered()).

The channel length of 高效積極模式 (18 hours) was picked on the design period 2005-2016 alone
(the best dev efficiency of 14/16/18/20 hours); 2017-2025 and the Dukascopy data
(research/efficient_dukascopy.py) are the check.

Gold 15m 2005-2025/3 (dev 2005-2016 / val 2017-2020 / test 2021-2025/3), day mode, costs and
overnight fees as everywhere else. Every position is managed exactly like a normal trade
(goldsignal.plans.simulate_plan); only the decision to open it changes.

    python research/efficient_active.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock, simulate_plan  # noqa: E402

PERIODS = {"dev": ("2005-01-01", "2016-12-31"), "val": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2025-03-21")}
OUT = ROOT / "results"


def load(path: str | None = None) -> pd.DataFrame:
    pq = Path(path) if path else ROOT / "data" / "cache_xau15.parquet"
    return pd.read_parquet(pq).loc[:"2025-03-21"]


# ----------------------------------------------------------------------------- signals

def signal_sets(df: pd.DataFrame, sym: str = "XAUUSD", hours=(24, 16, 12, 8), blackout=None) -> dict:
    """{(hours, squeeze): (sig array, features)} with the instrument's day-mode rules."""
    inst = INSTRUMENTS[sym]
    out = {}
    for h in hours:
        p0 = replace(inst.strategy_params(True, squeeze=False), breakout_hours=int(h))
        f = S.compute_features(df, params=p0)
        for sq in (False, True):
            p = replace(p0, squeeze_bars=S.SQUEEZE_BARS if sq else 0)
            out[(h, sq)] = (S.signals_from_rules(S.rule_table(df, f, p, blackout)), f, p)
    return out


# ----------------------------------------------------------------------------- simulator

class Book:
    """Positions opened one at a time; each one is an independent simulate_plan trade."""

    def __init__(self, df, atr1, plan, spread, slip, clock):
        self.df, self.atr1, self.plan = df, atr1, plan
        self.sp, self.sl, self.clock = spread, slip, clock
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.n = len(df)
        self.rows: list[dict] = []

    def open(self, i: int, d: int) -> dict | None:
        one = np.zeros(self.n, dtype=np.int64)
        one[i] = d
        tr = simulate_plan(self.df, one, self.atr1, self.plan, self.sp, self.sl, self.clock)
        if tr.empty:
            return None
        r = tr.iloc[0].to_dict()
        r["j0"] = int(r["sig_i"]) + 1
        self.rows.append(r)
        return r

    def stop_after(self, p: dict, i: int) -> float:
        """The trailing stop of open position p after the close of bar i (same ratchet as simulate_plan)."""
        j0, k = p["j0"], self.plan.trail_mult
        if i < j0:
            return p["init_stop"]
        a = self.atr1[j0:i + 1]
        if p["dir"] > 0:
            best = np.maximum.accumulate(np.maximum(self.h[j0:i + 1], p["entry"]))
            return float(max(p["init_stop"], np.nanmax(best - k * a)))
        best = np.minimum.accumulate(np.minimum(self.l[j0:i + 1] + self.sp[j0:i + 1], p["entry"]))
        return float(min(p["init_stop"], np.nanmin(best + k * a)))

    def safe(self, p: dict, i: int, rule: str) -> bool:
        if rule == "any":
            return True
        if rule == "locked":
            st = self.stop_after(p, i)
            return st >= p["entry"] if p["dir"] > 0 else st <= p["entry"]
        if rule.startswith("profit"):
            need = float(rule[6:] or 1.0)
            cur = (self.c[i] - p["entry"]) / p["risk"] if p["dir"] > 0 else (p["entry"] - (self.c[i] + self.sp[i])) / p["risk"]
            return cur >= need
        raise ValueError(rule)


def run_variant(df, base_sig, add_sig, atr1, plan, spread, slip, clock, add_rule="locked", max_open=2,
                flat_cooldown=True) -> pd.DataFrame:
    """base_sig may open a position when flat; while positions are open, add_sig (or base_sig)
    may add one in the same direction if every open position passes add_rule."""
    bk = Book(df, atr1, plan, spread, slip, clock)
    n = len(df)
    last_exit = -10 ** 9
    cand = np.flatnonzero((base_sig != 0) | (add_sig != 0))
    for i in cand:
        if i >= n - 1:
            continue
        open_pos = [p for p in bk.rows if p["exit_i"] >= i + 1]
        if not open_pos:
            d = int(base_sig[i])
            if d == 0 or (flat_cooldown and i < last_exit + plan.cooldown + 1):
                continue
        else:
            if len(open_pos) >= max_open:
                continue
            d = int(add_sig[i]) or int(base_sig[i])
            if d == 0 or any(p["dir"] != d for p in open_pos):
                continue
            if not all(bk.safe(p, i, add_rule) for p in open_pos):
                continue
        r = bk.open(i, d)
        if r is not None:
            r["add"] = bool(open_pos)
            last_exit = max(last_exit, int(r["exit_i"]))
    if not bk.rows:
        return pd.DataFrame()
    return pd.DataFrame(bk.rows).sort_values("sig_i").reset_index(drop=True)


def run_portfolio(specs: dict, cap_dir: int | None = None) -> pd.DataFrame:
    """Several instruments traded together. specs = {sym: dict(df=, base=, add=, atr1=, plan=, sp=, sl=, clock=,
    rule=, max_open=, cooldown=)}. Signals are taken in time order; each instrument follows run_variant's
    rules, and at most `cap_dir` positions may be open in the same direction across all instruments
    (gold and silver usually move together, so two longs are close to one double-size long)."""
    books, cands = {}, []
    for sym, sp_ in specs.items():
        df = sp_["df"]
        books[sym] = Book(df, sp_["atr1"], sp_["plan"], sp_["sp"], sp_["sl"], sp_["clock"])
        books[sym].last_exit = -10 ** 9
        idx = np.flatnonzero((sp_["base"] != 0) | (sp_["add"] != 0))
        idx = idx[idx < len(df) - 1]
        fill = df.index[idx + 1]
        cands += [(t, sym, int(i)) for t, i in zip(fill, idx)]
    cands.sort(key=lambda x: (x[0], x[1]))
    for t, sym, i in cands:
        sp_, bk = specs[sym], books[sym]
        open_here = [p for p in bk.rows if p["exit_i"] >= i + 1]
        if not open_here:
            d = int(sp_["base"][i])
            if d == 0 or (sp_.get("cooldown", True) and i < bk.last_exit + bk.plan.cooldown + 1):
                continue
        else:
            if len(open_here) >= sp_.get("max_open", 1):
                continue
            d = int(sp_["add"][i]) or int(sp_["base"][i])
            if d == 0 or any(p["dir"] != d for p in open_here):
                continue
            if not all(bk.safe(p, i, sp_.get("rule", "any")) for p in open_here):
                continue
        if cap_dir is not None:
            same = sum(1 for b in books.values() for p in b.rows
                       if p["dir"] == d and p["entry_time"] <= t < p["exit_time"] + pd.Timedelta(minutes=15))
            if same >= cap_dir:
                continue
        r = bk.open(i, d)
        if r is not None:
            r["add"] = bool(open_here)
            r["sym"] = sym
            bk.last_exit = max(bk.last_exit, int(r["exit_i"]))
    rows = [r for b in books.values() for r in b.rows]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


# ----------------------------------------------------------------------------- metrics

def stats(tr: pd.DataFrame, a: str, b: str) -> dict:
    m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
    x = tr[m].sort_values("exit_time")
    yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    if x.empty:
        return {"n": 0}
    r = x["R"].to_numpy()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq))
    streak = best = 0
    for v in r:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    ryr = float(r.sum()) / yrs
    return {"n": int(len(r)), "per_week": round(len(r) / (yrs * 52.18), 2), "win%": round(100 * float((r > 0).mean()), 1),
            "avgR": round(float(r.mean()), 3), "totR": round(float(r.sum()), 1), "R_yr": round(ryr, 1),
            "maxDD": round(dd, 1), "eff": round(ryr / dd, 2) if dd > 0 else None, "streak": best,
            "adds": int(x.get("add", pd.Series(False, index=x.index)).sum())}


def evaluate(tr: pd.DataFrame, start="2005-01-01", end="2025-03-21") -> dict:
    tr = tr[(tr["reason"] != "end")]
    out = {p: stats(tr, a, b) for p, (a, b) in PERIODS.items()}
    out["all"] = stats(tr, start, end)
    return out


VARIANTS = [
    # name, base (hours, squeeze), add (hours, squeeze) or None, add rule, max open, cooldown when flat
    ("一般設定（24h＋收窄，一次一張）", (24, True), None, "any", 1, True),
    ("現在的積極模式（12h，最多 2 張）", (12, False), (12, False), "any", 2, False),
    ("順勢加碼：保本後加 12h 訊號，最多 2 張", (24, True), (12, False), "locked", 2, True),
    ("順勢加碼：保本後加 12h 訊號，最多 3 張", (24, True), (12, False), "locked", 3, True),
    ("順勢加碼：+1R 後加 12h 訊號，最多 2 張", (24, True), (12, False), "profit1", 2, True),
    ("順勢加碼：+1R 後加 12h 訊號，最多 3 張", (24, True), (12, False), "profit1", 3, True),
    ("順勢加碼：保本後加 24h 訊號，最多 2 張", (24, True), (24, False), "locked", 2, True),
    ("順勢加碼：保本後加 8h 訊號，最多 3 張", (24, True), (8, False), "locked", 3, True),
    ("16h＋收窄，最多 2 張", (16, True), (16, True), "any", 2, True),
    ("16h＋收窄，保本後加 12h，最多 3 張", (16, True), (12, False), "locked", 3, True),
    ("12h，保本後才加，最多 2 張", (12, False), (12, False), "locked", 2, True),
    ("12h，+1R 後才加，最多 2 張", (12, False), (12, False), "profit1", 2, True),
    # second batch (same run): more channels / caps
    ("24h＋收窄，最多 2 張", (24, True), (24, True), "any", 2, True),
    ("24h＋收窄，第 2 張用 16h＋收窄", (24, True), (16, True), "any", 2, True),
    ("16h＋收窄，一次一張", (16, True), None, "any", 1, True),
    ("16h＋收窄，最多 3 張", (16, True), (16, True), "any", 3, True),
    ("16h＋收窄，+0.5R 後才加，最多 2 張", (16, True), (16, True), "profit0.5", 2, True),
    ("24h＋收窄，+0.5R 後才加 16h＋收窄", (24, True), (16, True), "profit0.5", 2, True),
    ("12h＋收窄，最多 2 張", (12, True), (12, True), "any", 2, True),
    ("16h（不收窄），最多 2 張", (16, False), (16, False), "any", 2, True),
    # third batch: the channel length around 16 hours (the mode uses 18)
    ("14h＋收窄，最多 2 張", (14, True), (14, True), "any", 2, True),
    ("高效積極模式：18h＋收窄，同向最多 2 張", (18, True), (18, True), "any", 2, True),
    ("20h＋收窄，最多 2 張", (20, True), (20, True), "any", 2, True),
]


def run_all(df: pd.DataFrame, sym: str = "XAUUSD", start="2005-01-01", end="2025-03-21", blackout=None,
            variants=VARIANTS) -> tuple[pd.DataFrame, dict]:
    inst = INSTRUMENTS[sym]
    hours = tuple(sorted({v[1][0] for v in variants} | {v[2][0] for v in variants if v[2]}))
    sets = signal_sets(df, sym, hours=hours, blackout=blackout)
    px = df["open"].to_numpy(float)
    sp, sl = px * inst.spread_pct, px * inst.slip_pct
    clock = Clock.build(df.index)
    rows, trades = [], {}
    for name, base, add, rule, mo, cd in variants:
        sig_b, f, p = sets[base]
        sig_a = sets[add][0] if add else np.zeros_like(sig_b)
        plan = p.plan(swap=True)
        t0 = time.time()
        tr = run_variant(df, sig_b, sig_a, f["h1_atr"].to_numpy(float), plan, sp, sl, clock, rule, mo, cd)
        tr = tr[tr["entry_time"] >= pd.Timestamp(start, tz="UTC")]
        trades[name] = tr
        ev = evaluate(tr, start, end)
        row = {"name": name, "secs": round(time.time() - t0, 1)}
        for per, d in ev.items():
            for k, v in d.items():
                row[f"{per}_{k}"] = v
        rows.append(row)
        print(f"  {name}: {ev['all'].get('per_week')}/wk avgR {ev['all'].get('avgR')} R/yr {ev['all'].get('R_yr')} "
              f"DD {ev['all'].get('maxDD')} eff {ev['all'].get('eff')} | dev {ev['dev'].get('eff')} val {ev['val'].get('eff')} "
              f"test {ev['test'].get('eff')}", flush=True)
    return pd.DataFrame(rows), trades


def _row(name: str, tr: pd.DataFrame, w=None) -> dict:
    """evaluate() with optional per-trade size weights (R x weight; avgR stays per trade)."""
    tr = tr[tr["reason"] != "end"].copy()
    if w is not None:
        tr["R"] = tr["R"] * w(tr)
    ev = evaluate(tr)
    row = {"name": name}
    for per, d in ev.items():
        for k, v in d.items():
            row[f"{per}_{k}"] = v
    return row


def robustness(df: pd.DataFrame, sym: str = "XAUUSD") -> pd.DataFrame:
    """2 positions with the squeeze filter: channel 14-20 hours x squeeze window / threshold."""
    inst = INSTRUMENTS[sym]
    px = df["open"].to_numpy(float)
    sp, sl = px * inst.spread_pct, px * inst.slip_pct
    clock = Clock.build(df.index)
    rows = []
    for h in (14, 16, 18, 20, 24):
        p0 = replace(inst.strategy_params(True, squeeze=False), breakout_hours=h)
        f = S.compute_features(df, params=p0)
        for bars, pct in ((16, 0.3), (8, 0.3), (24, 0.3), (16, 0.2), (16, 0.4)):
            p = replace(p0, squeeze_bars=bars, squeeze_pct=pct)
            sig = S.signals_from_rules(S.rule_table(df, f, p, None))
            mo = 1 if h == 24 else 2
            tr = run_variant(df, sig, sig, f["h1_atr"].to_numpy(float), p.plan(swap=True), sp, sl, clock, "any", mo, True)
            rows.append(_row(f"{h}h 收窄 {bars // 4}h/{int(pct * 100)}% 最多 {mo} 張", tr))
            print("  ", rows[-1]["name"], rows[-1]["all_eff"], flush=True)
    return pd.DataFrame(rows)


def tiered(df: pd.DataFrame, sym: str = "XAUUSD") -> pd.DataFrame:
    """Squeeze signals (16h) at full size plus the other signals (12h or 16h, no squeeze) at a
    smaller size, sharing the 2-3 slots: does trading the weaker signals small add anything?"""
    inst = INSTRUMENTS[sym]
    sets = signal_sets(df, sym, hours=(16, 12))
    px = df["open"].to_numpy(float)
    sp, sl = px * inst.spread_pct, px * inst.slip_pct
    clock = Clock.build(df.index)
    hi, f, p = sets[(16, True)]
    rows = []
    for lo_h in (12, 16):
        lo = sets[(lo_h, False)][0]
        u = np.where(hi != 0, hi, lo)
        for mo in (2, 3):
            tr = run_variant(df, u, u, f["h1_atr"].to_numpy(float), p.plan(swap=True), sp, sl, clock, "any", mo, False)
            for w in (1.0, 0.5, 0.33):
                rows.append(_row(f"16h 收窄訊號 1 份 + {lo_h}h 其他訊號 {w:g} 份，最多 {mo} 張", tr,
                                 lambda t, w=w: np.where(hi[t["sig_i"].to_numpy()] != 0, 1.0, w)))
                print("  ", rows[-1]["name"], rows[-1]["all_eff"], flush=True)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    df = load()
    res, trades = run_all(df)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "XAUUSD_efficient_variants.csv", index=False)
    show = ["name", "all_per_week", "all_avgR", "all_R_yr", "all_maxDD", "all_eff", "all_streak", "all_adds",
            "dev_eff", "val_eff", "test_eff", "dev_avgR", "val_avgR", "test_avgR"]
    print(res[show].to_string(index=False))
    (ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
    pd.to_pickle(trades, ROOT / "data" / "cache" / "efficient_trades.pkl")
    if "--quick" not in sys.argv:
        rob = robustness(df)
        rob.to_csv(OUT / "XAUUSD_efficient_robustness.csv", index=False)
        tie = tiered(df)
        tie.to_csv(OUT / "XAUUSD_efficient_tiered.csv", index=False)
        print(rob[show[:6] + show[8:11]].to_string(index=False))
        print(tie[["name", "all_per_week", "all_R_yr", "all_maxDD", "all_eff", "dev_eff", "val_eff", "test_eff"]].to_string(index=False))


if __name__ == "__main__":
    main()
