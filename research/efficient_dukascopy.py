"""Second-vendor check of the efficient active variants: Dukascopy 15m bars, gold AND silver,
2013/7-2026/10, plus a one-year money simulation for a $2,000 account (docs/RESEARCH.md section 11).

Same rules, costs and position handling as research/efficient_active.py (exact live 15m rules,
day mode, overnight fees); only the data differ. Dukascopy is not reachable from every network,
so this was run in the browser (Pyodide) on 1-minute BID candles aggregated to 15m there:

    result = run_all({"XAUUSD": df_gold, "XAGUSD": df_silver})

Locally, with 15m CSV files (time,open,high,low,close in UTC):

    python research/efficient_dukascopy.py data/dukascopy_XAUUSD_15m_2013.csv.gz data/dukascopy_XAGUSD_15m_2013.csv.gz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock  # noqa: E402

START, END = "2013-07-01", "2026-10-09"
PERIODS = [("2013/7–2016", "2013-07-01", "2016-12-31"), ("2017–2020", "2017-01-01", "2020-12-31"),
           ("2021–2025/3", "2021-01-01", "2025-03-31"), ("2025/4–2026/10", "2025-04-01", END)]

# name: base (hours, squeeze), add (hours, squeeze) or None, add rule, max open, cooldown when flat
VARIANTS = {
    "XAUUSD": {
        "normal": ((24, True), None, "any", 1, True),          # 一般設定
        "active": ((12, False), (12, False), "any", 2, False),  # 現在的積極模式
        "eff16": ((16, True), (16, True), "any", 2, True),      # 16h + 收窄，最多 2 張
        # "18sq_max2" below = 高效積極模式 for gold (goldsignal/active.py EFFICIENT)
        "16sq_max1": ((16, True), None, "any", 1, True),
        "16sq_max3": ((16, True), (16, True), "any", 3, True),
        "14sq_max2": ((14, True), (14, True), "any", 2, True),
        "18sq_max2": ((18, True), (18, True), "any", 2, True),
        "20sq_max2": ((20, True), (20, True), "any", 2, True),
        "16_max2": ((16, False), (16, False), "any", 2, True),
    },
    "XAGUSD": {
        "normal": ((24, False), None, "any", 1, True),
        "active": ((12, False), (12, False), "any", 2, False),
        "eff16": ((16, True), (16, True), "any", 2, True),
        "16_max2": ((16, False), (16, False), "any", 2, True),
        "24_max2": ((24, False), (24, False), "any", 2, True),
        "24sq_max1": ((24, True), None, "any", 1, True),
        "20_max2": ((20, False), (20, False), "any", 2, True),
        "20sq_max2": ((20, True), (20, True), "any", 2, True),
        "12sq_max2": ((12, True), (12, True), "any", 2, True),
    },
}

# gold + silver traded together (positions are independent; per-instrument caps as above)
COMBOS = {"normal": ("normal", "normal"), "active": ("active", "active"), "eff16+normal": ("eff16", "normal"),
          "eff16+16_max2": ("eff16", "16_max2"), "eff16+eff16": ("eff16", "eff16"), "eff16+24_max2": ("eff16", "24_max2"),
          "eff16+active": ("eff16", "active"), "eff16+24sq_max1": ("eff16", "24sq_max1")}
for _g in ("18sq_max2", "20sq_max2", "16sq_max1"):   # 18sq_max2 + normal = 高效積極模式
    for _s in ("normal", "24sq_max1"):
        COMBOS[f"{_g}+{_s}"] = (_g, _s)

PRICE = {"XAUUSD": None, "XAGUSD": None}   # filled from the last bar
OZ_PER_001 = {"XAUUSD": 1.0, "XAGUSD": 50.0}   # 0.01 lot


def frame(t_ms, o, h, l, c, scale=1e-3) -> pd.DataFrame:
    idx = pd.to_datetime(np.asarray(t_ms, dtype="int64"), unit="ms", utc=True)
    df = pd.DataFrame({"open": np.asarray(o, float) * scale, "high": np.asarray(h, float) * scale,
                       "low": np.asarray(l, float) * scale, "close": np.asarray(c, float) * scale, "volume": 0.0}, index=idx)
    return df[~df.index.duplicated()].sort_index()


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
    g, l_ = r[r > 0].sum(), -r[r <= 0].sum()
    ryr = float(r.sum()) / yrs
    return {"n": int(len(r)), "per_week": round(len(r) / (yrs * 52.18), 2), "win%": round(100 * float((r > 0).mean()), 1),
            "avgR": round(float(r.mean()), 3), "totR": round(float(r.sum()), 1), "R_yr": round(ryr, 1),
            "maxDD": round(dd, 1), "eff": round(ryr / dd, 2) if dd > 0 else None,
            "PF": round(float(g / l_), 2) if l_ > 0 else None, "streak": best}


def by_period(tr: pd.DataFrame) -> dict:
    out = {lab: stats(tr, a, b) for lab, a, b in PERIODS}
    out["all"] = stats(tr, START, END)
    return out


def run_symbol(df: pd.DataFrame, sym: str, variants: dict | None = None) -> dict[str, pd.DataFrame]:
    variants = variants or VARIANTS[sym]
    inst = INSTRUMENTS[sym]
    hours = sorted({v[0][0] for v in variants.values()} | {v[1][0] for v in variants.values() if v[1]})
    sets = E.signal_sets(df, sym, hours=tuple(hours))
    px = df["open"].to_numpy(float)
    sp, sl = px * inst.spread_pct, px * inst.slip_pct
    clock = Clock.build(df.index)
    out = {}
    for name, (base, add, rule, mo, cd) in variants.items():
        sig_b, f, p = sets[base]
        sig_a = sets[add][0] if add else np.zeros_like(sig_b)
        tr = E.run_variant(df, sig_b, sig_a, f["h1_atr"].to_numpy(float), p.plan(swap=True), sp, sl, clock, rule, mo, cd)
        tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp(START, tz="UTC"))].copy()
        tr["sym"] = sym
        out[name] = tr
    return out


# ----------------------------------------------------------------------------- money

def money(trs: list[pd.DataFrame], lots: dict, balance=2000.0, paths=20000, block=5, seed=7) -> dict:
    """One-year block bootstrap (blocks of `block` consecutive trades) of the trades 2013/7-2026/10.
    Every trade risks R x stop x oz dollars, the stop distance drawn from that instrument's trades of
    the last 12 months at today's price; `lots` = {sym: lots per signal}."""
    x = pd.concat(trs).sort_values("entry_time").reset_index(drop=True)
    yrs = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    n_yr = int(round(len(x) / yrs))
    syms = sorted(x["sym"].unique())
    code = x["sym"].map({s: k for k, s in enumerate(syms)}).to_numpy()
    R = x["R"].to_numpy(float)
    recent = x["entry_time"] >= pd.Timestamp(END, tz="UTC") - pd.Timedelta(days=365)
    pools, oz, med_stop = [], [], {}
    for s in syms:
        pct = (x.loc[recent & (x["sym"] == s), "risk"] / x.loc[recent & (x["sym"] == s), "entry"]).to_numpy(float)
        pools.append(pct * PRICE[s])
        med_stop[s] = round(float(np.median(pct * PRICE[s])), 2)
        oz.append(OZ_PER_001[s] * lots[s] / 0.01)
    oz = np.asarray(oz)
    rng = np.random.default_rng(seed)
    nb = -(-n_yr // block)
    starts = rng.integers(0, len(x) - block + 1, size=(paths, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :n_yr]
    stop = np.zeros(idx.shape)
    for k, pool in enumerate(pools):
        draw = pool[rng.integers(0, len(pool), size=idx.shape)]
        stop = np.where(code[idx] == k, draw, stop)
    pnl = R[idx] * stop * oz[code[idx]]
    eq = balance + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.full((paths, 1), balance), eq], axis=1), axis=1)[:, 1:]
    dd = np.max((peak - eq) / peak, axis=1) * 100
    ret = (eq[:, -1] / balance - 1) * 100
    # the real 2013-2026 sequence at today's typical stop
    hist = np.cumsum(R * np.array([med_stop[s] for s in syms])[code] * oz[code])
    hist_dd = float(np.max(np.maximum.accumulate(np.r_[0.0, hist])[1:] - hist))
    return {"trades_per_year": n_yr, "per_week": round(n_yr / 52.18, 2), "lots": lots, "stop_usd_median": med_stop,
            "med_ret": round(float(np.median(ret)), 1), "p5_ret": round(float(np.percentile(ret, 5)), 1),
            "p95_ret": round(float(np.percentile(ret, 95)), 1), "p_loss": round(float((ret < 0).mean() * 100), 1),
            "med_dd": round(float(np.median(dd)), 1), "p_dd30": round(float((dd >= 30).mean() * 100), 1),
            "p_dd50": round(float((dd >= 50).mean() * 100), 1), "hist_maxdd_usd": round(hist_dd), "n_hist": int(len(x))}


MONEY = [  # label, {sym: variant}, {sym: lots}
    ("gold_normal", {"XAUUSD": "normal"}, {"XAUUSD": 0.01}),
    ("gold_active", {"XAUUSD": "active"}, {"XAUUSD": 0.01}),
    ("gold_eff16", {"XAUUSD": "eff16"}, {"XAUUSD": 0.01}),
    ("gold_eff16_x2", {"XAUUSD": "eff16"}, {"XAUUSD": 0.02}),
    ("gold_normal_x2", {"XAUUSD": "normal"}, {"XAUUSD": 0.02}),
    ("both_normal", {"XAUUSD": "normal", "XAGUSD": "normal"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_active", {"XAUUSD": "active", "XAGUSD": "active"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_eff16+normal", {"XAUUSD": "eff16", "XAGUSD": "normal"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_eff16+16_max2", {"XAUUSD": "eff16", "XAGUSD": "16_max2"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_eff16+eff16", {"XAUUSD": "eff16", "XAGUSD": "eff16"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_eff16+24_max2", {"XAUUSD": "eff16", "XAGUSD": "24_max2"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_eff16x2+normal", {"XAUUSD": "eff16", "XAGUSD": "normal"}, {"XAUUSD": 0.02, "XAGUSD": 0.01}),
    ("gold_18sq_max2", {"XAUUSD": "18sq_max2"}, {"XAUUSD": 0.01}),
    ("gold_18sq_max2_x2", {"XAUUSD": "18sq_max2"}, {"XAUUSD": 0.02}),
    ("both_18sq_max2+normal", {"XAUUSD": "18sq_max2", "XAGUSD": "normal"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_18sq_max2x2+normal", {"XAUUSD": "18sq_max2", "XAGUSD": "normal"}, {"XAUUSD": 0.02, "XAGUSD": 0.01}),
    ("both_18sq_max2+24sq_max1", {"XAUUSD": "18sq_max2", "XAGUSD": "24sq_max1"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("both_eff16+24sq_max1", {"XAUUSD": "eff16", "XAGUSD": "24sq_max1"}, {"XAUUSD": 0.01, "XAGUSD": 0.01}),
    ("silver_normal", {"XAGUSD": "normal"}, {"XAGUSD": 0.01}),
    ("silver_active", {"XAGUSD": "active"}, {"XAGUSD": 0.01}),
]
MONEY_HOW = ("One-year block bootstrap (5-trade blocks, 20,000 paths) of the 15m trades above, $2,000 account, "
             "lots per signal as given (0.01 lot = gold 1 oz / silver 50 oz), stop distances drawn from each "
             "instrument's trades of the last 12 months at the last price; hist_maxdd_usd = worst drop of the actual "
             "2013-2026 sequence at today's median stop.")


def run_all(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    trades = {}
    for sym, df in frames.items():
        PRICE[sym] = float(df["close"].iloc[-1])
        trades[sym] = run_symbol(df, sym)
    res = {"how": ("Dukascopy 1-minute BID candles aggregated to 15m (2013-2026), exact live 15m rules in day mode "
                   "(research/efficient_active.py run_variant), trades entered 2013-07-01..2026-10-09. Costs: gold spread "
                   "0.012 %, silver 0.06 %, slippage per fill, overnight fees. eff = R per year / max drawdown (R)."),
           "periods": [p[0] for p in PERIODS] + ["all"],
           "instruments": {s: {n: by_period(t) for n, t in v.items()} for s, v in trades.items()}}
    if len(trades) == 2:
        res["both"] = {c: by_period(pd.concat([trades["XAUUSD"][g], trades["XAGUSD"][s]])) for c, (g, s) in COMBOS.items()}
    res["prices"] = {s: round(p, 3) for s, p in PRICE.items() if p}
    res["money"] = {lab: money([trades[s][v] for s, v in pick.items() if s in trades], lots)
                    for lab, pick, lots in MONEY if all(s in trades for s in pick)}
    res["money_how"] = MONEY_HOW
    return res, trades


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    if "volume" not in df:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def main():
    frames = {}
    for p in sys.argv[1:]:
        frames["XAGUSD" if "XAG" in Path(p).name.upper() else "XAUUSD"] = load_csv(p)
    res, _ = run_all(frames)
    (ROOT / "results" / "efficient_dukascopy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True),
                                                               encoding="utf-8")
    print(json.dumps(res["money"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
