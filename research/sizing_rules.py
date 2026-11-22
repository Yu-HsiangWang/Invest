"""Sizing rules aimed at +100 % a year with less ruin (docs/RESEARCH.md section 13).

Fixed-fraction sizing reaches a +100 % median year at 5-8 % risk per trade, but with 50-80 %
drawdowns. Rules that change the risk with the account's state are compared on the same trades:

  fixed f            every trade risks f of the account
  dd-linear f0       f = f0 x max(0.25, 1 - drawdown / 0.5)   (risk shrinks as the account falls)
  dd-step f0         f0 below a 20 % drawdown, f0 / 2 to 40 %, f0 / 4 beyond
  house-money a,b    risk = a x starting capital + b x profit so far (the starting capital is
                     protected; most of the bet is the year's winnings), as a fraction capped at 25 %

Scored four ways for each strategy (trades from research/target_100.py's cache):
  * a fresh account every calendar year 2005-2024 (the year's trades in order, sizing on the
    equity at each entry): median / worst year, years >= +100 %, worst drawdown - and the median
    year separately in 2005-2016 / 2017-2020 / 2021-2024 (the rules were not fitted, but the
    best one is picked after seeing the table, so it has to hold in each period);
  * the same with the user's real account: $2,000 each January, Mitrade lots (0.01 lot = 1 oz,
    whole 0.01 lots, at most 0.3 lots open), stop $ at today's gold price;
  * one-year block bootstraps (5-trade blocks, 20,000 paths): median, 5th percentile, chance of
    +100 %, chance of a 50 % drawdown;
  * the whole 2005-2025 history compounded.

    python research/sizing_rules.py     (needs the trade cache: python research/target_100.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache"
GOLD_PRICE = 3900.0
PERIODS = {"2005-16": (2005, 2016), "2017-20": (2017, 2020), "2021-24": (2021, 2024)}
STRATEGIES = ("active", "active_nosq", "efficient_hold", "efficient", "efficient_nosq", "active_hold", "active_nosq_hold",
              "efficient_nosq_hold")


# every rule works on scalars and on numpy arrays (the bootstrap sizes 20,000 accounts at once)
def rule_fixed(f):
    return lambda eq, peak, start: f + 0.0 * eq


def rule_dd_linear(f0, floor=0.25, dd_full=0.5):
    return lambda eq, peak, start: f0 * np.maximum(floor, 1 - (1 - eq / peak) / dd_full)


def rule_dd_step(f0):
    def r(eq, peak, start):
        dd = 1 - eq / peak
        return np.where(dd < 0.2, f0, np.where(dd < 0.4, f0 / 2, f0 / 4))
    return r


def rule_house(a, b, cap=0.25):
    return lambda eq, peak, start: np.where(eq > 0, np.minimum(cap, (a * start + b * np.maximum(eq - start, 0.0)) / np.maximum(eq, 1e-12)), 0.0)


RULES = {"fixed 3%": rule_fixed(0.03), "fixed 5%": rule_fixed(0.05), "fixed 8%": rule_fixed(0.08), "fixed 10%": rule_fixed(0.10),
         "dd-linear 8%": rule_dd_linear(0.08), "dd-linear 10%": rule_dd_linear(0.10), "dd-linear 15%": rule_dd_linear(0.15),
         "dd-step 8%": rule_dd_step(0.08), "dd-step 10%": rule_dd_step(0.10), "dd-step 15%": rule_dd_step(0.15),
         "house 3% + 20% of profit": rule_house(0.03, 0.20), "house 4% + 30% of profit": rule_house(0.04, 0.30),
         "house 5% + 40% of profit": rule_house(0.05, 0.40)}


def _events(tr: pd.DataFrame):
    ev = [(t, 0, i) for i, t in enumerate(tr["entry_time"])] + [(t, 1, i) for i, t in enumerate(tr["exit_time"])]
    ev.sort(key=lambda x: (x[0], x[1]))
    return ev


def run_events(tr: pd.DataFrame, rule, start: float = 1.0) -> tuple[float, float]:
    """Events in time order; each entry stakes rule(eq, peak, start) x eq. Returns (final, maxDD)."""
    R = tr["R"].to_numpy(float)
    eq = peak = start
    mdd = 0.0
    stake = np.zeros(len(tr))
    for _, kind, i in _events(tr):
        if kind == 0:
            stake[i] = float(rule(eq, peak, start)) * eq
        else:
            eq = max(eq + stake[i] * R[i], 0.0)
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak if peak > 0 else 1.0)
    return eq, mdd


def run_lots(tr: pd.DataFrame, rule, balance: float = 2000.0, max_lots: float = 0.3, price: float = GOLD_PRICE) -> tuple[float, float]:
    """The same with Mitrade lots (as research/target_100.lots_sim): floor(f x equity / stop $ per
    0.01 lot) x 0.01 lots, 0.01 lot when that rounds to zero unless it risks more than 3 f, never
    more than max_lots open. Returns (final $, maxDD)."""
    R = tr["R"].to_numpy(float)
    stop_usd = (tr["risk"] / tr["entry"]).to_numpy(float) * price
    eq = peak = balance
    mdd = 0.0
    lots = np.zeros(len(tr)); open_lots = 0.0
    for _, kind, i in _events(tr):
        if kind == 0:
            f = float(rule(eq, peak, balance))
            want = np.floor(f * eq / stop_usd[i] + 1e-9) / 100.0
            if want < 0.01:
                if stop_usd[i] > 3 * f * eq:
                    continue
                want = 0.01
            want = min(want, round(max_lots - open_lots, 2))
            if want < 0.01:
                continue
            lots[i] = want; open_lots = round(open_lots + want, 2)
        elif lots[i] > 0:
            eq += lots[i] * 100 * stop_usd[i] * R[i]
            open_lots = round(open_lots - lots[i], 2)
            peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
            if eq <= 0:
                return 0.0, 1.0
    return eq, mdd


def bootstrap(R: np.ndarray, n_year: int, rule, paths: int = 20000, block: int = 5, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    nb = -(-n_year // block)
    starts = rng.integers(0, len(R) - block + 1, size=(paths, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :n_year]
    eq = np.ones(paths); peak = np.ones(paths); mdd = np.zeros(paths)
    for k in range(n_year):
        f = rule(eq, peak, 1.0)
        eq = np.maximum(eq * (1 + f * R[idx[:, k]]), 0.0)
        peak = np.maximum(peak, eq)
        mdd = np.maximum(mdd, 1 - np.where(peak > 0, eq / peak, 0))
    ret = eq - 1
    return {"med": round(float(np.median(ret)), 3), "p5": round(float(np.percentile(ret, 5)), 3),
            "p_double": round(float((ret >= 1).mean()), 3), "p_loss": round(float((ret < 0).mean()), 3),
            "p_dd50": round(float((mdd >= 0.5).mean()), 3), "dd_med": round(float(np.median(mdd)), 3)}


def score(tr: pd.DataFrame, rule) -> dict:
    years, yrs, dds, lot_yrs, lot_dds = [], [], [], [], []
    for y in range(2005, 2025):
        t = tr[tr["entry_time"].dt.year == y].reset_index(drop=True)
        if len(t) < 3:
            continue
        e, d = run_events(t, rule)
        years.append(y); yrs.append(e - 1); dds.append(d)
        e2, d2 = run_lots(t, rule)
        lot_yrs.append(e2 / 2000 - 1); lot_dds.append(d2)
    years, yrs, dds, lot_yrs, lot_dds = map(np.array, (years, yrs, dds, lot_yrs, lot_dds))
    years_total = (pd.Timestamp("2025-03-21") - pd.Timestamp("2005-01-01")).days / 365.25
    e_all, d_all = run_events(tr.reset_index(drop=True), rule)
    R = tr["R"].to_numpy(float)
    out = {"fresh_median": round(float(np.median(yrs)), 3), "fresh_worst": round(float(yrs.min()), 3),
           "fresh_years_ge_100%": int((yrs >= 1).sum()), "fresh_years": int(len(yrs)), "fresh_years_lost": int((yrs < 0).sum()),
           "fresh_worst_dd": round(float(dds.max()), 3), "fresh_median_dd": round(float(np.median(dds)), 3)}
    for name, (a, b) in PERIODS.items():
        m = (years >= a) & (years <= b)
        out[f"fresh_median_{name}"] = round(float(np.median(yrs[m])), 3)
    out.update({"lots2000_median": round(float(np.median(lot_yrs)), 3), "lots2000_worst": round(float(lot_yrs.min()), 3),
                "lots2000_years_ge_100%": int((lot_yrs >= 1).sum()), "lots2000_worst_dd": round(float(lot_dds.max()), 3),
                "lots2000_years": {int(y): round(float(v), 3) for y, v in zip(years, lot_yrs)}})
    out.update({"hist_cagr": round(float(e_all ** (1 / years_total) - 1), 3) if e_all > 0 else -1.0, "hist_max_dd": round(d_all, 3),
                **{f"boot_{k}": v for k, v in bootstrap(R, int(round(len(R) / years_total)), rule).items()}})
    return out


def main():
    res = {"how": __doc__.split("\n\n")[1].strip() + "\n\n" + __doc__.split("\n\n")[2].strip(), "strategies": {}}
    for name in STRATEGIES:
        p = CACHE / f"target100_{name}.pkl"
        if not p.exists():
            continue
        tr = pd.read_pickle(p).sort_values("entry_time").reset_index(drop=True)
        res["strategies"][name] = {"trades_per_week": round(len(tr) / ((pd.Timestamp("2025-03-21") - pd.Timestamp("2005-01-01")).days / 7), 2),
                                   "avgR": round(float(tr["R"].mean()), 3), "rules": {}}
        print(f"{name}: {res['strategies'][name]['trades_per_week']}/wk avgR {res['strategies'][name]['avgR']:+.3f}", flush=True)
        for rn, rule in RULES.items():
            s = score(tr, rule)
            res["strategies"][name]["rules"][rn] = s
            print(f"   {rn:25s} fresh med {s['fresh_median']:+.0%} (05-16 {s['fresh_median_2005-16']:+.0%} 17-20 {s['fresh_median_2017-20']:+.0%} "
                  f"21-24 {s['fresh_median_2021-24']:+.0%}) worst {s['fresh_worst']:+.0%} >=100% {s['fresh_years_ge_100%']}/{s['fresh_years']} "
                  f"wDD {s['fresh_worst_dd']:.0%} | $2k lots med {s['lots2000_median']:+.0%} worst {s['lots2000_worst']:+.0%} | boot med {s['boot_med']:+.0%} "
                  f"p5 {s['boot_p5']:+.0%} P(x2) {s['boot_p_double']:.0%} P(DD50) {s['boot_p_dd50']:.0%} | CAGR {s['hist_cagr']:+.0%} DD {s['hist_max_dd']:.0%}",
                  flush=True)
    (OUT / "XAUUSD_sizing_rules.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
