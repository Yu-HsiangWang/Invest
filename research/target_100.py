"""Can the gold system make 100 % a year? (docs/RESEARCH.md section 13)

A strategy's yearly return is set by two things: its edge (R per trade, trades per year) and how
much of the account each trade risks (f). With compounding, the median growth rises with f up to
the Kelly fraction f* and FALLS beyond it, so every strategy has a ceiling - the median yearly
return at f* - that no amount of risk can lift. Below the ceiling, a 100 % target is a choice of f,
and the price of that choice is the drawdown.

For each strategy (trades from the live code, gold 15m 2005-2025/3, costs and day mode as always):
  * the actual history compounded at f (each trade risks f x the equity at its entry; positions
    that overlap are sized on the equity when each opens): CAGR, every calendar year, max drawdown;
  * one-year block bootstraps (5-trade blocks, 20,000 paths): median / 5th percentile year,
    chance of a 30 % / 50 % / 80 % fall from the peak within the year;
  * the Kelly fraction and the median-year ceiling;
  * the smallest f whose bootstrapped median year is >= +100 %;
  * a $2,000 account with Mitrade lots (0.01 lot = 1 oz, at most 0.3 lots open in total) at that f;
  * each period (2005-16 / 2017-20 / 2021-25/3) on its own, and one-year bootstraps of whole
    calendar months (which keep positions that were open together, together) - drawn from all of
    2005-2025, and from 2013/7-2025 and 2017-2025 only (--recent), the numbers docs/RESEARCH.md
    section 13 quotes.

Strategies: the three modes in day mode and holding overnight (_hold), the first 積極模式
(active_12h_old), and with --only: without the squeeze filter (_nosq) and with up to 3 / 4
same-direction positions (_max3 / _max4).

    python research/target_100.py
    python research/target_100.py --only active_nosq,active_max3,active_max4   # add strategies to the results
    python research/target_100.py --recent                                       # the month bootstraps by start year
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
from goldsignal import strategy as S  # noqa: E402
from goldsignal.active import active_params, mode_rules, run_mode, run_multi  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock  # noqa: E402

OUT = ROOT / "results"
START, END = "2005-01-01", "2025-03-21"
YEARS = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
F_GRID = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
GOLD_PRICE = 3900.0      # $/oz for the lot simulation (gold was ~$3,900 in Oct 2026)


# ----------------------------------------------------------------------------- trades

def strategy_trades(df: pd.DataFrame | None = None, only: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Closed trades 2005-2025/3 of each strategy, from the live code. `only` = names to build
    (default: all but the no-squeeze extras)."""
    df = E.load() if df is None else df
    inst = INSTRUMENTS["XAUUSD"]
    px = df["open"].to_numpy(float)
    sp, sl = px * inst.spread_pct, px * inst.slip_pct
    clock = Clock.build(df.index)
    out = {}
    want = (lambda k: k in only) if only else (lambda k: "nosq" not in k and "_max" not in k)
    for mode in ("normal", "efficient", "active"):
        for day in (True, False):
            r = mode_rules(inst, mode, day, True)
            k = f"{mode}{'' if day else '_hold'}"
            if want(k):
                f = S.compute_features(df, params=r.params)
                out[k] = run_mode(df, f, r, sp, sl, swap=True, clock=clock)
            k = f"{mode}_nosq{'' if day else '_hold'}"                 # the same mode without the squeeze filter
            if mode != "normal" and want(k):
                r2 = replace(r, params=replace(r.params, squeeze_bars=0))
                f = S.compute_features(df, params=r2.params)
                out[k] = run_mode(df, f, r2, sp, sl, swap=True, clock=clock)
            for n_open in (3, 4):                                       # pyramiding: more same-direction positions
                k = f"{mode}_max{n_open}{'' if day else '_hold'}"
                if mode != "normal" and want(k):
                    f = S.compute_features(df, params=r.params)
                    out[k] = run_mode(df, f, replace(r, max_open=n_open), sp, sl, swap=True, clock=clock)
    if want("active_12h_old"):
        p = active_params(inst.strategy_params(True, squeeze=False))      # the first 積極模式 (12h, no squeeze)
        f = S.compute_features(df, params=p)
        out["active_12h_old"] = run_multi(df, f, p, sp, sl, swap=True, max_open=2, clock=clock)
    for k, tr in out.items():
        out[k] = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp(START, tz="UTC"))].sort_values("entry_time").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- compounding

def compound_history(tr: pd.DataFrame, f: float) -> tuple[pd.Series, float]:
    """Equity path (start 1.0) when every trade risks f x the equity at its entry; returns the
    equity after each exit (indexed by exit time) and the max drawdown (fraction)."""
    ev = [(t, 0, i) for i, t in enumerate(tr["entry_time"])] + [(t, 1, i) for i, t in enumerate(tr["exit_time"])]
    ev.sort(key=lambda x: (x[0], x[1]))      # entries before exits at the same time
    R = tr["R"].to_numpy(float)
    eq, peak, mdd = 1.0, 1.0, 0.0
    stake = np.zeros(len(tr))
    times, vals = [], []
    for t, kind, i in ev:
        if kind == 0:
            stake[i] = f * eq
        else:
            eq = max(eq + stake[i] * R[i], 0.0)
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak if peak > 0 else 1.0)
            times.append(t); vals.append(eq)
    return pd.Series(vals, index=pd.DatetimeIndex(times)), mdd


def yearly_returns(path: pd.Series) -> pd.Series:
    """Calendar-year returns of an equity path (start of year = last value of the previous year)."""
    y = path.groupby(path.index.year).last()
    prev = y.shift(1)
    prev.iloc[0] = 1.0
    return y / prev - 1


def bootstrap_year(R: np.ndarray, n_year: int, f: float, paths: int = 20000, block: int = 5, seed: int = 3) -> dict:
    """One-year paths of n_year trades drawn in blocks of `block` consecutive trades; sequential
    compounding at f. Returns the median / percentiles of the year and drawdown probabilities."""
    rng = np.random.default_rng(seed)
    nb = -(-n_year // block)
    starts = rng.integers(0, len(R) - block + 1, size=(paths, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :n_year]
    g = np.maximum(1 + f * R[idx], 0.0)
    eq = np.cumprod(g, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((paths, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = np.max(1 - eq / peak, axis=1)
    ret = eq[:, -1] - 1
    return {"med": float(np.median(ret)), "p5": float(np.percentile(ret, 5)), "p25": float(np.percentile(ret, 25)),
            "p75": float(np.percentile(ret, 75)), "p_loss": float((ret < 0).mean()), "p_double": float((ret >= 1).mean()),
            "dd_med": float(np.median(dd)), "p_dd30": float((dd >= 0.3).mean()), "p_dd50": float((dd >= 0.5).mean()),
            "p_dd80": float((dd >= 0.8).mean())}


def kelly(R: np.ndarray) -> tuple[float, float]:
    """Growth-optimal fraction (sequential trades) and the mean log growth per trade there."""
    fs = np.linspace(0.001, min(0.99, 0.999 / max(1e-9, -R.min())), 2000)
    g = np.array([np.mean(np.log1p(f * R)) for f in fs])
    k = int(np.argmax(g))
    return float(fs[k]), float(g[k])


def lots_sim(tr: pd.DataFrame, f: float, balance: float = 2000.0, max_lots: float = 0.3, price: float = GOLD_PRICE) -> dict:
    """The actual history with a $2,000 account and Mitrade lots: each trade gets
    floor(f x equity / (stop $ per 0.01 lot)) x 0.01 lots, at least 0.01 lot (skipped if even that
    risks more than 3 f), and the open lots never exceed max_lots. Stop $ = the trade's stop
    distance as a % of price x today's gold price x 1 oz per 0.01 lot."""
    ev = [(t, 0, i) for i, t in enumerate(tr["entry_time"])] + [(t, 1, i) for i, t in enumerate(tr["exit_time"])]
    ev.sort(key=lambda x: (x[0], x[1]))
    R = tr["R"].to_numpy(float)
    stop_usd = (tr["risk"] / tr["entry"]).to_numpy(float) * price          # per 1 oz = 0.01 lot
    eq, peak, mdd = balance, balance, 0.0
    lots = np.zeros(len(tr)); open_lots = 0.0; skipped = capped = 0
    path_t, path_v = [], []
    for t, kind, i in ev:
        if kind == 0:
            want = np.floor(f * eq / stop_usd[i] + 1e-9) / 100.0
            if want < 0.01:
                if stop_usd[i] <= 3 * f * eq:          # 0.01 lot = 1 oz risks stop_usd[i]
                    want = 0.01
                else:
                    skipped += 1
                    continue
            room = round(max_lots - open_lots, 2)
            if want > room:
                capped += 1
                want = room
            if want < 0.01:
                skipped += 1
                continue
            lots[i] = want; open_lots = round(open_lots + want, 2)
        elif lots[i] > 0:
            eq += lots[i] * 100 * stop_usd[i] * R[i]
            open_lots = round(open_lots - lots[i], 2)
            peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
            path_t.append(t); path_v.append(eq)
            if eq <= 0:
                break
    path = pd.Series(path_v, index=pd.DatetimeIndex(path_t))
    yr = yearly_returns(path / balance)
    return {"final": round(float(eq), 0), "cagr": round(float((max(eq, 1e-9) / balance) ** (1 / YEARS) - 1), 3),
            "max_dd": round(mdd, 3), "skipped": skipped, "capped_by_0.3_lots": capped,
            "median_year": round(float(yr.median()), 3), "worst_year": round(float(yr.min()), 3),
            "years_ge_100%": int((yr >= 1).sum()), "years": int(len(yr))}


def year_by_year_2000(tr: pd.DataFrame, f: float, max_lots: float = 0.3) -> dict:
    """Start every calendar year 2005-2024 with a fresh $2,000 account at fraction f (Mitrade lots):
    the distribution of first-year returns - what a $2,000 account would actually face."""
    rets, dds = [], []
    for y in range(2005, 2025):
        t = tr[(tr["entry_time"].dt.year == y)]
        if len(t) < 3:
            continue
        r = lots_sim(t.reset_index(drop=True), f, max_lots=max_lots)
        rets.append(r["final"] / 2000 - 1); dds.append(r["max_dd"])
    rets, dds = np.array(rets), np.array(dds)
    return {"f": f, "years": int(len(rets)), "median": round(float(np.median(rets)), 3), "worst": round(float(rets.min()), 3),
            "best": round(float(rets.max()), 3), "years_ge_100%": int((rets >= 1).sum()), "years_lost": int((rets < 0).sum()),
            "median_dd": round(float(np.median(dds)), 3), "worst_dd": round(float(dds.max()), 3),
            "each": [round(float(x), 2) for x in rets]}


PERIODS = {"2005-16": ("2005-01-01", "2016-12-31"), "2017-20": ("2017-01-01", "2020-12-31"), "2021-25/3": ("2021-01-01", "2025-03-21")}


def period_stats(tr: pd.DataFrame, fs=(0.025, 0.0325, 0.05)) -> dict:
    """Each period on its own: trades, R per trade / per year, the largest fall of the cumulative R
    (exits in time order), efficiency = R per year / that fall, and the period compounded from a
    fresh start at a few risk fractions (CAGR and max drawdown)."""
    out = {}
    for name, (a, b) in PERIODS.items():
        t = tr[(tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))]
        yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        cum = t.sort_values("exit_time")["R"].cumsum().to_numpy()
        dd = float(np.max(np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum)) if len(cum) else 0.0
        row = {"per_week": round(len(t) / yrs / 52.18, 2), "avgR": round(float(t["R"].mean()), 3), "R_per_year": round(float(t["R"].sum() / yrs), 1),
               "max_dd_R": round(dd, 1), "eff": round(float(t["R"].sum() / yrs / dd), 2) if dd > 0 else None}
        for f in fs:
            path, mdd = compound_history(t.reset_index(drop=True), f)
            row[f"cagr_at_{f}"] = round(float(path.iloc[-1] ** (1 / yrs) - 1), 3) if len(path) and path.iloc[-1] > 0 else -1.0
            row[f"max_dd_at_{f}"] = round(mdd, 3)
        out[name] = row
    return out


def bootstrap_months(tr: pd.DataFrame, f: float, paths: int = 20000, seed: int = 11, start: str = "2005-01", end: str = "2025-02") -> dict:
    """One-year bootstraps of whole calendar months: 12 months drawn at random from 2005-2025, with
    every trade that entered in them, in order. Keeps the clustering of positions that are open at
    the same time (which matters with 3-4 positions in one direction); 5-trade blocks may not."""
    m = tr["entry_time"].dt.tz_convert(None).dt.to_period("M")
    months = pd.period_range(start, end, freq="M")
    groups = [tr["R"].to_numpy(float)[(m == p).to_numpy()] for p in months]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(paths, 12))
    ret, dd = np.empty(paths), np.empty(paths)
    for i in range(paths):
        r = np.concatenate([groups[j] for j in pick[i]])
        eq = np.cumprod(np.maximum(1 + f * r, 0.0)) if len(r) else np.ones(1)
        peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
        ret[i] = eq[-1] - 1
        dd[i] = float(np.max(1 - eq / peak)) if len(r) else 0.0
    return {"med": round(float(np.median(ret)), 3), "p5": round(float(np.percentile(ret, 5)), 3),
            "p_double": round(float((ret >= 1).mean()), 3), "p_loss": round(float((ret < 0).mean()), 3),
            "p_dd30": round(float((dd >= 0.3).mean()), 3), "p_dd50": round(float((dd >= 0.5).mean()), 3)}


def analyse(tr: pd.DataFrame) -> dict:
    R = tr["R"].to_numpy(float)
    n_year = int(round(len(R) / YEARS))
    fk, gk = kelly(R)
    out = {"n": int(len(R)), "per_year": n_year, "per_week": round(n_year / 52.18, 2), "avgR": round(float(R.mean()), 3),
           "win%": round(100 * float((R > 0).mean()), 1), "kelly_f": round(fk, 3),
           "ceiling_median_year": round(float(np.expm1(gk * n_year)), 2), "grid": []}
    for f in F_GRID + [round(fk, 3), round(fk / 2, 3)]:
        path, mdd = compound_history(tr, f)
        yr = yearly_returns(path)
        b = bootstrap_year(R, n_year, f)
        out["grid"].append({"f": f, "hist_cagr": round(float(path.iloc[-1] ** (1 / YEARS) - 1), 3) if path.iloc[-1] > 0 else -1.0,
                            "hist_max_dd": round(mdd, 3), "hist_median_year": round(float(yr.median()), 3),
                            "hist_worst_year": round(float(yr.min()), 3), "hist_years_ge_100%": int((yr >= 1).sum()),
                            **{f"boot_{k}": round(v, 3) for k, v in b.items()}})
    # the smallest f on a fine grid whose bootstrapped median year reaches +100 %
    f_need = None
    for f in np.round(np.arange(0.005, min(0.5, 2 * fk + 0.05), 0.0025), 4):
        if bootstrap_year(R, n_year, float(f), paths=8000)["med"] >= 1.0:
            f_need = float(f)
            break
    out["f_for_median_100"] = f_need
    if f_need:
        path, mdd = compound_history(tr, f_need)
        yr = yearly_returns(path)
        out["at_f_need"] = {"f": f_need, "hist_cagr": round(float(path.iloc[-1] ** (1 / YEARS) - 1), 3), "hist_max_dd": round(mdd, 3),
                            "hist_years": {int(k): round(float(v), 3) for k, v in yr.items()},
                            **{f"boot_{k}": round(v, 3) for k, v in bootstrap_year(R, n_year, f_need).items()},
                            "lots_2000": lots_sim(tr, f_need)}
    out["boot_median_max"] = round(max(g["boot_med"] for g in out["grid"]), 3)
    out["fresh_2000_each_year"] = [year_by_year_2000(tr, f) for f in (0.02, 0.03, 0.05, 0.08, 0.10)]
    out["periods"] = period_stats(tr)
    out["month_boot"] = {str(f): bootstrap_months(tr, f) for f in (0.02, 0.025, 0.03, 0.04, 0.05)}
    return out


SINCE = {"2005-2025/3": "2005-01", "2013/7-2025/3": "2013-07", "2017-2025/3": "2017-01"}
F_RECENT = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)


def recent_view(tr: pd.DataFrame) -> dict:
    """The month-block bootstrap drawing from all months, and only from recent months: is +100 %
    still reachable when the strong 2005-2012 trends are left out? For each start: median / 5th
    percentile / chance of a 50 % fall at several risks, the smallest risk with a +100 % median,
    the highest median at any risk (the ceiling), and the best median with at most a 20 % chance
    of a 50 % fall."""
    out = {}
    for name, start in SINCE.items():
        t = tr[tr["entry_time"] >= pd.Timestamp(start + "-01", tz="UTC")].reset_index(drop=True)
        rows = {str(f): bootstrap_months(t, f, paths=8000, start=start, end="2025-02") for f in F_RECENT}
        need = next((f for f in F_RECENT if rows[str(f)]["med"] >= 1.0), None)
        ok = [f for f in F_RECENT if rows[str(f)]["p_dd50"] <= 0.20]
        best = max(ok, key=lambda f: rows[str(f)]["med"]) if ok else None
        top = max(F_RECENT, key=lambda f: rows[str(f)]["med"])
        out[name] = {"by_f": rows, "f_for_median_100": need, "best_f_with_dd50_le_20%": best,
                     "median_there": rows[str(best)]["med"] if best else None,
                     "ceiling_median": rows[str(top)]["med"], "ceiling_f": top}
    return out


def main():
    if "--recent" in sys.argv:            # recent-months bootstraps of the cached trades, merged into the results
        path = OUT / "XAUUSD_target_100.json"
        res = json.loads(path.read_text(encoding="utf-8"))
        for k in res["strategies"]:
            tr = pd.read_pickle(ROOT / "data" / "cache" / f"target100_{k}.pkl")
            res["strategies"][k]["recent"] = rv = recent_view(tr)
            for name, v in rv.items():
                print(f"{k:20s} {name}: f for +100% median {v['f_for_median_100']}, ceiling {v['ceiling_median']:+.0%} at f={v['ceiling_f']}, "
                      f"best with P(DD50)<=20%: f={v['best_f_with_dd50_le_20%']} median {v['median_there']}", flush=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    trades = strategy_trades(only=only)
    path = OUT / "XAUUSD_target_100.json"
    old = json.loads(path.read_text(encoding="utf-8")) if (only and path.exists()) else {"strategies": {}}
    res = {"how": ("Gold 15m 2005-2025/3 (broker data), trades from the live code (goldsignal.active.run_mode), costs and "
                   "overnight fees included. f = fraction of the account each trade risks (stop distance x size). "
                   "hist_* = the actual 2005-2025 sequence compounded; boot_* = one-year block bootstraps (5-trade blocks)."),
           "strategies": old["strategies"]}
    for k, tr in trades.items():
        res["strategies"][k] = a = analyse(tr)
        print(f"{k:16s} n={a['n']:4d} {a['per_week']:.2f}/wk avgR {a['avgR']:+.3f} kelly {a['kelly_f']:.3f} "
              f"ceiling median {a['ceiling_median_year']:+.0%}  f for +100% median: {a['f_for_median_100']}", flush=True)
        for y in a["fresh_2000_each_year"]:
            print(f"     fresh $2,000 each year at f={y['f']}: median {y['median']:+.0%}, worst {y['worst']:+.0%}, best {y['best']:+.0%}, "
                  f">=+100% in {y['years_ge_100%']}/{y['years']} years, lost in {y['years_lost']}, worst DD {y['worst_dd']:.0%}", flush=True)
        for pn, pr in a["periods"].items():
            print(f"     {pn}: {pr['per_week']}/wk avgR {pr['avgR']:+.3f} {pr['R_per_year']}R/yr maxDD {pr['max_dd_R']}R eff {pr['eff']} | "
                  + " ".join(f"f={f}: CAGR {pr[f'cagr_at_{f}']:+.0%} DD {pr[f'max_dd_at_{f}']:.0%}" for f in (0.025, 0.0325, 0.05)), flush=True)
        print("     month-block bootstrap: " + " | ".join(f"f={f}: med {v['med']:+.0%} p5 {v['p5']:+.0%} P(x2) {v['p_double']:.0%} "
                                                         f"P(DD50) {v['p_dd50']:.0%}" for f, v in a["month_boot"].items()), flush=True)
        if a.get("at_f_need"):
            z = a["at_f_need"]
            print(f"     at f={z['f']}: hist CAGR {z['hist_cagr']:+.0%}, hist maxDD {z['hist_max_dd']:.0%}, boot median {z['boot_med']:+.0%}, "
                  f"p5 {z['boot_p5']:+.0%}, P(DD>=50%) {z['boot_p_dd50']:.0%}, P(DD>=80%) {z['boot_p_dd80']:.0%}; $2,000 lots: {z['lots_2000']}", flush=True)
        pd.to_pickle(tr, ROOT / "data" / "cache" / f"target100_{k}.pkl")
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
