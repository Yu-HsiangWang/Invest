"""Study 1: do the buy / sell zones and the trend state say anything about the next 3-12 months?

Pre-registered (docs/RESEARCH.md section 0): parameters fixed, periods IS <=2012 / VAL 2013-18 /
TEST 2019+, pooled across the tickers of a market; placebo = circular shifts of the zone series.

    python research/zone_study.py --market TW
    python research/zone_study.py --market US

Outputs results/<MARKET>_zone_forward.csv, results/<MARKET>_zone_strategies.csv,
results/<MARKET>_zone_placebo.csv, results/<MARKET>_zone_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import RESULTS, available, period_of  # noqa: E402
from stocksignal import zones as Z
from stocksignal.backtest import TW_ETF, TW_STOCK, US_STOCK, dca, metrics, simulate
from stocksignal.indicators import momentum, sma
from stocksignal.universe import by_market

HORIZONS = (63, 126, 252)


def states_for(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    ch = Z.channel_bands(c)
    dv = Z.deviation_bands(c)
    st = pd.DataFrame(index=df.index)
    st["channel"] = Z.channel_zone(ch["zscore"])
    st["deviation"] = Z.deviation_zone(dv["dev_rank"])
    st["trend"] = Z.trend_state(c)
    mom = momentum(c)
    st["mom"] = np.sign(mom).where(mom.notna())
    st["above200"] = (c > sma(c, 200)).astype(float).where(sma(c, 200).notna())
    # combos used by the dashboard's direction label
    st["trend_mom"] = (st["trend"].astype("Int64").astype(str) + "/" + st["mom"].astype("Int64").astype(str)).where(st["trend"].notna() & st["mom"].notna())
    st["zscore"] = ch["zscore"]
    st["dev_rank"] = dv["dev_rank"]
    return st


def forward_table(frames: dict[str, pd.DataFrame], states: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        pooled = []
        for sym, df in frames.items():
            c = df["close"]
            fwd = c.shift(-h) / c - 1.0
            d = pd.DataFrame({"fwd": fwd, "state": states[sym][col], "period": period_of(df.index), "sym": sym}).dropna()
            pooled.append(d)
        d = pd.concat(pooled)
        for per in ("IS", "VAL", "TEST", "ALL"):
            dd = d if per == "ALL" else d[d["period"] == per]
            if len(dd) < 50:
                continue
            base = dd["fwd"]
            for stv, g in dd.groupby("state"):
                if len(g) < 40:
                    continue
                rows.append({"zone_method": col, "horizon": h, "period": per, "state": stv, "n": len(g),
                             "n_eff": round(len(g) / h, 1), "n_sym": g["sym"].nunique(),
                             "mean": g["fwd"].mean(), "median": g["fwd"].median(), "hit": (g["fwd"] > 0).mean(),
                             "base_mean": base.mean(), "base_hit": (base > 0).mean(),
                             "excess_mean": g["fwd"].mean() - base.mean(), "excess_hit": (g["fwd"] > 0).mean() - (base > 0).mean()})
    return pd.DataFrame(rows)


def placebo(frames: dict[str, pd.DataFrame], states: dict[str, pd.DataFrame], col: str, target_state: float,
            h: int = 126, n_iter: int = 1000, seed: int = 0) -> dict:
    """Circular-shift placebo: how often does a shifted copy of the zone series beat the real excess return?"""
    rng = np.random.default_rng(seed)
    real_num = real_den = 0.0
    sims = np.zeros(n_iter)
    dens = np.zeros(n_iter)
    for sym, df in frames.items():
        c = df["close"]
        fwd = (c.shift(-h) / c - 1.0).to_numpy()
        st = states[sym][col].to_numpy()
        ok = ~np.isnan(fwd) & ~np.isnan(st)
        fwd_ok, st_ok = fwd[ok], st[ok]
        mask = st_ok == target_state
        if mask.sum() < 20:
            continue
        base = fwd_ok.mean()
        real_num += (fwd_ok[mask] - base).sum()
        real_den += mask.sum()
        n = len(st_ok)
        if n <= 2 * h + 10:
            continue
        shifts = rng.integers(h, n - h, size=n_iter)
        for i, s in enumerate(shifts):
            m = np.roll(mask, s)
            sims[i] += (fwd_ok[m] - base).sum()
            dens[i] += m.sum()
    if real_den == 0:
        return {"zone_method": col, "state": target_state, "h": h, "real_excess": np.nan, "placebo_p": np.nan}
    real = real_num / real_den
    sim = sims / np.where(dens > 0, dens, np.nan)
    p = float(np.nanmean(sim >= real)) if target_state < 0 else float(np.nanmean(sim <= real))
    return {"zone_method": col, "state": target_state, "h": h, "real_excess": real, "placebo_median": float(np.nanmedian(sim)),
            "placebo_p": p, "n_days": int(real_den)}


# ----------------------------------------------------------------------------- strategies
def strategy_targets(df: pd.DataFrame, st: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["close"]
    s200 = sma(c, 200)
    idx = df.index
    out = {}
    out["buy_hold"] = pd.Series(1.0, index=idx)
    # daily 200MA filter
    out["trend200"] = (c > s200).astype(float).where(s200.notna(), 0.0)
    # Faber: month-end close vs 10-month SMA (evaluated at the last trading day of each month)
    me = idx.to_period("M")
    is_me = np.r_[me[1:] != me[:-1], True]
    m_close = c[is_me]
    sma10m = m_close.rolling(10).mean()
    sig = (m_close > sma10m).astype(float).where(sma10m.notna(), 0.0)
    out["faber10m"] = sig.reindex(idx).ffill().fillna(0.0)
    # Zone strategies: in at cheap, out at rich (hold otherwise)
    for col, name in (("channel", "chan"), ("deviation", "dev")):
        z = st[col]
        pos = pd.Series(np.nan, index=idx)
        pos[z <= -1] = 1.0
        pos[z >= 1] = 0.0
        pos = pos.ffill().fillna(0.0)
        out[f"{name}_in-1_out+1"] = pos
        pos2 = pd.Series(np.nan, index=idx)
        pos2[z <= -1] = 1.0
        pos2[z >= 2] = 0.0
        out[f"{name}_in-1_out+2"] = pos2.ffill().fillna(0.0)
        # with a trend stop: also go to cash when below the 200MA and the MA is falling
        tr = st["trend"]
        pos3 = pos.copy()
        pos3[tr == -1] = 0.0
        out[f"{name}_in-1_out+1_trendstop"] = pos3
        # scaled: 50% at -1, 100% at -2, trim to 50% at +1, 0 at +2
        sc = pd.Series(np.nan, index=idx)
        sc[z == -1] = 0.5
        sc[z <= -2] = 1.0
        sc[z == 1] = 0.5
        sc[z >= 2] = 0.0
        out[f"{name}_scaled"] = sc.ffill().fillna(0.0)
    return out


def run_market(market: str, n_placebo: int) -> dict:
    inst = by_market(market)
    frames = available([i.symbol for i in inst])
    print(f"{market}: {len(frames)} / {len(inst)} symbols with data")
    states = {s: states_for(df) for s, df in frames.items()}
    kinds = {i.symbol: i.kind for i in inst}

    fw = pd.concat([forward_table(frames, states, col) for col in ("channel", "deviation", "trend", "mom", "trend_mom")])
    fw.to_csv(RESULTS / f"{market}_zone_forward.csv", index=False)

    pl = []
    for col in ("channel", "deviation"):
        for stv in (-2, -1, 1, 2):
            for h in (126, 252):
                pl.append(placebo(frames, states, col, stv, h=h, n_iter=n_placebo))
    pl = pd.DataFrame(pl)
    pl.to_csv(RESULTS / f"{market}_zone_placebo.csv", index=False)

    rows = []
    for sym, df in frames.items():
        costs = (TW_ETF if kinds[sym] == "etf" else TW_STOCK) if market == "TW" else US_STOCK
        per = period_of(df.index)
        tg = strategy_targets(df, states[sym])
        bench = None
        for name, target in tg.items():
            for p in ("IS", "VAL", "TEST", "ALL"):
                m = (per == p) if p != "ALL" else pd.Series(True, index=df.index)
                sub = df[m]
                if len(sub) < 400:
                    continue
                res = simulate(sub, target[m], costs)
                bh = sub["close"] / sub["close"].iloc[0]
                met = metrics(res, bench=bh)
                rows.append({"sym": sym, "kind": kinds[sym], "strategy": name, "period": p, **met})
        # DCA variants on the full history
        for col, name in (("channel", "chan"), ("deviation", "dev")):
            z = states[sym][col]
            w = pd.Series(1.0, index=df.index)
            w[z <= -1] = 2.0
            w[z <= -2] = 3.0
            w[z >= 1] = 0.5
            w[z >= 2] = 0.0
            plain = dca(df, pd.Series(1.0, index=df.index), costs)
            tilt = dca(df, w, costs)
            rows.append({"sym": sym, "kind": kinds[sym], "strategy": f"dca_{name}_tilt_vs_plain", "period": "ALL",
                         "final_multiple": tilt["equity"].iloc[-1] / plain["equity"].iloc[-1],
                         "cagr": tilt["equity"].iloc[-1] / tilt["invested"].iloc[-1] - 1,
                         "bench_cagr": plain["equity"].iloc[-1] / plain["invested"].iloc[-1] - 1})
    strat = pd.DataFrame(rows)
    strat.to_csv(RESULTS / f"{market}_zone_strategies.csv", index=False)

    # summary: median across symbols of excess CAGR and DD reduction per strategy/period
    summ = {}
    core = strat[~strat["strategy"].str.startswith("dca")]
    for (name, p), g in core.groupby(["strategy", "period"]):
        summ[f"{name}|{p}"] = {"n": int(len(g)), "median_cagr": float(g["cagr"].median()), "median_bench_cagr": float(g["bench_cagr"].median()),
                               "median_excess_cagr": float(g["excess_cagr"].median()), "share_beating_bh": float((g["excess_cagr"] > 0).mean()),
                               "median_maxdd": float(g["max_dd"].median()), "median_bench_maxdd": float(g["bench_max_dd"].median()),
                               "median_dd_reduction": float((1 - g["max_dd"] / g["bench_max_dd"]).median()),
                               "median_time_in_market": float(g["time_in_market"].median()),
                               "median_calmar": float(g["calmar"].median())}
    d = strat[strat["strategy"].str.startswith("dca")]
    for name, g in d.groupby("strategy"):
        summ[name] = {"n": int(len(g)), "median_final_ratio": float(g["final_multiple"].median()), "share_better": float((g["final_multiple"] > 1).mean())}
    out = {"market": market, "symbols": sorted(frames), "strategies": summ,
           "placebo": pl.to_dict(orient="records"),
           "forward": fw[(fw["horizon"] == 126)].to_dict(orient="records")}
    (RESULTS / f"{market}_zone_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW")
    ap.add_argument("--placebo", type=int, default=500)
    a = ap.parse_args()
    out = run_market(a.market, a.placebo)
    pd.set_option("display.width", 200)
    fw = pd.read_csv(RESULTS / f"{a.market}_zone_forward.csv")
    print(fw[(fw.horizon == 126) & (fw.zone_method.isin(["channel", "deviation", "trend"]))]
          .pivot_table(index=["zone_method", "state"], columns="period", values="excess_mean").round(3))
    print(pd.DataFrame(out["placebo"]).round(3))
    s = pd.DataFrame(out["strategies"]).T
    print(s[["median_excess_cagr", "share_beating_bh", "median_dd_reduction", "median_calmar", "median_time_in_market"]].round(3))


if __name__ == "__main__":
    main()
