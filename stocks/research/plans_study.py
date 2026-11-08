"""Study 3 (exploratory, decided AFTER seeing study 1): practical plans a long-only investor can follow.

The pre-registered in/out zone strategies lost to buy-and-hold (study 1). Here we test the two
ways the evidence says zones should be used instead:

A. "Reserve + buy-the-dip" DCA: invest a fixed amount monthly, but keep part of it in a cash
   reserve and deploy the reserve when the stock is in a cheap zone (-1: x2, -2: x3 of the monthly
   amount). No selling. Compare final wealth with plain DCA (same total contributions).
B. "Defensive hold": stay invested; only when the MARKET (index/ETF) 10-month SMA trend is down AND
   the stock is not in a cheap zone, cut exposure to 50%. Compare CAGR / max drawdown / Calmar
   with buy-and-hold and with the plain 10-month SMA filter.

    python research/plans_study.py --market TW
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import RESULTS, available, load_daily, period_of  # noqa: E402
from stocksignal import zones as Z  # noqa: E402
from stocksignal.backtest import TW_ETF, TW_STOCK, US_STOCK, metrics, simulate  # noqa: E402
from stocksignal.indicators import sma  # noqa: E402
from stocksignal.universe import by_market  # noqa: E402


def dca_reserve(df: pd.DataFrame, zone: pd.Series, costs, monthly: float = 1.0, reserve_frac: float = 0.3,
                mult_cheap: float = 2.0, mult_very_cheap: float = 3.0, max_reserve_months: float = 12.0) -> pd.DataFrame:
    """Monthly contribution `monthly`; invest (1-reserve_frac) normally, park the rest; when in a cheap
    zone at month start, invest up to mult x monthly from the reserve. Reserve capped at N months
    (beyond the cap the excess is invested anyway)."""
    z = zone.reindex(df.index).ffill()
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    month = df.index.to_period("M")
    first = np.r_[True, month[1:] != month[:-1]]
    shares = wallet = invested = 0.0
    eq = np.empty(len(df))
    for t in range(len(df)):
        if first[t] and t > 0:
            invested += monthly
            wallet += monthly
            zt = z.iloc[t - 1]
            if zt <= -2:
                spend = min(wallet, monthly * mult_very_cheap)
            elif zt <= -1:
                spend = min(wallet, monthly * mult_cheap)
            else:
                spend = min(wallet, monthly * (1 - reserve_frac))
            if wallet - spend > monthly * max_reserve_months:
                spend = wallet - monthly * max_reserve_months
            if spend > 0:
                shares += spend / (1 + costs.buy()) / o[t]
                wallet -= spend
        eq[t] = shares * c[t] + wallet
    return pd.DataFrame({"equity": eq, "invested": invested, "close": c}, index=df.index)


def dca_plain(df: pd.DataFrame, costs, monthly: float = 1.0) -> pd.DataFrame:
    return dca_reserve(df, pd.Series(0.0, index=df.index), costs, monthly, reserve_frac=0.0)


def run(market: str) -> dict:
    inst = by_market(market)
    frames = available([i.symbol for i in inst])
    kinds = {i.symbol: i.kind for i in inst}
    bench_sym = "^TWII" if market == "TW" else "SPY"
    bench = load_daily(bench_sym)["close"]
    # market trend: month-end close vs 10-month SMA, evaluated monthly (Faber), with the market index
    me = bench.index.to_period("M")
    is_me = np.r_[me[1:] != me[:-1], True]
    mclose = bench[is_me]
    mtrend = (mclose > mclose.rolling(10).mean()).astype(float).where(mclose.rolling(10).mean().notna())
    market_up = mtrend.reindex(bench.index).ffill()
    rows = []
    for sym, df in frames.items():
        costs = (TW_ETF if kinds[sym] == "etf" else TW_STOCK) if market == "TW" else US_STOCK
        c = df["close"]
        zc = Z.channel_zone(Z.channel_bands(c)["zscore"])
        zd = Z.deviation_zone(Z.deviation_bands(c)["dev_rank"])
        zone = pd.concat([zc, zd], axis=1).min(axis=1)                 # cheapest of the two methods
        mu = market_up.reindex(df.index).ffill()
        # B. defensive hold
        s200 = sma(c, 200)
        own_faber = pd.Series(np.nan, index=df.index)
        me2 = df.index.to_period("M")
        is_me2 = np.r_[me2[1:] != me2[:-1], True]
        mc2 = c[is_me2]
        own_faber[is_me2] = (mc2 > mc2.rolling(10).mean()).astype(float).where(mc2.rolling(10).mean().notna())
        own_faber = own_faber.ffill().fillna(1.0)
        targets = {
            "buy_hold": pd.Series(1.0, index=df.index),
            "faber10m_own": own_faber,
            "market_faber": mu.fillna(1.0),
            "defensive50": pd.Series(np.where((mu == 0) & (zone > -1), 0.5, 1.0), index=df.index),
            "defensive50_own": pd.Series(np.where((own_faber == 0) & (zone > -1), 0.5, 1.0), index=df.index),
            "defensive0_cheap_stays": pd.Series(np.where((mu == 0) & (zone > -1), 0.0, 1.0), index=df.index),
        }
        per = period_of(df.index)
        for name, tgt in targets.items():
            for p in ("IS", "VAL", "TEST", "ALL"):
                m = (per == p) if p != "ALL" else pd.Series(True, index=df.index)
                sub = df[m]
                if len(sub) < 400:
                    continue
                res = simulate(sub, tgt[m], costs)
                met = metrics(res, bench=sub["close"] / sub["close"].iloc[0])
                rows.append({"sym": sym, "kind": kinds[sym], "plan": name, "period": p, **met})
        # A. DCA variants (full history, and TEST-only for a recent view)
        for p in ("ALL", "TEST"):
            m = (per == p) if p != "ALL" else pd.Series(True, index=df.index)
            sub = df[m]
            if len(sub) < 750:
                continue
            plain = dca_plain(sub, costs)
            for rf, mc, mvc, tag in ((0.2, 2.0, 3.0, "r20"), (0.3, 2.0, 3.0, "r30"), (0.3, 3.0, 4.0, "r30x3")):
                tilt = dca_reserve(sub, zone[m], costs, reserve_frac=rf, mult_cheap=mc, mult_very_cheap=mvc)
                rows.append({"sym": sym, "kind": kinds[sym], "plan": f"dca_reserve_{tag}", "period": p,
                             "final_multiple": tilt["equity"].iloc[-1] / plain["equity"].iloc[-1],
                             "cagr": tilt["equity"].iloc[-1] / tilt["invested"].iloc[-1] - 1,
                             "bench_cagr": plain["equity"].iloc[-1] / plain["invested"].iloc[-1] - 1,
                             "max_dd": float((tilt["equity"] / tilt["equity"].cummax() - 1).min()),
                             "bench_max_dd": float((plain["equity"] / plain["equity"].cummax() - 1).min())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"{market}_plans.csv", index=False)
    summ = {}
    for (plan, p), g in out.groupby(["plan", "period"]):
        if plan.startswith("dca"):
            summ[f"{plan}|{p}"] = {"n": len(g), "median_final_ratio": float(g["final_multiple"].median()), "share_better": float((g["final_multiple"] > 1).mean()),
                                  "median_gain_vs_contrib": float(g["cagr"].median()), "median_plain_gain": float(g["bench_cagr"].median()),
                                  "median_dd": float(g["max_dd"].median()), "median_plain_dd": float(g["bench_max_dd"].median())}
        else:
            summ[f"{plan}|{p}"] = {"n": len(g), "median_cagr": float(g["cagr"].median()), "median_bench_cagr": float(g["bench_cagr"].median()),
                                  "median_excess_cagr": float(g["excess_cagr"].median()), "share_beating_bh": float((g["excess_cagr"] > 0).mean()),
                                  "median_maxdd": float(g["max_dd"].median()), "median_bench_maxdd": float(g["bench_max_dd"].median()),
                                  "median_dd_reduction": float((1 - g["max_dd"] / g["bench_max_dd"]).median()),
                                  "median_calmar": float(g["calmar"].median()), "median_time_in_market": float(g["time_in_market"].median())}
    (RESULTS / f"{market}_plans_summary.json").write_text(json.dumps({"market": market, "plans": summ}, ensure_ascii=False, indent=1), encoding="utf-8")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW")
    a = ap.parse_args()
    summ = run(a.market)
    pd.set_option("display.width", 250)
    df = pd.DataFrame(summ).T
    df["plan"] = [i.split("|")[0] for i in df.index]
    df["per"] = [i.split("|")[1] for i in df.index]
    core = df[~df.plan.str.startswith("dca")]
    print(core.pivot_table(index="plan", columns="per", values=["median_excess_cagr", "median_dd_reduction", "median_calmar"]).astype(float).round(2).to_string())
    d = df[df.plan.str.startswith("dca")]
    print(d[["plan", "per", "n", "median_final_ratio", "share_better", "median_gain_vs_contrib", "median_plain_gain", "median_dd", "median_plain_dd"]].to_string())


if __name__ == "__main__":
    main()
