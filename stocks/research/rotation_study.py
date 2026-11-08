"""Study 4 (exploratory): when new money arrives every month, which names on the watch-list should get it?

Monthly contribution split (a) equally across all names with data, (b) only across the cheapest
third by channel z-score, (c) only across the most expensive third, (d) cheapest third by
200-day deviation rank, (e) top third by 12-1 momentum (the classic alternative), (f) bottom third
by momentum. Long-only, never sells. Compares final wealth per unit contributed, per period.

    python research/rotation_study.py --market TW
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.common import RESULTS, available  # noqa: E402
from stocksignal import zones as Z  # noqa: E402
from stocksignal.backtest import TW_ETF, TW_STOCK, US_STOCK  # noqa: E402
from stocksignal.indicators import momentum  # noqa: E402
from stocksignal.universe import by_market  # noqa: E402

PERIODS = {"IS": ("2000-01-01", "2012-12-31"), "VAL": ("2013-01-01", "2018-12-31"), "TEST": ("2019-01-01", "2026-12-31"),
           "ALL": ("2000-01-01", "2026-12-31")}


def run(market: str, n_boot: int = 300) -> dict:
    inst = by_market(market)
    frames = available([i.symbol for i in inst])
    kinds = {i.symbol: i.kind for i in inst}
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()
    opens = pd.DataFrame({s: f["open"] for s, f in frames.items()}).reindex(closes.index)
    z = pd.DataFrame({s: Z.channel_bands(f["close"])["zscore"] for s, f in frames.items()}).reindex(closes.index)
    dr = pd.DataFrame({s: Z.deviation_bands(f["close"])["dev_rank"] for s, f in frames.items()}).reindex(closes.index)
    mom = pd.DataFrame({s: momentum(f["close"]) for s, f in frames.items()}).reindex(closes.index)
    month = closes.index.to_period("M")
    first = np.r_[True, month[1:] != month[:-1]]
    buy_cost = {s: ((TW_ETF if kinds[s] == "etf" else TW_STOCK) if market == "TW" else US_STOCK).buy() for s in frames}
    out = {}
    rng = np.random.default_rng(0)
    for per, (a, b) in PERIODS.items():
        idx = closes.index[(closes.index >= a) & (closes.index <= b)]
        if len(idx) < 500:
            continue
        rules = {"equal": None, "cheap_third_channel": ("z", "low"), "rich_third_channel": ("z", "high"),
                 "cheap_third_dev": ("dr", "low"), "rich_third_dev": ("dr", "high"), "mom_top_third": ("mom", "high"), "mom_bottom_third": ("mom", "low")}
        res = {}
        picks_log = []
        for name, rule in rules.items():
            shares = pd.Series(0.0, index=closes.columns)
            invested = 0.0
            for t in idx[first[closes.index.get_indexer(idx)]]:
                i = closes.index.get_loc(t)
                if i == 0:
                    continue
                prev = closes.index[i - 1]
                avail = closes.columns[closes.loc[prev].notna() & opens.loc[t].notna() & z.loc[prev].notna()]
                if len(avail) < 3:
                    continue
                if rule is None:
                    chosen = list(avail)
                else:
                    key = {"z": z, "dr": dr, "mom": mom}[rule[0]].loc[prev, avail].dropna()
                    if len(key) < 3:
                        continue
                    k = max(1, len(key) // 3)
                    chosen = list(key.nsmallest(k).index if rule[1] == "low" else key.nlargest(k).index)
                amt = 1.0 / len(chosen)
                for s in chosen:
                    shares[s] += amt / (1 + buy_cost[s]) / opens.loc[t, s]
                invested += 1.0
                if name != "equal":
                    picks_log.append({"date": t, "rule": name, "n": len(chosen)})
            last = closes.loc[idx].ffill().iloc[-1]                  # each name valued at its own last available close
            final = float((shares * last.fillna(0)).sum())
            res[name] = {"invested": invested, "final": final, "multiple": final / invested if invested else np.nan}
        # placebo: random third each month
        rand = []
        for _ in range(n_boot):
            shares = pd.Series(0.0, index=closes.columns)
            invested = 0.0
            for t in idx[first[closes.index.get_indexer(idx)]]:
                i = closes.index.get_loc(t)
                if i == 0:
                    continue
                prev = closes.index[i - 1]
                avail = closes.columns[closes.loc[prev].notna() & opens.loc[t].notna() & z.loc[prev].notna()]
                if len(avail) < 3:
                    continue
                k = max(1, len(avail) // 3)
                chosen = rng.choice(avail, size=k, replace=False)
                amt = 1.0 / k
                for s in chosen:
                    shares[s] += amt / (1 + buy_cost[s]) / opens.loc[t, s]
                invested += 1.0
            last = closes.loc[idx].ffill().iloc[-1]
            rand.append(float((shares * last.fillna(0)).sum()) / invested)
        rand = np.array(rand)
        for name in res:
            res[name]["placebo_pct"] = float((rand < res[name]["multiple"]).mean())
        res["random_third"] = {"multiple_median": float(np.median(rand)), "p5": float(np.percentile(rand, 5)), "p95": float(np.percentile(rand, 95))}
        out[per] = res
    (RESULTS / f"{market}_rotation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    out = run(a.market, a.boot)
    for per, res in out.items():
        print(per)
        for k, v in res.items():
            if k == "random_third":
                print(f"   random third: median x{v['multiple_median']:.2f} (5-95%: {v['p5']:.2f}-{v['p95']:.2f})")
            else:
                print(f"   {k:22s} final/invested x{v['multiple']:.2f}   beats {v['placebo_pct']*100:.0f}% of random picks")


if __name__ == "__main__":
    main()
