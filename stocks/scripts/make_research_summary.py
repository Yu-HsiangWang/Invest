"""Build results/research_summary.json for the dashboard from the research outputs.

    python scripts/make_research_summary.py
Contents: pooled forward-return tables by zone/trend (per market, 6 & 12 months), the P/E study (TW),
strategy comparisons (median across tickers), ETF defense table, broad-universe check.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.common import RESULTS, load_daily, period_of  # noqa: E402
from stocksignal import zones as Z  # noqa: E402

OUT = RESULTS / "research_summary.json"


def pooled_tables(market: str) -> dict:
    fw = pd.read_csv(RESULTS / f"{market}_zone_forward.csv")
    out = {}
    for method in ("channel", "deviation", "trend", "mom"):
        out[method] = {}
        for h in (126, 252):
            sub = fw[(fw.zone_method == method) & (fw.horizon == h) & (fw.period == "ALL")]
            out[method][str(h)] = {str(int(float(r.state))): {
                "n": int(r.n), "n_eff": float(r.n_eff), "n_sym": int(r.n_sym), "hit": round(float(r.hit), 3), "base_hit": round(float(r.base_hit), 3),
                "mean": round(float(r["mean"]), 4), "median": round(float(r["median"]), 4), "excess_mean": round(float(r.excess_mean), 4),
                "by_period": {p: round(float(x), 4) for p, x in fw[(fw.zone_method == method) & (fw.horizon == h) & (fw.state == r.state)].set_index("period")["excess_mean"].items()}}
                for _, r in sub.iterrows()}
    return out


def pe_study_tw() -> dict:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "daily" / "*.valuation.csv*"))):
        sym = os.path.basename(f).replace(".valuation.csv.gz", "").replace(".valuation.csv", "")
        v = pd.read_csv(f, index_col=0, parse_dates=True)
        d = load_daily(sym)
        if d is None:
            continue
        close = d["close_raw"].where(d["close_raw"].notna(), d["close"])
        pe = v["pe"].reindex(d.index).ffill()
        z = Z.pe_zone(Z.pe_bands(close, pe))
        for h in (126, 252):
            fwd = d["close"].shift(-h) / d["close"] - 1
            rows.append(pd.DataFrame({"fwd": fwd, "z": z, "per": period_of(d.index), "sym": sym, "h": h}).dropna())
    if not rows:
        return {}
    x = pd.concat(rows)
    out = {}
    for h in (126, 252):
        xh = x[x.h == h]
        base = xh.fwd.mean()
        bh = (xh.fwd > 0).mean()
        out[str(h)] = {}
        for zv, g in xh.groupby("z"):
            per = {p: round(float(gg.fwd.mean() - xh[xh.per == p].fwd.mean()), 4) for p, gg in g.groupby("per") if len(xh[xh.per == p]) > 100}
            out[str(h)][str(int(zv))] = {"n": int(len(g)), "n_eff": round(len(g) / h, 1), "n_sym": int(g.sym.nunique()), "hit": round(float((g.fwd > 0).mean()), 3),
                                        "base_hit": round(float(bh), 3), "mean": round(float(g.fwd.mean()), 4), "median": round(float(g.fwd.median()), 4),
                                        "excess_mean": round(float(g.fwd.mean() - base), 4), "by_period": per}
    return out


def strategies(market: str) -> dict:
    js = json.loads((RESULTS / f"{market}_zone_summary.json").read_text(encoding="utf-8"))["strategies"]
    names = {"buy_hold": "買進持有", "faber10m": "10 月均線（月底判斷）", "trend200": "200 日均線（每日）", "chan_in-1_out+1": "五線譜 −1σ 買／+1σ 賣",
             "chan_in-1_out+2": "五線譜 −1σ 買／+2σ 賣", "chan_scaled": "五線譜分批", "dev_scaled": "年線乖離分批", "dev_in-1_out+2": "年線乖離 −1 買／+2 賣"}
    out = {}
    for key, v in js.items():
        if "|" not in key:
            continue
        name, per = key.split("|")
        if name not in names:
            continue
        out.setdefault(names[name], {})[per] = {k: round(float(x), 4) for k, x in v.items()}
    return out


def etf_table(market: str) -> list[dict]:
    st = pd.read_csv(RESULTS / f"{market}_zone_strategies.csv")
    sub = st[(st.kind == "etf") & (st.period == "ALL") & (st.strategy.isin(["buy_hold", "faber10m"]))]
    rows = []
    for sym, g in sub.groupby("sym"):
        bh = g[g.strategy == "buy_hold"].iloc[0]
        fb = g[g.strategy == "faber10m"].iloc[0]
        rows.append({"sym": sym, "years": round(float(bh.years), 1), "bh_cagr": round(float(bh.cagr), 4), "bh_dd": round(float(bh.max_dd), 4),
                     "faber_cagr": round(float(fb.cagr), 4), "faber_dd": round(float(fb.max_dd), 4)})
    return rows


def broad(market: str) -> dict:
    p = RESULTS / f"broad_forward_{market}.csv"
    if not p.exists():
        return {}
    b = pd.read_csv(p)
    out = {}
    for method in ("channel", "trend", "mom"):
        sub = b[(b.horizon == 252) & (b.method == method)]
        out[method] = {}
        for stv, g in sub.groupby("state"):
            out[method][str(int(float(stv)))] = {"excess_by_period": {r.period: round(float(r.excess_mean), 4) for _, r in g.iterrows()},
                                          "hit": round(float(g[g.period == "ALL"].hit.iloc[0]), 3) if (g.period == "ALL").any() else None,
                                          "base_hit": round(float(g[g.period == "ALL"].base_hit.iloc[0]), 3) if (g.period == "ALL").any() else None,
                                          "n_sym": int(g[g.period == "ALL"].n_sym.iloc[0]) if (g.period == "ALL").any() else None}
    return out


def main():
    summary = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"), "markets": {}}
    for m in ("TW", "US"):
        summary["markets"][m] = {"pooled": pooled_tables(m), "strategies": strategies(m), "etf": etf_table(m), "broad": broad(m)}
    summary["markets"]["TW"]["pooled"]["pe"] = pe_study_tw()
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
