"""Money-management study for a small manual account (docs/RESEARCH.md section 8).

Part 1  trade management in R: same-day exits, breakeven moves, scale-outs, entry cut-offs.
Part 2  one-year account simulations in dollars ($2,000 / $5,000 accounts, 0.01-lot steps,
        100x leverage, 50 % margin stop-out) -> results/<SYMBOL>_money.json (dashboard).

    python research/position_sizing.py --source kaggle                       # gold 2005-2025/3
    python research/position_sizing.py --source dukascopy --symbol XAUUSD    # gold 2013-2026 + money table
    python research/position_sizing.py --source dukascopy --symbol XAGUSD    # silver 2013-2026 + money table

The Dukascopy runs download ~13 years of 1-minute bars on first use (cached in data/cache).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import metrics  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock, Plan, simulate_plan  # noqa: E402

OUT = ROOT / "results"
NY = "America/New_York"


# ----------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------

def load(source: str, sym: str) -> tuple[pd.DataFrame, str]:
    """15m bars and the first entry date used for statistics (indicator warm-up before it)."""
    if source == "kaggle":
        if sym != "XAUUSD":
            raise SystemExit("the Kaggle file only has gold")
        pq = ROOT / "data" / "cache_xau15.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
        else:
            from goldsignal.bars import load_kaggle_15m
            df = load_kaggle_15m(ROOT / "data" / "raw" / "XAU_15m_data.csv")
        return df.loc[:"2025-03-21"], "2005-01-01"
    from research.silver_check import load as duka_load  # cached Dukascopy download
    return duka_load(sym, date(2013, 1, 1), None), "2013-07-01"


def periods_for(source: str) -> dict:
    if source == "kaggle":
        return {"2005-16": ("2005-01-01", "2016-12-31"), "2017-20": ("2017-01-01", "2020-12-31"),
                "2021-25/3": ("2021-01-01", "2025-03-21")}
    return {"2013-19": ("2013-07-01", "2019-12-31"), "2020-22": ("2020-01-01", "2022-12-31"),
            "2023-26": ("2023-01-01", "2026-12-31")}


# ----------------------------------------------------------------------------------------
# part 1: trade management in R
# ----------------------------------------------------------------------------------------

def variants(sym: str) -> list[tuple[str, S.StrategyParams, Plan]]:
    inst = INSTRUMENTS[sym]
    base = inst.strategy_params(False)      # the app's entry rules (gold: with the 布林收窄濾網)
    day = inst.strategy_params(True)
    noon = replace(base, day_cutoff_ny=12.0)
    keep = inst.eod_keep_r
    return [
        ("A 原策略（可一直抱著）", base, Plan()),
        ("B 每天 16:45 全部平倉", base, Plan(eod="close")),
        ("C 16:45 只留停損已過成本的單", base, Plan(eod="keep_locked")),
        ("D 16:45 還在虧損才平", base, Plan(eod="close_losers", eod_k=0.0)),
        ("E 16:45 未達 +0.25R 就平", base, Plan(eod="close_losers", eod_k=0.25)),
        ("F 16:45 未達 +0.5R 就平", base, Plan(eod="close_losers", eod_k=0.5)),
        ("G +1R 後停損移到成本", base, Plan(be_trigger_r=1.0)),
        ("H +1R 平一半、其餘保本跟蹤", base, Plan(legs=(1.0, 0.0), be_trigger_r=1.0)),
        ("I +1R/+2R 各平 1/3、其餘跟蹤", base, Plan(legs=(1.0, 2.0, 0.0), be_trigger_r=1.0)),
        ("J +2R 先平 1/3、其餘跟蹤", base, Plan(legs=(2.0, 0.0, 0.0))),
        ("K 週五收盤前全部平倉", base, Plan(eod="friday")),
        ("L 紐約 12:00 後不開新倉", noon, Plan()),
        ("M 當日平倉模式（程式預設）", day, Plan(eod="close_losers", eod_k=keep)),
        ("N M + 16:45 留倉時停損收到 -0.5R", day, Plan(eod="close_losers", eod_k=keep, eod_tighten_r=-0.5)),
        ("O M + +2R 先平 1/3", day, Plan(eod="close_losers", eod_k=keep, legs=(2.0, 0.0, 0.0))),
    ]


def run_variant(df, f, clock, params, plan, sym) -> pd.DataFrame:
    inst = INSTRUMENTS[sym]
    t = S.rule_table(df, f, params)
    sig = S.signals_from_rules(t)
    px = df["open"].to_numpy(float)
    plan = replace(plan, trail_mult=params.trail_atr_h1, stop_mult=params.stop_atr_h1, cooldown=params.cooldown_bars, swap=True)
    return simulate_plan(df, sig, f["h1_atr"].to_numpy(float), plan, px * inst.spread_pct, px * inst.slip_pct, clock)


def summarize(tr: pd.DataFrame, label: str, start: str, periods: dict) -> dict:
    tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp(start, tz="UTC"))]
    span = (tr["entry_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 365.25
    m = metrics(tr, span)
    row = {"plan": label, "n/yr": round(len(tr) / span, 1), "win%": m["win%"], "avgR": m["avgR"], "PF": m["PF"],
           "maxDD_R": m["maxDD_R"], "streak": m["max_losing_streak"],
           "overnight%": round(100 * float((tr["nights"] > 0).mean())), "hold_h": round(float(tr["bars"].median()) / 4, 1)}
    for name, (a, b) in periods.items():
        k = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        row[name] = round(float(tr.loc[k, "R"].mean()), 3)
    return row


def hour_table(tr: pd.DataFrame) -> pd.DataFrame:
    ny = (tr["signal_time"] + pd.Timedelta(minutes=15)).dt.tz_convert(NY)
    bucket = pd.cut(ny.dt.hour, [-1, 2, 7, 11, 16, 23],
                    labels=["00-03 亞洲", "03-08 倫敦", "08-12 紐約上午", "12-17 紐約下午", "17-24 亞洲開盤"])
    g = tr.groupby(bucket, observed=False)["R"]
    return pd.DataFrame({"n": g.size(), "share%": (100 * g.size() / len(tr)).round(0), "avgR": g.mean().round(3)})


# ----------------------------------------------------------------------------------------
# part 2: dollars
# ----------------------------------------------------------------------------------------

def mc_accounts(r2, rrun, mae, stop_usd, start, rule, price, oz_per_lot=100.0, n_trades=22,
                n_paths=20000, block=5, seed=1, min_lot=0.01, max_lots=0.3, leverage=100.0,
                spread_usd=0.0, stop_out=0.5, scale_out=False) -> dict:
    """Vectorised block-bootstrap of one year of trades for many accounts at once.

    r2, rrun, mae : per historical trade - R of a leg that takes profit at +2R, R of a
                    runner leg (trailing stop only) and the worst adverse excursion in R
    stop_usd      : stop distances ($ per oz) to sample from (today's volatility regime)
    rule          : {"kind": "fixed", "lots": 0.02} or {"kind": "risk", "risk": 1.0}
    P&L = R x stop x ounces. Margin stop-out: if the adverse excursion would push equity below
    stop_out x margin, the position is liquidated there."""
    rng = np.random.default_rng(seed)
    r2, rrun, mae, stop_usd = (np.asarray(x, float) for x in (r2, rrun, mae, stop_usd))
    n = len(r2)
    nb = int(np.ceil(n_trades / block))
    idx = (rng.integers(0, n, size=(n_paths, nb))[:, :, None] + np.arange(block)) % n
    idx = idx.reshape(n_paths, -1)[:, :n_trades]
    stops = stop_usd[rng.integers(0, len(stop_usd), size=(n_paths, n_trades))]
    oz_unit = oz_per_lot * min_lot
    margin_unit = price * oz_unit / leverage
    cap = int(round(max_lots / min_lot))
    eq = np.full(n_paths, float(start))
    peak = eq.copy()
    mdd = np.zeros(n_paths)
    outs = np.zeros(n_paths)
    taken = np.zeros(n_paths)
    lots_sum = np.zeros(n_paths)
    for t in range(n_trades):
        s = stops[:, t]
        loss_unit = (s + spread_usd) * oz_unit
        if rule["kind"] == "fixed":
            units = np.full(n_paths, int(round(rule["lots"] / min_lot)))
        else:  # same rule as goldsignal.plans.size_position
            budget = eq * rule["risk"] / 100.0
            units = np.floor(budget / loss_unit + 1e-9).astype(int)
            units = np.where((units == 0) & (loss_unit <= 2.0 * budget), 1, units)
        units = np.minimum(units, cap)
        units = np.minimum(units, np.floor(eq / margin_unit + 1e-9).astype(int))  # broker margin check
        units = np.maximum(units, 0)
        k = idx[:, t]
        n2 = np.where(scale_out & (units >= 3), units // 3, 0)
        r_units = n2 * r2[k] + (units - n2) * rrun[k]
        margin = units * margin_unit
        worst = mae[k] * s * units * oz_unit
        so = (units > 0) & (eq - worst < stop_out * margin)
        eq = np.where(so, stop_out * margin, eq + r_units * s * oz_unit)
        outs += so
        taken += units > 0
        lots_sum += units * min_lot
        peak = np.maximum(peak, eq)
        mdd = np.maximum(mdd, 1.0 - eq / peak)
    return {"med_ret": round(100 * (float(np.median(eq)) / start - 1), 1),
            "p5_ret": round(100 * (float(np.percentile(eq, 5)) / start - 1), 1),
            "p95_ret": round(100 * (float(np.percentile(eq, 95)) / start - 1), 1),
            "p_loss": round(100 * float(np.mean(eq < start)), 1),
            "med_dd": round(100 * float(np.median(mdd)), 1),
            "p_dd30": round(100 * float(np.mean(mdd >= 0.30)), 1),
            "p_dd50": round(100 * float(np.mean(mdd >= 0.50)), 1),
            "p_stopout": round(100 * float(np.mean(outs > 0)), 1),
            "avg_lots": round(float(np.sum(lots_sum) / max(np.sum(taken), 1)), 3)}


LABELS = {"0.01": "固定 0.01 手", "0.02": "固定 0.02 手", "0.03": "固定 0.03 手", "0.03so": "0.03 手（+2R 先平 1/3）",
          "0.05": "固定 0.05 手", "0.10": "固定 0.10 手", "0.20": "固定 0.20 手", "0.30": "固定 0.30 手",
          "risk2": "依風險 2%（建議）", "risk1": "依風險 1%（建議）"}
RULES = {"0.01": ({"kind": "fixed", "lots": 0.01}, False), "0.02": ({"kind": "fixed", "lots": 0.02}, False),
         "0.03": ({"kind": "fixed", "lots": 0.03}, False), "0.03so": ({"kind": "fixed", "lots": 0.03}, True),
         "0.05": ({"kind": "fixed", "lots": 0.05}, False), "0.10": ({"kind": "fixed", "lots": 0.10}, False),
         "0.20": ({"kind": "fixed", "lots": 0.20}, False), "0.30": ({"kind": "fixed", "lots": 0.30}, False),
         "risk2": ({"kind": "risk", "risk": 2.0}, False), "risk1": ({"kind": "risk", "risk": 1.0}, False)}


def money_table(df, f, clock, sym: str, start: str, label: str) -> dict:
    """Day-mode trades -> one-year Monte Carlo for $2,000 and $5,000 accounts."""
    inst = INSTRUMENTS[sym]
    day = inst.strategy_params(True)
    tr = run_variant(df, f, clock, day, Plan(eod="close_losers", eod_k=inst.eod_keep_r, legs=(2.0, 0.0, 0.0)), sym)
    tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp(start, tz="UTC"))].reset_index(drop=True)
    legs = np.array(tr["legR"].to_list())
    years = (tr["entry_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 365.25
    per_year = int(round(len(tr) / years))
    price = float(df["close"].iloc[-1])
    recent = tr[tr["entry_time"] >= tr["entry_time"].iloc[-1] - pd.Timedelta(days=365)]
    stops = recent["risk_pct"].to_numpy() * price           # today's volatility regime, today's price
    rec = "risk2" if sym == "XAUUSD" else "risk1"
    keys = [rec, "0.01", "0.02", "0.03", "0.03so", "0.05", "0.10", "0.20", "0.30"] if sym == "XAUUSD" else \
           [rec, "0.01", "0.02", "0.03", "0.05", "0.10"]
    rows = []
    for bal in (2000.0, 5000.0):
        for key in keys:
            rule, so = RULES[key]
            r = mc_accounts(legs[:, 0], legs[:, -1], tr["mae"].to_numpy(), stops, bal, rule, price,
                            inst.default_oz_per_lot, n_trades=per_year, spread_usd=price * inst.spread_pct, scale_out=so)
            rows.append({"balance": int(bal), "rule": key, "label": LABELS[key], **r})
    med = float(np.median(stops))
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "price": round(price, inst.decimals), "oz_per_lot": inst.default_oz_per_lot, "leverage": 100, "stop_out_level": 0.5,
        "trades_per_year": per_year,
        "stop_usd": {"median": round(med, 3), "p10": round(float(np.percentile(stops, 10)), 3), "p90": round(float(np.percentile(stops, 90)), 3)},
        "recommended": {"2000": rec, "5000": rec},
        "caption": (f"把 {label} 當日平倉模式的 {len(tr)} 筆{inst.name}實際交易結果，隨機重抽成 20,000 條「一年約 {per_year} 筆」的路徑；"
                    f"停損距離用最近一年的水準（現價約每盎司 ${med:.{2 if inst.decimals > 2 else 0}f}）。「一般」＝中位數，「運氣差」＝最差的 5%，"
                    "「跌三成／腰斬」＝一年內資金曾從高點回落 30%／50% 的機率。已計入 100 倍槓桿的保證金與 50% 強平線。"),
        "source": f"{label}, day-mode trades (goldsignal rules), block bootstrap of 5-trade blocks, 20,000 one-year paths; research/position_sizing.py",
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("kaggle", "dukascopy"), default="kaggle")
    ap.add_argument("--symbol", choices=list(INSTRUMENTS), default="XAUUSD")
    ap.add_argument("--money", action="store_true", help="also run part 2 (always on for --source dukascopy)")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df, start = load(args.source, args.symbol)
    f = S.compute_features(df)
    clock = Clock.build(df.index)
    periods = periods_for(args.source)
    rows, trades = [], {}
    for label, params, plan in variants(args.symbol):
        tr = run_variant(df, f, clock, params, plan, args.symbol)
        trades[label] = tr
        rows.append(summarize(tr, label, start, periods))
    res = pd.DataFrame(rows).set_index("plan")
    print(res.to_string())
    a = trades[variants(args.symbol)[0][0]]
    a = a[(a["reason"] != "end") & (a["entry_time"] >= pd.Timestamp(start, tz="UTC"))]
    print("\nplan A by signal time (New York):")
    print(hour_table(a).to_string())
    print(f"\nplan A: {int((a['R'] < -1.2).sum())} of {len(a)} trades lost more than 1.2R (gaps); worst {a['R'].min():.2f}R")
    print("plan A: R by nights held\n", a.groupby(np.minimum(a["nights"], 4))["R"].agg(["size", "mean", "sum"]).round(3).to_string())
    OUT.mkdir(exist_ok=True)
    tag = f"{args.symbol}_{args.source}"
    (OUT / f"{tag}_exit_study.json").write_text(json.dumps(res.reset_index().to_dict("records"), ensure_ascii=False, indent=1, default=float))
    if args.source == "dukascopy" or args.money:
        label = "2013/7–2026/10" if args.source == "dukascopy" else "2005–2025/3"
        mt = money_table(df, f, clock, args.symbol, start, label)
        print(pd.DataFrame(mt["rows"]).to_string(index=False))
        path = OUT / (f"{args.symbol}_money.json" if args.source == "dukascopy" else f"{tag}_money.json")
        path.write_text(json.dumps(mt, ensure_ascii=False, indent=1))
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
