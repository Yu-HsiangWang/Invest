"""Second-vendor check of the stop-order entry (research/entry_mechanics.py) on Dukascopy 15m
gold and silver 2013/7-2026/10, with the $2,000 money simulation (docs/RESEARCH.md section 12.2).

Variants (fixed before running):
  gold   : close-breakout baselines (normal / 高效積極 / 積極) and stop entries on the 18h and
           24h channels with the squeeze filter (max 1 / 2 same-direction positions), 18h
           without the squeeze filter;
  silver : close-breakout baselines (normal / 積極) and stop entries on the 24h channel
           without the squeeze filter (silver's normal rules), max 1 / 2, and 24h + squeeze.

Decision rule: a stop-entry variant replaces 積極模式 if it trades at least as often and is
more efficient (R per year / max drawdown) in every Dukascopy period AND overall, for the
instrument; it is offered as a new mode only if the money simulation also shows a lower
chance of a 30 % drawdown than 積極模式 at the same lots.

    res = run_all({"XAUUSD": df_gold, "XAGUSD": df_silver})      # in Pyodide, like efficient_dukascopy
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
import efficient_dukascopy as D  # noqa: E402
import entry_mechanics as M  # noqa: E402
import open_source_algos as A  # noqa: E402

VARIANTS = {
    "XAUUSD": {
        "normal": ("close", 24, True, 1), "efficient": ("close", 18, True, 2), "active": ("active", 12, False, 2),
        "stop18sq_2": ("stop", 18, True, 2), "stop18sq_1": ("stop", 18, True, 1), "stop24sq_2": ("stop", 24, True, 2),
        "stop24sq_1": ("stop", 24, True, 1), "stop18_2": ("stop", 18, False, 2), "stop12sq_2": ("stop", 12, True, 2),
    },
    "XAGUSD": {
        "normal": ("close", 24, False, 1), "active": ("active", 12, False, 2),
        "stop24_1": ("stop", 24, False, 1), "stop24_2": ("stop", 24, False, 2), "stop24sq_1": ("stop", 24, True, 1),
        "stop24sq_2": ("stop", 24, True, 2), "stop18_2": ("stop", 18, False, 2),
    },
}
COMBOS = {"normal": ("normal", "normal"), "efficient": ("efficient", "normal"), "active": ("active", "active"),
          "stop18sq_2+normal": ("stop18sq_2", "normal"), "stop24sq_2+normal": ("stop24sq_2", "normal"),
          "stop18sq_2+stop24_1": ("stop18sq_2", "stop24_1"), "stop24sq_2+stop24_1": ("stop24sq_2", "stop24_1"),
          "stop18sq_2+stop24_2": ("stop18sq_2", "stop24_2"), "stop18_2+stop24_2": ("stop18_2", "stop24_2")}
MONEY = [("gold_" + k, {"XAUUSD": k}, {"XAUUSD": 0.01}) for k in ("normal", "efficient", "active", "stop18sq_2", "stop24sq_2", "stop18_2")] + \
        [("both_" + c, {"XAUUSD": g, "XAGUSD": s}, {"XAUUSD": 0.01, "XAGUSD": 0.01}) for c, (g, s) in COMBOS.items()] + \
        [("silver_" + k, {"XAGUSD": k}, {"XAGUSD": 0.01}) for k in ("normal", "active", "stop24_1", "stop24_2")]


def run_symbol(df: pd.DataFrame, sym: str) -> dict[str, pd.DataFrame]:
    H = A.Harness(df, sym)
    out = {}
    for name, (kind, hours, sq, mo) in VARIANTS[sym].items():
        if kind == "close":
            sig = H.sys_sig(hours, sq)
            tr = H.run1(sig) if mo == 1 else H.run2(sig, None, mo)
        elif kind == "active":
            sig = H.sys_sig(hours, sq)
            tr = H._trim(E.run_variant(df, sig, sig, H.atr1, H.plan, H.sp, H.sl, H.clock, "any", mo, False))
        else:
            si, d, fl = M.stop_entry_signals(H, hours, sq, 0.0)
            tr = M.run_with_fills(H, si, d, fl, mo)
        tr = tr[tr["entry_time"] >= pd.Timestamp(D.START, tz="UTC")].copy()
        tr["sym"] = sym
        out[name] = tr
    return out


def run_all(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    trades = {}
    for sym, df in frames.items():
        D.PRICE[sym] = float(df["close"].iloc[-1])
        trades[sym] = run_symbol(df, sym)
    res = {"how": ("Dukascopy 1-minute BID candles aggregated to 15m, 2013/7-2026/10; close-breakout baselines and stop-order "
                   "entries at the channel edge (fill = max(open, level) + spread + slippage), exits / costs / day mode as "
                   "everywhere else. eff = R per year / max drawdown (R)."),
           "periods": [p[0] for p in D.PERIODS] + ["all"],
           "variants": {s: {k: list(v) for k, v in VARIANTS[s].items()} for s in VARIANTS},
           "instruments": {s: {k: D.by_period(t) for k, t in v.items()} for s, v in trades.items()}}
    if len(trades) == 2:
        res["both"] = {c: D.by_period(pd.concat([trades["XAUUSD"][g], trades["XAGUSD"][s]])) for c, (g, s) in COMBOS.items()}
        res["money"] = {lab: D.money([trades[s][v] for s, v in pick.items()], lots) for lab, pick, lots in MONEY}
        res["money_how"] = D.MONEY_HOW
    return res, trades


def main():
    frames = {("XAGUSD" if "XAG" in Path(p).name.upper() else "XAUUSD"): D.load_csv(p) for p in sys.argv[1:]}
    res, _ = run_all(frames)
    (ROOT / "results" / "stop_entry_dukascopy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
