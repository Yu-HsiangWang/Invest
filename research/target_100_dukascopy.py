"""Second-vendor check of the two 積極模式 changes that came closest to +100 % a year on the broker
data (docs/RESEARCH.md section 13.5), on Dukascopy 15m gold 2013/7-2026/10 (run in the browser
like research/stop_entry_dukascopy.py, or locally on a CSV of 15m candles).

Candidates, against 積極模式 as it is (stop orders on the 18-hour channel, squeeze filter, at most
2 same-direction positions):
  max3 / max4  up to 3 / 4 same-direction positions (pyramiding)
  nosq         without the squeeze filter (more trades)
On 2005-2025/3 they needed 3.25 % / 2.5 % / 4.75 % risk per position for a +100 % median year
(5 % as it is), but pyramiding was worse than 2 positions in 2021-2025/3. They were found after
the fact, so they must hold on other data before they go into the dashboard.

Decision rule, fixed before running: a candidate replaces 積極模式's rules only if, on Dukascopy,
  (a) its efficiency (R per year / max drawdown in R) beats the current rules overall AND in at
      least 3 of the 4 periods (including 2025/4-2026/10 or 2021-2025/3), and
  (b) at the risk per position where its month-block bootstrap median year is +100 %, its chance
      of a 50 % fall within the year is lower than the current rules' at their own +100 % risk.

    res = run_all(df_gold)                                     # in Pyodide
    python research/target_100_dukascopy.py XAUUSD_15m.csv    # locally
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

import efficient_dukascopy as D  # noqa: E402
import entry_mechanics as M  # noqa: E402
import open_source_algos as A  # noqa: E402
import target_100 as T  # noqa: E402

CANDIDATES = {"current": (True, 2), "max3": (True, 3), "max4": (True, 4), "nosq": (False, 2)}   # (squeeze filter, max open)
F_GRID = np.round(np.arange(0.01, 0.0801, 0.0025), 4)
MONTHS = (D.START[:7], "2026-09")


def run_all(df: pd.DataFrame) -> dict:
    D.PRICE["XAUUSD"] = float(df["close"].iloc[-1])
    H = A.Harness(df, "XAUUSD")
    sigs = {sq: M.stop_entry_signals(H, 18, sq, 0.0) for sq in (True, False)}
    res = {"how": __doc__.split("\n\n")[3].strip(), "periods": [p[0] for p in D.PERIODS] + ["all"], "variants": {}}
    for name, (sq, mo) in CANDIDATES.items():
        si, d, fl = sigs[sq]
        tr = M.run_with_fills(H, si, d, fl, mo)
        tr = tr[(tr["entry_time"] >= pd.Timestamp(D.START, tz="UTC")) & (tr["reason"] != "end")].sort_values("entry_time").reset_index(drop=True)
        v = {"by_period": D.by_period(tr), "n": int(len(tr))}
        need = None
        for f in F_GRID:                       # the smallest risk whose month-block median year is >= +100 %
            if T.bootstrap_months(tr, float(f), paths=4000, start=MONTHS[0], end=MONTHS[1])["med"] >= 1.0:
                need = float(f)
                break
        v["f_for_median_100"] = need
        if need:
            v["at_need"] = T.bootstrap_months(tr, need, start=MONTHS[0], end=MONTHS[1])
        v["at_fixed"] = {str(f): T.bootstrap_months(tr, f, start=MONTHS[0], end=MONTHS[1]) for f in (0.025, 0.0325, 0.05)}
        res["variants"][name] = v
    base = res["variants"]["current"]
    for name, v in res["variants"].items():
        if name == "current":
            continue
        eff = [(v["by_period"][p].get("eff") or 0.0, base["by_period"][p].get("eff") or 0.0) for p in res["periods"]]
        wins = sum(a > b for a, b in eff[:-1])
        a_ok = eff[-1][0] > eff[-1][1] and wins >= 3
        b_ok = bool(v.get("at_need") and base.get("at_need") and v["at_need"]["p_dd50"] < base["at_need"]["p_dd50"])
        v["decision"] = {"efficiency_overall_and_3_of_4": bool(a_ok), "periods_won": int(wins), "lower_ruin_at_100%": b_ok,
                         "adopt": bool(a_ok and b_ok)}
    return res


def main():
    res = run_all(D.load_csv(sys.argv[1]))
    (ROOT / "results" / "target_100_dukascopy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["variants"].items():
        print(k, v["f_for_median_100"], v.get("at_need"), v.get("decision"))


if __name__ == "__main__":
    main()
