"""Re-create results/XAUUSD_backtest_summary.json (shown in the dashboard) from the
2004-2025 15-minute history.

The history file is the public 'XAU_15m_data.csv' (Kaggle dataset by
novandraanugrah, also mirrored in github.com/BaseMax/XAUUSD-LSTM). Put it at
data/raw/XAU_15m_data.csv, then run:

    python scripts/make_backtest_summary.py

Three variants are reported, with Mitrade's overnight financing charged:
  day          = the app default: 當日平倉模式 (no entries 12:00-17:00 NY, 16:45 NY check)
                 + 布林收窄濾網 (only breakouts out of a 15m Bollinger squeeze, docs/RESEARCH.md section 10)
  day_nofilter = the same without the 布林收窄濾網 (the rules before the chart-tool back-test)
  hold         = the original trade management (positions may run for days), same entries
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import by_year, metrics  # noqa: E402
from goldsignal.bars import load_kaggle_15m  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402

INST = INSTRUMENTS["XAUUSD"]
SPREAD_PCT, SLIP_PCT = INST.spread_pct, INST.slip_pct
PERIODS = [
    ("開發期 2005–2016", "2005-01-01", "2016-12-31"),
    ("驗證期 2017–2020", "2017-01-01", "2020-12-31"),
    ("測試期 2021–2025/3", "2021-01-01", "2025-03-21"),
]


def summarize(tr: pd.DataFrame, label: str) -> dict:
    out = []
    for name, a, b in PERIODS:
        m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
        yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        out.append({"label": name, **metrics(tr[m], yrs)})
    span = (tr["entry_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 365.25
    return {"label": label, "all": metrics(tr, span), "periods": out,
            "overnight_pct": round(100 * float((tr["nights"] > 0).mean()), 1),
            "hold_hours_median": float(tr["bars"].median() / 4)}


def main():
    src = ROOT / "data" / "raw" / "XAU_15m_data.csv"
    df = load_kaggle_15m(src).loc[:"2025-03-21"]
    px = df["open"].to_numpy(float)
    f = S.compute_features(df)
    res = {}
    trades = {}
    for mode, day, sq, label in (("day", True, None, "當日平倉模式＋布林收窄濾網"), ("day_nofilter", True, False, "當日平倉模式（不加濾網）"),
                                 ("hold", False, None, "原策略（可留倉過夜）")):
        p = INST.strategy_params(day, squeeze=sq)
        tr, _, _ = S.run(df, f, p, spread=px * SPREAD_PCT, slip=px * SLIP_PCT, swap=True)
        tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC"))]
        trades[mode] = tr
        res[mode] = summarize(tr, label)
    tr = trades["day"]
    allm = res["day"]["all"]
    months = (tr["entry_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 30.4
    eq = tr[["exit_time", "R"]].copy()
    eq["cum"] = eq["R"].cumsum()
    step = max(1, len(eq) // 300)
    out = {
        "caption": f"當日平倉模式＋布林收窄濾網 2005–2025/3 共 {allm['n']} 筆（約每月 {allm['n'] / months:.1f} 筆），累積 {allm['totR']:+.0f}R；"
                   f"已扣點差 0.012%、每次成交滑價 0.005% 與 Mitrade 隔夜費。",
        "params": INST.strategy_params(True).to_dict(),
        "costs": {"spread_pct": SPREAD_PCT, "slip_pct_per_fill": SLIP_PCT, "swap_long_pct_night": 0.0168, "swap_short_pct_night": 0.014},
        "mode": "day",
        "all": allm,
        "periods": res["day"]["periods"],
        "modes": res,
        "equity": [[ts.isoformat(), round(float(v), 2)] for ts, v in zip(eq["exit_time"].iloc[::step], eq["cum"].iloc[::step])],
        "by_year": by_year(tr).reset_index().rename(columns={"entry_time": "year"}).to_dict("records"),
        "long": metrics(tr[tr["dir"] > 0]),
        "short": metrics(tr[tr["dir"] < 0]),
    }
    path = ROOT / "results" / "XAUUSD_backtest_summary.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: out[k] for k in ("caption", "all", "periods")}, ensure_ascii=False, indent=1))
    print("day_nofilter:", json.dumps(res["day_nofilter"]["all"], ensure_ascii=False))
    print("hold:", json.dumps(res["hold"]["all"], ensure_ascii=False))


if __name__ == "__main__":
    main()
