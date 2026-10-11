"""Re-create results/modes_summary.json - the comparison of the three trading modes shown in the
dashboard (一般設定 / 高效積極模式 / 積極模式, goldsignal/active.py).

* gold on the 2004-2025 15-minute history (the same file as scripts/make_backtest_summary.py),
  traded by the exact live code (goldsignal.active.mode_rules / run_mode);
* gold and silver on Dukascopy 15m data 2013/7-2026/10 and the $2,000 money simulation, from
  results/efficient_dukascopy.json (research/efficient_dukascopy.py, run in the browser because
  Dukascopy is not reachable from every network).

    python scripts/make_mode_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.active import EFFICIENT, MAX_OPEN, ACTIVE_BREAKOUT_HOURS, MODE_LABEL, MODES, mode_rules, run_mode  # noqa: E402
from goldsignal.backtest import metrics  # noqa: E402
from goldsignal.bars import load_kaggle_15m  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402

PERIODS = [("2005–2016", "用來設計策略", "2005-01-01", "2016-12-31"), ("2017–2020", "驗證", "2017-01-01", "2020-12-31"),
           ("2021–2025/3", "測試", "2021-01-01", "2025-03-21")]

# which research variant (research/efficient_dukascopy.py) each mode is, per instrument / for both
DUKA = {"XAUUSD": {"normal": "normal", "efficient": "18sq_max2", "active": "active"},
        "XAGUSD": {"normal": "normal", "efficient": "normal", "active": "active"},
        "both": {"normal": "normal", "efficient": "18sq_max2+normal", "active": "active"}}
MONEY = {"gold": {"normal": "gold_normal", "efficient": "gold_18sq_max2", "active": "gold_active"},
         "both": {"normal": "both_normal", "efficient": "both_18sq_max2+normal", "active": "both_active"},
         "silver": {"normal": "silver_normal", "efficient": "silver_normal", "active": "silver_active"},
         "gold_x2": {"normal": "gold_normal_x2", "efficient": "gold_18sq_max2_x2"}}

RULES = {
    "normal": "一次一張；24 小時突破；黃金要布林收窄濾網，白銀不用。",
    "efficient": (f"黃金：{EFFICIENT['XAUUSD']['breakout_hours']} 小時突破＋布林收窄濾網，持倉中出現同方向的新訊號可以再開一張"
                  f"（最多同時 {EFFICIENT['XAUUSD']['max_open']} 張）；白銀：照一般設定。"),
    "active": f"{ACTIVE_BREAKOUT_HOURS} 小時突破、不用收窄濾網、出場不用冷卻，每個新訊號都可以再開一張（同一商品最多同時 {MAX_OPEN} 張）。",
}


def gold_kaggle() -> dict:
    pq = ROOT / "data" / "cache_xau15.parquet"
    df = pd.read_parquet(pq).loc[:"2025-03-21"] if pq.exists() else load_kaggle_15m(ROOT / "data" / "raw" / "XAU_15m_data.csv").loc[:"2025-03-21"]
    inst = INSTRUMENTS["XAUUSD"]
    px = df["open"].to_numpy(float)
    out = {}
    for mode in MODES:
        r = mode_rules(inst, mode, True, inst.squeeze_default)
        f = S.compute_features(df, params=r.params)
        tr = run_mode(df, f, r, px * inst.spread_pct, px * inst.slip_pct, swap=True)
        tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC"))].sort_values("exit_time")
        rows = []
        for label, sub, a, b in PERIODS:
            m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
            yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
            rows.append({"label": label, "sub": sub, **metrics(tr[m], yrs)})
        span = (pd.Timestamp("2025-03-21") - pd.Timestamp("2005-01-01")).days / 365.25
        allm = metrics(tr, span)
        allm["R_yr"] = round(allm["totR"] / span, 1)
        allm["eff"] = round(allm["R_yr"] / allm["maxDD_R"], 2) if allm.get("maxDD_R") else None
        out[mode] = {"periods": rows, "all": allm}
        print(f"  {mode}: {allm['n']} trades, {allm['per_week']}/wk, {allm['avgR']:+.3f}R, DD {allm['maxDD_R']}R, eff {allm['eff']}")
    return out


def main():
    duka = json.loads((ROOT / "results" / "efficient_dukascopy.json").read_text(encoding="utf-8"))
    res = {
        "labels": MODE_LABEL,
        "rules": RULES,
        "kaggle_gold": gold_kaggle(),
        "dukascopy": {k: {m: (duka["both"] if k == "both" else duka["instruments"][k])[v] for m, v in sel.items()}
                      for k, sel in DUKA.items()},
        "periods": duka["periods"],
        "money": {k: {m: duka["money"][v] for m, v in sel.items()} for k, sel in MONEY.items()},
        "balance": 2000,
        "prices": duka.get("prices"),
        "how": {"kaggle": "黃金券商 15 分 K 2005–2025/3，儀表板實際使用的程式（當日平倉模式）。",
                "dukascopy": duka["how"], "money": duka.get("money_how")},
        "caption": "三種模式的停損、移動停損、16:45 檢查都一樣，只差在什麼時候開新單。已扣點差、滑價與隔夜費。",
    }
    (ROOT / "results" / "modes_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
