"""Re-create results/<SYM>_active_summary.json - the 積極模式 back-test shown in the dashboard.

Gold: the exact live rules (goldsignal.active) on the 2004-2025 15-minute history (same file
as scripts/make_backtest_summary.py). Gold and silver on Dukascopy data were run in the browser
(Pyodide): the hourly version of the rules 2013-2026 (research/multi_market.py), the exact 15m
rules on the newest data 2025/4-2026/10, and a one-year money simulation for a $2,000 account;
those numbers are stored in results/active_dukascopy.json and this script merges everything.

    python scripts/make_active_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.active import ACTIVE_BREAKOUT_HOURS, MAX_OPEN, active_params, run_multi  # noqa: E402
from goldsignal.backtest import metrics  # noqa: E402
from goldsignal.bars import load_kaggle_15m  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402

PERIODS = [("2005–2016", "用來設計策略", "2005-01-01", "2016-12-31"), ("2017–2020", "驗證", "2017-01-01", "2020-12-31"),
           ("2021–2025/3", "測試", "2021-01-01", "2025-03-21")]


def gold_kaggle() -> dict:
    pq = ROOT / "data" / "cache_xau15.parquet"
    df = pd.read_parquet(pq).loc[:"2025-03-21"] if pq.exists() else load_kaggle_15m(ROOT / "data" / "raw" / "XAU_15m_data.csv").loc[:"2025-03-21"]
    inst = INSTRUMENTS["XAUUSD"]
    px = df["open"].to_numpy(float)
    out = {}
    for mode, p in (("active", active_params(inst.strategy_params(True, squeeze=False))), ("normal", inst.strategy_params(True))):
        f = S.compute_features(df, params=p)
        if mode == "active":
            tr = run_multi(df, f, p, px * inst.spread_pct, px * inst.slip_pct, swap=True, max_open=MAX_OPEN)
        else:
            tr, _, _ = S.run(df, f, p, px * inst.spread_pct, px * inst.slip_pct, swap=True)
        tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC"))].sort_values("exit_time")
        rows = []
        for label, sub, a, b in PERIODS:
            m = (tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
            yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
            rows.append({"label": label, "sub": sub, **metrics(tr[m], yrs)})
        span = (pd.Timestamp("2025-03-21") - pd.Timestamp("2005-01-01")).days / 365.25
        out[mode] = {"periods": rows, "all": metrics(tr, span)}
    return out


def main():
    duka = json.loads((ROOT / "results" / "active_dukascopy.json").read_text(encoding="utf-8"))
    g = gold_kaggle()
    for sym in ("XAUUSD", "XAGUSD"):
        d = duka["instruments"][sym]
        res = {
            "mode": "active",
            "rules": {"breakout_hours": ACTIVE_BREAKOUT_HOURS, "max_open": MAX_OPEN, "squeeze": False},
            "caption": (f"積極模式：{ACTIVE_BREAKOUT_HOURS} 小時突破、不用布林收窄濾網、同一商品最多同時 {MAX_OPEN} 張；"
                        "其餘規則（趨勢、停損、移動停損、16:45 檢查）與一般設定相同。已扣點差、滑價與隔夜費。"),
            "dukascopy": d,
            "both": duka["both"],
            "source": duka["how"],
        }
        rec = duka.get("recent_15m")
        if rec:
            res["recent_15m"] = {"window_label": rec["window_label"], "how": rec["how"], "normal": rec[sym]["normal"],
                                 "active": rec[sym]["active"], "both": rec["both"]}
        if duka.get("money"):
            res["money"] = duka["money"]
        if sym == "XAUUSD":
            res["kaggle"] = g
        (ROOT / "results" / f"{sym}_active_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                                                    encoding="utf-8")
    print(json.dumps({k: v["all"] for k, v in g.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
