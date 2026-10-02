"""Day mode (16:45 New York check), scale-out legs and position sizing."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goldsignal import strategy as S  # noqa: E402
from goldsignal.backtest import ExitRules, simulate  # noqa: E402
from goldsignal.plans import Clock, Plan, eod_decision, leg_targets, lots_table, simulate_plan, size_position  # noqa: E402
from tests.test_core import synthetic_m15  # noqa: E402


def _bars(start_ny: str, closes: list[float], spread: float = 0.0) -> pd.DataFrame:
    """15m bars starting at a New York wall-clock time; tiny ranges around the closes."""
    idx = pd.date_range(pd.Timestamp(start_ny, tz="America/New_York"), periods=len(closes), freq="15min").tz_convert("UTC")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.05, "low": np.minimum(o, c) - 0.05, "close": c, "volume": 1.0}, index=idx)


def test_plan_hold_matches_fast_simulator():
    df = synthetic_m15(6000)
    rng = np.random.default_rng(3)
    sig = np.zeros(len(df), dtype=np.int64)
    sig[rng.choice(np.arange(300, len(df) - 50), 40, replace=False)] = rng.choice([-1, 1], 40)
    atr = np.full(len(df), 1.5)
    c = df["close"].to_numpy()
    sp = np.full(len(df), 0.3)
    sl = np.full(len(df), 0.05)
    fast = simulate(df, sig, c - 2 * atr, c + 2 * atr, ExitRules(tp_r=0, trail_mult=5.0, cooldown=4), sp, sl, atr=atr)
    slow = simulate_plan(df, sig, atr, Plan(swap=False), sp, sl)
    assert len(fast) == len(slow) > 5
    assert np.allclose(fast["R"].to_numpy(), slow["R"].to_numpy())
    assert np.allclose(fast["stop_last"].to_numpy(), slow["stop_last"].to_numpy())


def test_eod_closes_losers_and_keeps_winners():
    # long entered 10:00 NY; at 16:45 it is flat (< +0.25R) -> closed by the 16:45 check
    n = 40
    flat = _bars("2026-09-15 10:00", [100.0] * n)
    sig = np.zeros(n, dtype=np.int64)
    sig[0] = 1
    atr = np.full(n, 1.0)  # stop 2 below, 1R = 2.0
    tr = simulate_plan(flat, sig, atr, Plan(eod="close_losers", eod_k=0.25, swap=False), np.zeros(n), np.zeros(n))
    assert tr.iloc[0]["reason"] == "eod"
    close_ny = (tr.iloc[0]["exit_time"] + pd.Timedelta(minutes=15)).tz_convert("America/New_York")
    assert (close_ny.hour, close_ny.minute) == (16, 45)
    # same trade but +1R by 16:45 -> kept overnight
    up = _bars("2026-09-15 10:00", list(np.linspace(100, 102, n)))
    tr2 = simulate_plan(up, sig, atr, Plan(eod="close_losers", eod_k=0.25, swap=False), np.zeros(n), np.zeros(n))
    assert tr2.iloc[0]["reason"] != "eod"


def test_eod_mask_never_fires_on_the_running_day():
    m15 = _bars("2026-09-15 09:00", [100.0] * 20)  # today until 13:45 NY, data stops there
    mask = Clock.build(m15.index).eod_mask(16.75)
    assert not mask.any()


def test_day_cutoff_blocks_afternoon_signals():
    df = synthetic_m15(9000)
    f = S.compute_features(df)
    base = S.rule_table(df, f, S.StrategyParams())
    day = S.rule_table(df, f, replace(S.StrategyParams(), day_cutoff_ny=12.0, eod_keep_r=0.25))
    pm = (f["ny_close_h"] >= 12.0) & (f["ny_close_h"] < 17.0)
    assert not (day["long_signal"] | day["short_signal"])[pm].any()
    assert (base["timing_ok"][~pm] == day["timing_ok"][~pm]).all()


def test_position_sizing_numbers():
    g = size_position(2000, 2.0, 30.0, 4180.0, 100.0, spread=0.5)        # gold, $2,000, 2 % risk
    assert g["lots"] == 0.01 and g["risk_usd"] == 30.5 and g["margin"] == 41.8
    assert size_position(5000, 2.0, 30.0, 4180.0, 100.0, spread=0.5)["lots"] == 0.03
    assert size_position(5000, 2.0, 30.0, 4180.0, 100.0, max_lots=0.02)["lots"] == 0.02   # user cap
    s = size_position(2000, 1.0, 0.74, 61.13, 5000.0, spread=0.037)      # silver: one min lot ~1.9 %
    assert s["lots"] == 0.01 and s["over_budget"]
    assert size_position(2000, 0.5, 0.9, 61.13, 5000.0)["skip"]          # 2.25 % > 2 x 0.5 %
    assert size_position(100, 50.0, 1.0, 4180.0, 100.0)["lots"] == 0.02  # margin allows only 2 x $41.8
    assert leg_targets(1) == [0.0] and leg_targets(2) == [0.0, 0.0] and leg_targets(3) == [2.0, 0.0, 0.0]
    rows = {r["lots"]: r for r in lots_table(2000, 30.0, 4180.0, 100.0)}
    assert rows[0.3]["margin"] == 1254.0 and abs(rows[0.3]["liq_move"] - 45.77) < 0.01
    e = eod_decision(1, 100.0, 2.0, 100.3, 0.25)
    assert not e["keep"] and abs(e["target_price"] - 100.5) < 1e-9
