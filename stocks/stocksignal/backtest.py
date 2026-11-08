"""Long-only daily simulator shared by the research scripts and the dashboard's back-test pages.

Conventions
-----------
* `target` is the desired exposure (0..1 of capital) decided at the CLOSE of day t.
* It is executed at the OPEN of day t+1 (default) or the close of t+1 - never at the close
  that produced the signal.
* Costs are a fraction of the traded notional, applied on buys and sells separately, plus
  slippage per side. Taiwan: commission 0.1425% (x discount) + 0.3% tax on sells (0.1% ETF).
* Cash earns `cash_rate` (annual). Default 0 so the comparisons are conservative for the
  timing strategies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import cagr, max_drawdown, sharpe


@dataclass(frozen=True)
class Costs:
    buy_fee: float = 0.0        # fraction of notional
    sell_fee: float = 0.0       # fraction of notional (includes transaction tax)
    slippage: float = 0.0005    # per side
    name: str = "generic"

    def buy(self) -> float:
        return self.buy_fee + self.slippage

    def sell(self) -> float:
        return self.sell_fee + self.slippage


# Taiwan: commission 0.1425% with a typical 60% online discount (=0.0855%), tax 0.3% stocks / 0.1% ETF.
TW_STOCK = Costs(buy_fee=0.001425 * 0.6, sell_fee=0.001425 * 0.6 + 0.003, slippage=0.0005, name="tw_stock")
TW_ETF = Costs(buy_fee=0.001425 * 0.6, sell_fee=0.001425 * 0.6 + 0.001, slippage=0.0005, name="tw_etf")
# US: zero commission broker; half-spread + slippage only.
US_STOCK = Costs(buy_fee=0.0, sell_fee=0.0, slippage=0.0005, name="us")
# US via Taiwanese sub-brokerage (複委託): ~0.15-0.5% each way with minimums. Use 0.25%.
US_SUBBROKER = Costs(buy_fee=0.0025, sell_fee=0.0025, slippage=0.0005, name="us_subbroker")


def simulate(df: pd.DataFrame, target: pd.Series, costs: Costs = US_STOCK, exec_at: str = "open",
             cash_rate: float = 0.0, start_capital: float = 1.0) -> pd.DataFrame:
    """Run the exposure schedule. Returns a frame with equity, exposure, cash, trades (notional traded).

    df needs columns open, close (adjusted consistently). `target` is aligned to df.index.
    """
    tgt = target.reindex(df.index).ffill().fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    equity = np.empty(n)
    expo = np.zeros(n)
    traded = np.zeros(n)
    shares = 0.0
    cash = float(start_capital)
    daily_cash = (1.0 + cash_rate) ** (1.0 / 252.0) - 1.0
    for t in range(n):
        if t > 0:
            cash *= 1.0 + daily_cash
            px = o[t] if exec_at == "open" else c[t]
            desired = tgt[t - 1]                      # decided at yesterday's close
            port_val = cash + shares * px
            want_val = desired * port_val
            cur_val = shares * px
            diff = want_val - cur_val
            if abs(diff) > 1e-9 * max(port_val, 1e-9):
                if diff > 0:
                    buy_val = min(diff, cash / (1.0 + costs.buy()))
                    shares += buy_val / px
                    cash -= buy_val * (1.0 + costs.buy())
                    traded[t] = buy_val
                else:
                    sell_val = min(-diff, shares * px)
                    shares -= sell_val / px
                    cash += sell_val * (1.0 - costs.sell())
                    traded[t] = sell_val
        equity[t] = cash + shares * c[t]
        expo[t] = (shares * c[t]) / equity[t] if equity[t] > 0 else 0.0
    out = pd.DataFrame({"equity": equity, "exposure": expo, "traded": traded, "close": c}, index=df.index)
    return out


def metrics(res: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    eq = res["equity"]
    r = eq.pct_change()
    out = {
        "cagr": cagr(eq),
        "vol": float(r.std() * np.sqrt(252)) if len(r.dropna()) > 1 else float("nan"),
        "sharpe": sharpe(r),
        "max_dd": max_drawdown(eq),
        "time_in_market": float((res["exposure"] > 0.01).mean()),
        "turnover_per_year": float(res["traded"].sum() / eq.mean() / max(len(eq) / 252.0, 1e-9)),
        "final_multiple": float(eq.iloc[-1] / eq.iloc[0]),
        "years": float(len(eq) / 252.0),
    }
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else float("nan")
    if bench is not None:
        b = bench.reindex(eq.index)
        out["bench_cagr"] = cagr(b)
        out["bench_max_dd"] = max_drawdown(b)
        out["bench_final_multiple"] = float(b.iloc[-1] / b.iloc[0])
        out["excess_cagr"] = out["cagr"] - out["bench_cagr"]
    return out


def trades_from_result(res: pd.DataFrame) -> pd.DataFrame:
    """Round trips: entry when exposure goes from ~0 to >0, exit when it returns to ~0."""
    ex = res["exposure"].to_numpy()
    eq = res["equity"].to_numpy()
    rows, in_pos, t0 = [], False, 0
    for t in range(len(ex)):
        if not in_pos and ex[t] > 0.01:
            in_pos, t0 = True, t
        elif in_pos and ex[t] <= 0.01:
            rows.append({"entry": res.index[t0], "exit": res.index[t], "days": t - t0, "ret": eq[t] / eq[t0 - 1 if t0 > 0 else t0] - 1.0})
            in_pos = False
    if in_pos:
        rows.append({"entry": res.index[t0], "exit": None, "days": len(ex) - t0, "ret": eq[-1] / eq[t0 - 1 if t0 > 0 else t0] - 1.0})
    return pd.DataFrame(rows)


def dca(df: pd.DataFrame, weights: pd.Series, costs: Costs = US_STOCK, monthly: float = 1.0,
        day_of_month: str = "first") -> pd.DataFrame:
    """Dollar-cost averaging with a zone-dependent multiplier.

    `weights` (aligned to df.index, decided at close) scales the monthly amount invested at the
    next trading day's open: 1 = normal, 2 = double, 0 = skip and keep the cash for later.
    Skipped cash accumulates and is deployed when the weight is > 1 (buy-the-dip wallet).
    Returns equity, invested (cumulative contributions), shares.
    """
    w = weights.reindex(df.index).ffill().fillna(1.0).to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    idx = df.index
    month = idx.to_period("M")
    first_of_month = np.r_[True, month[1:] != month[:-1]]
    shares, wallet, invested = 0.0, 0.0, 0.0
    eq = np.empty(len(df))
    inv = np.empty(len(df))
    for t in range(len(df)):
        if first_of_month[t] and t > 0:
            invested += monthly
            wallet += monthly
            mult = w[t - 1]
            spend = min(wallet, monthly * mult) if mult > 0 else 0.0
            if spend > 0:
                shares += spend / (1.0 + costs.buy()) / o[t]
                wallet -= spend
        eq[t] = shares * c[t] + wallet
        inv[t] = invested
    return pd.DataFrame({"equity": eq, "invested": inv, "close": c}, index=idx)


def conditional_forward(close: pd.Series, state: pd.Series, horizons=(63, 126, 252), min_n: int = 30) -> pd.DataFrame:
    """Forward returns by state: n, mean, median, hit-rate, and the same for 'all'.

    Observations overlap, so `n_eff` approximates the number of independent observations
    (n / horizon). Research use only.
    """
    rows = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        d = pd.DataFrame({"fwd": fwd, "state": state}).dropna()
        for st, g in list(d.groupby("state")) + [("all", d)]:
            if len(g) < min_n:
                continue
            rows.append({"horizon": h, "state": st, "n": int(len(g)), "n_eff": round(len(g) / h, 1),
                         "mean": float(g["fwd"].mean()), "median": float(g["fwd"].median()),
                         "hit": float((g["fwd"] > 0).mean()), "p10": float(g["fwd"].quantile(0.1)),
                         "p90": float(g["fwd"].quantile(0.9))})
    return pd.DataFrame(rows)


def block_bootstrap_diff(a: np.ndarray, b: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0) -> float:
    """Two-sided bootstrap p-value for mean(a) - mean(b) with overlapping/serially dependent data.

    Resamples blocks of consecutive observations from each array (moving-block bootstrap).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < block or len(b) < block:
        return float("nan")
    obs = a.mean() - b.mean()

    def _resample(x):
        nb = int(np.ceil(len(x) / block))
        starts = rng.integers(0, len(x) - block + 1, size=nb)
        return np.concatenate([x[s:s + block] for s in starts])[:len(x)]

    diffs = np.empty(n_boot)
    ac, bc = a - a.mean(), b - b.mean()
    for i in range(n_boot):
        diffs[i] = _resample(ac).mean() - _resample(bc).mean()
    return float((np.abs(diffs) >= abs(obs)).mean())
