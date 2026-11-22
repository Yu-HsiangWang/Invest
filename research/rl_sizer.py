"""Reinforcement learning B: learn which stop-order trades to take and how big (docs/RESEARCH.md
section 13).

Candidate trades = every stop-order fill of the 積極模式 entry on the 18-hour channel, WITHOUT the
squeeze filter (so the agent can take more trades than the mode if it finds them worth it). Each
candidate is simulated alone with the live exits (stop 2 ATR, trail 5 ATR, day mode, costs).

At each candidate the agent sees the 33 market features of research/rl_agent.py at the signal bar,
whether the squeeze filter would pass, the direction, the stop distance, how many positions are
open, the account's drawdown and the last three results (closed trades only: a trade still open
at the signal bar is not in them), and picks a size: skip, 0.5x, 1x or 2x
of the base risk F0 = 5 % of the account (about the risk at which 積極模式's median year is +100 %
over 2005-2025).
At most two positions, same direction (as the mode). Reward = log growth of the account,
log(1 + F0 x size x R): maximising it is maximising the compounded return.

PPO on 2005-2016 candidates (episodes = 150 consecutive candidates from a random start), the
checkpoint chosen on 2017-2020, scored once on 2021-2025/3; compared with the fixed rules on the
same candidates: take everything at 1x / 2x, and the mode's own rule (squeeze passes) at 1x, 1.5x
and 2x - the larger sizes matter, because a policy that simply bets more than 1x would beat the 1x
rules without choosing anything well.

Variant B2 changes the goal from "grow as fast as possible" to "grow, but pay for every new low":
reward = log growth - LAM x each increase of the (log) drawdown, with sizes up to 3x (15 %, about
the Kelly fraction) so that betting the maximum is no longer automatically best; the checkpoint is
chosen on the same objective in 2017-2020, and the fixed rules are scored on it too.

Needs: pip install stable-baselines3 gymnasium (brings PyTorch).

    python research/rl_sizer.py --variant B1 --steps 600000 --seeds 0 1 2
    python research/rl_sizer.py --variant B2 --lam 1 --sizes 0 0.5 1 1.5 2 3 --steps 600000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
import rl_agent as RA  # noqa: E402
from goldsignal import strategy as S  # noqa: E402
from goldsignal.active import mode_rules, run_multi, stop_entry_candidates  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock  # noqa: E402

OUT = ROOT / "results"
MODELS = ROOT / "data" / "cache" / "rl_models"
F0 = 0.05
SIZES = np.array([0.0, 0.5, 1.0, 2.0])
EP_LEN = 150
LAM = 0.0            # drawdown penalty (variant B2): each new step down from the peak (log) costs LAM x itself


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Every stop-order fill on the 18h channel without the squeeze filter, each traded alone."""
    inst = INSTRUMENTS["XAUUSD"]
    r = mode_rules(inst, "active", True, True)
    p_nosq = replace(r.params, squeeze_bars=0)
    f = S.compute_features(df, params=p_nosq)
    t = S.rule_table(df, f, p_nosq)
    sig, fills = stop_entry_candidates(df, f, t)
    sq_ok = S.rule_table(df, f, r.params)["squeeze_ok"].to_numpy(bool)     # the mode's squeeze test at the signal bar
    d2 = df.copy()
    o = d2["open"].to_numpy(float).copy()
    m = ~np.isnan(fills)
    o[m] = fills[m]
    d2["open"] = o
    px = df["open"].to_numpy(float)
    tr = run_multi(d2, f, p_nosq, px * inst.spread_pct, px * inst.slip_pct, swap=True, max_open=0,
                   clock=Clock.build(df.index), rules=t, cooldown_when_flat=False, same_dir=False, sig=sig)
    tr = tr[(tr["reason"] != "end") & (tr["entry_time"] >= pd.Timestamp("2005-01-01", tz="UTC"))].copy()
    tr["squeeze_ok"] = sq_ok[tr["sig_i"].to_numpy()]
    tr["risk_pct"] = tr["risk"] / tr["entry"]
    return tr.sort_values("sig_i").reset_index(drop=True)


def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class SizerEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, C: pd.DataFrame, X: np.ndarray, rows: np.ndarray, seed: int = 0, sequential: bool = False):
            super().__init__()
            self.C, self.Xc = C, X
            self.rows = rows                      # candidate row numbers of this split, in time order
            self.R = C["R"].to_numpy(float)
            self.si = C["sig_i"].to_numpy(int)
            self.xi = C["exit_i"].to_numpy(int)
            self.dir = C["dir"].to_numpy(int)
            self.sq = C["squeeze_ok"].to_numpy(float)
            self.rp = C["risk_pct"].to_numpy(float)
            self.k = X.shape[1] + 9
            self.observation_space = spaces.Box(-10, 10, shape=(self.k,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(SIZES))
            self.rng = np.random.default_rng(seed)
            self.sequential = sequential

        def _realize(self, now_i: int):
            """Book the taken trades that have exited by bar now_i (only those are known then)."""
            keep = []
            for x, R, s in sorted(self.pending):
                if x <= now_i:
                    self.eq *= max(1 + F0 * s * R, 1e-6)
                    self.peak = max(self.peak, self.eq)
                    self.last = [float(R)] + self.last[:2]
                else:
                    keep.append((x, R, s))
            self.pending = keep

        def _obs(self):
            j = self.rows[self.pos]
            self._realize(self.si[j])
            live = [(x, d) for x, d in self.open if x >= self.si[j] + 1]
            n_open = len(live)
            opp = float(any(d != self.dir[j] for _, d in live))
            dd = np.log(self.eq / self.peak)          # closed trades only
            last = (self.last + [0.0, 0.0, 0.0])[:3]  # the last three CLOSED results
            extra = np.array([self.dir[j], self.sq[j] - 0.5, np.clip(self.rp[j] * 100, 0, 5) - 1, n_open / 2, opp,
                              np.clip(dd, -3, 0), *np.clip(np.array(last) / 3, -1, 3)], dtype=np.float32)
            return np.concatenate([self.Xc[self.si[j]], extra])

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if self.sequential:
                self.pos, self.stop_at = 0, len(self.rows)
            else:
                self.pos = int(self.rng.integers(0, max(1, len(self.rows) - EP_LEN)))
                self.stop_at = min(len(self.rows), self.pos + EP_LEN)
            self.eq, self.peak, self.open, self.last, self.pending = 1.0, 1.0, [], [], []
            self.v_eq = self.v_peak = 0.0             # log equity in entry order, for the drawdown penalty
            self.taken = []
            return self._obs(), {}

        def step(self, action):
            j = self.rows[self.pos]
            live = [(x, d) for x, d in self.open if x >= self.si[j] + 1]
            size = SIZES[int(action)]
            if len(live) >= 2 or any(d != self.dir[j] for _, d in live):
                size = 0.0                          # the mode's limits: 2 positions, one direction
            r = 0.0
            if size > 0:
                g = max(1 + F0 * size * self.R[j], 1e-6)
                r = float(np.log(g))                # reward now; the account sees it only when the trade closes
                if LAM:
                    dd_old = self.v_peak - self.v_eq
                    self.v_eq += r
                    self.v_peak = max(self.v_peak, self.v_eq)
                    r -= LAM * max(0.0, (self.v_peak - self.v_eq) - dd_old)
                self.pending.append((int(self.xi[j]), float(self.R[j]), float(size)))
                self.taken.append((j, size))
            self.open = live + ([(self.xi[j], self.dir[j])] if size > 0 else [])
            self.pos += 1
            done = self.pos >= self.stop_at
            obs = self._obs() if not done else np.zeros(self.k, dtype=np.float32)
            return obs, r * 100, done, False, {}

    return SizerEnv


def split_rows(C: pd.DataFrame, split: str) -> np.ndarray:
    a, b = RA.SPLITS[split]
    m = (C["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (C["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))
    return np.flatnonzero(m.to_numpy())


def score(C: pd.DataFrame, taken: list[tuple[int, float]], split: str) -> dict:
    """Compound the chosen trades (sequentially, F0 x size per trade) over the split."""
    a, b = RA.SPLITS[split]
    years = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    if not taken:
        return {"trades": 0}
    j = np.array([x for x, _ in taken]); s = np.array([y for _, y in taken])
    R = C["R"].to_numpy(float)[j]
    g = np.maximum(1 + F0 * s * R, 1e-9)
    eq = np.cumprod(g)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    ldd = np.log(peak / eq)
    objective = (np.log(g).sum() - LAM * np.maximum(np.diff(np.r_[0.0, ldd]), 0).sum()) / years
    yrs = C["entry_time"].iloc[j].dt.year.to_numpy()
    by_year = pd.Series(np.log(g)).groupby(yrs).sum().apply(np.expm1)
    return {"trades": int(len(j)), "per_week": round(len(j) / years / 52.18, 2), "avgR": round(float(R.mean()), 3),
            "avg_size": round(float(s.mean()), 2), "cagr": round(float(eq[-1] ** (1 / years) - 1), 3),
            "max_dd": round(float(np.max(1 - eq / peak)), 3), "median_year": round(float(by_year.median()), 3),
            "worst_year": round(float(by_year.min()), 3), "objective": round(float(objective), 3),
            "years": {int(k): round(float(v), 3) for k, v in by_year.items()}}


def rule_policy(C: pd.DataFrame, rows: np.ndarray, rule) -> list[tuple[int, float]]:
    """A fixed rule under the same position limits (2 positions, same direction)."""
    si, xi, d = C["sig_i"].to_numpy(int), C["exit_i"].to_numpy(int), C["dir"].to_numpy(int)
    open_, taken = [], []
    for j in rows:
        open_ = [(x, dd) for x, dd in open_ if x >= si[j] + 1]
        s = rule(C.iloc[j])
        if s <= 0 or len(open_) >= 2 or any(dd != d[j] for _, dd in open_):
            continue
        open_.append((xi[j], d[j])); taken.append((int(j), float(s)))
    return taken


def run_policy(model, Env, C, X, rows) -> list[tuple[int, float]]:
    env = Env(C, X, rows, sequential=True)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(a)
    return env.taken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", default="B1")
    ap.add_argument("--lam", type=float, default=0.0, help="drawdown penalty (B2)")
    ap.add_argument("--sizes", type=float, nargs="+", default=None, help="size menu in units of F0")
    args = ap.parse_args()
    global LAM, SIZES
    LAM = args.lam
    if args.sizes:
        SIZES = np.array(args.sizes)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    df = E.load()
    data = RA.build_data(df)
    X = data["X"]
    C = candidates(df)
    print(f"{len(C)} candidates, {C['squeeze_ok'].mean():.0%} pass the squeeze filter, avgR {C['R'].mean():+.3f} "
          f"(squeeze {C.loc[C.squeeze_ok, 'R'].mean():+.3f} / not {C.loc[~C.squeeze_ok, 'R'].mean():+.3f})", flush=True)
    Env = make_env_class()
    rows = {s: split_rows(C, s) for s in RA.SPLITS}
    path = OUT / "XAUUSD_rl_sizer.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "runs" in old and "variants" not in old:            # the first run's layout
        old = {"variants": {"B1": {k: old[k] for k in ("sizes", "baselines", "runs") if k in old}}}
    res = {"how": __doc__.split("\n\n")[1].strip() + "\n\n" + __doc__.split("\n\n")[2].strip(), "F0": F0,
           "n_candidates": int(len(C)), "variants": old.get("variants", {})}
    var = res["variants"][args.variant] = {"lam": LAM, "sizes": SIZES.tolist(), "baselines": {}, "runs": []}
    base_rules = [("all candidates 1x", lambda r: 1.0), ("all candidates 2x", lambda r: 2.0),
                  ("squeeze passes 1x (= 積極模式)", lambda r: 1.0 if r["squeeze_ok"] else 0.0),
                  ("squeeze passes 1.5x", lambda r: 1.5 if r["squeeze_ok"] else 0.0),
                  ("squeeze passes 2x", lambda r: 2.0 if r["squeeze_ok"] else 0.0),
                  ("squeeze 2x, others 0.5x", lambda r: 2.0 if r["squeeze_ok"] else 0.5)]
    if SIZES.max() > 2:
        base_rules += [("all candidates 3x", lambda r: 3.0), ("squeeze passes 3x", lambda r: 3.0 if r["squeeze_ok"] else 0.0)]
    for name, rule in base_rules:
        var["baselines"][name] = {s: score(C, rule_policy(C, rows[s], rule), s) for s in RA.SPLITS}
        print(f"  {name}: " + ", ".join(f"{s} CAGR {v['cagr']:+.0%} DD {v['max_dd']:.0%} {v['per_week']}/wk obj {v['objective']:+.2f}"
                                        for s, v in var['baselines'][name].items()), flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        venv = DummyVecEnv([lambda i=i: Env(C, X, rows["train"], seed=seed * 100 + i) for i in range(8)])
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=6, gamma=0.99, learning_rate=3e-4,
                    ent_coef=0.01, seed=seed, verbose=0, policy_kwargs={"net_arch": [64, 64]})
        best = {"val": -1e9, "steps": None}
        hist = []

        class CB(BaseCallback):
            def __init__(self):
                super().__init__(); self.next = 50_000

            def _on_step(self):
                if self.num_timesteps >= self.next:
                    self.next += 50_000
                    v = score(C, run_policy(self.model, Env, C, X, rows["val"]), "val")
                    crit = v["objective"] if v.get("trades") else -9          # = log growth per year when LAM = 0
                    hist.append({"steps": self.num_timesteps, "val_cagr": v.get("cagr"), "val_dd": v.get("max_dd"), "val_per_week": v.get("per_week"),
                                 "val_objective": v.get("objective")})
                    if crit > best["val"]:
                        best.update(val=crit, steps=self.num_timesteps)
                        self.model.save(MODELS / f"sizer{args.variant}_seed{seed}_best")
                return True

        t0 = time.time()
        model.learn(total_timesteps=args.steps, callback=CB())
        chosen = PPO.load(MODELS / f"sizer{args.variant}_seed{seed}_best")
        run = {"seed": seed, "minutes": round((time.time() - t0) / 60, 1), "best_steps": best["steps"], "history": hist}
        for s in RA.SPLITS:
            run[s] = score(C, run_policy(chosen, Env, C, X, rows[s]), s)
        var["runs"].append(run)
        print(f"  seed {seed} @ {best['steps']}: " + ", ".join(f"{s} CAGR {run[s].get('cagr', 0):+.0%} DD {run[s].get('max_dd', 0):.0%} "
                                                              f"{run[s].get('per_week')}/wk size {run[s].get('avg_size')} obj {run[s].get('objective')}"
                                                              for s in RA.SPLITS), flush=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
