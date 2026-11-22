"""Reinforcement learning A: an end-to-end gold trading agent (docs/RESEARCH.md section 13).

Every 15-minute bar the agent chooses a target position - long, flat or short (1x notional) - and
holds it to the next bar. It sees what the rule system sees (multi-timeframe trend, EMAs, ADX, RSI,
ATRs, Bollinger width percentile, channel position, extension) plus recent price changes in ATR
units, tick-volume ratio, time of day / week and its own position, open profit and time in the
trade. Episode = one trading session (17:00 -> 16:45 New York); it is forced flat at the session's
last bar, like 當日平倉模式, so it never holds overnight.

Reward per bar = position x price change / daily ATR - costs / daily ATR, with the same costs as
the back-tests: half the 0.012 % spread plus 0.005 % slippage per unit of position change (a
round trip costs one spread and two slippages; a long -> short flip costs two units).

Training: PPO (stable-baselines3) on sessions 2005-2016; every EVAL_EVERY steps the deterministic
policy is scored on 2017-2020 (validation) and the best checkpoint is kept; 2021-2025/3 is scored
once, at the end, for the chosen checkpoint. Several seeds. Results are compared with the rule
system on the same days by the annualised Sharpe ratio of daily P&L - scale-free, so the agent
and the system are judged on the same yardstick whatever their position sizes. (The
"best_median_year_at_kelly" column, exp(Sharpe^2 / 2) - 1, is the normal-distribution
approximation; it badly understates strategies whose profit comes from a few large trends, like
the rule system - research/target_100.py does that calculation on the actual trades.)

Variants (results/XAUUSD_rl_agent.json, one entry each):
  A1  decides every 15-minute bar, trained with the real costs;
  A2  decides once an hour (holds 4 bars), trained with twice the costs (to discourage the
      over-trading A1 shows) - scored with the real costs like everything else.

Needs: pip install stable-baselines3 gymnasium (brings PyTorch).

    python research/rl_agent.py --variant A1 --steps 2000000 --seeds 0 1 2
    python research/rl_agent.py --variant A2 --every 4 --train-cost 2 --steps 2000000 --seeds 0 1 2
    python research/rl_agent.py --speed-test
    python research/rl_agent.py --merge other_run.json      # add another run's variants (made with --out)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import efficient_active as E  # noqa: E402
from goldsignal import strategy as S  # noqa: E402
from goldsignal.instruments import INSTRUMENTS  # noqa: E402
from goldsignal.plans import Clock  # noqa: E402

OUT = ROOT / "results"
MODELS = ROOT / "data" / "cache" / "rl_models"
SPLITS = {"train": ("2005-01-01", "2016-12-31"), "val": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2025-03-21")}
EVAL_EVERY = 250_000


# ----------------------------------------------------------------------------- data

def build_data(df: pd.DataFrame | None = None) -> dict:
    """Feature matrix X (n_bars x k, float32), prices, ATRs, costs and the sessions of each split."""
    df = E.load() if df is None else df
    inst = INSTRUMENTS["XAUUSD"]
    p = inst.strategy_params(True, squeeze=True)
    f = S.compute_features(df, params=p)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df else np.ones(len(df))
    a1 = f["h1_atr"].to_numpy(float)
    ad = f["d1_atr"].to_numpy(float)
    clock = Clock.build(df.index)

    def lag(x, k):
        return np.r_[np.full(k, np.nan), x[:-k]]

    cols = {}
    for k in (1, 2, 4, 8, 16, 32, 96):
        cols[f"ret{k}"] = np.clip((c - lag(c, k)) / a1, -12, 12) / 3
    cols["c_h1e20"] = np.clip((c - f["h1_ema20"].to_numpy()) / a1, -10, 10) / 3
    cols["c_h1e50"] = np.clip((c - f["h1_ema50"].to_numpy()) / a1, -10, 10) / 3
    cols["c_m15e20"] = np.clip((c - f["m15_ema20"].to_numpy()) / a1, -10, 10) / 3
    cols["h4_e20"] = np.clip((f["h4_close"] - f["h4_ema20"]).to_numpy() / f["h4_atr"].to_numpy(), -6, 6) / 2
    cols["d1_e20"] = np.clip((f["d1_close"] - f["d1_ema20"]).to_numpy() / ad, -6, 6) / 2
    cols["d1_e2050"] = np.clip((f["d1_ema20"] - f["d1_ema50"]).to_numpy() / ad, -6, 6) / 2
    for k in ("trend_d1", "trend_h4", "trend_h1"):
        cols[k] = f[k].to_numpy(float)
    for k in ("h1_adx", "h4_adx", "d1_adx"):
        cols[k] = f[k].to_numpy(float) / 50 - 0.5
    for k in ("m15_rsi", "h1_rsi", "h4_rsi", "d1_rsi"):
        cols[k] = f[k].to_numpy(float) / 100 - 0.5
    cols["bw_pct"] = f["m15_bw_pct"].to_numpy(float) - 0.5
    rng = (f["chan_hi"] - f["chan_lo"]).to_numpy(float)
    cols["chan_pos"] = np.clip((c - f["chan_lo"].to_numpy(float)) / np.where(rng > 0, rng, np.nan), -0.5, 1.5) - 0.5
    cols["d1_ext"] = np.clip(f["d1_ext"].to_numpy(float), -4, 4) / 2
    cols["a1_ad"] = np.clip(a1 / ad, 0, 1) * 2 - 0.5
    cols["m15_a1"] = np.clip(f["m15_atr"].to_numpy(float) / a1, 0, 2) - 0.5
    vm = pd.Series(v).rolling(96 * 5, min_periods=50).median().to_numpy()
    cols["vol_ratio"] = np.clip(np.log((v + 1) / (vm + 1)), -3, 3) / 2
    h = clock.sess_h
    cols["sess_sin"], cols["sess_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    dow = clock.sess_dow.astype(float)
    cols["dow_sin"], cols["dow_cos"] = np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)
    X = np.nan_to_num(np.column_stack(list(cols.values())).astype(np.float32))

    cost = c * (inst.spread_pct / 2 + inst.slip_pct)          # per unit of position change, in $
    sess = clock.session
    idx = df.index
    splits = {}
    for name, (a, b) in SPLITS.items():
        m = np.asarray((idx >= pd.Timestamp(a, tz="UTC")) & (idx < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)))
        ids = np.unique(sess[m])
        bounds = []
        for s_id in ids:
            w = np.flatnonzero((sess == s_id) & m)
            if len(w) >= 8 and np.all(np.isfinite(ad[w])) and np.all(ad[w] > 0):
                bounds.append((int(w[0]), int(w[-1])))
        splits[name] = bounds
    return {"X": X, "c": c, "ad": ad, "a1": a1, "cost": cost, "splits": splits, "cols": list(cols), "index": idx}


# ----------------------------------------------------------------------------- environment

def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class GoldSessionEnv(gym.Env):
        """One trading session per episode; actions 0 = flat, 1 = long, 2 = short."""
        metadata = {"render_modes": []}
        POS = np.array([0.0, 1.0, -1.0])

        def __init__(self, data: dict, split: str = "train", seed: int = 0, sequential: bool = False, every: int = 1):
            super().__init__()
            self.d = data
            self.every = every                      # bars per decision (the position is held in between)
            self.sessions = data["splits"][split]
            self.k = data["X"].shape[1] + 4
            self.observation_space = spaces.Box(-10, 10, shape=(self.k,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)
            self.rng = np.random.default_rng(seed)
            self.sequential = sequential
            self.ptr = 0

        def _obs(self):
            t = self.t
            unreal = self.pos * (self.d["c"][t] - self.entry) / self.d["a1"][t] if self.pos else 0.0
            extra = np.array([self.pos, np.clip(unreal, -10, 10) / 3, min(self.held, 192) / 96, (self.end - t) / 96], dtype=np.float32)
            return np.concatenate([self.d["X"][t], extra])

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if self.sequential:
                self.start, self.end = self.sessions[self.ptr % len(self.sessions)]
                self.ptr += 1
            else:
                self.start, self.end = self.sessions[self.rng.integers(len(self.sessions))]
            self.t = self.start
            self.pos, self.entry, self.held = 0.0, 0.0, 0
            self.pnl_pct = 0.0
            self.trades = []
            return self._obs(), {}

        def step(self, action):
            d, t = self.d, self.t
            new = self.POS[int(action)] if t < self.end else 0.0          # forced flat on the last bar
            ch = abs(new - self.pos)
            cost = ch * d["cost"][t]
            if new != self.pos:
                if self.pos != 0:
                    self.trades.append(self.pos * (d["c"][t] - self.entry) / self.entry - d["cost"][t] / d["c"][t] * 2 * 1.0)
                if new != 0:
                    self.entry, self.held = d["c"][t], 0
            self.pos = new
            if t < self.end:
                t2 = min(t + self.every, self.end)
                move = d["c"][t2] - d["c"][t]
                r = (new * move - cost) / d["ad"][t]
                self.pnl_pct += (new * move - cost) / d["c"][t]
                self.t = t2
                self.held += (t2 - t) if new else 0
                return self._obs(), float(r), False, False, {}
            r = -cost / d["ad"][t]
            self.pnl_pct += -cost / d["c"][t]
            return self._obs(), float(r), True, False, {"pnl_pct": self.pnl_pct, "trades": self.trades}

    return GoldSessionEnv


def evaluate(model, data: dict, split: str, cost_mult: float = 1.0, every: int = 1) -> dict:
    """Deterministic policy over every session of the split: daily P&L (% of notional), trades."""
    Env = make_env_class()
    d = dict(data)
    if cost_mult != 1.0:
        d = dict(data, cost=data["cost"] * cost_mult)
    env = Env(d, split, sequential=True, every=every)
    days, trades, dates, long_bars, short_bars, bars = [], [], [], 0, 0, 0
    for _ in range(len(env.sessions)):
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            t0 = env.t
            obs, r, done, _, info = env.step(a)
            n = max(env.t - t0, 1)
            bars += n
            long_bars += n * (env.pos > 0)
            short_bars += n * (env.pos < 0)
        days.append(info["pnl_pct"]); trades += info["trades"]
        dates.append(data["index"][env.end])
    return summarise(np.array(days), np.array(trades), dates, extra={"long%": round(100 * long_bars / bars, 1), "short%": round(100 * short_bars / bars, 1)})


def summarise(days: np.ndarray, trades: np.ndarray, dates, extra: dict | None = None) -> dict:
    years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 0.25)
    mu, sd = float(days.mean()), float(days.std(ddof=1)) if len(days) > 1 else 0.0
    sharpe = mu / sd * np.sqrt(252) if sd > 0 else 0.0
    eq = np.cumsum(days)
    mdd = float(np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq)) if len(eq) else 0.0
    out = {"days": int(len(days)), "sharpe": round(sharpe, 2), "mean_day_%": round(100 * mu, 4), "sd_day_%": round(100 * sd, 3),
           "total_%_1x": round(100 * float(days.sum()), 1), "max_dd_%_1x": round(100 * mdd, 1),
           "trades": int(len(trades)), "trades_per_week": round(len(trades) / years / 52.18, 2),
           "avg_trade_%": round(100 * float(trades.mean()), 4) if len(trades) else None,
           "win%": round(100 * float((trades > 0).mean()), 1) if len(trades) else None,
           "kelly_leverage": round(mu / sd ** 2, 1) if sd > 0 else None,
           "best_median_year_at_kelly": round(float(np.expm1(sharpe ** 2 / 2)), 2)}
    if extra:
        out.update(extra)
    return out


def system_daily(split: str, data: dict) -> dict:
    """The rule system's daily P&L over the same sessions, for comparison: each closed trade's
    % return (exit vs entry, costs included) booked on its exit day, 1x notional per position."""
    import pickle
    res = {}
    for mode in ("efficient", "active"):
        p = ROOT / "data" / "cache" / f"target100_{mode}.pkl"
        if not p.exists():
            continue
        tr = pd.read_pickle(p)
        a, b = SPLITS[split]
        tr = tr[(tr["entry_time"] >= pd.Timestamp(a, tz="UTC")) & (tr["entry_time"] < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1))]
        ret = (tr["R"] * tr["risk"] / tr["entry"]).to_numpy(float)          # % of notional
        day = pd.Series(ret, index=tr["exit_time"].dt.tz_convert("America/New_York").dt.normalize()).groupby(level=0).sum()
        bdays = pd.bdate_range(pd.Timestamp(a), pd.Timestamp(b), tz="America/New_York")
        day = day.reindex(bdays, fill_value=0.0)
        res[mode] = summarise(day.to_numpy(), ret, list(bdays))
    return res


def long_daily(split: str, data: dict) -> dict:
    """The naive benchmark: long 1x through every session of the split, flat at its end (one round
    trip of costs a day) - what an agent that only learned "gold goes up" would earn."""
    c, cost = data["c"], data["cost"]
    days, dates = [], []
    for a, b in data["splits"][split]:
        days.append((c[b] - c[a]) / c[a] - 2 * cost[a] / c[a])
        dates.append(data["index"][b])
    days = np.array(days)
    return summarise(days, days, dates)


# ----------------------------------------------------------------------------- training

def train(data: dict, steps: int, seed: int, tag: str = "ppo", every: int = 1, train_cost: float = 1.0,
          eval_every: int = EVAL_EVERY, threads: int = 1) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(threads)
    Env = make_env_class()
    tdata = dict(data, cost=data["cost"] * train_cost) if train_cost != 1.0 else data
    venv = DummyVecEnv([lambda i=i: Env(tdata, "train", seed=seed * 100 + i, every=every) for i in range(8)])
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=6, gamma=0.99 ** every, gae_lambda=0.95,
                learning_rate=2.5e-4, ent_coef=0.005, clip_range=0.2, seed=seed, verbose=0,
                policy_kwargs={"net_arch": [128, 128]})
    MODELS.mkdir(parents=True, exist_ok=True)
    best = {"val_sharpe": -1e9}
    history = []

    class EvalCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = eval_every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next:
                self.next += eval_every
                v = evaluate(self.model, data, "val", every=every)
                history.append({"steps": self.num_timesteps, **{k: v[k] for k in ("sharpe", "total_%_1x", "trades_per_week", "avg_trade_%")}})
                print(f"    seed {seed} {self.num_timesteps:>9,d} steps: val Sharpe {v['sharpe']:+.2f}, {v['trades_per_week']:.1f} trades/wk, "
                      f"avg trade {v['avg_trade_%']}%", flush=True)
                if v["sharpe"] > best["val_sharpe"]:
                    best.update(val_sharpe=v["sharpe"], steps=self.num_timesteps)
                    self.model.save(MODELS / f"{tag}_seed{seed}_best")
            return True

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=EvalCB())
    model.save(MODELS / f"{tag}_seed{seed}_last")
    chosen = PPO.load(MODELS / f"{tag}_seed{seed}_best") if (MODELS / f"{tag}_seed{seed}_best.zip").exists() else model
    out = {"seed": seed, "steps": steps, "minutes": round((time.time() - t0) / 60, 1), "best_steps": best.get("steps"),
           "history": history,
           "train": evaluate(chosen, data, "train", every=every), "val": evaluate(chosen, data, "val", every=every),
           "test": evaluate(chosen, data, "test", every=every),
           "test_cost_x2": evaluate(chosen, data, "test", cost_mult=2.0, every=every)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--speed-test", action="store_true")
    ap.add_argument("--variant", default="A1")
    ap.add_argument("--every", type=int, default=1, help="bars per decision")
    ap.add_argument("--train-cost", type=float, default=1.0, help="cost multiple used in training only")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(OUT / "XAUUSD_rl_agent.json"))
    ap.add_argument("--merge", default=None, help="add the variants of another result file (a run with --out) and the long benchmark")
    args = ap.parse_args()
    data = build_data()
    if args.merge:
        path = Path(args.out)
        res = json.loads(path.read_text(encoding="utf-8"))
        res.setdefault("variants", {}).update(json.loads(Path(args.merge).read_text(encoding="utf-8"))["variants"])
        res["long_every_session"] = {s: long_daily(s, data) for s in SPLITS}
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        return
    print("features:", len(data["cols"]), {k: len(v) for k, v in data["splits"].items()}, "sessions", flush=True)
    if args.speed_test:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        Env = make_env_class()
        venv = DummyVecEnv([lambda i=i: Env(data, "train", seed=i) for i in range(8)])
        m = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=6, policy_kwargs={"net_arch": [128, 128]}, verbose=0)
        t0 = time.time(); m.learn(total_timesteps=40_000); dt = time.time() - t0
        print(f"{40_000 / dt:.0f} steps/s; eval val:", end=" ", flush=True)
        t0 = time.time(); v = evaluate(m, data, "val"); print(f"{time.time() - t0:.0f}s", v)
        return
    path = Path(args.out)
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    res.update({"how": __doc__.split("\n\n")[1].strip(), "features": data["cols"],
                "system": {s: system_daily(s, data) for s in ("train", "val", "test")},
                "long_every_session": {s: long_daily(s, data) for s in SPLITS}})
    res.setdefault("variants", {})
    res["variants"][args.variant] = {"every": args.every, "train_cost": args.train_cost, "steps": args.steps, "runs": []}
    for seed in args.seeds:
        print(f"{args.variant} seed {seed}", flush=True)
        r = train(data, args.steps, seed, tag=f"ppo{args.variant}", every=args.every, train_cost=args.train_cost,
                  eval_every=args.eval_every, threads=args.threads)
        res["variants"][args.variant]["runs"].append(r)
        print(f"  chosen @ {r['best_steps']}: train Sharpe {r['train']['sharpe']}, val {r['val']['sharpe']}, test {r['test']['sharpe']} "
              f"(cost x2: {r['test_cost_x2']['sharpe']}), test trades/wk {r['test']['trades_per_week']}", flush=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
