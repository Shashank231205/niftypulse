"""Reinforcement-learning execution agent (PPO) on the market tape.

Task: buy `parent_lots` child lots within `horizon_s` seconds. Each second the agent
picks one action:
    0 WAIT     keep any resting order, do nothing new
    1 JOIN     rest a one-lot limit at the best bid (if none is resting)
    2 CROSS    buy one lot at the ask now
Anything unfilled at the deadline is crossed. Reward is the negative implementation
shortfall in bps against the arrival mid, including fees, so the agent learns when the
queue is worth waiting in and when to pay the spread.

Observation: remaining fraction, time fraction, spread, L1 imbalance, short-term OFI and
trade imbalance, the direction model's p(up) - p(down), and whether an order is resting
plus its queue ahead relative to displayed size.

Baselines evaluated on the same held-out episodes: cross everything at once, cross one
lot per step (TWAP-style), and join-then-cross-at-deadline.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from niftypulse.backtest.costs import CostModel
from niftypulse.backtest.simulator import RestingOrder, TapeSimulator
from niftypulse.core.config import PolicySettings

FloatArray = npt.NDArray[np.float64]

WAIT, JOIN, CROSS = 0, 1, 2
N_OBS = 9


@dataclass
class EpisodeResult:
    shortfall_bps: float
    passive_fills: int
    aggressive_fills: int


class ExecutionEnv:
    def __init__(
        self,
        sim: TapeSimulator,
        signals: FloatArray,
        costs: CostModel,
        parent_lots: int,
        horizon_s: int,
    ) -> None:
        """`signals[i]`: (imbalance_l1, ofi_l1_5s, trade_imbalance_5s, score) at tape step i."""
        self.sim, self.signals, self.costs = sim, signals, costs
        self.parent, self.horizon = parent_lots, horizon_s

    def reset(self, start: int) -> FloatArray:
        self.start, self.t = start, 0
        self.remaining = self.parent
        self.arrival = self.sim.mid(start)
        self.cost_bps = 0.0
        self.resting: RestingOrder | None = None
        self.passive = self.aggressive = 0
        return self._obs()

    def _i(self) -> int:
        return self.start + self.t

    def _obs(self) -> FloatArray:
        i = self._i()
        spread_bps = 1e4 * (self.sim.ask[i] - self.sim.bid[i]) / self.sim.mid(i)
        queue = 0.0
        if self.resting is not None:
            queue = self.resting.queue_ahead / max(self.sim.bid_qty[i], 1e-9)
        return np.array(
            [
                self.remaining / self.parent,
                self.t / self.horizon,
                spread_bps,
                *self.signals[i],
                float(self.resting is not None),
                min(queue, 5.0),
            ],
            dtype=np.float64,
        )

    def _fill(self, price: float, liquidity: str) -> None:
        fee = self.costs.fee_bps("buy", "maker" if liquidity == "maker" else "taker", 1.0)
        self.cost_bps += 1e4 * (price - self.arrival) / self.arrival + fee
        self.remaining -= 1
        if liquidity == "maker":
            self.passive += 1
        else:
            self.aggressive += 1

    def step(self, action: int) -> tuple[FloatArray, float, bool]:
        i = self._i()
        before = self.cost_bps
        if action == CROSS and self.remaining > 0:
            self._fill(self.sim.market_price(i, "buy"), "taker")
        elif action == JOIN and self.resting is None and self.remaining > 0:
            self.resting = self.sim.join(i, "buy", 1.0)
        self.t += 1
        nxt = self._i()
        if self.resting is not None and self.sim.advance(self.resting, nxt):
            self._fill(self.resting.price, "maker")
            self.resting = None
        done = self.remaining == 0 or self.t >= self.horizon or nxt + 1 >= len(self.sim)
        if done:
            while self.remaining > 0:
                self._fill(self.sim.market_price(nxt, "buy"), "taker")
        reward = -(self.cost_bps - before) / self.parent
        return self._obs(), reward, done

    def result(self) -> EpisodeResult:
        return EpisodeResult(self.cost_bps / self.parent, self.passive, self.aggressive)


class ActorCritic(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_OBS, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class ObsNormalizer:
    def __init__(self) -> None:
        self.mu = np.zeros(N_OBS)
        self.sd = np.ones(N_OBS)

    def fit(self, obs: FloatArray) -> None:
        self.mu, self.sd = obs.mean(axis=0), obs.std(axis=0) + 1e-6

    def __call__(self, obs: FloatArray) -> torch.Tensor:
        return torch.as_tensor((obs - self.mu) / self.sd, dtype=torch.float32)


class PPOAgent:
    def __init__(self, s: PolicySettings) -> None:
        torch.manual_seed(s.seed)
        self.s = s
        self.net = ActorCritic(s.ppo_hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=s.ppo_learning_rate)
        self.norm = ObsNormalizer()
        self.history: list[dict[str, float]] = []

    def act(self, obs: FloatArray, greedy: bool = False) -> tuple[int, float, float]:
        with torch.no_grad():
            logits, value = self.net(self.norm(obs[None, :]))
        dist = torch.distributions.Categorical(logits=logits[0])
        a = int(torch.argmax(logits[0])) if greedy else int(dist.sample())
        return a, float(dist.log_prob(torch.tensor(a))), float(value[0])

    def train(
        self, env: ExecutionEnv, starts: npt.NDArray[np.int64], rng: np.random.Generator
    ) -> None:
        s = self.s
        warm = np.vstack([env.reset(int(st)) for st in starts[: min(200, len(starts))]])
        self.norm.fit(warm)
        batch_episodes = 32
        for it in range(0, s.exec_episodes, batch_episodes):
            obs_l, act_l, logp_l, adv_l, ret_l, ep_costs = [], [], [], [], [], []
            for st in rng.choice(starts, size=batch_episodes):
                obs, done = env.reset(int(st)), False
                o_ep, a_ep, lp_ep, r_ep, v_ep = [], [], [], [], []
                while not done:
                    a, lp, v = self.act(obs)
                    nxt, r, done = env.step(a)
                    o_ep.append(obs)
                    a_ep.append(a)
                    lp_ep.append(lp)
                    r_ep.append(r)
                    v_ep.append(v)
                    obs = nxt
                adv, ret = self._gae(np.array(r_ep), np.array(v_ep))
                obs_l += o_ep
                act_l += a_ep
                logp_l += lp_ep
                adv_l += list(adv)
                ret_l += list(ret)
                ep_costs.append(env.result().shortfall_bps)
            self._update(
                np.array(obs_l), np.array(act_l), np.array(logp_l), np.array(adv_l), np.array(ret_l)
            )
            self.history.append(
                {"episodes": it + batch_episodes, "mean_shortfall_bps": float(np.mean(ep_costs))}
            )

    def _gae(self, rewards: FloatArray, values: FloatArray) -> tuple[FloatArray, FloatArray]:
        s = self.s
        adv = np.zeros_like(rewards)
        last = 0.0
        for t in reversed(range(len(rewards))):
            nxt_v = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + s.ppo_gamma * nxt_v - values[t]
            last = delta + s.ppo_gamma * s.ppo_lambda * last
            adv[t] = last
        return adv, adv + values

    def _update(
        self,
        obs: FloatArray,
        acts: FloatArray,
        old_logp: FloatArray,
        adv: FloatArray,
        ret: FloatArray,
    ) -> None:
        s = self.s
        x = self.norm(obs)
        a = torch.as_tensor(acts, dtype=torch.int64)
        olp = torch.as_tensor(old_logp, dtype=torch.float32)
        advantage = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        target = torch.as_tensor(ret, dtype=torch.float32)
        for _ in range(s.ppo_epochs):
            logits, value = self.net(x)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(a) - olp)
            clipped = torch.clamp(ratio, 1 - s.ppo_clip, 1 + s.ppo_clip)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
            value_loss = nn.functional.mse_loss(value, target)
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()


def run_policy(
    env: ExecutionEnv, start: int, policy: str, agent: PPOAgent | None = None
) -> EpisodeResult:
    obs, done = env.reset(start), False
    while not done:
        if policy == "ppo":
            assert agent is not None
            action = agent.act(obs, greedy=True)[0]
        elif policy == "cross_all":
            action = CROSS
        elif policy == "twap":
            action = CROSS if env.t % max(1, env.horizon // env.parent) == 0 else WAIT
        else:
            action = JOIN
        obs, _, done = env.step(action)
    return env.result()


def evaluate(
    env: ExecutionEnv, starts: npt.NDArray[np.int64], agent: PPOAgent
) -> dict[str, dict[str, float]]:
    out = {}
    for name in ("ppo", "cross_all", "twap", "join_then_cross"):
        results = [run_policy(env, int(st), name, agent) for st in starts]
        costs = np.array([r.shortfall_bps for r in results])
        passive = sum(r.passive_fills for r in results)
        total = passive + sum(r.aggressive_fills for r in results)
        out[name] = {
            "mean_shortfall_bps": float(costs.mean()),
            "stderr_bps": float(costs.std(ddof=1) / np.sqrt(len(costs))),
            "passive_fill_share": passive / total if total else 0.0,
            "episodes": len(costs),
        }
    return out
