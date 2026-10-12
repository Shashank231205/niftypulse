import numpy as np
import pytest

from niftypulse.backtest.costs import BinanceCosts, NseFuturesCosts
from niftypulse.backtest.simulator import TapeSimulator
from niftypulse.core.config import CostSettings, PolicySettings
from niftypulse.policy.bandit import ThompsonSelector
from niftypulse.policy.sizing import ConvexSizer
from niftypulse.regime.changepoint import BOCPD
from niftypulse.regime.kalman import DynamicHedgeRatio, LocalLinearTrend

NS = 1_000_000_000


def test_bocpd_flags_a_mean_shift() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(6, 1, 50)])
    cpd = BOCPD(hazard_lambda=200, max_run=400, recent_window=10)
    probs = [cpd.update(v)[0] for v in x]
    assert max(probs[:290]) < 0.5
    assert max(probs[300:320]) > 0.8


def test_kalman_recovers_drift_and_hedge_ratio() -> None:
    kf = LocalLinearTrend(level_var_per_s=1e-10, velocity_var_per_s=1e-12, obs_var=1e-8)
    for t in range(600):
        kf.update(np.log(100.0) + 2e-5 * t, t * NS)  # +0.2 bps per second
    assert kf.velocity_bps_per_s == pytest.approx(0.2, rel=0.1)
    hr = DynamicHedgeRatio(coef_var_per_s=1e-6, obs_var=1e-4)
    rng = np.random.default_rng(1)
    for t in range(2000):
        x = rng.normal()
        hr.update(x, 0.5 + 1.7 * x + 0.01 * rng.normal(), t * NS)
    assert hr.beta == pytest.approx(1.7, abs=0.05)


def test_thompson_bandit_learns_the_better_arm() -> None:
    bandit = ThompsonSelector(
        ["good", "bad"], n_contexts=1, prior_mean=0, prior_var=1, noise_var=1, seed=0
    )
    rng = np.random.default_rng(0)
    for _ in range(400):
        arm = bandit.choose(0)
        bandit.observe(0, arm, rng.normal(0.3 if arm == "good" else -0.3, 1))
    picks = [bandit.choose(0) for _ in range(200)]
    assert picks.count("good") > 180


def test_convex_sizer_respects_limits_and_signal_sign() -> None:
    s = PolicySettings(_env_file=None)  # type: ignore[call-arg]
    sizer = ConvexSizer(s, 3)
    w = sizer.solve(np.array([5.0, -5.0, 0.0]), np.eye(3), np.zeros(3))
    assert w[0] > 0 > w[1] and abs(w[2]) < 1e-6
    assert np.all(np.abs(w) <= s.max_position + 1e-6) and np.abs(w).sum() <= s.max_gross + 1e-6


def test_cost_models() -> None:
    assert BinanceCosts(2.0, 5.0).fee(10_000, "buy", "taker") == pytest.approx(5.0)
    nse = NseFuturesCosts(CostSettings(_env_file=None))  # type: ignore[call-arg]
    sell, buy = nse.fee(1_000_000, "sell", "taker"), nse.fee(1_000_000, "buy", "taker")
    assert sell > buy  # STT applies to the sell leg only
    assert 200 < sell < 300  # 20 brokerage + 200 STT + charges + GST


def _tape(**cols: list[float]):  # type: ignore[no-untyped-def]
    import pandas as pd

    n = len(next(iter(cols.values())))
    base = {
        "ts": [i * NS for i in range(n)],
        "bid": [100.0] * n,
        "ask": [100.1] * n,
        "bid_qty": [5.0] * n,
        "ask_qty": [5.0] * n,
        "sell_volume": [0.0] * n,
        "buy_volume": [0.0] * n,
        "trade_low": [np.nan] * n,
        "trade_high": [np.nan] * n,
    }
    base.update(cols)
    return TapeSimulator(pd.DataFrame(base))


def test_queue_fill_needs_queue_ahead_to_trade_through() -> None:
    sim = _tape(sell_volume=[0.0, 3.0, 3.0, 0.0])
    order = sim.join(0, "buy", 1.0)
    assert order.queue_ahead == 5.0
    assert not sim.advance(order, 1)  # 3 of 5 ahead traded
    assert sim.advance(order, 2)  # remaining 2 exhausted, we fill
    through = _tape(trade_low=[np.nan, 99.9])
    o2 = through.join(0, "buy", 1.0)
    assert through.advance(o2, 1)  # price traded below our bid
