import numpy as np
import pandas as pd

from niftypulse.research.lead_lag import (
    discretise,
    lead_lag_matrix,
    significance,
    transfer_entropy,
)

NS = 1_000_000_000


def _leader_follower(n: int, lag: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    leader = rng.normal(size=n)
    follower = np.roll(leader, lag) + 0.5 * rng.normal(size=n)
    return leader, follower


def test_discretise_gives_balanced_terciles() -> None:
    sym = discretise(np.random.default_rng(1).normal(size=3000))
    counts = np.bincount(sym, minlength=3)
    assert set(np.unique(sym)) == {0, 1, 2}
    assert counts.min() > 900


def test_te_is_directional_at_the_true_lag() -> None:
    x, y = _leader_follower(5000, lag=2)
    sx, sy = discretise(x), discretise(y)
    assert transfer_entropy(sx, sy, 2) > 0.2
    assert transfer_entropy(sy, sx, 2) < 0.01
    assert transfer_entropy(sx, sy, 2) > 10 * transfer_entropy(sx, sy, 5)


def test_independent_series_are_not_significant() -> None:
    rng = np.random.default_rng(3)
    sx, sy = discretise(rng.normal(size=4000)), discretise(rng.normal(size=4000))
    _, excess, p = significance(sx, sy, 1, surrogates=100, min_shift=200, seed=0)
    assert p > 0.05
    assert abs(excess) < 0.005


def test_matrix_finds_the_leader_from_mid_prices() -> None:
    x, y = _leader_follower(3000, lag=1, seed=5)
    ts = np.arange(3000, dtype=np.int64) * NS
    frame = pd.concat(
        [
            pd.DataFrame({"instrument": "A", "ts": ts, "mid": 100 * np.exp(np.cumsum(x) * 1e-4)}),
            pd.DataFrame({"instrument": "B", "ts": ts, "mid": 100 * np.exp(np.cumsum(y) * 1e-4)}),
        ]
    )
    rows = lead_lag_matrix(frame, [1, 3], surrogates=50, min_shift=100, seed=0)
    best = max(rows, key=lambda r: r.excess_bits)
    assert (best.source, best.target, best.lag_steps) == ("A", "B", 1)
    assert best.p_value < 0.05
