"""Contextual model selection by Thompson sampling.

Arms are trained model families; the context is the current regime. Each (regime, arm)
keeps a Gaussian posterior over its mean reward, where reward is the log-score gain over
the class prior once the label is known:

    reward = log p_model(y) - log p_prior(y)

so an arm earns credit only for beating a no-skill forecast. Rewards arrive after the
label horizon (delayed feedback); `observe` is called when they do. Sampling from the
posterior explores uncertain arms and exploits proven ones, and adapts as regimes shift.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class _Posterior:
    mean: float
    var: float
    n: int = 0


@dataclass
class ThompsonSelector:
    arms: list[str]
    n_contexts: int
    prior_mean: float
    prior_var: float
    noise_var: float
    seed: int
    _post: dict[tuple[int, str], _Posterior] = field(init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._post = {
            (c, a): _Posterior(self.prior_mean, self.prior_var)
            for c in range(self.n_contexts)
            for a in self.arms
        }
        self._rng = np.random.default_rng(self.seed)

    def choose(self, context: int) -> str:
        draws = {
            a: self._rng.normal(
                self._post[(context, a)].mean, np.sqrt(self._post[(context, a)].var)
            )
            for a in self.arms
        }
        return max(draws, key=lambda a: draws[a])

    def observe(self, context: int, arm: str, reward: float) -> None:
        """Conjugate Normal update with known observation noise."""
        p = self._post[(context, arm)]
        precision = 1.0 / p.var + 1.0 / self.noise_var
        p.mean = (p.mean / p.var + reward / self.noise_var) / precision
        p.var = 1.0 / precision
        p.n += 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            f"regime{c}/{a}": {"mean": p.mean, "sd": float(np.sqrt(p.var)), "n": p.n}
            for (c, a), p in self._post.items()
        }


def log_score_gain(
    proba: npt.NDArray[np.float64], prior: npt.NDArray[np.float64], y_idx: int
) -> float:
    return float(np.log(max(proba[y_idx], 1e-7)) - np.log(max(prior[y_idx], 1e-7)))
