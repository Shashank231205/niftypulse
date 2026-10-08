"""Hidden Markov regime model.

A Gaussian HMM is fitted on training data only, over (log realized vol, |return|, log
spread). States are re-ordered by volatility so 0 = calm and K-1 = most volatile,
which keeps regime ids stable across retrains.

Live and in research we use the *forward filter* p(state_t | x_1..x_t), not Viterbi:
Viterbi decodes each state using the whole sequence, including the future, which would
leak information into features and inflate backtests.
"""

import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

FloatArray = npt.NDArray[np.float64]


class RegimeHMM:
    def __init__(self, n_states: int, seed: int) -> None:
        self.n_states = n_states
        self._hmm = GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=200, random_state=seed
        )
        self.means: FloatArray | None = None
        self.covars: FloatArray | None = None
        self.transmat: FloatArray | None = None
        self.startprob: FloatArray | None = None

    def fit(self, x: FloatArray) -> "RegimeHMM":
        self._hmm.fit(x)
        order = np.argsort(self._hmm.means_[:, 0])
        self.means = self._hmm.means_[order]
        self.covars = self._hmm.covars_[order]
        self.transmat = self._hmm.transmat_[np.ix_(order, order)]
        self.startprob = self._hmm.startprob_[order]
        return self

    def filter(self, x: FloatArray) -> FloatArray:
        """Causal state probabilities, one row per observation."""
        return OnlineRegimeFilter(self).run(x)


class OnlineRegimeFilter:
    """Incremental forward algorithm: one O(K^2) update per observation."""

    def __init__(self, model: RegimeHMM) -> None:
        assert model.means is not None and model.covars is not None
        assert model.transmat is not None and model.startprob is not None
        self._model = model
        self._dists = [
            multivariate_normal(mean=m, cov=c, allow_singular=True)
            for m, c in zip(model.means, model.covars, strict=True)
        ]
        self.alpha: FloatArray = model.startprob.copy()
        self._started = False

    def update(self, obs: FloatArray) -> FloatArray:
        assert self._model.transmat is not None
        like = np.array([max(d.pdf(obs), 1e-300) for d in self._dists])
        prior = self.alpha if not self._started else self.alpha @ self._model.transmat
        post = prior * like
        total = post.sum()
        self.alpha = post / total if total > 0 else np.full(len(post), 1 / len(post))
        self._started = True
        return self.alpha

    def run(self, x: FloatArray) -> FloatArray:
        return np.vstack([self.update(row) for row in x])


def regime_observations(
    realized_vol_bps: FloatArray, abs_return_bps: FloatArray, spread_bps: FloatArray
) -> FloatArray:
    return np.column_stack(
        [
            np.log(np.maximum(realized_vol_bps, 1e-3)),
            np.log1p(np.abs(abs_return_bps)),
            np.log(np.maximum(spread_bps, 1e-3)),
        ]
    )
