"""Bayesian online change-point detection (Adams & MacKay, 2007).

Maintains a posterior over the run length r_t (time since the last change) with a
constant hazard 1/lambda and a Normal-Gamma prior on the segment's mean and precision,
which gives a Student-t predictive. Run lengths are truncated at `max_run` so each
update is O(max_run).

Under a constant hazard p(r_t = 0) always equals the hazard, so the useful alarm is the
probability that a change happened recently: P(r_t < recent_window).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

FloatArray = npt.NDArray[np.float64]


@dataclass
class BOCPD:
    hazard_lambda: float
    max_run: int
    recent_window: int
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    run_probs: FloatArray = field(default_factory=lambda: np.array([1.0]))
    mu: FloatArray = field(init=False)
    kappa: FloatArray = field(init=False)
    alpha: FloatArray = field(init=False)
    beta: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self._reset_params()

    def _reset_params(self) -> None:
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])

    def _predictive(self, x: float) -> FloatArray:
        """Student-t log density of x under each run length's posterior."""
        nu = 2 * self.alpha
        scale2 = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        z = (x - self.mu) ** 2 / (nu * scale2)
        return (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * math.pi * scale2)
            - (nu + 1) / 2 * np.log1p(z)
        )

    def update(self, x: float) -> tuple[float, float]:
        """Returns (P(change within recent_window), expected run length)."""
        h = 1.0 / self.hazard_lambda
        pred = np.exp(self._predictive(x))
        growth = self.run_probs * pred * (1 - h)
        cp = float((self.run_probs * pred * h).sum())
        probs = np.concatenate([[cp], growth])
        total = probs.sum()
        probs = probs / total if total > 0 else np.concatenate([[1.0], np.zeros(len(growth))])

        mu = np.concatenate([[self.mu0], (self.kappa * self.mu + x) / (self.kappa + 1)])
        kappa = np.concatenate([[self.kappa0], self.kappa + 1])
        alpha = np.concatenate([[self.alpha0], self.alpha + 0.5])
        beta = np.concatenate(
            [[self.beta0], self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1))]
        )
        if len(probs) > self.max_run:
            probs, mu, kappa, alpha, beta = (
                a[: self.max_run] for a in (probs, mu, kappa, alpha, beta)
            )
            probs = probs / probs.sum()
        self.run_probs, self.mu, self.kappa, self.alpha, self.beta = probs, mu, kappa, alpha, beta
        recent = float(probs[: self.recent_window].sum())
        expected = float((np.arange(len(probs)) * probs).sum())
        return recent, expected
