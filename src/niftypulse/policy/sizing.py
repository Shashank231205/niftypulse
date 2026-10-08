"""Position sizing as a convex program.

    maximise    mu' w  -  gamma * w' S w  -  kappa * ||w - w_prev||_1
    subject to  |w_i| <= max_position,  ||w||_1 <= max_gross

mu is the expected forward return in bps implied by the calibrated probabilities,
S a Ledoit-Wolf shrunk covariance of recent returns, and the L1 term charges turnover
so the book does not churn on small signal changes. Solved with CVXPY.
"""

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf

from niftypulse.core.config import PolicySettings

FloatArray = npt.NDArray[np.float64]


def expected_return_bps(proba: FloatArray, mean_up_bps: float, mean_down_bps: float) -> FloatArray:
    """E[r] = p_up * E[r | up] + p_down * E[r | down] (flat contributes ~0)."""
    return proba[:, 2] * mean_up_bps + proba[:, 0] * mean_down_bps


def shrunk_covariance(returns_bps: FloatArray) -> FloatArray:
    return np.asarray(LedoitWolf().fit(returns_bps).covariance_, dtype=np.float64)


class ConvexSizer:
    def __init__(self, settings: PolicySettings, n_assets: int) -> None:
        s = settings
        self._mu = cp.Parameter(n_assets)
        self._prev = cp.Parameter(n_assets)
        self._chol = cp.Parameter((n_assets, n_assets))
        self.w = cp.Variable(n_assets)
        risk = cp.sum_squares(self._chol.T @ self.w)
        objective = cp.Maximize(
            self._mu @ self.w
            - s.risk_aversion * risk
            - s.turnover_penalty_bps * cp.norm1(self.w - self._prev)
        )
        constraints = [cp.abs(self.w) <= s.max_position, cp.norm1(self.w) <= s.max_gross]
        self._problem = cp.Problem(objective, constraints)

    def solve(self, mu_bps: FloatArray, cov_bps: FloatArray, prev: FloatArray) -> FloatArray:
        """Parameterised once (DPP), so each re-solve reuses the compiled problem."""
        jitter = 1e-9 * np.eye(len(mu_bps))
        self._mu.value = mu_bps
        self._prev.value = prev
        self._chol.value = np.linalg.cholesky(cov_bps + jitter)
        self._problem.solve(solver=cp.CLARABEL)
        if self.w.value is None:
            return prev
        return np.asarray(self.w.value, dtype=np.float64)
