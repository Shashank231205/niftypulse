"""Kalman filters for noisy prices.

`LocalLinearTrend` treats the log mid as a hidden level plus velocity observed with
microstructure noise (bid-ask bounce). It yields a de-noised fair price, a drift
estimate in bps/s, and the normalised innovation (how surprising the last tick was).

`DynamicHedgeRatio` tracks y = alpha + beta * x with random-walk coefficients, e.g. a
perpetual against its spot or ETH against BTC. The residual z-score measures how far the
pair has moved from its current relationship.

Time steps are irregular, so process noise scales with elapsed seconds.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

_BPS = 1e4
_NS = 1_000_000_000


@dataclass
class LocalLinearTrend:
    level_var_per_s: float
    velocity_var_per_s: float
    obs_var: float
    x: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    p: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(2))
    last_ns: int = 0
    innovation_z: float = 0.0

    def update(self, log_mid: float, ts_ns: int) -> None:
        if self.last_ns == 0:
            self.x = np.array([log_mid, 0.0])
            self.p = np.diag([self.obs_var, self.velocity_var_per_s])
            self.last_ns = ts_ns
            return
        dt = max((ts_ns - self.last_ns) / _NS, 1e-3)
        self.last_ns = ts_ns
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = np.diag([self.level_var_per_s * dt, self.velocity_var_per_s * dt])
        x = f @ self.x
        p = f @ self.p @ f.T + q
        s = p[0, 0] + self.obs_var
        innovation = log_mid - x[0]
        k = p[:, 0] / s
        self.x = x + k * innovation
        self.p = p - np.outer(k, p[0, :])
        self.innovation_z = innovation / math.sqrt(s)

    @property
    def fair_log_price(self) -> float:
        return float(self.x[0])

    @property
    def velocity_bps_per_s(self) -> float:
        return float(self.x[1] * _BPS)


@dataclass
class DynamicHedgeRatio:
    coef_var_per_s: float
    obs_var: float
    theta: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 1.0]))
    p: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(2))
    last_ns: int = 0
    residual_z: float = 0.0

    def update(self, x: float, y: float, ts_ns: int) -> None:
        dt = max((ts_ns - self.last_ns) / _NS, 1e-3) if self.last_ns else 1.0
        self.last_ns = ts_ns
        p = self.p + np.eye(2) * self.coef_var_per_s * dt
        h = np.array([1.0, x])
        s = float(h @ p @ h) + self.obs_var
        resid = y - float(h @ self.theta)
        k = p @ h / s
        self.theta = self.theta + k * resid
        self.p = p - np.outer(k, h @ p)
        self.residual_z = resid / math.sqrt(s)

    @property
    def beta(self) -> float:
        return float(self.theta[1])
