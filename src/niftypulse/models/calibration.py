"""Probability calibration.

Class-weighted training ranks well but distorts probabilities (it inflates the rare
classes). `BiasedTemperature` fits one temperature plus a per-class bias on held-out
data: softmax(log p / T + b). The temperature fixes over/under-confidence, the bias undoes
the prior shift from weighting. `IsotonicCalibrator` is used for binary targets (fill
probability).
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

FloatArray = npt.NDArray[np.float64]


def _softmax(z: FloatArray) -> FloatArray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class BiasedTemperature:
    def __init__(self) -> None:
        self.temperature = 1.0
        self.bias = np.zeros(3)

    def fit(self, proba: FloatArray, y_idx: npt.NDArray[np.int64]) -> "BiasedTemperature":
        logp = np.log(np.clip(proba, 1e-7, 1.0))

        def nll(params: FloatArray) -> float:
            t = np.exp(params[0])
            b = np.array([params[1], 0.0, params[2]])
            p = _softmax(logp / t + b)
            return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12)))

        res = minimize(nll, x0=np.zeros(3), method="L-BFGS-B")
        self.temperature = float(np.exp(res.x[0]))
        self.bias = np.array([res.x[1], 0.0, res.x[2]])
        return self

    def transform(self, proba: FloatArray) -> FloatArray:
        logp = np.log(np.clip(proba, 1e-7, 1.0))
        return _softmax(logp / self.temperature + self.bias)


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, score: FloatArray, y: FloatArray) -> "IsotonicCalibrator":
        self._iso.fit(score, y)
        return self

    def transform(self, score: FloatArray) -> FloatArray:
        return np.asarray(self._iso.predict(score), dtype=np.float64)
