"""Cross-instrument lead-lag, measured with transfer entropy.

Transfer entropy TE(X -> Y) at lag k is how much X's move k steps ago reduces uncertainty
about Y's next move beyond what Y's own last move already tells us:

    TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-k})      (bits)

Unlike correlation it is asymmetric and catches non-linear dependence, so TE(perp -> spot)
far above TE(spot -> perp) says the perpetual leads. Returns are discretised into
(down, flat, up) by tercile. The plug-in estimator is biased upwards, so every value is
compared with circular-shift surrogates of X: a shift keeps X's own autocorrelation but
breaks its timing with Y. Reported "excess" is TE minus the surrogate mean, and the
p-value is the share of surrogates at or above the observed TE. Many pairs and lags are
tested at once, so each result also carries a Benjamini-Hochberg q-value.
"""

from collections import Counter
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

IntArray = npt.NDArray[np.int64]


def discretise(returns: npt.NDArray[np.float64]) -> IntArray:
    """Tercile symbols 0/1/2; ties at zero (no move) fall into the middle bin."""
    lo, hi = np.nanquantile(returns, [1 / 3, 2 / 3])
    if lo == hi:
        return np.where(returns < lo, 0, np.where(returns > hi, 2, 1)).astype(np.int64)
    return np.digitize(returns, [lo, hi]).astype(np.int64)


def _entropy(*columns: IntArray) -> float:
    counts = np.array(list(Counter(zip(*columns, strict=True)).values()), dtype=np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def transfer_entropy(source: IntArray, target: IntArray, lag: int) -> float:
    """TE(source -> target) in bits with history length 1 and source lag `lag` >= 1."""
    y_next = target[lag:]
    y_prev = target[lag - 1 : -1]
    x_lag = source[:-lag]
    # H(Yt|Yp) - H(Yt|Yp,X) = H(Yt,Yp) - H(Yp) - H(Yt,Yp,X) + H(Yp,X)
    return (
        _entropy(y_next, y_prev)
        - _entropy(y_prev)
        - _entropy(y_next, y_prev, x_lag)
        + _entropy(y_prev, x_lag)
    )


@dataclass(frozen=True, slots=True)
class LeadLag:
    source: str
    target: str
    lag_steps: int
    te_bits: float
    excess_bits: float
    p_value: float
    q_value: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "lag_steps": self.lag_steps,
            "te_bits": self.te_bits,
            "excess_bits": self.excess_bits,
            "p_value": self.p_value,
            "q_value": self.q_value,
        }


def significance(
    source: IntArray, target: IntArray, lag: int, surrogates: int, min_shift: int, seed: int
) -> tuple[float, float, float]:
    """(te, te minus surrogate mean, p-value) against circular-shift surrogates of source."""
    rng = np.random.default_rng(seed)
    observed = transfer_entropy(source, target, lag)
    n = len(source)
    shifts = rng.integers(min_shift, n - min_shift, size=surrogates)
    null = np.array([transfer_entropy(np.roll(source, s), target, lag) for s in shifts])
    p = (1 + int((null >= observed).sum())) / (1 + surrogates)
    return observed, observed - float(null.mean()), p


def benjamini_hochberg(p_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """False-discovery-rate adjusted p-values (step-up, monotone)."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def aligned_symbols(frame: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """One column of move symbols per instrument, on the shared sampling grid."""
    wide = frame.pivot_table(index="ts", columns="instrument", values=return_col)
    wide = wide.dropna()
    return wide.apply(lambda col: pd.Series(discretise(col.to_numpy()), index=col.index))


def step_returns_bps(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward one-step mid return per instrument; only past data, unlike the labels."""
    out = frame[["instrument", "ts", "mid"]].sort_values(["instrument", "ts"]).copy()
    out["step_ret_bps"] = out.groupby("instrument")["mid"].transform(
        lambda m: 1e4 * np.log(m).diff()
    )
    return out.dropna(subset=["step_ret_bps"])


def lead_lag_matrix(
    frame: pd.DataFrame, lags: list[int], surrogates: int, min_shift: int, seed: int
) -> list[LeadLag]:
    """TE for every ordered instrument pair and lag."""
    symbols = aligned_symbols(step_returns_bps(frame), "step_ret_bps")
    results = []
    for src, dst in permutations(symbols.columns, 2):
        x = symbols[src].to_numpy(dtype=np.int64)
        y = symbols[dst].to_numpy(dtype=np.int64)
        for lag in lags:
            te, excess, p = significance(x, y, lag, surrogates, min_shift, seed + lag)
            results.append(LeadLag(str(src), str(dst), lag, te, excess, p))
    q = benjamini_hochberg(np.array([r.p_value for r in results]))
    return [replace(r, q_value=float(qi)) for r, qi in zip(results, q, strict=True)]
