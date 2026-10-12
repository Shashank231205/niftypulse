"""Automated leakage checks, run before any result is trusted.

1. Shifted-label test: retrain with each instrument's labels rotated in time by a large
   random offset. That keeps the labels' own autocorrelation and class mix but breaks every
   link to the features, including slow ones such as volatility regimes (a short block
   shuffle keeps those, because a block keeps its local share of "flat" labels). Real skill
   must vanish; if the model still beats the baseline, something leaks.
2. Suspicious-feature scan: no legitimate feature should be strongly rank-correlated with
   the forward return at these horizons. A large |IC| almost always means a future value
   slipped into the feature.
3. Split overlap: every fold is checked for label windows reaching into its test block.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from niftypulse.validation.metrics import information_coefficient
from niftypulse.validation.purged_cv import PurgedWalkForward, assert_no_overlap


@dataclass(frozen=True, slots=True)
class LeakageReport:
    shuffled_skill: float
    shuffled_passed: bool
    suspicious_features: dict[str, float]
    folds_checked: int

    @property
    def passed(self) -> bool:
        return self.shuffled_passed and not self.suspicious_features

    def as_dict(self) -> dict[str, object]:
        return {
            "shuffled_skill": self.shuffled_skill,
            "shuffled_passed": self.shuffled_passed,
            "suspicious_features": self.suspicious_features,
            "folds_checked": self.folds_checked,
            "passed": self.passed,
        }


def shift_labels(
    instruments: npt.NDArray[np.object_],
    labels: npt.NDArray[np.float64],
    min_fraction: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Rotate each instrument's label series by a random offset of at least `min_fraction`."""
    rng = np.random.default_rng(seed)
    out = labels.copy()
    for key in np.unique(instruments):
        idx = np.flatnonzero(instruments == key)
        n = len(idx)
        lo = max(1, int(n * min_fraction))
        if n - lo <= lo:
            continue
        out[idx] = np.roll(labels[idx], int(rng.integers(lo, n - lo)))
    return out


def scan_features(
    frame: pd.DataFrame, feature_names: list[str], forward_return_col: str, max_abs_ic: float
) -> dict[str, float]:
    realized = frame[forward_return_col].to_numpy()
    flagged = {}
    for name in feature_names:
        ic = information_coefficient(frame[name].to_numpy(dtype=np.float64), realized)
        if np.isfinite(ic) and abs(ic) > max_abs_ic:
            flagged[name] = ic
    return flagged


def run_leakage_checks(
    frame: pd.DataFrame,
    feature_names: list[str],
    label_col: str,
    return_col: str,
    cv: PurgedWalkForward,
    skill_fn: Callable[[pd.DataFrame, pd.DataFrame, str], float],
    min_shift_fraction: float,
    max_abs_ic: float,
    max_shuffled_skill: float,
    seed: int = 11,
) -> LeakageReport:
    """`skill_fn(train, test, label_col)` trains a model and returns log-loss skill on test."""
    ts = frame["ts"].to_numpy()
    folds = list(cv.split(ts))
    for fold in folds:
        assert_no_overlap(ts, fold.train, fold.test, cv.label_horizon_s)

    shuffled = frame.copy()
    shuffled[label_col] = shift_labels(
        frame["instrument"].to_numpy(), frame[label_col].to_numpy(), min_shift_fraction, seed
    )
    last = folds[-1]
    skill = skill_fn(shuffled.iloc[last.train], shuffled.iloc[last.test], label_col)
    return LeakageReport(
        shuffled_skill=skill,
        shuffled_passed=skill <= max_shuffled_skill,
        suspicious_features=scan_features(frame, feature_names, return_col, max_abs_ic),
        folds_checked=len(folds),
    )
