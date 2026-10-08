"""Automated leakage checks, run before any result is trusted.

1. Shuffled-label test: retrain with labels permuted inside time blocks. Real skill must
   vanish; if the model still beats the baseline, something leaks (or the CV is broken).
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


def block_shuffle(
    labels: npt.NDArray[np.float64], block: int, seed: int
) -> npt.NDArray[np.float64]:
    """Permute whole blocks of labels: destroys feature-label links, keeps label autocorrelation."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    blocks = [labels[i : i + block] for i in range(0, n, block)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


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
    shuffle_block: int,
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
    shuffled[label_col] = block_shuffle(frame[label_col].to_numpy(), shuffle_block, seed)
    last = folds[-1]
    skill = skill_fn(shuffled.iloc[last.train], shuffled.iloc[last.test], label_col)
    return LeakageReport(
        shuffled_skill=skill,
        shuffled_passed=skill <= max_shuffled_skill,
        suspicious_features=scan_features(frame, feature_names, return_col, max_abs_ic),
        folds_checked=len(folds),
    )
