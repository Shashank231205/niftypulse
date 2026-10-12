"""Evaluation metrics, with uncertainty.

Class order everywhere is (down, flat, up) = (-1, 0, +1). Point estimates on noisy,
autocorrelated returns are misleading, so headline numbers come with stationary block
bootstrap confidence intervals.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

CLASSES = np.array([-1, 0, 1])

FloatArray = npt.NDArray[np.float64]


def to_class_index(labels: FloatArray) -> npt.NDArray[np.int64]:
    return (labels.astype(np.int64) + 1).astype(np.int64)


@dataclass(frozen=True, slots=True)
class ClassPrior:
    """Training class frequencies per instrument: the no-skill baseline to beat.

    A single pooled prior would credit a model merely for recognising which instrument a
    row belongs to (BTC has far more "flat" labels than SOL), which is not forecasting.
    """

    by_instrument: dict[str, FloatArray]
    overall: FloatArray

    @classmethod
    def fit(cls, instruments: npt.NDArray[np.object_], labels: FloatArray) -> "ClassPrior":
        y = to_class_index(labels)
        overall = np.bincount(y, minlength=3) / len(y)
        by = {
            str(k): np.bincount(y[instruments == k], minlength=3) / (instruments == k).sum()
            for k in np.unique(instruments)
        }
        return cls(by, overall)

    def rows(self, instruments: npt.NDArray[np.object_]) -> FloatArray:
        return np.array([self.by_instrument.get(str(k), self.overall) for k in instruments])


def expected_move(proba: FloatArray) -> FloatArray:
    """p(up) - p(down): a signed score in [-1, 1] used for ranking and trading."""
    return proba[:, 2] - proba[:, 0]


def expected_calibration_error(
    proba: FloatArray, y_idx: npt.NDArray[np.int64], bins: int = 15
) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(ece)


def information_coefficient(score: FloatArray, realized: FloatArray) -> float:
    """Spearman rank correlation between the signal and the realized forward return."""
    ok = np.isfinite(score) & np.isfinite(realized)
    if ok.sum() < 10:
        return float("nan")
    rho = stats.spearmanr(score[ok], realized[ok]).statistic
    return float(rho)


def directional_hit_rate(
    score: FloatArray, realized: FloatArray, min_abs_score: float
) -> tuple[float, float]:
    """Hit rate when the model takes a side, and the fraction of time it does."""
    active = np.abs(score) >= min_abs_score
    moved = active & (realized != 0)
    if not moved.any():
        return float("nan"), float(active.mean())
    return float((np.sign(score[moved]) == np.sign(realized[moved])).mean()), float(active.mean())


def block_bootstrap_ci(
    values: FloatArray,
    statistic: Callable[[FloatArray], float],
    block: int,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Stationary-style circular block bootstrap CI of `statistic` over row-aligned data."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2 * block:
        return float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        out[b] = statistic(values[idx])
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    n: int
    log_loss: float
    baseline_log_loss: float
    auc_macro: float
    balanced_accuracy: float
    ece: float
    ic: float
    ic_ci: tuple[float, float]
    hit_rate: float
    coverage: float
    class_balance: tuple[float, float, float]

    def as_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "n": self.n,
            "log_loss": self.log_loss,
            "baseline_log_loss": self.baseline_log_loss,
            "log_loss_skill": 1.0 - self.log_loss / self.baseline_log_loss,
            "auc_macro": self.auc_macro,
            "balanced_accuracy": self.balanced_accuracy,
            "ece": self.ece,
            "ic": self.ic,
            "ic_ci": list(self.ic_ci),
            "hit_rate": self.hit_rate,
            "coverage": self.coverage,
            "class_balance": list(self.class_balance),
        }


def classification_report(
    proba: FloatArray,
    labels: FloatArray,
    realized_bps: FloatArray,
    train_prior: FloatArray,
    min_abs_score: float,
    bootstrap_block: int,
) -> ClassificationReport:
    """`train_prior` is the no-skill baseline: one class distribution, or one per row."""
    y = to_class_index(labels)
    proba = np.clip(proba, 1e-7, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    baseline = np.broadcast_to(train_prior, (len(y), 3))
    present = np.unique(y)
    auc = (
        float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        if len(present) == 3
        else float("nan")
    )
    score = expected_move(proba)
    hit, coverage = directional_hit_rate(score, labels, min_abs_score)
    paired = np.column_stack([score, realized_bps])
    ci = block_bootstrap_ci(
        paired, lambda a: information_coefficient(a[:, 0], a[:, 1]), block=bootstrap_block
    )
    counts = np.bincount(y, minlength=3) / len(y)
    return ClassificationReport(
        n=len(y),
        log_loss=float(log_loss(y, proba, labels=[0, 1, 2])),
        baseline_log_loss=float(log_loss(y, baseline, labels=[0, 1, 2])),
        auc_macro=auc,
        balanced_accuracy=float(balanced_accuracy_score(y, proba.argmax(axis=1))),
        ece=expected_calibration_error(proba, y),
        ic=information_coefficient(score, realized_bps),
        ic_ci=ci,
        hit_rate=hit,
        coverage=coverage,
        class_balance=(float(counts[0]), float(counts[1]), float(counts[2])),
    )


def qlike(forecast_var: FloatArray, realized_var: FloatArray) -> float:
    """QLIKE loss (Patton, 2011): robust to noise in the realized-variance proxy."""
    f = np.maximum(forecast_var, 1e-12)
    r = np.maximum(realized_var, 1e-12)
    return float(np.mean(r / f - np.log(r / f) - 1.0))
