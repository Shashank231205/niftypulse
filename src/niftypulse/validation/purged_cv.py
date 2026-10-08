"""Time-series cross-validation without look-ahead.

Walk-forward: every fold trains strictly on the past and tests on the next block.
Purging: training rows whose label window (t, t + horizon] reaches into the test block
are dropped; otherwise the label itself leaks test-period prices into training.
Embargo: an extra gap after purging, because features computed from rolling windows are
serially correlated across the boundary (Lopez de Prado, 2018, ch. 7).

A final holdout block is split off first and never touched during model selection.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_NS = 1_000_000_000

IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Fold:
    number: int
    train: IndexArray
    test: IndexArray
    test_start_ns: int
    test_end_ns: int


@dataclass(frozen=True, slots=True)
class PurgedWalkForward:
    n_splits: int
    label_horizon_s: float
    embargo_s: float
    min_train_fraction: float = 0.3
    expanding: bool = True

    @property
    def gap_ns(self) -> int:
        return int((self.label_horizon_s + self.embargo_s) * _NS)

    def split(self, ts: npt.NDArray[np.int64]) -> Iterator[Fold]:
        """`ts` is each row's sample time in ns; rows need not be sorted."""
        order = np.argsort(ts, kind="stable")
        sorted_ts = ts[order]
        t0, t1 = int(sorted_ts[0]), int(sorted_ts[-1])
        first_test = t0 + int((t1 - t0) * self.min_train_fraction)
        edges = np.linspace(first_test, t1, self.n_splits + 1).astype(np.int64)
        window = int(edges[0] - t0)
        for k in range(self.n_splits):
            start, end = int(edges[k]), int(edges[k + 1])
            test_mask = (sorted_ts >= start) & (
                sorted_ts <= end if k == self.n_splits - 1 else sorted_ts < end
            )
            train_mask = sorted_ts < start - self.gap_ns
            if not self.expanding:
                train_mask &= sorted_ts >= start - self.gap_ns - window
            if not train_mask.any() or not test_mask.any():
                continue
            yield Fold(k, order[train_mask], order[test_mask], start, end)


def holdout_split(
    ts: npt.NDArray[np.int64], holdout_fraction: float, gap_s: float
) -> tuple[IndexArray, IndexArray]:
    """(development rows, holdout rows): the last `holdout_fraction` of time, after a gap."""
    t0, t1 = int(ts.min()), int(ts.max())
    cut = t1 - int((t1 - t0) * holdout_fraction)
    dev = np.flatnonzero(ts < cut - int(gap_s * _NS))
    hold = np.flatnonzero(ts >= cut)
    return dev.astype(np.int64), hold.astype(np.int64)


def assert_no_overlap(
    ts: npt.NDArray[np.int64], train: IndexArray, test: IndexArray, label_horizon_s: float
) -> None:
    """Fail loudly if any training label window reaches into the test period."""
    if len(train) == 0 or len(test) == 0:
        return
    latest_label_end = int(ts[train].max()) + int(label_horizon_s * _NS)
    if latest_label_end >= int(ts[test].min()):
        raise AssertionError("training labels overlap the test period: look-ahead leak")
