import numpy as np
import pandas as pd
import pytest

from niftypulse.models.calibration import BiasedTemperature
from niftypulse.validation.leakage import scan_features, shift_labels
from niftypulse.validation.metrics import ClassPrior, classification_report, qlike
from niftypulse.validation.purged_cv import PurgedWalkForward, assert_no_overlap, holdout_split

NS = 1_000_000_000


def test_walk_forward_is_purged_and_forward_only() -> None:
    ts = np.arange(10_000, dtype=np.int64) * NS
    cv = PurgedWalkForward(n_splits=4, label_horizon_s=30, embargo_s=60)
    folds = list(cv.split(ts))
    assert len(folds) == 4
    for f in folds:
        assert ts[f.train].max() + cv.gap_ns < ts[f.test].min()
        assert_no_overlap(ts, f.train, f.test, 30)


def test_overlap_check_catches_leak() -> None:
    ts = np.arange(100, dtype=np.int64) * NS
    with pytest.raises(AssertionError):
        assert_no_overlap(ts, np.arange(0, 50), np.arange(50, 100), label_horizon_s=10)


def test_holdout_is_last_block_behind_gap() -> None:
    ts = np.arange(1000, dtype=np.int64) * NS
    dev, hold = holdout_split(ts, 0.2, gap_s=10)
    assert ts[dev].max() + 10 * NS < ts[hold].min()
    assert len(hold) == pytest.approx(200, abs=2)


def test_shift_labels_rotates_each_instrument_far_enough() -> None:
    inst = np.repeat(np.array(["a", "b"], dtype=object), 1000)
    y = np.tile(np.arange(1000.0), 2)
    s = shift_labels(inst, y, 0.25, seed=1)
    for key in ("a", "b"):
        m = inst == key
        assert sorted(s[m]) == sorted(y[m])
        offset = int((y[m][0] - s[m][0]) % 1000)
        assert 250 <= offset <= 750


def test_feature_scan_flags_a_leaked_future_value() -> None:
    rng = np.random.default_rng(0)
    ret = rng.normal(size=2000)
    frame = pd.DataFrame(
        {"honest": rng.normal(size=2000), "leaked": ret + 0.01 * rng.normal(size=2000), "ret": ret}
    )
    flagged = scan_features(frame, ["honest", "leaked"], "ret", max_abs_ic=0.5)
    assert list(flagged) == ["leaked"]


def test_calibration_fixes_overconfidence() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 5000)
    logits = rng.normal(size=(5000, 3))
    logits[np.arange(5000), y] += 1.0
    sharp = np.exp(4 * logits) / np.exp(4 * logits).sum(axis=1, keepdims=True)  # overconfident
    calib = BiasedTemperature().fit(sharp, y)
    nll = lambda p: -np.mean(np.log(p[np.arange(len(y)), y]))  # noqa: E731
    assert calib.temperature > 1.5
    assert nll(calib.transform(sharp)) < nll(sharp)


def test_report_has_skill_against_prior_for_informative_model() -> None:
    rng = np.random.default_rng(5)
    labels = rng.choice([-1.0, 0.0, 1.0], size=3000, p=[0.25, 0.5, 0.25])
    proba = np.full((3000, 3), 0.1)
    proba[np.arange(3000), (labels + 1).astype(int)] = 0.8
    rep = classification_report(proba, labels, labels * 2.0, np.array([0.25, 0.5, 0.25]), 0.1, 100)
    assert rep.as_dict()["log_loss_skill"] > 0.3 and rep.ic > 0.5


def test_qlike_is_zero_for_perfect_forecast() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert qlike(v, v) == pytest.approx(0.0) and qlike(v * 2, v) > 0


def test_class_prior_is_per_instrument_with_pooled_fallback() -> None:
    inst = np.array(["btc"] * 4 + ["sol"] * 4, dtype=object)
    labels = np.array([0, 0, 0, 1, -1, 1, -1, 1], dtype=np.float64)
    prior = ClassPrior.fit(inst, labels)
    np.testing.assert_allclose(prior.by_instrument["btc"], [0, 0.75, 0.25])
    np.testing.assert_allclose(prior.by_instrument["sol"], [0.5, 0, 0.5])
    rows = prior.rows(np.array(["sol", "eth"], dtype=object))
    np.testing.assert_allclose(rows, [[0.5, 0, 0.5], prior.overall])
