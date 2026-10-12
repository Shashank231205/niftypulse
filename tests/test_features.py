import math

import numpy as np

from niftypulse.domain.events import Level
from niftypulse.features.engine import FeatureEngine, default_counterparts, feature_set_id
from niftypulse.features.labels import direction, forward_realized_vol_bps, forward_returns_bps
from niftypulse.features.microstructure import DecayedSum, MicrostructureState, _ofi_level
from tests.conftest import NS, book, trade


def test_decayed_sum_halves_after_tau_ln2() -> None:
    acc = DecayedSum(tau_s=1.0)
    acc.add(1.0, 0)
    assert math.isclose(acc.decay_to(int(math.log(2) * NS)), 0.5, rel_tol=1e-9)


def test_ofi_cases_follow_cont_kukanov_stoikov() -> None:
    assert _ofi_level(Level(100, 1), Level(100.1, 3), True) == 3  # bid improves
    assert _ofi_level(Level(100, 1), Level(100, 4), True) == 3  # bid size grows
    assert _ofi_level(Level(100, 2), Level(99.9, 5), True) == -2  # bid retreats
    assert _ofi_level(Level(101, 1), Level(100.9, 2), False) == -2  # ask improves (sell pressure)
    assert _ofi_level(Level(101, 3), Level(101, 1), False) == 2  # ask size shrinks


def test_features_match_names_and_respond_to_flow(feature_settings) -> None:  # type: ignore[no-untyped-def]
    s = feature_settings
    st = MicrostructureState(tuple(s.windows_s), tuple(s.rv_windows_s), s.ofi_levels, s.history_s)
    st.on_book(book(0.0, 100.0, 100.1, bid_qty=1, ask_qty=5))
    for i in range(10):
        st.on_trade(trade(0.1 + i * 0.1, 100.1, 1.0, "buy"))
    st.on_book(book(1.0, 100.05, 100.15, bid_qty=5, ask_qty=1))
    values = st.features(int(1.0 * NS))
    names = st.feature_names()
    assert len(values) == len(names) and all(math.isfinite(v) for v in values)
    named = dict(zip(names, values, strict=True))
    assert named["trade_imbalance_1s"] > 0.99
    assert named["imbalance_l1"] > 0
    assert named["mid_return_bps_1s"] > 0


def test_engine_cross_features_and_feature_set_id(feature_settings) -> None:  # type: ignore[no-untyped-def]
    keys = ["X|BTC", "Y|BTC"]
    assert default_counterparts(keys) == {"X|BTC": "Y|BTC", "Y|BTC": "X|BTC"}
    engine = FeatureEngine(feature_settings, keys)
    engine.on_event(book(0.0, 100.0, 100.1, key="X|BTC"))
    row = engine.on_event(book(0.5, 100.2, 100.3, key="Y|BTC"))
    assert row is not None and len(row.values) == len(engine.names)
    basis = dict(zip(engine.names, row.values, strict=True))["basis_bps"]
    assert basis > 0
    assert feature_set_id(feature_settings).startswith("micro-")


def test_labels_are_forward_looking_and_cost_aware(label_settings) -> None:  # type: ignore[no-untyped-def]
    ts = np.arange(10, dtype=np.int64) * NS
    mid = np.array([100, 100, 100, 100.5, 100.5, 100.5, 99, 99, 99, 99], dtype=float)
    ret = forward_returns_bps(ts, mid, np.array([2 * NS, 5 * NS]), 1.0)
    assert np.allclose(ret, [1e4 * np.log(100.5 / 100), 1e4 * np.log(99 / 100.5)])
    lab = direction(ret, np.array([1.0, 1.0]), label_settings)
    assert lab.tolist() == [1.0, -1.0]
    tiny = direction(np.array([0.3]), np.array([1.0]), label_settings)
    assert tiny.tolist() == [0.0]  # smaller than the half-spread: flat
    rv = forward_realized_vol_bps(ts, mid, np.array([0]), 3.0)
    assert rv[0] > 0
    assert np.isnan(forward_returns_bps(ts, mid, np.array([9 * NS]), 5.0))[0]


def test_labels_never_span_a_feed_gap() -> None:
    secs = np.concatenate([np.arange(0, 21), np.arange(600, 621)])  # outage from 20s to 600s
    ts = secs.astype(np.int64) * NS
    mid = np.where(secs < 600, 100.0, 110.0)
    sample = np.array([0, 18, 600], dtype=np.int64) * NS
    ret = forward_returns_bps(ts, mid, sample, 10.0, max_staleness_s=5.0)
    assert ret[0] == 0.0 and ret[2] == 0.0
    assert np.isnan(ret[1])  # the first price after 28s arrives at 600s
    rv = forward_realized_vol_bps(ts, mid, sample, 10.0, max_staleness_s=5.0)
    assert rv[0] == 0.0 and rv[2] == 0.0
    assert np.isnan(rv[1])
