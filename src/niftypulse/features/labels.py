"""Forward-looking targets, computed only from data strictly after the sample time.

- direction: -1 / 0 / +1 if the mid moves down / not enough / up over the horizon. A move
  only counts if it beats max(min_move_bps, spread_multiple x half-spread), i.e. it would
  have paid for crossing the spread.
- return: forward mid log-return in basis points (regression target).
- realized volatility: sqrt of the sum of squared mid log-returns over the next window.
"""

import numpy as np
import numpy.typing as npt

from niftypulse.core.config import LabelSettings

_NS = 1_000_000_000
_BPS = 1e4

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _no_gap(
    ts: IntArray, start: IntArray, end: IntArray, max_staleness_s: float
) -> npt.NDArray[np.bool_]:
    """True where no two consecutive events between start and end are further apart than
    `max_staleness_s`: the feed was live for the whole label window."""
    gaps = np.concatenate([[0], np.cumsum(np.diff(ts) > max_staleness_s * _NS)])
    return np.asarray(gaps[end] == gaps[start])


def forward_returns_bps(
    ts: IntArray,
    mid: FloatArray,
    sample_ts: IntArray,
    horizon_s: float,
    max_staleness_s: float = np.inf,
) -> FloatArray:
    """Return from the last mid at or before t to the first mid at or after t + h.

    If the feed went quiet for more than `max_staleness_s` anywhere in that window (an
    outage), the label is NaN rather than a return measured across the gap.
    """
    now_idx = np.searchsorted(ts, sample_ts, side="right") - 1
    target = sample_ts + int(horizon_s * _NS)
    fwd_idx = np.searchsorted(ts, target, side="left")
    out = np.full(len(sample_ts), np.nan)
    valid = (now_idx >= 0) & (fwd_idx < len(ts))
    valid[valid] &= _no_gap(ts, now_idx[valid], fwd_idx[valid], max_staleness_s)
    out[valid] = _BPS * np.log(mid[fwd_idx[valid]] / mid[now_idx[valid]])
    return out


def direction(ret_bps: FloatArray, half_spread_bps: FloatArray, s: LabelSettings) -> FloatArray:
    threshold = np.maximum(s.min_move_bps, s.spread_multiple * half_spread_bps)
    out = np.where(ret_bps > threshold, 1.0, np.where(ret_bps < -threshold, -1.0, 0.0))
    out[np.isnan(ret_bps)] = np.nan
    return out


def forward_realized_vol_bps(
    ts: IntArray,
    mid: FloatArray,
    sample_ts: IntArray,
    horizon_s: float,
    max_staleness_s: float = np.inf,
) -> FloatArray:
    r2 = np.concatenate([[0.0], np.diff(np.log(mid)) ** 2])
    cum = np.cumsum(r2)
    start = np.searchsorted(ts, sample_ts, side="right") - 1
    end = np.searchsorted(ts, sample_ts + int(horizon_s * _NS), side="right") - 1
    out = np.full(len(sample_ts), np.nan)
    valid = (start >= 0) & (sample_ts + int(horizon_s * _NS) <= ts[-1])
    target = sample_ts + int(horizon_s * _NS)
    valid[valid] &= _no_gap(ts, start[valid], end[valid], max_staleness_s) & (
        target[valid] - ts[end[valid]] <= max_staleness_s * _NS
    )
    out[valid] = _BPS * np.sqrt(cum[end[valid]] - cum[start[valid]])
    return out
