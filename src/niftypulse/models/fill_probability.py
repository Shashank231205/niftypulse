"""Passive fill probability and queue depletion.

Question: if we join the best bid (or ask) now, does our order fill within H seconds?

Labels come from replaying the lake. A new order joins the back of the queue, so the
queue ahead starts at the displayed size at that price. It shrinks by
- trades at our price (aggressors hitting the queue), and
- cancellations, inferred as any drop in displayed size not explained by trades
  (L2 data has no order ids; this is the standard approximation).
We fill when trades exhaust the remaining queue ahead, or immediately if price trades
through our level. The share of queue depletion caused by cancellations is recorded as
a second target (cancel dynamics).

A LightGBM classifier with isotonic calibration then predicts fill probability from the
live queue state, so the execution policy can trade off waiting vs crossing the spread.
"""

from dataclasses import dataclass
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from niftypulse.core.config import ModelSettings
from niftypulse.domain.events import BookSnapshot, MarketEvent, Trade
from niftypulse.models.calibration import IsotonicCalibrator
from niftypulse.models.registry import FloatArray

_NS = 1_000_000_000

Side = Literal["buy", "sell"]


@dataclass(slots=True)
class _Pending:
    side: Side
    price: float
    queue_ahead: float
    start_ns: int
    traded: float = 0.0
    cancelled: float = 0.0
    filled_ns: int = 0


def _queue_at(book: BookSnapshot, side: Side, price: float) -> float:
    levels = book.bids if side == "buy" else book.asks
    for lv in levels:
        if lv.price == price:
            return lv.qty
    return 0.0


def label_fills(events: list[MarketEvent], sample_every_s: float, horizon_s: float) -> pd.DataFrame:
    """One row per (sample time, side) for a single instrument's ordered events."""
    rows: list[dict[str, float | int | str]] = []
    pending: list[_Pending] = []
    done: list[_Pending] = []
    next_sample = 0
    for ev in events:
        if isinstance(ev, Trade):
            for p in pending:
                if p.filled_ns:
                    continue
                through = ev.price < p.price if p.side == "buy" else ev.price > p.price
                at_level = ev.price == p.price
                hits_us = (ev.aggressor == "sell") == (p.side == "buy")
                if through and hits_us:
                    p.filled_ns = ev.ts_exchange
                elif at_level and hits_us:
                    take = min(ev.qty, p.queue_ahead)
                    p.queue_ahead -= take
                    p.traded += take
                    if ev.qty > take:
                        p.filled_ns = ev.ts_exchange
            continue
        book = ev
        for p in pending:
            if p.filled_ns:
                continue
            shown = _queue_at(book, p.side, p.price)
            if shown < p.queue_ahead:
                p.cancelled += p.queue_ahead - shown
                p.queue_ahead = shown
        if book.ts_exchange >= next_sample and book.bids and book.asks:
            next_sample = book.ts_exchange + int(sample_every_s * _NS)
            for side, lv in (("buy", book.bids[0]), ("sell", book.asks[0])):
                pending.append(_Pending(side, lv.price, lv.qty, book.ts_exchange))  # type: ignore[arg-type]
        cutoff = book.ts_exchange - int(horizon_s * _NS)
        still: list[_Pending] = []
        for p in pending:
            (done if p.start_ns <= cutoff else still).append(p)
        pending = still
    for p in done:
        filled = bool(p.filled_ns) and p.filled_ns - p.start_ns <= int(horizon_s * _NS)
        depleted = p.traded + p.cancelled
        rows.append(
            {
                "ts": p.start_ns,
                "side": p.side,
                "filled": int(filled),
                "seconds_to_fill": (p.filled_ns - p.start_ns) / _NS if filled else np.nan,
                "cancel_share": p.cancelled / depleted if depleted > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


class FillProbabilityModel:
    """Calibrated P(fill within horizon | queue state, side)."""

    def __init__(self, settings: ModelSettings, features: list[str]) -> None:
        self.settings = settings
        self.features = features
        self._booster: lgb.Booster | None = None
        self._calibrator = IsotonicCalibrator()

    def _x(self, frame: pd.DataFrame) -> FloatArray:
        side = (frame["side"] == "buy").to_numpy(dtype=np.float64)[:, None]
        x = frame[self.features].to_numpy(dtype=np.float64)
        return np.nan_to_num(np.hstack([x, side]))

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, calib: pd.DataFrame) -> None:
        s = self.settings
        names = [*self.features, "side_is_buy"]
        dtrain = lgb.Dataset(self._x(train), train["filled"].to_numpy(), feature_name=names)
        dvalid = lgb.Dataset(self._x(valid), valid["filled"].to_numpy(), reference=dtrain)
        self._booster = lgb.train(
            {
                "objective": "binary",
                "learning_rate": s.gbm_learning_rate,
                "num_leaves": s.gbm_num_leaves,
                "min_data_in_leaf": s.gbm_min_child_samples,
                "lambda_l2": s.gbm_l2,
                "seed": s.seed,
                "verbosity": -1,
                "num_threads": s.n_threads,
            },
            dtrain,
            num_boost_round=s.gbm_max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(s.early_stopping_rounds, verbose=False)],
        )
        self._calibrator.fit(self._raw(calib), calib["filled"].to_numpy(dtype=np.float64))

    def _raw(self, frame: pd.DataFrame) -> FloatArray:
        assert self._booster is not None
        out = self._booster.predict(self._x(frame), num_iteration=self._booster.best_iteration)
        return np.asarray(out, dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        return self._calibrator.transform(self._raw(frame))
