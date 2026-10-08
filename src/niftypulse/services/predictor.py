"""Predictor: feature vectors -> regime -> bandit-selected, calibrated predictions.

Mirrors research exactly: rows are sampled on the same time grid, regime features come
from the shared `OnlineRegimeState`, and every arm in the production bundle scores the
row. The Thompson bandit picks which arm to publish for the current regime and learns
from realized outcomes once the horizon has passed. Each prediction is persisted with
its lineage: bundle id, feature set, code commit, source event and trace id.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus, current_trace_id, subject
from niftypulse.core.config import (
    bus_settings,
    db_settings,
    feature_settings,
    label_settings,
    model_settings,
    policy_settings,
    regime_settings,
)
from niftypulse.core.db import Database
from niftypulse.domain.events import AnyEvent, FeatureVector, Prediction
from niftypulse.features.engine import feature_set_id
from niftypulse.models.registry import ModelInput
from niftypulse.policy.bandit import ThompsonSelector, log_score_gain
from niftypulse.regime.features import RET_COL, SPREAD_COL, STEP_RET_COL, VOL_COL, OnlineRegimeState
from niftypulse.research.bundle import ModelBundle, load_production
from niftypulse.services.runtime import Service
from niftypulse.storage.repositories import PredictionRepository

log = telemetry.logger(__name__)

PREDICTIONS = Counter("predictor_predictions_total", "Published predictions", ["instrument", "arm"])
INFERENCE_SECONDS = Histogram(
    "predictor_inference_seconds",
    "All-arm inference per row",
    buckets=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5),
)

_NS = 1_000_000_000


@dataclass
class _Pending:
    due_ns: int
    mid: float
    half_spread_bps: float
    regime: int
    probas: dict[str, np.ndarray]


@dataclass
class _InstrumentState:
    regime: OnlineRegimeState
    rows: deque[dict[str, float]]
    next_grid_ns: int = 0
    pending: deque[_Pending] = field(default_factory=deque)


class Predictor(Service):
    name = "predictor"

    def __init__(self) -> None:
        super().__init__()
        self._bus = EventBus(bus_settings(), client_name=self.name, on_fatal=self.fail)
        self._db = Database(db_settings())
        self._step_ns = label_settings().sample_interval_ms * 1_000_000
        self._labels = label_settings()
        self._states: dict[str, _InstrumentState] = {}
        self._buffer: list[dict[str, object]] = []
        self._last_flush = time.monotonic()

    async def start(self) -> None:
        self._bundle, self._git_sha = await load_production(self._db)
        live_set = feature_set_id(feature_settings())
        if live_set != self._bundle.feature_set:
            raise RuntimeError(
                f"feature set mismatch: live {live_set}, bundle {self._bundle.feature_set}; "
                "retrain or fix config"
            )
        p = policy_settings()
        self._bandit = ThompsonSelector(
            list(self._bundle.arms),
            regime_settings().hmm_states,
            p.bandit_prior_mean,
            p.bandit_prior_var,
            p.bandit_noise_var,
            p.seed,
        )
        self._history_len = model_settings().seq_len * 2
        log.info(
            "predictor.bundle_loaded", bundle=self._bundle.bundle_id, arms=list(self._bundle.arms)
        )
        await self._bus.connect()
        self._bus.subscribe("feat.>", durable=self.name, handler=self._on_event)

    def _state(self, instrument: str) -> _InstrumentState:
        if instrument not in self._states:
            self._states[instrument] = _InstrumentState(
                OnlineRegimeState(self._bundle.hmm, regime_settings()),
                deque(maxlen=self._history_len),
            )
        return self._states[instrument]

    async def _on_event(self, event: AnyEvent) -> None:
        if not isinstance(event, FeatureVector):
            return
        b: ModelBundle = self._bundle
        mid, spread, *values = event.values
        st = self._state(event.instrument)
        self._settle(st, event.ts_exchange, mid)
        if event.ts_exchange < st.next_grid_ns:
            return
        st.next_grid_ns = event.ts_exchange - event.ts_exchange % self._step_ns + self._step_ns
        row = dict(zip(b.base_features, values, strict=True))
        regime_vals = st.regime.update(
            event.ts_exchange, mid, row[STEP_RET_COL], row[VOL_COL], row[RET_COL], row[SPREAD_COL]
        )
        row.update(zip(b.features[len(b.base_features) :], regime_vals, strict=True))
        row.update({"ts": float(event.ts_exchange), "mid": mid, "spread": spread})
        st.rows.append(row)

        frame = pd.DataFrame(list(st.rows))
        frame["instrument"] = event.instrument
        frame["ts"] = frame["ts"].astype(np.int64)
        with INFERENCE_SECONDS.time():
            data = ModelInput(frame, b.features)
            probas = {name: arm.predict_proba(data)[-1] for name, arm in b.arms.items()}
        regime = int(regime_vals[-1])
        arm = self._bandit.choose(regime)
        proba = probas[arm]
        vol = self._vol_forecast(frame, b)
        prediction = Prediction(
            instrument=event.instrument,
            ts_exchange=event.ts_exchange,
            ts_recv=event.ts_recv,
            horizon_s=int(b.horizon_s),
            p_down=float(proba[0]),
            p_flat=float(proba[1]),
            p_up=float(proba[2]),
            vol_forecast=vol,
            regime=regime,
            model_version=f"{b.bundle_id}/{arm}",
            feature_set=b.feature_set,
            source_event_id=event.event_id,
        )
        await self._bus.publish(subject("pred", event.instrument), prediction)
        PREDICTIONS.labels(event.instrument, arm).inc()
        half_spread_bps = 1e4 * 0.5 * spread / mid
        st.pending.append(
            _Pending(
                event.ts_exchange + int(b.horizon_s * _NS), mid, half_spread_bps, regime, probas
            )
        )
        self._buffer.append(
            {
                "event_id": prediction.event_id,
                "instrument": event.instrument,
                "ts_exchange": event.ts_exchange,
                "horizon_s": int(b.horizon_s),
                "p_up": prediction.p_up,
                "p_flat": prediction.p_flat,
                "p_down": prediction.p_down,
                "vol_forecast": vol,
                "regime": regime,
                "model_version": b.bundle_id,
                "feature_set": b.feature_set,
                "git_sha": self._git_sha,
                "source_event_id": event.event_id,
                "trace_id": current_trace_id(),
            }
        )
        if len(self._buffer) >= 200 or time.monotonic() - self._last_flush > 2.0:
            await self._flush()

    def _vol_forecast(self, frame: pd.DataFrame, b: ModelBundle) -> float:
        if b.vol_model is None:
            return float("nan")
        return float(b.vol_model.predict_vol(ModelInput(frame.tail(1), b.base_features))[0])

    def _settle(self, st: _InstrumentState, ts: int, mid: float) -> None:
        """Feed realized outcomes back to the bandit once each prediction's horizon passes."""
        while st.pending and st.pending[0].due_ns <= ts:
            p = st.pending.popleft()
            ret = 1e4 * math.log(mid / p.mid)
            threshold = max(
                self._labels.min_move_bps, self._labels.spread_multiple * p.half_spread_bps
            )
            y = 2 if ret > threshold else 0 if ret < -threshold else 1
            for arm, proba in p.probas.items():
                self._bandit.observe(
                    p.regime, arm, log_score_gain(proba, self._bundle.class_prior, y)
                )

    async def _flush(self) -> None:
        rows, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        async with self._db.session() as s:
            await PredictionRepository(s).add_many(rows)

    async def stop(self) -> None:
        if self._buffer:
            await self._flush()
        await self._bus.close()
        await self._db.close()
        log.info(
            "predictor.bandit_state",
            state=self._bandit.snapshot() if hasattr(self, "_bandit") else {},
        )


def main() -> None:
    Predictor().run()
