"""Monitor: live model quality, data drift, clock skew and staleness.

- Realized outcomes: each prediction is scored when its horizon passes (hit rate, log
  loss, IC over a rolling window) and its realized label is written back to Postgres.
- Drift: Population Stability Index of live feature values against the training
  deciles stored in the bundle. PSI > threshold raises an alert and is persisted.
- Clock: offset between this machine and the venue clock (NTP-style midpoint), so feed
  latency numbers can be corrected rather than trusted blindly.
- Staleness: an alert if predictions stop arriving.
"""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import httpx
import numpy as np
from prometheus_client import Gauge

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus, subject
from niftypulse.core.config import (
    bus_settings,
    db_settings,
    label_settings,
    monitor_settings,
)
from niftypulse.core.db import Database
from niftypulse.domain.events import (
    Alert,
    AnyEvent,
    BookSnapshot,
    FeatureVector,
    Prediction,
    now_ns,
)
from niftypulse.research.bundle import load_production
from niftypulse.services.runtime import Service
from niftypulse.storage.repositories import DriftRepository, PredictionRepository
from niftypulse.storage.schema import DriftReading
from niftypulse.validation.metrics import information_coefficient

log = telemetry.logger(__name__)

HIT_RATE = Gauge("monitor_hit_rate", "Rolling directional hit rate on moves", ["instrument"])
LOG_LOSS = Gauge("monitor_log_loss", "Rolling log loss", ["instrument"])
IC = Gauge("monitor_ic", "Rolling information coefficient", ["instrument"])
PSI = Gauge("monitor_psi", "Population stability index vs training", ["feature"])
CLOCK_OFFSET = Gauge("monitor_clock_offset_seconds", "Venue clock minus local clock")
PREDICTION_AGE = Gauge("monitor_seconds_since_prediction", "Prediction staleness")

_NS = 1_000_000_000

Severity = Literal["info", "warning", "critical"]


@dataclass
class _Open:
    event_id: str
    due_ns: int
    mid: float
    half_spread_bps: float
    proba: np.ndarray


def psi(reference_edges: list[float], live: np.ndarray) -> float:
    """PSI with the training deciles as bins (each holds ~1/k of training mass)."""
    edges = np.asarray(reference_edges)
    k = len(edges) - 1
    if k < 2 or len(live) == 0:
        return 0.0
    counts = np.histogram(np.clip(live, edges[0], edges[-1]), bins=edges)[0]
    actual = np.maximum(counts / counts.sum(), 1e-4)
    expected = np.full(k, 1.0 / k)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class Monitor(Service):
    name = "monitor"

    def __init__(self) -> None:
        super().__init__()
        self._s = monitor_settings()
        self._labels = label_settings()
        self._bus = EventBus(bus_settings(), client_name=self.name, on_fatal=self.fail)
        self._db = Database(db_settings())
        self._mids: dict[str, float] = {}
        self._half_spread: dict[str, float] = {}
        self._open: dict[str, deque[_Open]] = {}
        self._scored: dict[str, deque[tuple[np.ndarray, int, float]]] = {}
        self._live: deque[tuple[float, ...]] = deque(maxlen=self._s.drift_sample)
        self._last_prediction = time.monotonic()

    async def start(self) -> None:
        self._bundle, _ = await load_production(self._db)
        await self._bus.connect()
        self._bus.subscribe("md.book.>", durable=f"{self.name}-book", handler=self._on_book)
        self._bus.subscribe("pred.>", durable=f"{self.name}-pred", handler=self._on_prediction)
        self._bus.subscribe("feat.>", durable=f"{self.name}-feat", handler=self._on_features)
        self.spawn(self._drift_loop(), "drift")
        self.spawn(self._clock_loop(), "clock")
        self.spawn(self._staleness_loop(), "staleness")

    async def _on_book(self, event: AnyEvent) -> None:
        if not isinstance(event, BookSnapshot) or not event.bids or not event.asks:
            return
        mid = 0.5 * (event.bids[0].price + event.asks[0].price)
        self._mids[event.instrument] = mid
        self._half_spread[event.instrument] = (
            1e4 * 0.5 * (event.asks[0].price - event.bids[0].price) / mid
        )
        await self._score(event.instrument, event.ts_exchange, mid)

    async def _on_prediction(self, event: AnyEvent) -> None:
        if not isinstance(event, Prediction) or event.instrument not in self._mids:
            return
        self._last_prediction = time.monotonic()
        self._open.setdefault(event.instrument, deque()).append(
            _Open(
                event.event_id,
                event.ts_exchange + event.horizon_s * _NS,
                self._mids[event.instrument],
                self._half_spread[event.instrument],
                np.array([event.p_down, event.p_flat, event.p_up]),
            )
        )

    async def _on_features(self, event: AnyEvent) -> None:
        if isinstance(event, FeatureVector):
            self._live.append(event.values[2:])

    async def _score(self, instrument: str, ts: int, mid: float) -> None:
        queue = self._open.get(instrument)
        realized: list[tuple[str, int, float]] = []
        while queue and queue[0].due_ns <= ts:
            o = queue.popleft()
            ret = 1e4 * math.log(mid / o.mid)
            threshold = max(
                self._labels.min_move_bps, self._labels.spread_multiple * o.half_spread_bps
            )
            label = 1 if ret > threshold else -1 if ret < -threshold else 0
            window = self._scored.setdefault(instrument, deque(maxlen=self._s.rolling_window))
            window.append((o.proba, label, ret))
            realized.append((o.event_id, label, ret))
        if not realized:
            return
        self._publish_quality(instrument)
        async with self._db.session() as s:
            repo = PredictionRepository(s)
            for event_id, label, ret in realized:
                await repo.set_realized(event_id, label, ret)

    def _publish_quality(self, instrument: str) -> None:
        window = self._scored[instrument]
        proba = np.array([w[0] for w in window])
        labels = np.array([w[1] for w in window])
        rets = np.array([w[2] for w in window])
        score = proba[:, 2] - proba[:, 0]
        moved = labels != 0
        if moved.any():
            HIT_RATE.labels(instrument).set(float((np.sign(score[moved]) == labels[moved]).mean()))
        LOG_LOSS.labels(instrument).set(
            float(-np.mean(np.log(np.clip(proba[np.arange(len(labels)), labels + 1], 1e-7, 1))))
        )
        ic = information_coefficient(score, rets)
        if math.isfinite(ic):
            IC.labels(instrument).set(ic)

    async def _drift_loop(self) -> None:
        names = self._bundle.base_features
        while True:
            await asyncio.sleep(self._s.drift_interval_s)
            if len(self._live) < 100:
                continue
            live = np.array(self._live)
            breached = []
            async with self._db.session() as s:
                repo = DriftRepository(s)
                for j, name in enumerate(names):
                    edges = self._bundle.reference_bins.get(name)
                    if not edges:
                        continue
                    value = psi(edges, live[:, j])
                    PSI.labels(name).set(value)
                    hit = value > self._s.psi_threshold
                    if hit:
                        breached.append(name)
                    await repo.add(
                        DriftReading(
                            model_version=self._bundle.bundle_id,
                            metric="psi",
                            feature=name,
                            value=value,
                            threshold=self._s.psi_threshold,
                            breached=hit,
                        )
                    )
            if breached:
                await self._alert(
                    "warning", "feature_drift", f"PSI above threshold: {', '.join(breached[:8])}"
                )

    async def _clock_loop(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    t0 = time.time()
                    resp = await client.get(self._s.clock_endpoint)
                    t1 = time.time()
                    server = resp.json()["serverTime"] / 1000
                    CLOCK_OFFSET.set(server - (t0 + t1) / 2)
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    log.warning("monitor.clock_failed", error=repr(exc))
                await asyncio.sleep(self._s.clock_interval_s)

    async def _staleness_loop(self) -> None:
        alerted = False
        while True:
            await asyncio.sleep(self._s.prediction_stale_s / 3)
            age = time.monotonic() - self._last_prediction
            PREDICTION_AGE.set(age)
            if age > self._s.prediction_stale_s and not alerted:
                alerted = True
                await self._alert("critical", "predictions_stale", f"no predictions for {age:.0f}s")
            elif age <= self._s.prediction_stale_s:
                alerted = False

    async def _alert(self, severity: Severity, code: str, message: str) -> None:
        log.warning("monitor.alert", code=code, message=message)
        now = now_ns()
        await self._bus.publish(
            subject("alert.monitor", code),
            Alert(
                instrument="system",
                ts_exchange=now,
                ts_recv=now,
                severity=severity,
                code=code,
                message=message,
            ),
        )

    async def stop(self) -> None:
        await self._bus.close()
        await self._db.close()


def main() -> None:
    Monitor().run()
