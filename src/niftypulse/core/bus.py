"""NATS JetStream event bus.

Delivery guarantees:
- Publishes are de-duplicated by event_id (Nats-Msg-Id), so retries are safe.
- Consumers are durable pull consumers with explicit ack: at-least-once delivery.
- A handler failure is retried with backoff up to `max_deliver` times, then the message
  is copied to the dead-letter stream and acked, so one poison message cannot block a
  partition.
- Subjects are partitioned by instrument (`md.book.<instrument>`), which keeps per-
  instrument ordering while letting consumers in a queue group scale out.
"""

import asyncio
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import msgspec
import nats
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig, DiscardPolicy, RetentionPolicy, StreamConfig
from nats.js.errors import NotFoundError
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client import Counter, Histogram

from niftypulse.core import telemetry
from niftypulse.core.config import BusSettings
from niftypulse.domain.events import AnyEvent, Event, decode, encode, now_ns

log = telemetry.logger(__name__)
_tracer = telemetry.tracer(__name__)

PUBLISHED = Counter("bus_published_total", "Events published", ["subject_root"])
CONSUMED = Counter("bus_consumed_total", "Events consumed", ["consumer", "outcome"])
HANDLE_SECONDS = Histogram(
    "bus_handle_seconds",
    "Handler latency",
    ["consumer"],
    buckets=(1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0),
)
PIPELINE_LAG = Histogram(
    "bus_pipeline_lag_seconds",
    "Time from first receipt of the source tick to handling",
    ["consumer"],
    buckets=(1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0),
)


@dataclass(frozen=True, slots=True)
class StreamSpec:
    name: str
    subjects: tuple[str, ...]
    max_age_s: int
    max_bytes: int


def stream_specs(s: BusSettings) -> tuple[StreamSpec, ...]:
    """Stream topology is code; retention limits are settings."""
    return (
        StreamSpec("MARKETDATA", ("md.>",), s.marketdata_max_age_s, s.marketdata_max_bytes),
        StreamSpec("FEATURES", ("feat.>",), s.features_max_age_s, s.features_max_bytes),
        StreamSpec("SIGNALS", ("pred.>", "decision.>"), s.signals_max_age_s, s.signals_max_bytes),
        StreamSpec("ALERTS", ("alert.>",), s.alerts_max_age_s, s.alerts_max_bytes),
        StreamSpec("DEADLETTER", ("dlq.>",), s.deadletter_max_age_s, s.deadletter_max_bytes),
    )


Handler = Callable[[AnyEvent], Awaitable[None]]


class BufferedSink(Protocol):
    def add(self, event: AnyEvent) -> None: ...

    def should_flush(self) -> bool: ...

    async def flush(self) -> None: ...

    def discard(self) -> None: ...


def stream_for(subj: str, specs: tuple[StreamSpec, ...]) -> str:
    """Resolve the stream that captures a subject without asking the server.

    Consumers then need no permission on $JS.API.STREAM.NAMES (least privilege).
    """
    root = subj.split(".", 1)[0]
    for spec in specs:
        if any(pattern.split(".", 1)[0] == root for pattern in spec.subjects):
            return spec.name
    raise LookupError(f"no stream captures subject {subj!r}")


def subject(root: str, instrument: str) -> str:
    """NATS subject tokens cannot contain '.', '|' or spaces, which NSE keys do."""
    token = instrument.replace("|", "_").replace(".", "_").replace(" ", "_")
    return f"{root}.{token}"


class EventBus:
    def __init__(
        self,
        settings: BusSettings,
        client_name: str,
        on_fatal: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._settings = settings
        self._name = client_name
        self._on_fatal = on_fatal
        self._specs = stream_specs(settings)
        self._nc: Client | None = None
        self._js: JetStreamContext | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()

    @property
    def js(self) -> JetStreamContext:
        if self._js is None:
            raise RuntimeError("EventBus.connect() has not been called")
        return self._js

    async def connect(self) -> None:
        tls: ssl.SSLContext | None = None
        if self._settings.tls_ca_file:
            tls = ssl.create_default_context(cafile=self._settings.tls_ca_file)
        self._nc = await nats.connect(
            servers=[self._settings.url],
            name=self._name,
            user=self._settings.user,
            password=self._settings.password.get_secret_value(),
            tls=tls,
            connect_timeout=self._settings.connect_timeout_s,
            max_reconnect_attempts=self._settings.max_reconnect_attempts,
            disconnected_cb=self._on_disconnect,
            reconnected_cb=self._on_reconnect,
            error_cb=self._on_error,
        )
        self._js = self._nc.jetstream()
        log.info("bus.connected", url=self._settings.url)

    async def ensure_streams(self) -> None:
        """Create or update streams. Run once by the operator CLI, not by every service."""
        for spec in self._specs:
            config = StreamConfig(
                name=spec.name,
                subjects=list(spec.subjects),
                retention=RetentionPolicy.LIMITS,
                discard=DiscardPolicy.OLD,
                max_age=spec.max_age_s,
                max_bytes=spec.max_bytes,
                duplicate_window=self._settings.duplicate_window_s,
            )
            try:
                await self.js.stream_info(spec.name)
                await self.js.update_stream(config)
            except NotFoundError:
                await self.js.add_stream(config)
            log.info("bus.stream_ready", stream=spec.name)

    async def publish(self, subj: str, event: Event) -> None:
        with _tracer.start_as_current_span(f"publish {subj}", kind=SpanKind.PRODUCER) as span:
            span.set_attribute("messaging.message.id", event.event_id)
            headers = telemetry.inject_headers()
            headers["Nats-Msg-Id"] = event.event_id
            await self.js.publish(subj, encode(event), headers=headers)
        PUBLISHED.labels(subj.split(".", 1)[0]).inc()

    def subscribe(self, subj: str, durable: str, handler: Handler, *, batch: int = 256) -> None:
        """Start a durable pull consumer in the background."""
        self._watch(asyncio.create_task(self._consume(subj, durable, handler, batch)))

    def subscribe_buffered(
        self, subj: str, durable: str, sink: BufferedSink, *, ack_wait_s: float, max_pending: int
    ) -> None:
        """Consumer for sinks that write in batches (e.g. Parquet).

        Messages are acked only after `sink.flush()` has durably written them, so a crash
        between receipt and write causes redelivery, never loss. Redelivered duplicates are
        removed downstream by event_id.
        """
        self._watch(
            asyncio.create_task(
                self._consume_buffered(subj, durable, sink, ack_wait_s, max_pending)
            )
        )

    def _watch(self, task: asyncio.Task[None]) -> None:
        """A consumer loop that dies is fatal: report it instead of failing silently."""
        self._tasks.append(task)

        def _done(t: asyncio.Task[None]) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is None:
                return
            log.error("bus.consumer_crashed", error=repr(exc))
            if self._on_fatal is not None:
                self._on_fatal(exc)

        task.add_done_callback(_done)

    async def _consume_buffered(
        self, subj: str, durable: str, sink: BufferedSink, ack_wait_s: float, max_pending: int
    ) -> None:
        sub = await self.js.pull_subscribe(
            subj,
            durable=durable,
            stream=stream_for(subj, self._specs),
            config=ConsumerConfig(
                ack_wait=ack_wait_s,
                max_deliver=self._settings.max_deliver,
                max_ack_pending=max_pending,
            ),
        )
        pending: list[Msg] = []
        while True:
            stopping = self._stopping.is_set()
            if not stopping:
                try:
                    for msg in await sub.fetch(min(1024, max_pending), timeout=1.0):
                        try:
                            sink.add(decode(msg.data))
                        except msgspec.DecodeError as exc:
                            await self._on_handler_failure(msg, durable, exc, final=True)
                            continue
                        pending.append(msg)
                except TimeoutError:
                    pass
            if pending and (stopping or sink.should_flush()):
                await self._flush_sink(sink, pending, durable)
                pending = []
            if stopping:
                return

    async def _flush_sink(self, sink: BufferedSink, pending: list[Msg], durable: str) -> None:
        with _tracer.start_as_current_span(f"flush {durable}") as span:
            span.set_attribute("messaging.batch.message_count", len(pending))
            try:
                await sink.flush()
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                sink.discard()
                log.error(
                    "bus.flush_failed", consumer=durable, messages=len(pending), error=repr(exc)
                )
                for msg in pending:
                    await msg.nak(delay=self._settings.retry_max_delay_s)
                CONSUMED.labels(durable, "retried").inc(len(pending))
                return
        for msg in pending:
            await msg.ack()
        CONSUMED.labels(durable, "ok").inc(len(pending))

    async def _consume(self, subj: str, durable: str, handler: Handler, batch: int) -> None:
        sub = await self.js.pull_subscribe(
            subj,
            durable=durable,
            stream=stream_for(subj, self._specs),
            config=ConsumerConfig(
                ack_wait=self._settings.ack_wait_s,
                max_deliver=self._settings.max_deliver,
                max_ack_pending=self._settings.max_inflight,
            ),
        )
        while not self._stopping.is_set():
            try:
                msgs = await sub.fetch(batch, timeout=1.0)
            except TimeoutError:
                continue
            for msg in msgs:
                await self._dispatch(msg, durable, handler)

    async def _dispatch(self, msg: Msg, durable: str, handler: Handler) -> None:
        ctx = telemetry.extract_context(msg.headers)
        with (
            _tracer.start_as_current_span(
                f"consume {msg.subject}", context=ctx, kind=SpanKind.CONSUMER
            ) as span,
            HANDLE_SECONDS.labels(durable).time(),
        ):
            try:
                event = decode(msg.data)
                PIPELINE_LAG.labels(durable).observe((now_ns() - event.ts_recv) / 1e9)
                await handler(event)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                await self._on_handler_failure(msg, durable, exc)
                return
            await msg.ack()
            CONSUMED.labels(durable, "ok").inc()

    async def _on_handler_failure(
        self, msg: Msg, durable: str, exc: Exception, *, final: bool = False
    ) -> None:
        attempt = msg.metadata.num_delivered
        if final or attempt >= self._settings.max_deliver:
            headers = dict(msg.headers or {})
            headers.update({"x-error": repr(exc)[:500], "x-consumer": durable})
            await self.js.publish(f"dlq.{msg.subject}", msg.data, headers=headers)
            await msg.ack()
            CONSUMED.labels(durable, "dead_lettered").inc()
            log.error("bus.dead_lettered", subject=msg.subject, consumer=durable, error=repr(exc))
            return
        delay = min(
            2.0**attempt * self._settings.retry_base_delay_s, self._settings.retry_max_delay_s
        )
        await msg.nak(delay=delay)
        CONSUMED.labels(durable, "retried").inc()
        log.warning("bus.retry", subject=msg.subject, attempt=attempt, delay_s=delay)

    async def close(self) -> None:
        self._stopping.set()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._nc is not None and not self._nc.is_closed:
            await self._nc.drain()
        log.info("bus.closed")

    async def _on_disconnect(self) -> None:
        log.warning("bus.disconnected")

    async def _on_reconnect(self) -> None:
        log.info("bus.reconnected")

    async def _on_error(self, exc: Exception) -> None:
        log.error("bus.error", error=repr(exc))


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
