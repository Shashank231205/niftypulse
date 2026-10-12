"""Logs, traces and metrics for every service.

Every log line carries the active trace_id and span_id, so any log entry links
straight to its trace in Jaeger. Trace context travels between services inside NATS
message headers (W3C traceparent), which gives one trace per tick across the whole pipeline.
"""

import logging
from collections.abc import Mapping

import structlog
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from prometheus_client import start_http_server

from niftypulse.core.config import app_settings, telemetry_settings


def _add_trace_ids(
    _: object, __: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def setup(service: str) -> None:
    """Configure logging, tracing and the metrics endpoint for one service process."""
    app = app_settings()
    tel = telemetry_settings()

    logging.basicConfig(level=app.log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_trace_ids,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(app.log_level)),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service, env=app.env)

    provider = TracerProvider(
        resource=Resource.create({"service.name": service, "deployment.environment": app.env}),
        sampler=ParentBased(TraceIdRatioBased(tel.sample_ratio)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{tel.exporter_otlp_endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)

    start_http_server(tel.metrics_port)


def shutdown() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()


def inject_headers() -> dict[str, str]:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract_context(headers: Mapping[str, str] | None) -> Context:
    return propagate.extract(dict(headers or {}))


def logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
