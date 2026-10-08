"""FastAPI application: lifecycle, middleware and health endpoints."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace
from sqlalchemy import text
from starlette.middleware.base import RequestResponseEndpoint

from niftypulse.core import telemetry
from niftypulse.core.bus import EventBus
from niftypulse.core.config import bus_settings, db_settings, gateway_settings
from niftypulse.core.db import Database
from niftypulse.domain.events import AnyEvent
from niftypulse.services.gateway.auth import RateLimiter
from niftypulse.services.gateway.routes import encode_for_clients, router

log = telemetry.logger(__name__)
_tracer = telemetry.tracer(__name__)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = gateway_settings()
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_min)
    app.state.db = Database(db_settings())
    app.state.bus = EventBus(bus_settings(), client_name="gateway")
    app.state.listeners = set()
    await app.state.bus.connect()

    async def fan_out(event: AnyEvent) -> None:
        payload = encode_for_clients(event)
        for queue in list(app.state.listeners):
            if not queue.full():
                queue.put_nowait(payload)

    await app.state.bus.subscribe_live("pred.>", fan_out)
    yield
    await app.state.bus.close()
    await app.state.db.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="NiftyPulse API",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def trace_and_harden(request: Request, call_next: RequestResponseEndpoint) -> Response:
        with _tracer.start_as_current_span(
            f"{request.method} {request.url.path}", kind=trace.SpanKind.SERVER
        ) as span:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            ctx = span.get_span_context()
            if ctx.is_valid:
                response.headers["traceparent"] = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-01"
        response.headers.update(_SECURITY_HEADERS)
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("gateway.unhandled", error=repr(exc))
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        checks = {}
        try:
            async with request.app.state.db.session() as s:
                await asyncio.wait_for(s.execute(text("SELECT 1")), timeout=2.0)
            checks["database"] = "ok"
        except Exception as exc:  # readiness must report, not raise
            checks["database"] = f"error: {type(exc).__name__}"
        checks["bus"] = "ok" if request.app.state.bus.connected else "down"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse(checks, status_code=200 if ok else 503)

    app.include_router(router)
    return app


def main() -> None:
    settings = gateway_settings()
    telemetry.setup("gateway")
    try:
        uvicorn.run(
            create_app(),
            host=settings.host,
            port=settings.port,
            log_config=None,
            server_header=False,
            proxy_headers=False,
        )
    finally:
        telemetry.shutdown()
