"""HTTP and WebSocket routes."""

import asyncio
from typing import Annotated, Any

import msgspec
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, WebSocket, status
from fastapi.websockets import WebSocketDisconnect

from niftypulse.core.bus import current_trace_id
from niftypulse.services.gateway.auth import (
    Principal,
    decode_token,
    issue_token,
    require,
    verify_secret,
)
from niftypulse.storage.repositories import (
    AuditRepository,
    BacktestRepository,
    DriftRepository,
    ModelRepository,
    PredictionRepository,
    PrincipalRepository,
)

router = APIRouter(prefix="/v1")

Viewer = Annotated[Principal, Depends(require("viewer"))]
Admin = Annotated[Principal, Depends(require("admin"))]


async def _audit(
    request: Request, actor: str, action: str, resource: str, outcome: str, **detail: Any
) -> None:
    client = request.client.host if request.client else None
    async with request.app.state.db.session() as s:
        await AuditRepository(s).record(
            actor, action, resource, outcome, client, current_trace_id(), detail
        )


@router.post("/auth/token")
async def token(request: Request, x_api_key: Annotated[str, Header()]) -> dict[str, Any]:
    client = request.client.host if request.client else "unknown"
    if not request.app.state.limiter.allow(f"ip:{client}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    principal_id, _, secret = x_api_key.partition(".")
    async with request.app.state.db.session() as s:
        stored = await PrincipalRepository(s).get(principal_id)
    if stored is None or stored.disabled or not verify_secret(secret, stored.key_hash):
        await _audit(request, principal_id or "unknown", "auth.token", "token", "denied")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")
    await _audit(request, stored.id, "auth.token", "token", "ok", role=stored.role)
    settings = request.app.state.settings
    return {
        "access_token": issue_token(Principal(stored.id, stored.role), settings),
        "token_type": "bearer",
        "expires_in": settings.jwt_ttl_s,
    }


@router.get("/predictions/latest")
async def latest_predictions(
    request: Request,
    _: Viewer,
    instrument: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as s:
        rows = await PredictionRepository(s).latest(instrument, limit)
    return [
        {
            "event_id": r.event_id,
            "instrument": r.instrument,
            "ts_exchange": r.ts_exchange,
            "horizon_s": r.horizon_s,
            "p_down": r.p_down,
            "p_flat": r.p_flat,
            "p_up": r.p_up,
            "vol_forecast": r.vol_forecast,
            "regime": r.regime,
            "model_version": r.model_version,
            "feature_set": r.feature_set,
            "git_sha": r.git_sha,
            "trace_id": r.trace_id,
            "realized_label": r.realized_label,
            "realized_return_bps": r.realized_return,
        }
        for r in rows
    ]


@router.get("/models")
async def models(
    request: Request, _: Viewer, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as s:
        rows = await ModelRepository(s).list(limit)
    return [
        {
            "id": m.id,
            "family": m.family,
            "horizon_s": m.horizon_s,
            "feature_set": m.feature_set,
            "stage": m.stage,
            "git_sha": m.git_sha,
            "metrics": m.metrics,
            "created_at": m.created_at.isoformat(),
            "promoted_by": m.promoted_by,
        }
        for m in rows
    ]


@router.post("/models/{model_id}/promote")
async def promote(request: Request, model_id: str, principal: Admin) -> dict[str, str]:
    async with request.app.state.db.session() as s:
        repo = ModelRepository(s)
        if await repo.get(model_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown model")
        await repo.promote(model_id, principal.id)
    await _audit(request, principal.id, "model.promote", model_id, "ok")
    return {"promoted": model_id}


@router.get("/backtests")
async def backtests(
    request: Request, _: Viewer, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as s:
        rows = await BacktestRepository(s).list(limit)
    return [{"id": b.id, "model_version": b.model_version, "results": b.results} for b in rows]


@router.get("/drift")
async def drift(
    request: Request, _: Viewer, limit: Annotated[int, Query(ge=1, le=500)] = 100
) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as s:
        rows = await DriftRepository(s).latest(limit)
    return [
        {
            "feature": d.feature,
            "metric": d.metric,
            "value": d.value,
            "breached": d.breached,
            "at": d.created_at.isoformat(),
        }
        for d in rows
    ]


@router.websocket("/stream/predictions")
async def stream(ws: WebSocket) -> None:
    """Auth via the first message ({"token": ...}): tokens never appear in URLs or logs."""
    await ws.accept()
    try:
        first = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
        decode_token(str(first.get("token", "")), ws.app.state.settings)
    except (TimeoutError, HTTPException, ValueError, WebSocketDisconnect):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1000)
    ws.app.state.listeners.add(queue)
    try:
        while True:
            await ws.send_text((await queue.get()).decode())
    except WebSocketDisconnect:
        pass
    finally:
        ws.app.state.listeners.discard(queue)


def encode_for_clients(event: msgspec.Struct) -> bytes:
    return msgspec.json.encode(event)
