"""Repositories: the only code that talks SQL. Services depend on these, not on the ORM."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from niftypulse.storage.schema import (
    ApiPrincipal,
    AuditEntry,
    BacktestRun,
    DriftReading,
    ModelVersion,
    PaperFill,
    PredictionRecord,
)


class ModelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def register(self, model: ModelVersion) -> None:
        self._s.add(model)

    async def get(self, model_id: str) -> ModelVersion | None:
        return await self._s.get(ModelVersion, model_id)

    async def production(self) -> ModelVersion | None:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.stage == "production")
            .order_by(ModelVersion.promoted_at.desc())
            .limit(1)
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list(self, limit: int) -> Sequence[ModelVersion]:
        stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
        return (await self._s.execute(stmt)).scalars().all()

    async def promote(self, model_id: str, actor: str) -> None:
        """Exactly one production model: the previous one is archived in the same transaction."""
        await self._s.execute(
            update(ModelVersion).where(ModelVersion.stage == "production").values(stage="archived")
        )
        await self._s.execute(
            update(ModelVersion)
            .where(ModelVersion.id == model_id)
            .values(stage="production", promoted_by=actor, promoted_at=datetime.now(UTC))
        )


class PredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add_many(self, rows: list[dict[str, Any]]) -> None:
        """Idempotent on event_id: redelivered predictions are ignored."""
        if rows:
            stmt = (
                insert(PredictionRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["event_id"])
            )
            await self._s.execute(stmt)

    async def latest(self, instrument: str | None, limit: int) -> Sequence[PredictionRecord]:
        stmt = select(PredictionRecord).order_by(PredictionRecord.ts_exchange.desc()).limit(limit)
        if instrument:
            stmt = stmt.where(PredictionRecord.instrument == instrument)
        return (await self._s.execute(stmt)).scalars().all()

    async def set_realized(self, event_id: str, label: int, ret_bps: float) -> None:
        await self._s.execute(
            update(PredictionRecord)
            .where(PredictionRecord.event_id == event_id)
            .values(realized_label=label, realized_return=ret_bps)
        )


class FillRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, row: dict[str, Any]) -> None:
        stmt = (
            insert(PaperFill)
            .values(**row)
            .on_conflict_do_nothing(index_elements=["decision_event_id"])
        )
        await self._s.execute(stmt)


class DriftRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, reading: DriftReading) -> None:
        self._s.add(reading)

    async def latest(self, limit: int) -> Sequence[DriftReading]:
        stmt = select(DriftReading).order_by(DriftReading.created_at.desc()).limit(limit)
        return (await self._s.execute(stmt)).scalars().all()


class BacktestRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, run: BacktestRun) -> None:
        self._s.add(run)

    async def list(self, limit: int) -> Sequence[BacktestRun]:
        stmt = select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit)
        return (await self._s.execute(stmt)).scalars().all()


class PrincipalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, principal_id: str) -> ApiPrincipal | None:
        return await self._s.get(ApiPrincipal, principal_id)

    async def upsert(self, principal_id: str, role: str, key_hash: str) -> None:
        stmt = insert(ApiPrincipal).values(
            id=principal_id, role=role, key_hash=key_hash, disabled=False
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"], set_={"role": role, "key_hash": key_hash, "disabled": False}
        )
        await self._s.execute(stmt)


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def record(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        client_ip: str | None,
        trace_id: str | None,
        detail: dict[str, Any],
    ) -> None:
        self._s.add(
            AuditEntry(
                actor=actor,
                action=action,
                resource=resource,
                outcome=outcome,
                client_ip=client_ip,
                trace_id=trace_id,
                detail=detail,
            )
        )

    async def count(self) -> int:
        return int(
            (await self._s.execute(select(func.count()).select_from(AuditEntry))).scalar_one()
        )
