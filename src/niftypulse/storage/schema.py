"""Relational schema.

Ticks live in the Parquet lake, not here. Postgres holds what must be queryable,
consistent and auditable: model versions, predictions with full lineage, paper fills,
drift readings, backtest runs, API principals and the audit log.

Lineage: every prediction row records the model version, feature set, code commit,
source tick id and trace id, so any number on a dashboard can be traced back to the
exact model artifact, code and market event that produced it.
"""

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB}


class _Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ModelVersion(_Timestamped, Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    family: Mapped[str] = mapped_column(String(32))
    target: Mapped[str] = mapped_column(String(32))
    horizon_s: Mapped[int] = mapped_column(Integer)
    feature_set: Mapped[str] = mapped_column(String(64))
    git_sha: Mapped[str] = mapped_column(String(40))
    mlflow_run_id: Mapped[str] = mapped_column(String(64))
    artifact_uri: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    train_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    metrics: Mapped[dict[str, Any]]
    stage: Mapped[str] = mapped_column(String(16), default="candidate")
    promoted_by: Mapped[str | None] = mapped_column(String(64))
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PredictionRecord(_Timestamped, Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("event_id"),
        Index("ix_predictions_instrument_ts", "instrument", "ts_exchange"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(32))
    instrument: Mapped[str] = mapped_column(String(64))
    ts_exchange: Mapped[int] = mapped_column(BigInteger)
    horizon_s: Mapped[int] = mapped_column(Integer)
    p_up: Mapped[float] = mapped_column(Float)
    p_flat: Mapped[float] = mapped_column(Float)
    p_down: Mapped[float] = mapped_column(Float)
    vol_forecast: Mapped[float] = mapped_column(Float)
    regime: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    feature_set: Mapped[str] = mapped_column(String(64))
    git_sha: Mapped[str] = mapped_column(String(40))
    source_event_id: Mapped[str] = mapped_column(String(32))
    trace_id: Mapped[str | None] = mapped_column(String(32))
    realized_label: Mapped[int | None] = mapped_column(Integer)
    realized_return: Mapped[float | None] = mapped_column(Float)


class PaperFill(_Timestamped, Base):
    __tablename__ = "paper_fills"
    __table_args__ = (UniqueConstraint("decision_event_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    decision_event_id: Mapped[str] = mapped_column(String(32))
    instrument: Mapped[str] = mapped_column(String(64))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    costs: Mapped[float] = mapped_column(Float)
    costs_currency: Mapped[str] = mapped_column(String(8))
    policy_version: Mapped[str] = mapped_column(String(64))
    trace_id: Mapped[str | None] = mapped_column(String(32))


class DriftReading(_Timestamped, Base):
    __tablename__ = "drift_readings"
    __table_args__ = (Index("ix_drift_model_metric", "model_version", "metric"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    metric: Mapped[str] = mapped_column(String(32))
    feature: Mapped[str | None] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    breached: Mapped[bool]


class BacktestRun(_Timestamped, Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    policy_version: Mapped[str] = mapped_column(String(64))
    git_sha: Mapped[str] = mapped_column(String(40))
    data_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    data_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    config: Mapped[dict[str, Any]]
    results: Mapped[dict[str, Any]]


class ApiPrincipal(_Timestamped, Base):
    __tablename__ = "api_principals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(Text)
    disabled: Mapped[bool] = mapped_column(default=False)


class AuditEntry(Base):
    """Append-only. The application role has INSERT but no UPDATE or DELETE on this table."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_actor_at", "actor", "at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(128))
    outcome: Mapped[str] = mapped_column(String(16))
    client_ip: Mapped[str | None] = mapped_column(String(45))
    trace_id: Mapped[str | None] = mapped_column(String(32))
    detail: Mapped[dict[str, Any]]
