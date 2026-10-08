"""Async PostgreSQL engine and session factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from niftypulse.core.config import DbSettings


def create_engine(settings: DbSettings) -> AsyncEngine:
    return create_async_engine(
        settings.dsn,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_pre_ping=True,
        connect_args={"server_settings": {"statement_timeout": str(settings.statement_timeout_ms)}},
    )


class Database:
    def __init__(self, settings: DbSettings) -> None:
        self.engine = create_engine(settings)
        self._sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """One unit of work: commits on success, rolls back on any error."""
        async with self._sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        await self.engine.dispose()
