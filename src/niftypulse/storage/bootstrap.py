"""Create the database and its two roles. Idempotent; safe to re-run.

- owner role: owns the schema, used only by migrations.
- app role:   data access only, no DDL; granted by the migrations themselves.
"""

import asyncpg

from niftypulse.core import telemetry
from niftypulse.core.config import MigrationSettings, PostgresAdminSettings

log = telemetry.logger(__name__)


async def _ensure_role(conn: asyncpg.Connection, name: str, password: str) -> None:
    exists = await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = $1", name)
    ident = await conn.fetchval("SELECT quote_ident($1)", name)
    literal = await conn.fetchval("SELECT quote_literal($1)", password)
    verb = "ALTER" if exists else "CREATE"
    await conn.execute(f"{verb} ROLE {ident} LOGIN PASSWORD {literal}")
    log.info("db.role_ready", role=name, created=not exists)


async def bootstrap(admin: PostgresAdminSettings, db: MigrationSettings) -> None:
    conn = await asyncpg.connect(
        host=db.host,
        port=db.port,
        user=admin.superuser,
        password=admin.superuser_password.get_secret_value(),
        database="postgres",
    )
    try:
        await _ensure_role(conn, db.owner_user, db.owner_password.get_secret_value())
        await _ensure_role(conn, db.app_user, db.app_password.get_secret_value())
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db.name):
            name = await conn.fetchval("SELECT quote_ident($1)", db.name)
            owner = await conn.fetchval("SELECT quote_ident($1)", db.owner_user)
            await conn.execute(f"CREATE DATABASE {name} OWNER {owner}")
            log.info("db.created", database=db.name)
    finally:
        await conn.close()

    conn = await asyncpg.connect(
        host=db.host,
        port=db.port,
        user=admin.superuser,
        password=admin.superuser_password.get_secret_value(),
        database=db.name,
    )
    try:
        app = await conn.fetchval("SELECT quote_ident($1)", db.app_user)
        name = await conn.fetchval("SELECT quote_ident($1)", db.name)
        await conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        await conn.execute(f"REVOKE ALL ON DATABASE {name} FROM PUBLIC")
        await conn.execute(f"GRANT CONNECT ON DATABASE {name} TO {app}")
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {app}")
    finally:
        await conn.close()
