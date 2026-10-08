"""Typed settings, one class per concern.

Each service loads only the settings it needs, so the recorder never sees database
credentials and the gateway never sees broker keys. Environment-specific values have
no defaults: a missing value stops the process at boot with a clear error. Only
operational tunables (timeouts, limits) carry defaults, and they live here.
"""

from datetime import time
from functools import lru_cache
from urllib.parse import quote
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Base(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class AppSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="APP_")

    env: str
    log_level: str = "INFO"


class BusSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="NATS_")

    url: str
    user: str
    password: SecretStr
    tls_ca_file: str | None = None
    connect_timeout_s: float = 5.0
    max_reconnect_attempts: int = -1
    ack_wait_s: float = 30.0
    max_deliver: int = 5
    max_inflight: int = 1024
    duplicate_window_s: int = 120
    marketdata_max_age_s: int = 3 * 86_400
    marketdata_max_bytes: int = 20 * 2**30
    features_max_age_s: int = 86_400
    features_max_bytes: int = 10 * 2**30
    signals_max_age_s: int = 7 * 86_400
    signals_max_bytes: int = 5 * 2**30
    alerts_max_age_s: int = 30 * 86_400
    alerts_max_bytes: int = 2**30
    deadletter_max_age_s: int = 30 * 86_400
    deadletter_max_bytes: int = 2**30
    retry_base_delay_s: float = 0.1
    retry_max_delay_s: float = 10.0


class DbSettings(_Base):
    """Application connection. Uses the least-privilege app role, never the owner."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str
    port: int
    name: str
    app_user: str
    app_password: SecretStr
    pool_size: int = 10
    max_overflow: int = 5
    statement_timeout_ms: int = 5000

    @property
    def dsn(self) -> str:
        return self._dsn(self.app_user, self.app_password)

    def _dsn(self, user: str, password: SecretStr) -> str:
        pw = quote(password.get_secret_value(), safe="")
        return f"postgresql+asyncpg://{user}:{pw}@{self.host}:{self.port}/{self.name}"


class PostgresAdminSettings(_Base):
    """Cluster superuser, used only by `niftypulse db-bootstrap` to create roles."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    superuser: str
    superuser_password: SecretStr


class MigrationSettings(DbSettings):
    """Schema owner, used only by Alembic and the bootstrap command."""

    owner_user: str
    owner_password: SecretStr

    @property
    def owner_dsn(self) -> str:
        return self._dsn(self.owner_user, self.owner_password)


class TelemetrySettings(_Base):
    model_config = SettingsConfigDict(env_prefix="OTEL_")

    exporter_otlp_endpoint: str
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    metrics_port: int


class BrokerSettings(_Base):
    """Upstox app credentials and endpoints, used by the login flow and the feed."""

    model_config = SettingsConfigDict(env_prefix="UPSTOX_")

    api_key: SecretStr
    api_secret: SecretStr
    redirect_uri: str
    api_base_url: str
    instrument_master_url: str
    token_file: str
    token_expiry_local: time = time(3, 30)
    http_timeout_s: float = 10.0


class MarketSettings(_Base):
    """Exchange calendar facts. Defaults are NSE's published session times."""

    model_config = SettingsConfigDict(env_prefix="MARKET_")

    timezone: str = "Asia/Kolkata"
    session_open: time = time(9, 15)
    session_close: time = time(15, 30)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class NseUniverseSettings(_Base):
    """NSE subscription universe. Contracts are resolved daily, never pinned."""

    model_config = SettingsConfigDict(env_prefix="NSE_UNIVERSE_")

    futures: list[str]
    equities: list[str]
    indices: list[str]
    option_underlying: str
    option_index_key: str
    option_strikes_each_side: int = 10


class BinanceSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    spot_ws_url: str
    spot_rest_url: str
    usdm_ws_url: str
    usdm_rest_url: str
    spot_symbols: list[str]
    usdm_symbols: list[str]
    snapshot_limit: int = 1000
    depth_interval_ms: int = 100
    http_timeout_s: float = 10.0


class FeedSettings(_Base):
    """Recorder behaviour shared by every venue adapter."""

    model_config = SettingsConfigDict(env_prefix="FEED_")

    venues: list[str]
    publish_depth: int = 20
    queue_maxsize: int = 100_000
    ping_interval_s: float = 20.0
    reconnect_base_s: float = 1.0
    reconnect_max_s: float = 60.0
    stale_after_s: float = 15.0


class LakeSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="LAKE_")

    root: str
    flush_rows: int = 50_000
    flush_interval_s: float = 30.0
    ack_wait_multiplier: float = 3.0


class GatewaySettings(_Base):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_")

    host: str
    port: int
    jwt_secret: SecretStr
    jwt_ttl_s: int = 900
    rate_limit_per_min: int = 120


@lru_cache
def app_settings() -> AppSettings:
    return AppSettings()


@lru_cache
def bus_settings() -> BusSettings:
    return BusSettings()


@lru_cache
def db_settings() -> DbSettings:
    return DbSettings()


@lru_cache
def telemetry_settings() -> TelemetrySettings:
    return TelemetrySettings()


@lru_cache
def feed_settings() -> FeedSettings:
    return FeedSettings()


@lru_cache
def lake_settings() -> LakeSettings:
    return LakeSettings()


@lru_cache
def gateway_settings() -> GatewaySettings:
    return GatewaySettings()


@lru_cache
def broker_settings() -> BrokerSettings:
    return BrokerSettings()


@lru_cache
def nse_universe_settings() -> NseUniverseSettings:
    return NseUniverseSettings()


@lru_cache
def market_settings() -> MarketSettings:
    return MarketSettings()


@lru_cache
def binance_settings() -> BinanceSettings:
    return BinanceSettings()


@lru_cache
def migration_settings() -> MigrationSettings:
    return MigrationSettings()


@lru_cache
def postgres_admin_settings() -> PostgresAdminSettings:
    return PostgresAdminSettings()
