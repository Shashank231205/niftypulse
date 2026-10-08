"""Typed settings, one class per concern.

Each service loads only the settings it needs, so the recorder never sees database
credentials and the gateway never sees broker keys. Environment-specific values have
no defaults: a missing value stops the process at boot with a clear error. Only
operational tunables (timeouts, limits) carry defaults, and they live here.
"""

from functools import lru_cache

from pydantic import Field, PostgresDsn, SecretStr
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


class DbSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="DB_")

    dsn: PostgresDsn
    pool_size: int = 10
    max_overflow: int = 5
    statement_timeout_ms: int = 5000


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
    http_timeout_s: float = 10.0


class UniverseSettings(_Base):
    """What the recorder subscribes to. Contracts are resolved daily, never hardcoded."""

    model_config = SettingsConfigDict(env_prefix="UNIVERSE_")

    futures: list[str]
    equities: list[str]
    indices: list[str]
    option_underlying: str
    option_index_key: str
    option_strikes_each_side: int = 10


class FeedSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="FEED_")

    mode: str = "full"
    reconnect_base_s: float = 1.0
    reconnect_max_s: float = 60.0
    stale_after_s: float = 15.0


class LakeSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="LAKE_")

    root: str
    flush_rows: int = 50_000
    flush_interval_s: float = 30.0


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
def universe_settings() -> UniverseSettings:
    return UniverseSettings()
