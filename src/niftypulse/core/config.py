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


class FeatureSettings(_Base):
    """Feature definitions. Any change produces a new feature-set id."""

    model_config = SettingsConfigDict(env_prefix="FEATURES_")

    leader_instrument: str
    windows_s: list[float] = [1.0, 5.0, 30.0]
    rv_windows_s: list[float] = [5.0, 30.0, 300.0]
    ofi_levels: int = 5
    history_s: float = 60.0


class LabelSettings(_Base):
    """Targets. A move counts as up/down only if it beats the half-spread times
    `spread_multiple` (and at least `min_move_bps`), so labels are cost-aware."""

    model_config = SettingsConfigDict(env_prefix="LABELS_")

    horizons_s: list[float] = [1.0, 5.0, 30.0]
    vol_horizon_s: float = 30.0
    spread_multiple: float = 1.0
    min_move_bps: float = 0.5
    sample_interval_ms: int = 1000
    # feed outages: rows whose latest book, or whose forward price, is older than this
    # are dropped instead of being filled across the gap
    max_staleness_s: float = 5.0


class ModelSettings(_Base):
    """Hyperparameters for every model family; defaults are documented starting points."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    seed: int = 7
    n_threads: int = 4
    early_stopping_rounds: int = 50
    # quiet windows have exactly zero realized vol; flooring keeps log targets and QLIKE sane
    vol_floor_bps: float = 0.1
    logistic_c: float = 0.1
    gbm_learning_rate: float = 0.03
    gbm_num_leaves: int = 31
    gbm_min_child_samples: int = 200
    gbm_subsample: float = 0.8
    gbm_colsample: float = 0.8
    gbm_l2: float = 5.0
    gbm_max_rounds: int = 2000
    xgb_max_depth: int = 6
    seq_len: int = 30
    seq_hidden: int = 64
    seq_layers: int = 2
    seq_heads: int = 4
    seq_dropout: float = 0.1
    seq_epochs: int = 30
    seq_batch_size: int = 512
    seq_learning_rate: float = 1e-3
    seq_weight_decay: float = 1e-4
    seq_patience: int = 5
    student_hidden: int = 32
    distill_temperature: float = 2.0
    distill_alpha: float = 0.7
    prune_amount: float = 0.5


class ResearchSettings(_Base):
    """Validation protocol and artifact locations."""

    model_config = SettingsConfigDict(env_prefix="RESEARCH_")

    artifacts_root: str
    cv_splits: int = 5
    embargo_s: float = 60.0
    holdout_fraction: float = 0.2
    min_abs_score: float = 0.1
    bootstrap_block: int = 300
    leakage_min_shift_fraction: float = 0.25
    leakage_max_abs_ic: float = 0.5
    leakage_max_shuffled_skill: float = 0.01
    lead_lag_steps: list[int] = [1, 2, 5, 10, 30]
    lead_lag_surrogates: int = 200
    lead_lag_min_shift_steps: int = 300


class RegimeSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="REGIME_")

    seed: int = 7
    hmm_states: int = 3
    kalman_level_var: float = 1e-9
    kalman_velocity_var: float = 1e-12
    kalman_obs_var: float = 1e-8
    bocpd_hazard_lambda: float = 600.0
    bocpd_max_run: int = 600
    bocpd_recent_window: int = 10


class PolicySettings(_Base):
    """Decision layer: model selection, sizing and execution."""

    model_config = SettingsConfigDict(env_prefix="POLICY_")

    bandit_prior_mean: float = 0.0
    bandit_prior_var: float = 1.0
    bandit_noise_var: float = 1.0
    entry_threshold: float = 0.15
    max_position: float = 1.0
    max_gross: float = 2.0
    risk_aversion: float = 5.0
    turnover_penalty_bps: float = 1.0
    covariance_lookback: int = 600
    exec_parent_lots: int = 10
    exec_horizon_s: int = 60
    exec_episodes: int = 400
    ppo_epochs: int = 8
    ppo_clip: float = 0.2
    ppo_gamma: float = 0.99
    ppo_lambda: float = 0.95
    ppo_learning_rate: float = 3e-4
    ppo_hidden: int = 64
    paper_notional: float = 10_000.0
    seed: int = 7


class CostSettings(_Base):
    """Fee schedules (published rates, Oct 2026). Override in .env when venues change them."""

    model_config = SettingsConfigDict(env_prefix="COSTS_")

    binance_spot_maker_bps: float = 10.0
    binance_spot_taker_bps: float = 10.0
    binance_usdm_maker_bps: float = 2.0
    binance_usdm_taker_bps: float = 5.0
    nse_brokerage_per_order: float = 20.0
    nse_brokerage_cap_pct: float = 0.05
    nse_exchange_txn_pct: float = 0.00173
    nse_sebi_per_crore: float = 10.0
    nse_stt_sell_pct: float = 0.02
    nse_stamp_buy_pct: float = 0.002
    nse_gst_pct: float = 18.0


class MonitorSettings(_Base):
    model_config = SettingsConfigDict(env_prefix="MONITOR_")

    rolling_window: int = 2000
    drift_interval_s: float = 300.0
    drift_sample: int = 5000
    psi_threshold: float = 0.2
    clock_interval_s: float = 60.0
    clock_endpoint: str
    prediction_stale_s: float = 30.0


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


@lru_cache
def feature_settings() -> FeatureSettings:
    return FeatureSettings()


@lru_cache
def label_settings() -> LabelSettings:
    return LabelSettings()


@lru_cache
def model_settings() -> ModelSettings:
    return ModelSettings()


@lru_cache
def research_settings() -> ResearchSettings:
    return ResearchSettings()


@lru_cache
def regime_settings() -> RegimeSettings:
    return RegimeSettings()


@lru_cache
def policy_settings() -> PolicySettings:
    return PolicySettings()


@lru_cache
def cost_settings() -> CostSettings:
    return CostSettings()


@lru_cache
def monitor_settings() -> MonitorSettings:
    return MonitorSettings()
