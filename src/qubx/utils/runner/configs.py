import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from qubx.core.interfaces import IStrategy


def resolve_env_vars_recursive(value: Any) -> Any:
    """
    Recursively resolve environment variables in config values.

    Supports:
    - env:VARIABLE (legacy format, no default)
    - env:{VARIABLE} (new format, no default)
    - env:{VARIABLE:default} (new format with default)

    Args:
        value: Any config value (dict, list, string, or other)

    Returns:
        The value with all environment variables resolved

    Raises:
        ValueError: If env var not found and no default provided
    """
    if isinstance(value, dict):
        return {k: resolve_env_vars_recursive(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars_recursive(v) for v in value]
    elif isinstance(value, str):
        # New format: env:{VAR} or env:{VAR:default}
        if value.startswith("env:{") and value.endswith("}"):
            var_spec = value[5:-1]  # Extract content between env:{ and }
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
                return os.environ.get(var_name, default)
            else:
                env_value = os.environ.get(var_spec)
                if env_value is None:
                    raise ValueError(f"Environment variable '{var_spec}' not found")
                return env_value
        # Legacy format: env:VAR (no braces)
        elif value.startswith("env:"):
            var_name = value[4:]
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable '{var_name}' not found")
            return env_value
        return value
    else:
        return value


class StrictBaseModel(BaseModel):
    """Base model with strict validation that forbids extra fields."""

    model_config = ConfigDict(extra="forbid")


class ConnectorConfig(StrictBaseModel):
    connector: str
    params: dict = Field(default_factory=dict)


class ExchangeConfig(StrictBaseModel):
    connector: str
    universe: list[str]
    params: dict = Field(default_factory=dict)
    broker: ConnectorConfig | None = None
    account: ConnectorConfig | None = None
    base_currency: str | None = None


class StorageConfig(StrictBaseModel):
    storage: str
    args: dict = Field(default_factory=dict)


class TypedStorageConfig(StrictBaseModel):
    data_type: list[str] | str
    storages: list[StorageConfig]


class RestorerConfig(StrictBaseModel):
    type: str
    parameters: dict = Field(default_factory=dict)


class PrefetchConfig(StrictBaseModel):
    enabled: bool = True
    prefetch_period: str = "1w"
    cache_size_mb: int = 1000
    aux_data_names: list[str] = Field(default_factory=list)
    args: dict = Field(default_factory=dict)


class WarmupConfig(StrictBaseModel):
    data: StorageConfig
    custom_data: list[TypedStorageConfig] = Field(default_factory=list)
    restorer: RestorerConfig | None = None
    enable_funding: bool = False


class LoggingConfig(StrictBaseModel):
    logger: str
    position_interval: str | None = None
    portfolio_interval: str | None = None
    position_log_on_change: bool = False
    args: dict = Field(default_factory=dict)
    heartbeat_interval: str | None = None


class ExporterConfig(StrictBaseModel):
    exporter: str
    parameters: dict = Field(default_factory=dict)


class EmitterConfig(StrictBaseModel):
    """Configuration for a single metric emitter."""

    emitter: str
    parameters: dict = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class EmissionConfig(StrictBaseModel):
    """Configuration for metric emission."""

    stats_interval: str = "1m"  # Default interval for emitting strategy stats
    stats_to_emit: list[str] | None = None  # Optional list of specific stats to emit
    emitters: list[EmitterConfig] = Field(default_factory=list)


class NotifierConfig(StrictBaseModel):
    """Configuration for strategy lifecycle notifiers."""

    notifier: str
    parameters: dict = Field(default_factory=dict)


class StatePersistenceConfig(StrictBaseModel):
    """Configuration for state persistence."""

    type: str  # e.g., "RedisStatePersistence"
    parameters: dict = Field(default_factory=dict)
    snapshot_interval: str | None = "5s"  # Interval for periodic state snapshots (None to disable)


class HealthConfig(StrictBaseModel):
    emit_health: bool = False
    emit_interval: str = "10s"
    monitor_interval: str = "1s"
    buffer_size: int = 5000


class DataTypeThrottleConfig(StrictBaseModel):
    """Configuration for throttling a specific data type."""

    data_type: str  # e.g., "quote", "orderbook", "trade"
    max_frequency_hz: float = 2.0  # Maximum updates per second per instrument
    enabled: bool = True


class ThrottlingConfig(StrictBaseModel):
    """Configuration for data throttling to reduce processing overhead."""

    enabled: bool = True
    throttles: list[DataTypeThrottleConfig] = Field(default_factory=list)


class RateLimitingConfig(StrictBaseModel):
    """Configuration for exchange rate limiting."""

    backend: str = "local"  # "local" (in-memory) or "redis"
    redis_url: str | None = None  # Required when backend is "redis"
    egress_ip: str = "auto"  # "auto" for periodic discovery, or explicit IP
    ip_check_interval: int = 60  # Seconds between egress IP checks (when "auto")
    metrics_interval: str = "60s"  # Interval for emitting rate limit metrics (None to disable)


class LiveConfig(StrictBaseModel):
    read_only: bool = False
    base_currency: str | None = None
    exchanges: dict[str, ExchangeConfig]
    logging: LoggingConfig
    exporters: list[ExporterConfig] | None = None
    emission: EmissionConfig | None = None
    notifiers: list[NotifierConfig] | None = None
    warmup: WarmupConfig | None = None
    health: HealthConfig = Field(default_factory=HealthConfig)
    throttling: ThrottlingConfig | None = None
    aux: list[StorageConfig] | StorageConfig | None = None
    prefetch: PrefetchConfig = Field(default_factory=PrefetchConfig)
    state: StatePersistenceConfig | None = None
    rate_limiting: RateLimitingConfig | None = None


class SimulationConfig(StrictBaseModel):
    capital: float | dict[str, float]
    instruments: list[str]
    start: str
    stop: str
    data: StorageConfig
    custom_data: list[TypedStorageConfig] = Field(default_factory=list)
    commissions: dict | str | None = None
    base_currency: str | None = None
    n_jobs: int | None = None
    variate: dict = Field(default_factory=dict)
    debug: str | None = None
    run_separate_instruments: bool = False
    enable_funding: bool = False
    enable_inmemory_emitter: bool = False
    prefetch: PrefetchConfig | None = None
    aux: list[StorageConfig] | StorageConfig | None = None
    portfolio_log_freq: str | None = None
    trading_session: str | dict | None = None


class PluginsConfig(StrictBaseModel):
    """Configuration for plugin loading."""

    paths: list[str] = Field(default_factory=list)
    """Paths to scan for plugin .py files (for local development)."""

    modules: list[str] = Field(default_factory=list)
    """Module names to import (for pip-installed packages)."""


class ReleaseSourceConfig(StrictBaseModel):
    """Source repository for building a release."""

    repo: str
    """GitHub org/repo (e.g., 'xLydianSoftware/xincubator')."""

    ref: str
    """Git ref to build from — tag, branch, or commit SHA."""

    subdirectory: str | None = None
    """Optional subpath within the source repo where the strategy's package
    lives. Required for monorepo sources where the workspace root has no
    Python project (e.g., uv workspaces). When set, the release wheel build
    runs from ``<clone>/<subdirectory>`` instead of the workspace root.

    Example: ``subdirectory: "e2e-driver"`` for a strategy living in
    ``xLydianSoftware/exchanges/e2e-driver/``.

    Must be a relative path; absolute paths or paths that escape the clone
    via ``..`` are rejected at parse time."""

    @field_validator("subdirectory")
    @classmethod
    def _validate_subdirectory(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if os.path.isabs(v):
            raise ValueError(f"subdirectory must be a relative path, got {v!r}")
        # Reject `..` segments to keep the build path inside the clone.
        parts = os.path.normpath(v).split(os.sep)
        if any(p == ".." for p in parts):
            raise ValueError(f"subdirectory must not escape the clone via '..', got {v!r}")
        return os.path.normpath(v)


class ReleasePlatformConfig(StrictBaseModel):
    """Platform deployment configuration."""

    name: str
    """Release name on platform.xlydian.com."""

    exchanges: list[str] = Field(default_factory=list)
    """Exchange identifiers (e.g., ['binance'])."""

    image_tag: str | None = None
    """Qubx Docker image tag (e.g., '1.1.3.dev16'). Defaults to latest."""

    tags: list[str] = Field(default_factory=list)
    """Descriptive tags for the release."""


class ReleaseConfig(StrictBaseModel):
    """Configuration for automated release packaging and platform deployment."""

    source: ReleaseSourceConfig
    """Source repository and ref to build from."""

    platform: ReleasePlatformConfig | None = None
    """Platform deployment settings. If omitted, release is built but not deployed."""


class StrategyConfig(StrictBaseModel):
    name: str | None = None
    description: str | list[str] | None = None
    tags: str | list[str] | None = None
    strategy: str | list[str] | type[IStrategy]
    parameters: dict = Field(default_factory=dict)
    aux: list[StorageConfig] | StorageConfig | None = None
    plugins: PluginsConfig | None = None
    live: LiveConfig | None = None
    simulation: SimulationConfig | None = None
    release: ReleaseConfig | None = None


def normalize_aux_config(aux_config: list[StorageConfig] | StorageConfig | None) -> list[StorageConfig]:
    """
    Normalize aux config to a list of StorageConfig objects.

    Args:
        aux_config: Can be None, single StorageConfig, or list of StorageConfig

    Returns:
        List of StorageConfig objects (empty list if aux_config is None)
    """
    if aux_config is None:
        return []
    elif isinstance(aux_config, list):
        return aux_config
    else:
        return [aux_config]


def resolve_aux_config(
    global_aux: list[StorageConfig] | StorageConfig | None, section_aux: list[StorageConfig] | StorageConfig | None
) -> list[StorageConfig]:
    """
    Resolve aux config with section-specific overrides.

    Args:
        global_aux: Global aux config from StrategyConfig
        section_aux: Section-specific aux config (simulation/live)

    Returns:
        List of StorageConfig objects, with section config taking precedence
    """
    if section_aux is not None:
        return normalize_aux_config(section_aux)
    else:
        return normalize_aux_config(global_aux)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into base. Dicts merge recursively, everything else replaces."""
    merged = dict(base)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_strategy_config_from_yaml(
    path: Path | str,
    key: str | None = None,
    overrides_path: Path | str | None = None,
    resolve_env: bool = True,
) -> StrategyConfig:
    """
    Loads a strategy configuration from a YAML file.

    Environment variables in the config are resolved recursively.
    Supported formats:
    - env:VARIABLE (legacy format)
    - env:{VARIABLE} (new format)
    - env:{VARIABLE:default} (new format with default value)

    Args:
        path (str | Path): The path to the YAML file.
        key (str | None): The key to extract from the YAML file.
        overrides_path: Optional sparse YAML file to deep-merge on top of base config.
        resolve_env: Whether to resolve env: variables. Set to False for contexts
            like release where env vars may not be available.

    Returns:
        StrategyConfig: The parsed configuration with env vars resolved.
    """
    path = Path(os.path.expanduser(path))
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    with path.open("r") as f:
        config_dict = yaml.safe_load(f)
        if key:
            config_dict = config_dict[key]

        # Deep-merge overrides if provided
        if overrides_path:
            overrides_path = Path(os.path.expanduser(overrides_path))
            if overrides_path.exists():
                with overrides_path.open("r") as of:
                    overrides_dict = yaml.safe_load(of) or {}
                config_dict = _deep_merge(config_dict, overrides_dict)

        if resolve_env:
            config_dict = resolve_env_vars_recursive(config_dict)
        return StrategyConfig(**config_dict)


class ValidationResult(StrictBaseModel):
    """Result of configuration validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_strategy_config(path: Path | str, check_imports: bool = True) -> ValidationResult:
    """
    Validates a strategy configuration file.

    Args:
        path: Path to the strategy configuration YAML file.
        check_imports: Whether to verify strategy class can be imported (default: True).

    Returns:
        ValidationResult with validation status, errors, and warnings.
    """
    result = ValidationResult(valid=True)

    # Check if file exists
    path = Path(os.path.expanduser(path))
    if not path.exists():
        result.valid = False
        result.errors.append(f"Configuration file not found: {path}")
        return result

    # Try to load and parse YAML
    try:
        config = load_strategy_config_from_yaml(path)
    except yaml.YAMLError as e:
        result.valid = False
        result.errors.append(f"YAML parsing error: {e}")
        return result
    except Exception as e:
        result.valid = False
        result.errors.append(f"Configuration parsing error: {e}")
        return result

    # Validate strategy class can be imported
    if check_imports:
        if isinstance(config.strategy, str):
            try:
                from qubx.utils.misc import class_import

                class_import(config.strategy)
            except Exception as e:
                result.valid = False
                result.errors.append(f"Failed to import strategy '{config.strategy}': {e}")
        elif isinstance(config.strategy, list):
            for strat in config.strategy:
                try:
                    from qubx.utils.misc import class_import

                    class_import(strat)
                except Exception as e:
                    result.valid = False
                    result.errors.append(f"Failed to import strategy '{strat}': {e}")

    # Validate live configuration if present
    if config.live:
        if not config.live.exchanges:
            result.valid = False
            result.errors.append("Live configuration requires at least one exchange")

        for exchange_name, exchange_config in config.live.exchanges.items():
            if not exchange_config.universe:
                result.valid = False
                result.errors.append(f"Exchange '{exchange_name}' has no symbols in universe")

            if exchange_config.connector.lower() not in ["ccxt", "tardis", "xlighter"]:
                result.warnings.append(
                    f"Exchange '{exchange_name}' uses unknown connector: {exchange_config.connector}"
                )

    # Validate simulation configuration if present
    if config.simulation:
        if not config.simulation.instruments:
            result.valid = False
            result.errors.append("Simulation configuration requires at least one instrument")

        if config.simulation.capital <= 0:
            result.valid = False
            result.errors.append("Simulation capital must be positive")

        # Validate date format
        try:
            import pandas as pd

            pd.Timestamp(config.simulation.start)
            pd.Timestamp(config.simulation.stop)
        except Exception as e:
            result.valid = False
            result.errors.append(f"Invalid simulation date format: {e}")

    # Check that at least one mode (live or simulation) is configured
    if not config.live and not config.simulation:
        result.warnings.append("Configuration has neither 'live' nor 'simulation' section")

    return result
