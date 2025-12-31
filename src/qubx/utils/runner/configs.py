import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from qubx.core.interfaces import IStrategy


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


class ReaderConfig(StrictBaseModel):
    reader: str
    args: dict = Field(default_factory=dict)


class TypedReaderConfig(StrictBaseModel):
    data_type: list[str] | str
    readers: list[ReaderConfig]


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
    readers: list[TypedReaderConfig] = Field(default_factory=list)
    restorer: RestorerConfig | None = None
    enable_funding: bool = False


class LoggingConfig(StrictBaseModel):
    logger: str
    position_interval: str
    portfolio_interval: str
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


class LiveConfig(StrictBaseModel):
    read_only: bool = False
    exchanges: dict[str, ExchangeConfig]
    logging: LoggingConfig
    exporters: list[ExporterConfig] | None = None
    emission: EmissionConfig | None = None
    notifiers: list[NotifierConfig] | None = None
    warmup: WarmupConfig | None = None
    health: HealthConfig = Field(default_factory=HealthConfig)
    throttling: ThrottlingConfig | None = None
    aux: list[ReaderConfig] | ReaderConfig | None = None
    prefetch: PrefetchConfig = Field(default_factory=PrefetchConfig)


class SimulationConfig(StrictBaseModel):
    capital: float
    instruments: list[str]
    start: str
    stop: str
    data: list[TypedReaderConfig] = Field(default_factory=list)
    commissions: dict | str | None = None
    base_currency: str | None = None
    n_jobs: int | None = None
    variate: dict = Field(default_factory=dict)
    debug: str | None = None
    run_separate_instruments: bool = False
    enable_funding: bool = False
    enable_inmemory_emitter: bool = False
    prefetch: PrefetchConfig | None = None
    aux: list[ReaderConfig] | ReaderConfig | None = None
    portfolio_log_freq: str | None = None


class StrategyConfig(StrictBaseModel):
    name: str | None = None
    description: str | list[str] | None = None
    tags: str | list[str] | None = None
    strategy: str | list[str] | type[IStrategy]
    parameters: dict = Field(default_factory=dict)
    aux: list[ReaderConfig] | ReaderConfig | None = None
    live: LiveConfig | None = None
    simulation: SimulationConfig | None = None


def normalize_aux_config(aux_config: list[ReaderConfig] | ReaderConfig | None) -> list[ReaderConfig]:
    """
    Normalize aux config to a list of ReaderConfig objects.

    Args:
        aux_config: Can be None, single ReaderConfig, or list of ReaderConfig

    Returns:
        List of ReaderConfig objects (empty list if aux_config is None)
    """
    if aux_config is None:
        return []
    elif isinstance(aux_config, list):
        return aux_config
    else:
        return [aux_config]


def resolve_aux_config(
    global_aux: list[ReaderConfig] | ReaderConfig | None, section_aux: list[ReaderConfig] | ReaderConfig | None
) -> list[ReaderConfig]:
    """
    Resolve aux config with section-specific overrides.

    Args:
        global_aux: Global aux config from StrategyConfig
        section_aux: Section-specific aux config (simulation/live)

    Returns:
        List of ReaderConfig objects, with section config taking precedence
    """
    if section_aux is not None:
        return normalize_aux_config(section_aux)
    else:
        return normalize_aux_config(global_aux)


def load_strategy_config_from_yaml(path: Path | str, key: str | None = None) -> StrategyConfig:
    """
    Loads a strategy configuration from a YAML file.

    Args:
        path (str | Path): The path to the YAML file.
        key (str | None): The key to extract from the YAML file.

    Returns:
        StrategyConfig: The parsed configuration.
    """
    path = Path(os.path.expanduser(path))
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    with path.open("r") as f:
        config_dict = yaml.safe_load(f)
        if key:
            config_dict = config_dict[key]
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
