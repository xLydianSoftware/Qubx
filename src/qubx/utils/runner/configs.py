import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from qubx.core.interfaces import IStrategy


class ConnectorConfig(BaseModel):
    connector: str
    params: dict = Field(default_factory=dict)


class ExchangeConfig(BaseModel):
    connector: str
    universe: list[str]
    params: dict = Field(default_factory=dict)
    broker: ConnectorConfig | None = None
    account: ConnectorConfig | None = None


class ReaderConfig(BaseModel):
    reader: str
    args: dict = Field(default_factory=dict)


class TypedReaderConfig(BaseModel):
    data_type: list[str] | str
    readers: list[ReaderConfig]


class RestorerConfig(BaseModel):
    type: str
    parameters: dict = Field(default_factory=dict)


class PrefetchConfig(BaseModel):
    enabled: bool = True
    prefetch_period: str = "1w"
    cache_size_mb: int = 1000
    aux_data_names: list[str] = Field(default_factory=list)
    args: dict = Field(default_factory=dict)


class WarmupConfig(BaseModel):
    readers: list[TypedReaderConfig] = Field(default_factory=list)
    restorer: RestorerConfig | None = None
    prefetch: PrefetchConfig | None = None


class LoggingConfig(BaseModel):
    logger: str
    position_interval: str
    portfolio_interval: str
    args: dict = Field(default_factory=dict)
    heartbeat_interval: str = "1m"


class ExporterConfig(BaseModel):
    exporter: str
    parameters: dict = Field(default_factory=dict)


class EmitterConfig(BaseModel):
    """Configuration for a single metric emitter."""

    emitter: str
    parameters: dict = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class EmissionConfig(BaseModel):
    """Configuration for metric emission."""

    stats_interval: str = "1m"  # Default interval for emitting strategy stats
    stats_to_emit: list[str] | None = None  # Optional list of specific stats to emit
    emitters: list[EmitterConfig] = Field(default_factory=list)


class NotifierConfig(BaseModel):
    """Configuration for strategy lifecycle notifiers."""

    notifier: str
    parameters: dict = Field(default_factory=dict)


class HealthConfig(BaseModel):
    emit_interval: str = "10s"
    queue_monitor_interval: str = "1s"
    buffer_size: int = 5000


class LiveConfig(BaseModel):
    read_only: bool = False
    exchanges: dict[str, ExchangeConfig]
    logging: LoggingConfig
    exporters: list[ExporterConfig] | None = None
    emission: EmissionConfig | None = None
    notifiers: list[NotifierConfig] | None = None
    warmup: WarmupConfig | None = None
    health: HealthConfig = Field(default_factory=HealthConfig)
    aux: list[ReaderConfig] | ReaderConfig | None = None


class SimulationConfig(BaseModel):
    capital: float
    instruments: list[str]
    start: str
    stop: str
    data: list[TypedReaderConfig] = Field(default_factory=list)
    commissions: dict | str | None = None
    n_jobs: int | None = None
    variate: dict = Field(default_factory=dict)
    debug: str | None = None
    run_separate_instruments: bool = False
    enable_funding: bool = False
    enable_inmemory_emitter: bool = False
    prefetch: PrefetchConfig | None = None
    aux: list[ReaderConfig] | ReaderConfig | None = None


class StrategyConfig(BaseModel):
    name: str | None = None
    description: str | list[str] | None = None
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
    global_aux: list[ReaderConfig] | ReaderConfig | None,
    section_aux: list[ReaderConfig] | ReaderConfig | None
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
