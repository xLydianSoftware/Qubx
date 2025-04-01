import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from qubx.core.interfaces import IStrategy


class ExchangeConfig(BaseModel):
    connector: str
    universe: list[str]
    params: dict = Field(default_factory=dict)


class ReaderConfig(BaseModel):
    reader: str
    args: dict = Field(default_factory=dict)


class TypedReaderConfig(BaseModel):
    data_type: list[str] | str
    readers: list[ReaderConfig]


class RestorerConfig(BaseModel):
    type: str
    parameters: dict = Field(default_factory=dict)


class WarmupConfig(BaseModel):
    readers: list[TypedReaderConfig] = Field(default_factory=list)
    restorer: RestorerConfig | None = None


class LoggingConfig(BaseModel):
    logger: str
    position_interval: str
    portfolio_interval: str
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


class StrategyConfig(BaseModel):
    name: str | None = None
    strategy: str | list[str] | type[IStrategy]
    parameters: dict = Field(default_factory=dict)
    read_only: bool = False
    exchanges: dict[str, ExchangeConfig]
    logging: LoggingConfig
    aux: ReaderConfig | None = None
    exporters: list[ExporterConfig] | None = None
    emission: EmissionConfig | None = None
    notifiers: list[NotifierConfig] | None = None
    warmup: WarmupConfig | None = None
    health: HealthConfig = Field(default_factory=HealthConfig)


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


class StrategySimulationConfig(BaseModel):
    strategy: str | list[str]
    parameters: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)
    simulation: dict = Field(default_factory=dict)
    description: str | list[str] | None = None
    variate: dict = Field(default_factory=dict)


def load_simulation_config_from_yaml(path: Path | str) -> StrategySimulationConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return StrategySimulationConfig(**cfg)
