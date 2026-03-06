"""
Factory functions for creating various components used in strategy running and simulation.
"""

import inspect
import os
from typing import Any, Optional

from qubx import logger
from qubx.core.interfaces import IAccountViewer, IMetricEmitter, IStatePersistence, IStrategyNotifier, ITradeDataExport
from qubx.data.storage import IStorage
from qubx.data.storages.multi import MultiStorage
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.utils.misc import class_import
from qubx.utils.runner.configs import (
    EmissionConfig,
    ExporterConfig,
    NotifierConfig,
    StatePersistenceConfig,
    StorageConfig,
    TypedStorageConfig,
)


def resolve_env_vars(value: str | Any) -> str | Any:
    """
    Resolve environment variables in a value.
    If the value is a string and starts with 'env:', the rest is treated as an environment variable name.
    """
    if isinstance(value, str) and value.startswith("env:"):
        env_var = value[4:].strip()
        _value = os.environ.get(env_var)
        if _value is None:
            raise ValueError(f"Environment variable {env_var} not found")
        return _value
    return value


def construct_storage(storage_config: StorageConfig | None) -> IStorage | None:
    """
    Construct a storage from config using StorageRegistry.

    Handles both simple storage names (e.g., 'qdb') and URI-style names
    (e.g., 'qdb::quantlab', 'csv::/data/path/', 'mqdb::nebula').

    Args:
        storage_config: Storage configuration

    Returns:
        IStorage instance or None if storage_config is None

    Raises:
        ValueError: If the storage name is not registered in StorageRegistry
    """
    if storage_config is None:
        return None

    from qubx.data.registry import StorageRegistry

    storage_name = storage_config.storage
    kwargs = dict(storage_config.args)

    try:
        # - resolve storage class from registry
        storage_cls = StorageRegistry.get_class(storage_name)
    except ValueError as e:
        logger.error(
            f"Failed to resolve storage '{storage_name}'. "
            "Make sure it is registered via @storage() decorator or is a fully-qualified class name. "
            f"Available storages: {list(StorageRegistry.get_all_storages().keys())}. Error: {e}"
        )
        raise

    # - URI-style 'name::host' — first segment after '::' is the positional host/path arg
    if "::" in storage_name:
        db_path = storage_name.split("::", 1)[1]
        return storage_cls(db_path, **kwargs)

    return storage_cls(**kwargs)


def create_metric_emitters(
    emission_config: EmissionConfig, strategy_name: str, run_id: str | None = None
) -> IMetricEmitter | None:
    """
    Create metric emitters from the configuration.

    Args:
        emission_config: Configuration for metric emission
        strategy_name: Name of the strategy to be included in tags
        run_id: Optional run ID to be included in tags

    Returns:
        IMetricEmitter or None if no metric emitters are configured
    """
    if not emission_config.emitters:
        return None

    emitters = []
    stats_to_emit = emission_config.stats_to_emit
    stats_interval = emission_config.stats_interval

    for metric_config in emission_config.emitters:
        emitter_class_name = metric_config.emitter
        if "." not in emitter_class_name:
            emitter_class_name = f"qubx.emitters.{emitter_class_name}"

        try:
            emitter_class = class_import(emitter_class_name)

            # Copy parameters (env vars already resolved during config load)
            params: dict[str, Any] = dict(metric_config.parameters)

            # Add strategy_name if the emitter requires it and it's not already provided
            if "strategy_name" in inspect.signature(emitter_class).parameters and "strategy_name" not in params:
                params["strategy_name"] = strategy_name

            # Add stats_to_emit if the emitter supports it and it's not already provided
            if (
                "stats_to_emit" in inspect.signature(emitter_class).parameters
                and "stats_to_emit" not in params
                and stats_to_emit
            ):
                params["stats_to_emit"] = stats_to_emit

            # Add stats_interval if the emitter supports it and it's not already provided
            if "stats_interval" in inspect.signature(emitter_class).parameters and "stats_interval" not in params:
                params["stats_interval"] = stats_interval

            # Process tags and add strategy_name as a tag (env vars already resolved)
            tags = dict(metric_config.tags)
            tags["strategy"] = strategy_name
            if run_id is not None:
                tags["run_id"] = run_id

            # Add tags if the emitter supports it
            if "tags" in inspect.signature(emitter_class).parameters:
                params["tags"] = tags

            # Create the emitter instance
            emitter = emitter_class(**params)
            emitters.append(emitter)
            logger.info(f"Created metric emitter: {emitter_class_name}")
        except Exception as e:
            logger.error(f"Failed to create metric emitter {metric_config.emitter}: {e}")
            logger.opt(colors=False).error(f"Metric emitter parameters: {metric_config.parameters}")

    if not emitters:
        return None
    elif len(emitters) == 1:
        return emitters[0]
    else:
        return CompositeMetricEmitter(emitters, stats_interval=stats_interval)


def create_data_type_storages(storages_configs: list[TypedStorageConfig] | None) -> dict[str, IStorage]:
    """
    Create a dictionary mapping data types to readers based on the readers list.

    This function ensures that identical reader configurations are only instantiated once,
    and multiple data types can share the same reader instance if they have identical configurations.

    Args:
        readers_configs: The readers list containing reader definitions.
        account_manager: Optional account manager to inject into readers.

    Returns:
        A dictionary mapping data types to reader instances.
    """
    if storages_configs is None:
        return {}

    # First, create unique readers to avoid duplicate instantiation
    unique_readers = {}  # Maps reader config hash to reader instance
    data_type_to_storage = {}  # Maps data type to reader instance

    for typed_reader_config in storages_configs:
        data_types = typed_reader_config.data_type
        if isinstance(data_types, str):
            data_types = [data_types]
        readers_for_types = []

        for storage_config in typed_reader_config.storages:
            # Create a hashable representation of the reader config
            # Create a hashable key from reader name and stringified args
            if storage_config.args:
                args_str = str(storage_config.args)
                reader_key = f"{storage_config.storage}:{args_str}"
            else:
                reader_key = storage_config.storage

            # Check if we've already created this reader
            if reader_key not in unique_readers:
                try:
                    storage = construct_storage(storage_config)
                    if storage is None:
                        raise ValueError(f"Reader {storage_config.storage} could not be created")
                    unique_readers[reader_key] = storage
                except Exception as e:
                    logger.error(f"Reader {storage_config.storage} could not be created: {e}")
                    raise

            # Add the reader to the list for these data types
            readers_for_types.append(unique_readers[reader_key])

        # - wrap in MultiStorage when multiple storages cover the same data type
        if len(readers_for_types) > 1:
            multi = MultiStorage(readers_for_types)
            for data_type in data_types:
                data_type_to_storage[data_type] = multi
        elif len(readers_for_types) == 1:
            single_reader = readers_for_types[0]
            for data_type in data_types:
                data_type_to_storage[data_type] = single_reader

    return data_type_to_storage


def create_exporters(
    exporters: list[ExporterConfig] | None,
    strategy_name: str,
    account: Optional[IAccountViewer] = None,
) -> ITradeDataExport | None:
    """
    Create exporters from the configuration.

    Args:
        config: Strategy configuration
        strategy_name: Name of the strategy

    Returns:
        ITradeDataExport or None if no exporters are configured
    """
    if not exporters:
        return None

    _exporters = []

    for exporter_config in exporters:
        exporter_class_name = exporter_config.exporter
        if "." not in exporter_class_name:
            exporter_class_name = f"qubx.exporters.{exporter_class_name}"

        try:
            exporter_class = class_import(exporter_class_name)

            # Process parameters (env vars already resolved during config load)
            params = {}
            for key, value in exporter_config.parameters.items():
                # Handle formatter if specified
                if key == "formatter" and isinstance(value, dict):
                    formatter_class_name = value.get("class")
                    formatter_args = dict(value.get("args", {}))

                    if account and "account" not in formatter_args:
                        formatter_args["account"] = account

                    if formatter_class_name:
                        if "." not in formatter_class_name:
                            formatter_class_name = f"qubx.exporters.formatters.{formatter_class_name}"
                        formatter_class = class_import(formatter_class_name)
                        params[key] = formatter_class(**formatter_args)
                else:
                    params[key] = value

            # Add strategy_name if the exporter requires it and it's not already provided
            if "strategy_name" in inspect.signature(exporter_class).parameters and "strategy_name" not in params:
                params["strategy_name"] = strategy_name
            if account and "account" not in params:
                params["account"] = account

            # Create the exporter instance
            exporter = exporter_class(**params)
            _exporters.append(exporter)
            logger.info(f"Created exporter: {exporter_class_name}")

        except Exception as e:
            logger.error(f"Failed to create exporter {exporter_class_name}: {e}")
            logger.opt(colors=False).error(f"Exporter parameters: {exporter_config.parameters}")

    if not _exporters:
        return None

    # If there's only one exporter, return it directly
    if len(_exporters) == 1:
        return _exporters[0]

    # If there are multiple exporters, create a composite exporter
    from qubx.exporters.composite import CompositeExporter

    return CompositeExporter(_exporters)


def create_notifiers(notifiers: list[NotifierConfig] | None, strategy_name: str) -> IStrategyNotifier | None:
    """
    Create notifiers from the configuration.

    Args:
        notifiers: List of notifier configurations
        strategy_name: Name of the strategy

    Returns:
        IStrategyNotifier or None if no notifiers are configured
    """
    if not notifiers:
        return None

    _notifiers = []

    for notifier_config in notifiers:
        notifier_class_name = notifier_config.notifier
        if "." not in notifier_class_name:
            notifier_class_name = f"qubx.notifications.{notifier_class_name}"

        try:
            notifier_class = class_import(notifier_class_name)

            # Copy parameters (env vars already resolved during config load)
            params = dict(notifier_config.parameters)

            # Create throttler if configured or use default TimeWindowThrottler
            if "SlackNotifier" in notifier_class_name and ("throttle" not in params or params["throttle"] is None):
                # Import here to avoid circular imports
                from qubx.notifications.throttler import TimeWindowThrottler

                # Create default throttler with 10s window
                default_window = 10.0
                params["throttler"] = TimeWindowThrottler(window_seconds=default_window)
                logger.info(
                    f"Using default TimeWindowThrottler with window={default_window}s for {notifier_class_name}"
                )
            elif "throttle" in params:
                throttle_config = params.pop("throttle")

                if isinstance(throttle_config, dict):
                    throttler_type = throttle_config.get("type", "TimeWindow")
                    window_seconds = float(throttle_config.get("window_seconds", 10.0))
                    max_count = int(throttle_config.get("max_count", 3))

                    if throttler_type.lower() == "timewindow":
                        from qubx.notifications.throttler import TimeWindowThrottler

                        throttler = TimeWindowThrottler(window_seconds=window_seconds)
                        logger.info(f"Created TimeWindowThrottler with window_seconds={window_seconds}")
                    elif throttler_type.lower() == "countbased":
                        from qubx.notifications.throttler import CountBasedThrottler

                        throttler = CountBasedThrottler(max_count=max_count, window_seconds=window_seconds)
                        logger.info(
                            f"Created CountBasedThrottler with max_count={max_count}, window_seconds={window_seconds}"
                        )
                    elif throttler_type.lower() == "none":
                        from qubx.notifications.throttler import NoThrottling

                        throttler = NoThrottling()
                        logger.info("Created NoThrottling throttler")
                    else:
                        logger.warning(f"Unknown throttler type '{throttler_type}', defaulting to TimeWindowThrottler")
                        from qubx.notifications.throttler import TimeWindowThrottler

                        throttler = TimeWindowThrottler(window_seconds=window_seconds)

                    params["throttler"] = throttler
                elif isinstance(throttle_config, (int, float)):
                    # Simple case: just a window_seconds value
                    from qubx.notifications.throttler import TimeWindowThrottler

                    throttler = TimeWindowThrottler(window_seconds=float(throttle_config))
                    logger.info(f"Created TimeWindowThrottler with window_seconds={throttle_config}")
                    params["throttler"] = throttler

            # Create the notifier instance
            params["strategy_name"] = strategy_name
            notifier = notifier_class(**params)
            _notifiers.append(notifier)
            logger.info(f"Created notifier: {notifier_class_name}")

        except Exception as e:
            logger.error(f"Failed to create notifier {notifier_class_name}: {e}")
            logger.opt(colors=False).error(f"Notifier parameters: {notifier_config.parameters}")

    if not _notifiers:
        return None

    # If there's only one notifier, return it directly
    if len(_notifiers) == 1:
        return _notifiers[0]

    # If there are multiple notifiers, create a composite notifier
    from qubx.notifications.composite import CompositeNotifier

    return CompositeNotifier(_notifiers)


def construct_multi_storage(storage_configs: list[StorageConfig]) -> IStorage | None:
    """
    Construct auxiliary data storage from config.

    Args:
        storage_configs: List of storage configurations

    Returns:
        Single IStorage if only one config, MultiStorage if multiple configs, None if empty
    """
    if not storage_configs:
        return None

    elif len(storage_configs) == 1:
        return construct_storage(storage_configs[0])

    else:
        storages: list[IStorage] = []
        for config in storage_configs:
            try:
                s = construct_storage(config)
                if s is not None:
                    storages.append(s)
                    logger.debug(f"Created storage: {s.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Failed to create storage from config {config}: {e}")

        if not storages:
            logger.warning("No storages could be created from provided configs")
            return None
        elif len(storages) == 1:
            return storages[0]
        else:
            logger.info(f"Created MultiStorage with {len(storages)} storages")
            return MultiStorage(storages)


def create_state_persistence(
    config: StatePersistenceConfig | None,
    strategy_name: str,
) -> IStatePersistence | None:
    """
    Create state persistence from configuration.

    Args:
        config: State persistence configuration
        strategy_name: Name of the strategy

    Returns:
        IStatePersistence or None if no persistence is configured
    """
    if config is None:
        return None

    persistence_class_name = config.type
    if "." not in persistence_class_name:
        persistence_class_name = f"qubx.state.{persistence_class_name}"

    try:
        persistence_class = class_import(persistence_class_name)

        # Copy parameters (env vars already resolved during config load)
        params: dict[str, Any] = dict(config.parameters)

        # Add strategy_name if not already provided
        if "strategy_name" not in params:
            params["strategy_name"] = strategy_name

        persistence = persistence_class(**params)
        logger.info(f"Created state persistence: {persistence_class_name}")
        return persistence

    except Exception as e:
        logger.error(f"Failed to create state persistence {persistence_class_name}: {e}")
        logger.opt(colors=False).error(f"State persistence parameters: {config.parameters}")
        raise
