"""
Factory functions for creating various components used in strategy running and simulation.
"""

import inspect
import os
from typing import Any, Optional

from qubx import logger
from qubx.core.interfaces import IAccountViewer, IMetricEmitter, IStrategyLifecycleNotifier, ITradeDataExport
from qubx.data.composite import CompositeReader
from qubx.data.readers import DataReader
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.utils.misc import class_import
from qubx.utils.runner.configs import EmissionConfig, ExporterConfig, NotifierConfig, ReaderConfig, TypedReaderConfig


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


def construct_reader(reader_config: ReaderConfig | None) -> DataReader | None:
    if reader_config is None:
        return None

    from qubx.data.registry import ReaderRegistry

    try:
        # Use the ReaderRegistry.get method to construct the reader directly
        return ReaderRegistry.get(reader_config.reader, **reader_config.args)
    except ValueError as e:
        # Log the error and re-raise
        logger.error(f"Failed to construct reader: {e}")
        raise


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

            # Process parameters and resolve environment variables
            params: dict[str, Any] = {}
            for key, value in metric_config.parameters.items():
                params[key] = resolve_env_vars(value)

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

            # Process tags and add strategy_name as a tag
            tags = dict(metric_config.tags)
            for k, v in tags.items():
                tags[k] = resolve_env_vars(v)

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


def create_data_type_readers(readers_configs: list[TypedReaderConfig] | None) -> dict[str, DataReader]:
    """
    Create a dictionary mapping data types to readers based on the readers list.

    This function ensures that identical reader configurations are only instantiated once,
    and multiple data types can share the same reader instance if they have identical configurations.

    Args:
        readers_configs: The readers list containing reader definitions.

    Returns:
        A dictionary mapping data types to reader instances.
    """
    if readers_configs is None:
        return {}

    # First, create unique readers to avoid duplicate instantiation
    unique_readers = {}  # Maps reader config hash to reader instance
    data_type_to_reader = {}  # Maps data type to reader instance

    for typed_reader_config in readers_configs:
        data_types = typed_reader_config.data_type
        if isinstance(data_types, str):
            data_types = [data_types]
        readers_for_types = []

        for reader_config in typed_reader_config.readers:
            # Create a hashable representation of the reader config
            # Create a hashable key from reader name and stringified args
            if reader_config.args:
                args_str = str(reader_config.args)
                reader_key = f"{reader_config.reader}:{args_str}"
            else:
                reader_key = reader_config.reader

            # Check if we've already created this reader
            if reader_key not in unique_readers:
                try:
                    reader = construct_reader(reader_config)
                    if reader is None:
                        raise ValueError(f"Reader {reader_config.reader} could not be created")
                    unique_readers[reader_key] = reader
                except Exception as e:
                    logger.error(f"Reader {reader_config.reader} could not be created: {e}")
                    raise

            # Add the reader to the list for these data types
            readers_for_types.append(unique_readers[reader_key])

        # Create a composite reader if needed, or use the single reader
        if len(readers_for_types) > 1:
            composite_reader = CompositeReader(readers_for_types)
            for data_type in data_types:
                data_type_to_reader[data_type] = composite_reader
        elif len(readers_for_types) == 1:
            single_reader = readers_for_types[0]
            for data_type in data_types:
                data_type_to_reader[data_type] = single_reader

    return data_type_to_reader


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

            # Process parameters and resolve environment variables
            params = {}
            for key, value in exporter_config.parameters.items():
                resolved_value = resolve_env_vars(value)

                # Handle formatter if specified
                if key == "formatter" and isinstance(resolved_value, dict):
                    formatter_class_name = resolved_value.get("class")
                    formatter_args = resolved_value.get("args", {})

                    # Resolve env vars in formatter args
                    for fmt_key, fmt_value in formatter_args.items():
                        formatter_args[fmt_key] = resolve_env_vars(fmt_value)

                    if account and "account" not in formatter_args:
                        formatter_args["account"] = account

                    if formatter_class_name:
                        if "." not in formatter_class_name:
                            formatter_class_name = f"qubx.exporters.formatters.{formatter_class_name}"
                        formatter_class = class_import(formatter_class_name)
                        params[key] = formatter_class(**formatter_args)
                else:
                    params[key] = resolved_value

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


def create_lifecycle_notifiers(
    notifiers: list[NotifierConfig] | None, strategy_name: str
) -> IStrategyLifecycleNotifier | None:
    """
    Create lifecycle notifiers from the configuration.

    Args:
        notifiers: List of notifier configurations
        strategy_name: Name of the strategy

    Returns:
        IStrategyLifecycleNotifier or None if no lifecycle notifiers are configured
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

            # Process parameters and resolve environment variables
            params = {}
            for key, value in notifier_config.parameters.items():
                params[key] = resolve_env_vars(value)

            # Create throttler if configured or use default TimeWindowThrottler
            if "SlackLifecycleNotifier" in notifier_class_name and (
                "throttle" not in params or params["throttle"] is None
            ):
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
            notifier = notifier_class(**params)
            _notifiers.append(notifier)
            logger.info(f"Created lifecycle notifier: {notifier_class_name}")

        except Exception as e:
            logger.error(f"Failed to create lifecycle notifier {notifier_class_name}: {e}")
            logger.opt(colors=False).error(f"Lifecycle notifier parameters: {notifier_config.parameters}")

    if not _notifiers:
        return None

    # If there's only one notifier, return it directly
    if len(_notifiers) == 1:
        return _notifiers[0]

    # If there are multiple notifiers, create a composite notifier
    from qubx.notifications.composite import CompositeLifecycleNotifier

    return CompositeLifecycleNotifier(_notifiers)
