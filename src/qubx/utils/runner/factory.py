"""
Factory functions for creating various components used in strategy running and simulation.
"""

import inspect
import os
from typing import Any

from qubx import logger
from qubx.core.interfaces import IMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.utils.misc import class_import
from qubx.utils.runner.configs import EmissionConfig


def _resolve_env_vars(value: str) -> str:
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


def create_metric_emitters(emission_config: EmissionConfig, strategy_name: str) -> IMetricEmitter | None:
    """
    Create metric emitters from the configuration.

    Args:
        emission_config: Configuration for metric emission
        strategy_name: Name of the strategy to be included in tags

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
                params[key] = _resolve_env_vars(value)

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
                tags[k] = _resolve_env_vars(v)

            tags["strategy"] = strategy_name

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
