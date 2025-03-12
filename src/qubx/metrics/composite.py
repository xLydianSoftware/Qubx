"""
Composite Metric Emitter.

This module provides a composite implementation of IMetricEmitter that delegates to multiple emitters.
"""

from typing import Dict, List

from qubx import logger
from qubx.core.basics import dt_64
from qubx.core.interfaces import IStrategyContext
from qubx.metrics.base import BaseMetricEmitter


class CompositeMetricEmitter(BaseMetricEmitter):
    """
    Composite metric emitter that delegates to multiple emitters.

    This emitter can be used to send metrics to multiple destinations
    by combining multiple emitters into one.
    """

    def __init__(self, emitters: List[BaseMetricEmitter], stats_interval: str = "1m"):
        """
        Initialize the Composite Metric Emitter.

        Args:
            emitters: List of emitters to delegate to
            stats_interval: Interval for emitting strategy stats (default: "1m")
        """
        super().__init__(stats_interval=stats_interval)
        self._emitters = emitters

    def emit_gauge(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Emit a gauge metric to all configured emitters.

        Args:
            name: Name of the metric
            value: Current value of the metric
            tags: Dictionary of tags/labels for the metric
        """
        for emitter in self._emitters:
            try:
                emitter.emit_gauge(name, value, tags)
            except Exception as e:
                logger.error(f"Error emitting gauge to {emitter.__class__.__name__}: {e}")

    def emit_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] | None = None) -> None:
        """
        Emit a counter metric to all configured emitters.

        Args:
            name: Name of the metric
            value: Amount to increment the counter
            tags: Dictionary of tags/labels for the metric
        """
        for emitter in self._emitters:
            try:
                emitter.emit_counter(name, value, tags)
            except Exception as e:
                logger.error(f"Error emitting counter to {emitter.__class__.__name__}: {e}")

    def emit_summary(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Emit a summary metric to all configured emitters.

        Args:
            name: Name of the metric
            value: Value to add to the summary
            tags: Dictionary of tags/labels for the metric
        """
        for emitter in self._emitters:
            try:
                emitter.emit_summary(name, value, tags)
            except Exception as e:
                logger.error(f"Error emitting summary to {emitter.__class__.__name__}: {e}")

    def emit_strategy_stats(self, context: IStrategyContext) -> None:
        """
        Emit standard strategy statistics to all configured emitters.

        Args:
            context: The strategy context to get statistics from
        """
        # Store context for later use in notify method
        self._context = context

        for emitter in self._emitters:
            try:
                emitter.emit_strategy_stats(context)
            except Exception as e:
                logger.error(f"Error emitting strategy stats to {emitter.__class__.__name__}: {e}")

    def notify(self, timestamp: dt_64) -> None:
        """
        Notify all emitters of a time update.

        Args:
            timestamp: The current timestamp
        """
        # Store the timestamp for our own tracking
        super().notify(timestamp)

        # Also notify all child emitters
        for emitter in self._emitters:
            try:
                emitter.notify(timestamp)
            except Exception as e:
                logger.error(f"Error notifying {emitter.__class__.__name__}: {e}")
