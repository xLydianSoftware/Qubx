"""
Composite Metric Emitter.

This module provides a composite implementation of IMetricEmitter that delegates to multiple emitters.
"""

from typing import Dict, List

from qubx import logger
from qubx.core.interfaces import IMetricEmitter


class CompositeMetricEmitter(IMetricEmitter):
    """
    Composite metric emitter that delegates to multiple emitters.

    This emitter can be used to send metrics to multiple destinations
    by combining multiple emitters into one.
    """

    def __init__(self, emitters: List[IMetricEmitter]):
        """
        Initialize the Composite Metric Emitter.

        Args:
            emitters: List of emitters to delegate to
        """
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
