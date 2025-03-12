"""
Null Metric Emitter.

This module provides a no-op implementation of IMetricEmitter that can be used as a default.
"""

from typing import Dict

from qubx.metrics.base import BaseMetricEmitter


class NullMetricEmitter(BaseMetricEmitter):
    """
    A no-op implementation of IMetricEmitter.

    This emitter does nothing and can be used as a default when no metric emitter is provided.
    It inherits the notify and emit_strategy_stats methods from BaseMetricEmitter but
    overrides the actual emission methods to do nothing.
    """

    def emit_gauge(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Do nothing implementation of emit_gauge.

        Args:
            name: Name of the metric
            value: Current value of the metric
            tags: Dictionary of tags/labels for the metric
        """
        pass

    def emit_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] | None = None) -> None:
        """
        Do nothing implementation of emit_counter.

        Args:
            name: Name of the metric
            value: Amount to increment the counter
            tags: Dictionary of tags/labels for the metric
        """
        pass

    def emit_summary(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Do nothing implementation of emit_summary.

        Args:
            name: Name of the metric
            value: Value to add to the summary
            tags: Dictionary of tags/labels for the metric
        """
        pass
