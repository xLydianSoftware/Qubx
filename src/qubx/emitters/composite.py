"""
Composite Metric Emitter.

This module provides a composite implementation of IMetricEmitter that delegates to multiple emitters.
"""

from typing import Dict, List, Optional

from qubx import logger
from qubx.core.basics import Signal, dt_64
from qubx.core.interfaces import IAccountViewer, IMetricEmitter, IStrategyContext
from qubx.emitters.base import BaseMetricEmitter


class CompositeMetricEmitter(BaseMetricEmitter):
    """
    A composite metric emitter that delegates to multiple emitters.

    This emitter can be used to send metrics to multiple monitoring systems at once.
    """

    def __init__(
        self,
        emitters: List[IMetricEmitter],
        stats_to_emit: Optional[List[str]] = None,
        stats_interval: str = "1m",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Composite Metric Emitter.

        Args:
            emitters: List of emitters to delegate to
            stats_to_emit: Optional list of specific stats to emit
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
        """
        super().__init__(stats_to_emit, stats_interval, tags)
        self._emitters = emitters

    def _emit_impl(self, name: str, value: float, tags: Dict[str, str], timestamp: dt_64 | None = None) -> None:
        """
        Implementation of emit for the composite emitter.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric
        """
        for emitter in self._emitters:
            try:
                emitter.emit(name, value, tags, timestamp)
            except Exception as e:
                logger.error(f"Error emitting metric to {emitter.__class__.__name__}: {e}")

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

    def emit_signals(self, time: dt_64, signals: list["Signal"], account: "IAccountViewer") -> None:
        """
        Emit signals to all configured emitters.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information
        """
        for emitter in self._emitters:
            try:
                emitter.emit_signals(time, signals, account)
            except Exception as e:
                logger.error(f"Error emitting signals to {emitter.__class__.__name__}: {e}")

    def notify(self, context: IStrategyContext) -> None:
        for emitter in self._emitters:
            try:
                emitter.notify(context)
            except Exception as e:
                logger.error(f"Error notifying {emitter.__class__.__name__}: {e}")
