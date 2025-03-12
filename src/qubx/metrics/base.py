"""
Base Metric Emitter.

This module provides a base implementation of IMetricEmitter that can be extended by other emitters.
"""

from typing import Dict, List, Optional, Set

import pandas as pd

from qubx import logger
from qubx.core.basics import dt_64
from qubx.core.interfaces import IMetricEmitter, IStrategyContext


class BaseMetricEmitter(IMetricEmitter):
    """
    A base implementation of IMetricEmitter.

    This class provides common functionality for metric emitters, including
    the notify method for time-based emission of metrics.
    """

    # Default set of statistics to emit
    DEFAULT_STATS = {
        "total_capital",
        "net_leverage",
        "gross_leverage",
        "universe_size",
        "position_count",
        "position_pnl",
        "position_unrealized_pnl",
        "position_leverage",
    }

    def __init__(
        self, stats_to_emit: Optional[List[str]] = None, stats_interval: str = "1m", tags: Dict[str, str] | None = None
    ):
        """
        Initialize the Base Metric Emitter.

        Args:
            stats_to_emit: Optional list of specific stats to emit. If None, all default stats are emitted.
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
        """
        self._stats_to_emit: Set[str] = set(stats_to_emit) if stats_to_emit else self.DEFAULT_STATS
        self._stats_interval = pd.Timedelta(stats_interval)
        self._default_tags = tags or {}
        self._last_emission_time = None
        self._context = None

    def _merge_tags(self, tags: Dict[str, str] | None = None) -> Dict[str, str]:
        """
        Merge default tags with provided tags.

        Args:
            tags: Additional tags to merge with default tags

        Returns:
            Dictionary of merged tags
        """
        if tags is None:
            return dict(self._default_tags)

        merged_tags = dict(self._default_tags)
        merged_tags.update(tags)
        return merged_tags

    def _emit_gauge_impl(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Implementation of emit_gauge to be overridden by subclasses.

        Args:
            name: Name of the metric
            value: Current value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
        """
        pass

    def _emit_counter_impl(self, name: str, value: float = 1.0, tags: Dict[str, str] | None = None) -> None:
        """
        Implementation of emit_counter to be overridden by subclasses.

        Args:
            name: Name of the metric
            value: Amount to increment the counter
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
        """
        pass

    def _emit_summary_impl(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Implementation of emit_summary to be overridden by subclasses.

        Args:
            name: Name of the metric
            value: Value to add to the summary
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
        """
        pass

    def emit_gauge(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Emit a gauge metric with merged tags.

        Args:
            name: Name of the metric
            value: Current value of the metric
            tags: Dictionary of tags/labels for the metric
        """
        merged_tags = self._merge_tags(tags)
        self._emit_gauge_impl(name, value, merged_tags)

    def emit_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] | None = None) -> None:
        """
        Emit a counter metric with merged tags.

        Args:
            name: Name of the metric
            value: Amount to increment the counter
            tags: Dictionary of tags/labels for the metric
        """
        merged_tags = self._merge_tags(tags)
        self._emit_counter_impl(name, value, merged_tags)

    def emit_summary(self, name: str, value: float, tags: Dict[str, str] | None = None) -> None:
        """
        Emit a summary metric with merged tags.

        Args:
            name: Name of the metric
            value: Value to add to the summary
            tags: Dictionary of tags/labels for the metric
        """
        merged_tags = self._merge_tags(tags)
        self._emit_summary_impl(name, value, merged_tags)

    def emit_strategy_stats(self, context: IStrategyContext) -> None:
        """
        Emit standard strategy statistics.

        This method emits standard statistics about the strategy's state, such as
        total capital, leverage, and position information.

        Args:
            context: The strategy context to get statistics from
        """
        try:
            # Store context for later use in notify method
            self._context = context

            # Strategy-level metrics
            if "total_capital" in self._stats_to_emit:
                self.emit_gauge("total_capital", context.get_total_capital())

            if "net_leverage" in self._stats_to_emit:
                self.emit_gauge("net_leverage", context.get_net_leverage())

            if "gross_leverage" in self._stats_to_emit:
                self.emit_gauge("gross_leverage", context.get_gross_leverage())

            if "universe_size" in self._stats_to_emit:
                self.emit_gauge("universe_size", len(context.instruments))

            if "position_count" in self._stats_to_emit:
                positions = context.get_positions()
                active_positions = [p for i, p in positions.items() if abs(p.quantity) > i.min_size]
                self.emit_gauge("position_count", len(active_positions))

            # Position-level metrics
            positions = context.get_positions()
            total_capital = context.get_total_capital()

            for instrument, position in positions.items():
                # Skip positions that are effectively zero
                if abs(position.quantity) <= instrument.min_size:
                    continue

                symbol = instrument.symbol
                tags = {"symbol": symbol, "exchange": instrument.exchange}

                if "position_pnl" in self._stats_to_emit:
                    self.emit_gauge("position_pnl", position.pnl, tags)

                if "position_unrealized_pnl" in self._stats_to_emit:
                    # Call the method to get the value
                    self.emit_gauge("position_unrealized_pnl", position.unrealized_pnl(), tags)

                if "position_leverage" in self._stats_to_emit and total_capital > 0:
                    # Calculate position leverage as (position value / total capital) * 100
                    # Use the current price from the market data
                    quote = context.quote(instrument)
                    if quote is not None:
                        current_price = quote.mid_price()
                        position_value = abs(position.quantity * current_price)
                        position_leverage = (position_value / total_capital) * 100
                        self.emit_gauge("position_leverage", position_leverage, tags)

        except Exception as e:
            logger.error(f"[BaseMetricEmitter] Failed to emit strategy stats: {e}")

    def notify(self, timestamp: dt_64) -> None:
        """
        Notify the metric emitter of a time update.

        This method checks if enough time has passed since the last emission
        and emits metrics if necessary.

        Args:
            timestamp: The current timestamp
        """
        # If context is not set, we can't emit metrics
        if self._context is None:
            return

        # Convert to pandas timestamp for easier time calculations
        current_time = pd.Timestamp(timestamp)

        # If this is the first notification, initialize the last emission time
        if self._last_emission_time is None:
            self._last_emission_time = current_time
            return

        # Check if enough time has passed since the last emission
        elapsed = current_time - self._last_emission_time

        if elapsed >= self._stats_interval:
            logger.debug(f"[{self.__class__.__name__}] Emitting metrics at {current_time}")
            self.emit_strategy_stats(self._context)
            self._last_emission_time = current_time
