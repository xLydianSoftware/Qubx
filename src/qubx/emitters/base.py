"""
Base Metric Emitter.

This module provides a base implementation of IMetricEmitter that can be extended by other emitters.
"""

from typing import Dict, List, Optional, Set

import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer, IMetricEmitter, IStrategyContext, ITimeProvider


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
        "price",
        "position_count",
        "position_pnl",
        "position_unrealized_pnl",
        "position_leverage",
    }

    def __init__(
        self, stats_to_emit: Optional[List[str]] = None, stats_interval: str = "1m", tags: dict[str, str] | None = None
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
        self._time_provider = None

    def _merge_tags(self, tags: dict[str, str] | None = None, instrument: Instrument | None = None) -> dict[str, str]:
        """
        Merge default tags with provided tags and instrument tags if provided.

        Args:
            tags: Additional tags to merge with default tags
            instrument: Optional instrument to add symbol and exchange tags from

        Returns:
            Dict[str, str]: Merged tags dictionary
        """
        result = self._default_tags.copy()

        if tags:
            result.update(tags)

        if instrument:
            result.update({"symbol": instrument.symbol, "exchange": instrument.exchange})

        return result

    def _emit_impl(self, name: str, value: float, tags: Dict[str, str], timestamp: dt_64 | None = None) -> None:
        """
        Implementation of emit to be overridden by subclasses.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric
        """
        pass

    def emit(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        timestamp: dt_64 | None = None,
        instrument: Instrument | None = None,
    ) -> None:
        """
        Emit a metric with the given name, value, and optional tags.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Optional dictionary of tags/labels to include with the metric
            timestamp: Optional timestamp for the metric (defaults to current time)
            instrument: Optional instrument to add symbol and exchange tags from
        """
        if self._time_provider is not None and timestamp is None:
            timestamp = self._time_provider.time()
        merged_tags = self._merge_tags(tags, instrument)
        self._emit_impl(name, float(value), merged_tags, timestamp)

    def set_time_provider(self, time_provider: ITimeProvider) -> None:
        """
        Set the time provider for the metric emitter.

        Args:
            time_provider: The time provider to use
        """
        self._time_provider = time_provider

    def emit_strategy_stats(self, context: IStrategyContext) -> None:
        """
        Emit standard strategy statistics.

        This method emits standard statistics about the strategy's state, such as
        total capital, leverage, and position information.

        Args:
            context: The strategy context to get statistics from
        """
        try:
            # Get current timestamp
            current_time = context.time()

            tags = {"type": "stats"}

            # Strategy-level metrics
            if "total_capital" in self._stats_to_emit:
                self.emit("total_capital", context.get_total_capital(), tags, timestamp=current_time)

            if "net_leverage" in self._stats_to_emit:
                self.emit("net_leverage", context.get_net_leverage(), tags, timestamp=current_time)

            if "gross_leverage" in self._stats_to_emit:
                self.emit("gross_leverage", context.get_gross_leverage(), tags, timestamp=current_time)

            if "universe_size" in self._stats_to_emit:
                self.emit("universe_size", len(context.instruments), tags, timestamp=current_time)

            if "position_count" in self._stats_to_emit:
                positions = context.get_positions()
                active_positions = [p for i, p in positions.items() if abs(p.quantity) > i.min_size]
                self.emit("position_count", len(active_positions), tags, timestamp=current_time)

            # Position-level metrics
            positions = context.get_positions()
            total_capital = context.get_total_capital()

            for instrument, position in positions.items():
                pos_tags = {"type": "stats"}

                if "price" in self._stats_to_emit and (q := context.quote(instrument)) is not None:
                    self.emit("price", q.mid_price(), pos_tags, timestamp=current_time, instrument=instrument)

                if "position_pnl" in self._stats_to_emit:
                    self.emit("position_pnl", position.pnl, pos_tags, timestamp=current_time, instrument=instrument)

                if "position_unrealized_pnl" in self._stats_to_emit:
                    # Call the method to get the value
                    self.emit(
                        "position_unrealized_pnl",
                        position.unrealized_pnl(),
                        pos_tags,
                        timestamp=current_time,
                        instrument=instrument,
                    )

                if "position_leverage" in self._stats_to_emit and total_capital > 0:
                    position_leverage = context.account.get_leverage(instrument) * 100
                    self.emit(
                        "position_leverage", position_leverage, pos_tags, timestamp=current_time, instrument=instrument
                    )

        except Exception as e:
            logger.error(f"[BaseMetricEmitter] Failed to emit strategy stats: {e}")

    def emit_signals(
        self,
        time: dt_64,
        signals: list["Signal"],
        account: "IAccountViewer",
        target_positions: list["TargetPosition"] | None = None,
    ) -> None:
        """
        Emit signals to the monitoring system.

        Base implementation does nothing - subclasses should override this method
        to implement specific signal emission logic.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information like total capital, leverage, etc.
            target_positions: Optional list of target positions generated from the signals
        """
        pass

    def notify(self, context: IStrategyContext) -> None:
        """
        Notify the metric emitter of a time update.

        This method checks if enough time has passed since the last emission
        and emits metrics if necessary.

        Args:
            context: The strategy context to get statistics from
        """
        if not context.is_live_or_warmup:
            return

        # Convert to pandas timestamp for easier time calculations
        current_time = context.time()

        # If this is the first notification, initialize the last emission time
        if self._last_emission_time is None:
            self._last_emission_time = current_time
            return

        # Check if enough time has passed since the last emission
        elapsed = current_time - self._last_emission_time

        if elapsed >= self._stats_interval:
            # logger.debug(f"[{self.__class__.__name__}] Emitting metrics at {current_time}")
            self.emit_strategy_stats(context)
            self._last_emission_time = current_time
