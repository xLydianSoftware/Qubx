"""
Composite Metric Emitter.

This module provides a composite implementation of IMetricEmitter that delegates to multiple emitters.
"""

import datetime
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import pandas as pd

from qubx import logger
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition, dt_64
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

    def emit_signals(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        signals: list[Signal],
        account: IAccountViewer,
        target_positions: list[TargetPosition] | None = None,
    ) -> None:
        """
        Emit signals to all configured emitters.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information
            target_positions: List of target positions (optional)
        """
        for emitter in self._emitters:
            try:
                emitter.emit_signals(time, signals, account, target_positions)
            except Exception as e:
                logger.error(f"Error emitting signals to {emitter.__class__.__name__}: {e}")

    def emit_deals(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        instrument: Instrument,
        deals: list[Deal],
        account: IAccountViewer,
    ) -> None:
        """
        Emit deals to all configured emitters.

        Args:
            time: Timestamp when the deals were generated
            instrument: Instrument the deals belong to
            deals: List of deals to emit
            account: Account viewer to get account information
        """
        for emitter in self._emitters:
            try:
                emitter.emit_deals(time, instrument, deals, account)
            except Exception as e:
                logger.error(f"Error emitting deals to {emitter.__class__.__name__}: {e}")

    def ensure_table(
        self,
        table: str,
        columns: dict[str, str],
        symbol_columns: Sequence[str] = (),
        dedup_keys: Sequence[str] | None = None,
        partition_by: str = "DAY",
    ) -> None:
        """
        Declare a strategy-owned table on all configured emitters.

        Args:
            table: Table name (e.g. "frab.trades")
            columns: Column name -> type (DOUBLE | LONG | STRING | BOOLEAN | TIMESTAMP)
            symbol_columns: Column names to create as indexed SYMBOL columns
            dedup_keys: Optional designated-timestamp-first dedup key columns
            partition_by: Partitioning unit (default DAY)
        """
        for emitter in self._emitters:
            try:
                emitter.ensure_table(
                    table, columns, symbol_columns=symbol_columns, dedup_keys=dedup_keys, partition_by=partition_by
                )
            except Exception as e:
                logger.error(f"Error ensuring table on {emitter.__class__.__name__}: {e}")

    def emit_record(
        self,
        table: str,
        record: dict[str, Any],
        symbol_columns: Sequence[str] = (),
        timestamp: dt_64 | None = None,
    ) -> None:
        """
        Emit one structured row to a strategy-owned table on all configured emitters.

        Args:
            table: Table name (should have been declared via ensure_table)
            record: Column name -> value for this row
            symbol_columns: Which keys are SYMBOL columns (used if the table was not declared)
            timestamp: Designated timestamp of the row (defaults to context time / now)
        """
        for emitter in self._emitters:
            try:
                emitter.emit_record(table, record, symbol_columns=symbol_columns, timestamp=timestamp)
            except Exception as e:
                logger.error(f"Error emitting record to {emitter.__class__.__name__}: {e}")

    def set_context(self, context: IStrategyContext) -> None:
        """
        Set the strategy context for all child emitters.

        Args:
            context: The strategy context to use
        """
        for emitter in self._emitters:
            try:
                emitter.set_context(context)
            except Exception as e:
                logger.error(f"Error setting context on {emitter.__class__.__name__}: {e}")

    def notify(self, context: IStrategyContext) -> None:
        for emitter in self._emitters:
            try:
                emitter.notify(context)
            except Exception as e:
                logger.error(f"Error notifying {emitter.__class__.__name__}: {e}")

    def stop(self) -> None:
        """Stop all child emitters."""
        for emitter in self._emitters:
            try:
                emitter.stop()
            except Exception as e:
                logger.error(f"[CompositeMetricEmitter] Error stopping {emitter.__class__.__name__}: {e}")
