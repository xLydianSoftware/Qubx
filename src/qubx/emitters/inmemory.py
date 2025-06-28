"""
In-Memory Metric Emitter.

This module provides an implementation of IMetricEmitter that stores metrics in memory
using a pandas DataFrame for easy access and analysis.
"""

from typing import Optional

import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, dt_64
from qubx.core.interfaces import IStrategyContext
from qubx.emitters.base import BaseMetricEmitter
from qubx.utils.ntp import time_now


class InMemoryMetricEmitter(BaseMetricEmitter):
    """
    Emits metrics to an in-memory DataFrame.

    This emitter stores all metrics in a pandas DataFrame for easy access and analysis.
    Useful for testing, debugging, or real-time analysis of strategy metrics.
    """

    def __init__(
        self,
        stats_to_emit: list[str] | None = None,
        stats_interval: str = "1m",
        tags: dict[str, str] | None = None,
        max_rows: int | None = None,
    ):
        """
        Initialize the In-Memory Metric Emitter.

        Args:
            stats_to_emit: Optional list of specific stats to emit
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
            max_rows: Maximum number of rows to keep in memory (oldest entries are dropped)
        """
        super().__init__(stats_to_emit, stats_interval, tags)

        # Store emitted rows as a list of dicts for efficiency
        self._rows: list[dict] = []
        self._tag_columns = set()
        self._max_rows = max_rows
        logger.info(f"[InMemoryMetricEmitter] Initialized with max_rows={max_rows}")

    def notify(self, context: IStrategyContext) -> None:
        """
        Notify the metric emitter of a time update.

        This method checks if enough time has passed since the last emission
        and emits metrics if necessary.

        Args:
            context: The strategy context to get statistics from
        """
        # Convert to pandas timestamp for easier time calculations
        current_time = pd.Timestamp(context.time())

        # Floor current time to the stats interval
        floored_current = current_time.floor(self._stats_interval)

        if self._last_emission_time is None:
            self._last_emission_time = floored_current
            return

        floored_last = self._last_emission_time

        # Only emit if we've entered a new interval
        if floored_current > floored_last:
            self.emit_strategy_stats(context)
            self._last_emission_time = floored_current

    def _emit_impl(self, name: str, value: float, tags: dict[str, str], timestamp: dt_64 | None = None) -> None:
        """
        Implementation of emit for in-memory storage.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric
        """
        try:
            # Use NTP-synchronized time if no timestamp provided
            current_timestamp = timestamp if timestamp is not None else time_now()

            # Convert numpy datetime64 to pandas Timestamp
            if isinstance(current_timestamp, dt_64):
                current_timestamp = pd.Timestamp(current_timestamp)

            # Extract symbol and exchange from tags
            symbol = tags.pop("symbol", None)
            exchange = tags.pop("exchange", None)

            # Create base row data
            row_data = {
                "timestamp": current_timestamp,
                "name": name,
                "value": value,
                "symbol": symbol,
                "exchange": exchange,
            }

            # Add any additional tags as columns
            for tag_key, tag_value in tags.items():
                row_data[tag_key] = tag_value
                self._tag_columns.add(tag_key)

            self._rows.append(row_data)
            # Trim list if max_rows is set
            if self._max_rows is not None and len(self._rows) > self._max_rows:
                self._rows = self._rows[-self._max_rows :]

        except Exception as e:
            logger.error(f"[InMemoryMetricEmitter] Failed to emit metric {name}: {e}")

    def get_dataframe(
        self,
        instrument: Instrument | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        metric_name: str | None = None,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Get the metrics DataFrame with optional filtering.
        """
        if not self._rows:
            df = pd.DataFrame(columns=["timestamp", "name", "value", "symbol", "exchange"])
        else:
            df = pd.DataFrame(self._rows)
            # Ensure correct dtypes
            df = df.astype(
                {
                    "timestamp": "datetime64[ns]",
                    "name": "string",
                    "value": "float64",
                    "symbol": "string",
                    "exchange": "string",
                }
            )
        if copy:
            df = df.copy()

        # Filter by instrument (takes precedence over symbol/exchange)
        if instrument is not None:
            df = df[(df["symbol"] == instrument.symbol) & (df["exchange"] == instrument.exchange)]
        else:
            if symbol is not None:
                df = df[df["symbol"] == symbol]
            if exchange is not None:
                df = df[df["exchange"] == exchange]
        if metric_name is not None:
            df = df[df["name"] == metric_name]
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]
        return df

    def get_latest_metrics(
        self, instrument: Instrument | None = None, symbol: str | None = None, exchange: str | None = None
    ) -> pd.DataFrame:
        """
        Get the latest metrics for each metric name, optionally filtered by instrument.

        Args:
            instrument: Filter by specific instrument
            symbol: Filter by symbol (ignored if instrument is provided)
            exchange: Filter by exchange (ignored if instrument is provided)

        Returns:
            pd.DataFrame: DataFrame with latest metrics
        """
        df = self.get_dataframe(instrument=instrument, symbol=symbol, exchange=exchange, copy=False)

        if df.empty:
            return df

        # Group by metric name and get the latest timestamp for each
        latest_df = df.loc[df.groupby("name")["timestamp"].idxmax()]
        return latest_df.reset_index(drop=True)

    def get_metric_summary(self) -> pd.DataFrame:
        """
        Get a summary of all metrics including count, latest timestamp, and unique instruments.

        Returns:
            pd.DataFrame: Summary statistics for each metric
        """
        if not self._rows:
            return pd.DataFrame(columns=["metric_name", "count", "latest_timestamp", "unique_instruments"])

        summary_data = []

        for metric_name in set(row["name"] for row in self._rows):
            metric_df = [row for row in self._rows if row["name"] == metric_name]

            # Count unique instruments (combinations of symbol and exchange)
            unique_instruments = len(
                set((row["symbol"], row["exchange"]) for row in metric_df if row["symbol"] and row["exchange"])
            )

            summary_data.append(
                {
                    "metric_name": metric_name,
                    "count": len(metric_df),
                    "latest_timestamp": max(row["timestamp"] for row in metric_df),
                    "unique_instruments": unique_instruments,
                }
            )

        return pd.DataFrame(summary_data)

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._rows.clear()
        logger.info("[InMemoryMetricEmitter] Cleared all stored metrics")

    def get_instruments(self) -> list[tuple[str, str]]:
        """
        Get all unique instruments (symbol, exchange pairs) in the stored metrics.

        Returns:
            list[tuple[str, str]]: List of (symbol, exchange) tuples
        """
        return [(row["symbol"], row["exchange"]) for row in self._rows if row["symbol"] and row["exchange"]]

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the stored DataFrame."""
        return self.get_dataframe(copy=False).shape

    @property
    def memory_usage(self) -> pd.Series:
        """Get memory usage of the stored DataFrame."""
        return self.get_dataframe(copy=False).memory_usage(deep=True)
