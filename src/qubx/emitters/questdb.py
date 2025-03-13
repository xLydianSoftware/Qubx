"""
QuestDB Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to QuestDB.
"""

import datetime
from typing import Dict, List, Optional

import pandas as pd
from questdb.ingress import Sender

from qubx import logger
from qubx.core.basics import dt_64
from qubx.core.interfaces import IStrategyContext
from qubx.emitters.base import BaseMetricEmitter


class QuestDBMetricEmitter(BaseMetricEmitter):
    """
    Emits metrics to QuestDB using the QuestDB ingress client.

    This emitter sends metrics to QuestDB with custom timestamps and tags.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        table_name: str = "qubx_metrics",
        stats_to_emit: Optional[List[str]] = None,
        stats_interval: str = "1m",
        flush_interval: str = "5s",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the QuestDB Metric Emitter.

        Args:
            host: QuestDB server host
            port: QuestDB server port
            table_name: Name of the table to store metrics in
            stats_to_emit: Optional list of specific stats to emit
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
        """
        # Initialize default tags with strategy name
        default_tags = tags or {}

        super().__init__(stats_to_emit, stats_interval, default_tags)

        self._host = host
        self._port = port
        self._table_name = table_name
        self._conn_str = f"http::addr={host}:{port};"
        self._flush_interval = pd.Timedelta(flush_interval)
        self._sender = self._try_get_sender()
        self._last_flush = None

    def notify(self, context: IStrategyContext) -> None:
        super().notify(context)

        if self._last_flush is None:
            self._last_flush = pd.Timestamp.now()
            return

        if pd.Timestamp.now() - self._last_flush >= self._flush_interval:
            if self._sender is not None:
                try:
                    self._sender.flush()
                except:
                    pass
            self._last_flush = pd.Timestamp.now()

    def __del__(self):
        """Close the connection when the object is destroyed."""
        if hasattr(self, "_sender") and self._sender is not None:
            try:
                self._sender.close()
            except Exception:
                pass

    def _convert_timestamp(self, timestamp: dt_64) -> datetime.datetime:
        """Convert numpy.datetime64 to datetime."""
        # Convert to seconds since epoch as a float
        timestamp_seconds = float(timestamp) / 1e9  # Convert nanoseconds to seconds
        return datetime.datetime.fromtimestamp(timestamp_seconds)

    def _emit_impl(self, name: str, value: float, tags: Dict[str, str], timestamp: dt_64 | None = None) -> None:
        """
        Implementation of emit for QuestDB.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric
        """
        if self._sender is None:
            return

        try:
            # Prepare symbols (tags) and columns (values)
            symbols = {"metric_name": name}
            symbols.update(tags)  # Add all tags as symbols

            columns = {"value": value}  # Add the value as a column

            # Use the provided timestamp if available, otherwise use current time
            dt_timestamp = self._convert_timestamp(timestamp) if timestamp is not None else datetime.datetime.now()

            # Send the row to QuestDB
            self._sender.row(self._table_name, symbols=symbols, columns=columns, at=dt_timestamp)

        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to emit metric {name}: {e}")

    def _try_get_sender(self) -> Sender | None:
        try:
            _sender = Sender.from_conf(self._conn_str)
            _sender.establish()
            logger.info(f"[QuestDBMetricEmitter] Initialized QuestDB at {self._host}:{self._port}")
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to connect to QuestDB: {e}")
            _sender = None
        return _sender
