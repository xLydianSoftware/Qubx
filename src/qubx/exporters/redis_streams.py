"""
Redis Streams Exporter for trading data.

This module provides an implementation of ITradeDataExport that exports trading data to Redis Streams.
"""

from typing import List, Optional

import redis

from qubx import logger
from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer, ITradeDataExport
from qubx.exporters.formatters import DefaultFormatter, IExportFormatter


class RedisStreamsExporter(ITradeDataExport):
    """
    Exports trading data to Redis Streams.

    This exporter can be configured to export signals, target positions, and leverage changes.
    """

    def __init__(
        self,
        redis_url: str,
        strategy_name: str,
        export_signals: bool = True,
        export_targets: bool = True,
        export_position_changes: bool = True,
        signals_stream: Optional[str] = None,
        targets_stream: Optional[str] = None,
        position_changes_stream: Optional[str] = None,
        max_stream_length: int = 1000,
        formatter: Optional[IExportFormatter] = None,
    ):
        """
        Initialize the Redis Streams Exporter.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            strategy_name: Name of the strategy (used in stream keys if not provided)
            export_signals: Whether to export signals
            export_targets: Whether to export target positions
            export_position_changes: Whether to export position changes
            signals_stream: Custom stream name for signals (default: "strategy:{strategy_name}:signals")
            targets_stream: Custom stream name for target positions (default: "strategy:{strategy_name}:targets")
            position_changes_stream: Custom stream name for position changes (default: "strategy:{strategy_name}:position_changes")
            max_stream_length: Maximum length of each stream
            formatter: Formatter to use for formatting data (default: DefaultFormatter)
        """
        self._redis = redis.from_url(redis_url)
        self._strategy_name = strategy_name

        self._export_signals = export_signals
        self._export_targets = export_targets
        self._export_position_changes = export_position_changes

        self._signals_stream = signals_stream or f"strategy:{strategy_name}:signals"
        self._targets_stream = targets_stream or f"strategy:{strategy_name}:targets"
        self._position_changes_stream = position_changes_stream or f"strategy:{strategy_name}:position_changes"

        self._max_stream_length = max_stream_length
        self._formatter = formatter or DefaultFormatter()

        self._instrument_to_previous_leverage = {}

        logger.info(
            f"[RedisStreamsExporter] Initialized for strategy '{strategy_name}' with "
            f"signals: {export_signals}, targets: {export_targets}, position_changes: {export_position_changes}"
        )

    def export_signals(self, time: dt_64, signals: List[Signal], account: IAccountViewer) -> None:
        """
        Export signals to Redis Stream.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to export
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        if not self._export_signals or not signals:
            return

        try:
            for signal in signals:
                # Format the signal using the formatter
                data = self._formatter.format_signal(time, signal, account)

                # Add to Redis stream
                self._redis.xadd(self._signals_stream, data, maxlen=self._max_stream_length, approximate=True)

            logger.debug(f"[RedisStreamsExporter] Exported {len(signals)} signals to {self._signals_stream}")
        except Exception as e:
            logger.error(f"[RedisStreamsExporter] Failed to export signals: {e}")

    def export_target_positions(self, time: dt_64, targets: List[TargetPosition], account: IAccountViewer) -> None:
        """
        Export target positions to Redis Stream.

        Args:
            time: Timestamp when the target positions were generated
            targets: List of target positions to export
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        if not self._export_targets or not targets:
            return

        try:
            for target in targets:
                # Format the target position using the formatter
                data = self._formatter.format_target_position(time, target, account)

                # Add to Redis stream
                self._redis.xadd(self._targets_stream, data, maxlen=self._max_stream_length, approximate=True)

            logger.debug(f"[RedisStreamsExporter] Exported {len(targets)} target positions to {self._targets_stream}")
        except Exception as e:
            logger.error(f"[RedisStreamsExporter] Failed to export target positions: {e}")

    def export_position_changes(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> None:
        """
        Export leverage changes to Redis Stream.

        Args:
            time: Timestamp when the leverage change occurred
            instrument: The instrument for which the leverage changed
            price: Price at which the leverage changed
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        if not self._export_position_changes:
            return

        previous_leverage = self._instrument_to_previous_leverage.get(instrument)
        new_leverage = account.get_leverage(instrument)

        try:
            # Format the leverage change using the formatter
            data = self._formatter.format_position_change(time, instrument, price, account)

            # Add to Redis stream
            self._redis.xadd(self._position_changes_stream, data, maxlen=self._max_stream_length, approximate=True)

            logger.debug(
                f"[RedisStreamsExporter] Exported position change for {instrument}: "
                f"{previous_leverage} -> {new_leverage} @ {price}"
            )
        except Exception as e:
            logger.error(f"[RedisStreamsExporter] Failed to export position change: {e}")
