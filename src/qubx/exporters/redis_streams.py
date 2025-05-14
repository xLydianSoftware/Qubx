"""
Redis Streams Exporter for trading data.

This module provides an implementation of ITradeDataExport that exports trading data to Redis Streams.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, cast

import redis
from redis.typing import EncodableT, FieldT

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
        max_workers: int = 2,
        account: Optional[IAccountViewer] = None,
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
            max_workers: Maximum number of worker threads for Redis operations
            account: Optional account viewer to get account information like total capital, leverage, etc.
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

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="redis_exporter")

        if account:
            self._instrument_to_previous_leverage = dict(account.get_leverages())

        logger.info(
            f"[RedisStreamsExporter] Initialized for strategy '{strategy_name}' with "
            f"signals: {export_signals}, targets: {export_targets}, position_changes: {export_position_changes}"
        )

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            self._executor.shutdown(wait=False)
        except:
            pass

    def _prepare_for_redis(self, data: Dict[str, Any]) -> Dict[FieldT, EncodableT]:
        """
        Prepare data for Redis by ensuring all values are strings.

        Args:
            data: Dictionary with string keys and any values

        Returns:
            Dictionary with keys and values compatible with Redis
        """
        # Convert all values to strings and cast the result to the type expected by Redis
        string_dict = {k: str(v) for k, v in data.items()}
        return cast(Dict[FieldT, EncodableT], string_dict)

    def _add_to_redis_stream(self, stream: str, data: Dict[str, Any]) -> None:
        """
        Add data to a Redis stream in a background thread.

        Args:
            stream: The name of the Redis stream
            data: The data to add to the stream
        """
        try:
            # Prepare data for Redis
            redis_data = self._prepare_for_redis(data)

            # Submit the task to the executor
            self._executor.submit(self._add_to_redis_stream_impl, stream, redis_data, self._max_stream_length)
        except Exception as e:
            logger.exception(f"[RedisStreamsExporter] Failed to queue Redis stream operation")

    def _add_to_redis_stream_impl(self, stream: str, redis_data: Dict[FieldT, EncodableT], max_length: int) -> bool:
        """
        Implementation that actually adds data to Redis stream (called from worker thread).

        Args:
            stream: The name of the Redis stream
            redis_data: The data to add to the stream
            max_length: The maximum length of the stream

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Add to Redis stream
            self._redis.xadd(stream, redis_data, maxlen=max_length, approximate=True)
            return True
        except Exception:
            logger.exception(f"[RedisStreamsExporter] Failed to add to Redis stream {stream}")
            return False

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

                # Add to Redis stream in background thread
                self._add_to_redis_stream(self._signals_stream, data)

            logger.debug(f"[RedisStreamsExporter] Queued {len(signals)} signals for export to {self._signals_stream}")
        except Exception:
            logger.exception(f"[RedisStreamsExporter] Failed to export signals")

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

                # Add to Redis stream in background thread
                self._add_to_redis_stream(self._targets_stream, data)

            logger.debug(
                f"[RedisStreamsExporter] Queued {len(targets)} target positions for export to {self._targets_stream}"
            )
        except Exception:
            logger.exception(f"[RedisStreamsExporter] Failed to export target positions")

    def export_position_changes(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> None:
        """
        Export position changes to Redis Stream.

        Args:
            time: Timestamp when the position change occurred
            instrument: The instrument for which the position change occurred
            price: Price at which the position change occurred
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        if not self._export_position_changes:
            return

        previous_leverage = self._instrument_to_previous_leverage.get(instrument, 0.0)
        new_leverage = account.get_leverage(instrument)
        self._instrument_to_previous_leverage[instrument] = new_leverage

        try:
            # Format the leverage change using the formatter
            data = self._formatter.format_position_change(time, instrument, price, account)

            # Add to Redis stream in background thread
            self._add_to_redis_stream(self._position_changes_stream, data)

            logger.debug(
                f"[RedisStreamsExporter] Queued position change for {instrument}: "
                f"{previous_leverage:0.2%} -> {new_leverage:0.2%} @ {price}"
            )
        except Exception:
            logger.exception(f"[RedisStreamsExporter] Failed to export position change")
