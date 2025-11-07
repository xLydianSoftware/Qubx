"""OrderBook handler for Lighter WebSocket messages"""

from typing import Any, Awaitable, Callable, Literal

import numpy as np

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.series import OrderBook, Quote, time_as_nsec
from qubx.core.utils import recognize_time
from qubx.utils.hft.orderbook import LOB
from qubx.utils.misc import AsyncThreadLoop

from .base import BaseHandler


class OrderbookHandler(BaseHandler[OrderBook]):
    """
    Handler for Lighter orderbook messages with state maintenance.

    Maintains full orderbook state by processing:
    1. Initial snapshots (type="subscribed/order_book") - Full orderbook
    2. Updates (type="update/order_book") - Incremental changes

    Lighter format:
    ```json
    {
      "channel": "order_book:0",
      "type": "subscribed/order_book" or "update/order_book",
      "timestamp": 1760041996048,
      "order_book": {
        "code": 0,
        "asks": [{"price": "4332.75", "size": "0.6998"}, ...],
        "bids": [{"price": "4332.50", "size": "1.2345"}, ...],
        "offset": 995817
      }
    }
    ```
    """

    def __init__(
        self,
        market_id: int,
        instrument: Instrument,
        max_levels: int = 200,
        tick_size_pct: float = 0,
        max_buffer_size: int = 10,
        resubscribe_callback: Callable[[], Awaitable[None]] | None = None,
        async_loop: AsyncThreadLoop | None = None,
        generate_quote: bool = False,
        buffer_overflow_resolution: Literal["resubscribe", "drain_buffer"] = "drain_buffer",
    ):
        """
        Initialize orderbook handler with LOB state maintainer.

        Args:
            market_id: Lighter market ID to handle
            tick_size: Minimum price increment for this market
            max_levels: Maximum number of levels to include in output (default: 200)
            tick_size_pct: Percentage for dynamic tick sizing (e.g., 0.01 for 0.01%)
            instrument: Instrument for price rounding (required if tick_size_pct is set)
            max_buffer_size: Maximum number of out-of-order messages to buffer (default: 10)
            resubscribe_callback: Callback to trigger resubscription on buffer overflow or corruption
            async_loop: AsyncThreadLoop for submitting async tasks (required for resubscription)
        """
        super().__init__()
        self.market_id = market_id
        self.tick_size = instrument.tick_size
        self.max_levels = max_levels
        self.tick_size_pct = tick_size_pct
        self.instrument = instrument
        self._lob = LOB(depth=max_levels)
        self._generate_quote = generate_quote
        self._buffer_overflow_resolution = buffer_overflow_resolution

        # Offset tracking for message ordering
        self._last_offset: int | None = None
        self._buffer: dict[int, dict[str, Any]] = {}
        self._max_buffer_size = max_buffer_size
        self._resubscribe_callback = resubscribe_callback
        self._async_loop = async_loop
        self._is_resubscribing = False  # Flag to ignore messages during resubscription

    def can_handle(self, message: dict[str, Any]) -> bool:
        channel = message.get("channel", "")
        msg_type = message.get("type", "")
        expected_channel = f"order_book:{self.market_id}"
        is_orderbook_msg = msg_type in ["subscribed/order_book", "update/order_book"]
        return channel == expected_channel and is_orderbook_msg

    def _handle_impl(self, message: dict[str, Any]) -> OrderBook | Quote | None:
        """
        Process Lighter orderbook message and return current state.

        This handler maintains orderbook state internally by:
        1. Applying initial snapshot on first "subscribed/order_book" message
        2. Buffering out-of-order "update/order_book" messages
        3. Applying updates in order when gaps are filled
        4. Triggering resubscription on buffer overflow or crossed orderbook

        Args:
            message: Raw Lighter orderbook message

        Returns:
            Current OrderBook state, or None if not yet initialized or empty

        Raises:
            ValueError: If message format is invalid
        """
        orderbook = self._process_update_message(message)
        if self._generate_quote and isinstance(orderbook, OrderBook):
            return orderbook.to_quote()
        return orderbook

    def _process_update_message(self, message: dict[str, Any]) -> OrderBook | None:
        is_snapshot = message.get("type") == "subscribed/order_book"
        book = message.get("order_book")
        if book is None:
            self._warning("Missing order_book in message")
            return None

        # If resubscribing, ignore all updates and only accept snapshots
        if self._is_resubscribing:
            if is_snapshot:
                # This is the new snapshot we're waiting for - reset flag and process
                self._info(
                    f"Received snapshot after resubscription for market {self.market_id}, resuming normal processing"
                )
                self._is_resubscribing = False
            else:
                # Ignore all updates during resubscription
                return None

        # Extract offset for message ordering
        offset = book.get("offset")

        # Handle snapshot messages (reset state)
        if is_snapshot:
            self._last_offset = offset
            self._buffer.clear()
            result = self._apply_message(message)
            return result

        # Handle update messages with offset-based ordering
        if offset is None:
            # No offset provided, apply immediately (backward compatibility)
            result = self._apply_message(message)
            # Check for crossed orderbook (check even if result is None)
            if self._is_orderbook_crossed():
                self._warning(f"Crossed orderbook detected for market {self.market_id}, triggering resubscription")
                self._trigger_resubscription()
                return None
            return result

        # Check if this is the next expected message
        expected_offset = self._last_offset + 1 if self._last_offset is not None else offset

        if offset == expected_offset:
            # Apply this message
            result = self._apply_message(message)
            self._last_offset = offset

            # Check for crossed orderbook (check even if result is None)
            if self._is_orderbook_crossed():
                self._warning(f"Crossed orderbook detected for market {self.market_id}, triggering resubscription")
                self._trigger_resubscription()
                return None

            # Try to drain buffer for consecutive messages
            drained_result = self._drain_buffer()
            return drained_result if drained_result is not None else result

        elif offset > expected_offset:
            # Out of order - buffer this message
            self._buffer[offset] = message
            self._check_buffer_overflow()
            # Return None since we're not applying this message yet
            return None

        else:
            # Old or duplicate message (offset <= last_offset), skip it
            return None

    def _get_tick_size(self) -> float:
        """
        Get tick size based on tick_size_pct and current mid price.
        """
        if self.tick_size_pct is not None and self.tick_size_pct > 0:
            try:
                mid_price = self._lob.get_mid()
                if not np.isnan(mid_price):
                    raw_tick_size = max(mid_price * self.tick_size_pct / 100.0, self.instrument.tick_size)
                    return self.instrument.round_price_down(raw_tick_size)
            except Exception:
                # LOB is empty or error occurred, fall back to default tick size
                pass
        return self.tick_size

    def _apply_message(self, message: dict[str, Any]) -> OrderBook | None:
        """
        Apply a message to the LOB and return the resulting OrderBook.

        Args:
            message: Lighter orderbook message to apply

        Returns:
            Current OrderBook state, or None if empty/invalid
        """
        # Extract timestamp (milliseconds) and convert to nanoseconds
        timestamp_ms = message.get("timestamp")
        if timestamp_ms is None:
            raise ValueError("Missing timestamp in orderbook message")

        # Convert to datetime64, then to int64 nanoseconds for LOB
        timestamp_dt = recognize_time(timestamp_ms)
        timestamp_ns = time_as_nsec(timestamp_dt)

        book = message.get("order_book")
        if book is None:
            self._warning("Missing order_book in message")
            return None

        is_update = message.get("type") == "update/order_book"

        # Parse bids and asks into numpy arrays (Nx2 format: [price, size])
        bids_list = book.get("bids", [])
        asks_list = book.get("asks", [])

        if bids_list:
            bids = np.array([[float(bid["price"]), float(bid["size"])] for bid in bids_list], dtype=np.float64)
        else:
            bids = None

        if asks_list:
            asks = np.array([[float(ask["price"]), float(ask["size"])] for ask in asks_list], dtype=np.float64)
        else:
            asks = None

        # Update LOB state (is_snapshot=True for initial subscription, False for updates)
        self._lob.update(
            timestamp=timestamp_ns,
            bids=bids,
            asks=asks,
            is_snapshot=not is_update,
            is_sorted=True,  # Lighter sends sorted data
        )

        return self._lob.get_orderbook(
            tick_size=self._get_tick_size(),
            levels=self.max_levels,
            sizes_in_quoted=False,
        )

    def _drain_buffer(self) -> OrderBook | None:
        """
        Process consecutive buffered messages after filling a gap.

        Returns:
            OrderBook from last applied message, or None
        """
        result = None
        while self._last_offset is not None:
            next_offset = self._last_offset + 1
            if next_offset in self._buffer:
                # Found next consecutive message
                message = self._buffer.pop(next_offset)
                result = self._apply_message(message)
                self._last_offset = next_offset

                # Check for crossed orderbook after each update
                if self._is_orderbook_crossed():
                    self._warning(f"Crossed orderbook detected for market {self.market_id}, triggering resubscription")
                    self._trigger_resubscription()
                    return None
            else:
                # No more consecutive messages
                break
        return result

    def _drain_buffer_ignore_missing(self) -> OrderBook | None:
        """
        Process buffered messages ignoring missing messages.

        Returns:
            OrderBook from last applied message, or None
        """
        result = None
        offsets = sorted(self._buffer.keys())
        for offset in offsets:
            self._last_offset = offset
            message = self._buffer.pop(offset)
            result = self._apply_message(message)
        if self._is_orderbook_crossed():
            self._warning(f"Crossed orderbook detected for market {self.market_id}, triggering resubscription")
            self._trigger_resubscription()
            return None
        return result

    def _check_buffer_overflow(self) -> None:
        """Check if buffer has overflowed and trigger resubscription if needed."""
        if len(self._buffer) >= self._max_buffer_size:
            if self._buffer_overflow_resolution == "resubscribe":
                self._warning(
                    f"Orderbook message buffer overflow for market {self.market_id} "
                    f"(size={len(self._buffer)}, max={self._max_buffer_size}), triggering resubscription"
                )
                self._trigger_resubscription()
            elif self._buffer_overflow_resolution == "drain_buffer":
                self._warning(
                    f"Orderbook message buffer overflow for market {self.market_id} "
                    f"(size={len(self._buffer)}, max={self._max_buffer_size}), draining buffer"
                )
                self._drain_buffer_ignore_missing()
            else:
                raise ValueError(f"Invalid buffer overflow resolution: {self._buffer_overflow_resolution}")

    def _is_orderbook_crossed(self) -> bool:
        """
        Check if the current orderbook is crossed (best_bid >= best_ask).

        A crossed orderbook indicates corrupted data and should trigger resubscription.

        Returns:
            True if orderbook has data and is crossed, False otherwise
        """
        try:
            # Use public methods to get top of book
            best_bid = self._lob.get_bid()
            best_ask = self._lob.get_ask()

            # Check if both sides have data and are crossed
            if not np.isnan(best_bid) and not np.isnan(best_ask):
                return best_bid >= best_ask
        except Exception:
            # If we can't access the data, assume it's not crossed
            pass
        return False

    def _trigger_resubscription(self) -> None:
        """Trigger resubscription callback to get fresh orderbook snapshot."""
        # Set flag to ignore incoming messages until snapshot arrives
        self._is_resubscribing = True
        self._info(f"Entering resubscription mode for market {self.market_id}, ignoring updates until snapshot arrives")

        # Clear buffer and reset state
        self._buffer.clear()
        self._last_offset = None

        # Call resubscription callback if available
        if self._resubscribe_callback is not None and self._async_loop is not None:
            # Submit async callback via AsyncThreadLoop (handles cross-thread execution)
            self._async_loop.submit(self._resubscribe_callback())

    def reset(self) -> None:
        """
        Reset handler state on reconnection.

        Clears the message buffer and offset tracking to ensure clean state
        after WebSocket reconnection.
        """
        self._buffer.clear()
        self._last_offset = None

    def _debug(self, msg: str) -> None:
        logger.debug(f"<yellow>[{self.instrument}]</yellow> {msg}")

    def _info(self, msg: str) -> None:
        logger.info(f"<yellow>[{self.instrument}]</yellow> {msg}")

    def _warning(self, msg: str) -> None:
        logger.warning(f"<yellow>[{self.instrument}]</yellow> {msg}")

    def _error(self, msg: str) -> None:
        logger.error(f"<yellow>[{self.instrument}]</yellow> {msg}")
