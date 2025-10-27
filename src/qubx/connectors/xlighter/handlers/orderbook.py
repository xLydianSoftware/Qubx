"""OrderBook handler for Lighter WebSocket messages"""

from typing import Any

import numpy as np

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.series import OrderBook, time_as_nsec
from qubx.core.utils import recognize_time
from qubx.utils.hft.orderbook import LOB

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
    ):
        """
        Initialize orderbook handler with LOB state maintainer.

        Args:
            market_id: Lighter market ID to handle
            tick_size: Minimum price increment for this market
            max_levels: Maximum number of levels to include in output (default: 200)
            tick_size_pct: Percentage for dynamic tick sizing (e.g., 0.01 for 0.01%)
            instrument: Instrument for price rounding (required if tick_size_pct is set)
        """
        super().__init__()
        self.market_id = market_id
        self.tick_size = instrument.tick_size
        self.max_levels = max_levels
        self.tick_size_pct = tick_size_pct
        self.instrument = instrument
        self._lob = LOB(depth=max_levels)

    def can_handle(self, message: dict[str, Any]) -> bool:
        channel = message.get("channel", "")
        msg_type = message.get("type", "")
        expected_channel = f"order_book:{self.market_id}"
        is_orderbook_msg = msg_type in ["subscribed/order_book", "update/order_book"]
        return channel == expected_channel and is_orderbook_msg

    def _handle_impl(self, message: dict[str, Any]) -> OrderBook | None:
        """
        Process Lighter orderbook message and return current state.

        This handler maintains orderbook state internally by:
        1. Applying initial snapshot on first "subscribed/order_book" message
        2. Applying incremental updates from "update/order_book" messages
        3. Returning current orderbook state after each message

        Args:
            message: Raw Lighter orderbook message

        Returns:
            Current OrderBook state, or None if not yet initialized or empty

        Raises:
            ValueError: If message format is invalid
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
            logger.warning("Missing order_book in message")
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
