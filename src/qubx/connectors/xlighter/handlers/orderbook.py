"""OrderBook handler for Lighter WebSocket messages"""

from typing import Any, Optional

import numpy as np

from qubx.core.series import OrderBook

from ..orderbook_maintainer import OrderBookMaintainer
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

    The handler uses OrderBookMaintainer to track state across messages.
    Updates with size "0.0000" indicate removal of price level.
    """

    def __init__(self, market_id: int, tick_size: float, max_levels: Optional[int] = 200):
        """
        Initialize orderbook handler with state maintainer.

        Args:
            market_id: Lighter market ID to handle
            tick_size: Minimum price increment for this market
            max_levels: Maximum number of levels to include in output (default: 200)
        """
        super().__init__()
        self.market_id = market_id
        self.tick_size = tick_size
        self.max_levels = max_levels

        # Create maintainer for this market
        self._maintainer = OrderBookMaintainer(market_id=market_id, tick_size=tick_size)

    def can_handle(self, message: dict[str, Any]) -> bool:
        """Check if message is orderbook for this market"""
        channel = message.get("channel", "")
        msg_type = message.get("type", "")

        # Check if it's an orderbook message for our market
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

        time_ns = int(timestamp_ms * 1_000_000)  # ms -> ns

        # Process message through maintainer (applies snapshot or update)
        processed = self._maintainer.process_message(message)
        if not processed:
            return None

        # Get current orderbook state
        orderbook = self._maintainer.get_orderbook(time_ns, max_levels=self.max_levels)
        return orderbook

    @property
    def is_initialized(self) -> bool:
        """Check if orderbook has received initial snapshot"""
        return self._maintainer.is_initialized

    def reset(self) -> None:
        """Reset orderbook state (useful for reconnection scenarios)"""
        self._maintainer.reset()

    def get_depth(self) -> tuple[int, int]:
        """Get current orderbook depth (num_bids, num_asks)"""
        return self._maintainer.get_depth()
