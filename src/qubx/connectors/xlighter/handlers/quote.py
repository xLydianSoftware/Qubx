"""Quote handler for Lighter WebSocket messages"""

from typing import Any

from qubx.core.series import Quote

from .base import BaseHandler


class QuoteHandler(BaseHandler[Quote]):
    """
    Handler for deriving quotes from Lighter orderbook messages.

    Extracts top-of-book (best bid/ask) from orderbook updates to create Quote objects.

    Lighter orderbook format:
    ```json
    {
      "channel": "order_book:0",
      "timestamp": 1760041996048,
      "order_book": {
        "asks": [{"price": "4332.75", "size": "0.6998"}, ...],
        "bids": [{"price": "4332.50", "size": "1.2345"}, ...]
      }
    }
    ```

    Quote will contain:
    - time: timestamp in nanoseconds
    - bid: best bid price
    - ask: best ask price
    - bid_size: best bid size
    - ask_size: best ask size
    """

    def __init__(self, market_id: int):
        """
        Initialize quote handler.

        Args:
            market_id: Lighter market ID to handle
        """
        super().__init__()
        self.market_id = market_id

    def can_handle(self, message: dict[str, Any]) -> bool:
        """Check if message is orderbook for this market"""
        channel = message.get("channel", "")
        msg_type = message.get("type", "")

        # Check if it's an orderbook message for our market
        expected_channel = f"order_book:{self.market_id}"
        is_orderbook_msg = msg_type in ["subscribed/order_book", "update/order_book"]

        return channel == expected_channel and is_orderbook_msg

    def _handle_impl(self, message: dict[str, Any]) -> Quote | None:
        """
        Extract quote from Lighter orderbook message.

        Args:
            message: Raw Lighter orderbook message

        Returns:
            Quote object with best bid/ask, or None if incomplete

        Raises:
            ValueError: If message format is invalid
        """
        # Extract timestamp (milliseconds) and convert to nanoseconds
        timestamp_ms = message.get("timestamp")
        if timestamp_ms is None:
            raise ValueError("Missing timestamp in orderbook message")

        time_ns = int(timestamp_ms * 1_000_000)  # ms -> ns

        # Extract orderbook data
        order_book_data = message.get("order_book")
        if not order_book_data:
            raise ValueError("Missing order_book in message")

        # Parse asks and bids
        asks_raw = order_book_data.get("asks", [])
        bids_raw = order_book_data.get("bids", [])

        # Find best bid and ask (filtering out zero sizes)
        best_bid = self._find_best_bid(bids_raw)
        best_ask = self._find_best_ask(asks_raw)

        # Skip if we don't have both sides
        if best_bid is None or best_ask is None:
            return None

        bid_price, bid_size = best_bid
        ask_price, ask_size = best_ask

        # Create Quote
        return Quote(
            time=time_ns,
            bid=bid_price,
            ask=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
        )

    def _find_best_bid(self, bids: list[dict]) -> tuple[float, float] | None:
        """
        Find best (highest) bid with non-zero size.

        Args:
            bids: List of {"price": str, "size": str} dicts

        Returns:
            (price, size) tuple, or None if no valid bids
        """
        if not bids:
            return None

        # Find highest price with non-zero size
        best = None
        best_price = 0.0

        for level in bids:
            price = float(level["price"])
            size = float(level["size"])

            if size > 0 and price > best_price:
                best_price = price
                best = (price, size)

        return best

    def _find_best_ask(self, asks: list[dict]) -> tuple[float, float] | None:
        """
        Find best (lowest) ask with non-zero size.

        Args:
            asks: List of {"price": str, "size": str} dicts

        Returns:
            (price, size) tuple, or None if no valid asks
        """
        if not asks:
            return None

        # Find lowest price with non-zero size
        best = None
        best_price = float("inf")

        for level in asks:
            price = float(level["price"])
            size = float(level["size"])

            if size > 0 and price < best_price:
                best_price = price
                best = (price, size)

        return best
