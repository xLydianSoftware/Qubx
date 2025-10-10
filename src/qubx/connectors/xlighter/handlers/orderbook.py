"""OrderBook handler for Lighter WebSocket messages"""

from typing import Any, Optional

import numpy as np

from qubx.core.basics import Instrument
from qubx.core.series import OrderBook
from qubx.utils.orderbook import accumulate_orderbook_levels

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

    def __init__(
        self,
        market_id: int,
        tick_size: float,
        max_levels: Optional[int] = 200,
        tick_size_pct: Optional[float] = None,
        instrument: Optional[Instrument] = None,
    ):
        """
        Initialize orderbook handler with state maintainer.

        Args:
            market_id: Lighter market ID to handle
            tick_size: Minimum price increment for this market
            max_levels: Maximum number of levels to include in output (default: 200)
            tick_size_pct: Percentage for dynamic tick sizing (e.g., 0.01 for 0.01%)
            instrument: Instrument for price rounding (required if tick_size_pct is set)
        """
        super().__init__()
        self.market_id = market_id
        self.tick_size = tick_size
        self.max_levels = max_levels
        self.tick_size_pct = tick_size_pct
        self.instrument = instrument

        # Validate parameters
        if tick_size_pct is not None and tick_size_pct > 0:
            if instrument is None:
                raise ValueError("instrument is required when tick_size_pct is set")

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
        orderbook = self._maintainer.get_orderbook(time_ns, max_levels=None)  # Get full orderbook first
        if orderbook is None:
            return None

        # Apply aggregation if tick_size_pct is specified
        if self.tick_size_pct is not None and self.tick_size_pct > 0:
            orderbook = self._aggregate_orderbook(orderbook, time_ns)

        # Apply max_levels limit if aggregation wasn't applied
        elif self.max_levels is not None:
            # Trim to max_levels
            bids = orderbook.bids[: self.max_levels] if len(orderbook.bids) > self.max_levels else orderbook.bids
            asks = orderbook.asks[: self.max_levels] if len(orderbook.asks) > self.max_levels else orderbook.asks

            if len(bids) > 0 and len(asks) > 0:
                orderbook = OrderBook(
                    time=orderbook.time,
                    top_bid=float(bids[0][0]),
                    top_ask=float(asks[0][0]),
                    tick_size=orderbook.tick_size,
                    bids=bids,
                    asks=asks,
                )

        return orderbook

    def _aggregate_orderbook(self, orderbook: OrderBook, time_ns: int) -> OrderBook | None:
        """
        Aggregate orderbook levels using percentage-based tick sizing.

        This matches CCXT's ccxt_convert_orderbook behavior:
        1. Calculate mid price from top of book
        2. Calculate dynamic tick size as percentage of mid
        3. Aggregate levels into buckets using accumulate_orderbook_levels

        Args:
            orderbook: Raw orderbook from maintainer
            time_ns: Timestamp in nanoseconds

        Returns:
            Aggregated OrderBook, or None if aggregation fails
        """
        if self.instrument is None or self.tick_size_pct is None:
            return orderbook

        # Calculate mid price
        mid_price = (orderbook.top_bid + orderbook.top_ask) / 2.0

        # Calculate dynamic tick size as percentage of mid
        raw_tick_size = max(mid_price * self.tick_size_pct / 100.0, self.instrument.tick_size)
        tick_size = self.instrument.round_price_down(raw_tick_size)

        # Determine number of levels to aggregate
        levels = self.max_levels if self.max_levels is not None else 200

        # Prepare buffers for accumulation
        bids_buffer = np.zeros(levels, dtype=np.float64)
        asks_buffer = np.zeros(levels, dtype=np.float64)

        # Apply accumulation to bids and asks
        top_bid, bids_accumulated = accumulate_orderbook_levels(
            orderbook.bids, bids_buffer, tick_size, True, levels, False  # sizes_in_quoted=False
        )
        top_ask, asks_accumulated = accumulate_orderbook_levels(
            orderbook.asks, asks_buffer, tick_size, False, levels, False  # sizes_in_quoted=False
        )

        # Convert accumulated buffers to [price, size] arrays
        # For bids: start from top_bid and go down
        bids_result = []
        for i, size in enumerate(bids_accumulated):
            if size > 0:  # Only include non-zero levels
                price = top_bid - i * tick_size
                bids_result.append([price, size])

        # For asks: start from top_ask and go up
        asks_result = []
        for i, size in enumerate(asks_accumulated):
            if size > 0:  # Only include non-zero levels
                price = top_ask + i * tick_size
                asks_result.append([price, size])

        # Check we have data
        if not bids_result or not asks_result:
            return None

        # Convert to numpy arrays
        bids_array = np.array(bids_result, dtype=np.float64)
        asks_array = np.array(asks_result, dtype=np.float64)

        # Create new OrderBook with aggregated data
        return OrderBook(
            time=time_ns,
            top_bid=float(bids_array[0][0]),
            top_ask=float(asks_array[0][0]),
            tick_size=tick_size,
            bids=bids_array,
            asks=asks_array,
        )

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
