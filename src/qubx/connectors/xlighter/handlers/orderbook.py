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

        # Get full orderbook from maintainer (returns 2D [price, size] arrays)
        raw_orderbook = self._maintainer.get_raw_levels(time_ns)
        if raw_orderbook is None:
            return None

        bids_2d, asks_2d = raw_orderbook

        # Always aggregate into uniform tick_size grid
        # This converts 2D [price, size] to 1D size arrays on uniform price grid
        orderbook = self._aggregate_orderbook(bids_2d, asks_2d, time_ns)

        return orderbook

    def _aggregate_orderbook(self, bids_2d: np.ndarray, asks_2d: np.ndarray, time_ns: int) -> OrderBook | None:
        """
        Aggregate orderbook levels into uniform tick_size grid.

        Converts arbitrary price levels into uniform grid with fixed tick_size intervals.
        This is required because OrderBook expects prices on a uniform grid.

        Tick size calculation:
        - If tick_size_pct > 0: Calculate dynamic tick as percentage of mid price
        - Otherwise: Use instrument's minimum tick_size

        Args:
            bids_2d: 2D array of [price, size] for bids, sorted descending by price
            asks_2d: 2D array of [price, size] for asks, sorted ascending by price
            time_ns: Timestamp in nanoseconds

        Returns:
            OrderBook with 1D size arrays on uniform price grid, or None if empty
        """
        if len(bids_2d) == 0 or len(asks_2d) == 0:
            return None

        # Get top of book
        top_bid = float(bids_2d[0][0])
        top_ask = float(asks_2d[0][0])

        # Calculate tick size
        if self.tick_size_pct is not None and self.tick_size_pct > 0:
            # Dynamic tick size based on percentage of mid price
            if self.instrument is None:
                raise ValueError("instrument is required when tick_size_pct is set")
            mid_price = (top_bid + top_ask) / 2.0
            raw_tick_size = max(mid_price * self.tick_size_pct / 100.0, self.instrument.tick_size)
            tick_size = self.instrument.round_price_down(raw_tick_size)
        else:
            # Use minimum tick size
            tick_size = self.tick_size

        # Determine number of levels
        levels = self.max_levels if self.max_levels is not None else 200

        # Prepare buffers for accumulation
        bids_buffer = np.zeros(levels, dtype=np.float64)
        asks_buffer = np.zeros(levels, dtype=np.float64)

        # Apply accumulation to aggregate into uniform grid
        top_bid_agg, bids_accumulated = accumulate_orderbook_levels(
            bids_2d, bids_buffer, tick_size, True, levels, False  # is_bid=True, sizes_in_quoted=False
        )
        top_ask_agg, asks_accumulated = accumulate_orderbook_levels(
            asks_2d, asks_buffer, tick_size, False, levels, False  # is_bid=False, sizes_in_quoted=False
        )

        # Make explicit copies to ensure arrays aren't modified and are C-contiguous
        bid_sizes = np.ascontiguousarray(bids_accumulated, dtype=np.float64)
        ask_sizes = np.ascontiguousarray(asks_accumulated, dtype=np.float64)

        # Create OrderBook with aggregated data
        return OrderBook(
            time=time_ns,
            top_bid=top_bid_agg,
            top_ask=top_ask_agg,
            tick_size=tick_size,
            bids=bid_sizes,
            asks=ask_sizes,
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
