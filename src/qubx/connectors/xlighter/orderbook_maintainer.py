"""
OrderBook state maintainer for Lighter exchange.

Handles initial snapshots and incremental updates to maintain current orderbook state.
Similar pattern to CCXT Pro but implemented explicitly for Lighter.
"""

from typing import Optional

import numpy as np


class OrderBookMaintainer:
    """
    Maintains orderbook state by applying snapshots and updates.

    Lighter sends:
    1. Initial snapshot (type="subscribed/order_book") - Full orderbook
    2. Updates (type="update/order_book") - Partial updates with changed levels

    This class maintains the full orderbook state and applies updates efficiently.

    Usage:
        ```python
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Process initial snapshot
        snapshot_msg = {"type": "subscribed/order_book", "order_book": {...}}
        maintainer.process_message(snapshot_msg)

        # Process updates
        update_msg = {"type": "update/order_book", "order_book": {...}}
        maintainer.process_message(update_msg)

        # Get current orderbook as Qubx OrderBook
        orderbook = maintainer.get_orderbook(timestamp_ns)
        ```
    """

    def __init__(self, market_id: int, tick_size: float):
        """
        Initialize orderbook maintainer.

        Args:
            market_id: Lighter market ID
            tick_size: Minimum price increment for this market
        """
        self.market_id = market_id
        self.tick_size = tick_size

        # State: price -> size mappings
        self._bids: dict[float, float] = {}  # {price: size}
        self._asks: dict[float, float] = {}  # {price: size}

        # Tracking
        self._initialized = False
        self._last_offset: Optional[int] = None

    @property
    def is_initialized(self) -> bool:
        """Check if orderbook has been initialized with snapshot"""
        return self._initialized

    def reset(self) -> None:
        """Reset orderbook state"""
        self._bids.clear()
        self._asks.clear()
        self._initialized = False
        self._last_offset = None

    def process_message(self, message: dict) -> bool:
        """
        Process orderbook message (snapshot or update).

        Args:
            message: Raw Lighter orderbook message

        Returns:
            True if message was processed, False if should be skipped

        Raises:
            ValueError: If message format is invalid
        """
        msg_type = message.get("type")
        if msg_type not in ["subscribed/order_book", "update/order_book"]:
            return False

        # Extract orderbook data
        order_book_data = message.get("order_book")
        if not order_book_data:
            raise ValueError("Missing order_book in message")

        # Track offset for debugging
        offset = message.get("offset")
        if offset is not None:
            self._last_offset = offset

        # Process based on type
        if msg_type == "subscribed/order_book":
            return self._process_snapshot(order_book_data)
        else:  # update/order_book
            return self._process_update(order_book_data)

    def _process_snapshot(self, order_book_data: dict) -> bool:
        """
        Process initial snapshot - replaces entire orderbook.

        Args:
            order_book_data: OrderBook data from message

        Returns:
            True if snapshot was processed
        """
        # Clear existing state
        self._bids.clear()
        self._asks.clear()

        # Load bids
        bids_raw = order_book_data.get("bids", [])
        for level in bids_raw:
            price = float(level["price"])
            size = float(level["size"])
            if size > 0:  # Only store non-zero sizes
                self._bids[price] = size

        # Load asks
        asks_raw = order_book_data.get("asks", [])
        for level in asks_raw:
            price = float(level["price"])
            size = float(level["size"])
            if size > 0:  # Only store non-zero sizes
                self._asks[price] = size

        self._initialized = True
        return True

    def _process_update(self, order_book_data: dict) -> bool:
        """
        Process incremental update - modifies existing orderbook.

        Updates contain only changed levels. Size "0.0000" indicates removal.

        Args:
            order_book_data: OrderBook data from message

        Returns:
            True if update was processed
        """
        if not self._initialized:
            # Skip updates until we have initial snapshot
            return False

        # Apply bid updates
        bids_raw = order_book_data.get("bids", [])
        for level in bids_raw:
            price = float(level["price"])
            size = float(level["size"])

            if size > 0:
                self._bids[price] = size
            elif price in self._bids:
                del self._bids[price]

        # Apply ask updates
        asks_raw = order_book_data.get("asks", [])
        for level in asks_raw:
            price = float(level["price"])
            size = float(level["size"])

            if size > 0:
                self._asks[price] = size
            elif price in self._asks:
                del self._asks[price]

        return True

    def get_raw_levels(self, timestamp_ns: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Get raw orderbook levels as 2D [price, size] arrays.

        Args:
            timestamp_ns: Timestamp in nanoseconds (unused, for compatibility)

        Returns:
            Tuple of (bids_2d, asks_2d) where each is shape (N, 2) with [price, size] rows,
            or None if not initialized or empty
        """
        if not self._initialized:
            return None

        if not self._bids or not self._asks:
            return None

        # Convert to sorted numpy arrays [price, size]
        asks = self._get_sorted_levels(self._asks, is_ask=True, max_levels=None)
        bids = self._get_sorted_levels(self._bids, is_ask=False, max_levels=None)

        if len(asks) == 0 or len(bids) == 0:
            return None

        return (bids, asks)

    def _get_sorted_levels(
        self, levels: dict[float, float], is_ask: bool, max_levels: Optional[int]
    ) -> np.ndarray:
        """
        Convert price->size dict to sorted numpy array.

        Args:
            levels: Dict of {price: size}
            is_ask: True for asks, False for bids
            max_levels: Maximum number of levels to return

        Returns:
            2D numpy array [[price, size], ...] sorted appropriately
        """
        if not levels:
            return np.array([]).reshape(0, 2)

        # Convert to list of [price, size] pairs
        level_list = [[price, size] for price, size in levels.items()]

        # Convert to numpy array
        result = np.array(level_list, dtype=np.float64)

        # Sort by price
        # Asks: ascending (lowest first), Bids: descending (highest first)
        price_col = result[:, 0]
        if is_ask:
            sort_idx = np.argsort(price_col)
        else:
            sort_idx = np.argsort(price_col)[::-1]

        sorted_result = result[sort_idx]

        # Limit to max levels if specified
        if max_levels is not None and len(sorted_result) > max_levels:
            sorted_result = sorted_result[:max_levels]

        return sorted_result

    def get_top_of_book(self) -> Optional[tuple[float, float]]:
        """
        Get top bid and ask prices.

        Returns:
            (top_bid, top_ask) tuple, or None if not initialized
        """
        if not self._initialized or not self._bids or not self._asks:
            return None

        top_bid = max(self._bids.keys())
        top_ask = min(self._asks.keys())

        return (top_bid, top_ask)

    def get_depth(self) -> tuple[int, int]:
        """
        Get current orderbook depth.

        Returns:
            (num_bid_levels, num_ask_levels) tuple
        """
        return (len(self._bids), len(self._asks))

    def __repr__(self) -> str:
        bid_depth, ask_depth = self.get_depth()
        status = "initialized" if self._initialized else "uninitialized"
        tob = self.get_top_of_book()
        tob_str = f"bid={tob[0]:.2f}, ask={tob[1]:.2f}" if tob else "empty"
        return (
            f"OrderBookMaintainer(market_id={self.market_id}, "
            f"status={status}, depth=({bid_depth}, {ask_depth}), {tob_str})"
        )
