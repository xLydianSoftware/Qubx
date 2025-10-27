"""Type stubs for the LOB (Limit Order Book) Cython module."""

import numpy as np
from numpy.typing import NDArray

from qubx.core.series import OrderBook

class LOB:
    """
    A high-performance Limit Order Book (LOB) implementation using Cython.

    Attributes:
        timestamp (int): The current timestamp in milliseconds.
        depth (int): The depth of the order book.
        safe_depth (int): The safe depth of the order book (may be larger than depth).
        buffer_size (int): The buffer size for the order book.
    """

    timestamp: int
    depth: int
    safe_depth: int
    buffer_size: int

    def __init__(
        self,
        timestamp: int = -1,
        bids: NDArray[np.float64] | None = None,
        asks: NDArray[np.float64] | None = None,
        depth: int = 100,
        apply_shadow_depth: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the LOB.

        Args:
            timestamp: The initial timestamp in milliseconds. Defaults to -1.
            bids: The initial bid levels as Nx2 array of [price, size]. Defaults to None.
            asks: The initial ask levels as Nx2 array of [price, size]. Defaults to None.
            depth: The depth of the order book. Defaults to 100.
            apply_shadow_depth: Whether to maintain higher levels than depth for accuracy at speed cost. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        ...

    def update(
        self,
        timestamp: int,
        bids: NDArray[np.float64] | None,
        asks: NDArray[np.float64] | None,
        is_snapshot: bool,
        is_sorted: bool = True,
    ) -> None:
        """
        Update the order book with new bid and ask levels.

        Args:
            timestamp: The new timestamp in milliseconds.
            bids: The new bid levels as Nx2 array of [price, size].
            asks: The new ask levels as Nx2 array of [price, size].
            is_snapshot: Whether the update is a snapshot (True) or delta update (False).
            is_sorted: Whether the levels are already sorted. Defaults to True.
        """
        ...

    def get_mid(self) -> float:
        """
        Get the mid price.

        Returns:
            The mid price calculated as (best_bid + best_ask) / 2.
        """
        ...

    def get_bids(self, copy: bool = False) -> NDArray[np.float64]:
        """
        Get the bid levels.

        Args:
            copy: Whether to return a copy of the array. Defaults to False.

        Returns:
            The bid levels as Nx2 array of [price, size].
        """
        ...

    def get_asks(self, copy: bool = False) -> NDArray[np.float64]:
        """
        Get the ask levels.

        Args:
            copy: Whether to return a copy of the array. Defaults to False.

        Returns:
            The ask levels as Nx2 array of [price, size].
        """
        ...

    def get_bid(self) -> float:
        """
        Get the best bid price.

        Returns:
            The best bid price, or NaN if no bids available.
        """
        ...

    def get_ask(self) -> float:
        """
        Get the best ask price.

        Returns:
            The best ask price, or NaN if no asks available.
        """
        ...

    def get_bid_sz(self) -> float:
        """
        Get the size of the best bid.

        Returns:
            The size of the best bid.
        """
        ...

    def get_ask_sz(self) -> float:
        """
        Get the size of the best ask.

        Returns:
            The size of the best ask.
        """
        ...

    def as_tuple(self, copy: bool = True) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the order book as a tuple.

        Args:
            copy: Whether to return copies of the arrays. Defaults to True.

        Returns:
            A tuple of (timestamp, bids, asks).
        """
        ...

    def as_dict(self, copy: bool = True) -> dict[str, int | NDArray[np.float64]]:
        """
        Get the order book as a dictionary.

        Args:
            copy: Whether to return copies of the arrays. Defaults to True.

        Returns:
            A dictionary with keys "ts", "b" (bids), and "a" (asks).
        """
        ...

    def get_orderbook(
        self, tick_size: float, levels: int, sizes_in_quoted: bool = False
    ) -> OrderBook | None:
        """
        Generate OrderBook from current LOB state with tick-size aggregation.

        Args:
            tick_size: Price tick size for aggregation into uniform grid.
            levels: Number of price levels to include in output.
            sizes_in_quoted: Whether sizes should be in quoted currency (price * size). Defaults to False.

        Returns:
            OrderBook object with aggregated levels, or None if state is empty or invalid.

        Note:
            For levels==1, returns best bid/ask without aggregation for performance.
            For levels>1, aggregates raw orderbook into uniform tick-size grid.
        """
        ...
