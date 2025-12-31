# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as np

from qubx.core.series cimport OrderBook
from qubx.utils.hft.numba_utils import argsort
from qubx.utils.orderbook import accumulate_orderbook_levels

cdef double[:, :] two_d_float_array(int depth):
    cdef np.ndarray arr = np.zeros((depth, 2), dtype=np.float64)
    return arr

cdef class LOB:
    """
    A class representing a Limit Order Book (LOB).

    Attributes:
        timestamp (np.int64_t): The current timestamp.
        depth (int): The depth of the order book.
        safe_depth (int): The safe depth of the order book.
        bids (double[:, :]): The bid levels.
        asks (double[:, :]): The ask levels.
        timestamps_buffer (RingBuffer): The buffer for timestamps.
        bids_buffer (RingBuffer): The buffer for bid levels.
        asks_buffer (RingBuffer): The buffer for ask levels.
        mids_buffer (RingBuffer): The buffer for mid prices.
    """
    cdef:
        public np.int64_t timestamp
        public int depth, safe_depth, buffer_size
        public double[:, :] bids, asks, bids_cache, asks_cache

    def __init__(
        self,
        np.int64_t timestamp=-1,
        double[:, :] bids=None,
        double[:, :] asks=None,
        int depth=100,
        bint apply_shadow_depth=False,
        **kwargs
    ):
        """
        Initialize the LOB.

        Args:
            timestamp (np.int64_t, optional): The initial timestamp in ms. Defaults to -1.
            bids (double[:, :], optional): The initial bid levels. Defaults to None.
            asks (double[:, :], optional): The initial ask levels. Defaults to None.
            depth (int, optional): The depth of the order book. Defaults to 100.
            apply_shadow_depth (bint, optional): Whether to maintain higher levels than depth for accuracy at speed cost. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self.timestamp = timestamp 
        self.depth = depth 
        self.safe_depth = depth*3 if apply_shadow_depth else depth

        self.bids = two_d_float_array(self.safe_depth)
        self.asks = two_d_float_array(self.safe_depth)
        self.bids_cache = two_d_float_array(self.safe_depth)
        self.asks_cache = two_d_float_array(self.safe_depth)
        
        cdef int bid_idx, ask_idx, i
        
        if bids is not None:
            bids_np = np.asarray(bids)
            bids_np = bids_np[argsort(bids_np[:,0])][::-1]
            bid_idx = min(bids_np.shape[0], self.safe_depth)
            for i in range(bid_idx):
                self.bids[i, 0] = bids_np[i, 0]
                self.bids[i, 1] = bids_np[i, 1]
            self.bids[bid_idx:, 1] = 0
        
        if asks is not None:            
            asks_np = np.asarray(asks)
            asks_np = asks_np[argsort(asks_np[:,0])]
            ask_idx = min(asks_np.shape[0], self.safe_depth)
            for i in range(ask_idx):
                self.asks[i, 0] = asks_np[i, 0]
                self.asks[i, 1] = asks_np[i, 1]
            self.asks[ask_idx:, 1] = 0

    def update(
        self, np.int64_t timestamp, double[:, :] bids, double[:, :] asks, bint is_snapshot, bint is_sorted=True
    ):
        """
        Update the order book with new bid and ask levels.

        Args:
            timestamp (np.int64_t): The new timestamp.
            bids (double[:, :]): The new bid levels.
            asks (double[:, :]): The new ask levels.
            is_snapshot (bint): Whether the update is a snapshot or delta update.
            is_sorted (bint, optional): Whether the levels are already sorted (increasing/decreasing). Defaults to True.
        """
        cdef:
            bint bids_provided = bids is not None and bids.shape[0] != 0
            bint asks_provided = asks is not None and asks.shape[0] != 0
            bint bids_is_increasing, asks_is_increasing
            int i
            int bid_idx, ask_idx
            int old_idx, new_idx, cache_idx
            double old_price, old_volume, new_price, new_volume
            double[:, :] temp_view
            np.ndarray bids_np, asks_np
       
        if is_sorted:
            if bids_provided:
                bids_is_increasing = bids[0, 0] < bids[bids.shape[0]-1, 0]
                if bids_is_increasing:
                    bids_np = np.asarray(bids)[::-1]
                    bids = bids_np
            
            if asks_provided:
                asks_is_increasing = asks[0, 0] < asks[asks.shape[0]-1, 0]
                if not asks_is_increasing:
                    asks_np = np.asarray(asks)[::-1]
                    asks = asks_np
     
        if not is_sorted:
            if bids_provided:
                bids_np = np.asarray(bids)
                bids_np = bids_np[argsort(bids_np[:,0])][::-1]
                bids = bids_np
            if asks_provided:
                asks_np = np.asarray(asks)
                asks_np = asks_np[argsort(asks_np[:,0])]
                asks = asks_np
        
        if is_snapshot:
            if bids_provided:
                bid_idx = min(bids.shape[0], self.safe_depth)
                for i in range(bid_idx):
                    self.bids[i, :] = bids[i, :]
                self.bids[bid_idx:, 1] = 0
            
            if asks_provided:
                ask_idx = min(asks.shape[0], self.safe_depth)
                for i in range(ask_idx):
                    self.asks[i, :] = asks[i, :]
                self.asks[ask_idx:, 1] = 0
        else:
            if bids_provided:
                old_idx = 0
                new_idx = 0
                cache_idx = 0
                
                while old_idx < self.safe_depth and new_idx < bids.shape[0] and cache_idx < self.safe_depth:
                    old_price = self.bids[old_idx, 0]
                    old_volume = self.bids[old_idx, 1]
                    new_price = bids[new_idx, 0]
                    new_volume = bids[new_idx, 1]

                    if old_volume == 0:
                        old_idx = self.safe_depth
                        break
                        
                    if old_price > new_price:  
                        self.bids_cache[cache_idx,:] = self.bids[old_idx,:]
                        cache_idx += 1
                        old_idx += 1
                    elif old_price < new_price: 
                        if new_volume > 0:
                            self.bids_cache[cache_idx,:] = bids[new_idx,:]
                            cache_idx += 1
                        new_idx += 1
                    else: 
                        if new_volume > 0:
                            self.bids_cache[cache_idx,:] = bids[new_idx,:]
                            cache_idx += 1
                        old_idx += 1
                        new_idx += 1
               
                while old_idx < self.safe_depth and cache_idx < self.safe_depth:
                    if self.bids[old_idx, 1] == 0:
                        break
                    self.bids_cache[cache_idx,:] = self.bids[old_idx,:]
                    old_idx += 1
                    cache_idx += 1
                
                while new_idx < bids.shape[0] and cache_idx < self.safe_depth:
                    if bids[new_idx, 1] > 0:
                        self.bids_cache[cache_idx,:] = bids[new_idx,:]
                        cache_idx += 1
                    new_idx += 1
                  
                if cache_idx < self.safe_depth:
                    self.bids_cache[cache_idx:, 1] = 0
                
                temp_view = self.bids
                self.bids = self.bids_cache
                self.bids_cache = temp_view
            
            if asks_provided:
                old_idx = 0
                new_idx = 0
                cache_idx = 0
                
                while old_idx < self.safe_depth and new_idx < asks.shape[0] and cache_idx < self.safe_depth:
                    old_price = self.asks[old_idx, 0]
                    old_volume = self.asks[old_idx, 1]
                    new_price = asks[new_idx, 0]
                    new_volume = asks[new_idx, 1]
                    
                    if old_volume == 0:
                        old_idx = self.safe_depth
                        break
                        
                    if old_price < new_price:
                        self.asks_cache[cache_idx,:] = self.asks[old_idx,:]
                        cache_idx += 1
                        old_idx += 1
                    elif old_price > new_price:
                        if new_volume > 0:
                            self.asks_cache[cache_idx,:] = asks[new_idx,:]
                            cache_idx += 1
                        new_idx += 1
                    else: 
                        if new_volume > 0:
                            self.asks_cache[cache_idx,:] = asks[new_idx,:]
                            cache_idx += 1
                        old_idx += 1
                        new_idx += 1
                        
                while old_idx < self.safe_depth and cache_idx < self.safe_depth:
                    if self.asks[old_idx, 1] == 0:
                        break
                    self.asks_cache[cache_idx,:] = self.asks[old_idx,:]
                    old_idx += 1
                    cache_idx += 1
                
                while new_idx < asks.shape[0] and cache_idx < self.safe_depth:
                    if asks[new_idx, 1] > 0:
                        self.asks_cache[cache_idx,:] = asks[new_idx,:]
                        cache_idx += 1
                    new_idx += 1
                
                if cache_idx < self.safe_depth:
                    self.asks_cache[cache_idx:, 1] = 0
                
                temp_view = self.asks
                self.asks = self.asks_cache
                self.asks_cache = temp_view
        
        self.timestamp = timestamp

    cpdef double get_mid(self):
        """
        Get the mid price.

        Returns:
            double: The mid price.
        """
        cdef double best_bid = self.get_bid()
        cdef double best_ask = self.get_ask()
        return (best_bid + best_ask) / 2

    cpdef np.ndarray get_bids(self,copy=False):
        """
        Get the bid levels.

        Returns:
            np.ndarray: The bid levels.
        """
        bids = np.asarray(self.bids[:self.depth, :])
        return bids.copy() if copy else bids

    cpdef np.ndarray get_asks(self,copy=False):
        """
        Get the ask levels.

        Returns:
            np.ndarray: The ask levels.
        """
        asks = np.asarray(self.asks[:self.depth, :])
        return asks.copy() if copy else asks

    cpdef double get_bid(self):
        """
        Get the best bid price.

        Returns:
            double: The best bid price.
        """
        return self.bids[0, 0] if self.bids[0, 1] != 0 else np.nan

    cpdef double get_ask(self):
        """
        Get the best ask price.

        Returns:
            double: The best ask price.
        """
        return self.asks[0, 0] if self.asks[0, 1] != 0 else np.nan

    cpdef double get_bid_sz(self):
        """
        Get the size of the best bid.

        Returns:
            double: The size of the best bid.
        """
        return self.bids[0, 1]

    cpdef double get_ask_sz(self):
        """
        Get the size of the best ask.

        Returns:
            double: The size of the best ask.
        """
        return self.asks[0, 1]

    cpdef tuple as_tuple(self,copy=True):
        """
        Get the order book as a tuple.

        Returns:
            tuple: The order book as a tuple (timestamp, bids, asks).
        """
        return (self.timestamp, self.get_bids(copy=copy), self.get_asks(copy=copy))

    cpdef dict as_dict(self,copy=True):
        """
        Get the order book as a dictionary.

        Returns:
            dict: The order book as a dictionary.
        """
        return {
            "ts": self.timestamp,
            "b": self.get_bids(copy=copy),
            "a": self.get_asks(copy=copy)
        }

    cpdef OrderBook get_orderbook(self, double tick_size, int levels, bint sizes_in_quoted=False):
        """
        Generate OrderBook from current LOB state with tick-size aggregation.

        Args:
            tick_size: Price tick size for aggregation into uniform grid
            levels: Number of price levels to include in output
            sizes_in_quoted: Whether sizes should be in quoted currency (price * size)

        Returns:
            OrderBook object with aggregated levels, or None if state is empty/invalid

        Note:
            For levels==1, returns best bid/ask without aggregation for performance.
            For levels>1, aggregates raw orderbook into uniform tick-size grid using
            the accumulate_orderbook_levels function.
        """
        # Check if orderbook has data
        if self.bids[0, 1] == 0 or self.asks[0, 1] == 0:
            return None

        cdef:
            double best_bid = self.bids[0, 0]
            double best_ask = self.asks[0, 0]
            int actual_levels = min(levels, self.depth)
            np.ndarray sorted_bids, sorted_asks
            np.ndarray bids_buffer, asks_buffer
            double top_bid_agg, top_ask_agg

        # Detect crossed orderbook (should not happen with clean data)
        if best_bid >= best_ask:
            return None

        # Optimization: for depth==1, skip aggregation and return best bid/ask directly
        if levels == 1:
            return OrderBook(
                time=self.timestamp,
                top_bid=best_bid,
                top_ask=best_ask,
                tick_size=tick_size,
                bids=np.array([self.bids[0, 1]], dtype=np.float64),
                asks=np.array([self.asks[0, 1]], dtype=np.float64),
            )

        # Get bids/asks as numpy arrays (slice to requested levels)
        sorted_bids = np.asarray(self.bids[:actual_levels, :], dtype=np.float64)
        sorted_asks = np.asarray(self.asks[:actual_levels, :], dtype=np.float64)

        # Allocate buffers for aggregated levels
        bids_buffer = np.zeros(actual_levels, dtype=np.float64)
        asks_buffer = np.zeros(actual_levels, dtype=np.float64)

        # Apply tick-size aggregation using numba function
        # This accumulates raw levels into uniform price grid based on tick_size
        top_bid_agg, bids_buffer = accumulate_orderbook_levels(
            sorted_bids,
            bids_buffer,
            tick_size,
            True,  # is_bid
            actual_levels,
            sizes_in_quoted,
        )

        top_ask_agg, asks_buffer = accumulate_orderbook_levels(
            sorted_asks,
            asks_buffer,
            tick_size,
            False,  # is_bid
            actual_levels,
            sizes_in_quoted,
        )

        return OrderBook(
            time=self.timestamp,
            top_bid=top_bid_agg,
            top_ask=top_ask_agg,
            tick_size=tick_size,
            bids=bids_buffer,
            asks=asks_buffer,
        )
