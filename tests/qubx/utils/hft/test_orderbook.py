"""Unit tests for the LOB (Limit Order Book) Cython module."""

import numpy as np

from qubx.core.series import OrderBook
from qubx.utils.hft.orderbook import LOB


class TestLOBInitialization:
    """Test LOB initialization scenarios."""

    def test_init_empty(self):
        """Test initialization with no data."""
        lob = LOB(timestamp=1000, depth=10)
        assert lob.timestamp == 1000
        assert lob.depth == 10
        assert lob.safe_depth == 10
        assert np.isnan(lob.get_bid())
        assert np.isnan(lob.get_ask())

    def test_init_with_bids_asks(self):
        """Test initialization with bid and ask data."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0], [98.0, 15.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0], [103.0, 22.0]])

        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        assert lob.timestamp == 1000
        assert lob.get_bid() == 100.0
        assert lob.get_ask() == 101.0
        assert lob.get_bid_sz() == 10.0
        assert lob.get_ask_sz() == 12.0

    def test_init_with_unsorted_data(self):
        """Test initialization with unsorted data."""
        # Bids unsorted (should be sorted descending)
        bids = np.array([[98.0, 15.0], [100.0, 10.0], [99.0, 20.0]])
        # Asks unsorted (should be sorted ascending)
        asks = np.array([[103.0, 22.0], [101.0, 12.0], [102.0, 18.0]])

        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Should be sorted correctly
        assert lob.get_bid() == 100.0
        assert lob.get_ask() == 101.0

        # Check full orderbook is sorted
        bids_result = lob.get_bids()
        asks_result = lob.get_asks()

        # Bids should be descending (first non-zero entries)
        assert bids_result[0, 0] == 100.0
        assert bids_result[1, 0] == 99.0
        assert bids_result[2, 0] == 98.0

        # Asks should be ascending
        assert asks_result[0, 0] == 101.0
        assert asks_result[1, 0] == 102.0
        assert asks_result[2, 0] == 103.0

    def test_init_with_shadow_depth(self):
        """Test initialization with shadow depth enabled."""
        lob = LOB(timestamp=1000, depth=10, apply_shadow_depth=True)
        assert lob.depth == 10
        assert lob.safe_depth == 30  # 3x depth


class TestLOBSnapshotUpdate:
    """Test snapshot updates (full orderbook replacement)."""

    def test_snapshot_update_basic(self):
        """Test basic snapshot update."""
        lob = LOB(timestamp=1000, depth=10)

        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])

        lob.update(timestamp=2000, bids=bids, asks=asks, is_snapshot=True)

        assert lob.timestamp == 2000
        assert lob.get_bid() == 100.0
        assert lob.get_ask() == 101.0

    def test_snapshot_replaces_existing_data(self):
        """Test that snapshot completely replaces existing data."""
        bids1 = np.array([[100.0, 10.0], [99.0, 20.0], [98.0, 15.0]])
        asks1 = np.array([[101.0, 12.0], [102.0, 18.0], [103.0, 22.0]])
        lob = LOB(timestamp=1000, bids=bids1, asks=asks1, depth=10)

        # New snapshot with different levels
        bids2 = np.array([[105.0, 5.0], [104.0, 8.0]])
        asks2 = np.array([[106.0, 6.0], [107.0, 9.0]])

        lob.update(timestamp=2000, bids=bids2, asks=asks2, is_snapshot=True)

        assert lob.get_bid() == 105.0
        assert lob.get_ask() == 106.0

        # Old levels should be gone
        bids_result = lob.get_bids()
        # Third level should have zero size
        assert bids_result[2, 1] == 0.0

    def test_snapshot_with_unsorted_data(self):
        """Test snapshot update with unsorted input."""
        lob = LOB(timestamp=1000, depth=10)

        bids = np.array([[98.0, 15.0], [100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[103.0, 22.0], [101.0, 12.0], [102.0, 18.0]])

        lob.update(timestamp=2000, bids=bids, asks=asks, is_snapshot=True, is_sorted=False)

        assert lob.get_bid() == 100.0
        assert lob.get_ask() == 101.0


class TestLOBDeltaUpdate:
    """Test delta updates (incremental orderbook updates)."""

    def test_delta_add_new_levels(self):
        """Test adding new price levels via delta update."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Add new bid level
        new_bids = np.array([[98.0, 15.0]])
        lob.update(timestamp=2000, bids=new_bids, asks=None, is_snapshot=False)

        bids_result = lob.get_bids()
        assert bids_result[0, 0] == 100.0  # Best bid unchanged
        assert bids_result[2, 0] == 98.0  # New level added
        assert bids_result[2, 1] == 15.0

    def test_delta_update_existing_level(self):
        """Test updating an existing price level."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Update existing bid level
        updated_bids = np.array([[99.0, 25.0]])  # Change size from 20 to 25
        lob.update(timestamp=2000, bids=updated_bids, asks=None, is_snapshot=False)

        bids_result = lob.get_bids()
        assert bids_result[1, 0] == 99.0
        assert bids_result[1, 1] == 25.0

    def test_delta_remove_level_with_zero_size(self):
        """Test removing a level by setting size to zero."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0], [98.0, 15.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Remove middle bid level
        remove_bids = np.array([[99.0, 0.0]])
        lob.update(timestamp=2000, bids=remove_bids, asks=None, is_snapshot=False)

        bids_result = lob.get_bids()
        assert bids_result[0, 0] == 100.0
        assert bids_result[1, 0] == 98.0  # 99 should be removed, 98 moves up
        assert bids_result[1, 1] == 15.0

    def test_delta_insert_better_price(self):
        """Test inserting a new best bid/ask."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Insert better bid
        new_bids = np.array([[101.0, 5.0]])  # This would cross - but LOB doesn't validate
        # Actually, let's use a proper better bid
        new_bids = np.array([[100.5, 5.0]])
        lob.update(timestamp=2000, bids=new_bids, asks=None, is_snapshot=False)

        assert lob.get_bid() == 100.5
        assert lob.get_bid_sz() == 5.0

    def test_delta_complex_merge(self):
        """Test complex delta update with multiple changes."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0], [98.0, 15.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0], [103.0, 22.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Multiple changes: add, update, remove
        updated_bids = np.array([
            [101.0, 8.0],   # New best bid
            [99.0, 0.0],    # Remove
            [97.0, 30.0],   # Add new level
        ])
        lob.update(timestamp=2000, bids=updated_bids, asks=None, is_snapshot=False)

        bids_result = lob.get_bids()
        assert bids_result[0, 0] == 101.0  # New best
        assert bids_result[1, 0] == 100.0  # Existing
        assert bids_result[2, 0] == 98.0   # 99 removed, so 98 is next
        assert bids_result[3, 0] == 97.0   # New level


class TestLOBAccessorMethods:
    """Test accessor methods."""

    def test_get_mid(self):
        """Test mid price calculation."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[102.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        assert lob.get_mid() == 101.0

    def test_as_tuple(self):
        """Test as_tuple method."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        ts, b, a = lob.as_tuple()
        assert ts == 1000
        assert b.shape == (10, 2)
        assert a.shape == (10, 2)
        assert b[0, 0] == 100.0

    def test_as_dict(self):
        """Test as_dict method."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        result = lob.as_dict()
        assert result["ts"] == 1000
        assert result["b"].shape == (10, 2)
        assert result["a"].shape == (10, 2)

    def test_get_bids_asks_copy(self):
        """Test that copy parameter works correctly."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Without copy
        b1 = lob.get_bids(copy=False)
        b2 = lob.get_bids(copy=False)
        assert b1 is not b2  # Different array objects but share memory

        # With copy
        b3 = lob.get_bids(copy=True)
        b3[0, 1] = 999.0  # Modify copy
        b4 = lob.get_bids(copy=False)
        assert b4[0, 1] == 10.0  # Original unchanged


class TestLOBEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_update(self):
        """Test update with None for both bids and asks."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Update with no changes
        lob.update(timestamp=2000, bids=None, asks=None, is_snapshot=False)

        assert lob.timestamp == 2000
        assert lob.get_bid() == 100.0
        assert lob.get_ask() == 101.0

    def test_update_only_bids(self):
        """Test updating only bids while asks remain unchanged."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        new_bids = np.array([[99.5, 15.0]])
        lob.update(timestamp=2000, bids=new_bids, asks=None, is_snapshot=False)

        assert lob.get_ask() == 101.0  # Unchanged
        assert lob.get_bids()[1, 0] == 99.5  # New bid added

    def test_update_only_asks(self):
        """Test updating only asks while bids remain unchanged."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        new_asks = np.array([[101.5, 15.0]])
        lob.update(timestamp=2000, bids=None, asks=new_asks, is_snapshot=False)

        assert lob.get_bid() == 100.0  # Unchanged
        assert lob.get_asks()[1, 0] == 101.5  # New ask added

    def test_depth_limit(self):
        """Test that orderbook respects depth limit."""
        # Create more levels than depth
        bids = np.array([[100 - i, 10.0] for i in range(20)])
        asks = np.array([[101 + i, 10.0] for i in range(20)])

        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=5)

        bids_result = lob.get_bids()
        asks_result = lob.get_asks()

        # Should only return 5 levels
        assert bids_result.shape[0] == 5
        assert asks_result.shape[0] == 5

        # Check best levels are correct
        assert bids_result[0, 0] == 100.0
        assert asks_result[0, 0] == 101.0

    def test_large_orderbook(self):
        """Test with a larger orderbook."""
        depth = 100
        bids = np.array([[1000 - i * 0.1, 10.0 + i] for i in range(depth)])
        asks = np.array([[1001 + i * 0.1, 10.0 + i] for i in range(depth)])

        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=depth)

        assert lob.get_bid() == 1000.0
        assert lob.get_ask() == 1001.0
        assert abs(lob.get_mid() - 1000.5) < 0.01


class TestLOBPerformance:
    """Performance-related tests (not actual benchmarks, just sanity checks)."""

    def test_repeated_delta_updates(self):
        """Test that repeated delta updates don't cause issues."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        # Perform many delta updates
        for i in range(100):
            new_bids = np.array([[100.0 - i * 0.01, 10.0 + i]])
            lob.update(timestamp=2000 + i, bids=new_bids, asks=None, is_snapshot=False)

        # Should still be functional
        assert lob.timestamp == 2000 + 99
        assert lob.get_bid() == 100.0

    def test_alternating_snapshots_and_deltas(self):
        """Test alternating between snapshots and delta updates."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        for i in range(20):
            new_bids = np.array([[100.0 + i, 10.0]])
            is_snapshot = (i % 2 == 0)
            lob.update(
                timestamp=2000 + i,
                bids=new_bids,
                asks=None,
                is_snapshot=is_snapshot,
            )

        # Should handle both update types
        assert lob.get_bid() == 100.0 + 19


class TestLOBGetOrderBook:
    """Test LOB.get_orderbook() method for generating OrderBook objects."""

    def test_get_orderbook_basic(self):
        """Test basic get_orderbook functionality."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0], [98.0, 15.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0], [103.0, 22.0]])
        lob = LOB(timestamp=1000000000000, bids=bids, asks=asks, depth=10)  # Use nanoseconds

        ob = lob.get_orderbook(tick_size=1.0, levels=3)

        assert isinstance(ob, OrderBook)
        assert ob.time == 1000000000000
        assert ob.tick_size == 1.0
        assert ob.top_bid == 100.0
        assert ob.top_ask == 101.0
        assert len(ob.bids) == 3
        assert len(ob.asks) == 3

    def test_get_orderbook_depth_1(self):
        """Test depth==1 optimization (no aggregation)."""
        bids = np.array([[100.0, 10.0], [99.0, 20.0]])
        asks = np.array([[101.0, 12.0], [102.0, 18.0]])
        lob = LOB(timestamp=2000000000000, bids=bids, asks=asks, depth=10)

        ob = lob.get_orderbook(tick_size=0.5, levels=1)

        assert isinstance(ob, OrderBook)
        assert ob.top_bid == 100.0
        assert ob.top_ask == 101.0
        assert len(ob.bids) == 1
        assert len(ob.asks) == 1
        assert ob.bids[0] == 10.0  # Best bid size
        assert ob.asks[0] == 12.0  # Best ask size

    def test_get_orderbook_empty_returns_none(self):
        """Test that get_orderbook returns None for empty LOB."""
        lob = LOB(timestamp=1000, depth=10)

        ob = lob.get_orderbook(tick_size=1.0, levels=5)

        assert ob is None

    def test_get_orderbook_crossed_returns_none(self):
        """Test that get_orderbook returns None for crossed orderbook."""
        # Artificially create crossed orderbook (should not happen in practice)
        bids = np.array([[102.0, 10.0]])  # Bid higher than ask
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000, bids=bids, asks=asks, depth=10)

        ob = lob.get_orderbook(tick_size=1.0, levels=5)

        assert ob is None

    def test_get_orderbook_with_aggregation(self):
        """Test tick-size aggregation with levels > 1."""
        # Create orderbook with fractional prices that need aggregation
        bids = np.array([
            [100.05, 5.0],
            [100.02, 3.0],
            [99.98, 7.0],
            [99.50, 10.0],
        ])
        asks = np.array([
            [100.12, 6.0],
            [100.18, 4.0],
            [100.55, 8.0],
            [101.00, 12.0],
        ])
        lob = LOB(timestamp=3000000000000, bids=bids, asks=asks, depth=10)

        # Aggregate with 0.5 tick size
        ob = lob.get_orderbook(tick_size=0.5, levels=4)

        assert isinstance(ob, OrderBook)
        # Aggregation should combine levels within same tick bucket
        assert len(ob.bids) == 4
        assert len(ob.asks) == 4
        # Check that aggregation happened (some levels combined)
        assert ob.bids[0] > 0  # First bucket has size
        assert ob.top_bid == 100.05  # Best price preserved

    def test_get_orderbook_different_tick_sizes(self):
        """Test get_orderbook with different tick sizes."""
        bids = np.array([[100.0, 10.0], [99.5, 20.0], [99.0, 15.0]])
        asks = np.array([[101.0, 12.0], [101.5, 18.0], [102.0, 22.0]])
        lob = LOB(timestamp=4000000000000, bids=bids, asks=asks, depth=10)

        # Small tick size
        ob1 = lob.get_orderbook(tick_size=0.1, levels=3)
        assert ob1.tick_size == 0.1

        # Large tick size
        ob2 = lob.get_orderbook(tick_size=1.0, levels=3)
        assert ob2.tick_size == 1.0

        # Both should have valid orderbooks
        assert ob1.top_bid == 100.0
        assert ob2.top_bid == 100.0

    def test_get_orderbook_with_sizes_in_quoted(self):
        """Test sizes_in_quoted parameter."""
        bids = np.array([[100.0, 1.0], [99.0, 2.0]])
        asks = np.array([[101.0, 1.0], [102.0, 2.0]])
        lob = LOB(timestamp=5000000000000, bids=bids, asks=asks, depth=10)

        # Without sizes_in_quoted
        ob1 = lob.get_orderbook(tick_size=1.0, levels=2, sizes_in_quoted=False)
        assert ob1.bids[0] == 1.0  # Base size

        # With sizes_in_quoted (price * size)
        ob2 = lob.get_orderbook(tick_size=1.0, levels=2, sizes_in_quoted=True)
        # Should be aggregated as price * size
        assert ob2.bids[0] >= 100.0  # At least 100 * 1.0

    def test_get_orderbook_preserves_timestamp(self):
        """Test that OrderBook preserves LOB timestamp."""
        timestamp_ns = 123456789000000  # Example nanosecond timestamp
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=timestamp_ns, bids=bids, asks=asks, depth=10)

        ob = lob.get_orderbook(tick_size=1.0, levels=1)

        assert ob.time == timestamp_ns

    def test_get_orderbook_respects_levels_limit(self):
        """Test that get_orderbook respects the levels parameter."""
        bids = np.array([[100.0 - i, 10.0] for i in range(20)])
        asks = np.array([[101.0 + i, 10.0] for i in range(20)])
        lob = LOB(timestamp=6000000000000, bids=bids, asks=asks, depth=100)

        # Request only 5 levels
        ob = lob.get_orderbook(tick_size=1.0, levels=5)

        assert len(ob.bids) == 5
        assert len(ob.asks) == 5

    def test_get_orderbook_after_updates(self):
        """Test get_orderbook after delta updates."""
        bids = np.array([[100.0, 10.0]])
        asks = np.array([[101.0, 12.0]])
        lob = LOB(timestamp=1000000000000, bids=bids, asks=asks, depth=10)

        # Initial orderbook
        ob1 = lob.get_orderbook(tick_size=1.0, levels=3)
        assert ob1.top_bid == 100.0

        # Update with new levels
        new_bids = np.array([[100.5, 15.0]])
        lob.update(timestamp=2000000000000, bids=new_bids, asks=None, is_snapshot=False)

        # Get updated orderbook
        ob2 = lob.get_orderbook(tick_size=1.0, levels=3)
        assert ob2.top_bid == 100.5
        assert ob2.time == 2000000000000
