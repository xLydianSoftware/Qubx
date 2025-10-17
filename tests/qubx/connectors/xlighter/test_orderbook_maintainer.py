"""Tests for OrderBookMaintainer"""
import pytest

from qubx.connectors.xlighter.orderbook_maintainer import OrderBookMaintainer


class TestOrderBookMaintainer:
    """Test OrderBookMaintainer functionality"""

    def test_initialization(self):
        """Test basic initialization"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        assert maintainer.market_id == 0
        assert maintainer.tick_size == 0.01
        assert not maintainer.is_initialized
        assert maintainer.get_depth() == (0, 0)
        assert maintainer.get_top_of_book() is None

    def test_process_snapshot(self):
        """Test processing initial snapshot"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        snapshot_msg = {
            "type": "subscribed/order_book",
            "channel": "order_book:0",
            "timestamp": 1760041996048,
            "offset": 995816,
            "order_book": {
                "code": 0,
                "asks": [
                    {"price": "100.00", "size": "1.0"},
                    {"price": "100.01", "size": "2.0"},
                    {"price": "100.02", "size": "3.0"},
                ],
                "bids": [
                    {"price": "99.99", "size": "1.5"},
                    {"price": "99.98", "size": "2.5"},
                    {"price": "99.97", "size": "3.5"},
                ],
            },
        }

        result = maintainer.process_message(snapshot_msg)

        assert result is True
        assert maintainer.is_initialized
        assert maintainer.get_depth() == (3, 3)  # 3 bids, 3 asks

        tob = maintainer.get_top_of_book()
        assert tob is not None
        assert tob[0] == 99.99  # top bid
        assert tob[1] == 100.00  # top ask

    def test_process_update_before_snapshot(self):
        """Test that updates are skipped before snapshot"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        update_msg = {
            "type": "update/order_book",
            "channel": "order_book:0",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [],
            },
        }

        result = maintainer.process_message(update_msg)

        assert result is False
        assert not maintainer.is_initialized

    def test_process_update_add_level(self):
        """Test adding a new price level via update"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Initial snapshot
        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        # Update: add new ask level
        update_msg = {
            "type": "update/order_book",
            "order_book": {
                "asks": [{"price": "100.01", "size": "2.0"}],
                "bids": [],
            },
        }
        result = maintainer.process_message(update_msg)

        assert result is True
        assert maintainer.get_depth() == (1, 2)  # 1 bid, 2 asks

    def test_process_update_modify_level(self):
        """Test modifying existing price level via update"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Initial snapshot
        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        # Update: modify ask size
        update_msg = {
            "type": "update/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "5.0"}],  # Changed size
                "bids": [],
            },
        }
        maintainer.process_message(update_msg)

        # Get orderbook and check
        ob = maintainer.get_orderbook(timestamp_ns=1000000000)
        assert ob is not None
        assert ob.asks[0][0] == 100.00
        assert ob.asks[0][1] == 5.0  # Size should be updated

    def test_process_update_remove_level(self):
        """Test removing price level via update (size = 0)"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Initial snapshot
        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [
                    {"price": "100.00", "size": "1.0"},
                    {"price": "100.01", "size": "2.0"},
                ],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        assert maintainer.get_depth() == (1, 2)

        # Update: remove one ask level (size = 0)
        update_msg = {
            "type": "update/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "0.0"}],  # Remove this level
                "bids": [],
            },
        }
        maintainer.process_message(update_msg)

        assert maintainer.get_depth() == (1, 1)  # One ask removed

    def test_get_orderbook(self):
        """Test getting OrderBook object"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [
                    {"price": "100.00", "size": "1.0"},
                    {"price": "100.01", "size": "2.0"},
                ],
                "bids": [
                    {"price": "99.99", "size": "1.5"},
                    {"price": "99.98", "size": "2.5"},
                ],
            },
        }
        maintainer.process_message(snapshot_msg)

        timestamp_ns = 1760041996048 * 1_000_000
        ob = maintainer.get_orderbook(timestamp_ns)

        assert ob is not None
        assert ob.time == timestamp_ns
        assert ob.top_bid == 99.99
        assert ob.top_ask == 100.00
        assert ob.tick_size == 0.01
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2

        # Check sorting: bids descending, asks ascending
        assert ob.bids[0][0] == 99.99  # Highest bid first
        assert ob.bids[1][0] == 99.98
        assert ob.asks[0][0] == 100.00  # Lowest ask first
        assert ob.asks[1][0] == 100.01

    def test_get_orderbook_with_max_levels(self):
        """Test limiting orderbook levels"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [
                    {"price": "100.00", "size": "1.0"},
                    {"price": "100.01", "size": "2.0"},
                    {"price": "100.02", "size": "3.0"},
                    {"price": "100.03", "size": "4.0"},
                    {"price": "100.04", "size": "5.0"},
                ],
                "bids": [
                    {"price": "99.99", "size": "1.0"},
                    {"price": "99.98", "size": "2.0"},
                    {"price": "99.97", "size": "3.0"},
                    {"price": "99.96", "size": "4.0"},
                    {"price": "99.95", "size": "5.0"},
                ],
            },
        }
        maintainer.process_message(snapshot_msg)

        # Get with max 2 levels
        ob = maintainer.get_orderbook(timestamp_ns=1000000000, max_levels=2)

        assert ob is not None
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        assert ob.bids[0][0] == 99.99  # Top 2 bids
        assert ob.bids[1][0] == 99.98
        assert ob.asks[0][0] == 100.00  # Top 2 asks
        assert ob.asks[1][0] == 100.01

    def test_get_orderbook_before_initialization(self):
        """Test that get_orderbook returns None before snapshot"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        ob = maintainer.get_orderbook(timestamp_ns=1000000000)
        assert ob is None

    def test_get_orderbook_empty_sides(self):
        """Test that get_orderbook returns None if one side is empty"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Snapshot with only bids (no asks)
        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        ob = maintainer.get_orderbook(timestamp_ns=1000000000)
        assert ob is None  # Can't create orderbook with only one side

    def test_reset(self):
        """Test resetting orderbook state"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Initialize with snapshot
        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        assert maintainer.is_initialized
        assert maintainer.get_depth() == (1, 1)

        # Reset
        maintainer.reset()

        assert not maintainer.is_initialized
        assert maintainer.get_depth() == (0, 0)
        assert maintainer.get_top_of_book() is None

    def test_snapshot_replaces_previous_state(self):
        """Test that new snapshot replaces previous orderbook"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # First snapshot
        snapshot1 = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot1)
        assert maintainer.get_depth() == (1, 1)

        # Second snapshot (replaces first)
        snapshot2 = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [
                    {"price": "200.00", "size": "2.0"},
                    {"price": "200.01", "size": "3.0"},
                ],
                "bids": [
                    {"price": "199.99", "size": "2.0"},
                    {"price": "199.98", "size": "3.0"},
                    {"price": "199.97", "size": "4.0"},
                ],
            },
        }
        maintainer.process_message(snapshot2)
        assert maintainer.get_depth() == (3, 2)  # New state

        tob = maintainer.get_top_of_book()
        assert tob[0] == 199.99  # New top bid
        assert tob[1] == 200.00  # New top ask

    def test_filter_zero_sizes_in_snapshot(self):
        """Test that zero sizes are filtered out in snapshot"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [
                    {"price": "100.00", "size": "1.0"},
                    {"price": "100.01", "size": "0.0"},  # Should be filtered
                ],
                "bids": [
                    {"price": "99.99", "size": "0.0"},  # Should be filtered
                    {"price": "99.98", "size": "1.0"},
                ],
            },
        }
        maintainer.process_message(snapshot_msg)

        assert maintainer.get_depth() == (1, 1)  # Only non-zero levels

    def test_multiple_updates_sequence(self):
        """Test realistic sequence of snapshot + multiple updates"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Initial snapshot
        snapshot_msg = {
            "type": "subscribed/order_book",
            "offset": 100,
            "order_book": {
                "asks": [{"price": "100.00", "size": "10.0"}],
                "bids": [{"price": "99.99", "size": "10.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        # Update 1: Modify bid
        update1 = {
            "type": "update/order_book",
            "offset": 101,
            "order_book": {
                "bids": [{"price": "99.99", "size": "15.0"}],  # Increased size
                "asks": [],
            },
        }
        maintainer.process_message(update1)

        ob = maintainer.get_orderbook(1000000000)
        assert ob.bids[0][1] == 15.0  # Updated size

        # Update 2: Add new level
        update2 = {
            "type": "update/order_book",
            "offset": 102,
            "order_book": {
                "bids": [{"price": "99.98", "size": "5.0"}],  # New level
                "asks": [],
            },
        }
        maintainer.process_message(update2)

        assert maintainer.get_depth() == (2, 1)

        # Update 3: Remove level
        update3 = {
            "type": "update/order_book",
            "offset": 103,
            "order_book": {
                "bids": [{"price": "99.98", "size": "0.0"}],  # Remove
                "asks": [],
            },
        }
        maintainer.process_message(update3)

        assert maintainer.get_depth() == (1, 1)


class TestOrderBookMaintainerEdgeCases:
    """Test edge cases and error handling"""

    def test_missing_order_book_field(self):
        """Test handling of message without order_book field"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        invalid_msg = {
            "type": "subscribed/order_book",
            "channel": "order_book:0",
            # Missing "order_book" field
        }

        with pytest.raises(ValueError, match="Missing order_book"):
            maintainer.process_message(invalid_msg)

    def test_unknown_message_type(self):
        """Test that unknown message types are skipped"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        unknown_msg = {
            "type": "unknown_type",
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [],
            },
        }

        result = maintainer.process_message(unknown_msg)
        assert result is False

    def test_empty_orderbook_snapshot(self):
        """Test snapshot with empty bids and asks"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        empty_snapshot = {
            "type": "subscribed/order_book",
            "order_book": {"asks": [], "bids": []},
        }

        maintainer.process_message(empty_snapshot)

        assert maintainer.is_initialized  # Marked as initialized
        assert maintainer.get_depth() == (0, 0)
        assert maintainer.get_orderbook(1000000000) is None  # But can't create OB

    def test_repr(self):
        """Test string representation"""
        maintainer = OrderBookMaintainer(market_id=0, tick_size=0.01)

        # Before initialization
        repr_before = repr(maintainer)
        assert "uninitialized" in repr_before
        assert "market_id=0" in repr_before

        # After initialization
        snapshot_msg = {
            "type": "subscribed/order_book",
            "order_book": {
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.99", "size": "1.0"}],
            },
        }
        maintainer.process_message(snapshot_msg)

        repr_after = repr(maintainer)
        assert "initialized" in repr_after
        assert "bid=99.99" in repr_after
        assert "ask=100.00" in repr_after
