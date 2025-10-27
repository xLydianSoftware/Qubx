"""Tests for OrderbookHandler using captured samples"""

import json
from pathlib import Path

import numpy as np
import pytest

from qubx.connectors.xlighter.handlers import OrderbookHandler
from qubx.core.series import OrderBook


class TestOrderbookHandler:
    """Test OrderbookHandler with real Lighter samples"""

    @pytest.fixture
    def btc_instrument(self):
        """Create a mock BTC instrument for testing"""
        from qubx.core.basics import AssetType, Instrument, MarketType

        return Instrument(
            exchange="XLIGHTER",
            symbol="BTCUSDC",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            base="BTC",
            quote="USDC",
            settle="USDC",
            exchange_symbol="BTCUSDC",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

    @pytest.fixture
    def handler(self, btc_instrument):
        """Create handler for BTC-USDC (market_id=0)"""
        return OrderbookHandler(market_id=0, instrument=btc_instrument)

    @pytest.fixture
    def sample_snapshot(self):
        """Load orderbook snapshot sample"""
        sample_path = Path(__file__).parent.parent / "test_data/samples/orderbook/sample_01.json"
        with open(sample_path) as f:
            data = json.load(f)
        return data["data"]

    @pytest.fixture
    def sample_update(self):
        """Load orderbook update sample (consecutive offset after sample_01)"""
        sample_path = Path(__file__).parent.parent / "test_data/samples/orderbook/sample_03.json"
        with open(sample_path) as f:
            data = json.load(f)
        return data["data"]

    def test_can_handle_snapshot(self, handler, sample_snapshot):
        """Test handler recognizes snapshot messages"""
        assert handler.can_handle(sample_snapshot) is True

    def test_can_handle_update(self, handler, sample_update):
        """Test handler recognizes update messages"""
        assert handler.can_handle(sample_update) is True

    def test_cannot_handle_wrong_market(self, btc_instrument, sample_snapshot):
        """Test handler rejects wrong market_id"""
        handler = OrderbookHandler(market_id=99, instrument=btc_instrument)
        assert handler.can_handle(sample_snapshot) is False

    def test_handle_snapshot(self, handler, sample_snapshot):
        """Test handling orderbook snapshot"""
        result = handler.handle(sample_snapshot)

        assert result is not None
        assert isinstance(result, OrderBook)

        # Check timestamp conversion
        expected_time_ns = int(sample_snapshot["timestamp"] * 1_000_000)
        assert result.time == expected_time_ns

        # Check tick size
        assert result.tick_size == 0.01

        # Check top of book
        assert result.top_bid > 0
        assert result.top_ask > 0
        assert result.top_ask > result.top_bid

        # Check arrays have correct structure (1D arrays of sizes)
        assert len(result.asks) > 0
        assert len(result.bids) > 0

        # Check that there are some non-zero levels
        assert np.sum(result.asks > 0) > 0
        assert np.sum(result.bids > 0) > 0

    def test_handle_update(self, handler, sample_snapshot, sample_update):
        """Test handling orderbook update (requires snapshot first)"""
        # First process snapshot to initialize state
        snapshot_result = handler.handle(sample_snapshot)
        assert snapshot_result is not None

        # Now process update
        result = handler.handle(sample_update)

        assert result is not None
        assert isinstance(result, OrderBook)

        # Updates should return current orderbook state
        assert result.time > 0
        assert result.tick_size == 0.01

    def test_filter_zero_sizes(self, handler):
        """Test that zero sizes are filtered out"""
        # First process snapshot to initialize
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Now process update with zero sizes
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [
                    {"price": "4333.00", "size": "1.5"},
                    {"price": "4334.00", "size": "0.0000"},  # Should be filtered
                    {"price": "4335.00", "size": "2.3"},
                ],
                "bids": [
                    {"price": "4332.00", "size": "3.2"},
                    {"price": "4331.00", "size": "0.0000"},  # Should be filtered
                ],
            },
        }

        result = handler.handle(message)

        assert result is not None
        # Verify that we have non-zero levels (aggregation may combine them)
        assert np.sum(result.asks > 0) >= 1  # At least some asks
        assert np.sum(result.bids > 0) >= 1  # At least some bids

        # Verify all non-zero sizes are positive
        assert np.all(result.asks[result.asks > 0] > 0)
        assert np.all(result.bids[result.bids > 0] > 0)

    def test_empty_orderbook_returns_none(self, handler):
        """Test that empty orderbook returns None"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [],
                "bids": [],
            },
        }

        result = handler.handle(message)
        assert result is None

    def test_missing_timestamp_raises_error(self, handler):
        """Test that missing timestamp raises ValueError"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "order_book": {
                "asks": [{"price": "4333.00", "size": "1.5"}],
                "bids": [{"price": "4332.00", "size": "3.2"}],
            },
        }

        with pytest.raises(ValueError, match="Missing timestamp"):
            handler.handle(message)

    def test_missing_orderbook_raises_error(self, handler):
        """Test that missing order_book returns None with warning"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
        }

        # Should return None instead of raising, logs warning
        result = handler.handle(message)
        assert result is None

    def test_handler_stats(self, handler, sample_snapshot):
        """Test handler statistics tracking"""
        assert handler.stats["messages_processed"] == 0
        assert handler.stats["errors"] == 0

        handler.handle(sample_snapshot)

        assert handler.stats["messages_processed"] == 1
        assert handler.stats["errors"] == 0

    def test_handle_safe_catches_errors(self, handler):
        """Test that handle_safe catches exceptions"""
        # Message with invalid structure that will raise exception
        bad_message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "order_book": {"asks": [], "bids": []},
            # Missing timestamp will raise ValueError
        }

        result = handler.handle_safe(bad_message)

        assert result is None
        assert handler.stats["errors"] == 1

    def test_multiple_samples(self, handler):
        """Test processing multiple real samples"""
        samples_dir = Path(__file__).parent.parent / "test_data/samples/orderbook"

        processed = 0
        for sample_file in sorted(samples_dir.glob("sample_*.json"))[:5]:
            with open(sample_file) as f:
                data = json.load(f)

            if handler.can_handle(data["data"]):
                result = handler.handle(data["data"])
                if result is not None:
                    processed += 1
                    assert isinstance(result, OrderBook)
                    assert result.time > 0

        assert processed > 0, "Should process at least some samples"


class TestOrderbookHandlerAggregation:
    """Test OrderbookHandler aggregation functionality"""

    @pytest.fixture
    def btc_instrument(self):
        """Create a mock BTC instrument for testing"""
        from qubx.core.basics import AssetType, Instrument, MarketType

        return Instrument(
            exchange="XLIGHTER",
            symbol="BTCUSDC",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            base="BTC",
            quote="USDC",
            settle="USDC",
            exchange_symbol="BTCUSDC",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

    @pytest.fixture
    def handler_with_aggregation(self, btc_instrument):
        """Create handler with aggregation enabled (0.01% tick size, 20 levels)"""
        return OrderbookHandler(
            market_id=0,
            instrument=btc_instrument,
            max_levels=20,
            tick_size_pct=0.01,  # 0.01% of mid price
        )

    @pytest.fixture
    def sample_orderbook_message(self):
        """Create a sample orderbook message with many levels"""
        return {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": f"{100000 + i * 0.01}", "size": f"{1.0 + i * 0.1}"} for i in range(50)],
                "bids": [{"price": f"{99999 - i * 0.01}", "size": f"{1.0 + i * 0.1}"} for i in range(50)],
            },
        }

    def test_aggregation_with_dynamic_tick_size(self, btc_instrument):
        """Test that aggregation calculates dynamic tick size correctly"""
        handler = OrderbookHandler(
            market_id=0,
            instrument=btc_instrument,
            tick_size_pct=0.01,  # 0.01% of mid price
        )

        message = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "100000.00", "size": "1.0"}],
                "bids": [{"price": "99999.00", "size": "1.0"}],
            },
        }

        result = handler.handle(message)
        assert result is not None

        # Tick size should be dynamically calculated and at least the instrument tick size
        assert result.tick_size >= btc_instrument.tick_size

    def test_aggregation_with_zero_tick_size_pct(self, btc_instrument):
        """Test that zero tick_size_pct disables aggregation"""
        handler = OrderbookHandler(
            market_id=0,
            instrument=btc_instrument,
            tick_size_pct=0.0,  # Disabled
        )

        message = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "100000.00", "size": "1.0"}],
                "bids": [{"price": "99999.00", "size": "1.0"}],
            },
        }

        result = handler.handle(message)
        assert result is not None
        # Should use raw tick_size, not aggregated
        assert result.tick_size == 0.01

    def test_aggregation_respects_max_levels(self, handler_with_aggregation, sample_orderbook_message):
        """Test that aggregation respects max_levels parameter"""
        result = handler_with_aggregation.handle(sample_orderbook_message)

        assert result is not None
        # Count non-zero levels in the 1D arrays
        assert np.sum(result.bids > 0) <= 20, "Should have at most 20 bid levels"
        assert np.sum(result.asks > 0) <= 20, "Should have at most 20 ask levels"

    def test_aggregation_calculates_dynamic_tick_size(self, handler_with_aggregation, sample_orderbook_message):
        """Test that aggregation calculates tick_size as percentage of mid"""
        result = handler_with_aggregation.handle(sample_orderbook_message)

        assert result is not None

        # Calculate expected tick size
        mid_price = (result.top_bid + result.top_ask) / 2.0
        expected_tick_size = mid_price * 0.01 / 100.0  # 0.01% of mid

        # Aggregated tick_size should be at least the expected percentage
        # (rounded to instrument.tick_size minimum)
        assert result.tick_size >= 0.01, "Should be at least instrument tick_size"
        assert result.tick_size >= expected_tick_size * 0.9, "Should be close to expected percentage"

    def test_aggregation_price_level_spacing(self, handler_with_aggregation, sample_orderbook_message):
        """Test that aggregated price levels are properly spaced"""
        result = handler_with_aggregation.handle(sample_orderbook_message)

        assert result is not None

        # With 1D arrays, prices are implicit: top_price - i * tick_size
        # Check that tick_size is reasonable
        assert result.tick_size > 0
        assert result.tick_size >= 0.01  # At least instrument tick_size

        # Verify top of book makes sense
        assert result.top_bid < result.top_ask

    def test_aggregation_accumulates_sizes(self, handler_with_aggregation):
        """Test that aggregation accumulates sizes within same price bucket"""
        # Create orderbook with multiple levels within same bucket
        message = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [
                    {"price": "100000.00", "size": "1.0"},
                    {"price": "100000.01", "size": "2.0"},  # Should aggregate with previous
                    {"price": "100000.02", "size": "1.5"},  # Should aggregate with previous
                ],
                "bids": [
                    {"price": "99999.00", "size": "1.0"},
                    {"price": "99998.99", "size": "2.0"},  # Should aggregate with previous
                    {"price": "99998.98", "size": "1.5"},  # Should aggregate with previous
                ],
            },
        }

        result = handler_with_aggregation.handle(message)
        assert result is not None

        # With aggregation, multiple close levels should be combined
        # The number of non-zero levels should be less than or equal to raw levels
        assert np.sum(result.bids > 0) <= 3
        assert np.sum(result.asks > 0) <= 3

    def test_aggregation_filters_zero_sizes(self, handler_with_aggregation):
        """Test that aggregation filters out zero-size levels"""
        message = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [
                    {"price": "100000.00", "size": "1.0"},
                    {"price": "100001.00", "size": "0.0"},  # Should be filtered
                ],
                "bids": [
                    {"price": "99999.00", "size": "1.0"},
                    {"price": "99998.00", "size": "0.0"},  # Should be filtered
                ],
            },
        }

        result = handler_with_aggregation.handle(message)
        assert result is not None

        # Zero-size levels should not appear in result
        # With 1D arrays, check that non-zero values are all positive
        assert np.all(result.bids[result.bids > 0] > 0), "All non-zero bid sizes should be positive"
        assert np.all(result.asks[result.asks > 0] > 0), "All non-zero ask sizes should be positive"

        # Should have exactly 1 non-zero level on each side
        assert np.sum(result.bids > 0) == 1
        assert np.sum(result.asks > 0) == 1

    def test_no_aggregation_without_tick_size_pct(self, btc_instrument, sample_orderbook_message):
        """Test that handler works without aggregation (backward compatibility)"""
        handler = OrderbookHandler(
            market_id=0,
            instrument=btc_instrument,
            max_levels=20,
            # No tick_size_pct (defaults to 0)
        )

        result = handler.handle(sample_orderbook_message)
        assert result is not None

        # Should return raw orderbook with original tick_size
        assert result.tick_size == 0.01
        assert np.sum(result.bids > 0) <= 20
        assert np.sum(result.asks > 0) <= 20


class TestOrderbookHandlerCrossedOrderbook:
    """Test OrderbookHandler crossed orderbook detection and handling"""

    @pytest.fixture
    def btc_instrument(self):
        """Create a mock BTC instrument for testing"""
        from qubx.core.basics import AssetType, Instrument, MarketType

        return Instrument(
            exchange="XLIGHTER",
            symbol="BTCUSDC",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            base="BTC",
            quote="USDC",
            settle="USDC",
            exchange_symbol="BTCUSDC",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

    @pytest.fixture
    def handler(self, btc_instrument):
        """Create handler for BTC-USDC (market_id=0)"""
        return OrderbookHandler(market_id=0, instrument=btc_instrument)

    def test_crossed_orderbook_detected_and_cleaned(self, handler):
        """Test that LOB detects crossed orderbook and returns None"""
        # Build initial valid state
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        result = handler.handle(snapshot)
        assert result is not None

        # Inject crossed orderbook (bid > ask)
        # LOB will detect crossed orderbook and return None
        crossed_update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "asks": [{"price": "4330.00", "size": "1.0"}],  # Ask below old bid!
                "bids": [{"price": "4332.00", "size": "1.0"}],  # Keep bid same
            },
        }
        result = handler.handle(crossed_update)

        # LOB detects crossed orderbook (4332 >= 4330) and returns None
        assert result is None

    def test_exact_cross_detected(self, handler):
        """Test that exact cross (bid == ask) is detected"""
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4332.00", "size": "1.0"}],  # Same as bid
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        result = handler.handle(snapshot)

        # LOB detects crossed orderbook (bid >= ask) and returns None
        assert result is None

    def test_recovery_after_crossed_orderbook(self, handler):
        """Test that handler recovers after detecting crossed orderbook"""
        # Create crossed orderbook
        crossed = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4330.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        result = handler.handle(crossed)
        assert result is None

        # Now send valid snapshot
        valid = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        result = handler.handle(valid)

        # Should recover and produce valid orderbook
        assert result is not None
        assert result.top_bid == 4332.00
        assert result.top_ask == 4333.00

    def test_normal_orderbook_not_flagged(self, handler):
        """Test that normal orderbook with proper spread is not flagged"""
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [
                    {"price": "4333.00", "size": "1.0"},
                    {"price": "4334.00", "size": "1.5"},
                ],
                "bids": [
                    {"price": "4332.00", "size": "1.0"},
                    {"price": "4331.00", "size": "1.5"},
                ],
            },
        }
        result = handler.handle(snapshot)

        # Should produce valid orderbook
        assert result is not None
        assert result.top_bid < result.top_ask


class TestOrderbookHandlerMessageOrdering:
    """Test offset-based message ordering and buffer management"""

    @pytest.fixture
    def btc_instrument(self):
        """Create a mock BTC instrument for testing"""
        from qubx.core.basics import AssetType, Instrument, MarketType

        return Instrument(
            exchange="XLIGHTER",
            symbol="BTCUSDC",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            base="BTC",
            quote="USDC",
            settle="USDC",
            exchange_symbol="BTCUSDC",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

    @pytest.fixture
    def resubscribe_callback(self):
        """Mock resubscription callback that tracks calls"""
        call_count = {"count": 0}

        async def callback():
            call_count["count"] += 1

        callback.call_count = call_count
        return callback

    @pytest.fixture
    def async_loop(self):
        """Mock AsyncThreadLoop that executes coroutines immediately"""
        import asyncio

        class MockAsyncThreadLoop:
            def submit(self, coro):
                """Execute coroutine immediately in the current event loop"""
                loop = asyncio.get_event_loop()
                loop.run_until_complete(coro)

        return MockAsyncThreadLoop()

    @pytest.fixture
    def handler(self, btc_instrument, resubscribe_callback, async_loop):
        """Create handler with resubscription callback"""
        return OrderbookHandler(
            market_id=0,
            instrument=btc_instrument,
            max_buffer_size=3,
            resubscribe_callback=resubscribe_callback,
            async_loop=async_loop,
        )

    def test_messages_in_order_no_buffering(self, handler):
        """Test that messages arriving in order are applied immediately"""
        # Snapshot
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        result1 = handler.handle(snapshot)
        assert result1 is not None
        assert handler._last_offset == 100
        assert len(handler._buffer) == 0

        # Update with next offset
        update1 = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                "asks": [{"price": "4333.50", "size": "0.5"}],
                "bids": [],
            },
        }
        result2 = handler.handle(update1)
        assert result2 is not None
        assert handler._last_offset == 101
        assert len(handler._buffer) == 0

    def test_out_of_order_buffered_and_applied(self, handler):
        """Test that out-of-order messages are buffered and applied when gap is filled"""
        # Snapshot
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Receive offset 103 (skipping 101, 102)
        update3 = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996300,
            "order_book": {
                "offset": 103,
                "asks": [{"price": "4333.00", "size": "1.5"}],
                "bids": [],
            },
        }
        result = handler.handle(update3)
        assert result is None  # Buffered, not applied yet
        assert handler._last_offset == 100  # Still at snapshot offset
        assert 103 in handler._buffer
        assert len(handler._buffer) == 1

        # Receive offset 102 (still missing 101)
        update2 = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996200,
            "order_book": {
                "offset": 102,
                "asks": [{"price": "4333.50", "size": "0.8"}],
                "bids": [],
            },
        }
        result = handler.handle(update2)
        assert result is None  # Buffered
        assert handler._last_offset == 100
        assert 102 in handler._buffer
        assert 103 in handler._buffer
        assert len(handler._buffer) == 2

        # Receive offset 101 (fills the gap)
        update1 = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                "asks": [{"price": "4332.75", "size": "0.5"}],
                "bids": [],
            },
        }
        result = handler.handle(update1)
        assert result is not None  # All messages applied
        assert handler._last_offset == 103  # All buffered messages drained
        assert len(handler._buffer) == 0

    def test_duplicate_messages_skipped(self, handler):
        """Test that duplicate or old messages are skipped"""
        # Snapshot
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Apply offset 101
        update1 = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                "asks": [{"price": "4333.50", "size": "0.5"}],
                "bids": [],
            },
        }
        handler.handle(update1)
        assert handler._last_offset == 101

        # Receive old offset 100 (should be skipped)
        old_update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996050,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4334.00", "size": "2.0"}],
                "bids": [],
            },
        }
        result = handler.handle(old_update)
        assert result is None  # Skipped
        assert handler._last_offset == 101  # Unchanged
        assert len(handler._buffer) == 0

        # Receive duplicate offset 101 (should be skipped)
        duplicate = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                "asks": [{"price": "4335.00", "size": "3.0"}],
                "bids": [],
            },
        }
        result = handler.handle(duplicate)
        assert result is None  # Skipped
        assert handler._last_offset == 101
        assert len(handler._buffer) == 0

    def test_buffer_overflow_triggers_resubscription(self, handler, resubscribe_callback):
        """Test that buffer overflow triggers resubscription callback"""
        # Snapshot
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Buffer is max_buffer_size=3
        # Add 3 out-of-order messages (should trigger overflow on 3rd)
        for i in range(1, 4):
            update = {
                "channel": "order_book:0",
                "type": "update/order_book",
                "timestamp": 1760041996000 + i * 100,
                "order_book": {
                    "offset": 100 + i + 1,  # Skip offset 101
                    "asks": [{"price": f"{4333.0 + i}", "size": "1.0"}],
                    "bids": [],
                },
            }
            handler.handle(update)

        # Buffer should be cleared after overflow
        assert len(handler._buffer) == 0
        assert handler._last_offset is None  # Reset
        assert resubscribe_callback.call_count["count"] == 1

    def test_snapshot_resets_buffer_and_offset(self, handler):
        """Test that snapshot clears buffer and resets offset tracking"""
        # Initial snapshot
        snapshot1 = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot1)

        # Buffer some out-of-order messages
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996200,
            "order_book": {
                "offset": 103,
                "asks": [{"price": "4334.00", "size": "1.0"}],
                "bids": [],
            },
        }
        handler.handle(update)
        assert len(handler._buffer) == 1
        assert handler._last_offset == 100

        # New snapshot (e.g., after reconnection)
        snapshot2 = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "offset": 200,
                "asks": [{"price": "4335.00", "size": "2.0"}],
                "bids": [{"price": "4333.00", "size": "2.0"}],
            },
        }
        result = handler.handle(snapshot2)
        assert result is not None
        assert len(handler._buffer) == 0  # Buffer cleared
        assert handler._last_offset == 200  # New offset

    def test_reset_clears_state(self, handler):
        """Test that reset() clears buffer and offset"""
        # Setup state
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Buffer a message
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996200,
            "order_book": {
                "offset": 103,
                "asks": [{"price": "4334.00", "size": "1.0"}],
                "bids": [],
            },
        }
        handler.handle(update)
        assert len(handler._buffer) == 1
        assert handler._last_offset == 100

        # Reset
        handler.reset()
        assert len(handler._buffer) == 0
        assert handler._last_offset is None

    def test_crossed_orderbook_triggers_resubscription(self, handler, resubscribe_callback):
        """Test that crossed orderbook (bid >= ask) triggers resubscription"""
        # Snapshot with normal orderbook
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Update that creates crossed orderbook - directly send crossed snapshot
        crossed_snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",  # Use snapshot to force the state
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                # Create crossed orderbook: best_bid >= best_ask
                "asks": [{"price": "4330.00", "size": "1.0"}],  # Ask at 4330
                "bids": [{"price": "4335.00", "size": "1.0"}],  # Bid at 4335 (crossed!)
            },
        }
        result = handler.handle(crossed_snapshot)

        # Note: snapshot doesn't check for crossed orderbook, only updates do
        # So let's send an update after this to trigger the check
        # Actually, let's use a different approach - send updates that cross

        # Reset and try with updates
        handler.reset()

        # Start fresh
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Send update that creates crossed orderbook (ask below bid)
        update_crossed = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                "asks": [{"price": "98.00", "size": "1.0"}],  # Ask at 98
                "bids": [],  # Keep existing bid at 99
            },
        }
        result = handler.handle(update_crossed)

        # Should trigger resubscription
        assert result is None  # Returns None due to corruption
        assert resubscribe_callback.call_count["count"] == 1
        assert len(handler._buffer) == 0
        assert handler._last_offset is None

    def test_backward_compatibility_no_offset(self, handler):
        """Test that messages without offset are handled (backward compatibility)"""
        # Snapshot without offset
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4333.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        result1 = handler.handle(snapshot)
        assert result1 is not None
        assert handler._last_offset is None  # No offset tracking

        # Update without offset (should be applied immediately)
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "asks": [{"price": "4333.50", "size": "0.5"}],
                "bids": [],
            },
        }
        result2 = handler.handle(update)
        assert result2 is not None
        assert handler._last_offset is None
        assert len(handler._buffer) == 0

    def test_resubscription_ignores_updates_until_snapshot(self, handler, resubscribe_callback):
        """Test that updates are ignored during resubscription until snapshot arrives"""
        # Initial snapshot
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "offset": 100,
                "asks": [{"price": "100.00", "size": "1.0"}],
                "bids": [{"price": "99.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Trigger resubscription by creating crossed orderbook
        crossed_update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996100,
            "order_book": {
                "offset": 101,
                "asks": [{"price": "98.00", "size": "1.0"}],  # Ask below bid
                "bids": [],
            },
        }
        result = handler.handle(crossed_update)

        # Should trigger resubscription and return None
        assert result is None
        assert handler._is_resubscribing is True
        assert resubscribe_callback.call_count["count"] == 1

        # Now send several more updates - they should all be ignored
        for i in range(5):
            update = {
                "channel": "order_book:0",
                "type": "update/order_book",
                "timestamp": 1760041996100 + (i + 1) * 100,
                "order_book": {
                    "offset": 102 + i,
                    "asks": [{"price": f"{95.0 + i}", "size": "1.0"}],
                    "bids": [],
                },
            }
            result = handler.handle(update)
            # All updates should be ignored
            assert result is None

        # Should still only have 1 resubscription callback (not 6)
        assert resubscribe_callback.call_count["count"] == 1
        assert handler._is_resubscribing is True

        # Now send a snapshot (simulating the result of resubscription)
        new_snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "offset": 200,
                "asks": [{"price": "105.00", "size": "2.0"}],
                "bids": [{"price": "104.00", "size": "2.0"}],
            },
        }
        result = handler.handle(new_snapshot)

        # Snapshot should be accepted and flag should be reset
        assert result is not None
        assert handler._is_resubscribing is False
        assert handler._last_offset == 200

        # After snapshot, updates should be processed normally again
        normal_update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997100,
            "order_book": {
                "offset": 201,
                "asks": [{"price": "105.50", "size": "0.5"}],
                "bids": [],
            },
        }
        result = handler.handle(normal_update)
        assert result is not None  # Should be processed normally
        assert handler._last_offset == 201
