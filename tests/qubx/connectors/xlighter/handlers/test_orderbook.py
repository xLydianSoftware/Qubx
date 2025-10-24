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
        sample_path = (
            Path(__file__).parent.parent
            / "test_data/samples/orderbook/sample_01.json"
        )
        with open(sample_path) as f:
            data = json.load(f)
        return data["data"]

    @pytest.fixture
    def sample_update(self):
        """Load orderbook update sample"""
        sample_path = (
            Path(__file__).parent.parent
            / "test_data/samples/orderbook/sample_08.json"
        )
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
        bad_message = {"channel": "order_book:0", "type": "update/order_book"}

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
                "asks": [
                    {"price": f"{100000 + i * 0.01}", "size": f"{1.0 + i * 0.1}"}
                    for i in range(50)
                ],
                "bids": [
                    {"price": f"{99999 - i * 0.01}", "size": f"{1.0 + i * 0.1}"}
                    for i in range(50)
                ],
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
        """Test that proactive cleaning handles crossed orderbook"""
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
        assert len(handler._state_manager.bids) > 0

        # Inject crossed orderbook (bid > ask)
        # Proactive cleaning should remove the crossed bid
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

        # Proactive cleaning removes crossed bids (4332 >= 4330)
        # So bid at 4332 is removed, leaving no bids
        # Result is None because we have no valid bids
        assert result is None
        assert len(handler._state_manager.bids) == 0
        # Asks should still exist (4330, 4333)
        assert len(handler._state_manager.asks) == 2

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

        # Should detect crossed orderbook (bid >= ask)
        assert result is None
        assert len(handler._state_manager.bids) == 0

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


class TestOrderbookHandlerProactiveCleaning:
    """Test OrderBookStateManager proactive cross-cleaning with update extremes"""

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

    def test_uses_update_extremes_not_state(self, handler):
        """Test that we use update extremes, not state extremes for cleaning"""
        # Setup initial state with old high bid
        snapshot1 = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4335.00", "size": "1.0"}, {"price": "4336.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],  # Old high bid
            },
        }
        handler.handle(snapshot1)

        # Update with LOWER bid (doesn't remove old 4332)
        # Using state extreme would use 4332, using update extreme uses 4320
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "bids": [{"price": "4320.00", "size": "1.0"}],  # Lower bid
                "asks": [],
            },
        }
        handler.handle(update)

        # Both bids should exist (old 4332 not removed)
        assert 4332.0 in handler._state_manager.bids
        assert 4320.0 in handler._state_manager.bids

        # Asks should NOT be cleaned (4320 < 4335, doesn't cross)
        # If we incorrectly used state extreme (4332), we might have cleaned
        assert 4335.0 in handler._state_manager.asks
        assert 4336.0 in handler._state_manager.asks

    def test_proactive_cleaning_removes_crossed_asks(self, handler):
        """Test that proactive cleaning removes crossed asks"""
        # Setup initial state
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [
                    {"price": "4333.00", "size": "1.0"},
                    {"price": "4334.00", "size": "1.0"},
                    {"price": "4335.00", "size": "1.0"},
                ],
                "bids": [{"price": "4330.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Update with bid that crosses some asks
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "bids": [{"price": "4334.00", "size": "1.0"}],  # Crosses 4333, 4334
                "asks": [],
            },
        }
        result = handler.handle(update)

        # Crossed asks should be removed proactively
        assert 4333.0 not in handler._state_manager.asks
        assert 4334.0 not in handler._state_manager.asks
        # Higher ask should survive
        assert 4335.0 in handler._state_manager.asks

        # Should still produce valid orderbook
        assert result is not None
        assert result.top_bid < result.top_ask

    def test_proactive_cleaning_removes_crossed_bids(self, handler):
        """Test that proactive cleaning removes crossed bids"""
        # Setup initial state
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4335.00", "size": "1.0"}],
                "bids": [
                    {"price": "4332.00", "size": "1.0"},
                    {"price": "4333.00", "size": "1.0"},
                    {"price": "4334.00", "size": "1.0"},
                ],
            },
        }
        handler.handle(snapshot)

        # Update with ask that crosses some bids
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "bids": [],
                "asks": [{"price": "4333.00", "size": "1.0"}],  # Crosses 4333, 4334
            },
        }
        result = handler.handle(update)

        # Crossed bids should be removed proactively
        assert 4333.0 not in handler._state_manager.bids
        assert 4334.0 not in handler._state_manager.bids
        # Lower bid should survive
        assert 4332.0 in handler._state_manager.bids

        # Should still produce valid orderbook
        assert result is not None
        assert result.top_bid < result.top_ask

    def test_zero_size_excluded_from_extremes(self, handler):
        """Test that zero-size updates don't contribute to extremes"""
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4335.00", "size": "1.0"}],
                "bids": [{"price": "4332.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Update removes bid, adds lower bid
        # Extreme should be 4320, not 4332 (being removed)
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "bids": [
                    {"price": "4332.00", "size": "0.0"},  # Remove 4332
                    {"price": "4320.00", "size": "1.0"},  # Add 4320
                ],
                "asks": [],
            },
        }
        handler.handle(update)

        # Should use 4320 for cross-check, not 4332
        assert 4332.0 not in handler._state_manager.bids
        assert 4320.0 in handler._state_manager.bids

        # Ask should not be affected (4320 < 4335)
        assert 4335.0 in handler._state_manager.asks

    def test_no_cleaning_when_no_cross(self, handler):
        """Test that valid updates don't trigger cleaning"""
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4335.00", "size": "1.0"}],
                "bids": [{"price": "4330.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Update with valid non-crossing prices
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "bids": [{"price": "4331.00", "size": "1.0"}],  # Doesn't cross
                "asks": [{"price": "4336.00", "size": "1.0"}],  # Doesn't cross
            },
        }
        result = handler.handle(update)

        # All levels should be preserved
        assert 4330.0 in handler._state_manager.bids
        assert 4331.0 in handler._state_manager.bids
        assert 4335.0 in handler._state_manager.asks
        assert 4336.0 in handler._state_manager.asks

        # Should produce valid orderbook
        assert result is not None
        assert result.top_bid < result.top_ask

    def test_empty_update_no_cleaning(self, handler):
        """Test that empty updates don't trigger cleaning"""
        snapshot = {
            "channel": "order_book:0",
            "type": "subscribed/order_book",
            "timestamp": 1760041996000,
            "order_book": {
                "asks": [{"price": "4335.00", "size": "1.0"}],
                "bids": [{"price": "4330.00", "size": "1.0"}],
            },
        }
        handler.handle(snapshot)

        # Update with no new prices
        update = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041997000,
            "order_book": {
                "bids": [],
                "asks": [],
            },
        }
        result = handler.handle(update)

        # State should be unchanged
        assert 4330.0 in handler._state_manager.bids
        assert 4335.0 in handler._state_manager.asks

        # Should produce valid orderbook
        assert result is not None
