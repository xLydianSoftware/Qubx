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
    def handler(self):
        """Create handler for BTC-USDC (market_id=0)"""
        return OrderbookHandler(market_id=0, tick_size=0.01)

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

    def test_cannot_handle_wrong_market(self, sample_snapshot):
        """Test handler rejects wrong market_id"""
        handler = OrderbookHandler(market_id=99, tick_size=0.01)
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

        # Check arrays have correct structure
        assert len(result.asks) > 0
        assert len(result.bids) > 0

        # Asks should be sorted ascending
        ask_prices = [level[0] for level in result.asks]
        assert ask_prices == sorted(ask_prices)

        # Bids should be sorted descending
        bid_prices = [level[0] for level in result.bids]
        assert bid_prices == sorted(bid_prices, reverse=True)

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
        # Only non-zero levels should remain
        assert len(result.asks) == 2
        assert len(result.bids) == 1

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
        """Test that missing order_book raises ValueError"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
        }

        with pytest.raises(ValueError, match="Missing order_book"):
            handler.handle(message)

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
