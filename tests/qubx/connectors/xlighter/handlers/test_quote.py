"""Tests for QuoteHandler using captured samples"""

import json
from pathlib import Path

import pytest

from qubx.connectors.xlighter.handlers import QuoteHandler
from qubx.core.series import Quote


class TestQuoteHandler:
    """Test QuoteHandler with real Lighter samples"""

    @pytest.fixture
    def handler(self):
        """Create handler for BTC-USDC (market_id=0)"""
        return QuoteHandler(market_id=0)

    @pytest.fixture
    def sample_orderbook(self):
        """Load orderbook snapshot sample"""
        sample_path = (
            Path(__file__).parent.parent
            / "test_data/samples/orderbook/sample_01.json"
        )
        with open(sample_path) as f:
            data = json.load(f)
        return data["data"]

    def test_can_handle_orderbook(self, handler, sample_orderbook):
        """Test handler recognizes orderbook messages"""
        assert handler.can_handle(sample_orderbook) is True

    def test_cannot_handle_wrong_market(self, sample_orderbook):
        """Test handler rejects wrong market_id"""
        handler = QuoteHandler(market_id=99)
        assert handler.can_handle(sample_orderbook) is False

    def test_handle_orderbook_to_quote(self, handler, sample_orderbook):
        """Test extracting quote from orderbook"""
        result = handler.handle(sample_orderbook)

        assert result is not None
        assert isinstance(result, Quote)

        # Check timestamp conversion
        expected_time_ns = int(sample_orderbook["timestamp"] * 1_000_000)
        assert result.time == expected_time_ns

        # Check bid/ask values
        assert result.bid > 0
        assert result.ask > 0
        assert result.ask > result.bid  # Ask should be higher than bid

        # Check sizes
        assert result.bid_size > 0
        assert result.ask_size > 0

        # Check mid price
        mid = result.mid_price()
        assert result.bid < mid < result.ask

    def test_quote_from_update(self, handler):
        """Test extracting quote from orderbook update"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [
                    {"price": "4333.50", "size": "2.5"},
                    {"price": "4334.00", "size": "1.0"},
                ],
                "bids": [
                    {"price": "4332.50", "size": "3.0"},
                    {"price": "4331.00", "size": "2.0"},
                ],
            },
        }

        result = handler.handle(message)

        assert result is not None
        assert result.bid == 4332.50  # Best bid (highest)
        assert result.bid_size == 3.0
        assert result.ask == 4333.50  # Best ask (lowest)
        assert result.ask_size == 2.5

    def test_filters_zero_sizes(self, handler):
        """Test that zero sizes are filtered when finding best bid/ask"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [
                    {"price": "4333.00", "size": "0.0000"},  # Zero size
                    {"price": "4333.50", "size": "2.5"},  # Best ask
                ],
                "bids": [
                    {"price": "4333.00", "size": "0.0000"},  # Zero size
                    {"price": "4332.50", "size": "3.0"},  # Best bid
                ],
            },
        }

        result = handler.handle(message)

        assert result is not None
        assert result.bid == 4332.50
        assert result.ask == 4333.50

    def test_missing_bid_returns_none(self, handler):
        """Test that missing bids returns None"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [{"price": "4333.50", "size": "2.5"}],
                "bids": [],
            },
        }

        result = handler.handle(message)
        assert result is None

    def test_missing_ask_returns_none(self, handler):
        """Test that missing asks returns None"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [],
                "bids": [{"price": "4332.50", "size": "3.0"}],
            },
        }

        result = handler.handle(message)
        assert result is None

    def test_all_zero_sizes_returns_none(self, handler):
        """Test that all zero sizes returns None"""
        message = {
            "channel": "order_book:0",
            "type": "update/order_book",
            "timestamp": 1760041996048,
            "order_book": {
                "asks": [{"price": "4333.50", "size": "0.0000"}],
                "bids": [{"price": "4332.50", "size": "0.0000"}],
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
                "asks": [{"price": "4333.50", "size": "2.5"}],
                "bids": [{"price": "4332.50", "size": "3.0"}],
            },
        }

        with pytest.raises(ValueError, match="Missing timestamp"):
            handler.handle(message)

    def test_handler_stats(self, handler, sample_orderbook):
        """Test handler statistics tracking"""
        assert handler.stats["messages_processed"] == 0

        handler.handle(sample_orderbook)

        assert handler.stats["messages_processed"] == 1

    def test_multiple_samples(self, handler):
        """Test processing multiple real orderbook samples"""
        samples_dir = Path(__file__).parent.parent / "test_data/samples/orderbook"

        processed = 0
        for sample_file in sorted(samples_dir.glob("sample_*.json"))[:10]:
            with open(sample_file) as f:
                data = json.load(f)

            if handler.can_handle(data["data"]):
                result = handler.handle(data["data"])
                if result is not None:
                    processed += 1
                    assert isinstance(result, Quote)
                    assert result.bid > 0
                    assert result.ask > 0
                    assert result.ask > result.bid

        assert processed > 0, "Should process at least some samples"

    def test_quote_consistency(self, handler, sample_orderbook):
        """Test quote values are consistent with orderbook top-of-book"""
        result = handler.handle(sample_orderbook)

        assert result is not None

        # Extract asks and bids from sample
        order_book_data = sample_orderbook["order_book"]
        asks_raw = order_book_data.get("asks", [])
        bids_raw = order_book_data.get("bids", [])

        # Find actual best bid/ask from raw data
        if asks_raw:
            valid_asks = [(float(a["price"]), float(a["size"])) for a in asks_raw if float(a["size"]) > 0]
            if valid_asks:
                best_ask = min(valid_asks, key=lambda x: x[0])
                assert result.ask == best_ask[0]
                assert result.ask_size == best_ask[1]

        if bids_raw:
            valid_bids = [(float(b["price"]), float(b["size"])) for b in bids_raw if float(b["size"]) > 0]
            if valid_bids:
                best_bid = max(valid_bids, key=lambda x: x[0])
                assert result.bid == best_bid[0]
                assert result.bid_size == best_bid[1]
