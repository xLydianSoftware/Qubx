"""Tests for TradesHandler using captured samples"""

import json
from pathlib import Path

import pytest

from qubx.connectors.xlighter.handlers import TradesHandler
from qubx.core.series import Trade


class TestTradesHandler:
    """Test TradesHandler with real Lighter samples"""

    @pytest.fixture
    def handler(self):
        """Create handler for BTC-USDC (market_id=0)"""
        return TradesHandler(market_id=0)

    @pytest.fixture
    def sample_trades(self):
        """Load trades sample with regular trades"""
        sample_path = (
            Path(__file__).parent.parent
            / "test_data/samples/trades/sample_07.json"
        )
        with open(sample_path) as f:
            data = json.load(f)
        # Note: sample_07 is for market_id=1, let's use a different sample
        return data["data"]

    @pytest.fixture
    def sample_liquidations(self):
        """Load trades sample with liquidation trades"""
        sample_path = (
            Path(__file__).parent.parent
            / "test_data/samples/trades/sample_01.json"
        )
        with open(sample_path) as f:
            data = json.load(f)
        return data["data"]

    def test_can_handle_trade_message(self, handler, sample_trades):
        """Test handler recognizes trade messages for correct market"""
        # This sample is for market_id=1, so should not handle with market_id=0 handler
        handler_1 = TradesHandler(market_id=1)
        assert handler_1.can_handle(sample_trades) is True

    def test_cannot_handle_wrong_market(self, sample_trades):
        """Test handler rejects wrong market_id"""
        handler = TradesHandler(market_id=99)
        assert handler.can_handle(sample_trades) is False

    def test_handle_regular_trades(self, sample_trades):
        """Test handling regular trades"""
        handler = TradesHandler(market_id=1)  # sample is for market_id=1
        result = handler.handle(sample_trades)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

        for trade in result:
            assert isinstance(trade, Trade)
            assert trade.time > 0
            assert trade.price > 0
            assert trade.size > 0
            assert trade.side in [0, 1]  # BUY or SELL
            assert trade.trade_id > 0

    def test_handle_liquidation_trades(self, sample_liquidations):
        """Test handling liquidation trades"""
        handler = TradesHandler(market_id=0)  # liquidations are for market_id=0
        result = handler.handle(sample_liquidations)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

        for trade in result:
            assert isinstance(trade, Trade)
            assert trade.time > 0
            assert trade.price > 0
            assert trade.size > 0

    def test_trade_side_from_is_maker_ask(self, handler):
        """Test side determination from is_maker_ask"""
        # is_maker_ask=true -> taker bought -> side=BUY (0)
        message_maker_ask_true = {
            "channel": "trade:0",
            "type": "update/trade",
            "trades": [
                {
                    "trade_id": 123,
                    "timestamp": 1760041996404,
                    "price": "4332.50",
                    "size": "1.5",
                    "is_maker_ask": True,
                }
            ],
        }

        result = handler.handle(message_maker_ask_true)
        assert result is not None
        assert len(result) == 1
        assert result[0].side == 0  # BUY

        # is_maker_ask=false -> taker sold -> side=SELL (1)
        message_maker_ask_false = {
            "channel": "trade:0",
            "type": "update/trade",
            "trades": [
                {
                    "trade_id": 124,
                    "timestamp": 1760041996404,
                    "price": "4332.50",
                    "size": "1.5",
                    "is_maker_ask": False,
                }
            ],
        }

        result = handler.handle(message_maker_ask_false)
        assert result is not None
        assert len(result) == 1
        assert result[0].side == 1  # SELL

    def test_timestamp_conversion(self, handler):
        """Test timestamp conversion from milliseconds to nanoseconds"""
        message = {
            "channel": "trade:0",
            "type": "update/trade",
            "trades": [
                {
                    "trade_id": 123,
                    "timestamp": 1760041996404,  # milliseconds
                    "price": "4332.50",
                    "size": "1.5",
                    "is_maker_ask": True,
                }
            ],
        }

        result = handler.handle(message)
        assert result is not None
        expected_time_ns = 1760041996404 * 1_000_000
        assert result[0].time == expected_time_ns

    def test_empty_trades_returns_none(self, handler):
        """Test that message with no trades returns None"""
        message = {
            "channel": "trade:0",
            "type": "update/trade",
            "trades": [],
            "liquidation_trades": None,
        }

        result = handler.handle(message)
        assert result is None

    def test_handles_both_trade_types(self, handler):
        """Test handling message with both regular and liquidation trades"""
        message = {
            "channel": "trade:0",
            "type": "update/trade",
            "trades": [
                {
                    "trade_id": 123,
                    "timestamp": 1760041996404,
                    "price": "4332.50",
                    "size": "1.5",
                    "is_maker_ask": True,
                }
            ],
            "liquidation_trades": [
                {
                    "trade_id": 124,
                    "timestamp": 1760041996405,
                    "price": "4332.60",
                    "size": "2.5",
                    "is_maker_ask": False,
                }
            ],
        }

        result = handler.handle(message)
        assert result is not None
        assert len(result) == 2

    def test_handler_stats(self, handler, sample_liquidations):
        """Test handler statistics tracking"""
        assert handler.stats["messages_processed"] == 0

        handler.handle(sample_liquidations)

        assert handler.stats["messages_processed"] == 1

    def test_multiple_samples(self):
        """Test processing multiple real samples"""
        samples_dir = Path(__file__).parent.parent / "test_data/samples/trades"

        # Try both market IDs
        for market_id in [0, 1]:
            handler = TradesHandler(market_id=market_id)
            processed = 0

            for sample_file in sorted(samples_dir.glob("sample_*.json"))[:5]:
                with open(sample_file) as f:
                    data = json.load(f)

                if handler.can_handle(data["data"]):
                    result = handler.handle(data["data"])
                    if result is not None:
                        processed += 1
                        assert all(isinstance(t, Trade) for t in result)

            if processed > 0:
                assert processed > 0, f"Should process samples for market_id={market_id}"
