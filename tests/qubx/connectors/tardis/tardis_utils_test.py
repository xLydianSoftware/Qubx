import numpy as np
import pandas as pd
import pytest

from qubx.connectors.tardis.utils import (
    tardis_convert_orderbook,
    tardis_convert_quote,
    tardis_convert_trade,
    tardis_parse_message,
)
from qubx.core.lookups import lookup
from qubx.core.series import OrderBook


class TestTardisOrderbookRelatedFunctions:
    @pytest.fixture
    def test_instrument(self):
        instrument = lookup.find_symbol("BITFINEX.F", "BTCUSDT")
        assert instrument is not None
        return instrument

    @pytest.fixture
    def sample_orderbook_message(self):
        return {
            "type": "book_snapshot",
            "symbol": "BTCF0:USTF0",
            "exchange": "bitfinex-derivatives",
            "name": "book_snapshot_10_1s",
            "depth": 10,
            "interval": 1000,
            "bids": [
                {"price": 82130, "amount": 0.00008},
                {"price": 82129, "amount": 0.36098271},
                {"price": 82128, "amount": 0.0761},
                {"price": 82125, "amount": 0.0786},
                {"price": 82123, "amount": 0.00008},
                {"price": 82122, "amount": 0.121816},
                {"price": 82120, "amount": 0.0609094},
                {"price": 82118, "amount": 0.00008},
                {"price": 82117, "amount": 0.4217111},
                {"price": 82115, "amount": 0.0758},
            ],
            "asks": [
                {"price": 82137, "amount": 0.2758},
                {"price": 82138, "amount": 0.27175744},
                {"price": 82139, "amount": 0.5080156},
                {"price": 82140, "amount": 0.30909064},
                {"price": 82142, "amount": 0.0608934},
                {"price": 82145, "amount": 0.87322354},
                {"price": 82147, "amount": 0.0004},
                {"price": 82148, "amount": 0.36088},
                {"price": 82151, "amount": 0.00008},
                {"price": 82153, "amount": 0.00008},
            ],
            "timestamp": "2025-04-03T19:24:44.817Z",
            "localTimestamp": "2025-04-03T19:24:44.823Z",
        }

    @pytest.fixture
    def sample_quote_message(self):
        return {
            "type": "book_snapshot",
            "symbol": "BTCF0:USTF0",
            "exchange": "bitfinex-derivatives",
            "name": "quote",
            "depth": 1,
            "interval": 0,
            "bids": [{"price": 82169, "amount": 0.00082259}],
            "asks": [{"price": 82173, "amount": 0.0832}],
            "timestamp": "2025-04-03T19:26:15.987Z",
            "localTimestamp": "2025-04-03T19:26:15.994Z",
        }

    @pytest.fixture
    def sample_trade_message(self):
        return {
            "type": "trade",
            "symbol": "BTCF0:USTF0",
            "exchange": "bitfinex-derivatives",
            "id": "1746045857",
            "price": 82163,
            "amount": 0.0004,
            "side": "buy",
            "timestamp": "2025-04-03T19:26:44.902Z",
            "localTimestamp": "2025-04-03T19:26:44.913Z",
        }

    def test_orderbook_creation(self):
        """Test basic OrderBook creation to identify issues."""
        # Convert timestamp to nanoseconds as integer
        timestamp_ns = int(pd.Timestamp("2025-04-03T19:24:44.817Z").timestamp() * 1_000_000_000)

        # Create simple bids and asks arrays
        bids = np.array([[82130.0, 0.1], [82129.0, 0.2]], dtype=np.float64)
        asks = np.array([[82137.0, 0.3], [82138.0, 0.4]], dtype=np.float64)

        try:
            # Try to create an OrderBook directly
            ob = OrderBook(
                time=timestamp_ns,
                top_bid=82130.0,
                top_ask=82137.0,
                tick_size=0.1,
                bids=bids,
                asks=asks,
            )
            assert ob is not None
            assert ob.top_bid == 82130.0
            assert ob.top_ask == 82137.0
        except Exception as e:
            pytest.fail(f"Failed to create OrderBook directly: {e}")

    def test_tardis_parse_message(self):
        message_str = '{"type":"trade","symbol":"BTCF0:USTF0","price":82163,"amount":0.0004,"side":"buy"}'
        parsed = tardis_parse_message(message_str)
        assert parsed["type"] == "trade"
        assert parsed["symbol"] == "BTCF0:USTF0"
        assert parsed["price"] == 82163
        assert parsed["amount"] == 0.0004
        assert parsed["side"] == "buy"

        # Test invalid JSON
        invalid_message = "{invalid json}"
        empty_result = tardis_parse_message(invalid_message)
        assert empty_result == {}

    def test_tardis_convert_orderbook(self, test_instrument, sample_orderbook_message):
        # Extract raw data for debugging
        bids = []
        asks = []
        for b in sample_orderbook_message["bids"]:
            bids.append((float(b["price"]), float(b["amount"])))
        for a in sample_orderbook_message["asks"]:
            asks.append((float(a["price"]), float(a["amount"])))

        raw_bids = np.array(bids, dtype=np.float64)
        raw_asks = np.array(asks, dtype=np.float64)

        print(f"Raw bids shape: {raw_bids.shape}, first bid: {raw_bids[0]}")
        print(f"Raw asks shape: {raw_asks.shape}, first ask: {raw_asks[0]}")

        # Test with default parameters
        orderbook = tardis_convert_orderbook(sample_orderbook_message, test_instrument)

        assert orderbook is not None
        assert orderbook.top_bid == 82130.0
        assert orderbook.top_ask == 82137.0
        assert orderbook.tick_size > 0

        # Check that the bids and asks have been accumulated correctly
        assert orderbook.bids.shape[0] == 50  # Default levels
        assert orderbook.asks.shape[0] == 50

        # Test with custom parameters
        custom_levels = 20
        custom_tick_pct = 0.02
        orderbook = tardis_convert_orderbook(
            sample_orderbook_message, test_instrument, levels=custom_levels, tick_size_pct=custom_tick_pct
        )

        assert orderbook is not None
        assert orderbook.bids.shape[0] == custom_levels
        assert orderbook.asks.shape[0] == custom_levels

        # Test with instrument tick size
        orderbook = tardis_convert_orderbook(sample_orderbook_message, test_instrument, tick_size_pct=0)

        assert orderbook is not None
        assert orderbook.tick_size == test_instrument.tick_size

        # Test with empty bids/asks
        empty_message = sample_orderbook_message.copy()
        empty_message["bids"] = []
        result = tardis_convert_orderbook(empty_message, test_instrument)
        assert result is None

    def test_tardis_convert_quote(self, test_instrument, sample_quote_message):
        quote = tardis_convert_quote(sample_quote_message, test_instrument)

        assert quote is not None
        assert quote.bid == 82169.0
        assert quote.ask == 82173.0
        assert quote.bid_size == 0.00082259
        assert quote.ask_size == 0.0832

        # Test with invalid message
        invalid_message = {"type": "quote", "timestamp": "2025-04-03T19:26:15.987Z"}
        result = tardis_convert_quote(invalid_message, test_instrument)
        assert result is None

        # Test with traditional format
        traditional_message = {
            "best_bid_price": 82169,
            "best_ask_price": 82173,
            "best_bid_size": 0.5,
            "best_ask_size": 0.3,
            "timestamp": "2025-04-03T19:26:15.987Z",
        }
        quote = tardis_convert_quote(traditional_message, test_instrument)
        assert quote is not None
        assert quote.bid == 82169.0
        assert quote.ask == 82173.0
        assert quote.bid_size == 0.5
        assert quote.ask_size == 0.3

    def test_tardis_convert_trade(self, test_instrument, sample_trade_message):
        # Test buy side
        trade = tardis_convert_trade(sample_trade_message, test_instrument)
        assert trade is not None
        assert trade.price == 82163.0
        assert trade.size == 0.0004
        assert trade.side == 1  # Buy side should be 1

        # Test sell side
        sell_message = sample_trade_message.copy()
        sell_message["side"] = "sell"
        sell_trade = tardis_convert_trade(sell_message, test_instrument)
        assert sell_trade is not None  # Check for None before accessing attributes
        assert sell_trade.side == -1  # Sell side should be -1

        # Test trades array format
        trades_message = {
            "trades": [
                {"price": 82160, "size": 0.1, "side": "sell", "timestamp": "2025-04-03T19:26:44.000Z"},
                {"price": 82163, "size": 0.0004, "side": "buy", "timestamp": "2025-04-03T19:26:44.902Z"},
            ],
            "timestamp": "2025-04-03T19:26:45.000Z",
        }
        trade = tardis_convert_trade(trades_message, test_instrument)
        assert trade is not None
        assert trade.price == 82163.0  # Should get the last trade
        assert trade.size == 0.0004
        assert trade.side == 1

        # Test with empty message
        empty_message = {"type": "trades", "timestamp": "2025-04-03T19:26:45.000Z"}
        result = tardis_convert_trade(empty_message, test_instrument)
        assert result is None
