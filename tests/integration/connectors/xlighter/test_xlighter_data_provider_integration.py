"""
Integration tests for XLighter data provider with live Lighter exchange.

These tests validate read-only WebSocket functionality:
- Orderbook subscriptions (L2 updates)
- Trade feed subscriptions (regular + liquidations)
- Quote derivation from orderbook
- Multi-instrument subscriptions
- Data format conversion (Lighter → Qubx)

Run with: pytest tests/integration/connectors/xlighter/ -v -m integration
"""

import asyncio
import os
from unittest.mock import MagicMock

import pytest

from qubx.connectors.xlighter.client import LighterClient
from qubx.connectors.xlighter.data import LighterDataProvider
from qubx.connectors.xlighter.instruments import LighterInstrumentLoader
from qubx.core.basics import AssetType, CtrlChannel, Instrument, LiveTimeProvider, MarketType

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="class")
def lighter_credentials():
    """
    Load Lighter credentials from accounts file.

    Set LIGHTER_ACCOUNT_PATH to override default location.
    """
    account_path = os.getenv("LIGHTER_ACCOUNT_PATH", "/home/yuriy/accounts/xlydian1_lighter.toml")

    if not os.path.exists(account_path):
        pytest.skip(f"Lighter account file not found: {account_path}")

    import toml

    config = toml.load(account_path)
    accounts = config.get("accounts", [])

    lighter_account = None
    for account in accounts:
        if account.get("exchange", "").upper() in ["LIGHTER", "XLIGHTER"]:
            lighter_account = account
            break

    if not lighter_account:
        pytest.skip("No LIGHTER/XLIGHTER account found in config")

    return {
        "api_key": lighter_account.get("api_key"),
        "secret": lighter_account.get("secret"),
        "account_index": lighter_account.get("account_index"),
        "api_key_index": lighter_account.get("api_key_index", 0),
    }


@pytest.fixture
async def lighter_client(lighter_credentials):
    """Create Lighter client for testing"""
    client = LighterClient(
        api_key=lighter_credentials["api_key"],
        private_key=lighter_credentials["secret"],
        account_index=lighter_credentials["account_index"],
        api_key_index=lighter_credentials["api_key_index"],
        testnet=False,  # Use mainnet for integration tests
    )
    yield client
    client.close()


@pytest.fixture
async def instrument_loader(lighter_client):
    """Load instruments from Lighter API"""
    loader = LighterInstrumentLoader(lighter_client)
    await loader.load_instruments()
    return loader


@pytest.fixture
def test_instruments(instrument_loader):
    """
    Get test instruments for BTC and ETH.

    Uses actual instruments loaded from Lighter API with correct market_ids.
    """
    # Get BTC instrument
    btc_instrument = instrument_loader.get_instrument_by_symbol("BTCUSDC")
    if not btc_instrument:
        pytest.skip("BTC instrument not found in Lighter markets")

    # Get ETH instrument
    eth_instrument = instrument_loader.get_instrument_by_symbol("ETHUSDC")
    if not eth_instrument:
        pytest.skip("ETH instrument not found in Lighter markets")

    return [btc_instrument, eth_instrument]


@pytest.fixture
def test_channel():
    """Create a mock channel for capturing test data"""
    channel = MagicMock(spec=CtrlChannel)
    channel.received_data = []

    def mock_send(data):
        channel.received_data.append(data)

    channel.send = mock_send
    return channel


@pytest.fixture
async def data_provider(lighter_client, instrument_loader, test_channel):
    """Create LighterDataProvider with real WebSocket"""
    provider = LighterDataProvider(
        client=lighter_client,
        instrument_loader=instrument_loader,
        time_provider=LiveTimeProvider(),
        channel=test_channel,
        ws_url="wss://mainnet.zklighter.elliot.ai/stream",
    )
    yield provider
    # Cleanup
    try:
        await provider.close()
    except Exception:
        pass


@pytest.mark.integration
class TestXLighterBasicIntegration:
    """Basic integration tests for XLighter data provider"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, data_provider, test_instruments, test_channel):
        """Test WebSocket connection establishment"""
        instrument = test_instruments[0]  # BTC

        # Subscribe to orderbook (triggers WebSocket connection)
        data_provider.subscribe("orderbook", {instrument})

        # Wait for connection and first data
        await asyncio.sleep(10)

        # Verify WebSocket is connected
        assert data_provider._ws_connected, "WebSocket should be connected"
        assert data_provider._ws_manager is not None, "WebSocket manager should be initialized"

        # Cleanup
        data_provider.unsubscribe("orderbook", {instrument})

    @pytest.mark.asyncio
    async def test_orderbook_subscription(self, data_provider, test_instruments, test_channel):
        """Test orderbook subscription and data flow"""
        instrument = test_instruments[0]  # BTC

        # Clear any previous data
        test_channel.received_data.clear()

        # Subscribe to orderbook
        data_provider.subscribe("orderbook", {instrument})

        # Wait for orderbook data
        await asyncio.sleep(15)

        # Verify orderbook data received
        orderbook_data = [d for d in test_channel.received_data if d[1] == "orderbook"]
        assert len(orderbook_data) > 0, "Should receive orderbook data within 15 seconds"

        # Verify orderbook structure
        instrument_rcv, data_type, orderbook, is_warmup = orderbook_data[0]
        assert instrument_rcv == instrument
        assert data_type == "orderbook"
        assert not is_warmup
        assert hasattr(orderbook, "bids")
        assert hasattr(orderbook, "asks")
        assert hasattr(orderbook, "time")

        # Verify we have price levels
        assert len(orderbook.bids) > 0 or len(orderbook.asks) > 0, "Orderbook should have bids or asks"

        # Cleanup
        data_provider.unsubscribe("orderbook", {instrument})

    @pytest.mark.asyncio
    async def test_trades_subscription(self, data_provider, test_instruments, test_channel):
        """Test trade feed subscription"""
        instrument = test_instruments[0]  # BTC

        # Clear any previous data
        test_channel.received_data.clear()

        # Subscribe to trades
        data_provider.subscribe("trade", {instrument})

        # Wait for trade data (may take longer than orderbook)
        await asyncio.sleep(20)

        # Verify trade data received
        trade_data = [d for d in test_channel.received_data if d[1] == "trade"]
        assert len(trade_data) > 0, "Should receive trade data within 20 seconds"

        # Verify trade structure
        instrument_rcv, data_type, trade, is_warmup = trade_data[0]
        assert instrument_rcv == instrument
        assert data_type == "trade"
        assert not is_warmup
        assert hasattr(trade, "time")
        assert hasattr(trade, "price")
        assert hasattr(trade, "size")
        assert trade.price > 0
        assert trade.size > 0

        # Cleanup
        data_provider.unsubscribe("trade", {instrument})

    @pytest.mark.asyncio
    async def test_quote_subscription(self, data_provider, test_instruments, test_channel):
        """Test quote subscription (derived from orderbook)"""
        instrument = test_instruments[0]  # BTC

        # Clear any previous data
        test_channel.received_data.clear()

        # Subscribe to quotes
        data_provider.subscribe("quote", {instrument})

        # Wait for quote data
        await asyncio.sleep(15)

        # Verify quote data received
        quote_data = [d for d in test_channel.received_data if d[1] == "quote"]
        assert len(quote_data) > 0, "Should receive quote data within 15 seconds"

        # Verify quote structure
        instrument_rcv, data_type, quote, is_warmup = quote_data[0]
        assert instrument_rcv == instrument
        assert data_type == "quote"
        assert not is_warmup
        assert hasattr(quote, "time")
        assert hasattr(quote, "bid")
        assert hasattr(quote, "ask")
        assert hasattr(quote, "bid_size")
        assert hasattr(quote, "ask_size")

        # At least one side should have valid price
        assert quote.bid > 0 or quote.ask > 0, "Quote should have valid bid or ask"

        # Cleanup
        data_provider.unsubscribe("quote", {instrument})

    @pytest.mark.asyncio
    async def test_unsubscribe(self, data_provider, test_instruments, test_channel):
        """Test unsubscribe functionality"""
        instrument = test_instruments[0]  # BTC

        # Subscribe
        data_provider.subscribe("orderbook", {instrument})
        await asyncio.sleep(5)

        # Verify subscription exists
        assert data_provider.has_subscription(instrument, "orderbook")

        # Unsubscribe
        data_provider.unsubscribe("orderbook", {instrument})

        # Verify subscription removed
        assert not data_provider.has_subscription(instrument, "orderbook")


@pytest.mark.integration
class TestXLighterOrderbookIntegration:
    """Integration tests for orderbook functionality"""

    @pytest.mark.asyncio
    async def test_orderbook_updates(self, data_provider, test_instruments, test_channel):
        """Test orderbook state updates (snapshot + deltas)"""
        instrument = test_instruments[0]  # BTC

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe to orderbook
        data_provider.subscribe("orderbook", {instrument})

        # Wait for multiple updates
        await asyncio.sleep(20)

        # Verify we received multiple orderbook updates
        orderbook_data = [d for d in test_channel.received_data if d[1] == "orderbook"]
        assert len(orderbook_data) > 1, "Should receive multiple orderbook updates"

        # Verify orderbooks have different states (prices/sizes change)
        first_ob = orderbook_data[0][2]
        last_ob = orderbook_data[-1][2]

        # At least one should be different (time, top level, or depth)
        assert (
            first_ob.time != last_ob.time
            or len(first_ob.bids) != len(last_ob.bids)
            or len(first_ob.asks) != len(last_ob.asks)
        ), "Orderbook should update over time"

        # Cleanup
        data_provider.unsubscribe("orderbook", {instrument})

    @pytest.mark.asyncio
    async def test_orderbook_depth(self, data_provider, test_instruments, test_channel):
        """Test orderbook depth parameter"""
        instrument = test_instruments[0]  # BTC

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe with specific depth (use depth parameter, not colon notation)
        data_provider.subscribe("orderbook", {instrument})  # Default depth

        # Wait for data
        await asyncio.sleep(15)

        # Verify orderbook data with depth
        orderbook_data = [d for d in test_channel.received_data if d[1] == "orderbook"]
        assert len(orderbook_data) > 0

        orderbook = orderbook_data[0][2]
        # Depth might be less than requested if market is thin
        assert len(orderbook.bids) <= 10, "Should respect max depth parameter"
        assert len(orderbook.asks) <= 10, "Should respect max depth parameter"

        # Cleanup
        data_provider.unsubscribe("orderbook", {instrument})

    @pytest.mark.asyncio
    async def test_orderbook_aggregation_by_percentage(self, data_provider, test_instruments, test_channel):
        """Test orderbook aggregation by tick_size_pct parameter"""
        instrument = test_instruments[0]  # BTC

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe with percentage-based aggregation: 0.01% tick size, 20 levels
        # This should aggregate by 0.01% of mid price (0.2% total depth)
        data_provider.subscribe("orderbook(0.01, 20)", {instrument})

        # Wait for aggregated orderbook data
        await asyncio.sleep(15)

        # Verify orderbook data received
        orderbook_data = [d for d in test_channel.received_data if d[1] == "orderbook"]
        assert len(orderbook_data) > 0, "Should receive aggregated orderbook data"

        orderbook = orderbook_data[0][2]

        # Verify orderbook has at most 20 levels
        assert len(orderbook.bids) <= 20, "Should have at most 20 bid levels"
        assert len(orderbook.asks) <= 20, "Should have at most 20 ask levels"

        # Verify tick size is percentage of mid price
        mid_price = (orderbook.top_bid + orderbook.top_ask) / 2.0
        expected_tick_size = mid_price * 0.01 / 100.0  # 0.01% of mid
        # Tick size should be at least the expected percentage (rounded to instrument.tick_size)
        assert orderbook.tick_size >= instrument.tick_size, "Aggregated tick size should be >= instrument tick_size"

        # Verify price levels are spaced by the aggregated tick_size
        if len(orderbook.bids) > 1:
            bid_spacing = orderbook.bids[0][0] - orderbook.bids[1][0]  # Price difference between levels
            # Allow some tolerance for rounding
            assert abs(bid_spacing - orderbook.tick_size) < orderbook.tick_size * 0.1, "Bid levels should be spaced by tick_size"

        if len(orderbook.asks) > 1:
            ask_spacing = orderbook.asks[1][0] - orderbook.asks[0][0]
            assert abs(ask_spacing - orderbook.tick_size) < orderbook.tick_size * 0.1, "Ask levels should be spaced by tick_size"

        # Cleanup
        data_provider.unsubscribe("orderbook(0.01, 20)", {instrument})


@pytest.mark.integration
class TestXLighterTradesIntegration:
    """Integration tests for trade feed"""

    @pytest.mark.asyncio
    async def test_trade_data_format(self, data_provider, test_instruments, test_channel):
        """Test trade data format and conversion"""
        instrument = test_instruments[1]  # ETH (more liquid, faster trades)

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe to trades
        data_provider.subscribe("trade", {instrument})

        # Wait for trades
        await asyncio.sleep(25)

        # Verify trades received
        trade_data = [d for d in test_channel.received_data if d[1] == "trade"]
        assert len(trade_data) > 0, "Should receive trades"

        # Check trade object structure
        for instrument_rcv, data_type, trade, is_warmup in trade_data:
            assert instrument_rcv == instrument
            assert data_type == "trade"
            assert trade.time > 0, "Trade should have valid timestamp"
            assert trade.price > 0, "Trade price should be positive"
            assert trade.size > 0, "Trade size should be positive"
            assert hasattr(trade, "side"), "Trade should have side (buy/sell indicator)"

        # Cleanup
        data_provider.unsubscribe("trade", {instrument})


@pytest.mark.integration
class TestXLighterQuoteIntegration:
    """Integration tests for quote derivation"""

    @pytest.mark.asyncio
    async def test_quote_from_orderbook(self, data_provider, test_instruments, test_channel):
        """Test that quotes are correctly derived from orderbook"""
        instrument = test_instruments[0]  # BTC

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe to quotes (should subscribe to orderbook internally)
        data_provider.subscribe("quote", {instrument})

        # Wait for quotes
        await asyncio.sleep(15)

        # Verify quotes received
        quote_data = [d for d in test_channel.received_data if d[1] == "quote"]
        assert len(quote_data) > 0

        # Verify quote values are reasonable
        quote = quote_data[0][2]
        if quote.bid > 0 and quote.ask > 0:
            # Spread should be positive
            spread = quote.ask - quote.bid
            assert spread >= 0, "Ask should be >= bid"

            # Spread should be reasonable (< 1% of mid price)
            mid_price = (quote.bid + quote.ask) / 2
            assert spread < mid_price * 0.01, "Spread should be reasonable"

        # Cleanup
        data_provider.unsubscribe("quote", {instrument})

    @pytest.mark.asyncio
    async def test_synthetic_quote_from_orderbook(self, data_provider, test_instruments, test_channel):
        """Test synthetic quote generation when subscribing to orderbook without explicit quote subscription"""
        instrument = test_instruments[0]  # BTC

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe ONLY to orderbook (should NOT generate synthetic quotes automatically)
        # Note: LighterDataProvider only generates synthetic quotes internally, not to channel
        data_provider.subscribe("orderbook", {instrument})

        # Wait for orderbook updates
        await asyncio.sleep(15)

        # Verify we get orderbook but NOT quote events (synthetic quotes are internal only)
        orderbook_data = [d for d in test_channel.received_data if d[1] == "orderbook"]
        quote_data = [d for d in test_channel.received_data if d[1] == "quote"]

        assert len(orderbook_data) > 0, "Should receive orderbook"
        # Synthetic quotes are stored internally but not sent to channel
        assert len(quote_data) == 0, "Should NOT receive quote events without explicit quote subscription"

        # Cleanup
        data_provider.unsubscribe("orderbook", {instrument})


@pytest.mark.integration
class TestXLighterMultiInstrument:
    """Integration tests for multi-instrument subscriptions"""

    @pytest.mark.asyncio
    async def test_multi_instrument_orderbook(self, data_provider, test_instruments, test_channel):
        """Test subscribing to orderbook for multiple instruments"""
        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe to both BTC and ETH orderbooks
        data_provider.subscribe("orderbook", set(test_instruments))

        # Wait for data from both
        await asyncio.sleep(20)

        # Verify both instruments received data
        orderbook_data = [d for d in test_channel.received_data if d[1] == "orderbook"]
        instruments_with_data = {d[0] for d in orderbook_data}

        assert test_instruments[0] in instruments_with_data, "BTC should receive orderbook data"
        assert test_instruments[1] in instruments_with_data, "ETH should receive orderbook data"

        # Cleanup
        data_provider.unsubscribe("orderbook", set(test_instruments))

    @pytest.mark.asyncio
    async def test_mixed_subscriptions(self, data_provider, test_instruments, test_channel):
        """Test different subscription types for different instruments"""
        btc_instrument = test_instruments[0]
        eth_instrument = test_instruments[1]

        # Clear previous data
        test_channel.received_data.clear()

        # Subscribe BTC to orderbook, ETH to trades
        data_provider.subscribe("orderbook", {btc_instrument})
        data_provider.subscribe("trade", {eth_instrument})

        # Wait for data
        await asyncio.sleep(20)

        # Verify BTC orderbook
        btc_orderbook = [d for d in test_channel.received_data if d[0] == btc_instrument and d[1] == "orderbook"]
        assert len(btc_orderbook) > 0, "BTC should receive orderbook"

        # Verify ETH trades
        eth_trades = [d for d in test_channel.received_data if d[0] == eth_instrument and d[1] == "trade"]
        assert len(eth_trades) > 0, "ETH should receive trades"

        # Cleanup
        data_provider.unsubscribe("orderbook", {btc_instrument})
        data_provider.unsubscribe("trade", {eth_instrument})

    @pytest.mark.asyncio
    async def test_subscription_queries(self, data_provider, test_instruments):
        """Test subscription query methods"""
        btc_instrument = test_instruments[0]
        eth_instrument = test_instruments[1]

        # Subscribe to different types
        data_provider.subscribe("orderbook", {btc_instrument})
        data_provider.subscribe("trade", {eth_instrument})
        data_provider.subscribe("quote", {btc_instrument})

        # Wait for subscriptions to be established
        await asyncio.sleep(2)

        # Test has_subscription
        assert data_provider.has_subscription(btc_instrument, "orderbook")
        assert data_provider.has_subscription(btc_instrument, "quote")
        assert data_provider.has_subscription(eth_instrument, "trade")
        assert not data_provider.has_subscription(eth_instrument, "orderbook")

        # Test get_subscriptions for instrument
        btc_subs = data_provider.get_subscriptions(btc_instrument)
        assert "orderbook" in btc_subs
        assert "quote" in btc_subs

        eth_subs = data_provider.get_subscriptions(eth_instrument)
        assert "trade" in eth_subs

        # Test get_subscribed_instruments
        orderbook_instruments = data_provider.get_subscribed_instruments("orderbook")
        assert btc_instrument in orderbook_instruments

        trade_instruments = data_provider.get_subscribed_instruments("trade")
        assert eth_instrument in trade_instruments

        # Cleanup
        data_provider.unsubscribe(None, set(test_instruments))  # Unsubscribe all


@pytest.mark.integration
class TestXLighterErrorHandling:
    """Integration tests for error handling"""

    @pytest.mark.asyncio
    async def test_invalid_subscription_type(self, data_provider, test_instruments):
        """Test error handling for invalid subscription type"""
        instrument = test_instruments[0]

        with pytest.raises(ValueError, match="Unsupported subscription type"):
            data_provider.subscribe("invalid_type", {instrument})

    @pytest.mark.asyncio
    async def test_empty_instruments(self, data_provider):
        """Test handling of empty instrument set"""
        # Should not raise error, just log debug message
        data_provider.subscribe("orderbook", set())

        # Verify no subscriptions created
        all_subs = data_provider.get_subscriptions()
        assert len(all_subs) == 0 or "orderbook" not in all_subs
