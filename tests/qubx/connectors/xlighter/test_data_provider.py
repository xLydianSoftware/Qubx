"""Tests for LighterDataProvider"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from qubx.connectors.xlighter.client import LighterClient
from qubx.connectors.xlighter.data import LighterDataProvider
from qubx.connectors.xlighter.instruments import LighterInstrumentLoader
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType


@pytest.fixture
def mock_client():
    """Mock LighterClient"""
    client = Mock(spec=LighterClient)
    return client


@pytest.fixture
def mock_instrument_loader():
    """Mock LighterInstrumentLoader with sample instruments"""
    loader = Mock(spec=LighterInstrumentLoader)

    # Create sample instruments
    btc_instrument = Instrument(
        symbol="BTC-USDC",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="XLIGHTER",
        base="BTC",
        quote="USDC",
        settle="USDC",
        exchange_symbol="BTC-USDC",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        min_notional=5.0,
    )

    eth_instrument = Instrument(
        symbol="ETH-USDC",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="XLIGHTER",
        base="ETH",
        quote="USDC",
        settle="USDC",
        exchange_symbol="ETH-USDC",
        tick_size=0.01,
        lot_size=0.01,
        min_size=0.01,
        min_notional=5.0,
    )

    loader.get_market_id = Mock(side_effect=lambda symbol: {"BTC-USDC": 0, "ETH-USDC": 1}.get(symbol))
    loader.instruments = {"BTC-USDC": btc_instrument, "ETH-USDC": eth_instrument}

    return loader


@pytest.fixture
def mock_time_provider():
    """Mock time provider"""
    time_provider = Mock()
    time_provider.time = Mock(return_value=1000000000)
    return time_provider


@pytest.fixture
def mock_channel():
    """Mock control channel"""
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock()
    return channel


@pytest.fixture
def data_provider(mock_client, mock_instrument_loader, mock_time_provider, mock_channel):
    """Create LighterDataProvider with mocks"""
    return LighterDataProvider(
        client=mock_client,
        instrument_loader=mock_instrument_loader,
        time_provider=mock_time_provider,
        channel=mock_channel,
    )


class TestLighterDataProviderInit:
    """Test initialization"""

    def test_init(self, data_provider, mock_client, mock_instrument_loader):
        """Test basic initialization"""
        assert data_provider.client == mock_client
        assert data_provider.instrument_loader == mock_instrument_loader
        assert data_provider.is_simulation is False
        assert data_provider._ws_connected is False


class TestSubscriptionManagement:
    """Test subscription tracking and management"""

    def test_subscribe_single_instrument(self, data_provider, mock_instrument_loader):
        """Test subscribing single instrument"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("orderbook", {btc})

                assert "orderbook" in data_provider._subscriptions
                assert btc in data_provider._subscriptions["orderbook"]

    def test_subscribe_multiple_instruments(self, data_provider, mock_instrument_loader):
        """Test subscribing multiple instruments"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("trade", {btc, eth})

                assert "trade" in data_provider._subscriptions
                assert btc in data_provider._subscriptions["trade"]
                assert eth in data_provider._subscriptions["trade"]

    def test_subscribe_reset(self, data_provider, mock_instrument_loader):
        """Test reset flag replaces existing subscriptions"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                # First subscription
                data_provider.subscribe("orderbook", {btc})
                assert btc in data_provider._subscriptions["orderbook"]

                # Second subscription with reset
                data_provider.subscribe("orderbook", {eth}, reset=True)
                assert eth in data_provider._subscriptions["orderbook"]
                assert btc not in data_provider._subscriptions["orderbook"]

    def test_subscribe_add_mode(self, data_provider, mock_instrument_loader):
        """Test add mode preserves existing subscriptions"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                # First subscription
                data_provider.subscribe("orderbook", {btc})
                # Second subscription without reset (add mode)
                data_provider.subscribe("orderbook", {eth}, reset=False)

                assert btc in data_provider._subscriptions["orderbook"]
                assert eth in data_provider._subscriptions["orderbook"]

    def test_subscribe_empty_instruments(self, data_provider):
        """Test subscribing with empty instrument set"""
        with patch.object(data_provider, "_connect_websocket"):
            data_provider.subscribe("orderbook", set())
            # Should not create subscription
            assert "orderbook" not in data_provider._subscriptions


class TestUnsubscribe:
    """Test unsubscription"""

    def test_unsubscribe_single_instrument(self, data_provider, mock_instrument_loader):
        """Test unsubscribing single instrument"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                # Subscribe
                data_provider.subscribe("orderbook", {btc})
                assert btc in data_provider._subscriptions["orderbook"]

                # Unsubscribe
                data_provider.unsubscribe("orderbook", {btc})
                assert "orderbook" not in data_provider._subscriptions

    def test_unsubscribe_partial(self, data_provider, mock_instrument_loader):
        """Test partial unsubscription"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                # Subscribe both
                data_provider.subscribe("trade", {btc, eth})

                # Unsubscribe one
                data_provider.unsubscribe("trade", {btc})

                assert "trade" in data_provider._subscriptions
                assert btc not in data_provider._subscriptions["trade"]
                assert eth in data_provider._subscriptions["trade"]

    def test_unsubscribe_all_types(self, data_provider, mock_instrument_loader):
        """Test unsubscribing from all subscription types"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                # Subscribe to multiple types
                data_provider.subscribe("orderbook", {btc})
                data_provider.subscribe("trade", {btc})

                # Unsubscribe from all (subscription_type=None)
                data_provider.unsubscribe(None, {btc})

                assert btc not in data_provider._subscriptions.get("orderbook", set())
                assert btc not in data_provider._subscriptions.get("trade", set())


class TestSubscriptionQueries:
    """Test subscription query methods"""

    def test_has_subscription(self, data_provider, mock_instrument_loader):
        """Test has_subscription check"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("orderbook", {btc})

                assert data_provider.has_subscription(btc, "orderbook") is True
                assert data_provider.has_subscription(eth, "orderbook") is False
                assert data_provider.has_subscription(btc, "trade") is False

    def test_get_subscriptions_for_instrument(self, data_provider, mock_instrument_loader):
        """Test getting subscriptions for specific instrument"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("orderbook", {btc})
                data_provider.subscribe("trade", {btc})

                subs = data_provider.get_subscriptions(btc)
                assert "orderbook" in subs
                assert "trade" in subs
                assert len(subs) == 2

    def test_get_subscriptions_all(self, data_provider, mock_instrument_loader):
        """Test getting all subscription types"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("orderbook", {btc})
                data_provider.subscribe("trade", {btc})

                subs = data_provider.get_subscriptions()
                assert "orderbook" in subs
                assert "trade" in subs

    def test_get_subscribed_instruments_for_type(self, data_provider, mock_instrument_loader):
        """Test getting subscribed instruments for subscription type"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("orderbook", {btc, eth})

                instruments = data_provider.get_subscribed_instruments("orderbook")
                assert btc in instruments
                assert eth in instruments
                assert len(instruments) == 2

    def test_get_subscribed_instruments_all(self, data_provider, mock_instrument_loader):
        """Test getting all subscribed instruments"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]
        eth = mock_instrument_loader.instruments["ETH-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with patch.object(data_provider, "_subscribe_instrument"):
                data_provider.subscribe("orderbook", {btc})
                data_provider.subscribe("trade", {eth})

                instruments = data_provider.get_subscribed_instruments()
                assert btc in instruments
                assert eth in instruments


class TestHandlerCreation:
    """Test handler creation"""

    def test_create_orderbook_handler(self, data_provider, mock_instrument_loader):
        """Test creating orderbook handler"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        handler = data_provider._create_handler("orderbook", btc, market_id=0, depth=100)

        assert handler is not None
        assert handler.market_id == 0
        assert handler.tick_size == 0.01
        assert handler.max_levels == 100

    def test_create_trade_handler(self, data_provider, mock_instrument_loader):
        """Test creating trade handler"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        handler = data_provider._create_handler("trade", btc, market_id=0)

        assert handler is not None
        assert handler.market_id == 0

    def test_create_quote_handler(self, data_provider, mock_instrument_loader):
        """Test creating quote handler"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        handler = data_provider._create_handler("quote", btc, market_id=0)

        assert handler is not None
        assert handler.market_id == 0

    def test_create_unknown_handler_raises(self, data_provider, mock_instrument_loader):
        """Test creating unknown handler type raises error"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with pytest.raises(ValueError, match="Unknown subscription type"):
            data_provider._create_handler("unknown", btc, market_id=0)


class TestWarmup:
    """Test warmup functionality"""

    def test_warmup_empty_config(self, data_provider):
        """Test warmup with empty config"""
        data_provider.warmup({})
        # Should not raise, just log

    def test_warmup_trade(self, data_provider, mock_instrument_loader):
        """Test warmup for trade data"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_warmup_instrument") as mock_warmup:
            data_provider.warmup({("trade", btc): "1h"})
            mock_warmup.assert_called_once_with("trade", btc, "1h")

    def test_warmup_orderbook_skipped(self, data_provider, mock_instrument_loader):
        """Test that orderbook warmup is skipped (realtime only)"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        # Should not raise, just log that it's skipped
        data_provider.warmup({("orderbook", btc): "10min"})


class TestValidation:
    """Test validation and error handling"""

    def test_subscribe_unsupported_type_raises(self, data_provider, mock_instrument_loader):
        """Test subscribing to unsupported type raises error"""
        btc = mock_instrument_loader.instruments["BTC-USDC"]

        with patch.object(data_provider, "_connect_websocket"):
            with pytest.raises(ValueError, match="Unsupported subscription type"):
                data_provider.subscribe("unsupported_type", {btc})

    def test_subscribe_instrument_without_market_id_raises(self, data_provider):
        """Test subscribing instrument without market_id raises error"""
        unknown_instrument = Instrument(
            symbol="UNKNOWN-USDC",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="XLIGHTER",
            base="UNKNOWN",
            quote="USDC",
            settle="USDC",
            exchange_symbol="UNKNOWN-USDC",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
            min_notional=5.0,
        )

        with patch.object(data_provider, "_connect_websocket"):
            with pytest.raises(ValueError, match="Market ID not found"):
                data_provider._subscribe_instrument("orderbook", unknown_instrument)


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations"""

    async def test_close(self, data_provider):
        """Test closing provider"""
        mock_ws = AsyncMock()
        data_provider._ws_manager = mock_ws

        await data_provider.close()

        mock_ws.disconnect.assert_called_once()
