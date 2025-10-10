"""Tests for LighterBroker"""

import asyncio
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

from qubx.connectors.xlighter.broker import LighterBroker
from qubx.connectors.xlighter.client import LighterClient
from qubx.connectors.xlighter.instruments import LighterInstrumentLoader
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
from qubx.core.exceptions import InvalidOrderParameters, OrderNotFound


@pytest.fixture
def mock_client():
    """Mock LighterClient"""
    client = Mock(spec=LighterClient)
    client.create_order = AsyncMock()
    client.cancel_order = AsyncMock()
    return client


@pytest.fixture
def mock_instrument_loader():
    """Mock LighterInstrumentLoader"""
    loader = Mock(spec=LighterInstrumentLoader)

    # Create sample instruments
    btc = Instrument(
        symbol="BTCUSDC",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="XLIGHTER",
        base="BTC",
        quote="USDC",
        settle="USDC",
        exchange_symbol="BTCUSDC",  # Lighter exchange still uses BTC-USDC format
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        min_notional=5.0,
    )

    loader.get_market_id = Mock(side_effect=lambda symbol: {"BTCUSDC": 0}.get(symbol))
    loader.instruments = {"BTCUSDC": btc}

    return loader


@pytest.fixture
def mock_channel():
    """Mock control channel"""
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock()
    return channel


@pytest.fixture
def mock_time_provider():
    """Mock time provider"""
    time_provider = Mock()
    time_provider.time = Mock(return_value=1000000000)
    return time_provider


@pytest.fixture
def mock_account():
    """Mock account processor"""
    account = Mock()
    account.get_orders = Mock(return_value={})
    return account


@pytest.fixture
def mock_data_provider():
    """Mock data provider"""
    return Mock()


@pytest.fixture
def broker(mock_client, mock_instrument_loader, mock_channel, mock_time_provider, mock_account, mock_data_provider):
    """Create LighterBroker with mocks"""
    return LighterBroker(
        client=mock_client,
        instrument_loader=mock_instrument_loader,
        channel=mock_channel,
        time_provider=mock_time_provider,
        account=mock_account,
        data_provider=mock_data_provider,
    )


class TestBrokerInit:
    """Test broker initialization"""

    def test_init(self, broker, mock_client, mock_instrument_loader):
        """Test basic initialization"""
        assert broker.client == mock_client
        assert broker.instrument_loader == mock_instrument_loader
        assert broker.is_simulated_trading is False


class TestCreateOrder:
    """Test order creation"""

    @pytest.mark.asyncio
    async def test_create_market_order(self, broker, mock_client, mock_instrument_loader):
        """Test creating market order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock successful order creation
        mock_response = Mock()
        mock_response.tx_hash = "0x123"
        mock_client.create_order.return_value = (None, mock_response, None)

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="market",
            amount=1.0,
            price=None,
            client_id=None,
            time_in_force="gtc",
        )

        # Verify order creation called correctly
        mock_client.create_order.assert_called_once()
        call_kwargs = mock_client.create_order.call_args[1]
        assert call_kwargs["market_id"] == 0
        assert call_kwargs["is_buy"] is True
        assert call_kwargs["size"] == 1.0
        assert call_kwargs["order_type"] == 1  # MARKET

        # Verify order object
        assert order.instrument == btc
        assert order.side == "buy"
        assert order.type == "MARKET"
        assert order.quantity == 1.0
        assert order.price == 0.0

    @pytest.mark.asyncio
    async def test_create_limit_order(self, broker, mock_client, mock_instrument_loader):
        """Test creating limit order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock successful order creation
        mock_response = Mock()
        mock_response.tx_hash = "0x456"
        mock_client.create_order.return_value = (None, mock_response, None)

        order = await broker._create_order(
            instrument=btc,
            order_side="sell",
            order_type="limit",
            amount=0.5,
            price=50000.0,
            client_id="test_order_1",
            time_in_force="gtc",
        )

        # Verify order creation
        mock_client.create_order.assert_called_once()
        call_kwargs = mock_client.create_order.call_args[1]
        assert call_kwargs["market_id"] == 0
        assert call_kwargs["is_buy"] is False
        assert call_kwargs["size"] == 0.5
        assert call_kwargs["price"] == 50000.0
        assert call_kwargs["order_type"] == 0  # LIMIT

        # Verify order object
        assert order.side == "sell"
        assert order.type == "LIMIT"
        assert order.quantity == 0.5
        assert order.price == 50000.0
        assert order.client_id == "test_order_1"

    @pytest.mark.asyncio
    async def test_create_order_with_ioc(self, broker, mock_client, mock_instrument_loader):
        """Test creating order with IOC time in force"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        mock_response = Mock()
        mock_response.tx_hash = "0x789"
        mock_client.create_order.return_value = (None, mock_response, None)

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="limit",
            amount=1.0,
            price=49000.0,
            client_id=None,
            time_in_force="ioc",
        )

        # Verify TIF is IOC
        call_kwargs = mock_client.create_order.call_args[1]
        assert call_kwargs["time_in_force"] == 0  # IOC

        assert order.time_in_force == "ioc"

    @pytest.mark.asyncio
    async def test_create_order_post_only(self, broker, mock_client, mock_instrument_loader):
        """Test creating post-only order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        mock_response = Mock()
        mock_response.tx_hash = "0xabc"
        mock_client.create_order.return_value = (None, mock_response, None)

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="limit",
            amount=1.0,
            price=49000.0,
            client_id=None,
            time_in_force="post_only",
        )

        # Verify post_only flag
        call_kwargs = mock_client.create_order.call_args[1]
        assert call_kwargs["post_only"] is True
        assert call_kwargs["time_in_force"] == 2  # POST_ONLY

    @pytest.mark.asyncio
    async def test_create_order_reduce_only(self, broker, mock_client, mock_instrument_loader):
        """Test creating reduce-only order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        mock_response = Mock()
        mock_response.tx_hash = "0xdef"
        mock_client.create_order.return_value = (None, mock_response, None)

        order = await broker._create_order(
            instrument=btc,
            order_side="sell",
            order_type="market",
            amount=0.5,
            price=None,
            client_id=None,
            time_in_force="gtc",
            reduce_only=True,
        )

        # Verify reduce_only flag
        call_kwargs = mock_client.create_order.call_args[1]
        assert call_kwargs["reduce_only"] is True

        assert order.options.get("reduce_only") is True

    @pytest.mark.asyncio
    async def test_create_order_generates_client_id(self, broker, mock_client, mock_instrument_loader):
        """Test that client_id is generated if not provided"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        mock_response = Mock()
        mock_response.tx_hash = "0x999"
        mock_client.create_order.return_value = (None, mock_response, None)

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="market",
            amount=1.0,
            price=None,
            client_id=None,
            time_in_force="gtc",
        )

        # Client ID should be generated
        assert order.client_id is not None
        assert len(order.client_id) > 0

    @pytest.mark.asyncio
    async def test_create_order_invalid_type(self, broker, mock_instrument_loader):
        """Test that invalid order type raises error"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        with pytest.raises(InvalidOrderParameters, match="Invalid order type"):
            await broker._create_order(
                instrument=btc,
                order_side="buy",
                order_type="invalid",
                amount=1.0,
                price=None,
                client_id=None,
                time_in_force="gtc",
            )

    @pytest.mark.asyncio
    async def test_create_limit_order_without_price(self, broker, mock_instrument_loader):
        """Test that limit order without price raises error"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        with pytest.raises(InvalidOrderParameters, match="Limit orders require a price"):
            await broker._create_order(
                instrument=btc,
                order_side="buy",
                order_type="limit",
                amount=1.0,
                price=None,
                client_id=None,
                time_in_force="gtc",
            )

    @pytest.mark.asyncio
    async def test_create_order_unknown_instrument(self, broker, mock_client):
        """Test creating order for unknown instrument"""
        unknown = Instrument(
            symbol="UNKNOWNUSDC",
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

        with pytest.raises(InvalidOrderParameters, match="Market ID not found"):
            await broker._create_order(
                instrument=unknown,
                order_side="buy",
                order_type="market",
                amount=1.0,
                price=None,
                client_id=None,
                time_in_force="gtc",
            )

    @pytest.mark.asyncio
    async def test_create_order_api_error(self, broker, mock_client, mock_instrument_loader):
        """Test handling API error during order creation"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock API error
        mock_client.create_order.return_value = (None, None, "API Error: Insufficient funds")

        with pytest.raises(InvalidOrderParameters, match="Lighter API error"):
            await broker._create_order(
                instrument=btc,
                order_side="buy",
                order_type="market",
                amount=1.0,
                price=None,
                client_id=None,
                time_in_force="gtc",
            )


class TestCancelOrder:
    """Test order cancellation"""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, broker, mock_client, mock_account, mock_instrument_loader):
        """Test successful order cancellation"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock existing order
        import numpy as np

        from qubx.core.basics import Order

        existing_order = Order(
            id="123",
            type="LIMIT",
            instrument=btc,
            time=np.datetime64(1000000000, "ns"),
            quantity=1.0,
            price=50000.0,
            side="BUY",
            status="OPEN",
            time_in_force="GTC",
            client_id="client_1",
        )
        mock_account.get_orders.return_value = {"123": existing_order}

        # Mock successful cancellation
        mock_response = Mock()
        mock_response.tx_hash = "0xccc"
        mock_client.cancel_order.return_value = (None, mock_response, None)

        result = await broker._cancel_order("123")

        # Verify cancellation called
        mock_client.cancel_order.assert_called_once()
        call_kwargs = mock_client.cancel_order.call_args[1]
        assert call_kwargs["order_id"] == 123
        assert call_kwargs["market_id"] == 0

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, broker, mock_account):
        """Test cancelling order that doesn't exist"""
        mock_account.get_orders.return_value = {}

        with pytest.raises(OrderNotFound, match="Order not found"):
            await broker._cancel_order("nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_order_api_error(self, broker, mock_client, mock_account, mock_instrument_loader):
        """Test API error during cancellation"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock existing order
        import numpy as np

        from qubx.core.basics import Order

        existing_order = Order(
            id="123",
            type="LIMIT",
            instrument=btc,
            time=np.datetime64(1000000000, "ns"),
            quantity=1.0,
            price=50000.0,
            side="BUY",
            status="OPEN",
            time_in_force="GTC",
        )
        mock_account.get_orders.return_value = {"123": existing_order}

        # Mock cancellation error
        mock_client.cancel_order.return_value = (None, None, "Order already cancelled")

        result = await broker._cancel_order("123")

        assert result is False


class TestCancelOrders:
    """Test cancelling all orders for instrument"""

    def test_cancel_all_orders(self, broker, mock_account, mock_instrument_loader):
        """Test cancelling all orders for an instrument"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock orders
        import numpy as np

        from qubx.core.basics import Order

        order1 = Order(
            id="1",
            type="LIMIT",
            instrument=btc,
            time=np.datetime64(1000000000, "ns"),
            quantity=1.0,
            price=50000.0,
            side="BUY",
            status="OPEN",
            time_in_force="GTC",
        )
        order2 = Order(
            id="2",
            type="LIMIT",
            instrument=btc,
            time=np.datetime64(1000000000, "ns"),
            quantity=1.0,
            price=51000.0,
            side="SELL",
            status="OPEN",
            time_in_force="GTC",
        )

        mock_account.get_orders.return_value = {"1": order1, "2": order2}

        with patch.object(broker, "cancel_order_async") as mock_cancel:
            broker.cancel_orders(btc)

            # Should call cancel for each order
            assert mock_cancel.call_count == 2


class TestUpdateOrder:
    """Test order modification"""

    def test_update_order(self, broker, mock_account, mock_instrument_loader):
        """Test updating order (cancel + replace)"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock existing order
        import numpy as np

        from qubx.core.basics import Order

        existing_order = Order(
            id="123",
            type="LIMIT",
            instrument=btc,
            time=np.datetime64(1000000000, "ns"),
            quantity=1.0,
            price=50000.0,
            side="BUY",
            status="OPEN",
            time_in_force="GTC",
            options={"reduce_only": False},
        )
        mock_account.get_orders.return_value = {"123": existing_order}

        with patch.object(broker, "cancel_order") as mock_cancel:
            with patch.object(broker, "send_order") as mock_send:
                mock_send.return_value = existing_order

                broker.update_order("123", price=49000.0, amount=0.5)

                # Should cancel old order
                mock_cancel.assert_called_once_with("123")

                # Should create new order
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["price"] == 49000.0
                assert call_kwargs["amount"] == 0.5
