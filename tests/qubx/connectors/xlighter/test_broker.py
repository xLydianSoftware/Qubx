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

    # Create sample instruments (BTC market: tick_size=0.1, lot_size=0.00001)
    btc = Instrument(
        symbol="BTCUSDC",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="XLIGHTER",
        base="BTC",
        quote="USDC",
        settle="USDC",
        exchange_symbol="BTCUSDC",
        tick_size=0.1,  # price_decimals = 1
        lot_size=0.00001,  # size_decimals = 5
        min_size=0.00001,
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
    data_provider = Mock()

    # Mock quote for market orders
    mock_quote = Mock()
    mock_quote.mid_price = Mock(return_value=50000.0)  # BTC mid price
    data_provider.get_quote = Mock(return_value=mock_quote)

    return data_provider


@pytest.fixture
def mock_ws_manager():
    """Mock WebSocket manager"""
    ws_manager = Mock()
    ws_manager.send_tx = AsyncMock(return_value={"tx_id": "test_tx_123", "status": "sent"})
    ws_manager.send_batch_tx = AsyncMock(return_value={"batch_id": "test_batch_123", "count": 1, "status": "sent"})
    return ws_manager


@pytest.fixture
def broker(mock_client, mock_instrument_loader, mock_ws_manager, mock_channel, mock_time_provider, mock_account, mock_data_provider):
    """Create LighterBroker with mocks"""
    loop = asyncio.get_event_loop()

    # Setup mock signer client
    mock_signer = Mock()
    mock_signer.sign_create_order = Mock(return_value=('{"hash": "0xabc123"}', None))
    mock_signer.sign_cancel_order = Mock(return_value=('{"hash": "0xcancel"}', None))
    mock_client.signer_client = mock_signer
    mock_client._loop = loop

    # Setup mock account.process_order
    mock_account.process_order = Mock()

    return LighterBroker(
        client=mock_client,
        instrument_loader=mock_instrument_loader,
        ws_manager=mock_ws_manager,
        channel=mock_channel,
        time_provider=mock_time_provider,
        account=mock_account,
        data_provider=mock_data_provider,
        loop=loop,
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
    async def test_create_market_order(self, broker, mock_client, mock_ws_manager, mock_instrument_loader):
        """Test creating market order with slippage protection"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="market",
            amount=1.0,
            price=None,
            client_id="test_order",
            time_in_force="gtc",
        )

        # Verify signing was called correctly
        mock_client.signer_client.sign_create_order.assert_called_once()
        call_kwargs = mock_client.signer_client.sign_create_order.call_args[1]
        assert call_kwargs["market_index"] == 0
        assert call_kwargs["is_ask"] is False  # buy = not ask
        assert call_kwargs["base_amount"] == int(1.0 * 1e5)  # 1.0 * 10^5 (size_decimals=5)
        assert call_kwargs["order_type"] == 1  # MARKET
        assert call_kwargs["time_in_force"] == 0  # IOC for market orders
        assert call_kwargs["order_expiry"] == 0  # IOC expiry

        # Verify slippage protection price was calculated
        # mid_price = 50000, default slippage = 5%, buy order
        # protected_price = 50000 * 1.05 = 52500
        expected_protected_price = 52500.0
        assert call_kwargs["price"] == int(expected_protected_price * 1e1)  # 52500 * 10^1 (price_decimals=1)

        # Verify WebSocket submission was called
        mock_ws_manager.send_tx.assert_called_once()

        # Verify order object
        assert order.instrument == btc
        assert order.side == "buy"
        assert order.type == "MARKET"
        assert order.quantity == 1.0
        assert order.price == expected_protected_price
        assert order.status == "NEW"

    @pytest.mark.asyncio
    async def test_create_limit_order(self, broker, mock_client, mock_ws_manager, mock_instrument_loader):
        """Test creating limit order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        order = await broker._create_order(
            instrument=btc,
            order_side="sell",
            order_type="limit",
            amount=0.5,
            price=50000.0,
            client_id="test_order_1",
            time_in_force="gtc",
        )

        # Verify signing
        mock_client.signer_client.sign_create_order.assert_called_once()
        call_kwargs = mock_client.signer_client.sign_create_order.call_args[1]
        assert call_kwargs["market_index"] == 0
        assert call_kwargs["is_ask"] is True  # sell = ask
        assert call_kwargs["base_amount"] == int(0.5 * 1e5)  # 0.5 * 10^5 (size_decimals=5)
        assert call_kwargs["price"] == int(50000.0 * 1e1)  # 50000 * 10^1 (price_decimals=1)
        assert call_kwargs["order_type"] == 0  # LIMIT
        assert call_kwargs["time_in_force"] == 1  # GOOD_TILL_TIME for limit orders
        assert call_kwargs["order_expiry"] == -1  # 28-day expiry

        # Verify WebSocket submission
        mock_ws_manager.send_tx.assert_called_once()

        # Verify order object
        assert order.side == "sell"
        assert order.type == "LIMIT"
        assert order.quantity == 0.5
        assert order.price == 50000.0
        # client_id is converted to string of client_order_index hash
        assert order.client_id == str(abs(hash("test_order_1")) % (10**9))
        assert order.status == "NEW"

    @pytest.mark.asyncio
    async def test_create_order_with_ioc(self, broker, mock_client, mock_instrument_loader):
        """Test creating order with IOC time in force"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="limit",
            amount=1.0,
            price=49000.0,
            client_id="test_ioc",
            time_in_force="ioc",
        )

        # Verify TIF is IOC
        call_kwargs = mock_client.signer_client.sign_create_order.call_args[1]
        assert call_kwargs["time_in_force"] == 0  # IOC

        assert order.time_in_force == "ioc"

    @pytest.mark.asyncio
    async def test_create_order_post_only(self, broker, mock_client, mock_instrument_loader):
        """Test creating post-only order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        order = await broker._create_order(
            instrument=btc,
            order_side="buy",
            order_type="limit",
            amount=1.0,
            price=49000.0,
            client_id="test_post",
            time_in_force="post_only",
        )

        # Verify post_only flag
        call_kwargs = mock_client.signer_client.sign_create_order.call_args[1]
        assert call_kwargs["time_in_force"] == 2  # POST_ONLY

    @pytest.mark.asyncio
    async def test_create_order_reduce_only(self, broker, mock_client, mock_instrument_loader):
        """Test creating reduce-only market order"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        order = await broker._create_order(
            instrument=btc,
            order_side="sell",
            order_type="market",
            amount=0.5,
            price=None,
            client_id="test_reduce",
            time_in_force="gtc",
            reduce_only=True,
        )

        # Verify reduce_only flag and market order parameters
        call_kwargs = mock_client.signer_client.sign_create_order.call_args[1]
        assert call_kwargs["reduce_only"] == 1  # True as int
        assert call_kwargs["time_in_force"] == 0  # IOC for market orders
        assert call_kwargs["order_expiry"] == 0  # IOC expiry

        # Verify slippage protection for SELL order
        # mid_price = 50000, default slippage = 5%, sell order
        # protected_price = 50000 * 0.95 = 47500
        expected_protected_price = 47500.0
        assert call_kwargs["price"] == int(expected_protected_price * 1e1)

        assert order.options.get("reduce_only") is True
        assert order.price == expected_protected_price

    @pytest.mark.asyncio
    async def test_create_order_generates_client_id(self, broker, mock_client, mock_instrument_loader):
        """Test that client_id is generated if not provided"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

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

        # Verify slippage protection was applied (BUY order)
        expected_protected_price = 52500.0  # 50000 * 1.05
        assert order.price == expected_protected_price

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

        # Mock signing error
        mock_client.signer_client.sign_create_order.return_value = (None, "API Error: Insufficient funds")

        with pytest.raises(InvalidOrderParameters, match="Order signing failed"):
            await broker._create_order(
                instrument=btc,
                order_side="buy",
                order_type="market",
                amount=1.0,
                price=None,
                client_id="test_err",
                time_in_force="gtc",
            )


class TestCancelOrder:
    """Test order cancellation"""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, broker, mock_client, mock_ws_manager, mock_account, mock_instrument_loader):
        """Test successful order cancellation"""
        btc = mock_instrument_loader.instruments["BTCUSDC"]

        # Mock existing order
        import numpy as np

        from qubx.core.basics import Order

        existing_order = Order(
            id="test_tx_123",
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
        mock_account.get_orders.return_value = {"test_tx_123": existing_order}

        # Store the client_order_index that would have been set during creation
        broker._client_order_indices["client_1"] = abs(hash("client_1")) % (10**9)

        result = await broker._cancel_order("test_tx_123")

        # Verify signing was called
        mock_client.signer_client.sign_cancel_order.assert_called_once()
        call_kwargs = mock_client.signer_client.sign_cancel_order.call_args[1]
        assert call_kwargs["market_index"] == 0
        assert call_kwargs["order_index"] == broker._client_order_indices["client_1"]

        # Verify WebSocket submission
        mock_ws_manager.send_tx.assert_called_once()

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
            id="test_tx_123",
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
        mock_account.get_orders.return_value = {"test_tx_123": existing_order}

        # Store the client_order_index
        broker._client_order_indices["client_1"] = abs(hash("client_1")) % (10**9)

        # Mock cancellation error
        mock_client.signer_client.sign_cancel_order.return_value = (None, "Order already cancelled")

        result = await broker._cancel_order("test_tx_123")

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

    @pytest.mark.asyncio
    async def test_update_order(self, broker, mock_client, mock_ws_manager, mock_account, mock_instrument_loader):
        """Test updating order via native modification"""
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
            client_id="client_123",
            options={"reduce_only": False},
        )
        mock_account.get_orders.return_value = {"123": existing_order}

        # Store the client_order_index that would have been set during creation
        broker._client_order_indices["client_123"] = abs(hash("client_123")) % (10**9)

        # Mock sign_modify_order
        mock_client.signer_client.sign_modify_order = Mock(return_value=('{"hash": "0xmodify"}', None))

        # Update order
        result = await broker._modify_order("123", price=49000.0, amount=0.5)

        # Verify sign_modify_order was called
        mock_client.signer_client.sign_modify_order.assert_called_once()
        call_kwargs = mock_client.signer_client.sign_modify_order.call_args[1]
        assert call_kwargs["market_index"] == 0
        assert call_kwargs["order_index"] == broker._client_order_indices["client_123"]
        assert call_kwargs["base_amount"] == int(0.5 * 1e5)  # 0.5 * 10^5 (size_decimals=5)
        assert call_kwargs["price"] == int(49000.0 * 1e1)  # 49000 * 10^1 (price_decimals=1)
        assert call_kwargs["trigger_price"] == 0

        # Verify WebSocket submission
        mock_ws_manager.send_tx.assert_called_once()

        # Verify returned order has updated values
        assert result.price == 49000.0
        assert result.quantity == 0.5
