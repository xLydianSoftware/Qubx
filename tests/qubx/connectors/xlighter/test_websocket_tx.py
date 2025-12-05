"""Tests for WebSocket transaction submission"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.connectors.xlighter.constants import TX_TYPE_CANCEL_ORDER, TX_TYPE_CREATE_ORDER
from qubx.connectors.xlighter.websocket import LighterWebSocketManager


@pytest.fixture
def ws_manager():
    """Create a WebSocket manager for testing"""
    # Create a mock client with signer_client for auth token generation
    mock_client = MagicMock()
    mock_client.signer_client = MagicMock()
    mock_client.signer_client.create_auth_token_with_expiry = MagicMock(
        return_value=("mock_token", None)
    )
    return LighterWebSocketManager(client=mock_client, testnet=False)


class TestWebSocketTransactionSending:
    """Test WebSocket transaction sending methods"""

    @pytest.mark.asyncio
    async def test_send_tx_creates_correct_message(self, ws_manager):
        """Test that send_tx creates correct WebSocket message"""
        # Mock the send method
        ws_manager.send = AsyncMock()

        # Sample tx_info from signer
        tx_info = '{"hash": "0xabc123", "nonce": 1, "account_index": 123}'
        tx_type = TX_TYPE_CREATE_ORDER
        tx_id = "test_tx_123"

        # Send transaction
        response = await ws_manager.send_tx(tx_type=tx_type, tx_info=tx_info, tx_id=tx_id)

        # Verify send was called with correct message
        ws_manager.send.assert_called_once()
        sent_message = ws_manager.send.call_args[0][0]

        assert sent_message["type"] == "jsonapi/sendtx"
        assert sent_message["data"]["id"] == tx_id
        assert sent_message["data"]["tx_type"] == tx_type
        assert sent_message["data"]["tx_info"]["hash"] == "0xabc123"
        assert sent_message["data"]["tx_info"]["nonce"] == 1

        # Verify response
        assert response["tx_id"] == tx_id
        assert response["tx_type"] == tx_type
        assert response["status"] == "sent"

    @pytest.mark.asyncio
    async def test_send_tx_with_dict_tx_info(self, ws_manager):
        """Test send_tx with tx_info already as dict"""
        ws_manager.send = AsyncMock()

        # tx_info as dict (not string)
        tx_info_dict = {"hash": "0xdef456", "nonce": 2}
        tx_type = TX_TYPE_CANCEL_ORDER

        response = await ws_manager.send_tx(tx_type=tx_type, tx_info=tx_info_dict)

        # Verify send was called
        ws_manager.send.assert_called_once()
        sent_message = ws_manager.send.call_args[0][0]

        assert sent_message["data"]["tx_info"]["hash"] == "0xdef456"
        assert response["status"] == "sent"

    @pytest.mark.asyncio
    async def test_send_tx_auto_generates_id(self, ws_manager):
        """Test that send_tx auto-generates ID if not provided"""
        ws_manager.send = AsyncMock()

        tx_info = '{"hash": "0x123"}'
        response = await ws_manager.send_tx(tx_type=TX_TYPE_CREATE_ORDER, tx_info=tx_info)

        # Verify ID was generated (UUID format)
        tx_id = response["tx_id"]
        assert len(tx_id) == 36  # UUID length
        assert "-" in tx_id

    @pytest.mark.asyncio
    async def test_send_batch_tx_creates_correct_message(self, ws_manager):
        """Test that send_batch_tx creates correct WebSocket message"""
        ws_manager.send = AsyncMock()

        # Multiple transactions
        tx_types = [TX_TYPE_CREATE_ORDER, TX_TYPE_CREATE_ORDER, TX_TYPE_CANCEL_ORDER]
        tx_infos = [
            '{"hash": "0x1", "nonce": 1}',
            '{"hash": "0x2", "nonce": 2}',
            '{"hash": "0x3", "nonce": 3}',
        ]
        batch_id = "batch_123"

        response = await ws_manager.send_batch_tx(tx_types=tx_types, tx_infos=tx_infos, batch_id=batch_id)

        # Verify send was called with correct message
        ws_manager.send.assert_called_once()
        sent_message = ws_manager.send.call_args[0][0]

        assert sent_message["type"] == "jsonapi/sendtxbatch"
        assert sent_message["data"]["id"] == batch_id
        assert sent_message["data"]["tx_types"] == tx_types
        assert len(sent_message["data"]["tx_infos"]) == 3
        assert sent_message["data"]["tx_infos"][0]["hash"] == "0x1"
        assert sent_message["data"]["tx_infos"][2]["hash"] == "0x3"

        # Verify response
        assert response["batch_id"] == batch_id
        assert response["count"] == 3
        assert response["status"] == "sent"

    @pytest.mark.asyncio
    async def test_send_batch_tx_validates_length_mismatch(self, ws_manager):
        """Test that send_batch_tx raises error if tx_types and tx_infos have different lengths"""
        tx_types = [TX_TYPE_CREATE_ORDER, TX_TYPE_CREATE_ORDER]
        tx_infos = ['{"hash": "0x1"}']  # Only 1 tx_info

        with pytest.raises(ValueError, match="must have same length"):
            await ws_manager.send_batch_tx(tx_types=tx_types, tx_infos=tx_infos)

    @pytest.mark.asyncio
    async def test_send_batch_tx_validates_max_size(self, ws_manager):
        """Test that send_batch_tx raises error if batch exceeds 50 transactions"""
        tx_types = [TX_TYPE_CREATE_ORDER] * 51
        tx_infos = ['{"hash": "0x1"}'] * 51

        with pytest.raises(ValueError, match="cannot exceed 50"):
            await ws_manager.send_batch_tx(tx_types=tx_types, tx_infos=tx_infos)

    @pytest.mark.asyncio
    async def test_send_batch_tx_validates_empty_batch(self, ws_manager):
        """Test that send_batch_tx raises error for empty batch"""
        with pytest.raises(ValueError, match="Cannot send empty batch"):
            await ws_manager.send_batch_tx(tx_types=[], tx_infos=[])

    @pytest.mark.asyncio
    async def test_send_batch_tx_auto_generates_id(self, ws_manager):
        """Test that send_batch_tx auto-generates batch ID if not provided"""
        ws_manager.send = AsyncMock()

        tx_types = [TX_TYPE_CREATE_ORDER]
        tx_infos = ['{"hash": "0x1"}']

        response = await ws_manager.send_batch_tx(tx_types=tx_types, tx_infos=tx_infos)

        # Verify batch ID was generated (UUID format)
        batch_id = response["batch_id"]
        assert len(batch_id) == 36  # UUID length
        assert "-" in batch_id


class TestBrokerWebSocketOrders:
    """Test broker order creation via WebSocket"""

    @pytest.mark.asyncio
    async def test_broker_create_order_uses_websocket(self):
        """Test that broker creates order via WebSocket submission"""
        import asyncio
        from unittest.mock import Mock

        from qubx.connectors.xlighter.broker import LighterBroker
        from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
        from tests.qubx.core.utils_test import DummyTimeProvider

        # Create mocks
        mock_client = MagicMock()
        mock_client.signer_client = MagicMock()
        mock_client.signer_client.sign_create_order = MagicMock(return_value=('{"hash": "0xabc123", "nonce": 1}', None))

        mock_ws_manager = MagicMock()
        mock_ws_manager.send_tx = AsyncMock(return_value={"tx_id": "order_123", "status": "sent"})
        mock_ws_manager.next_nonce = AsyncMock(return_value=1)  # Add async next_nonce

        mock_channel = CtrlChannel("test")
        mock_time_provider = DummyTimeProvider()
        mock_account = Mock()
        mock_account.process_order = Mock()  # Add process_order mock

        # Mock get_position to return proper Position object
        from qubx.core.basics import Position

        def get_position_mock(instrument):
            return Position(instrument=instrument)
        mock_account.get_position = Mock(side_effect=get_position_mock)

        mock_data_provider = Mock()
        mock_loop = asyncio.get_event_loop()

        # Create broker
        broker = LighterBroker(
            client=mock_client,
            ws_manager=mock_ws_manager,
            channel=mock_channel,
            time_provider=mock_time_provider,
            account=mock_account,
            data_provider=mock_data_provider,
            loop=mock_loop,
        )

        # Create test instrument (mimicking BTC market decimals)
        # BTC market has: price_decimals=1 (tick_size=0.1), size_decimals=5 (lot_size=0.00001)
        # exchange_symbol="0" is the numeric market_id
        instrument = Instrument(
            symbol="BTCUSDC",
            exchange="LIGHTER",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            base="BTC",
            quote="USDC",
            settle="USDC",
            exchange_symbol="0",  # market_id as string
            tick_size=0.1,  # price_decimals = 1
            lot_size=0.00001,  # size_decimals = 5
            min_size=0.00001,
        )

        # Send order
        await broker._create_order_ws(
            instrument=instrument,
            order_side="BUY",
            order_type="limit",
            amount=0.1,
            price=40000.0,
            client_id="123461",
            time_in_force="gtc",
        )

        # Verify signing was called
        mock_client.signer_client.sign_create_order.assert_called_once()
        call_args = mock_client.signer_client.sign_create_order.call_args[1]
        assert call_args["market_index"] == 0
        # BTC market: size_decimals=5, price_decimals=1
        assert call_args["base_amount"] == int(0.1 * 1e5)  # 0.1 * 10^5 = 10000
        assert call_args["price"] == int(40000 * 1e1)  # 40000 * 10^1 = 400000
        assert call_args["is_ask"] == False  # BUY = not ask

        # Verify WebSocket submission was called
        mock_ws_manager.send_tx.assert_called_once()
        ws_call_args = mock_ws_manager.send_tx.call_args[1]
        assert ws_call_args["tx_type"] == TX_TYPE_CREATE_ORDER
        assert ws_call_args["tx_info"] == '{"hash": "0xabc123", "nonce": 1}'
        # tx_id is the client_id
        assert "tx_id" in ws_call_args
        assert ws_call_args["tx_id"] == "123461"

    @pytest.mark.asyncio
    async def test_broker_cancel_order_uses_websocket(self):
        """Test that broker cancels order via WebSocket submission"""
        import asyncio
        from unittest.mock import Mock

        from qubx.connectors.xlighter.broker import LighterBroker
        from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType, Order
        from tests.qubx.core.utils_test import DummyTimeProvider

        # Create mocks
        mock_client = MagicMock()
        mock_client.signer_client = MagicMock()
        mock_client.signer_client.sign_cancel_order = MagicMock(return_value=('{"hash": "0xcancel"}', None))

        mock_ws_manager = MagicMock()
        mock_ws_manager.send_tx = AsyncMock(return_value={"tx_id": "cancel_123", "status": "sent"})
        mock_ws_manager.next_nonce = AsyncMock(return_value=1)  # Add async next_nonce

        mock_channel = CtrlChannel("test")
        mock_time_provider = DummyTimeProvider()
        mock_account = Mock()
        mock_data_provider = Mock()
        mock_loop = asyncio.get_event_loop()

        # Create test instrument and order
        # exchange_symbol="0" is the numeric market_id
        instrument = Instrument(
            symbol="BTCUSDC",
            exchange="LIGHTER",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            base="BTC",
            quote="USDC",
            settle="USDC",
            exchange_symbol="0",  # market_id as string
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        test_order = Order(
            id="order_123",
            type="LIMIT",
            instrument=instrument,
            time=0,
            quantity=0.1,
            price=40000.0,
            side="BUY",
            status="OPEN",
            time_in_force="gtc",
        )

        mock_account.get_orders = Mock(return_value={"order_123": test_order})

        # Create broker
        broker = LighterBroker(
            client=mock_client,
            ws_manager=mock_ws_manager,
            channel=mock_channel,
            time_provider=mock_time_provider,
            account=mock_account,
            data_provider=mock_data_provider,
            loop=mock_loop,
        )

        # Cancel order - pass the Order object, not just the ID
        result = await broker._cancel_order(test_order)

        # Verify signing was called
        mock_client.signer_client.sign_cancel_order.assert_called_once()
        call_args = mock_client.signer_client.sign_cancel_order.call_args[1]
        assert call_args["market_index"] == 0

        # Verify WebSocket submission was called
        mock_ws_manager.send_tx.assert_called_once()
        ws_call_args = mock_ws_manager.send_tx.call_args[1]
        assert ws_call_args["tx_type"] == TX_TYPE_CANCEL_ORDER
        assert ws_call_args["tx_info"] == '{"hash": "0xcancel"}'

        # Verify result
        assert result
