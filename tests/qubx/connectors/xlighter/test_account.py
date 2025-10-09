"""Tests for LighterAccountProcessor"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
from qubx.connectors.xlighter.account import LighterAccountProcessor
from qubx.connectors.xlighter.client import LighterClient
from qubx.connectors.xlighter.instruments import LighterInstrumentLoader
from qubx.connectors.xlighter.websocket import LighterWebSocketManager


@pytest.fixture
def mock_time_provider():
    """Mock time provider"""
    provider = MagicMock()
    provider.now.return_value = datetime(2025, 10, 9, 12, 0, 0).timestamp() * 1_000_000_000
    return provider


@pytest.fixture
def mock_channel():
    """Mock control channel"""
    channel = MagicMock()
    channel.control = MagicMock()
    channel.control.is_set = MagicMock(return_value=True)
    return channel


@pytest.fixture
def mock_client():
    """Mock LighterClient"""
    return MagicMock(spec=LighterClient)


@pytest.fixture
def mock_instrument_loader():
    """Mock LighterInstrumentLoader with test instruments"""
    loader = MagicMock(spec=LighterInstrumentLoader)

    # Create test instruments
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

    # Setup mappings
    loader.market_id_to_ticker = {
        0: "BTC-USDC",
        1: "ETH-USDC",
    }
    loader.ticker_to_market_id = {
        "BTC-USDC": 0,
        "ETH-USDC": 1,
    }
    loader.instruments_cache = {
        "XLIGHTER:SWAP:BTC-USDC": btc_instrument,
        "XLIGHTER:SWAP:ETH-USDC": eth_instrument,
    }

    return loader


@pytest.fixture
def mock_ws_manager():
    """Mock WebSocket manager"""
    manager = MagicMock(spec=LighterWebSocketManager)
    manager.subscribe_account_all = AsyncMock()
    manager.subscribe_user_stats = AsyncMock()
    manager.subscribe_executed_transaction = AsyncMock()
    return manager


@pytest.fixture
def account_processor(
    mock_client, mock_instrument_loader, mock_ws_manager, mock_channel, mock_time_provider
):
    """Create LighterAccountProcessor for testing"""
    return LighterAccountProcessor(
        account_id="225671",
        client=mock_client,
        instrument_loader=mock_instrument_loader,
        ws_manager=mock_ws_manager,
        channel=mock_channel,
        time_provider=mock_time_provider,
        base_currency="USDC",
        initial_capital=100_000,
    )


class TestLighterAccountProcessorInit:
    """Test initialization"""

    def test_initialization(self, account_processor):
        """Test basic initialization"""
        assert account_processor.account_id == "225671"
        assert account_processor.base_currency == "USDC"
        assert account_processor._lighter_account_index == 225671
        assert not account_processor._is_running

    def test_set_subscription_manager(self, account_processor):
        """Test setting subscription manager"""
        mock_manager = MagicMock()
        account_processor.set_subscription_manager(mock_manager)
        assert account_processor._subscription_manager == mock_manager


class TestLighterAccountProcessorLifecycle:
    """Test start/stop lifecycle"""

    @pytest.mark.asyncio
    async def test_start(self, account_processor, mock_ws_manager):
        """Test starting subscriptions"""
        # Start processor
        account_processor.start()

        # Should be running
        assert account_processor._is_running

        # Should have created subscription tasks
        await asyncio.sleep(0.1)  # Let tasks start

        # Verify subscriptions were called
        mock_ws_manager.subscribe_account_all.assert_called_once()
        mock_ws_manager.subscribe_user_stats.assert_called_once()
        mock_ws_manager.subscribe_executed_transaction.assert_called_once()

        # Cleanup
        account_processor.stop()

    def test_stop(self, account_processor):
        """Test stopping subscriptions"""
        account_processor.start()
        assert account_processor._is_running

        account_processor.stop()
        assert not account_processor._is_running


class TestAccountAllHandler:
    """Test account_all message handling"""

    @pytest.mark.asyncio
    async def test_handle_positions(self, account_processor):
        """Test handling position updates"""
        message = {
            "channel": "account_all/225671",
            "account": {
                "positions": [
                    {
                        "market_index": 0,
                        "symbol": "BTC-USDC",
                        "position": "1.5",
                        "sign": 1,  # Long
                        "avg_entry_price": "43500.00",
                        "position_value": "65250.00",
                    },
                    {
                        "market_index": 1,
                        "symbol": "ETH-USDC",
                        "position": "10.0",
                        "sign": -1,  # Short
                        "avg_entry_price": "2300.00",
                        "position_value": "23000.00",
                    },
                ]
            },
        }

        await account_processor._handle_account_all_message(message)

        # Check positions were updated
        positions = account_processor.get_positions()
        assert len(positions) == 2

        # Check BTC position
        btc_instrument = account_processor.instrument_loader.instruments_cache["XLIGHTER:SWAP:BTC-USDC"]
        btc_pos = account_processor.get_position(btc_instrument)
        assert btc_pos.quantity == 1.5
        assert btc_pos.position_avg_price_funds == 43500.00

        # Check ETH position (short)
        eth_instrument = account_processor.instrument_loader.instruments_cache["XLIGHTER:SWAP:ETH-USDC"]
        eth_pos = account_processor.get_position(eth_instrument)
        assert eth_pos.quantity == -10.0
        assert eth_pos.position_avg_price_funds == 2300.00

    @pytest.mark.asyncio
    async def test_handle_balance(self, account_processor):
        """Test handling balance updates"""
        message = {
            "channel": "account_all/225671",
            "account": {
                "balance": "125000.50",
                "positions": [],
            },
        }

        await account_processor._handle_account_all_message(message)

        # Check balance was updated
        balances = account_processor.get_balances()
        assert "USDC" in balances
        assert balances["USDC"].total == 125000.50

    @pytest.mark.asyncio
    async def test_handle_orders(self, account_processor):
        """Test handling order updates"""
        message = {
            "channel": "account_all/225671",
            "account": {
                "orders": [
                    {
                        "order_id": "123456",
                        "client_order_id": "789",
                        "market_index": 0,
                        "is_ask": False,
                        "price": "43000.00",
                        "initial_base_amount": "1.0",
                        "remaining_base_amount": "0.5",
                        "timestamp": 1234567890,
                        "order_type": 0,  # Limit
                        "time_in_force": 1,  # GTC
                    }
                ],
                "positions": [],
            },
        }

        await account_processor._handle_account_all_message(message)

        # Check order was added
        orders = account_processor.get_orders()
        assert "123456" in orders

        order = orders["123456"]
        assert order.side == "BUY"
        assert order.price == 43000.00
        assert order.quantity == 1.0
        assert order.filled_quantity == 0.5
        assert order.status == "partially_filled"


class TestUserStatsHandler:
    """Test user_stats message handling"""

    @pytest.mark.asyncio
    async def test_handle_user_stats(self, account_processor):
        """Test handling user stats updates"""
        message = {
            "channel": "user_stats/225671",
            "stats": {
                "portfolio_value": "150000.00",
                "margin_usage": "45000.00",
                "available_balance": "105000.00",
                "leverage": "2.5",
            },
        }

        await account_processor._handle_user_stats_message(message)

        # Check balance was updated from portfolio value
        balances = account_processor.get_balances()
        assert balances["USDC"].total == 150000.00
        assert balances["USDC"].free == 105000.00
        assert balances["USDC"].locked == 45000.00


class TestExecutedTransactionHandler:
    """Test executed_transaction message handling"""

    @pytest.mark.asyncio
    async def test_handle_fills_buyer(self, account_processor):
        """Test handling fills when we are the buyer"""
        message = {
            "channel": "executed_transaction",
            "txs": [
                {
                    "tx_hash": "0xabc123",
                    "l1_address": "0x...",
                    "account_index": 225671,
                    "trades": {
                        "0": [  # BTC-USDC
                            {
                                "trade_id": 212690112,
                                "market_id": 0,
                                "size": "0.5",
                                "price": "43500.00",
                                "is_maker_ask": False,
                                "timestamp": 1760040869198,
                                "bid_order_id": "123",
                                "ask_order_id": "456",
                                "bid_account_id": 225671,
                                "ask_account_id": 999,
                            }
                        ]
                    },
                }
            ],
        }

        await account_processor._handle_executed_transaction_message(message)

        # Check transaction was processed
        assert "0xabc123" in account_processor._processed_tx_hashes

    @pytest.mark.asyncio
    async def test_handle_fills_seller(self, account_processor):
        """Test handling fills when we are the seller"""
        message = {
            "channel": "executed_transaction",
            "txs": [
                {
                    "tx_hash": "0xdef456",
                    "l1_address": "0x...",
                    "account_index": 225671,
                    "trades": {
                        "1": [  # ETH-USDC
                            {
                                "trade_id": 212690113,
                                "market_id": 1,
                                "size": "2.0",
                                "price": "2300.00",
                                "is_maker_ask": True,
                                "timestamp": 1760040869199,
                                "bid_order_id": "789",
                                "ask_order_id": "101112",
                                "bid_account_id": 888,
                                "ask_account_id": 225671,
                            }
                        ]
                    },
                }
            ],
        }

        await account_processor._handle_executed_transaction_message(message)

        # Check transaction was processed
        assert "0xdef456" in account_processor._processed_tx_hashes

    @pytest.mark.asyncio
    async def test_ignore_other_account_fills(self, account_processor):
        """Test that fills for other accounts are ignored"""
        message = {
            "channel": "executed_transaction",
            "txs": [
                {
                    "tx_hash": "0xghi789",
                    "l1_address": "0x...",
                    "account_index": 999999,  # Different account
                    "trades": {
                        "0": [
                            {
                                "trade_id": 212690114,
                                "market_id": 0,
                                "size": "1.0",
                                "price": "43000.00",
                                "bid_account_id": 999999,
                                "ask_account_id": 888,
                            }
                        ]
                    },
                }
            ],
        }

        await account_processor._handle_executed_transaction_message(message)

        # Check transaction was NOT processed
        assert "0xghi789" not in account_processor._processed_tx_hashes

    @pytest.mark.asyncio
    async def test_deduplicate_transactions(self, account_processor):
        """Test that duplicate transactions are not processed twice"""
        message = {
            "channel": "executed_transaction",
            "txs": [
                {
                    "tx_hash": "0xjkl012",
                    "account_index": 225671,
                    "trades": {
                        "0": [
                            {
                                "trade_id": 212690115,
                                "market_id": 0,
                                "size": "0.1",
                                "price": "44000.00",
                                "bid_account_id": 225671,
                                "ask_account_id": 777,
                                "bid_order_id": "999",
                                "ask_order_id": "888",
                            }
                        ]
                    },
                }
            ],
        }

        # Process first time
        await account_processor._handle_executed_transaction_message(message)
        assert "0xjkl012" in account_processor._processed_tx_hashes

        # Process second time (duplicate)
        initial_count = len(account_processor._processed_tx_hashes)
        await account_processor._handle_executed_transaction_message(message)

        # Should still only have the same hash once (sets handle duplicates automatically)
        assert len(account_processor._processed_tx_hashes) == initial_count
        assert "0xjkl012" in account_processor._processed_tx_hashes


class TestHelperMethods:
    """Test helper methods"""

    def test_get_instrument_for_market_id(self, account_processor):
        """Test getting instrument from market ID"""
        btc_instrument = account_processor._get_instrument_for_market_id(0)
        assert btc_instrument is not None
        assert btc_instrument.symbol == "BTC-USDC"

        eth_instrument = account_processor._get_instrument_for_market_id(1)
        assert eth_instrument is not None
        assert eth_instrument.symbol == "ETH-USDC"

        # Unknown market ID
        unknown = account_processor._get_instrument_for_market_id(999)
        assert unknown is None

    def test_convert_lighter_trades_to_deals(self, account_processor):
        """Test converting Lighter trades to Deals"""
        btc_instrument = account_processor.instrument_loader.instruments_cache["XLIGHTER:SWAP:BTC-USDC"]

        trades = [
            {
                "trade_id": 100,
                "market_id": 0,
                "size": "0.5",
                "price": "43500.00",
                "is_maker_ask": False,
                "timestamp": 1234567890,
                "bid_order_id": "123",
                "ask_order_id": "456",
                "bid_account_id": 225671,  # We are buyer
                "ask_account_id": 999,
            },
            {
                "trade_id": 101,
                "market_id": 0,
                "size": "0.3",
                "price": "43600.00",
                "is_maker_ask": True,
                "timestamp": 1234567891,
                "bid_order_id": "789",
                "ask_order_id": "101112",
                "bid_account_id": 888,
                "ask_account_id": 225671,  # We are seller
            },
        ]

        tx = {"tx_hash": "0xtest"}

        deals = account_processor._convert_lighter_trades_to_deals(trades, btc_instrument, tx)

        assert len(deals) == 2

        # Check first deal (buy)
        assert deals[0].side == "BUY"
        assert deals[0].quantity == 0.5
        assert deals[0].price == 43500.00
        assert deals[0].trade_id == "100"
        assert deals[0].order_id == "123"

        # Check second deal (sell)
        assert deals[1].side == "SELL"
        assert deals[1].quantity == 0.3
        assert deals[1].price == 43600.00
        assert deals[1].trade_id == "101"
        assert deals[1].order_id == "101112"


class TestAccountViewer:
    """Test IAccountViewer methods"""

    def test_get_base_currency(self, account_processor):
        """Test getting base currency"""
        assert account_processor.get_base_currency() == "USDC"

    @pytest.mark.asyncio
    async def test_get_positions(self, account_processor):
        """Test getting positions"""
        # Setup position
        message = {
            "channel": "account_all/225671",
            "account": {
                "positions": [
                    {
                        "market_index": 0,
                        "symbol": "BTC-USDC",
                        "position": "1.0",
                        "sign": 1,
                        "avg_entry_price": "43000.00",
                    }
                ]
            },
        }

        await account_processor._handle_account_all_message(message)

        positions = account_processor.get_positions()
        assert len(positions) == 1

    def test_get_orders(self, account_processor):
        """Test getting orders"""
        # Initially empty
        orders = account_processor.get_orders()
        assert len(orders) == 0
