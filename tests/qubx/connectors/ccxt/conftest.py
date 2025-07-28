"""
Shared fixtures for CCXT data provider tests.

This module provides common test fixtures and utilities for testing
the new component-based CCXT data provider architecture.
"""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ccxt.pro import Exchange

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.connectors.ccxt.subscription_orchestrator import SubscriptionOrchestrator
from qubx.connectors.ccxt.warmup_service import WarmupService
from qubx.connectors.ccxt.handlers import DataTypeHandlerFactory
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
from tests.qubx.core.utils_test import DummyTimeProvider


@pytest.fixture
def mock_instruments():
    """Create test instruments for various markets."""
    return [
        Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTC/USDT:USDT",
            tick_size=0.1,
            lot_size=0.001,
            min_size=0.001,
        ),
        Instrument(
            symbol="ETHUSDT", 
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETH/USDT:USDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        ),
        Instrument(
            symbol="ADAUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ADA", 
            quote="USDT",
            settle="USDT",
            exchange_symbol="ADA/USDT",
            tick_size=0.0001,
            lot_size=1.0,
            min_size=1.0,
        ),
    ]


@pytest.fixture
def mock_exchange():
    """Create a mock CCXT exchange for testing."""
    exchange = MagicMock(spec=Exchange)
    exchange.name = "mock_exchange" 
    exchange.asyncio_loop = asyncio.new_event_loop()
    exchange.apiKey = "test_api_key"
    exchange.sandbox = True
    
    # Mock exchange methods
    exchange.watch_ohlcv_for_symbols = AsyncMock()
    exchange.watch_trades_for_symbols = AsyncMock()
    exchange.watch_order_book_for_symbols = AsyncMock()
    exchange.watch_bids_asks = AsyncMock()
    exchange.un_watch_trades_for_symbols = AsyncMock()
    exchange.fetch_ohlcv = AsyncMock()
    exchange.fetch_trades = AsyncMock()
    exchange.find_timeframe = MagicMock(return_value="1m")
    exchange.close = AsyncMock()
    
    return exchange


@pytest.fixture
def mock_ctrl_channel():
    """Create a mock control channel for testing."""
    channel = MagicMock(spec=CtrlChannel)
    channel.send = MagicMock()
    channel.send_async = AsyncMock()
    return channel


@pytest.fixture
def mock_async_thread_loop():
    """Create a mock AsyncThreadLoop for controlled testing."""
    
    class MockAsyncThreadLoop:
        def __init__(self):
            self.submitted_tasks = []
            self.running_futures = {}

        def submit(self, coro):
            """Submit a coroutine and return a mock Future."""
            future = MagicMock(spec=Future)
            future.running.return_value = True
            future.cancel = MagicMock()
            future.result = MagicMock(return_value=None)

            # Store the coroutine for inspection
            self.submitted_tasks.append(coro)
            self.running_futures[id(future)] = {
                "future": future, 
                "coro": coro, 
                "cancelled": False
            }

            return future

        def cancel_future(self, future):
            """Mark a future as cancelled."""
            future_id = id(future)
            if future_id in self.running_futures:
                self.running_futures[future_id]["cancelled"] = True
                future.running.return_value = False
                future.cancel.return_value = True

    return MockAsyncThreadLoop()


@pytest.fixture
def time_provider():
    """Create a dummy time provider for testing."""
    return DummyTimeProvider()


@pytest.fixture
def subscription_manager():
    """Create a SubscriptionManager instance for testing."""
    return SubscriptionManager()


@pytest.fixture
def connection_manager(subscription_manager):
    """Create a ConnectionManager instance for testing."""
    return ConnectionManager(
        exchange_id="test_exchange",
        max_ws_retries=3,
        subscription_manager=subscription_manager
    )


@pytest.fixture
def subscription_orchestrator(subscription_manager, connection_manager):
    """Create a SubscriptionOrchestrator instance for testing."""
    return SubscriptionOrchestrator(
        exchange_id="test_exchange",
        subscription_manager=subscription_manager,
        connection_manager=connection_manager
    )


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider for handler testing."""
    data_provider = MagicMock()
    data_provider._exchange_id = "test_exchange"
    data_provider._last_quotes = {}
    data_provider.channel = MagicMock()
    data_provider.time_provider = DummyTimeProvider()
    data_provider._time_msec_nbars_back = MagicMock(return_value=1640995200000)
    data_provider._get_exch_timeframe = MagicMock(return_value="1m")
    data_provider._get_exch_symbol = MagicMock(return_value="BTC/USDT:USDT")
    return data_provider


@pytest.fixture
def handler_factory(mock_data_provider, mock_exchange):
    """Create a DataTypeHandlerFactory instance for testing."""
    return DataTypeHandlerFactory(
        data_provider=mock_data_provider,
        exchange=mock_exchange,
        exchange_id="test_exchange"
    )


@pytest.fixture
def warmup_service(handler_factory, mock_ctrl_channel, mock_async_thread_loop):
    """Create a WarmupService instance for testing."""
    return WarmupService(
        handler_factory=handler_factory,
        channel=mock_ctrl_channel,
        exchange_id="test_exchange",
        async_loop=mock_async_thread_loop,
        warmup_timeout=30
    )


# Sample data fixtures for testing
@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    return [
        [1640995200000, 47000.0, 47100.0, 46900.0, 47050.0, 1000.0],
        [1640995260000, 47050.0, 47150.0, 46950.0, 47100.0, 1100.0],
        [1640995320000, 47100.0, 47200.0, 47000.0, 47150.0, 1200.0],
    ]


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return [
        {
            "id": "123456789",
            "timestamp": 1640995200000,
            "datetime": "2022-01-01T00:00:00.000Z",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.1,
            "price": 47000.0,
            "cost": 4700.0,
            "fee": {"cost": 4.7, "currency": "USDT"}
        }
    ]


@pytest.fixture 
def sample_orderbook_data():
    """Sample orderbook data for testing."""
    return {
        "symbol": "BTC/USDT:USDT",
        "bids": [[47000.0, 0.5], [46999.0, 1.0], [46998.0, 2.0]],
        "asks": [[47001.0, 0.8], [47002.0, 1.5], [47003.0, 2.5]],
        "timestamp": 1640995200000,
        "datetime": "2022-01-01T00:00:00.000Z"
    }