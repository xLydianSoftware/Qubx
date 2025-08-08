"""
Unit tests for OrderBookDataHandler.

Tests both bulk and individual subscription approaches for orderbook data,
ensuring proper integration with the refactored subscription orchestrator.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from qubx.connectors.ccxt.handlers.orderbook import OrderBookDataHandler
from qubx.core.basics import AssetType, CtrlChannel, DataType, Instrument, MarketType


@pytest.fixture
def btc_instrument():
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="binance",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def eth_instrument():
    return Instrument(
        symbol="ETHUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="binance",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def mock_exchange_with_bulk_support():
    """Mock exchange that supports bulk orderbook watching."""
    exchange = Mock()
    exchange.has = {"watchOrderBookForSymbols": True}
    exchange.watch_order_book_for_symbols = AsyncMock()
    exchange.un_watch_order_book_for_symbols = AsyncMock()
    exchange.name = "binance"
    return exchange


@pytest.fixture
def mock_exchange_without_bulk_support():
    """Mock exchange that doesn't support bulk orderbook watching."""
    exchange = Mock()
    exchange.has = {"watchOrderBookForSymbols": False}
    exchange.watch_order_book = AsyncMock()
    exchange.un_watch_order_book = AsyncMock()
    exchange.name = "binance"
    return exchange


@pytest.fixture
def mock_data_provider():
    data_provider = Mock()
    data_provider.time_provider.time.return_value = np.datetime64("2024-01-01T00:00:00", "s")
    data_provider._health_monitor.record_data_arrival = Mock()
    data_provider.has_subscription.return_value = False  # No quote subscription exists
    data_provider._last_quotes = {}
    return data_provider


@pytest.fixture
def orderbook_channel():
    return CtrlChannel("orderbook_test")


@pytest.fixture
def sample_ccxt_orderbook():
    return {
        "symbol": "BTCUSDT",
        "bids": [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 0.5]],
        "asks": [[50001.0, 1.5], [50002.0, 1.0], [50003.0, 2.0]],
        "timestamp": 1640995200000,  # 2022-01-01T00:00:00Z
        "datetime": "2022-01-01T00:00:00Z",
        "nonce": None,
    }


class TestOrderBookHandlerBulkSubscriptions:
    """Test bulk subscription functionality."""

    def test_bulk_subscription_when_supported(
        self, mock_exchange_with_bulk_support, mock_data_provider, btc_instrument, orderbook_channel
    ):
        """Test that bulk subscription is used when exchange supports it."""
        instruments = {btc_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_with_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="bulk_orderbook_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
            depth=100,
            tick_size_pct=0.01,
        )

        # Should use bulk subscription (single subscriber_func, not individual)
        assert config.subscriber_func is not None
        assert config.instrument_subscribers is None
        assert config.requires_market_type_batching is True
        assert config.stream_name == "bulk_orderbook_stream"

    @pytest.mark.asyncio
    @patch("qubx.connectors.ccxt.handlers.orderbook.instrument_to_ccxt_symbol")
    @patch("qubx.connectors.ccxt.handlers.orderbook.ccxt_find_instrument")
    @patch("qubx.connectors.ccxt.handlers.orderbook.ccxt_convert_orderbook")
    async def test_bulk_subscription_processes_data(
        self,
        mock_convert,
        mock_find_instrument,
        mock_instrument_to_ccxt_symbol,
        mock_exchange_with_bulk_support,
        mock_data_provider,
        btc_instrument,
        orderbook_channel,
        sample_ccxt_orderbook,
    ):
        """Test that bulk subscription processes orderbook data correctly."""
        # Setup mocks
        mock_instrument_to_ccxt_symbol.return_value = "BTCUSDT"
        mock_find_instrument.return_value = btc_instrument
        mock_orderbook = Mock()
        mock_orderbook.time = np.datetime64("2024-01-01T00:00:00", "ns")
        mock_orderbook.to_quote.return_value = Mock()  # Mock the quote conversion
        mock_convert.return_value = mock_orderbook

        instruments = {btc_instrument}
        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_with_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Mock exchange response - should be called with ["BTCUSDT"]
        mock_exchange_with_bulk_support.watch_order_book_for_symbols.return_value = sample_ccxt_orderbook

        # Set up channel to capture sent data
        sent_data = []
        orderbook_channel.send = lambda data: sent_data.append(data)

        # Call the subscriber function (market type batching wrapper takes no arguments)
        await config.subscriber_func()

        # Verify exchange was called with the correct symbols
        mock_exchange_with_bulk_support.watch_order_book_for_symbols.assert_called_once_with(["BTCUSDT"])

        # Should emit orderbook data
        orderbook_data = [data for data in sent_data if data[1] == DataType.ORDERBOOK]
        assert len(orderbook_data) == 1

        # Verify the data
        instrument, data_type, orderbook, is_historical = orderbook_data[0]
        assert instrument.symbol == "BTCUSDT"
        assert data_type == DataType.ORDERBOOK
        assert orderbook == mock_orderbook
        assert not is_historical


class TestOrderBookHandlerIndividualSubscriptions:
    """Test individual subscription functionality."""

    def test_individual_subscription_when_no_bulk_support(
        self, mock_exchange_without_bulk_support, mock_data_provider, btc_instrument, orderbook_channel
    ):
        """Test that individual subscription is used when exchange doesn't support bulk."""
        instruments = {btc_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="individual_orderbook_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
            depth=50,
            tick_size_pct=0.005,
        )

        # Should use individual subscription
        assert config.subscriber_func is None
        assert config.instrument_subscribers is not None
        assert config.requires_market_type_batching is False
        assert config.stream_name == "individual_orderbook_stream"

        # Should have one subscriber per instrument
        assert len(config.instrument_subscribers) == len(instruments)
        assert btc_instrument in config.instrument_subscribers
        assert callable(config.instrument_subscribers[btc_instrument])

    def test_individual_subscription_creates_unsubscribers(
        self, mock_exchange_without_bulk_support, mock_data_provider, btc_instrument, eth_instrument, orderbook_channel
    ):
        """Test that individual subscription creates unsubscriber functions."""
        instruments = {btc_instrument, eth_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Should create unsubscribers for each instrument
        assert config.instrument_unsubscribers is not None
        assert len(config.instrument_unsubscribers) == len(instruments)
        assert btc_instrument in config.instrument_unsubscribers
        assert eth_instrument in config.instrument_unsubscribers
        assert callable(config.instrument_unsubscribers[btc_instrument])
        assert callable(config.instrument_unsubscribers[eth_instrument])

    @pytest.mark.asyncio
    @patch("qubx.connectors.ccxt.handlers.orderbook.instrument_to_ccxt_symbol")
    async def test_individual_subscriber_calls_correct_exchange_method(
        self,
        mock_instrument_to_ccxt_symbol,
        mock_exchange_without_bulk_support,
        mock_data_provider,
        btc_instrument,
        orderbook_channel,
        sample_ccxt_orderbook,
    ):
        """Test that individual subscriber calls watch_order_book for single instrument."""
        # Setup mock to return expected symbol format
        mock_instrument_to_ccxt_symbol.return_value = "BTCUSDT"

        instruments = {btc_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Mock exchange response
        mock_exchange_without_bulk_support.watch_order_book.return_value = sample_ccxt_orderbook

        # Get the individual subscriber for BTC
        btc_subscriber = config.instrument_subscribers[btc_instrument]

        # Mock the processing method to avoid conversion complexity
        with patch.object(handler, "_process_orderbook", return_value=True) as mock_process:
            # Call the individual subscriber
            await btc_subscriber()

            # Should call watch_order_book with correct symbol
            mock_exchange_without_bulk_support.watch_order_book.assert_called_once_with("BTCUSDT")

            # Should process the orderbook
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    @patch("qubx.connectors.ccxt.handlers.orderbook.instrument_to_ccxt_symbol")
    async def test_individual_unsubscriber_calls_correct_exchange_method(
        self,
        mock_instrument_to_ccxt_symbol,
        mock_exchange_without_bulk_support,
        mock_data_provider,
        btc_instrument,
        orderbook_channel,
    ):
        """Test that individual unsubscriber calls un_watch_order_book for single instrument."""
        # Setup mock to return expected symbol format
        mock_instrument_to_ccxt_symbol.return_value = "BTCUSDT"

        instruments = {btc_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Get the individual unsubscriber for BTC
        btc_unsubscriber = config.instrument_unsubscribers[btc_instrument]

        # Call the individual unsubscriber
        await btc_unsubscriber()

        # Should call un_watch_order_book with correct symbol
        mock_exchange_without_bulk_support.un_watch_order_book.assert_called_once_with("BTCUSDT")

    @pytest.mark.asyncio
    @patch("qubx.connectors.ccxt.handlers.orderbook.instrument_to_ccxt_symbol")
    @patch("qubx.connectors.ccxt.handlers.orderbook.ccxt_convert_orderbook")
    async def test_individual_subscriber_processes_data(
        self,
        mock_convert,
        mock_instrument_to_ccxt_symbol,
        mock_exchange_without_bulk_support,
        mock_data_provider,
        btc_instrument,
        orderbook_channel,
        sample_ccxt_orderbook,
    ):
        """Test that individual subscriber processes orderbook data correctly."""
        # Setup mocks
        mock_instrument_to_ccxt_symbol.return_value = "BTCUSDT"
        mock_orderbook = Mock()
        mock_orderbook.time = np.datetime64("2024-01-01T00:00:00", "ns")
        mock_orderbook.to_quote.return_value = Mock()  # Mock the quote conversion
        mock_convert.return_value = mock_orderbook

        instruments = {btc_instrument}
        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Mock exchange response
        mock_exchange_without_bulk_support.watch_order_book.return_value = sample_ccxt_orderbook

        # Set up channel to capture sent data
        sent_data = []
        orderbook_channel.send = lambda data: sent_data.append(data)

        # Get the individual subscriber for BTC
        btc_subscriber = config.instrument_subscribers[btc_instrument]

        # Call the individual subscriber
        await btc_subscriber()

        # Should emit orderbook data
        orderbook_data = [data for data in sent_data if data[1] == DataType.ORDERBOOK]
        assert len(orderbook_data) == 1

        # Verify the data
        instrument, data_type, orderbook, is_historical = orderbook_data[0]
        assert instrument.symbol == "BTCUSDT"
        assert data_type == DataType.ORDERBOOK
        assert orderbook == mock_orderbook
        assert not is_historical

    def test_individual_subscription_with_multiple_instruments(
        self, mock_exchange_without_bulk_support, mock_data_provider, btc_instrument, eth_instrument, orderbook_channel
    ):
        """Test individual subscription with multiple instruments creates separate subscribers."""
        instruments = {btc_instrument, eth_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="binance",
        )

        config = handler.prepare_subscription(
            name="multi_instrument_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Should create separate subscribers for each instrument
        assert len(config.instrument_subscribers) == 2
        assert btc_instrument in config.instrument_subscribers
        assert eth_instrument in config.instrument_subscribers

        # Each subscriber should be different (different closures)
        btc_subscriber = config.instrument_subscribers[btc_instrument]
        eth_subscriber = config.instrument_subscribers[eth_instrument]
        assert btc_subscriber != eth_subscriber


class TestOrderBookDataProcessing:
    """Test orderbook data processing logic."""

    @patch("qubx.connectors.ccxt.handlers.orderbook.ccxt_convert_orderbook")
    def test_process_orderbook_sends_data_correctly(
        self, mock_convert, mock_data_provider, btc_instrument, orderbook_channel, sample_ccxt_orderbook
    ):
        """Test that _process_orderbook sends data correctly."""
        # Setup mock
        mock_orderbook = Mock()
        mock_orderbook.time = np.datetime64("2024-01-01T00:00:00", "ns")
        mock_orderbook.to_quote.return_value = Mock()  # Mock the quote conversion
        mock_convert.return_value = mock_orderbook

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=Mock(),
            exchange_id="test",
        )

        # Set up channel to capture sent data
        sent_data = []
        orderbook_channel.send = lambda data: sent_data.append(data)

        # Process the orderbook
        result = handler._process_orderbook(
            sample_ccxt_orderbook, btc_instrument, DataType.ORDERBOOK, orderbook_channel, 100, 0.01
        )

        assert result is True
        assert len(sent_data) == 1

        instrument, data_type, orderbook, is_historical = sent_data[0]
        assert instrument == btc_instrument
        assert data_type == DataType.ORDERBOOK
        assert orderbook == mock_orderbook
        assert not is_historical

    @patch("qubx.connectors.ccxt.handlers.orderbook.ccxt_convert_orderbook")
    def test_process_orderbook_handles_none_result(
        self, mock_convert, mock_data_provider, btc_instrument, orderbook_channel, sample_ccxt_orderbook
    ):
        """Test that _process_orderbook handles None result from conversion."""
        mock_convert.return_value = None  # Conversion failed

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=Mock(),
            exchange_id="test",
        )

        # Set up channel to capture sent data
        sent_data = []
        orderbook_channel.send = lambda data: sent_data.append(data)

        # Process the orderbook
        result = handler._process_orderbook(
            sample_ccxt_orderbook, btc_instrument, DataType.ORDERBOOK, orderbook_channel, 100, 0.01
        )

        assert result is False
        assert len(sent_data) == 0  # No data should be sent

    def test_handler_data_type_property(self, mock_data_provider):
        """Test that handler returns correct data type."""
        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=Mock(),
            exchange_id="test",
        )

        assert handler.data_type == "orderbook"


class TestOrderBookHandlerEdgeCases:
    """Test edge cases and error handling."""

    def test_no_unsubscriber_when_exchange_doesnt_support(self, mock_data_provider, btc_instrument, orderbook_channel):
        """Test that no unsubscriber is created when exchange doesn't support it."""
        exchange = Mock()
        exchange.has = {"watchOrderBookForSymbols": False}
        exchange.watch_order_book = AsyncMock()
        # Remove un_watch_order_book method to simulate no support
        if hasattr(exchange, "un_watch_order_book"):
            delattr(exchange, "un_watch_order_book")

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=exchange,
            exchange_id="test",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments={btc_instrument},
        )

        # Should not create unsubscribers
        assert config.instrument_unsubscribers is None

    @pytest.mark.asyncio
    async def test_individual_subscriber_error_handling(
        self, mock_exchange_without_bulk_support, mock_data_provider, btc_instrument, orderbook_channel
    ):
        """Test that individual subscriber handles errors properly."""
        instruments = {btc_instrument}

        handler = OrderBookDataHandler(
            data_provider=mock_data_provider,
            exchange=mock_exchange_without_bulk_support,
            exchange_id="test",
        )

        config = handler.prepare_subscription(
            name="test_stream",
            sub_type=DataType.ORDERBOOK,
            channel=orderbook_channel,
            instruments=instruments,
        )

        # Mock exchange to raise an error
        mock_exchange_without_bulk_support.watch_order_book.side_effect = Exception("Connection failed")

        # Get the individual subscriber for BTC
        btc_subscriber = config.instrument_subscribers[btc_instrument]

        # Should re-raise the exception for connection manager to handle
        with pytest.raises(Exception, match="Connection failed"):
            await btc_subscriber()
