from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from qubx.connectors.tardis.data import TardisDataProvider
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.core.lookups import lookup


class TestTardisDataProvider:
    @pytest.fixture(scope="module")
    def mock_time_provider(self) -> MagicMock:
        mock = MagicMock()
        mock.time.return_value = np.datetime64("2023-01-01T00:00:00")
        return mock

    @pytest.fixture(scope="module")
    def mock_channel(self) -> MagicMock:
        mock = MagicMock(spec=CtrlChannel)
        return mock

    @pytest.fixture(scope="module")
    def test_instrument(self) -> Instrument:
        instrument = lookup.find_symbol("BITFINEX.F", "BTCUSDT")
        assert instrument is not None
        return instrument

    @pytest.fixture(scope="module")
    def second_instrument(self) -> Instrument:
        instrument = lookup.find_symbol("BITFINEX.F", "ETHUSDT")
        assert instrument is not None
        return instrument

    @pytest.fixture(scope="session")
    def test_trade_message(self) -> Dict[str, Any]:
        return {
            "type": "trade",
            "exchange": "bitfinex",
            "symbol": "BTCUSDT",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "price": 50000,
            "amount": 1.0,
            "side": "buy",
        }

    @pytest.fixture(scope="session")
    def test_orderbook_message(self) -> Dict[str, Any]:
        return {
            "type": "book_snapshot",
            "exchange": "bitfinex",
            "symbol": "BTCUSDT",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "bids": [{"price": 50000, "amount": 1.0}],
            "asks": [{"price": 50100, "amount": 1.0}],
        }

    @pytest.fixture(scope="module")
    def mock_trade(self) -> MagicMock:
        return MagicMock(time=1672531200000000000, price=50000.0, size=1.0, side=1)

    @pytest.fixture(scope="module")
    def mock_orderbook(self) -> MagicMock:
        mock = MagicMock()
        mock.to_quote.return_value = MagicMock()
        return mock

    @pytest.fixture
    def mock_convert_trade(self, mock_trade: MagicMock) -> Generator[MagicMock, None, None]:
        with patch("qubx.connectors.tardis.data.tardis_convert_trade", return_value=mock_trade) as mock:
            yield mock

    @pytest.fixture
    def mock_convert_orderbook(self, mock_orderbook: MagicMock) -> Generator[MagicMock, None, None]:
        with patch("qubx.connectors.tardis.data.tardis_convert_orderbook", return_value=mock_orderbook) as mock:
            yield mock

    @pytest.fixture
    def data_provider(
        self, mock_time_provider: MagicMock, mock_channel: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> Generator[TardisDataProvider, None, None]:
        # Patch asyncio-related functionality
        monkeypatch.setattr("qubx.connectors.tardis.data.AsyncThreadLoop", MagicMock())
        monkeypatch.setattr("threading.Thread", MagicMock())

        # Patch _start_websocket_connection to return an AsyncMock to prevent the warning
        async_mock = AsyncMock()
        monkeypatch.setattr(TardisDataProvider, "_start_websocket_connection", lambda self: async_mock)

        # Create a simplified provider without real thread or asyncio operations
        provider = TardisDataProvider(
            host="localhost",
            port=8000,
            exchange="bitfinex",
            time_provider=mock_time_provider,
            channel=mock_channel,
        )

        # Mock the internal loop and thread
        provider._loop = MagicMock()
        provider._loop.submit.return_value.result.return_value = None
        provider._event_loop = MagicMock()
        provider._thread = MagicMock()

        # Pre-initialize internal structures that would normally be created during connection
        # We're accessing protected members for testing purposes only
        provider._symbol_to_instrument = {}
        setattr(provider, "_last_quotes", {})

        # Skip actual close operations in teardown
        with patch.object(TardisDataProvider, "close"):
            yield provider

        # After yield - we can reset the mock since mock_channel is a MagicMock instance
        # even though it's spec'd as CtrlChannel
        mock_channel.reset_mock()

    @pytest.fixture(autouse=True)
    def reset_mocks(self, mock_channel: MagicMock) -> Generator[None, None, None]:
        # This fixture will automatically run before each test
        # We can reset the mock since mock_channel is a MagicMock instance
        # even though it's spec'd as CtrlChannel
        mock_channel.reset_mock()
        yield
        # Reset after test too
        mock_channel.reset_mock()

    def test_subscribe_and_get_subscriptions(
        self, data_provider: TardisDataProvider, test_instrument: Instrument
    ) -> None:
        # Test subscribing to a single data type
        data_provider.subscribe(DataType.TRADE, {test_instrument})

        # Verify subscription was added
        assert test_instrument in data_provider.get_subscribed_instruments(DataType.TRADE)
        assert DataType.TRADE in data_provider.get_subscriptions(test_instrument)
        assert data_provider.has_subscription(test_instrument, DataType.TRADE)

        # Subscribe to another data type
        data_provider.subscribe(DataType.ORDERBOOK[0.01, 10], {test_instrument})

        # Verify both subscriptions exist
        assert test_instrument in data_provider.get_subscribed_instruments(DataType.TRADE)
        assert test_instrument in data_provider.get_subscribed_instruments(DataType.ORDERBOOK)
        assert len(data_provider.get_subscriptions(test_instrument)) == 2
        assert data_provider.has_subscription(test_instrument, DataType.TRADE)
        assert data_provider.has_subscription(test_instrument, DataType.ORDERBOOK)

        # Check that subscription parameters are stored correctly
        orderbook_sub = f"{DataType.ORDERBOOK}(0.01, 10)"
        data_type, params = DataType.from_str(orderbook_sub)
        assert data_type == DataType.ORDERBOOK
        assert params == {"tick_size_pct": 0.01, "depth": 10}

    def test_unsubscribe(self, data_provider: TardisDataProvider, test_instrument: Instrument) -> None:
        # First subscribe to multiple data types
        data_provider.subscribe(DataType.TRADE, {test_instrument})
        data_provider.subscribe(DataType.QUOTE, {test_instrument})
        data_provider.subscribe(DataType.ORDERBOOK[0.01, 10], {test_instrument})

        # Verify all subscriptions
        assert len(data_provider.get_subscriptions(test_instrument)) == 3

        # Unsubscribe from one data type
        data_provider.unsubscribe(DataType.TRADE, {test_instrument})

        # Verify remaining subscriptions
        assert not data_provider.has_subscription(test_instrument, DataType.TRADE)
        assert data_provider.has_subscription(test_instrument, DataType.QUOTE)
        assert data_provider.has_subscription(test_instrument, DataType.ORDERBOOK)
        assert len(data_provider.get_subscriptions(test_instrument)) == 2

        # Unsubscribe from all data types
        data_provider.unsubscribe(None, {test_instrument})

        # Verify no subscriptions remain
        assert len(data_provider.get_subscriptions(test_instrument)) == 0
        assert not data_provider.has_subscription(test_instrument, DataType.QUOTE)
        assert not data_provider.has_subscription(test_instrument, DataType.ORDERBOOK)

    def test_multiple_instruments(
        self, data_provider: TardisDataProvider, test_instrument: Instrument, second_instrument: Instrument
    ) -> None:
        # Subscribe with both instruments
        data_provider.subscribe(DataType.TRADE, {test_instrument, second_instrument})

        # Verify both are subscribed
        instruments = data_provider.get_subscribed_instruments(DataType.TRADE)
        assert len(instruments) == 2
        assert test_instrument in instruments
        assert second_instrument in instruments

        # Unsubscribe one instrument
        data_provider.unsubscribe(DataType.TRADE, {test_instrument})

        # Verify only one remains
        instruments = data_provider.get_subscribed_instruments(DataType.TRADE)
        assert len(instruments) == 1
        assert second_instrument in instruments
        assert test_instrument not in instruments

    def test_reset_subscription(
        self, data_provider: TardisDataProvider, test_instrument: Instrument, second_instrument: Instrument
    ) -> None:
        # First subscribe to one instrument
        data_provider.subscribe(DataType.TRADE, {test_instrument})

        # Verify subscription
        assert len(data_provider.get_subscribed_instruments(DataType.TRADE)) == 1
        assert test_instrument in data_provider.get_subscribed_instruments(DataType.TRADE)

        # Reset subscription with a new instrument
        data_provider.subscribe(DataType.TRADE, {second_instrument}, reset=True)

        # Verify only new instrument is subscribed
        instruments = data_provider.get_subscribed_instruments(DataType.TRADE)
        assert len(instruments) == 1
        assert second_instrument in instruments
        assert test_instrument not in instruments

    @pytest.mark.asyncio
    async def test_process_tardis_message_trade(
        self,
        data_provider: TardisDataProvider,
        test_instrument: Instrument,
        mock_convert_trade: MagicMock,
        test_trade_message: Dict[str, Any],
    ) -> None:
        # Reset subscriptions and internal state
        # We're accessing protected members for testing purposes only
        # pylint: disable=protected-access
        data_provider._subscriptions = {}  # type: ignore
        data_provider._subscriptions = {DataType.TRADE: {test_instrument}}  # type: ignore
        data_provider._symbol_to_instrument = {"BTCUSDT": test_instrument}  # type: ignore

        # We can call reset_mock because the channel is a MagicMock instance
        # even though it's spec'd as CtrlChannel
        data_provider.channel.reset_mock()  # type: ignore

        # Process the message
        await data_provider._process_tardis_message(test_trade_message)

        # Verify the channel was called with the correct data
        data_provider.channel.send.assert_called_once()
        call_args = data_provider.channel.send.call_args[0][0]
        assert call_args[0] == test_instrument
        assert call_args[1] == DataType.TRADE
        assert call_args[3] is False  # not historical

    @pytest.mark.asyncio
    async def test_process_tardis_message_orderbook(
        self,
        data_provider: TardisDataProvider,
        test_instrument: Instrument,
        mock_convert_orderbook: MagicMock,
        test_orderbook_message: Dict[str, Any],
    ) -> None:
        # Reset subscriptions and internal state
        # We're accessing protected members for testing purposes only
        # pylint: disable=protected-access
        data_provider._subscriptions = {}  # type: ignore
        data_provider._subscriptions = {DataType.ORDERBOOK: {test_instrument}}  # type: ignore
        data_provider._symbol_to_instrument = {"BTCUSDT": test_instrument}  # type: ignore
        data_provider._subscription_params = {
            (DataType.ORDERBOOK, test_instrument): {"depth": 20, "tick_size_pct": 0.05}
        }  # type: ignore

        # We can call reset_mock because the channel is a MagicMock instance
        # even though it's spec'd as CtrlChannel
        data_provider.channel.reset_mock()  # type: ignore

        # Process the message
        await data_provider._process_tardis_message(test_orderbook_message)

        # Verify the orderbook conversion was called with correct parameters
        mock_convert_orderbook.assert_called_once()
        call_args = mock_convert_orderbook.call_args[0]
        assert call_args[0] == test_orderbook_message
        assert call_args[1] == test_instrument
        assert call_args[2] == 20  # levels from params
        assert call_args[3] == 0.05  # tick_size_pct from params

        # Verify the channel was called correctly
        data_provider.channel.send.assert_called_once()
        channel_args = data_provider.channel.send.call_args[0][0]
        assert channel_args[0] == test_instrument
        assert channel_args[1] == DataType.ORDERBOOK
        assert channel_args[3] is False  # not historical

    @pytest.mark.asyncio
    async def test_process_tardis_message_orderbook_default_params(
        self,
        data_provider: TardisDataProvider,
        test_instrument: Instrument,
        mock_convert_orderbook: MagicMock,
        test_orderbook_message: Dict[str, Any],
    ) -> None:
        # Reset subscriptions and internal state
        # We're accessing protected members for testing purposes only
        # pylint: disable=protected-access
        data_provider._subscriptions = {}  # type: ignore
        data_provider._subscriptions = {DataType.ORDERBOOK: {test_instrument}}  # type: ignore
        data_provider._symbol_to_instrument = {"BTCUSDT": test_instrument}  # type: ignore
        data_provider._subscription_params = {}  # No custom params  # type: ignore

        # We can call reset_mock because the channel is a MagicMock instance
        # even though it's spec'd as CtrlChannel
        data_provider.channel.reset_mock()  # type: ignore

        # Process the message
        await data_provider._process_tardis_message(test_orderbook_message)

        # Verify the orderbook conversion was called with default parameters
        mock_convert_orderbook.assert_called_once()
        call_args = mock_convert_orderbook.call_args[0]
        assert call_args[0] == test_orderbook_message
        assert call_args[1] == test_instrument
        assert call_args[2] == 50  # default levels
        assert call_args[3] == 0.01  # default tick_size_pct
