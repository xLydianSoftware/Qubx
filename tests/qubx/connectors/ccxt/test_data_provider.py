"""Simple unit tests for CcxtDataProvider focusing on core functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.handlers.ohlc import OhlcDataHandler
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
from qubx.core.series import Bar, Quote


@pytest.fixture
def btc_instrument():
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="TEST",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def mock_exchange():
    # Create mock raw exchange
    raw_exchange = Mock()
    raw_exchange.name = "TEST"
    raw_exchange.apiKey = None  # This makes is_read_only return True
    raw_exchange.asyncio_loop = asyncio.new_event_loop()
    raw_exchange.find_timeframe = Mock(return_value="1m")
    raw_exchange.close = AsyncMock()

    # Create mock ExchangeManager that wraps the raw exchange
    exchange_manager = Mock()
    exchange_manager.exchange = raw_exchange  # This is the key part!
    exchange_manager.force_recreation = Mock(return_value=True)
    exchange_manager.reset_recreation_count_if_needed = Mock()
    exchange_manager._exchange_name = "TEST"

    return exchange_manager


@pytest.fixture
def mock_time_provider():
    return Mock()


@pytest.fixture
def ctrl_channel():
    channel = CtrlChannel("test")
    channel.control.set()
    return channel


@pytest.fixture
def data_provider(mock_exchange, mock_time_provider, ctrl_channel):
    return CcxtDataProvider(
        exchange_manager=mock_exchange,
        time_provider=mock_time_provider,
        channel=ctrl_channel,
        max_ws_retries=3,
        warmup_timeout=10,
    )


class TestBasicFunctionality:
    """Test basic data provider functionality."""

    def test_initialization(self, data_provider):
        """Test that data provider initializes correctly."""
        assert data_provider._exchange_id == "TEST"
        assert data_provider.is_simulation is False
        assert data_provider.exchange() == "TEST"

    def test_subscribe_calls_orchestrator(self, data_provider, btc_instrument):
        """Test that subscribe calls the orchestrator with correct parameters."""
        mock_handler = Mock()

        with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=mock_handler):
            with patch.object(data_provider._subscription_orchestrator, "execute_subscription") as mock_orchestrator:
                data_provider.subscribe("trades", [btc_instrument])

                mock_orchestrator.assert_called_once()
                call_args = mock_orchestrator.call_args
                assert call_args.kwargs["subscription_type"] == "trades"
                assert btc_instrument in call_args.kwargs["instruments"]

    def test_subscribe_unsupported_type_raises_error(self, data_provider, btc_instrument):
        """Test that subscribing to unsupported type raises error."""
        with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=None):
            with pytest.raises(ValueError, match="Subscription type .* is not supported"):
                data_provider.subscribe("unsupported", [btc_instrument])

    def test_unsubscribe_empty_does_nothing(self, data_provider, btc_instrument):
        """Test that unsubscribing from empty subscription does nothing."""
        # Mock to return empty list (no subscriptions)
        with patch.object(data_provider._subscription_manager, "get_subscribed_instruments", return_value=[]):
            with patch.object(data_provider._subscription_orchestrator, "execute_unsubscription") as mock_unsub:
                data_provider.unsubscribe("trades", [btc_instrument])
                mock_unsub.assert_not_called()

    def test_quote_management(self, data_provider, btc_instrument):
        """Test quote getting and setting."""
        # Initially no quote
        assert data_provider.get_quote(btc_instrument) is None

        # Set a quote
        test_quote = Quote(
            time=1234567890,
            bid=50000.0,
            ask=50001.0,
            bid_size=1.0,
            ask_size=1.0,
        )
        data_provider._last_quotes[btc_instrument] = test_quote

        # Get the quote
        retrieved = data_provider.get_quote(btc_instrument)
        assert retrieved == test_quote

    def test_is_read_only_property(self, data_provider):
        """Test read-only property based on API key."""
        # With API key
        data_provider._exchange_manager.exchange.apiKey = "test_key"
        assert data_provider.is_read_only is False

        # Without API key
        data_provider._exchange_manager.exchange.apiKey = None
        assert data_provider.is_read_only is True

        # With empty API key
        data_provider._exchange_manager.exchange.apiKey = ""
        assert data_provider.is_read_only is True

    def test_get_ohlc_success(self, data_provider, btc_instrument):
        """Test successful OHLC data retrieval."""
        # Create a real OhlcDataHandler mock
        mock_ohlc_handler = Mock(spec=OhlcDataHandler)
        expected_bars = [
            Bar(time=1234567890, open=50000, high=50100, low=49900, close=50050, volume=100),
        ]
        mock_ohlc_handler.get_historical_ohlc = AsyncMock(return_value=expected_bars)

        # Mock the async loop submission to return immediately
        mock_future = Mock()
        mock_future.result.return_value = expected_bars

        # Create a mock AsyncThreadLoop
        mock_async_thread_loop = Mock()
        mock_async_thread_loop.submit.return_value = mock_future

        with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=mock_ohlc_handler):
            # Patch AsyncThreadLoop in the data module
            with patch("qubx.connectors.ccxt.data.AsyncThreadLoop", return_value=mock_async_thread_loop):
                bars = data_provider.get_ohlc(btc_instrument, "1m", 1)
                assert len(bars) == 1
                assert bars[0].open == 50000

    def test_get_ohlc_no_handler_raises_error(self, data_provider, btc_instrument):
        """Test OHLC retrieval when handler is not available."""
        with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=None):
            with pytest.raises(ValueError, match="OHLC handler not available"):
                data_provider.get_ohlc(btc_instrument, "1m", 10)

    def test_get_ohlc_wrong_handler_type_raises_error(self, data_provider, btc_instrument):
        """Test OHLC retrieval when wrong handler type is returned."""
        mock_wrong_handler = Mock()  # Not an OhlcDataHandler

        with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=mock_wrong_handler):
            with pytest.raises(ValueError, match="Expected OhlcDataHandler"):
                data_provider.get_ohlc(btc_instrument, "1m", 10)

    def test_close_calls_exchange_close(self, data_provider):
        """Test that close method calls exchange close."""
        with patch.object(data_provider._subscription_manager, "get_subscriptions", return_value=[]):
            data_provider.close()
            data_provider._exchange_manager.exchange.close.assert_called_once()

    def test_close_handles_errors_gracefully(self, data_provider):
        """Test that close handles errors without raising."""
        # Make exchange.close raise an error
        data_provider._exchange_manager.exchange.close.side_effect = Exception("Connection error")

        with patch.object(data_provider._subscription_manager, "get_subscriptions", return_value=[]):
            # Should not raise
            data_provider.close()

            # Exchange close should have been attempted
            data_provider._exchange_manager.exchange.close.assert_called_once()


class TestUnsubscriptionLogic:
    """Test unsubscription logic with proper mocking."""

    def test_complete_unsubscription(self, data_provider, btc_instrument):
        """Test complete unsubscription when no instruments remain."""
        # Mock subscription manager to return the instrument, then empty after removal
        with patch.object(data_provider._subscription_manager, "get_subscribed_instruments") as mock_get_subs:
            # First call returns btc_instrument, second call (after removal) returns empty
            mock_get_subs.side_effect = [[btc_instrument], []]

            with patch.object(data_provider._subscription_manager, "remove_subscription"):
                with patch.object(data_provider._subscription_orchestrator, "execute_unsubscription") as mock_unsub:
                    data_provider.unsubscribe("trades", [btc_instrument])

                    # Should call unsubscription since no instruments remain
                    mock_unsub.assert_called_once()

    def test_partial_unsubscription(self, data_provider, btc_instrument):
        """Test partial unsubscription triggers resubscription without collapsing desired set."""
        eth_instrument = Instrument(
            symbol="ETHUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="TEST",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=Mock()):
            with patch.object(data_provider._subscription_orchestrator, "execute_subscription") as mock_exec:
                # Seed subscription state with two instruments
                data_provider.subscribe("trades", [btc_instrument, eth_instrument])

                # Now remove BTC => should resubscribe with remaining ETH only (via subscribe(reset=True))
                data_provider.unsubscribe("trades", [btc_instrument])

                # Expect two execute_subscription calls:
                # - initial subscribe: BTC+ETH
                # - partial-unsub rebuild: ETH only
                assert mock_exec.call_count == 2
                initial_set = mock_exec.call_args_list[0].kwargs["instruments"]
                remaining_set = mock_exec.call_args_list[1].kwargs["instruments"]

                assert btc_instrument in initial_set and eth_instrument in initial_set
                assert eth_instrument in remaining_set and btc_instrument not in remaining_set


def _make_instrument(idx: int) -> Instrument:
    """Create deterministic unique instruments for bulk subscription regression tests."""
    base = f"ASSET{idx}"
    quote = "USD"
    return Instrument(
        symbol=f"{base}{quote}",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="TEST",
        base=base,
        quote=quote,
        settle=quote,
        exchange_symbol=f"{base}{quote}",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


def test_stale_recovery_does_not_collapse_bulk_subscription_set(data_provider):
    """
    Regression test for the observed live issue:
    bulk N instruments -> unsubscribe(stale) triggers rebuild to N-1 -> subscribe(stale) must rebuild to N (not 1).
    """
    instruments = [_make_instrument(i) for i in range(40)]
    stale = instruments[0]

    with patch.object(data_provider._data_type_handler_factory, "get_handler", return_value=Mock()):
        with patch.object(data_provider._subscription_orchestrator, "execute_subscription") as mock_exec:
            data_provider.subscribe("quote", instruments)
            data_provider.unsubscribe("quote", [stale])  # rebuild to 39
            data_provider.subscribe("quote", [stale])  # rebuild back to 40 (union)

            sizes = [len(call.kwargs["instruments"]) for call in mock_exec.call_args_list]
            assert sizes == [40, 39, 40]

            # Provider state should also reflect the full desired set.
            assert len(data_provider.get_subscribed_instruments("quote")) == 40


class TestDelegation:
    """Test that data provider correctly delegates to composed components."""

    def test_get_subscriptions_delegates_to_manager(self, data_provider, btc_instrument):
        """Test that get_subscriptions delegates to subscription manager."""
        with patch.object(
            data_provider._subscription_manager, "get_subscriptions", return_value=["trades"]
        ) as mock_get_subs:
            result = data_provider.get_subscriptions()
            mock_get_subs.assert_called_once_with(None)
            assert result == ["trades"]

            # Test with instrument parameter
            data_provider.get_subscriptions(btc_instrument)
            mock_get_subs.assert_called_with(btc_instrument)

    def test_has_subscription_delegates_to_manager(self, data_provider, btc_instrument):
        """Test that has_subscription delegates to subscription manager."""
        with patch.object(data_provider._subscription_manager, "has_subscription", return_value=True) as mock_has_sub:
            result = data_provider.has_subscription(btc_instrument, "trades")
            mock_has_sub.assert_called_once_with(btc_instrument, "trades")
            assert result is True

    def test_has_pending_subscription_delegates_to_manager(self, data_provider, btc_instrument):
        """Test that has_pending_subscription delegates to subscription manager."""
        with patch.object(
            data_provider._subscription_manager, "has_pending_subscription", return_value=False
        ) as mock_has_pending:
            result = data_provider.has_pending_subscription(btc_instrument, "trades")
            mock_has_pending.assert_called_once_with(btc_instrument, "trades")
            assert result is False

    def test_warmup_delegates_to_service(self, data_provider, btc_instrument):
        """Test that warmup delegates to warmup service."""
        warmups = {("ohlc", btc_instrument): "1m"}
        with patch.object(data_provider._warmup_service, "execute_warmup") as mock_warmup:
            data_provider.warmup(warmups)
            mock_warmup.assert_called_once_with(warmups)
