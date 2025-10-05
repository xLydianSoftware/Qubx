from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.basics import Instrument, MarketType, Position
from qubx.core.exceptions import OrderNotFound
from qubx.core.interfaces import IAccountProcessor, IBroker, IStrategyContext, ITimeProvider
from qubx.core.mixins.trading import ClientIdStore, TradingManager


class MockTimeProvider(ITimeProvider):
    def __init__(self, timestamp=None):
        self._timestamp = timestamp or pd.Timestamp("2023-01-01").asm8

    def time(self):
        return self._timestamp


class MockStrategyContext(IStrategyContext):
    def __init__(self, timestamp=None):
        self._timestamp = timestamp or pd.Timestamp("2023-01-01").asm8
        self.emitted_signals = []

    def time(self):
        return self._timestamp

    def emit_signal(self, signal):
        """Mock implementation to track emitted signals."""
        if isinstance(signal, list):
            self.emitted_signals.extend(signal)
        else:
            self.emitted_signals.append(signal)


@pytest.fixture
def time_provider():
    return MockTimeProvider()


@pytest.fixture
def strategy_context():
    return MockStrategyContext()


@pytest.fixture
def client_id_store():
    return ClientIdStore()


def test_initialize_from_timestamp():
    """Test that order_id is initialized correctly from timestamp."""
    # Given a time provider with a known timestamp
    timestamp = pd.Timestamp("2023-01-01").asm8
    time_provider = MockTimeProvider(timestamp)
    store = ClientIdStore()

    # When initializing order_id from timestamp
    result = store._initialize_id_from_timestamp(time_provider)

    # Then the order_id is derived correctly
    expected = timestamp.astype("int64") // 100_000_000
    assert result == expected


def test_create_id(client_id_store):
    """Test ID creation from symbol and order ID."""
    # When creating an ID
    result = client_id_store._create_id("BTCUSD", 12345)

    # Then the ID follows the expected format
    assert result == "qubx_BTCUSD_12345"


def test_generate_id(time_provider, client_id_store):
    """Test that generated IDs are unique and incremental."""
    # When generating multiple IDs
    ids = [client_id_store.generate_id(time_provider, "BTCUSD") for _ in range(5)]

    # Then all IDs are unique
    assert len(ids) == len(set(ids)), "Generated IDs should be unique"

    # And they follow an incremental pattern
    for i in range(1, len(ids)):
        id1_parts = ids[i - 1].split("_")
        id2_parts = ids[i].split("_")
        assert int(id2_parts[2]) == int(id1_parts[2]) + 1, "IDs should increment by 1"


def test_unique_ids_across_symbols(time_provider, client_id_store):
    """Test that IDs are unique across different symbols."""
    # When generating IDs for different symbols
    id1 = client_id_store.generate_id(time_provider, "BTCUSD")
    id2 = client_id_store.generate_id(time_provider, "ETHUSD")

    # Then the IDs are unique
    assert id1 != id2

    # And they follow the expected format with different symbols
    assert "BTCUSD" in id1
    assert "ETHUSD" in id2

    # And the numerical part increments
    id1_num = int(id1.split("_")[2])
    id2_num = int(id2.split("_")[2])
    assert id2_num == id1_num + 1


# TradingManager Tests


@pytest.fixture
def mock_instrument():
    """Create a mock instrument for testing."""
    instrument = Mock(spec=Instrument)
    instrument.symbol = "BTCUSDT"
    instrument.exchange = "BINANCE.UM"
    instrument.market_type = MarketType.SWAP
    instrument.min_size = 0.001

    # Mock the signal method to return a signal object
    def mock_signal(context, signal_value, comment="", group=""):
        signal_obj = Mock()
        signal_obj.signal = signal_value
        signal_obj.comment = comment
        signal_obj.group = group
        signal_obj.instrument = instrument
        return signal_obj

    instrument.signal = mock_signal
    return instrument


@pytest.fixture
def mock_spot_instrument():
    """Create a mock spot instrument for testing."""
    instrument = Mock(spec=Instrument)
    instrument.symbol = "BTCUSDT"
    instrument.exchange = "BINANCE.SPOT"
    instrument.market_type = MarketType.SPOT
    instrument.min_size = 0.001
    return instrument


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = Mock(spec=IBroker)
    broker.exchange.return_value = "BINANCE.UM"
    # Add cancel_order_async method to the mock
    broker.cancel_order_async = Mock(return_value=None)
    return broker


@pytest.fixture
def mock_account():
    """Create a mock account processor for testing."""
    account = Mock(spec=IAccountProcessor)
    return account


@pytest.fixture
def trading_manager(strategy_context, mock_broker, mock_account):
    """Create a TradingManager instance for testing."""
    return TradingManager(
        context=strategy_context, brokers=[mock_broker], account=mock_account, strategy_name="test_strategy"
    )


class TestTradingManagerClosePosition:
    """Test cases for close_position method."""

    def test_close_position_with_open_long_position(
        self, trading_manager, mock_instrument, mock_account, strategy_context
    ):
        """Test closing an open long position."""
        # Given an open long position
        position = Mock(spec=Position)
        position.quantity = 1.5
        mock_account.get_position.return_value = position

        # When closing the position
        trading_manager.close_position(mock_instrument)

        # Then a signal with 0 value is emitted
        assert len(strategy_context.emitted_signals) == 1
        signal = strategy_context.emitted_signals[0]
        assert signal.signal == 0
        assert signal.instrument == mock_instrument
        mock_account.get_position.assert_called_once_with(mock_instrument)

    def test_close_position_with_open_short_position(
        self, trading_manager, mock_instrument, mock_account, strategy_context
    ):
        """Test closing an open short position."""
        # Given an open short position
        position = Mock(spec=Position)
        position.quantity = -2.0
        mock_account.get_position.return_value = position

        # When closing the position
        trading_manager.close_position(mock_instrument)

        # Then a signal with 0 value is emitted
        assert len(strategy_context.emitted_signals) == 1
        signal = strategy_context.emitted_signals[0]
        assert signal.signal == 0
        assert signal.instrument == mock_instrument
        mock_account.get_position.assert_called_once_with(mock_instrument)

    def test_close_position_with_closed_position(
        self, trading_manager, mock_instrument, mock_account, strategy_context
    ):
        """Test closing a position that is already closed."""
        # Given a closed position
        position = Mock(spec=Position)
        position.quantity = 0.0
        mock_account.get_position.return_value = position

        # When closing the position
        trading_manager.close_position(mock_instrument)

        # Then no signal is emitted
        assert len(strategy_context.emitted_signals) == 0
        mock_account.get_position.assert_called_once_with(mock_instrument)

    def test_close_position_with_very_small_position(
        self, trading_manager, mock_instrument, mock_account, strategy_context
    ):
        """Test closing a position that is below minimum size but not zero."""
        # Given a position below minimum size but not zero
        position = Mock(spec=Position)
        position.quantity = 0.0001  # Below min_size of 0.001, but not zero
        mock_account.get_position.return_value = position

        # When closing the position
        trading_manager.close_position(mock_instrument)

        # Then a signal is emitted because quantity != 0 (new logic checks quantity == 0)
        assert len(strategy_context.emitted_signals) == 1
        signal = strategy_context.emitted_signals[0]
        assert signal.signal == 0
        assert signal.instrument == mock_instrument
        mock_account.get_position.assert_called_once_with(mock_instrument)


class TestTradingManagerClosePositions:
    """Test cases for close_positions method."""

    def test_close_positions_all_markets(self, trading_manager, mock_account):
        """Test closing all positions regardless of market type."""
        # Given multiple open positions
        btc_swap = Mock(spec=Instrument)
        btc_swap.market_type = MarketType.SWAP
        btc_spot = Mock(spec=Instrument)
        btc_spot.market_type = MarketType.SPOT
        eth_future = Mock(spec=Instrument)
        eth_future.market_type = MarketType.FUTURE

        pos1 = Mock(spec=Position)
        pos1.is_open.return_value = True
        pos2 = Mock(spec=Position)
        pos2.is_open.return_value = True
        pos3 = Mock(spec=Position)
        pos3.is_open.return_value = False  # Closed position

        positions = {btc_swap: pos1, btc_spot: pos2, eth_future: pos3}
        mock_account.get_positions.return_value = positions

        # Mock close_position method
        trading_manager.close_position = Mock()

        # When closing all positions
        trading_manager.close_positions()

        # Then close_position is called for each open position
        assert trading_manager.close_position.call_count == 2
        trading_manager.close_position.assert_any_call(btc_swap, False)
        trading_manager.close_position.assert_any_call(btc_spot, False)

    def test_close_positions_by_market_type_spot(self, trading_manager, mock_account):
        """Test closing positions filtered by SPOT market type."""
        # Given positions of different market types
        btc_swap = Mock(spec=Instrument)
        btc_swap.market_type = MarketType.SWAP
        btc_spot = Mock(spec=Instrument)
        btc_spot.market_type = MarketType.SPOT
        eth_spot = Mock(spec=Instrument)
        eth_spot.market_type = MarketType.SPOT

        pos1 = Mock(spec=Position)
        pos1.is_open.return_value = True
        pos2 = Mock(spec=Position)
        pos2.is_open.return_value = True
        pos3 = Mock(spec=Position)
        pos3.is_open.return_value = True

        positions = {btc_swap: pos1, btc_spot: pos2, eth_spot: pos3}
        mock_account.get_positions.return_value = positions

        # Mock close_position method
        trading_manager.close_position = Mock()

        # When closing only SPOT positions
        trading_manager.close_positions(market_type=MarketType.SPOT)

        # Then close_position is called only for SPOT instruments
        assert trading_manager.close_position.call_count == 2
        trading_manager.close_position.assert_any_call(btc_spot, False)
        trading_manager.close_position.assert_any_call(eth_spot, False)

    def test_close_positions_by_market_type_future(self, trading_manager, mock_account):
        """Test closing positions filtered by FUTURE market type."""
        # Given positions of different market types
        btc_swap = Mock(spec=Instrument)
        btc_swap.market_type = MarketType.SWAP
        btc_future = Mock(spec=Instrument)
        btc_future.market_type = MarketType.FUTURE

        pos1 = Mock(spec=Position)
        pos1.is_open.return_value = True
        pos2 = Mock(spec=Position)
        pos2.is_open.return_value = True

        positions = {btc_swap: pos1, btc_future: pos2}
        mock_account.get_positions.return_value = positions

        # Mock close_position method
        trading_manager.close_position = Mock()

        # When closing only FUTURE positions
        trading_manager.close_positions(market_type=MarketType.FUTURE)

        # Then close_position is called only for FUTURE instruments
        trading_manager.close_position.assert_called_once_with(btc_future, False)

    def test_close_positions_no_open_positions(self, trading_manager, mock_account):
        """Test closing positions when no positions are open."""
        # Given no open positions
        btc_swap = Mock(spec=Instrument)
        btc_swap.market_type = MarketType.SWAP

        pos1 = Mock(spec=Position)
        pos1.is_open.return_value = False

        positions = {btc_swap: pos1}
        mock_account.get_positions.return_value = positions

        # Mock close_position method
        trading_manager.close_position = Mock()

        # When closing all positions
        trading_manager.close_positions()

        # Then close_position is not called
        trading_manager.close_position.assert_not_called()

    def test_close_positions_empty_positions_dict(self, trading_manager, mock_account):
        """Test closing positions when positions dictionary is empty."""
        # Given empty positions
        mock_account.get_positions.return_value = {}

        # Mock close_position method
        trading_manager.close_position = Mock()

        # When closing all positions
        trading_manager.close_positions()

        # Then close_position is not called
        trading_manager.close_position.assert_not_called()

    def test_close_positions_mixed_open_closed(self, trading_manager, mock_account):
        """Test closing positions with mix of open and closed positions."""
        # Given mix of open and closed positions
        btc_swap = Mock(spec=Instrument)
        btc_swap.market_type = MarketType.SWAP
        eth_spot = Mock(spec=Instrument)
        eth_spot.market_type = MarketType.SPOT
        ada_future = Mock(spec=Instrument)
        ada_future.market_type = MarketType.FUTURE

        pos1 = Mock(spec=Position)
        pos1.is_open.return_value = True  # Open
        pos2 = Mock(spec=Position)
        pos2.is_open.return_value = False  # Closed
        pos3 = Mock(spec=Position)
        pos3.is_open.return_value = True  # Open

        positions = {btc_swap: pos1, eth_spot: pos2, ada_future: pos3}
        mock_account.get_positions.return_value = positions

        # Mock close_position method
        trading_manager.close_position = Mock()

        # When closing all positions
        trading_manager.close_positions()

        # Then close_position is called only for open positions
        assert trading_manager.close_position.call_count == 2
        trading_manager.close_position.assert_any_call(btc_swap, False)
        trading_manager.close_position.assert_any_call(ada_future, False)


class TestTradingManagerCancelOrder:
    """Test cases for cancel_order and cancel_order_async methods."""

    def test_cancel_order_success(self, trading_manager, mock_broker, mock_account):
        """Test successful synchronous order cancellation."""
        # Given a broker that successfully cancels orders
        mock_broker.cancel_order.return_value = True
        order_id = "test_order_123"
        exchange = "BINANCE.UM"

        # When canceling an order
        result = trading_manager.cancel_order(order_id, exchange)

        # Then the operation succeeds
        assert result is True
        mock_broker.cancel_order.assert_called_once_with(order_id)
        mock_account.remove_order.assert_called_once_with(order_id, exchange)

    def test_cancel_order_failure(self, trading_manager, mock_broker, mock_account):
        """Test failed synchronous order cancellation."""
        # Given a broker that fails to cancel orders
        mock_broker.cancel_order.return_value = False
        order_id = "test_order_456"
        exchange = "BINANCE.UM"

        # When canceling an order
        result = trading_manager.cancel_order(order_id, exchange)

        # Then the operation fails
        assert result is False
        mock_broker.cancel_order.assert_called_once_with(order_id)
        # Account removal should not be called when cancellation fails
        mock_account.remove_order.assert_not_called()

    def test_cancel_order_empty_id(self, trading_manager):
        """Test cancel_order with empty order ID."""
        # When canceling with empty order ID
        result = trading_manager.cancel_order("")

        # Then it returns False immediately
        assert result is False

    def test_cancel_order_order_not_found_exception(self, trading_manager, mock_broker, mock_account):
        """Test cancel_order when OrderNotFound exception is raised."""
        # Given a broker that raises OrderNotFound
        mock_broker.cancel_order.side_effect = OrderNotFound("Order not found")
        order_id = "missing_order_789"
        exchange = "BINANCE.UM"

        # When canceling the order
        result = trading_manager.cancel_order(order_id, exchange)

        # Then it returns False and still tries to remove from account
        assert result is False
        mock_account.remove_order.assert_called_once_with(order_id, exchange)

    def test_cancel_order_unexpected_exception(self, trading_manager, mock_broker, mock_account):
        """Test cancel_order when unexpected exception is raised."""
        # Given a broker that raises an unexpected exception
        mock_broker.cancel_order.side_effect = Exception("Unexpected error")
        order_id = "error_order_999"
        exchange = "BINANCE.UM"

        # When canceling the order
        result = trading_manager.cancel_order(order_id, exchange)

        # Then it returns False and doesn't remove from account
        assert result is False
        mock_account.remove_order.assert_not_called()

    def test_cancel_order_async_success(self, trading_manager, mock_broker, mock_account):
        """Test successful asynchronous order cancellation."""
        # Given a broker with async cancel support
        mock_broker.cancel_order_async.return_value = None
        order_id = "async_order_123"
        exchange = "BINANCE.UM"

        # When canceling an order async
        result = trading_manager.cancel_order_async(order_id, exchange)

        # Then it returns None and removes from account optimistically
        assert result is None
        mock_broker.cancel_order_async.assert_called_once_with(order_id)
        mock_account.remove_order.assert_called_once_with(order_id, exchange)

    def test_cancel_order_async_empty_id(self, trading_manager):
        """Test cancel_order_async with empty order ID."""
        # When canceling with empty order ID
        result = trading_manager.cancel_order_async("")

        # Then it returns None immediately
        assert result is None

    def test_cancel_order_async_order_not_found_exception(self, trading_manager, mock_broker, mock_account):
        """Test cancel_order_async when OrderNotFound exception is raised."""
        # Given a broker that raises OrderNotFound
        mock_broker.cancel_order_async.side_effect = OrderNotFound("Order not found")
        order_id = "missing_async_order_789"
        exchange = "BINANCE.UM"

        # When canceling the order async
        result = trading_manager.cancel_order_async(order_id, exchange)

        # Then it returns None and still tries to remove from account
        assert result is None
        mock_account.remove_order.assert_called_once_with(order_id, exchange)

    def test_cancel_order_default_exchange(self, trading_manager, mock_broker, mock_account):
        """Test that default exchange is used when none specified."""
        # Given a broker that successfully cancels orders
        mock_broker.cancel_order.return_value = True
        order_id = "default_exchange_order"

        # When canceling without specifying exchange
        result = trading_manager.cancel_order(order_id)

        # Then the default exchange is used
        assert result is True
        mock_broker.cancel_order.assert_called_once_with(order_id)
        # Default exchange should be from first broker
        mock_account.remove_order.assert_called_once_with(order_id, "BINANCE.UM")
