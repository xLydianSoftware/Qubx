from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Instrument, MarketType, Order, OrderOrigin, OrderStatus, Position
from qubx.core.events import OrderCancelRejectedEvent, OrderUpdateRejectedEvent
from qubx.core.exceptions import InvalidOrderSize, OrderAlreadyTerminal, OrderNotFound
from qubx.core.interfaces import IStrategyContext, ITimeProvider
from qubx.core.lookups import lookup
from qubx.core.mixins.trading import ClientIdStore, TradingManager
from qubx.health.dummy import DummyHealthMonitor


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
def mock_connector():
    """Create a mock connector for testing."""
    connector = Mock()
    connector.exchange_name = "BINANCE.UM"
    connector.make_client_id = lambda s: s
    return connector


@pytest.fixture
def mock_account():
    """Create a mock account manager for testing."""
    account = Mock()
    return account


@pytest.fixture
def trading_manager(strategy_context, mock_connector, mock_account):
    """Create a TradingManager instance for testing."""
    return TradingManager(
        context=strategy_context,
        connectors={"BINANCE.UM": mock_connector},
        account_manager=mock_account,
        strategy_name="test_strategy",
        health_monitor=DummyHealthMonitor(),
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


def _live_order(order_id="test_order_123"):
    """A resolvable, non-terminal order living on BINANCE.UM."""
    instrument = Mock()
    instrument.exchange = "BINANCE.UM"
    order = Mock()
    order.client_order_id = "qubx_BTCUSDT_1"
    order.venue_order_id = order_id
    order.instrument = instrument
    order.status = OrderStatus.ACCEPTED
    return order


class TestTradingManagerTradeOrderShape:
    """The Order trade() builds carries None price for market orders (no fake 0.0)."""

    def test_market_order_has_none_price(self, trading_manager, mock_connector, mock_account):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        mock_account.get_position.return_value = None

        trading_manager.trade(instr, 0.1)  # no price => market order

        registered = mock_account.add_order.call_args.args[0]
        assert registered.type == "MARKET"
        assert registered.price is None

    def test_limit_order_keeps_price(self, trading_manager, mock_connector, mock_account):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        mock_account.get_position.return_value = None

        trading_manager.trade(instr, 0.1, price=50_000.0)

        registered = mock_account.add_order.call_args.args[0]
        assert registered.type == "LIMIT"
        assert registered.price == 50_000.0


class TestTradingManagerTradeSubmitFailure:
    """trade() registers the order before submitting, and cleans up on a synchronous raise."""

    def test_sync_submit_raise_removes_order_no_phantom(self, trading_manager, mock_connector, mock_account):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        mock_account.get_position.return_value = None  # flat: skip position-reducing math
        mock_connector.submit_order.side_effect = RuntimeError("framework reject")

        with pytest.raises(RuntimeError):
            trading_manager.trade(instr, 0.1, price=50_000.0)

        # the order is registered before submit, then dropped on the raise — not left as a
        # phantom in-flight order, and NOT a fake REJECTED transition.
        mock_account.add_order.assert_called_once()
        registered = mock_account.add_order.call_args.args[0]
        mock_account.remove_order.assert_called_once_with("BINANCE.UM", registered.client_order_id)
        mock_account.transition_order.assert_not_called()


class TestTradingManagerCancelOrder:
    """Cancel routing through IConnector + AccountManager."""

    def test_cancel_order_success(self, trading_manager, mock_connector, mock_account):
        """A live order transitions to PENDING_CANCEL and routes to the connector."""
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        result = trading_manager.cancel_order(order_id="test_order_123", exchange="BINANCE.UM")

        assert result is True
        mock_account.transition_order.assert_called_once_with(
            "BINANCE.UM", order.client_order_id, OrderStatus.PENDING_CANCEL
        )
        mock_connector.cancel_order.assert_called_once_with(
            client_order_id=order.client_order_id, venue_order_id=order.venue_order_id
        )

    def test_cancel_terminal_is_idempotent_noop(self, trading_manager, mock_connector, mock_account):
        """Cancelling a terminal order is a no-op that still reports success."""
        order = _live_order()
        order.status = OrderStatus.FILLED
        mock_account.find_order_by_id.return_value = order

        result = trading_manager.cancel_order(order_id="test_order_123", exchange="BINANCE.UM")

        assert result is True
        mock_connector.cancel_order.assert_not_called()
        mock_account.transition_order.assert_not_called()

    def test_cancel_already_pending_cancel_is_noop(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        order.status = OrderStatus.PENDING_CANCEL
        mock_account.find_order_by_id.return_value = order

        assert trading_manager.cancel_order(order_id="test_order_123") is True
        mock_connector.cancel_order.assert_not_called()
        mock_account.transition_order.assert_not_called()

    def test_cancel_order_empty_id(self, trading_manager):
        """Empty/ambiguous identifiers raise ValueError."""
        with pytest.raises(ValueError):
            trading_manager.cancel_order(order_id="")

    def test_cancel_order_not_found_raises(self, trading_manager, mock_account):
        """Cancelling an unknown order raises OrderNotFound (after the venue->cid fallback)."""
        mock_account.find_order_by_id.return_value = None
        mock_account.find_order_by_client_id.return_value = None
        with pytest.raises(OrderNotFound):
            trading_manager.cancel_order(order_id="missing_order_789", exchange="BINANCE.UM")

    def test_cancel_order_resolves_exchange_from_order(self, trading_manager, mock_connector, mock_account):
        """When no exchange is given, the order's own exchange is used."""
        order = _live_order()
        mock_account.find_order_by_id.return_value = order

        assert trading_manager.cancel_order(order_id="test_order_123") is True
        mock_connector.cancel_order.assert_called_once_with(
            client_order_id=order.client_order_id, venue_order_id=order.venue_order_id
        )


class _AccountRoutingContext(MockStrategyContext):
    """process_event applies straight to the AM — the PM dispatch leg
    (ProcessingManager._dispatch_account) minus strategy callbacks."""

    def __init__(self, account_manager):
        super().__init__()
        self._am = account_manager
        self.routed_events = []

    def process_event(self, event):
        self.routed_events.append(event)
        self._am.apply(event)


class TestTradingManagerCancelUpdateFailure:
    """Synchronous connector raise on cancel/update: the exception propagates to the
    caller AND the order reverts to its pre-pending status via the synthetic reject
    routed through the PM (not stuck PENDING_CANCEL/PENDING_UPDATE)."""

    @pytest.fixture
    def failure_setup(self, mock_connector):
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        am = SimulatedAccountManager(
            connectors={"BINANCE.UM": mock_connector},
            base_currencies={"BINANCE.UM": "USDT"},
            time=MockTimeProvider(),
        )
        context = _AccountRoutingContext(am)
        tm = TradingManager(
            context=context,
            connectors={"BINANCE.UM": mock_connector},
            account_manager=am,
            strategy_name="test_strategy",
            health_monitor=DummyHealthMonitor(),
        )
        order = Order(
            client_order_id="qubx_BTCUSDT_1",
            venue_order_id="V1",
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=instrument,
            submitted_at=pd.Timestamp("2023-01-01").asm8,
            quantity=0.1,
            price=50_000.0,
            side="BUY",
            status=OrderStatus.ACCEPTED,
            time_in_force="gtc",
        )
        am.add_order(order)
        return tm, am, context, order

    def test_cancel_sync_raise_propagates_and_reverts_order(self, failure_setup, mock_connector):
        tm, am, context, order = failure_setup
        mock_connector.cancel_order.side_effect = ConnectionError("venue unreachable")

        with pytest.raises(ConnectionError):
            tm.cancel_order(client_order_id=order.client_order_id)

        assert am.get_order(order.client_order_id).status is OrderStatus.ACCEPTED
        (event,) = context.routed_events
        assert isinstance(event, OrderCancelRejectedEvent)
        assert event.client_order_id == order.client_order_id
        assert "venue unreachable" in event.reason

    def test_update_sync_raise_propagates_and_reverts_order(self, failure_setup, mock_connector):
        tm, am, context, order = failure_setup
        mock_connector.update_order.side_effect = ConnectionError("venue unreachable")

        with pytest.raises(ConnectionError):
            tm.update_order(price=51_000.0, amount=0.1, client_order_id=order.client_order_id)

        assert am.get_order(order.client_order_id).status is OrderStatus.ACCEPTED
        (event,) = context.routed_events
        assert isinstance(event, OrderUpdateRejectedEvent)
        assert event.client_order_id == order.client_order_id
        assert "venue unreachable" in event.reason


class TestTradingManagerUpdateOrderGuards:
    """update_order pre-connector guards: terminal misuse raises, in-flight update no-ops."""

    def test_update_filled_raises_terminal_connector_not_called(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        order.status = OrderStatus.FILLED
        mock_account.find_order_by_id.return_value = order

        with pytest.raises(OrderAlreadyTerminal):
            trading_manager.update_order(price=51_000.0, amount=0.1, order_id="test_order_123")

        mock_connector.update_order.assert_not_called()
        mock_account.transition_order.assert_not_called()

    def test_update_while_pending_update_is_silent_noop(self, trading_manager, mock_connector, mock_account):
        order = _live_order()
        order.status = OrderStatus.PENDING_UPDATE
        mock_account.find_order_by_id.return_value = order

        assert trading_manager.update_order(price=51_000.0, amount=0.1, order_id="test_order_123") is None

        mock_connector.update_order.assert_not_called()
        mock_account.transition_order.assert_not_called()


class TestAdjustSizeMinNotional:
    """Test cases for _adjust_size with min notional rounding."""

    @pytest.fixture
    def ada_instrument(self):
        """Create a real Instrument with lot_size=0.1 and min_notional set."""
        return Instrument(
            symbol="ADAUSDC",
            market_type=MarketType.SWAP,
            exchange="TEST",
            base="ADA",
            quote="USDC",
            settle="USDC",
            exchange_symbol="ADAUSDC",
            tick_size=0.0001,
            lot_size=0.1,
            min_size=1.0,
            min_notional=25.0,
        )

    @pytest.fixture
    def notional_trading_manager(self, strategy_context, mock_connector, mock_account):
        """Create a TradingManager with a mock context that returns a quote."""
        tm = TradingManager(
            context=strategy_context,
            connectors={"BINANCE.UM": mock_connector},
            account_manager=mock_account,
            strategy_name="test_strategy",
            health_monitor=DummyHealthMonitor(),
        )
        return tm

    def _setup_quote(self, strategy_context, mid_price: float):
        """Set up the context to return a quote with the given mid price."""
        mock_quote = Mock()
        mock_quote.mid_price.return_value = mid_price
        strategy_context.quote = Mock(return_value=mock_quote)

    def _setup_no_position(self, mock_account):
        """Set up account to report no existing position (not position-reducing)."""
        position = Mock(spec=Position)
        position.quantity = 0.0
        mock_account.get_position.return_value = position

    def test_bug_case_amount_at_precision_below_min_notional(
        self, notional_trading_manager, ada_instrument, strategy_context, mock_account
    ):
        """Test the original bug: amount=36.7 with min_size=36.755 should return 36.8, not raise."""
        # Given: mid_price such that min_notional/mid_price ≈ 36.755
        # 25.0 / 0.68 = 36.7647..., let's use a price that gives ~36.755
        mid_price = 25.0 / 36.755
        self._setup_quote(strategy_context, mid_price)
        self._setup_no_position(mock_account)

        # When adjusting size 36.7 (at precision, below notional-derived min)
        result = notional_trading_manager._adjust_size(ada_instrument, 36.7)

        # Then it should round up to 36.8 instead of raising
        assert result == 36.8

    def test_normal_round_down_above_min(
        self, notional_trading_manager, ada_instrument, strategy_context, mock_account
    ):
        """Test that amounts well above min_size are rounded down normally."""
        mid_price = 25.0 / 36.755
        self._setup_quote(strategy_context, mid_price)
        self._setup_no_position(mock_account)

        result = notional_trading_manager._adjust_size(ada_instrument, 37.0)
        assert result == 37.0

    def test_round_up_amount_above_min(self, notional_trading_manager, ada_instrument, strategy_context, mock_account):
        """Test that amount=36.76 rounds up to 36.8 via step 2 (round_size_up of amount)."""
        mid_price = 25.0 / 36.755
        self._setup_quote(strategy_context, mid_price)
        self._setup_no_position(mock_account)

        result = notional_trading_manager._adjust_size(ada_instrument, 36.76)
        assert result == 36.8

    def test_genuinely_too_small_raises(self, notional_trading_manager, ada_instrument, strategy_context, mock_account):
        """Test that genuinely small amounts raise InvalidOrderSize."""
        mid_price = 25.0 / 36.755
        self._setup_quote(strategy_context, mid_price)
        self._setup_no_position(mock_account)

        with pytest.raises(InvalidOrderSize):
            notional_trading_manager._adjust_size(ada_instrument, 10.0)

    def test_borderline_reject(self, notional_trading_manager, ada_instrument, strategy_context, mock_account):
        """Test that amount=36.6 (more than one lot step below min) raises."""
        mid_price = 25.0 / 36.755
        self._setup_quote(strategy_context, mid_price)
        self._setup_no_position(mock_account)

        with pytest.raises(InvalidOrderSize):
            notional_trading_manager._adjust_size(ada_instrument, 36.6)

    def test_no_quote_falls_back_to_min_size(
        self, notional_trading_manager, ada_instrument, strategy_context, mock_account
    ):
        """Test that without a quote, min_size (not min_notional) is used."""
        strategy_context.quote = Mock(return_value=None)
        self._setup_no_position(mock_account)

        # min_size=1.0, so 5.0 should be fine
        result = notional_trading_manager._adjust_size(ada_instrument, 5.0)
        assert result == 5.0

    def test_position_reducing_uses_lot_size(
        self, notional_trading_manager, ada_instrument, strategy_context, mock_account
    ):
        """Test that position-reducing orders use lot_size, not min_notional."""
        mid_price = 25.0 / 36.755
        self._setup_quote(strategy_context, mid_price)

        # Set up an existing long position of 50 ADA
        position = Mock(spec=Position)
        position.quantity = 50.0
        mock_account.get_position.return_value = position

        # Selling 5 ADA (reducing position) — below min_notional but should work
        result = notional_trading_manager._adjust_size(ada_instrument, -5.0)
        assert result == 5.0
