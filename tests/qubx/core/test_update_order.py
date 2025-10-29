"""
Tests for update_order functionality across the trading stack.

This module tests the update_order implementation in:
- CcxtBroker (with mocked CCXT exchange)
- TradingManager
- StrategyContext delegation
- SimulatedBroker
"""

from unittest.mock import Mock, patch

import pytest

from qubx.backtester.broker import SimulatedBroker
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.basics import MarketType, Order, dt_64
from qubx.core.exceptions import BadRequest, OrderNotFound
from qubx.core.mixins.trading import TradingManager


# Test fixtures
@pytest.fixture
def mock_instrument():
    """Create a mock instrument for testing."""
    instrument = Mock()
    instrument.symbol = "BTCUSDT"
    instrument.exchange = "BINANCE.UM"
    instrument.market_type = MarketType.SWAP
    instrument.min_size = 0.001
    instrument.lot_size = 0.001
    instrument.tick_size = 0.1
    instrument.round_size_down = Mock(side_effect=lambda x: float(x))
    instrument.round_price_down = Mock(side_effect=lambda x: float(x))
    instrument.round_price_up = Mock(side_effect=lambda x: float(x))
    return instrument


@pytest.fixture
def mock_limit_order(mock_instrument):
    """Create a mock limit order for testing."""
    return Order(
        id="test_order_123",
        type="LIMIT",
        instrument=mock_instrument,
        time=dt_64("2024-01-01T10:00:00"),
        quantity=1.5,
        price=50000.0,
        side="BUY",
        status="OPEN",
        time_in_force="gtc",
        client_id="qubx_BTCUSDT_12345",
    )


class TestCcxtBrokerUpdateOrder:
    """Test CcxtBroker update_order functionality."""

    def test_update_order_exchange_supports_editing(self, mock_instrument, mock_limit_order):
        """Test update_order when exchange supports direct editing."""
        # Setup mocks
        mock_exchange_manager = Mock()
        mock_exchange = Mock()
        mock_exchange.has = {"editOrder": True}
        mock_exchange_manager.exchange = mock_exchange

        mock_account = Mock()
        mock_account.get_orders.return_value = {"test_order_123": mock_limit_order}

        mock_loop = Mock()
        future_result = Mock()
        updated_order = Order(
            id="test_order_123",
            type="LIMIT",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=2.0,  # Updated amount
            price=51000.0,  # Updated price
            side="BUY",
            status="OPEN",
            time_in_force="gtc",
            client_id="qubx_BTCUSDT_12345",
        )
        future_result.result.return_value = (updated_order, None)
        mock_loop.submit.return_value = future_result

        # Create broker with mocks
        broker = CcxtBroker(
            exchange_manager=mock_exchange_manager,
            channel=Mock(),
            time_provider=Mock(),
            account=mock_account,
            data_provider=Mock(),
        )
        # Mock the _exchange_manager.exchange.asyncio_loop directly
        mock_exchange_manager.exchange.asyncio_loop = Mock()

        # Create a mock that returns our test future
        def mock_submit(coro):
            return future_result

        # Mock the AsyncThreadLoop.submit method
        with patch("qubx.utils.misc.AsyncThreadLoop.submit", side_effect=mock_submit):
            # Execute update_order
            result = broker.update_order("test_order_123", 51000.0, 2.0)

        # Verify
        assert result == updated_order
        mock_account.process_order.assert_called_once_with(updated_order)

    def test_update_order_exchange_no_support_fallback(self, mock_instrument, mock_limit_order):
        """Test update_order fallback strategy when exchange doesn't support editing."""
        # Setup mocks
        mock_exchange_manager = Mock()
        mock_exchange = Mock()
        mock_exchange.has = {"editOrder": False}  # No edit support
        mock_exchange_manager.exchange = mock_exchange

        mock_account = Mock()
        mock_account.get_orders.return_value = {"test_order_123": mock_limit_order}

        # Create broker with mocks
        broker = CcxtBroker(
            exchange_manager=mock_exchange_manager,
            channel=Mock(),
            time_provider=Mock(),
            account=mock_account,
            data_provider=Mock(),
        )

        new_order = Order(
            id="new_order_456",
            type="LIMIT",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=2.0,
            price=51000.0,
            side="BUY",
            status="OPEN",
            time_in_force="gtc",
            client_id="qubx_BTCUSDT_12345",
        )

        # Mock the cancel and send methods
        broker.cancel_order = Mock(return_value=True)
        broker.send_order = Mock(return_value=new_order)
        mock_account.process_order = Mock()

        # Execute update_order
        result = broker.update_order("test_order_123", 51000.0, 2.0)

        # Verify fallback strategy was used
        broker.cancel_order.assert_called_once_with("test_order_123")
        broker.send_order.assert_called_once()
        assert result == new_order

    def test_update_order_not_found(self):
        """Test update_order raises OrderNotFound when order doesn't exist."""
        mock_exchange_manager = Mock()
        mock_account = Mock()
        mock_account.get_orders.return_value = {}  # No orders

        broker = CcxtBroker(
            exchange_manager=mock_exchange_manager,
            channel=Mock(),
            time_provider=Mock(),
            account=mock_account,
            data_provider=Mock(),
        )

        with pytest.raises(OrderNotFound):
            broker.update_order("nonexistent_order", 50000.0, 1.0)

    def test_update_order_non_updatable_status(self, mock_instrument):
        """Test that updating orders with non-updatable status raises BadRequest."""
        # Test FILLED order (should be rejected)
        filled_order = Order(
            id="filled_order_123",
            type="LIMIT",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=1.0,
            price=50000.0,
            side="BUY",
            status="FILLED",  # Cannot update filled orders
            time_in_force="gtc",
            client_id="qubx_BTCUSDT_12345",
        )

        mock_exchange_manager = Mock()
        mock_account = Mock()
        mock_account.get_orders.return_value = {"filled_order_123": filled_order}

        broker = CcxtBroker(
            exchange_manager=mock_exchange_manager,
            channel=Mock(),
            time_provider=Mock(),
            account=mock_account,
            data_provider=Mock(),
        )

        with pytest.raises(BadRequest, match="status 'FILLED' cannot be updated"):
            broker.update_order("filled_order_123", 50000.0, 1.0)

        # Test CANCELED order (should also be rejected)
        canceled_order = Order(
            id="canceled_order_456",
            type="LIMIT",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=1.0,
            price=49000.0,
            side="SELL",
            status="CANCELED",  # Cannot update canceled orders
            time_in_force="gtc",
        )

        mock_account.get_orders.return_value = {"canceled_order_456": canceled_order}
        with pytest.raises(BadRequest, match="status 'CANCELED' cannot be updated"):
            broker.update_order("canceled_order_456", 48000.0, 1.0)

        # Test CLOSED order (should also be rejected)
        closed_order = Order(
            id="closed_order_789",
            type="MARKET",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=1.0,
            price=50000.0,
            side="BUY",
            status="CLOSED",  # Cannot update closed orders
            time_in_force="ioc",
        )

        mock_account.get_orders.return_value = {"closed_order_789": closed_order}
        with pytest.raises(BadRequest, match="status 'CLOSED' cannot be updated"):
            broker.update_order("closed_order_789", 51000.0, 2.0)

    def test_update_order_market_order_allowed_if_pending(self, mock_instrument):
        """Test that even MARKET orders can be updated if they're still in updatable status."""
        pending_market_order = Order(
            id="pending_market_order_789",
            type="MARKET",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=1.0,
            price=51000.0,
            side="BUY",
            status="PENDING",  # Still updatable
            time_in_force="gtc",
        )

        mock_exchange_manager = Mock()
        mock_exchange_manager.exchange.has.get.return_value = False  # Force fallback
        mock_account = Mock()
        mock_account.get_orders.return_value = {"pending_market_order_789": pending_market_order}

        # Mock the fallback behavior (cancel + recreate)
        mock_cancel_order = Mock(return_value=True)
        mock_send_order = Mock(return_value=pending_market_order)

        broker = CcxtBroker(
            exchange_manager=mock_exchange_manager,
            channel=Mock(),
            time_provider=Mock(),
            account=mock_account,
            data_provider=Mock(),
        )

        # Patch the methods that fallback uses
        with (
            patch.object(broker, "cancel_order", mock_cancel_order),
            patch.object(broker, "send_order", mock_send_order),
        ):
            # This should NOT raise BadRequest - orders with PENDING status are updatable
            result = broker.update_order("pending_market_order_789", 52000.0, 2.0)

            # Verify fallback was used (cancel + send_order)
            mock_cancel_order.assert_called_once_with("pending_market_order_789")
            mock_send_order.assert_called_once_with(
                instrument=pending_market_order.instrument,
                order_side=pending_market_order.side,
                order_type=pending_market_order.type,  # Preserve original order type
                amount=2.0,
                price=52000.0,
                client_id=pending_market_order.client_id,
                time_in_force=pending_market_order.time_in_force,
            )
            assert result == pending_market_order


class TestTradingManagerUpdateOrder:
    """Test TradingManager update_order functionality."""

    def test_update_order_delegates_to_broker(self, mock_limit_order):
        """Test TradingManager delegates update_order to the appropriate broker."""
        mock_broker = Mock()
        updated_order = mock_limit_order
        updated_order.price = 51000.0
        updated_order.quantity = 2.0
        mock_broker.update_order.return_value = updated_order
        mock_broker.exchange.return_value = "BINANCE.UM"

        mock_account = Mock()
        mock_account.get_orders.return_value = {"test_order_123": mock_limit_order}

        trading_manager = TradingManager(
            context=Mock(), brokers=[mock_broker], account=mock_account, strategy_name="test_strategy"
        )
        trading_manager._exchange_to_broker = {"BINANCE.UM": mock_broker}

        result = trading_manager.update_order("test_order_123", 51000.0, 2.0, "BINANCE.UM")

        # Verify delegation
        mock_broker.update_order.assert_called_once_with("test_order_123", 51000.0, 2.0)
        mock_account.process_order.assert_called_once_with(updated_order)
        assert result == updated_order


class TestSimulatedBrokerUpdateOrder:
    """Test SimulatedBroker update_order functionality."""

    def test_update_order_cancel_and_recreate(self, mock_instrument, mock_limit_order):
        """Test SimulatedBroker uses cancel+recreate strategy."""
        mock_account = Mock()
        mock_account.get_orders.return_value = {"test_order_123": mock_limit_order}

        new_order = Order(
            id="new_order_789",
            type="LIMIT",
            instrument=mock_instrument,
            time=dt_64("2024-01-01T10:00:00"),
            quantity=2.0,  # Updated
            price=51000.0,  # Updated
            side="BUY",
            status="OPEN",
            time_in_force="gtc",
            client_id="qubx_BTCUSDT_12345",
        )

        broker = SimulatedBroker(Mock(), mock_account, Mock())
        broker.cancel_order = Mock()
        broker.send_order = Mock(return_value=new_order)

        result = broker.update_order("test_order_123", 51000.0, 2.0)

        # Verify cancel+recreate strategy
        broker.cancel_order.assert_called_once_with("test_order_123")
        broker.send_order.assert_called_once()
        assert result == new_order

    def test_update_order_not_found_backtester(self):
        """Test SimulatedBroker raises OrderNotFound when order doesn't exist."""
        mock_account = Mock()
        mock_account.get_orders.return_value = {}  # No orders found

        broker = SimulatedBroker(Mock(), mock_account, Mock())

        with pytest.raises(OrderNotFound):
            broker.update_order("nonexistent_order", 50000.0, 1.0)
