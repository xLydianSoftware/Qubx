from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.basics import OrderStatus, Position
from qubx.core.exceptions import OrderNotFound
from qubx.core.mixins.trading import TradingManager
from qubx.health.dummy import DummyHealthMonitor


def _order(order_id="exchange_order_1", client_order_id="cid_1", status=OrderStatus.ACCEPTED):
    instrument = Mock()
    instrument.exchange = "BINANCE.UM"
    instrument.symbol = "BTCUSDT"
    instrument.round_size_down = lambda x: float(x)
    instrument.round_price_down = lambda x: float(x)
    instrument.round_price_up = lambda x: float(x)
    instrument.min_size = 0.0
    instrument.min_notional = 0.0
    instrument.lot_size = 0.0
    instrument.quantity_multiplier = 1.0
    order = Mock()
    order.client_order_id = client_order_id
    order.venue_order_id = order_id
    order.instrument = instrument
    order.status = status
    return order


@pytest.fixture
def trading_manager():
    connector = Mock()
    connector.exchange_name = "BINANCE.UM"
    connector.make_client_id = lambda s: s

    account_manager = Mock()
    account_manager.get_position = lambda instrument: Position(instrument=instrument)

    mock_context = Mock()
    mock_context.time.return_value = pd.Timestamp("2024-01-01").asm8
    mock_context.quote.return_value = None

    return TradingManager(
        context=mock_context,
        connectors={"BINANCE.UM": connector},
        account_manager=account_manager,
        health_monitor=DummyHealthMonitor(),
        strategy_name="test",
    )


def test_cancel_order_requires_exactly_one_identifier(trading_manager):
    with pytest.raises(ValueError):
        trading_manager.cancel_order(exchange="BINANCE.UM")
    with pytest.raises(ValueError):
        trading_manager.cancel_order(order_id="o1", client_order_id="c1", exchange="BINANCE.UM")


def test_cancel_order_routes_client_order_id(trading_manager):
    order = _order(order_id="exchange_order_1", client_order_id="cid_1")
    trading_manager._account_manager.find_order_by_client_id.return_value = order
    trading_manager._account_manager.find_order_by_id.return_value = None

    ok = trading_manager.cancel_order(client_order_id="cid_1", exchange="BINANCE.UM")

    assert ok is True
    trading_manager._account_manager.transition_order.assert_called_once_with(
        "BINANCE.UM", "cid_1", OrderStatus.PENDING_CANCEL
    )
    trading_manager._exchange_to_connector["BINANCE.UM"].cancel_order.assert_called_once_with(
        client_order_id="cid_1", venue_order_id="exchange_order_1"
    )


def test_cancel_order_routes_order_id(trading_manager):
    order = _order(order_id="exchange_order_2", client_order_id="cid_2")
    trading_manager._account_manager.find_order_by_id.return_value = order
    trading_manager._account_manager.find_order_by_client_id.return_value = None

    trading_manager.cancel_order(order_id="exchange_order_2", exchange="BINANCE.UM")

    trading_manager._exchange_to_connector["BINANCE.UM"].cancel_order.assert_called_once_with(
        client_order_id="cid_2", venue_order_id="exchange_order_2"
    )


def test_cancel_unknown_order_raises(trading_manager):
    trading_manager._account_manager.find_order_by_client_id.return_value = None
    trading_manager._account_manager.find_order_by_id.return_value = None
    with pytest.raises(OrderNotFound):
        trading_manager.cancel_order(client_order_id="cid_1", exchange="BINANCE.UM")


def test_update_order_routes_client_order_id(trading_manager):
    order = _order(order_id="exchange_order_1", client_order_id="cid_1")
    trading_manager._account_manager.find_order_by_client_id.return_value = order
    trading_manager._account_manager.find_order_by_id.return_value = None

    trading_manager.update_order(client_order_id="cid_1", price=123.0, amount=1.0, exchange="BINANCE.UM")

    trading_manager._account_manager.transition_order.assert_called_once_with(
        "BINANCE.UM", "cid_1", OrderStatus.PENDING_UPDATE
    )
    trading_manager._exchange_to_connector["BINANCE.UM"].update_order.assert_called_once_with(
        client_order_id="cid_1", venue_order_id="exchange_order_1", price=123.0, quantity=1.0
    )


def test_update_order_requires_exactly_one_identifier(trading_manager):
    with pytest.raises(ValueError):
        trading_manager.update_order(price=123.0, amount=1.0, exchange="BINANCE.UM")
    with pytest.raises(ValueError):
        trading_manager.update_order(
            order_id="o1", client_order_id="c1", price=123.0, amount=1.0, exchange="BINANCE.UM"
        )
