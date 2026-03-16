from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.core.mixins.trading import TradingManager
from qubx.health.dummy import DummyHealthMonitor


@pytest.fixture
def trading_manager():
    mock_broker = Mock()
    mock_broker.exchange.return_value = "BINANCE.UM"
    mock_broker.cancel_order.return_value = True
    mock_broker.cancel_order_async.return_value = None
    mock_broker.update_order.return_value = Mock()
    mock_broker.update_order_async.return_value = "cid_1"

    mock_account = Mock()

    mock_context = Mock()
    mock_context.time.return_value = pd.Timestamp("2024-01-01").asm8

    tm = TradingManager(
        context=mock_context,
        brokers=[mock_broker],
        account=mock_account,
        health_monitor=DummyHealthMonitor(),
        strategy_name="test",
    )
    tm._exchange_to_broker = {"BINANCE.UM": mock_broker}
    return tm


def test_cancel_order_requires_exactly_one_identifier(trading_manager):
    with pytest.raises(ValueError):
        trading_manager.cancel_order(exchange="BINANCE.UM")
    with pytest.raises(ValueError):
        trading_manager.cancel_order(order_id="o1", client_order_id="c1", exchange="BINANCE.UM")


def test_cancel_order_routes_client_order_id(trading_manager):
    order = Mock()
    order.id = "exchange_order_1"
    order.client_id = "cid_1"

    trading_manager._account.find_order_by_client_id.return_value = order
    trading_manager._account.find_order_by_id.return_value = None

    ok = trading_manager.cancel_order(client_order_id="cid_1", exchange="BINANCE.UM")

    assert ok is True
    trading_manager._exchange_to_broker["BINANCE.UM"].cancel_order.assert_called_once_with(order_id=None, client_order_id="cid_1")
    trading_manager._account.remove_order.assert_called_once_with("exchange_order_1", "BINANCE.UM")


def test_cancel_order_async_routes_order_id(trading_manager):
    order = Mock()
    order.id = "exchange_order_2"
    order.client_id = None

    trading_manager._account.find_order_by_id.return_value = order
    trading_manager._account.find_order_by_client_id.return_value = None

    trading_manager.cancel_order_async(order_id="exchange_order_2", exchange="BINANCE.UM")

    trading_manager._exchange_to_broker["BINANCE.UM"].cancel_order_async.assert_called_once_with(
        order_id="exchange_order_2", client_order_id=None
    )
    trading_manager._account.remove_order.assert_called_once_with("exchange_order_2", "BINANCE.UM")


def test_update_order_routes_client_order_id(trading_manager):
    trading_manager._account.find_order_by_client_id.return_value = None
    trading_manager._account.find_order_by_id.return_value = None

    trading_manager.update_order(client_order_id="cid_1", price=123.0, amount=1.0, exchange="BINANCE.UM")

    trading_manager._exchange_to_broker["BINANCE.UM"].update_order.assert_called_once_with(
        order_id=None, client_order_id="cid_1", price=123.0, amount=1.0
    )


def test_update_order_async_requires_exactly_one_identifier(trading_manager):
    with pytest.raises(ValueError):
        trading_manager.update_order_async(price=123.0, amount=1.0, exchange="BINANCE.UM")
    with pytest.raises(ValueError):
        trading_manager.update_order_async(order_id="o1", client_order_id="c1", price=123.0, amount=1.0, exchange="BINANCE.UM")


