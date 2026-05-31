from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.basics import OrderOrigin, OrderStatus
from qubx.core.exceptions import OrderAlreadyTerminal
from qubx.core.mixins.trading import TradingManager


class _Stub:
    def generate_id(self, ctx, symbol):
        return f"qubx-{symbol}-1"


class _Ctx:
    def time(self):
        return np.datetime64("2026-05-28")


def _setup():
    am = MagicMock()
    connector = MagicMock()
    connector.exchange_name = "binance"
    connector.make_client_id = lambda s: s
    instrument = MagicMock(exchange="binance", symbol="BTCUSDT")
    # Instrument size/price adjustment is mocked through pass-through methods so
    # the test focuses on the new connector/AM path, not rounding logic.
    instrument.round_size_down = lambda x: x
    instrument.round_size_up = lambda x: x
    instrument.round_price_down = lambda x: x
    instrument.round_price_up = lambda x: x
    instrument.min_size = 0.0
    instrument.min_notional = 0.0
    instrument.lot_size = 0.0
    instrument.quantity_multiplier = 1.0
    tm = TradingManager.__new__(TradingManager)
    tm._account_manager = am
    tm._connectors = {"binance": connector}
    tm._context = _Ctx()
    tm._client_id_store = _Stub()
    tm._health_monitor = MagicMock()
    # The shared size-adjustment helpers read the old _account for reduce-only
    # checks; a mock keeps them inert so the test isolates the new connector path.
    tm._account = MagicMock()
    tm._account.get_position.return_value = MagicMock(quantity=0.0)
    return tm, am, connector, instrument


def test_trade_builds_submitted_and_calls_connector_then_am():
    tm, am, connector, inst = _setup()
    order = tm.trade(inst, amount=1.0, price=50_000.0)
    assert order.status is OrderStatus.SUBMITTED
    assert order.client_order_id == "qubx-BTCUSDT-1"
    assert order.origin is OrderOrigin.FRAMEWORK
    connector.submit_order.assert_called_once()
    am.add_order.assert_called_once_with("binance", order)


def test_cancel_terminal_is_noop():
    tm, am, connector, inst = _setup()
    am.get_order.return_value = MagicMock(
        status=OrderStatus.FILLED, instrument=inst, client_order_id="c",
    )
    tm.cancel("c")
    connector.cancel_order.assert_not_called()


def test_update_terminal_raises():
    tm, am, connector, inst = _setup()
    am.get_order.return_value = MagicMock(
        status=OrderStatus.FILLED, instrument=inst, client_order_id="c",
    )
    with pytest.raises(OrderAlreadyTerminal):
        tm.update("c", price=51_000.0)
