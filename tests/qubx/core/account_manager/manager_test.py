from typing import TypeVar

import numpy as np

from qubx.core.account_manager.events import OrderAcceptedEvent, OrderCanceledEvent, OrderFilledEvent
from qubx.core.account_manager.manager import AccountManager
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import (
    AssetBalance,
    Deal,
    ITimeProvider,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
)
from qubx.core.lookups import lookup

T0 = np.datetime64("2026-05-28T00:00:00", "ns")
T1 = np.datetime64("2026-05-28T00:01:00", "ns")

_btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
assert _btc is not None
BTC = _btc
EX = BTC.exchange

_T = TypeVar("_T")


def _present(value: _T | None) -> _T:
    assert value is not None
    return value


class _Time(ITimeProvider):
    def time(self) -> np.datetime64:
        return T1


def _am(*exchanges: str) -> AccountManager:
    return AccountManager({ex: "USDT" for ex in (exchanges or (EX,))}, _Time())


def _order(cid: str = "c1", status: OrderStatus = OrderStatus.SUBMITTED, venue_id=None) -> Order:
    return Order(
        client_id=cid,
        type=OrderType.LIMIT,
        instrument=BTC,
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_id=venue_id,
        price=100.0,
        last_updated_at=T0 if status.is_terminal else None,
        origin=OrderOrigin.FRAMEWORK,
    )


def _fill(trade_id: str = "t1", amount: float = 0.5, price: float = 50_000.0) -> Deal:
    return Deal(id=trade_id, order_id="v1", time=T0, amount=amount, price=price, aggressive=True)


def _seed(state: AccountState, amount: float) -> None:
    state._update_balance("USDT", AssetBalance(exchange=state.exchange, currency="USDT", free=amount, locked=0.0, total=amount))


def test_apply_routes_by_instrument_and_delegates_to_reducer():
    am = _am()
    am.get_state(EX)._add_order(_order("c1", OrderStatus.SUBMITTED))
    r = am.apply(OrderAcceptedEvent(timestamp=T0, instrument=BTC, client_order_id="c1", venue_order_id="V1"))
    assert r.order is not None and r.order.status is OrderStatus.ACCEPTED
    assert r.order.last_updated_at == T1  # the AM's clock, not the event timestamp


def test_apply_routes_by_order_id_across_exchanges():
    am = _am(EX, "OKX")
    am.get_state("OKX")._add_order(_order("c1", OrderStatus.ACCEPTED))
    r = am.apply(OrderCanceledEvent(timestamp=T0, client_order_id="c1"))  # no instrument, 2 exchanges
    assert r.order is not None and r.order.status is OrderStatus.CANCELED
    assert _present(am.get_state("OKX").get_order("c1")).status is OrderStatus.CANCELED


def test_apply_unroutable_returns_empty():
    am = _am(EX, "OKX")  # 2 exchanges, no fallback
    r = am.apply(OrderCanceledEvent(timestamp=T0, client_order_id="unknown"))
    assert r.order is None


def test_apply_fill_books_position_via_manager():
    am = _am()
    am.get_state(EX)._add_order(_order("c1", OrderStatus.ACCEPTED))
    r = am.apply(OrderFilledEvent(timestamp=T0, instrument=BTC, client_order_id="c1", fill=_fill("t1", 0.5)))
    assert r.position is not None and r.position.quantity == 0.5
    assert _present(am.get_position(BTC)).quantity == 0.5


def test_get_orders_aggregates_and_filters_terminal():
    am = _am(EX, "OKX")
    am.get_state(EX)._add_order(_order("a", OrderStatus.ACCEPTED))
    am.get_state("OKX")._add_order(_order("b", OrderStatus.ACCEPTED))
    am.get_state("OKX")._add_order(_order("c", OrderStatus.FILLED))  # terminal, retained in state
    assert set(am.get_orders().keys()) == {"a", "b"}  # terminal 'c' excluded
    assert set(am.get_orders(exchange="OKX").keys()) == {"b"}


def test_total_capital_aggregates_across_exchanges():
    am = _am(EX, "OKX")
    _seed(am.get_state(EX), 1000.0)
    _seed(am.get_state("OKX"), 500.0)
    assert am.get_total_capital() == 1500.0
    assert am.get_total_capital(exchange=EX) == 1000.0
    assert am.get_available_margin() == 1500.0  # no positions -> no initial margin


def test_margin_ratio_no_maint_is_100():
    assert _am().get_margin_ratio() == 100.0


def test_get_order_by_exchange_and_shortcuts():
    am = _am(EX, "OKX")
    am.get_state("OKX")._add_order(_order("c1", OrderStatus.ACCEPTED))
    assert _present(am.get_order("c1", exchange="OKX")).status is OrderStatus.ACCEPTED  # direct
    assert am.get_order("c1", exchange=EX) is None  # wrong exchange
    assert _present(am.get_order("c1")).client_id == "c1"  # multi-exchange scan fallback
    # single-exchange shortcut (no scan)
    one = _am()
    one.get_state(EX)._add_order(_order("x", OrderStatus.ACCEPTED))
    assert _present(one.get_order("x")).client_id == "x"
