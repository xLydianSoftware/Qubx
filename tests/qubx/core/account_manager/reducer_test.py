from typing import TypeVar

import numpy as np

from qubx.core.account_manager.events import (
    DealEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    PositionUpdateEvent,
)
from qubx.core.account_manager.reducer import apply
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import Deal, Order, OrderChange, OrderOrigin, OrderSide, OrderStatus, OrderType, Position
from qubx.core.lookups import lookup

T0 = np.datetime64("2026-05-28T00:00:00", "ns")
T1 = np.datetime64("2026-05-28T00:01:00", "ns")

_T = TypeVar("_T")


def _present(value: _T | None) -> _T:
    assert value is not None
    return value


def _state() -> AccountState:
    return AccountState("binance", "USDT")


def _fill(trade_id: str = "t1", amount: float = 0.5, price: float = 100.0) -> Deal:
    return Deal(id=trade_id, order_id="v1", time=T0, amount=amount, price=price, aggressive=True)


def _order(state: AccountState, cid: str = "c1", status: OrderStatus = OrderStatus.SUBMITTED, venue_id=None) -> Order:
    order = Order(
        client_id=cid,
        type=OrderType.LIMIT,
        instrument=None,  # type: ignore[arg-type]
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_id=venue_id,
        price=100.0,
        last_updated_at=T0 if status.is_terminal else None,
        origin=OrderOrigin.FRAMEWORK,
    )
    state._add_order(order)
    return order


def test_accept_transitions_and_sets_venue_id():
    state = _state()
    _order(state)
    r = apply(state, OrderAcceptedEvent(timestamp=T0, client_order_id="c1", venue_order_id="V1"), T1)
    assert r.order is not None
    assert r.order.status is OrderStatus.ACCEPTED
    assert r.order.venue_id == "V1"
    assert r.order.last_updated_at == T1
    assert r.order_change is OrderChange.ACCEPTED
    assert r.deal is None and r.position is None


def test_accept_on_terminal_is_noop():
    state = _state()
    _order(state, status=OrderStatus.FILLED)
    r = apply(state, OrderAcceptedEvent(timestamp=T0, client_order_id="c1"), T1)
    assert r.order is None


def test_accept_unknown_order_is_noop():
    r = apply(_state(), OrderAcceptedEvent(timestamp=T0, client_order_id="nope"), T1)
    assert r.order is None


def test_cancel_transitions_to_terminal():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderCanceledEvent(timestamp=T0, client_order_id="c1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.CANCELED
    assert r.order_change is OrderChange.CANCELED


def test_cancel_on_terminal_is_noop():
    state = _state()
    _order(state, status=OrderStatus.FILLED)
    r = apply(state, OrderCanceledEvent(timestamp=T0, client_order_id="c1"), T1)
    assert r.order is None


def test_expire_transitions_to_terminal():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderExpiredEvent(timestamp=T0, client_order_id="c1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.EXPIRED
    assert r.order_change is OrderChange.EXPIRED


def test_reject_transitions_and_records_reason():
    state = _state()
    _order(state)
    r = apply(state, OrderRejectedEvent(timestamp=T0, client_order_id="c1", reason="insufficient margin"), T1)
    assert r.order is not None
    assert r.order.status is OrderStatus.REJECTED
    assert r.order.rejected_reason == "insufficient margin"
    assert r.order_change is OrderChange.REJECTED


def test_reject_unknown_order_is_noop():
    r = apply(_state(), OrderRejectedEvent(timestamp=T0, client_order_id="nope", reason="x"), T1)
    assert r.order is None


def test_reject_resolves_by_venue_id():
    state = _state()
    _order(state, cid="c1", status=OrderStatus.ACCEPTED, venue_id="V1")
    r = apply(state, OrderRejectedEvent(timestamp=T0, client_order_id="", venue_order_id="V1", reason="x"), T1)
    assert r.order is not None and r.order.client_id == "c1"
    assert r.order.status is OrderStatus.REJECTED


def test_fill_with_embedded_deal():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderFilledEvent(timestamp=T0, client_order_id="c1", fill=_fill("t1", 0.5)), T1)
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.order_change is OrderChange.FILLED
    assert _present(r.deal).id == "t1"
    assert r.order.filled_quantity == 0.5


def test_fill_without_deal_split_stream():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderFilledEvent(timestamp=T0, client_order_id="c1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.deal is None
    assert r.order.filled_quantity == 0.0  # deal arrives separately via DealEvent


def test_fill_duplicate_deal_not_double_counted():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.5)
    state._apply_fill("c1", f, T0)  # already applied (e.g. earlier DealEvent)
    r = apply(state, OrderFilledEvent(timestamp=T0, client_order_id="c1", fill=f), T1)
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.deal is None  # deduped -> no on_execution
    assert r.order.filled_quantity == 0.5


def test_partial_fill_first_transitions():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderPartiallyFilledEvent(timestamp=T0, client_order_id="c1", fill=_fill("t1", 0.3)), T1)
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED
    assert r.order_change is OrderChange.PARTIALLY_FILLED
    assert _present(r.deal).id == "t1"
    assert r.order.filled_quantity == 0.3


def test_partial_fill_before_accept():
    state = _state()
    _order(state, status=OrderStatus.SUBMITTED)
    r = apply(state, OrderPartiallyFilledEvent(timestamp=T0, client_order_id="c1", fill=_fill("t1", 0.3)), T1)
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED


def test_subsequent_partial_fill_is_execution_only():
    state = _state()
    _order(state, status=OrderStatus.PARTIALLY_FILLED)
    r = apply(state, OrderPartiallyFilledEvent(timestamp=T0, client_order_id="c1", fill=_fill("t2", 0.2)), T1)
    assert r.order is None  # no status transition
    assert _present(r.deal).id == "t2"


def test_partial_fill_while_pending_is_execution_only():
    state = _state()
    _order(state, status=OrderStatus.PENDING_CANCEL)
    r = apply(state, OrderPartiallyFilledEvent(timestamp=T0, client_order_id="c1", fill=_fill("t1", 0.2)), T1)
    assert r.order is None  # pending status not disturbed
    assert _present(r.deal).id == "t1"
    assert _present(state.get_active_order("c1")).status is OrderStatus.PENDING_CANCEL


def test_deal_event_applies_and_reports_execution_only():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, DealEvent(timestamp=T0, client_order_id="c1", deal=_fill("t1", 0.4)), T1)
    assert r.order is None  # deal event never changes status
    assert _present(r.deal).id == "t1"
    assert _present(state.get_active_order("c1")).filled_quantity == 0.4


def test_deal_event_duplicate_is_noop():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.4)
    apply(state, DealEvent(timestamp=T0, client_order_id="c1", deal=f), T1)
    r = apply(state, DealEvent(timestamp=T0, client_order_id="c1", deal=f), T1)
    assert r.order is None and r.deal is None


def test_dedup_across_deal_and_order_stream():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.5)
    r1 = apply(state, DealEvent(timestamp=T0, client_order_id="c1", deal=f), T1)
    assert _present(r1.deal).id == "t1"
    r2 = apply(state, OrderFilledEvent(timestamp=T0, client_order_id="c1", fill=f), T1)
    assert r2.order is not None and r2.order.status is OrderStatus.FILLED
    assert r2.deal is None  # already applied via DealEvent
    assert r2.order.filled_quantity == 0.5


def test_unhandled_event_type_is_noop():
    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    r = apply(_state(), PositionUpdateEvent(timestamp=T0, position=Position(inst)), T1)
    assert r.order is None and r.deal is None and r.position is None
