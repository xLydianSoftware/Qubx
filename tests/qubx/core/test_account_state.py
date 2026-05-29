import numpy as np

from qubx.core.account_state import AccountState
from qubx.core.basics import Deal, Order, OrderOrigin, OrderStatus


def _make_order(cid="qubx-1", status=OrderStatus.SUBMITTED, venue_id=None):
    return Order(
        client_order_id=cid,
        venue_order_id=venue_id,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
    )


def _make_fill(trade_id="t1", amount=0.5, price=50_000.0):
    return Deal(
        trade_id=trade_id,
        order_id="V1",
        time=np.datetime64("2026-05-28T00:00:00"),
        amount=amount,
        price=price,
        aggressive=True,
    )


def test_add_order_inserts_and_indexes_inflight():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    assert state.get_order("qubx-1") is not None
    assert "qubx-1" in state._inflight_index


def test_add_order_terminal_not_indexed_inflight():
    state = AccountState(exchange="binance")
    state._add_order(_make_order(status=OrderStatus.FILLED))
    assert "qubx-1" not in state._inflight_index


def test_add_order_with_venue_id_indexes_by_venue():
    state = AccountState(exchange="binance")
    state._add_order(_make_order(venue_id="VENUE_ABC"))
    assert state.get_order_by_venue_id("VENUE_ABC").client_order_id == "qubx-1"


def test_set_venue_id_indexes_by_venue():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    state._set_venue_id("qubx-1", "VENUE_ABC")
    assert state.get_order_by_venue_id("VENUE_ABC").client_order_id == "qubx-1"
    assert state.get_order("qubx-1").venue_order_id == "VENUE_ABC"


def test_get_order_by_unknown_venue_id_returns_none():
    state = AccountState(exchange="binance")
    assert state.get_order_by_venue_id("NOPE") is None


def test_remove_order_drains_indexes():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    state._set_venue_id("qubx-1", "VENUE_ABC")
    state._remove_order("qubx-1")
    assert state.get_order("qubx-1") is None
    assert state.get_order_by_venue_id("VENUE_ABC") is None
    assert "qubx-1" not in state._inflight_index


def test_transition_to_accepted_drains_inflight():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    state._transition_order("qubx-1", OrderStatus.ACCEPTED, np.datetime64("2026-05-28T00:01:00"))
    order = state.get_order("qubx-1")
    assert order.status is OrderStatus.ACCEPTED
    assert order.last_updated_at == np.datetime64("2026-05-28T00:01:00")
    assert "qubx-1" not in state._inflight_index


def test_transition_to_terminal_populates_evict_index():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    now = np.datetime64("2026-05-28T00:01:00")
    state._transition_order("qubx-1", OrderStatus.FILLED, now)
    assert "qubx-1" not in state._inflight_index
    assert state._pending_evict_index["qubx-1"] == now


def test_transition_back_to_pending_re_indexes_inflight():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    state._transition_order("qubx-1", OrderStatus.ACCEPTED, np.datetime64("2026-05-28T00:01:00"))
    # transitioning to ACCEPTED drains the inflight index
    assert "qubx-1" not in state._inflight_index
    state._transition_order("qubx-1", OrderStatus.PENDING_CANCEL, np.datetime64("2026-05-28T00:02:00"))
    # a non-terminal, non-accepted/partial status re-indexes as inflight
    assert "qubx-1" in state._inflight_index


def test_apply_fill_accumulates_quantity_and_avg_price():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    now = np.datetime64("2026-05-28T00:01:00")
    state._apply_fill("qubx-1", _make_fill(trade_id="t1", amount=0.5, price=50_000.0), now)
    state._apply_fill("qubx-1", _make_fill(trade_id="t2", amount=0.5, price=51_000.0), now)
    order = state.get_order("qubx-1")
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price == 50_500.0


def test_apply_fill_dedup_by_trade_id():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    now = np.datetime64("2026-05-28T00:01:00")
    fill = _make_fill(trade_id="t1", amount=0.5)
    state._apply_fill("qubx-1", fill, now)
    state._apply_fill("qubx-1", fill, now)
    assert state.get_order("qubx-1").filled_quantity == 0.5


def test_apply_fill_uses_abs_amount_for_sell():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    now = np.datetime64("2026-05-28T00:01:00")
    state._apply_fill("qubx-1", _make_fill(trade_id="t1", amount=-0.5), now)
    assert state.get_order("qubx-1").filled_quantity == 0.5


def test_evict_to_history_moves_order_and_drains_indexes():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    state._set_venue_id("qubx-1", "VENUE_ABC")
    state._transition_order("qubx-1", OrderStatus.FILLED, np.datetime64("2026-05-28T00:01:00"))
    state._evict_to_history("qubx-1")
    assert "qubx-1" not in state.active_orders
    assert state.get_order_by_venue_id("VENUE_ABC") is None
    assert "qubx-1" not in state._pending_evict_index
    # still resolvable via terminal history
    assert state.get_order("qubx-1") is not None


def test_get_order_falls_back_to_terminal_history():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    state._transition_order("qubx-1", OrderStatus.FILLED, np.datetime64("2026-05-28T00:01:00"))
    state._evict_to_history("qubx-1")
    order = state.get_order("qubx-1")
    assert order is not None
    assert order.status is OrderStatus.FILLED


def test_get_orders_returns_copy():
    state = AccountState(exchange="binance")
    state._add_order(_make_order())
    orders = state.get_orders()
    orders.clear()
    assert state.get_order("qubx-1") is not None


def test_set_and_get_position():
    from qubx.core.basics import Position

    state = AccountState(exchange="binance")
    inst = object()
    pos = Position.__new__(Position)
    state._set_position(inst, pos)
    assert state.get_position(inst) is pos
    assert state.get_positions()[inst] is pos
    assert state.get_position(object()) is None


def test_update_and_get_balance():
    from qubx.core.basics import Balance

    state = AccountState(exchange="binance")
    bal = Balance(exchange="binance", currency="USDT", free=100.0, locked=0.0, total=100.0)
    state._update_balance("USDT", bal)
    assert state.get_balance("USDT") is bal
    assert state.get_balance("BTC") is None
