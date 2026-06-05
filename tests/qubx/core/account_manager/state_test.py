"""Unit tests for AccountState — the pure per-exchange data + index store.

Harvested from the #301 redesign branch and adapted to this branch's API:
underscore-prefixed framework mutators (`_add_order`, `_transition_order`, ...),
`Order.client_id`/`venue_id` (no `time`), `Deal(id=...)`, `AssetBalance`, and
read methods that hand out read-only `MappingProxyType` views.
"""

from typing import TypeVar

import numpy as np
import pytest

from qubx.core.account_manager.state import AccountState
from qubx.core.basics import (
    AssetBalance,
    Deal,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.lookups import lookup

T0 = np.datetime64("2026-05-28T00:00:00", "ns")
T1 = np.datetime64("2026-05-28T00:01:00", "ns")
T2 = np.datetime64("2026-05-28T00:02:00", "ns")

_T = TypeVar("_T")


def _present(value: _T | None) -> _T:
    """Narrow an Optional lookup result: fail loudly on None, return the value."""
    assert value is not None
    return value


def _order(
    cid: str = "qubx-1",
    status: OrderStatus = OrderStatus.SUBMITTED,
    venue_id: str | None = None,
    last_updated_at: np.datetime64 | None = None,
) -> Order:
    return Order(
        client_id=cid,
        type=OrderType.LIMIT,
        instrument=None,  # type: ignore[arg-type]  # indexing never reads order.instrument
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_id=venue_id,
        price=50_000.0,
        last_updated_at=last_updated_at,
        origin=OrderOrigin.FRAMEWORK,
    )


def _fill(trade_id: str = "t1", amount: float = 0.5, price: float = 50_000.0, order_id: str = "v1") -> Deal:
    return Deal(id=trade_id, order_id=order_id, time=T0, amount=amount, price=price, aggressive=True)


def _instrument(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol("BINANCE.UM", symbol)
    assert inst is not None, f"fixture instrument {symbol} not found in lookup"
    return inst


# --------------------------------------------------------------------------- #
# add_order / indexing
# --------------------------------------------------------------------------- #


def test_add_order_inserts_and_indexes_inflight():
    state = AccountState("binance")
    state._add_order(_order())
    assert state.get_order("qubx-1") is not None
    assert "qubx-1" in state._inflight_index


def test_add_order_with_venue_id_indexes_by_venue():
    state = AccountState("binance")
    state._add_order(_order(venue_id="VENUE_ABC"))
    found = state.get_order_by_venue_id("VENUE_ABC")
    assert found is not None and found.client_id == "qubx-1"


def test_add_terminal_order_populates_evict_index_not_inflight():
    state = AccountState("binance")
    state._add_order(_order(status=OrderStatus.FILLED, last_updated_at=T1))
    assert "qubx-1" not in state._inflight_index
    assert state._pending_evict_index["qubx-1"] == T1


def test_add_terminal_order_without_last_updated_at_raises():
    state = AccountState("binance")
    with pytest.raises(ValueError, match="last_updated_at"):
        state._add_order(_order(status=OrderStatus.CANCELED, last_updated_at=None))


# --------------------------------------------------------------------------- #
# venue id
# --------------------------------------------------------------------------- #


def test_set_venue_id_indexes_and_updates_order():
    state = AccountState("binance")
    state._add_order(_order())
    state._set_venue_id("qubx-1", "VENUE_ABC")
    assert _present(state.get_order_by_venue_id("VENUE_ABC")).client_id == "qubx-1"
    assert _present(state.get_order("qubx-1")).venue_id == "VENUE_ABC"


def test_set_venue_id_drops_stale_key_on_repoint():
    state = AccountState("binance")
    state._add_order(_order(venue_id="OLD"))
    state._set_venue_id("qubx-1", "NEW")
    assert state.get_order_by_venue_id("OLD") is None
    assert _present(state.get_order_by_venue_id("NEW")).client_id == "qubx-1"


def test_get_order_by_unknown_venue_id_returns_none():
    state = AccountState("binance")
    assert state.get_order_by_venue_id("NOPE") is None


# --------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------- #


def test_transition_to_accepted_drains_inflight():
    state = AccountState("binance")
    state._add_order(_order())
    order = state._transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    assert order.status is OrderStatus.ACCEPTED
    assert order.last_updated_at == T1
    assert "qubx-1" not in state._inflight_index


def test_transition_to_terminal_populates_evict_index():
    state = AccountState("binance")
    state._add_order(_order())
    state._transition_order("qubx-1", OrderStatus.FILLED, T1)
    assert "qubx-1" not in state._inflight_index
    assert state._pending_evict_index["qubx-1"] == T1


def test_transition_back_to_pending_re_indexes_inflight():
    state = AccountState("binance")
    state._add_order(_order())
    state._transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    assert "qubx-1" not in state._inflight_index
    state._transition_order("qubx-1", OrderStatus.PENDING_CANCEL, T2)
    assert "qubx-1" in state._inflight_index


# --------------------------------------------------------------------------- #
# fills
# --------------------------------------------------------------------------- #


def test_apply_fill_accumulates_quantity_and_avg_price():
    state = AccountState("binance")
    state._add_order(_order())
    state._apply_fill("qubx-1", _fill("t1", amount=0.5, price=50_000.0), T1)
    state._apply_fill("qubx-1", _fill("t2", amount=0.5, price=51_000.0), T2)
    order = _present(state.get_order("qubx-1"))
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price == 50_500.0


def test_apply_fill_dedup_by_trade_id():
    state = AccountState("binance")
    state._add_order(_order())
    fill = _fill("t1", amount=0.5)
    state._apply_fill("qubx-1", fill, T1)
    state._apply_fill("qubx-1", fill, T1)  # same trade id -> ignored
    assert _present(state.get_order("qubx-1")).filled_quantity == 0.5


def test_apply_fill_uses_magnitude_for_sell():
    # filled_quantity is unsigned magnitude (direction lives in order.side), matching
    # Order.quantity / the OME's positive-amount requirement. Deal.amount is signed,
    # so a sell fill accumulates abs(amount). Keeps `filled_quantity >= quantity`
    # well-defined for both sides.
    state = AccountState("binance")
    state._add_order(_order())
    state._apply_fill("qubx-1", _fill("t1", amount=-0.5, price=100.0), T1)
    order = _present(state.get_order("qubx-1"))
    assert order.filled_quantity == 0.5
    assert order.avg_fill_price == 100.0


def test_apply_fill_accumulates_magnitude_across_sell_fills():
    # Guards the _recompute_avg / _apply_fill coupling: two sell fills must not
    # cancel to new_qty == 0 (the bug if filled_quantity were unsigned but the
    # avg used signed amounts).
    state = AccountState("binance")
    state._add_order(_order())
    state._apply_fill("qubx-1", _fill("t1", amount=-0.5, price=100.0), T1)
    state._apply_fill("qubx-1", _fill("t2", amount=-0.5, price=102.0), T2)
    order = _present(state.get_order("qubx-1"))
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price == 101.0


# --------------------------------------------------------------------------- #
# eviction / terminal history
# --------------------------------------------------------------------------- #


def test_remove_order_drains_indexes_and_moves_to_history():
    state = AccountState("binance")
    state._add_order(_order())
    state._set_venue_id("qubx-1", "VENUE_ABC")
    state._apply_fill("qubx-1", _fill("t1"), T1)
    state._transition_order("qubx-1", OrderStatus.FILLED, T1)
    state._remove_order("qubx-1")

    assert "qubx-1" not in state._active_orders
    assert state.get_order_by_venue_id("VENUE_ABC") is None
    assert "qubx-1" not in state._inflight_index
    assert "qubx-1" not in state._pending_evict_index
    assert "qubx-1" not in state._seen_trade_ids
    # still resolvable via the terminal-history ring buffer
    assert state.get_order("qubx-1") is not None


def test_remove_unknown_order_is_noop():
    state = AccountState("binance")
    state._remove_order("does-not-exist")  # must not raise


def test_get_order_falls_back_to_terminal_history():
    state = AccountState("binance")
    state._add_order(_order())
    state._transition_order("qubx-1", OrderStatus.FILLED, T1)
    state._remove_order("qubx-1")
    order = state.get_order("qubx-1")
    assert order is not None
    assert order.status is OrderStatus.FILLED


# --------------------------------------------------------------------------- #
# read views are read-only
# --------------------------------------------------------------------------- #


def test_get_orders_returns_readonly_view():
    state = AccountState("binance")
    state._add_order(_order())
    orders = state.get_orders()
    assert orders["qubx-1"].client_id == "qubx-1"
    with pytest.raises(TypeError):
        orders["x"] = _order("x")  # type: ignore[index]  # MappingProxyType is read-only by design


# --------------------------------------------------------------------------- #
# balances / positions (identity-preserving updates)
# --------------------------------------------------------------------------- #


def test_update_balance_inserts_then_preserves_identity():
    state = AccountState("binance")
    bal = AssetBalance(exchange="binance", currency="USDT", free=100.0, locked=0.0, total=100.0)
    state._update_balance("USDT", bal)
    assert state.get_balance("USDT") is bal  # stored by reference on first insert
    assert state.get_balance("BTC") is None

    newer = AssetBalance(exchange="binance", currency="USDT", free=50.0, locked=0.0, total=50.0)
    state._update_balance("USDT", newer)
    # identity preserved: set() mutates the existing object in place
    assert state.get_balance("USDT") is bal
    assert _present(state.get_balance("USDT")).free == 50.0
    assert _present(state.get_balance("USDT")).total == 50.0


def test_update_position_inserts_then_resets_existing():
    state = AccountState("binance")
    inst = _instrument("BTCUSDT")
    pos = Position(inst)
    state._update_position(pos)
    assert state.get_position(inst) is pos

    newer = Position(inst, quantity=1.0, pos_average_price=50_000.0)
    state._update_position(newer)
    # identity preserved: reset_by_position mutates the existing object in place
    assert state.get_position(inst) is pos
    assert _present(state.get_position(inst)).quantity == 1.0
