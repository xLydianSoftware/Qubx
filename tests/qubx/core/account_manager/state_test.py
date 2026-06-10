"""Unit tests for AccountState — the pure per-exchange data + index store.

Ported from PR #302's state_test.py and adapted to this branch's API:
unprefixed framework mutators (`add_order`, `transition_order`, ...),
`Order.client_order_id`/`venue_order_id`, `Deal(trade_id=...)`, `Balance`,
and the remove_order/evict_to_history split (PR's `_remove_order` retains the
order in history — that is our `evict_to_history`; our `remove_order` is the
full drop for never-reached-the-venue submits).
"""

from typing import TypeVar

import numpy as np
import pytest

from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.basics import (
    Balance,
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
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=None,  # type: ignore[arg-type]  # indexing never reads order.instrument
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_order_id=venue_id,
        price=50_000.0,
        last_updated_at=last_updated_at,
        origin=OrderOrigin.FRAMEWORK,
    )


def _fill(trade_id: str = "t1", amount: float = 0.5, price: float = 50_000.0, order_id: str = "v1") -> Deal:
    return Deal(trade_id=trade_id, order_id=order_id, time=T0, amount=amount, price=price, aggressive=True)


def _instrument(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol("BINANCE.UM", symbol)
    assert inst is not None, f"fixture instrument {symbol} not found in lookup"
    return inst


def test_base_currency_stored_uppercased():
    assert AccountState("binance", "usdt").base_currency == "USDT"


# --------------------------------------------------------------------------- #
# add_order / indexing
# --------------------------------------------------------------------------- #


def test_add_order_inserts_and_indexes_inflight():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    assert state.get_order("qubx-1") is not None
    assert "qubx-1" in state._inflight_index


def test_add_order_with_venue_id_indexes_by_venue():
    state = AccountState("binance", "USDT")
    state.add_order(_order(venue_id="VENUE_ABC"))
    found = state.get_order_by_venue_id("VENUE_ABC")
    assert found is not None and found.client_order_id == "qubx-1"


def test_add_terminal_order_populates_evict_index_not_inflight():
    state = AccountState("binance", "USDT")
    state.add_order(_order(status=OrderStatus.FILLED, last_updated_at=T1))
    assert "qubx-1" not in state._inflight_index
    assert state._pending_evict_index["qubx-1"] == T1


def test_add_terminal_order_without_last_updated_at_raises():
    state = AccountState("binance", "USDT")
    with pytest.raises(ValueError, match="last_updated_at"):
        state.add_order(_order(status=OrderStatus.CANCELED, last_updated_at=None))


def test_add_order_refuses_duplicate_active_cid():
    # F11: silent overwrite orphaned the caller's Order reference and left stale
    # index entries — a duplicate active cid is a framework bug and must fail loudly.
    state = AccountState("binance", "USDT")
    first = _order()
    state.add_order(first)
    with pytest.raises(ValueError, match="duplicate"):
        state.add_order(_order())
    assert state.get_order("qubx-1") is first


# --------------------------------------------------------------------------- #
# venue id
# --------------------------------------------------------------------------- #


def test_set_venue_id_indexes_and_updates_order():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.set_venue_id("qubx-1", "VENUE_ABC")
    assert _present(state.get_order_by_venue_id("VENUE_ABC")).client_order_id == "qubx-1"
    assert _present(state.get_order("qubx-1")).venue_order_id == "VENUE_ABC"


def test_set_venue_id_drops_stale_key_on_repoint():
    state = AccountState("binance", "USDT")
    state.add_order(_order(venue_id="OLD"))
    state.set_venue_id("qubx-1", "NEW")
    assert state.get_order_by_venue_id("OLD") is None
    assert _present(state.get_order_by_venue_id("NEW")).client_order_id == "qubx-1"


def test_get_order_by_unknown_venue_id_returns_none():
    state = AccountState("binance", "USDT")
    assert state.get_order_by_venue_id("NOPE") is None


# --------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------- #


def test_transition_to_accepted_drains_inflight():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    order = state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    assert order.status is OrderStatus.ACCEPTED
    assert order.last_updated_at == T1
    assert "qubx-1" not in state._inflight_index


def test_transition_to_terminal_populates_evict_index():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.FILLED, T1)
    assert "qubx-1" not in state._inflight_index
    assert state._pending_evict_index["qubx-1"] == T1


def test_transition_back_to_pending_re_indexes_inflight():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    assert "qubx-1" not in state._inflight_index
    state.transition_order("qubx-1", OrderStatus.PENDING_CANCEL, T2)
    assert "qubx-1" in state._inflight_index


# --------------------------------------------------------------------------- #
# fills
# --------------------------------------------------------------------------- #


def test_apply_fill_accumulates_quantity_and_avg_price():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.apply_fill("qubx-1", _fill("t1", amount=0.5, price=50_000.0), T1)
    state.apply_fill("qubx-1", _fill("t2", amount=0.5, price=51_000.0), T2)
    order = _present(state.get_order("qubx-1"))
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price == 50_500.0


def test_apply_fill_dedup_by_trade_id():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    fill = _fill("t1", amount=0.5)
    assert state.apply_fill("qubx-1", fill, T1) is True
    assert state.apply_fill("qubx-1", fill, T1) is False  # same trade id -> ignored
    assert _present(state.get_order("qubx-1")).filled_quantity == 0.5


def test_apply_fill_uses_magnitude_for_sell():
    # filled_quantity is unsigned magnitude (direction lives in order.side), matching
    # Order.quantity / the OME's positive-amount requirement. Deal.amount is signed,
    # so a sell fill accumulates abs(amount). Keeps `filled_quantity >= quantity`
    # well-defined for both sides.
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.apply_fill("qubx-1", _fill("t1", amount=-0.5, price=100.0), T1)
    order = _present(state.get_order("qubx-1"))
    assert order.filled_quantity == 0.5
    assert order.avg_fill_price == 100.0


def test_apply_fill_accumulates_magnitude_across_sell_fills():
    # Guards the avg-price / apply_fill coupling: two sell fills must not
    # cancel to new_qty == 0 (the bug if filled_quantity were unsigned but the
    # avg used signed amounts).
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.apply_fill("qubx-1", _fill("t1", amount=-0.5, price=100.0), T1)
    state.apply_fill("qubx-1", _fill("t2", amount=-0.5, price=102.0), T2)
    order = _present(state.get_order("qubx-1"))
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price == 101.0


# --------------------------------------------------------------------------- #
# eviction / terminal history
# --------------------------------------------------------------------------- #


def test_remove_order_drains_indexes_and_moves_to_history():
    # diverges from PR #302: PR's `_remove_order` retains the order in the history
    # ring buffer — ours splits that into evict_to_history (this behavior) and
    # remove_order (full drop). This pins the history-retaining path.
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.set_venue_id("qubx-1", "VENUE_ABC")
    state.apply_fill("qubx-1", _fill("t1"), T1)
    state.transition_order("qubx-1", OrderStatus.FILLED, T1)
    state.evict_to_history("qubx-1")

    assert "qubx-1" not in state._active_orders
    assert state.get_order_by_venue_id("VENUE_ABC") is None
    assert "qubx-1" not in state._inflight_index
    assert "qubx-1" not in state._pending_evict_index
    assert "qubx-1" not in state._seen_trade_ids
    # still resolvable via the terminal-history ring buffer
    assert state.get_order("qubx-1") is not None


def test_remove_unknown_order_is_noop():
    state = AccountState("binance", "USDT")
    state.evict_to_history("does-not-exist")  # must not raise
    state.remove_order("does-not-exist")  # must not raise


def test_get_order_falls_back_to_terminal_history():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.FILLED, T1)
    state.evict_to_history("qubx-1")
    order = state.get_order("qubx-1")
    assert order is not None
    assert order.status is OrderStatus.FILLED


# --------------------------------------------------------------------------- #
# read views don't leak internal state
# --------------------------------------------------------------------------- #


def test_get_orders_returns_readonly_view():
    # diverges from PR #302: get_orders hands out a defensive dict copy rather than a
    # MappingProxyType — mutating the returned mapping must not affect internal state.
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    orders = state.get_orders()
    assert orders["qubx-1"].client_order_id == "qubx-1"
    orders.clear()
    assert state.get_order("qubx-1") is not None


# --------------------------------------------------------------------------- #
# balances / positions (identity-preserving updates)
# --------------------------------------------------------------------------- #


def test_update_balance_inserts_then_preserves_identity():
    state = AccountState("binance", "USDT")
    bal = Balance(exchange="binance", currency="USDT", free=100.0, locked=0.0, total=100.0)
    state.update_balance("USDT", bal)
    assert state.get_balance("USDT") is bal  # stored by reference on first insert
    assert state.get_balance("BTC") is None

    newer = Balance(exchange="binance", currency="USDT", free=50.0, locked=0.0, total=50.0)
    state.update_balance("USDT", newer)
    # identity preserved: update mutates the existing object in place
    assert state.get_balance("USDT") is bal
    assert _present(state.get_balance("USDT")).free == 50.0
    assert _present(state.get_balance("USDT")).total == 50.0


def test_update_position_inserts_then_resets_existing():
    state = AccountState("binance", "USDT")
    inst = _instrument("BTCUSDT")
    pos = Position(inst)
    state.set_position(inst, pos)
    assert state.get_position(inst) is pos

    newer = Position(inst, quantity=1.0, pos_average_price=50_000.0)
    state.set_position(inst, newer)
    # identity preserved: reset_by_position mutates the existing object in place
    assert state.get_position(inst) is pos
    assert _present(state.get_position(inst)).quantity == 1.0


# --------------------------------------------------------------------------- #
# side-tables: pre_pending_status / retry_count
# --------------------------------------------------------------------------- #


def test_pre_pending_captures_status_on_entry():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    assert state.get_pre_pending("qubx-1") is None
    state.transition_order("qubx-1", OrderStatus.PENDING_CANCEL, T2)
    assert state.get_pre_pending("qubx-1") == OrderStatus.ACCEPTED


def test_pre_pending_preserved_across_pending_to_pending():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    state.transition_order("qubx-1", OrderStatus.PENDING_UPDATE, T2)
    state.transition_order("qubx-1", OrderStatus.PENDING_CANCEL, T2)
    assert state.get_pre_pending("qubx-1") == OrderStatus.ACCEPTED


def test_pre_pending_cleared_on_leaving_pending():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    state.transition_order("qubx-1", OrderStatus.PENDING_CANCEL, T2)
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T2)
    assert state.get_pre_pending("qubx-1") is None


def test_retry_count_bump_and_default():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    assert state.get_retry("qubx-1") == 0
    assert state.bump_retry("qubx-1") == 1
    assert state.bump_retry("qubx-1") == 2
    assert state.get_retry("qubx-1") == 2


def test_retry_count_resets_on_transition():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.bump_retry("qubx-1")
    state.bump_retry("qubx-1")
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    assert state.get_retry("qubx-1") == 0


def test_remove_order_drops_side_tables():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    state.transition_order("qubx-1", OrderStatus.ACCEPTED, T1)
    state.transition_order("qubx-1", OrderStatus.PENDING_CANCEL, T2)
    state.bump_retry("qubx-1")
    assert state.get_pre_pending("qubx-1") == OrderStatus.ACCEPTED
    assert state.get_retry("qubx-1") == 1
    state.evict_to_history("qubx-1")
    assert state.get_pre_pending("qubx-1") is None
    assert state.get_retry("qubx-1") == 0


# --------------------------------------------------------------------------- #
# active-order lookups
# --------------------------------------------------------------------------- #


def test_get_active_order_excludes_terminal_history():
    state = AccountState("binance", "USDT")
    state.add_order(_order())
    assert _present(state.get_active_order("qubx-1")).client_order_id == "qubx-1"
    state.transition_order("qubx-1", OrderStatus.FILLED, T1)
    state.evict_to_history("qubx-1")
    assert state.get_active_order("qubx-1") is None  # evicted
    assert state.get_order("qubx-1") is not None  # but still in history


def test_get_active_order_unknown_returns_none():
    state = AccountState("binance", "USDT")
    assert state.get_active_order("nope") is None


def test_get_inflight_orders_tracks_index():
    state = AccountState("binance", "USDT")
    state.add_order(_order("a"))  # SUBMITTED -> inflight
    state.add_order(_order("b"))  # SUBMITTED -> inflight
    state.transition_order("b", OrderStatus.ACCEPTED, T1)  # leaves inflight
    inflight = {o.client_order_id for o in state.get_inflight_orders()}
    assert inflight == {"a"}
    state.transition_order("b", OrderStatus.PENDING_CANCEL, T2)  # re-enters inflight
    assert {o.client_order_id for o in state.get_inflight_orders()} == {"a", "b"}


def test_get_inflight_orders_empty_when_none():
    state = AccountState("binance", "USDT")
    state.add_order(_order(status=OrderStatus.FILLED, last_updated_at=T1))
    assert state.get_inflight_orders() == []


# --------------------------------------------------------------------------- #
# snapshot ratchet / terminal pruning
# --------------------------------------------------------------------------- #


def test_snapshot_as_of_ratchet():
    state = AccountState("binance", "USDT")
    assert state.get_last_snapshot_as_of() is None
    state.mark_snapshot_applied(T1)
    assert state.get_last_snapshot_as_of() == T1
    state.mark_snapshot_applied(T2)
    assert state.get_last_snapshot_as_of() == T2


def test_prune_terminal_orders_removes_only_expired():
    state = AccountState("binance", "USDT")
    state.add_order(_order("old", status=OrderStatus.FILLED, last_updated_at=T0))
    state.add_order(_order("recent", status=OrderStatus.FILLED, last_updated_at=T2))
    state.prune_terminal_orders(T2, np.timedelta64(90, "s"))
    assert state.get_active_order("old") is None
    assert state.get_order("old") is not None  # moved to history
    assert state.get_active_order("recent") is not None


def test_prune_terminal_orders_ignores_non_terminal():
    state = AccountState("binance", "USDT")
    state.add_order(_order("live"))  # SUBMITTED, never in the evict index
    state.prune_terminal_orders(T2, np.timedelta64(0, "s"))
    assert state.get_active_order("live") is not None


def test_venue_figures_roundtrip():
    state = AccountState("binance", "USDT")
    assert state.get_venue_figures() is None
    figures = VenueAccountFigures(as_of=T1, equity=12_345.0, margin_ratio=42.0)
    state.set_venue_figures(figures)
    assert state.get_venue_figures() is figures
    assert _present(state.get_venue_figures()).equity == 12_345.0
