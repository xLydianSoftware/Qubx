"""Scenario tests for the Reconciler (stage 2) — situation I: ResolveMissingOrder.

The Reconciler is pure: entry points return list[Action] and mutate the in-memory
AccountState they are handed. So a live-trading scenario is a plain data test — build
state + a Reconciler, drive on_snapshot / on_tick / on_event with explicit `now`, and
assert on (state, returned actions, live task keys). No connector mocks.

See docs/account-management/reconciliation-redesign.md (+ .canvas).

These tests define the intended API (names are proposed here, adjust at implementation).
"""

import numpy as np

from qubx import QubxLogConfig
from qubx.core.account_manager.diffs import Differ
from qubx.core.account_manager.reconciler import (
    Reconciler,
    RequestStatus,
)
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import Order, OrderOrigin, OrderSide, OrderStatus, OrderType
from qubx.core.events import AccountSnapshot, OrderCanceledEvent
from qubx.core.lookups import lookup

EXCHANGE = "BINANCE.UM"

T0 = np.datetime64("2026-05-28T00:00:30", "ns")  # snapshot as_of / spawn time
SETTLED = np.datetime64("2026-05-28T00:00:00", "ns")  # order last change — past grace


def _inst(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol(EXCHANGE, symbol)
    assert inst is not None
    return inst


_GEN = object()  # sentinel: generate a unique venue id from the cid


def _order(cid: str, *, venue_id=_GEN, status: OrderStatus = OrderStatus.ACCEPTED) -> Order:
    # default: a unique exchange order id derived from the cid (so two distinct orders never
    # collide). Pass an explicit value to override, or `venue_id=None` for an unacked order.
    vid = f"v_{cid}" if venue_id is _GEN else venue_id
    return Order(
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=_inst(),
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_order_id=vid,  # type: ignore
        price=50_000.0,
        submitted_at=SETTLED,
        last_updated_at=SETTLED,
        origin=OrderOrigin.FRAMEWORK,
    )


def _local(*orders: Order) -> AccountState:
    st = AccountState(EXCHANGE, "USDT")
    for o in orders:
        st.add_order(o)
    return st


def _origin(*, as_of=T0, open_orders=None) -> AccountSnapshot:
    return AccountSnapshot(exchange=EXCHANGE, as_of=as_of, open_orders=open_orders)


def _reconciler() -> Reconciler:
    # short windows so scenarios are a few ticks; differ grace 5s
    return Reconciler(Differ(grace="5s"), snapshot_interval="30s", missing_wait="2s", missing_max_retries=2)


def _wait_seconds(base, s: int):
    return base + np.timedelta64(s, "s")


# --------------------------------------------------------------------------- #
# I. ResolveMissingOrder
# --------------------------------------------------------------------------- #


def test_missing_order_spawns_task_without_terminalizing():
    st = _local(_order("X1"))
    rec = _reconciler()
    actions = rec.on_snapshot(st, _origin(open_orders=[]), T0)  # order absent from snapshot
    assert rec.active_keys() == {"X1"}  # a ResolveMissingOrder task now owns the cid
    assert actions == []  # nothing fetched yet — we WAIT first
    assert st.get_order("X1").status == OrderStatus.ACCEPTED  # NOT blind-terminalized


def test_missing_order_fetches_status_after_wait_window():
    rec = _reconciler()

    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)
    assert rec.on_tick(st, _wait_seconds(T0, 1)) == []  # still inside wait_before_fetch (2s)

    actions = rec.on_tick(st, _wait_seconds(T0, 3))  # past the wait window
    assert actions == [RequestStatus(cid="X1", venue_id="v_X1", instrument=_inst())]


def test_missing_order_resolved_by_arriving_event_drops_task():
    rec = _reconciler()

    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)

    # the venue's real answer arrives as a normal order event; the task sees it terminal
    rec.on_event(
        st, OrderCanceledEvent(instrument=_inst(), client_order_id="X1", venue_order_id="v_X1"), _wait_seconds(T0, 4)
    )
    assert rec.active_keys() == set()  # task done & dropped — no LOST


def test_unacked_missing_order_detected_by_cid_fallback_ends_lost():
    # X1 never got its venue ack (venue_order_id=None) and is absent from the snapshot —
    # the Differ can't match by venue id, falls back to cid, still flags it missing.
    rec = _reconciler()
    st = _local(_order("X1", venue_id=None))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)
    assert rec.active_keys() == {"X1"}
    rec.on_tick(st, _wait_seconds(T0, 3))  # retry 1
    rec.on_tick(st, _wait_seconds(T0, 5))  # retry 2
    rec.on_tick(st, _wait_seconds(T0, 7))  # give up
    assert st.get_order("X1").status == OrderStatus.LOST  # type: ignore


def test_missing_order_with_no_answer_ends_lost():
    rec = _reconciler()

    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)

    # drive ticks past the wait window + exhaust the 2 retries with no venue answer
    rec.on_tick(st, _wait_seconds(T0, 3))  # retry 1 -> RequestStatus
    rec.on_tick(st, _wait_seconds(T0, 5))  # retry 2 -> RequestStatus
    rec.on_tick(st, _wait_seconds(T0, 7))  # exhausted -> give up
    assert st.get_order("X1").status == OrderStatus.LOST  # type: ignore # new terminal state
    assert rec.active_keys() == set()


def test_repeated_snapshot_keeps_one_task_per_missing_order():
    rec = _reconciler()
    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)
    rec.on_snapshot(st, _origin(as_of=_wait_seconds(T0, 1), open_orders=[]), _wait_seconds(T0, 1))  # still missing
    assert rec.active_keys() == {"X1"}  # one task per key — not duplicated


def test_missing_one_order_with_no_answer_ends_lost():
    QubxLogConfig.set_log_level("DEBUG")

    rec = _reconciler()

    # same venue; each order gets its own unique exchange id (v_X1, v_X2 by default).
    # X1 is local-only (missing from the snapshot); X2 is present in both.
    st = _local(_order("X1"), _order("X2"))
    rec.on_snapshot(st, _origin(open_orders=[_order("X2")]), T0)

    # drive ticks past the wait window + exhaust the 2 retries with no venue answer
    rec.on_tick(st, _wait_seconds(T0, 3))  # retry 1 -> RequestStatus
    rec.on_tick(st, _wait_seconds(T0, 5))  # retry 2 -> RequestStatus
    rec.on_tick(st, _wait_seconds(T0, 7))  # exhausted -> give up

    assert st.get_order("X1").status == OrderStatus.LOST  # type: ignore # new terminal state
    assert st.get_order("X2").status == OrderStatus.ACCEPTED  # type: ignore
    assert rec.active_keys() == set()
