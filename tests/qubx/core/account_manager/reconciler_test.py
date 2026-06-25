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
    RouteEvent,
)
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import Order, OrderOrigin, OrderSide, OrderStatus, OrderType
from qubx.core.events import AccountSnapshot, OrderCanceledEvent, OrderLostEvent, OrderPartiallyFilledEvent
from qubx.core.lookups import lookup

EXCHANGE = "BINANCE.UM"

T0 = np.datetime64("2026-05-28T00:00:30", "ns")  # snapshot as_of / spawn time
SETTLED = np.datetime64("2026-05-28T00:00:00", "ns")  # order last change — past grace


def _inst(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol(EXCHANGE, symbol)
    assert inst is not None
    return inst


_GEN = object()  # sentinel: generate a unique venue id from the cid


def _order(
    cid: str,
    *,
    venue_id=_GEN,
    status: OrderStatus = OrderStatus.ACCEPTED,
    last_updated_at: np.datetime64 = SETTLED,
    price=50_000.0,
    quantity=1.0,
    filled=0.0,
) -> Order:
    # default: a unique exchange order id derived from the cid (so two distinct orders never
    # collide). Pass an explicit value to override, or `venue_id=None` for an unacked order.
    # `last_updated_at` is the order's venue update ts (a later snapshot leg carries a newer one).
    vid = f"v_{cid}" if venue_id is _GEN else venue_id
    return Order(
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=_inst(),
        quantity=quantity,
        filled_quantity=filled,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_order_id=vid,  # type: ignore
        price=price,
        submitted_at=SETTLED,
        last_updated_at=last_updated_at,
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


def _passed_seconds(base, s: int):
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
    assert rec.on_tick(st, _passed_seconds(T0, 1)) == []  # still inside wait_before_fetch (2s)

    actions = rec.on_tick(st, _passed_seconds(T0, 3))  # past the wait window
    assert actions == [RequestStatus(cid="X1", venue_id="v_X1", instrument=_inst())]


def test_missing_order_resolved_by_arriving_event_drops_task():
    rec = _reconciler()

    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)

    # the venue's real answer arrives as a normal order event; the task sees it terminal
    rec.on_event(
        st, OrderCanceledEvent(instrument=_inst(), client_order_id="X1", venue_order_id="v_X1"), _passed_seconds(T0, 4)
    )
    assert rec.active_keys() == set()  # task done & dropped — no LOST


def test_unacked_missing_order_detected_by_cid_fallback_ends_lost():
    # X1 never got its venue ack (venue_order_id=None) and is absent from the snapshot —
    # the Differ can't match by venue id, falls back to cid, still flags it missing.
    rec = _reconciler()
    st = _local(_order("X1", venue_id=None))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)
    assert rec.active_keys() == {"X1"}
    rec.on_tick(st, _passed_seconds(T0, 3))  # retry 1
    rec.on_tick(st, _passed_seconds(T0, 5))  # retry 2
    out = rec.on_tick(st, _passed_seconds(T0, 7))  # give up -> routes a LOST event (no silent mutate)
    assert [type(a) for a in out] == [RouteEvent]
    assert isinstance(out[0].event, OrderLostEvent) and out[0].event.client_order_id == "X1"
    assert rec.active_keys() == set()


def test_missing_order_with_no_answer_ends_lost():
    rec = _reconciler()

    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)

    # drive ticks past the wait window + exhaust the 2 retries with no venue answer
    rec.on_tick(st, _passed_seconds(T0, 3))  # retry 1 -> RequestStatus
    rec.on_tick(st, _passed_seconds(T0, 5))  # retry 2 -> RequestStatus
    out = rec.on_tick(st, _passed_seconds(T0, 7))  # exhausted -> routes a LOST event
    assert [type(a) for a in out] == [RouteEvent]
    assert isinstance(out[0].event, OrderLostEvent) and out[0].event.client_order_id == "X1"
    assert rec.active_keys() == set()


def test_repeated_snapshot_keeps_one_task_per_missing_order():
    rec = _reconciler()
    st = _local(_order("X1"))
    rec.on_snapshot(st, _origin(open_orders=[]), T0)
    rec.on_snapshot(st, _origin(as_of=_passed_seconds(T0, 1), open_orders=[]), _passed_seconds(T0, 1))  # still missing
    assert rec.active_keys() == {"X1"}  # one task per key — not duplicated


def test_missing_one_order_with_no_answer_ends_lost():
    QubxLogConfig.set_log_level("DEBUG")

    rec = _reconciler()

    # same venue; each order gets its own unique exchange id (v_X1, v_X2 by default).
    # X1 is local-only (missing from the snapshot); X2 is present in both.
    st = _local(_order("X1"), _order("X2"))
    rec.on_snapshot(st, _origin(open_orders=[_order("X2")]), T0)

    # drive ticks past the wait window + exhaust the 2 retries with no venue answer
    rec.on_tick(st, _passed_seconds(T0, 3))  # retry 1 -> RequestStatus
    rec.on_tick(st, _passed_seconds(T0, 5))  # retry 2 -> RequestStatus
    out = rec.on_tick(st, _passed_seconds(T0, 7))  # exhausted -> routes a LOST event for X1

    assert [type(a) for a in out] == [RouteEvent]
    assert isinstance(out[0].event, OrderLostEvent) and out[0].event.client_order_id == "X1"
    assert st.get_order("X2").status == OrderStatus.ACCEPTED  # type: ignore # untouched
    assert rec.active_keys() == set()
    QubxLogConfig.set_log_level("ERROR")


# NOTE: a terminal order (FILLED/CANCELED/...) never appears in a venue's open_orders, so a
# "snapshot reports it terminal" case is not real — that status arrives via the RequestStatus
# fetch reply (the connector replays the real event on the bus). Hence no reappear-FILLED test.


def test_missing_order_reappears_open_in_snapshot_resolves_without_lost():
    QubxLogConfig.set_log_level("DEBUG")
    # the dangerous one: order missing in snap1 (a race), still OPEN in snap2 — no diff,
    # but the task must resolve on reappearance instead of grinding to LOST.
    rec = _reconciler()
    st = _local(_order("order1", status=OrderStatus.ACCEPTED))

    rec.on_snapshot(st, _origin(open_orders=[]), T0)  # missing -> task
    rec.on_tick(st, _passed_seconds(T0, 3))  # retry 1
    rec.on_snapshot(
        st,
        _origin(as_of=_passed_seconds(T0, 5), open_orders=[_order("order1", status=OrderStatus.ACCEPTED)]),
        _passed_seconds(T0, 5),
    )
    assert rec.active_keys() == set()  # resolved by reappearance

    rec.on_tick(st, _passed_seconds(T0, 8))  # drive well past give-up
    rec.on_tick(st, _passed_seconds(T0, 10))
    assert st.get_order("order1").status == OrderStatus.ACCEPTED  # type: ignore # still live, never LOST
    QubxLogConfig.set_log_level("ERROR")


def test_status_resolved():
    # A present order the snapshot shows more-filled than local (we missed the WS fill).
    # status/filled are reconciled in-mem from the snapshot (venue-ts guarded), AND a fill
    # event is routed so the strategy is notified (the WS update may never arrive).
    rec = _reconciler()
    st = _local(_order("order1", status=OrderStatus.ACCEPTED))

    a = rec.on_snapshot(
        st,
        _origin(
            as_of=_passed_seconds(T0, 5),
            open_orders=[
                _order(
                    "order1",
                    status=OrderStatus.PARTIALLY_FILLED,
                    filled=0.5,
                    last_updated_at=_passed_seconds(T0, 4),  # newer venue ts than local
                )
            ],
        ),
        _passed_seconds(T0, 5),
    )

    o = st.get_order("order1")
    assert o.status == OrderStatus.PARTIALLY_FILLED  # type: ignore # reconciled in-mem
    assert o.filled_quantity == 0.5  # type: ignore
    assert [type(x) for x in a] == [RouteEvent]  # and a fill event sent for notification
    assert isinstance(a[0].event, OrderPartiallyFilledEvent) and a[0].event.client_order_id == "order1"


def test_snapshot_does_not_wipe_pending_cancel_marker():
    QubxLogConfig.set_log_level("DEBUG")
    # our cancel is in flight (PENDING_CANCEL). A snapshot poll still showing the order live
    # (non-terminal) must NOT wipe the pending marker — the venue resolves the race itself.
    rec = _reconciler()
    st = _local(_order("order1", status=OrderStatus.PENDING_CANCEL))
    a = rec.on_snapshot(
        st,
        _origin(
            as_of=_passed_seconds(T0, 5),
            open_orders=[_order("order1", status=OrderStatus.ACCEPTED, last_updated_at=_passed_seconds(T0, 4))],
        ),
        _passed_seconds(T0, 5),
    )
    assert a == []  # nothing reconciled / routed
    assert st.get_order("order1").status == OrderStatus.PENDING_CANCEL  # type: ignore # marker preserved
    QubxLogConfig.set_log_level("ERROR")
