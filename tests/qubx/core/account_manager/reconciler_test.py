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
from qubx.core.account_manager import reducer
from qubx.core.account_manager.diffs import Differ
from qubx.core.account_manager.reconciler import (
    HIST_DEALS_LOOKBACK,
    Reconciler,
    RequestFundingPayments,
    RequestHistDeals,
    RequestSnapshot,
    RequestStatus,
    RouteEvent,
)
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import Balance, Deal, Instrument, Order, OrderOrigin, OrderSide, OrderStatus, OrderType, Position
from qubx.core.events import (
    AccountSnapshot,
    DealEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderLostEvent,
    OrderPartiallyFilledEvent,
)
from qubx.core.lookups import lookup

EXCHANGE = "BINANCE.UM"

T0 = np.datetime64("2026-05-28T00:00:30", "ns")  # snapshot as_of / spawn time
SETTLED = np.datetime64("2026-05-28T00:00:00", "ns")  # order last change — past grace
D_ON = lambda: QubxLogConfig.set_log_level("DEBUG")
D_OFF = lambda: QubxLogConfig.set_log_level("ERROR")


def _inst(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol(EXCHANGE, symbol)
    assert inst is not None
    return inst


_GEN = object()  # sentinel: generate a unique venue id from the cid


def _order(
    cid: str,
    *,
    instrument: Instrument = _inst(),
    venue_id=_GEN,
    status: OrderStatus = OrderStatus.ACCEPTED,
    last_update_time: np.datetime64 | None = SETTLED,
    price=50_000.0,
    quantity=1.0,
    filled=0.0,
) -> Order:
    # default: a unique exchange order id derived from the cid (so two distinct orders never
    # collide). Pass an explicit value to override, or `venue_id=None` for an unacked order.
    # last_update_time = the order's update timestamp (Differ grace gate + reconcile guard).
    # instrument defaults to BTCUSDT — pass _inst("ETHUSDT") etc. for multi-instrument scenarios.
    vid = f"v_{cid}" if venue_id is _GEN else venue_id
    return Order(
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=instrument,
        quantity=quantity,
        filled_quantity=filled,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_order_id=vid,  # type: ignore
        price=price,
        submitted_at=SETTLED,
        last_update_time=last_update_time,
        origin=OrderOrigin.FRAMEWORK,
    )


def _local(*orders: Order) -> AccountState:
    st = AccountState(EXCHANGE, "USDT")
    for o in orders:
        st.add_order(o)
    return st


def _origin(*, as_of=T0, open_orders=None, positions=None, balances=None) -> AccountSnapshot:
    return AccountSnapshot(
        exchange=EXCHANGE, as_of=as_of, open_orders=open_orders, positions=positions, balances=balances
    )


def _balance(currency: str, *, free: float, locked: float = 0.0) -> Balance:
    return Balance(exchange=EXCHANGE, currency=currency, free=free, locked=locked, total=free + locked)


def _position(
    qty: float, *, instrument: Instrument = _inst(), avg: float = 59_000.0, r_pnl: float = 0.0, ts: np.datetime64 = T0
) -> Position:
    # a position carrying its own venue update time (ts) — the reconcile watermark / hist-deals
    # `since`. r_pnl is locally-accumulated accounting the snapshot must NOT clobber.
    p = Position(instrument, quantity=qty, pos_average_price=avg, r_pnl=r_pnl)
    p.last_update_time = ts  # type: ignore
    return p


def _deal(
    ts: np.datetime64,
    *,
    instrument: Instrument = _inst(),
    amount: float = 0.002,
    price: float = 59_100.0,
    trade_id: str = "td",
    cid: str | None = None,
    venue_id: str | None = None,
) -> DealEvent:
    # A DealEvent for driving on_event / reducer.apply in scenarios. Default: instrument-addressed
    # only (no cid/venue id) — pass cid and/or venue_id to address a specific order, amount<0 to sell,
    # ts as the venue trade time (vs the position watermark), and a unique trade_id to exercise dedup.
    return DealEvent(
        instrument=instrument,
        client_order_id=cid,
        venue_order_id=venue_id,
        deal=Deal(
            trade_id=trade_id,
            order_id=venue_id or "v",
            time=ts,
            amount=amount,
            price=price,
            aggressive=True,
        ),
    )


def _reconciler() -> Reconciler:
    # short windows so scenarios are a few ticks; differ grace 5s. Funding sweep off so the
    # startup sweep doesn't leak into unrelated action assertions (own section below).
    return Reconciler(
        Differ(grace="5s"),
        snapshot_interval="30s",
        missing_wait="2s",
        missing_max_retries=2,
        position_confirm_wait="2s",
        order_confirm_wait="2s",
        order_confirm_max_retries=2,
        funding_sweep_enabled=False,
    )


def _passed_seconds(base, s: int):
    return base + np.timedelta64(s, "s")


def _mark_in_session(st: AccountState) -> None:
    # Simulate a prior applied snapshot so the reconcile under test is IN-SESSION — not the FIRST
    # reconcile after start, which intentionally adopts venue positions WITHOUT requesting hist-deals
    # (that is the restorer's job). Recovery machinery is exercised on in-session drift.
    st.mark_snapshot_applied(_passed_seconds(T0, -20))


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
    D_ON()

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
    D_OFF()


# NOTE: a terminal order (FILLED/CANCELED/...) never appears in a venue's open_orders, so a
# "snapshot reports it terminal" case is not real — that status arrives via the RequestStatus
# fetch reply (the connector replays the real event on the bus). Hence no reappear-FILLED test.


def test_missing_order_reappears_open_in_snapshot_resolves_without_lost():
    D_ON()
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
    D_OFF()


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
                    last_update_time=_passed_seconds(T0, 4),  # newer VENUE ts than local
                )
            ],
        ),
        _passed_seconds(T0, 5),
    )

    o = st.get_order("order1")
    assert o.status == OrderStatus.PARTIALLY_FILLED  # type: ignore # reconciled in-mem
    assert o.filled_quantity == 0.5  # type: ignore
    assert [type(x) for x in a] == [RouteEvent]  # and a fill event sent for notification
    assert isinstance(a[0].event, OrderPartiallyFilledEvent) and a[0].event.client_order_id == "order1"  # type: ignore


def test_snapshot_does_not_wipe_pending_cancel_marker():
    D_ON()
    # our cancel is in flight (PENDING_CANCEL). A snapshot poll still showing the order live
    # (non-terminal) must NOT wipe the pending marker — the venue resolves the race itself.
    rec = _reconciler()
    st = _local(_order("order1", status=OrderStatus.PENDING_CANCEL))
    a = rec.on_snapshot(
        st,
        _origin(
            as_of=_passed_seconds(T0, 5),
            open_orders=[_order("order1", status=OrderStatus.ACCEPTED, last_update_time=_passed_seconds(T0, 4))],
        ),
        _passed_seconds(T0, 5),
    )
    assert a == []  # nothing reconciled / routed
    assert st.get_order("order1").status == OrderStatus.PENDING_CANCEL  # type: ignore # marker preserved
    D_OFF()


# --------------------------------------------------------------------------- #
# I.b AwaitOrderConfirm (order sent, awaiting venue confirmation)
# --------------------------------------------------------------------------- #


def test_order_sent_spawns_await_confirm_task():
    rec = _reconciler()
    st = _local(_order("X1", status=OrderStatus.SUBMITTED, venue_id=None))
    a = rec.on_order_sent(st, st.get_order("X1"), T0)  # type: ignore
    assert a == []  # nothing fetched yet — we WAIT for the venue ack
    assert rec.active_keys() == {"X1"}


def test_await_confirm_resolved_by_accept_event_drops_task():
    rec = _reconciler()
    st = _local(_order("X1", status=OrderStatus.SUBMITTED, venue_id=None))
    rec.on_order_sent(st, st.get_order("X1"), T0)  # type: ignore
    # - the venue's ack arrives on the normal event path -> the task is satisfied
    rec.on_event(
        st,
        OrderAcceptedEvent(instrument=_inst(), client_order_id="X1", venue_order_id="v_X1", accepted_at=T0),
        _passed_seconds(T0, 1),
    )
    assert rec.active_keys() == set()  # confirmed -> dropped, no LOST


def test_await_confirm_drops_when_order_no_longer_inflight():
    # the order got confirmed in state without an event routed to the task; the next tick sees it
    # no longer in-flight -> done (no fetch, no LOST).
    rec = _reconciler()
    st = _local(_order("X1", status=OrderStatus.SUBMITTED, venue_id=None))
    rec.on_order_sent(st, st.get_order("X1"), T0)  # type: ignore
    st.get_order("X1").status = OrderStatus.ACCEPTED  # type: ignore # confirmed out-of-band
    out = rec.on_tick(st, _passed_seconds(T0, 3))  # tick sees it no longer in-flight
    assert rec.active_keys() == set()  # task dropped...
    assert not any(isinstance(a, (RequestStatus, RouteEvent)) for a in out)  # ...with no fetch/LOST


def test_await_confirm_unanswered_ends_lost():
    rec = _reconciler()
    st = _local(_order("X1", status=OrderStatus.SUBMITTED, venue_id=None))
    rec.on_order_sent(st, st.get_order("X1"), T0)  # type: ignore
    rec.on_tick(st, _passed_seconds(T0, 3))  # retry 1 -> RequestStatus
    rec.on_tick(st, _passed_seconds(T0, 5))  # retry 2 -> RequestStatus
    out = rec.on_tick(st, _passed_seconds(T0, 7))  # exhausted -> route LOST
    assert [type(a) for a in out] == [RouteEvent]
    assert isinstance(out[0].event, OrderLostEvent) and out[0].event.client_order_id == "X1"  # type: ignore
    assert rec.active_keys() == set()


def test_await_confirm_fetches_status_after_wait_window():
    rec = _reconciler()
    st = _local(_order("X1", status=OrderStatus.SUBMITTED, venue_id=None))
    rec.on_order_sent(st, st.get_order("X1"), T0)  # type: ignore
    rec.on_tick(st, T0)  # consume the initial snapshot-due so later ticks are clean
    assert rec.on_tick(st, _passed_seconds(T0, 1)) == []  # inside the wait window (2s)
    out = rec.on_tick(st, _passed_seconds(T0, 3))  # past it -> fetch status
    assert out == [RequestStatus(cid="X1", venue_id=None, instrument=_inst())]


# --------------------------------------------------------------------------- #
# II. ConfirmPositionBySnapshot (position size diff -> recover missed deals)
# --------------------------------------------------------------------------- #


def test_first_snapshot_position_diff_does_not_request_hist_deals():
    # On the FIRST reconciliation after start, adopting the venue position is the position
    # RESTORER's job, not the reconciler's. A size diff still applies the snapshot (size/avg
    # venue-authoritative, r_pnl preserved) and watermarks it, but must NOT spawn
    # ConfirmPositionBySnapshot / request hist-deals — doing so on start would re-fetch trades the
    # restorer already accounted for and double-count against the restored r_pnl.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.003, avg=59_000.0, r_pnl=5.0, ts=_passed_seconds(T0, -10)))

    a = rec.on_snapshot(st, _origin(positions=[_position(0.005, avg=59_100.0, ts=T0)]), T0)

    pos = st.get_position(_inst())
    assert pos.quantity == 0.005  # type: ignore # venue size adopted
    assert pos.r_pnl == 5.0  # type: ignore # restored/local accounting preserved
    assert rec.active_keys() == set()  # NO ConfirmPositionBySnapshot on the first reconcile
    assert a == []


def test_in_session_position_size_diff_spawns_confirm_task():
    # AFTER the first reconcile (in-session), a size diff means we missed a live deal: apply the
    # snapshot surgically (r_pnl survives) and spawn ConfirmPositionBySnapshot to recover it.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.003, avg=59_000.0, r_pnl=5.0, ts=_passed_seconds(T0, -10)))
    # first snapshot (matches local) consumes the first-reconcile allowance without a hist request
    rec.on_snapshot(
        st,
        _origin(positions=[_position(0.003, avg=59_000.0, ts=_passed_seconds(T0, -10))], as_of=_passed_seconds(T0, -5)),
        _passed_seconds(T0, -5),
    )

    a = rec.on_snapshot(st, _origin(positions=[_position(0.005, avg=59_100.0, ts=T0)]), T0)

    pos = st.get_position(_inst())
    assert pos.quantity == 0.005  # type: ignore
    assert pos.r_pnl == 5.0  # type: ignore
    assert rec.active_keys() == {"BTCUSDT"}  # ConfirmPositionBySnapshot task owns the symbol
    assert a == []


def test_venue_figures_applied_from_snapshot():
    # venue-reported equity/margins are captured; total_capital prefers the venue equity.
    rec = _reconciler()
    st = _local()
    rec.on_snapshot(st, AccountSnapshot(exchange=EXCHANGE, as_of=T0, equity=12_345.0, available_margin=10_000.0), T0)
    figs = st.get_venue_figures()
    assert figs is not None and figs.equity == 12_345.0  # type: ignore
    assert st.total_capital() == 12_345.0


def test_original_order_missing_recovered_order():
    # the venue reports an order we don't track -> recovers it (framework order seen back =
    # RECOVERED), keeping its cid/status.
    rec = _reconciler()
    st = _local()  # no local orders
    rec.on_snapshot(st, _origin(open_orders=[_order("ext1", venue_id="v_ext1", status=OrderStatus.ACCEPTED)]), T0)
    o = st.get_order("ext1")
    assert o is not None  # type: ignore
    assert o.status == OrderStatus.ACCEPTED  # type: ignore
    assert o.origin == OrderOrigin.RECOVERED  # type: ignore
    assert o.venue_order_id == "v_ext1"  # type: ignore


def test_one_failing_diff_atom_does_not_abort_the_snapshot(monkeypatch):
    # A handler raising on one atom must not sink the rest of the snapshot reconcile
    # (balances/figures/other positions). Each atom is applied in isolation.
    rec = _reconciler()
    st = _local()
    snap = _origin(
        open_orders=[_order("ext1", venue_id="v_ext1", status=OrderStatus.ACCEPTED)],  # -> _recover_order
        balances=[_balance("USDT", free=1000.0)],  # -> apply_balance_snapshot
    )

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(rec, "_recover_order", _boom)
    rec.on_snapshot(st, snap, T0)  # must NOT raise

    bal = st.get_balance("USDT")
    assert bal is not None and bal.total == 1000.0  # type: ignore  # later atom still applied


def test_stale_snapshot_is_dropped():
    # the as_of ratchet: a snapshot at/before the last applied as_of is rejected wholesale —
    # no diffs applied, no tasks spawned.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.003, avg=59_000.0, ts=_passed_seconds(T0, -10)))

    rec.on_snapshot(st, _origin(as_of=T0, positions=[_position(0.005, avg=59_000.0, ts=T0)]), T0)
    assert st.get_position(_inst()).quantity == 0.005  # type: ignore # first snapshot applied

    # a stale snapshot (as_of < the applied T0) must be ignored entirely
    a = rec.on_snapshot(
        st,
        _origin(as_of=_passed_seconds(T0, -5), positions=[_position(0.009, avg=59_000.0, ts=_passed_seconds(T0, -5))]),
        _passed_seconds(T0, 1),
    )
    assert a == []
    assert st.get_position(_inst()).quantity == 0.005  # unchanged by the stale snapshot


def test_position_confirm_late_deal_arrives_drops_task_without_fetch():
    # the missed deal arrives on its own (late WS) within the wait window -> recovered, drop,
    # no hist fetch needed.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.003, ts=_passed_seconds(T0, -10)))
    _mark_in_session(st)
    rec.on_snapshot(st, _origin(positions=[_position(0.005, ts=T0)]), T0)
    assert rec.active_keys() == {"BTCUSDT"}

    a = rec.on_event(st, _deal(_passed_seconds(T0, 1)), _passed_seconds(T0, 1))
    assert rec.active_keys() == set()  # recovered by the arriving deal -> task dropped
    assert a == []  # no RequestHistDeals


def test_position_confirm_no_deal_fetches_hist_deals_after_window():
    D_ON()
    # no late deal by the window -> fetch the missed deals (since = position venue ts) so the
    # ledger has them; the connector replays them as DealEvents (reducer venue-ts guard skips
    # re-booking). One-shot -> task dropped.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.003, ts=_passed_seconds(T0, -10)))
    _mark_in_session(st)
    rec.on_snapshot(st, _origin(positions=[_position(0.005, ts=T0)]), T0)

    assert rec.on_tick(st, _passed_seconds(T0, 1)) == []  # inside the wait window (2s)
    a = rec.on_tick(st, _passed_seconds(T0, 3))  # past the window, no deal seen
    assert a == [RequestHistDeals(instrument=_inst(), since=T0 - HIST_DEALS_LOOKBACK)]
    assert rec.active_keys() == set()  # one-shot fetch -> dropped
    D_OFF()


def test_position_in_sync_does_not_spawn_task():
    D_ON()
    # snapshot agrees with local -> no diff, no task, no fetch.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.005, avg=59_100.0, ts=_passed_seconds(T0, -10)))
    a = rec.on_snapshot(st, _origin(positions=[_position(0.005, avg=59_100.0, ts=T0)]), T0)
    assert a == []
    assert rec.active_keys() == set()
    D_OFF()


def test_local_position_missing_flattens_and_spawns_confirm_task():
    D_ON()
    # local holds 0.005 but the snapshot (positions observed) lists none -> the venue says
    # flat, so we missed the closing deals. Flatten locally (r_pnl preserved) and spawn a
    # ConfirmPositionBySnapshot task to recover the closing deals for the record.
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.005, avg=59_000.0, r_pnl=7.0, ts=_passed_seconds(T0, -10)))
    _mark_in_session(st)

    a = rec.on_snapshot(st, _origin(positions=[]), T0)  # observed + empty -> absent at venue

    pos = st.get_position(_inst())
    assert pos.quantity == 0.0  # type: ignore # flattened (venue authoritative)
    assert pos.r_pnl == 7.0  # type: ignore # local accounting preserved
    assert rec.active_keys() == {"BTCUSDT"}  # confirm task to recover the closing deals
    assert a == []
    D_OFF()


# --------------------------------------------------------------------------- #
# III. Balances (inline apply from snapshot — no task)
# --------------------------------------------------------------------------- #


def test_balance_mismatch_applied_from_snapshot():
    D_ON()
    rec = _reconciler()
    st = _local()
    st.update_balance("USDT", _balance("USDT", free=400.0))
    a = rec.on_snapshot(st, _origin(balances=[_balance("USDT", free=465.0)]), T0)
    assert st.get_balance("USDT").total == 465.0  # type: ignore # applied from the snapshot
    assert a == []
    assert rec.active_keys() == set()  # balances are inline — never spawn a task
    D_OFF()


def test_original_balance_missing_applied_from_snapshot():
    D_ON()
    # the snapshot reports a currency we don't track locally -> recover it.
    rec = _reconciler()
    st = _local()
    a = rec.on_snapshot(st, _origin(balances=[_balance("USDT", free=465.0)]), T0)
    assert st.get_balance("USDT").total == 465.0  # type: ignore
    assert a == []
    assert rec.active_keys() == set()
    D_OFF()


def test_position_decrease_the_deals_arrived_with_small_latency():
    # The missed CLOSING deal (traded BEFORE the snapshot, venue ts <= watermark) is DELIVERED late
    # by WS but still within the confirm window. AM0 routes every DealEvent to BOTH reducer.apply
    # (ledger) AND reconciler.on_event (task notification) — the arriving deal must satisfy the
    # confirm task so NO RequestHistDeals is fetched (we already received it live).
    D_ON()
    rec = _reconciler()
    st = _local()  # no local orders — the missed close's order already filled+evicted at the venue
    # local LONG 0.005 @ 59_000; the venue closed 0.002 we missed -> snapshot shows 0.003
    st.set_position(_inst(), _position(0.005, avg=59_000.0, ts=_passed_seconds(T0, -10)))
    _mark_in_session(st)

    # snapshot decrease -> size-only reconcile (+ watermark=T0 + confirm task), pnl NOT realized
    rec.on_snapshot(st, _origin(positions=[_position(0.003, avg=59_000.0, ts=T0)]), T0)
    assert st.get_position(_inst()).quantity == 0.003  # type: ignore
    assert rec.active_keys() == {"BTCUSDT"}
    assert rec.on_tick(st, T0) == []  # nothing fetched at spawn (inside the window)

    # the FULL missed close (-0.002, == the missed delta) arrives late but within the 2s window. It
    # TRADED before the snapshot (ts = T0-1 <= watermark) and is merely DELIVERED at T0+1 (latency).
    hist = _deal(_passed_seconds(T0, -1), amount=-0.002, price=59_500.0, trade_id="hist1", venue_id="v_hist")

    # ledger leg: watermark books r_pnl-only -> size stays 0.003, realized PnL lands
    res = reducer.apply(st, hist, _passed_seconds(T0, 1))
    assert res.position is not None  # type: ignore # r_pnl realized
    assert st.get_position(_inst()).quantity == 0.003  # type: ignore # size unchanged (no double-move)
    assert st.get_position(_inst()).r_pnl == 1.0  # type: ignore # 0.002 * (59_500 - 59_000)

    # reconciler leg: the arriving deal FULLY covers the missed delta -> task satisfied, dropped
    assert rec.on_event(st, hist, _passed_seconds(T0, 1)) == []
    assert rec.active_keys() == set()  # fully recovered by the arriving deal

    # window elapses -> NO RequestHistDeals (we already got the whole missed delta live)
    assert rec.on_tick(st, _passed_seconds(T0, 3)) == []
    D_OFF()


def test_position_decrease_partial_deal_still_fetches_remainder():
    # Only PART of the missed delta arrives live (-0.001 of -0.002). The confirm task must NOT treat
    # itself as recovered — when the window elapses it still fetches the remainder via RequestHistDeals
    # (the hist reply is deduped against the -0.001 we already booked-skipped).
    D_ON()
    rec = _reconciler()
    st = _local()
    st.set_position(_inst(), _position(0.005, avg=59_000.0, ts=_passed_seconds(T0, -10)))
    _mark_in_session(st)
    rec.on_snapshot(st, _origin(positions=[_position(0.003, avg=59_000.0, ts=T0)]), T0)
    assert rec.active_keys() == {"BTCUSDT"}  # missed delta = 0.002

    # only half the missed delta is delivered live within the window
    half = _deal(_passed_seconds(T0, -1), amount=-0.001, price=59_500.0, trade_id="hist1", venue_id="v_hist")
    reducer.apply(st, half, _passed_seconds(T0, 1))
    assert rec.on_event(st, half, _passed_seconds(T0, 1)) == []  # partial — does NOT drop yet
    assert rec.active_keys() == {"BTCUSDT"}  # still armed: 0.001 of 0.002 outstanding

    # window elapses with the remainder still missing -> fetch it for the record
    out = rec.on_tick(st, _passed_seconds(T0, 3))
    assert out == [RequestHistDeals(instrument=_inst(), since=T0 - HIST_DEALS_LOOKBACK)]
    assert rec.active_keys() == set()  # one-shot fetch -> dropped
    D_OFF()


def test_position_decrease_then_pushed_historical_deals_end_to_end():
    # END-TO-END (manual AM0 wiring): snapshot shrinks the position, the confirm task fetches the
    # missed deals, and we PUSH those historical deals back through reducer.apply (what the connector
    # does on a RequestHistDeals reply). Size stays venue-authoritative; the recovered close realizes
    # its r_pnl (pnl-only book) and a re-delivery is deduped.
    D_ON()
    rec = _reconciler()
    st = _local()  # no local orders — the missed close's order already filled+evicted at the venue
    # local LONG 0.005 @ 59_000; the venue closed 0.002 we missed -> snapshot shows 0.003
    st.set_position(_inst(), _position(0.005, avg=59_000.0, ts=_passed_seconds(T0, -10)))
    _mark_in_session(st)

    # 1) snapshot decrease -> size-only reconcile (+ watermark + confirm task), pnl not yet realized
    rec.on_snapshot(st, _origin(positions=[_position(0.003, avg=59_000.0, ts=T0)]), T0)
    pos = st.get_position(_inst())
    assert pos.quantity == 0.003  # type: ignore # size corrected from the snapshot
    assert pos.r_pnl == 0.0  # type: ignore # snapshot is size-only — no realized pnl yet
    assert rec.active_keys() == {"BTCUSDT"}

    assert rec.on_tick(st, T0) == []  # inside the window
    assert rec.on_tick(st, _passed_seconds(T0, 1)) == []

    # 2) no late WS deal by the window -> fetch the missed deals since the venue watermark
    out = rec.on_tick(st, _passed_seconds(T0, 3))
    assert out == [RequestHistDeals(instrument=_inst(), since=T0 - HIST_DEALS_LOOKBACK)]
    assert rec.active_keys() == set()  # one-shot fetch task dropped

    # 3) the connector pushes the fetched HISTORICAL deal back as a normal DealEvent (addressed by
    #    venue id only — its order already filled+evicted; the reducer recovers it external)
    hist = _deal(_passed_seconds(T0, -2), amount=-0.002, price=59_500.0, trade_id="hist1", venue_id="v_hist")
    res = reducer.apply(st, hist, _passed_seconds(T0, 3))

    pos = st.get_position(_inst())
    assert res.deal is not None  # type: ignore # recorded (logged + trade-id dedup)
    assert res.position is not None  # type: ignore # r_pnl realized (pnl-only book)
    assert pos.quantity == 0.003  # type: ignore # size stays venue-authoritative (NOT driven to 0.001)
    assert pos.r_pnl == 1.0  # type: ignore # 0.002 * (59_500 - 59_000) realized from the recovered close

    # 4) a re-delivery of the SAME hist deal is deduped (no double realization)
    res2 = reducer.apply(st, hist, _passed_seconds(T0, 4))
    assert res2.deal is None  # type: ignore # trade-id already seen
    assert st.get_position(_inst()).quantity == 0.003  # type: ignore
    assert st.get_position(_inst()).r_pnl == 1.0  # type: ignore # not double-realized
    D_OFF()


# --------------------------------------------------------------------------- #
# Full vs light snapshot cadence — steady-state snapshots are light (positions+balance
# only) to stay off the order throttle; a periodic full sweep + the first-ever request
# stay full so unknown/algo orders are still discovered.
# --------------------------------------------------------------------------- #


def test_first_snapshot_is_full_then_light_then_full_after_sweep():
    rec = Reconciler(Differ(grace="5s"), snapshot_interval="30s", full_snapshot_interval="5m")
    st = _local()
    ex = st.exchange

    # first request after start must be FULL — no local order state yet, so we can't gate the
    # all-orders + algo discovery fetch (matches the initial-discovery contract)
    assert RequestSnapshot(ex, include_orders=True) in rec.on_tick(st, T0)

    # a due request inside the full-sweep window is LIGHT (positions + balance only)
    assert RequestSnapshot(ex, include_orders=False) in rec.on_tick(st, _passed_seconds(T0, 30))
    assert RequestSnapshot(ex, include_orders=False) in rec.on_tick(st, _passed_seconds(T0, 60))

    # once the full-sweep interval elapses, the next due request is FULL again (WS-drop backstop)
    assert RequestSnapshot(ex, include_orders=True) in rec.on_tick(st, _passed_seconds(T0, 300))


# --------------------------------------------------------------------------- #
# Funding sweep — hourly-aligned RequestFundingPayments (startup, then every hour
# boundary + 10min offset): floor anchor + next-due time, funding_sweep_enabled kill switch.
# --------------------------------------------------------------------------- #


def _funding_reconciler() -> Reconciler:
    return Reconciler(Differ(grace="5s"))  # funding sweep enabled by default


def _funding_actions(actions) -> list[RequestFundingPayments]:
    return [a for a in actions if isinstance(a, RequestFundingPayments)]


def _at(hhmmss: str) -> np.datetime64:
    return np.datetime64(f"2026-05-28T{hhmmss}", "ns")


def test_funding_sweep_kill_switch_disables_everything():
    # funding_sweep_enabled=False -> no startup sweep, no hourly schedule
    st = _local()
    rec = Reconciler(Differ(grace="5s"), funding_sweep_enabled=False)
    assert _funding_actions(rec.on_tick(st, T0)) == []
    assert _funding_actions(rec.on_tick(st, _passed_seconds(T0, 7200))) == []


def test_funding_sweep_startup_floor_defaults_to_now():
    # no restored last_funding_time anywhere -> floor = connect time (empty startup window)
    st = _local()
    rec = _funding_reconciler()
    assert _funding_actions(rec.on_tick(st, T0)) == [RequestFundingPayments(EXCHANGE, since=T0)]
    # nothing further until the next hour boundary + offset
    assert _funding_actions(rec.on_tick(st, _at("01:09:59"))) == []


def test_funding_sweep_startup_floor_from_restored_positions():
    # restored last_funding_time -> exclusive +1ns floor (that settlement is already inside
    # restored cumulative_funding); positions without one are ignored
    st = _local()
    settled = np.datetime64("2026-05-27T16:00:00", "ns")
    pos = _position(1.0)
    pos.last_funding_time = settled  # type: ignore
    st.set_position(pos.instrument, pos)
    other = _position(2.0, instrument=_inst("ETHUSDT"))  # last_funding_time stays NaT
    st.set_position(other.instrument, other)

    rec = _funding_reconciler()
    expected = settled + np.timedelta64(1, "ns")
    assert _funding_actions(rec.on_tick(st, T0)) == [RequestFundingPayments(EXCHANGE, since=expected)]


def test_funding_sweep_hourly_schedule():
    # fires once per hour at boundary + 10min offset; window = max(floor, now - 30m)
    st = _local()
    rec = _funding_reconciler()
    rec.on_tick(st, T0)  # startup (T0 = 00:00:30) -> next due 01:10:00

    assert _funding_actions(rec.on_tick(st, _at("01:09:59"))) == []  # before boundary + offset
    assert _funding_actions(rec.on_tick(st, _at("01:10:30"))) == [
        RequestFundingPayments(EXCHANGE, since=_at("00:40:30"))  # now - 30m lookback
    ]
    assert _funding_actions(rec.on_tick(st, _at("01:30:00"))) == []  # rescheduled to 02:10
    assert _funding_actions(rec.on_tick(st, _at("02:10:30"))) == [
        RequestFundingPayments(EXCHANGE, since=_at("01:40:30"))  # second hour re-fires
    ]


def test_funding_sweep_window_floor_clamped():
    # a sweep shortly after startup never reaches below the floor (restored-state guard)
    st = _local()
    rec = _funding_reconciler()
    start = _at("00:59:00")
    rec.on_tick(st, start)  # startup -> floor = 00:59:00, next due 01:10:00

    assert _funding_actions(rec.on_tick(st, _at("01:10:30"))) == [
        RequestFundingPayments(EXCHANGE, since=start)  # 01:10:30 - 30m reaches before the floor
    ]
