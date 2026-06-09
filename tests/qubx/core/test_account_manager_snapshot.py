from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig
from qubx.core.basics import (
    Balance,
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderStatus,
    Position,
)
from qubx.core.events import AccountSnapshot, AccountSnapshotEvent


class _T:
    def __init__(self, t="2026-05-28T00:00:00"):
        self.t = np.datetime64(t)

    def time(self):
        return self.t

    def adv(self, ms):
        self.t = self.t + np.timedelta64(ms, "ms")


def _instrument(symbol="BTCUSDT", exchange="binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _am(exchange="binance", cfg=None):
    am = AccountManager.__new__(AccountManager)
    am._init_state(
        connectors={exchange: MagicMock()},
        base_currencies={exchange: "USDT"},
        time=_T(),
        cfg=cfg or AccountManagerConfig(snapshot_check_threshold_ms=5_000),
        account_id="test",
        tcc=None,
    )
    return am


def _order(cid, status, time, vid=None, origin=OrderOrigin.FRAMEWORK, instrument=None, qty=1.0):
    return Order(
        client_order_id=cid,
        venue_order_id=vid,
        origin=origin,
        type="LIMIT",
        instrument=instrument,
        time=np.datetime64(time),
        quantity=qty,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
    )


def _snap_event(
    exchange="binance",
    as_of="2026-05-28T01:00:00",
    open_orders=None,
    positions=None,
    balances=None,
    equity=None,
    available_margin=None,
    margin_ratio=None,
):
    return AccountSnapshotEvent(
        instrument=None,
        snapshot=AccountSnapshot(
            exchange=exchange,
            as_of=np.datetime64(as_of),
            open_orders=open_orders,
            positions=positions,
            balances=balances,
            equity=equity,
            available_margin=available_margin,
            margin_ratio=margin_ratio,
        ),
    )


def test_missing_order_past_grace_transitions_terminal():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # order submitted at t0, vid V1, now snapshot at t0+1h with no open orders -> past grace
    state.add_order(_order("cid-1", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    assert state.get_order("cid-1").status is OrderStatus.REJECTED
    assert state.get_order("cid-1").rejected_reason == "reconcile: missing from snapshot"


def test_missing_order_within_grace_not_terminated():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:59:59", vid="V1", instrument=inst))
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    # snapshot as_of is 1s after order.time which is below the 5s grace window
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    assert state.get_order("cid-1").status is OrderStatus.ACCEPTED


def test_stale_snapshot_does_not_clobber_fresh_state():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    existing = _order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    existing.last_updated_at = np.datetime64("2026-05-28T02:00:00")
    existing.filled_quantity = 0.5
    state.add_order(existing)
    # snapshot older than the order's last_updated_at must not overwrite
    snap_order = _order("cid-1", OrderStatus.FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 1.0
    am._time.t = np.datetime64("2026-05-28T03:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    assert state.get_order("cid-1").filled_quantity == 0.5
    assert state.get_order("cid-1").status is OrderStatus.ACCEPTED


def test_out_of_order_snapshot_skipped():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am._time.t = np.datetime64("2026-05-28T03:00:00")
    am.apply(_snap_event(as_of="2026-05-28T02:00:00", open_orders=[]))
    assert state.get_order("cid-1").status is OrderStatus.REJECTED
    # re-add a fresh order, then deliver an older snapshot — must be skipped
    state.add_order(_order("cid-2", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V2", instrument=inst))
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    assert state.get_order("cid-2").status is OrderStatus.SUBMITTED


def test_external_order_materialized_from_snapshot():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    snap_order = _order("manual-123", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VX", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    materialized = state.get_order_by_venue_id("VX")
    assert materialized is not None
    assert materialized.origin is OrderOrigin.EXTERNAL
    assert materialized.client_order_id == "ext:VX"


def test_recovered_order_materialized_from_snapshot():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # Realistic framework cid as produced by ClientIdStore._create_id ("qubx_<sym>_<n>").
    # Reconcile must classify it as RECOVERED — using the doc's illustrative "qubx-"
    # would silently misclassify every recovered framework order as EXTERNAL.
    snap_order = _order("qubx_BTCUSDT_1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VY", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    materialized = state.get_order("qubx_BTCUSDT_1")
    assert materialized is not None
    assert materialized.origin is OrderOrigin.RECOVERED
    assert materialized.client_order_id == "qubx_BTCUSDT_1"


def test_existing_order_updated_from_fresh_snapshot():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    existing = _order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    existing.last_updated_at = np.datetime64("2026-05-28T00:30:00")
    state.add_order(existing)
    snap_order = _order("cid-1", OrderStatus.PARTIALLY_FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 0.4
    snap_order.avg_fill_price = 49_900.0
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    updated = state.get_order("cid-1")
    assert updated.status is OrderStatus.PARTIALLY_FILLED
    assert updated.filled_quantity == 0.4
    assert updated.avg_fill_price == 49_900.0
    assert updated.last_updated_at == np.datetime64("2026-05-28T01:00:00")


def test_snapshot_terminalization_runs_full_transition_machinery():
    # D8 regression: a snapshot that terminalizes an active order must go through
    # transition_order — audit trail, counters, inflight cleared, eviction registered —
    # not a bare status write that left the order a permanent hidden resident polled
    # forever by the inflight sweep.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    assert len(state.get_inflight_orders()) == 1
    snap_order = _order("cid-1", OrderStatus.FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 1.0
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.FILLED
    # the stale inflight entry is gone -> no pointless venue polling on the next tick
    assert state.get_inflight_orders() == []
    # audit trail + counters recorded the transition
    assert [(t.from_status, t.to_status) for t in order.transitions] == [(OrderStatus.SUBMITTED, OrderStatus.FILLED)]
    assert state.get_transition_counts()["filled"] == 1
    # eviction registered: the retention sweep moves it to history instead of leaking it
    am._time.adv(31_000)
    am._sweep_terminal_evictions()
    assert state.get_active_order("cid-1") is None
    assert state.get_order("cid-1") is not None  # findable in the terminal-history ring


def test_snapshot_does_not_wipe_pending_marker_with_live_status():
    # The snapshot is a poll of venue state; an outstanding cancel may still be in flight.
    # A live venue status (ACCEPTED) must NOT clear PENDING_CANCEL — the venue resolves
    # the race itself (canceled, or cancel-rejected which reverts via pre_pending).
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    snap_order = _order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 0.3
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.PENDING_CANCEL
    assert state.get_pre_pending("cid-1") is OrderStatus.ACCEPTED  # revert target intact
    assert len(state.get_inflight_orders()) == 1  # sweep keeps polling the cancel
    # non-status property drift still reconciled
    assert order.filled_quantity == 0.3
    assert order.last_updated_at == np.datetime64("2026-05-28T01:00:00")


def test_snapshot_terminal_status_resolves_pending_marker():
    # A terminal snapshot status IS the venue's resolution of the outstanding request —
    # it falls through the pending guard and terminalizes via the machinery.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    snap_order = _order("cid-1", OrderStatus.CANCELED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    assert state.get_order("cid-1").status is OrderStatus.CANCELED
    assert state.get_pre_pending("cid-1") is None
    assert state.get_inflight_orders() == []


def test_snapshot_illegal_edge_forced_with_indices_fixed():
    # can_transition(PARTIALLY_FILLED, ACCEPTED) is False, but the snapshot is venue-
    # authoritative: the write is forced (logged) and STILL routed through
    # transition_order so the audit and indices stay consistent.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.PARTIALLY_FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    snap_order = _order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.ACCEPTED
    assert [(t.from_status, t.to_status) for t in order.transitions] == [
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.ACCEPTED)
    ]
    assert state.get_inflight_orders() == []


def test_snapshot_resurrects_locally_terminal_order_and_unregisters_eviction():
    # We locally terminalized (e.g. sweep give-up REJECTED) but the venue still shows the
    # order open: the forced resurrect must pop the pending-evict entry, or the retention
    # sweep would evict a live order to history.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am.transition_order("binance", "cid-1", OrderStatus.REJECTED)
    snap_order = _order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    assert state.get_order("cid-1").status is OrderStatus.ACCEPTED
    am._time.adv(31_000)
    am._sweep_terminal_evictions()
    assert state.get_active_order("cid-1") is not None  # NOT evicted — it is live again


def test_position_updated_from_snapshot():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    snap_pos = Position(instrument=inst, quantity=2.0, pos_average_price=50_000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", positions=[snap_pos]))
    assert state.get_position(inst) is snap_pos
    assert state.get_position(inst).quantity == 2.0


def test_balance_updated_from_snapshot():
    am = _am()
    state = am._states["binance"]
    snap_bal = Balance(exchange="binance", currency="USDT", free=900.0, locked=100.0, total=1000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", balances=[snap_bal]))
    assert state.get_balance("USDT") is snap_bal
    assert state.get_balance("USDT").total == 1000.0


def test_snapshot_updates_existing_position_and_balance_in_place():
    # A snapshot carrying the same instrument/currency with new values updates the
    # EXISTING position/balance objects in place (identity preserved — code across the
    # framework holds references), never swapping in the snapshot's throwaway objects.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.set_position(inst, Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0))
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))
    pos_ref = state.get_position(inst)
    bal_ref = state.get_balance("USDT")

    snap_pos = Position(instrument=inst, quantity=3.0, pos_average_price=49_000.0)
    snap_bal = Balance(exchange="binance", currency="USDT", free=400.0, locked=100.0, total=500.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", positions=[snap_pos], balances=[snap_bal]))

    # identity preserved (same objects), values overwritten from the snapshot
    assert state.get_position(inst) is pos_ref
    assert state.get_position(inst) is not snap_pos
    assert state.get_position(inst).quantity == 3.0
    assert state.get_balance("USDT") is bal_ref
    assert state.get_balance("USDT") is not snap_bal
    assert state.get_balance("USDT").total == 500.0


def test_reconcile_applies_to_state():
    # Reconcile mutates AccountState silently (no per-event callback) — the strategy is
    # notified once via on_account_update at the PM layer. Here we assert the state effects.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # one missing (-> terminal), one unknown (-> materialized), plus position and balance
    state.add_order(_order("cid-gone", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="VG", instrument=inst))
    snap_order = _order("manual-9", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VNEW", instrument=inst)
    snap_pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    snap_bal = Balance(exchange="binance", currency="USDT", total=1000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(
        _snap_event(
            as_of="2026-05-28T01:00:00",
            open_orders=[snap_order],
            positions=[snap_pos],
            balances=[snap_bal],
        )
    )
    # missing from snapshot + older than the grace window -> terminal (REJECTED, was SUBMITTED)
    assert state.get_order("cid-gone").status is OrderStatus.REJECTED
    # unknown venue id in the snapshot -> materialized into the cache
    assert state.get_order_by_venue_id("VNEW") is not None
    # position / balance overwritten from the snapshot
    assert state.get_position(inst).quantity == 1.0
    assert state.get_balance("USDT").total == 1000.0


def test_snapshot_sets_venue_figures_and_metrics_prefer_them():
    am = _am()
    state = am._states["binance"]
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))
    am.apply(_snap_event(equity=5000.0, available_margin=4000.0, margin_ratio=42.0))
    figures = state.get_venue_figures()
    assert figures is not None
    assert figures.equity == 5000.0
    assert figures.available_margin == 4000.0
    assert figures.margin_ratio == 42.0
    assert figures.as_of == np.datetime64("2026-05-28T01:00:00")
    # per-metric prefer-venue over the derived values
    assert am.get_total_capital("binance") == 5000.0
    assert am.get_available_margin("binance") == 4000.0
    assert am.get_margin_ratio("binance") == 42.0


def test_snapshot_without_figures_keeps_previous_capture():
    # Absence means "not observed" (failed balance leg / venue lacking them),
    # not "gone" — a later figure-less snapshot must not clear the capture.
    am = _am()
    state = am._states["binance"]
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", equity=5000.0))
    am.apply(_snap_event(as_of="2026-05-28T02:00:00"))
    figures = state.get_venue_figures()
    assert figures is not None
    assert figures.equity == 5000.0
    assert figures.as_of == np.datetime64("2026-05-28T01:00:00")


def test_sim_snapshot_never_sets_venue_figures():
    # SimulatedConnector builds AccountSnapshot(exchange, as_of, open_orders) only —
    # no figure fields — so backtests always derive every metric from state.
    am = _am()
    state = am._states["binance"]
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))
    am.apply(_snap_event(open_orders=[]))
    assert state.get_venue_figures() is None
    assert am.get_total_capital("binance") == 1000.0
    assert am.get_margin_ratio("binance") == 100.0
