from unittest.mock import MagicMock

import numpy as np

from qubx import logger
from qubx.core.account_manager import AccountManager, AccountManagerConfig
from qubx.core.basics import (
    Balance,
    Deal,
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderStatus,
    Position,
)
from qubx.core.events import AccountSnapshot, AccountSnapshotEvent, DealEvent, OrderFilledEvent


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
    return AccountManager(
        connectors={exchange: MagicMock()},
        base_currencies={exchange: "USDT"},
        time=_T(),
        cfg=cfg or AccountManagerConfig(snapshot_check_threshold_ms=5_000),
        account_id="test",
    )


def _order(cid, status, time, vid=None, origin=OrderOrigin.FRAMEWORK, instrument=None, qty=1.0):
    return Order(
        client_order_id=cid,
        venue_order_id=vid,
        origin=origin,
        type="LIMIT",
        instrument=instrument,
        submitted_at=np.datetime64(time),
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


def _exhaust_fetch_budget(state, cid, n=5):
    # n matches AccountManagerConfig.inflight_check_retries default — the shared
    # per-order fetch budget reconcile reuses for missing orders.
    for _ in range(n):
        state.bump_retry(cid)


def test_missing_order_past_grace_requests_status_fetch_not_terminalized():
    # F7: a missing-past-grace order may have FILLED during a WS gap — the first response
    # is a status fetch through the connector, never a blind terminalization.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # order submitted at t0, vid V1, now snapshot at t0+1h with no open orders -> past grace
    state.add_order(_order("cid-1", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.SUBMITTED  # untouched until the venue answers
    am._connectors["binance"].request_order_status.assert_called_once_with(client_order_id="cid-1", venue_order_id="V1")
    assert state.get_retry("cid-1") == 1
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.missing == [order]
    assert result.reconcile_diff.terminated == []


def test_missing_order_exhausted_fetch_budget_terminalized():
    # Give-up: once the fetch budget is exhausted the order terminalizes exactly as the
    # pre-F7 behavior (REJECTED for never-acked SUBMITTED, reason recorded, no fetch).
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    _exhaust_fetch_budget(state, "cid-1")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.REJECTED
    assert order.rejected_reason == "reconcile: missing from snapshot"
    am._connectors["binance"].request_order_status.assert_not_called()
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.terminated == [order]
    assert result.reconcile_diff.missing == []


def test_missing_order_resolved_via_fetch_applies_true_status():
    # The connector's answer to the status fetch replays through the normal event path:
    # the true FILLED lands (no false CANCELED) and the fetch budget resets.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    assert state.get_order("cid-1").status is OrderStatus.ACCEPTED
    assert state.get_retry("cid-1") == 1

    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=None))
    assert state.get_order("cid-1").status is OrderStatus.FILLED
    assert state.get_retry("cid-1") == 0  # transition_order reset the budget


def test_missing_order_terminalizes_after_n_snapshot_cycles():
    # The loop: every snapshot cycle that still misses the order burns one fetch attempt;
    # after inflight_check_retries cycles the give-up terminalizes (ACCEPTED -> CANCELED).
    cfg = AccountManagerConfig(snapshot_check_threshold_ms=5_000, inflight_check_retries=2)
    am = _am(cfg=cfg)
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))

    for hour in ("01", "02", "03"):
        am._time.t = np.datetime64(f"2026-05-28T{hour}:00:00")
        am.apply(_snap_event(as_of=f"2026-05-28T{hour}:00:00", open_orders=[]))

    assert state.get_order("cid-1").status is OrderStatus.CANCELED
    assert am._connectors["binance"].request_order_status.call_count == 2


def test_missing_order_within_grace_not_terminated():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:59:59", vid="V1", instrument=inst))
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    # snapshot as_of is 1s after order.submitted_at which is below the 5s grace window
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    assert state.get_order("cid-1").status is OrderStatus.ACCEPTED


def test_open_orders_none_failed_fetch_leg_skips_missing_handling():
    # F16: open_orders=None means the connector's order-fetch leg FAILED — it must never
    # be treated as "venue has no orders". Live orders past grace stay untouched: no fetch
    # rescue, no budget burn, no give-up terminalization (even with the budget already
    # spent). A regression to `if snapshot.open_orders:` flips this to mass-handling.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-fresh", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VF", instrument=inst))
    state.add_order(_order("cid-spent", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="VS", instrument=inst))
    _exhaust_fetch_budget(state, "cid-spent")
    snap_bal = Balance(exchange="binance", currency="USDT", free=1000.0, total=1000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=None, balances=[snap_bal]))

    assert state.get_order("cid-fresh").status is OrderStatus.ACCEPTED
    assert state.get_order("cid-spent").status is OrderStatus.SUBMITTED
    am._connectors["binance"].request_order_status.assert_not_called()
    assert state.get_retry("cid-fresh") == 0
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.missing == []
    assert result.reconcile_diff.terminated == []
    # only the orders leg is absent — the rest of the snapshot still applies
    assert state.get_balance("USDT").total == 1000.0


def test_open_orders_empty_list_engages_missing_handling():
    # F16, the [] half of the None-vs-[] matrix: an empty list IS the venue's answer
    # ("no open orders"), so missing-handling engages — fetch rescue while budget lasts,
    # give-up terminalization once it is spent.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-fresh", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VF", instrument=inst))
    state.add_order(_order("cid-spent", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="VS", instrument=inst))
    _exhaust_fetch_budget(state, "cid-spent")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))

    fresh = state.get_order("cid-fresh")
    assert fresh.status is OrderStatus.ACCEPTED  # fetch rescue, not terminalized
    am._connectors["binance"].request_order_status.assert_called_once_with(
        client_order_id="cid-fresh", venue_order_id="VF"
    )
    assert state.get_retry("cid-fresh") == 1
    spent = state.get_order("cid-spent")
    assert spent.status is OrderStatus.REJECTED  # budget gone -> give-up
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.missing == [fresh]
    assert result.reconcile_diff.terminated == [spent]


def test_missing_accepted_order_gives_up_to_canceled():
    # F16: give-up terminalization is status-aware — a venue-acked ACCEPTED order resolves
    # to CANCELED (only never-acked SUBMITTED becomes REJECTED).
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    _exhaust_fetch_budget(state, "cid-1")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.CANCELED
    assert order.rejected_reason == "reconcile: missing from snapshot"
    am._connectors["binance"].request_order_status.assert_not_called()
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.terminated == [order]


def test_missing_order_without_timestamps_skipped():
    # F16: an order with neither last_updated_at nor submitted_at cannot be aged — it is
    # treated as just-seen and skipped before any missing handling (no fetch, no give-up
    # even with the budget spent), so a just-submitted order racing the snapshot survives.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    order = _order("cid-u", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VU", instrument=inst)
    order.submitted_at = None
    state.add_order(order)
    _exhaust_fetch_budget(state, "cid-u")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))

    assert state.get_order("cid-u").status is OrderStatus.ACCEPTED
    am._connectors["binance"].request_order_status.assert_not_called()
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.missing == []
    assert result.reconcile_diff.terminated == []


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
    _exhaust_fetch_budget(state, "cid-1")  # fetch budget spent -> the applied snapshot terminalizes
    am._time.t = np.datetime64("2026-05-28T03:00:00")
    am.apply(_snap_event(as_of="2026-05-28T02:00:00", open_orders=[]))
    assert state.get_order("cid-1").status is OrderStatus.REJECTED
    # re-add a fresh order, then deliver an older snapshot — must be skipped
    state.add_order(_order("cid-2", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V2", instrument=inst))
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    assert state.get_order("cid-2").status is OrderStatus.SUBMITTED


def test_external_order_materialized_from_snapshot():
    # Producer-classified EXTERNAL (AccountSnapshot contract) -> synthesized ext:<vid> cid.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    snap_order = _order(
        "manual-123",
        OrderStatus.ACCEPTED,
        "2026-05-28T00:00:00",
        vid="VX",
        origin=OrderOrigin.EXTERNAL,
        instrument=inst,
    )
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
    # Sim snapshots carry the order as the strategy created it (origin=FRAMEWORK, the
    # _order default): seen back via a snapshot it must materialize as RECOVERED, keep-cid.
    snap_order = _order("qubx_BTCUSDT_1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VY", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    materialized = state.get_order("qubx_BTCUSDT_1")
    assert materialized is not None
    assert materialized.origin is OrderOrigin.RECOVERED
    assert materialized.client_order_id == "qubx_BTCUSDT_1"


def test_recovered_order_with_venue_mangled_cid_keeps_cid():
    # OKX strips "_" from cids, so a recovered framework order arrives as "qubxBTCUSDT1"
    # with origin=RECOVERED assigned by the connector (venue-aware prefix). Reconcile must
    # trust that origin — re-classifying with the default "qubx_" prefix would misread it
    # as EXTERNAL and bury the strategy's own order under a synthetic ext:<vid> cid.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    snap_order = _order(
        "qubxBTCUSDT1",
        OrderStatus.ACCEPTED,
        "2026-05-28T00:00:00",
        vid="VZ",
        origin=OrderOrigin.RECOVERED,
        instrument=inst,
    )
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    materialized = state.get_order("qubxBTCUSDT1")
    assert materialized is not None
    assert materialized.origin is OrderOrigin.RECOVERED
    assert materialized.client_order_id == "qubxBTCUSDT1"


def test_unacked_order_within_grace_matched_by_cid_captures_venue_id():
    # F11 regression: a framework order whose create ack was lost (venue_order_id=None)
    # appears in the snapshot under our own cid. It must match by cid — capturing the
    # venue id and updating in place — never materialize a RECOVERED twin.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("qubx_BTCUSDT_7", OrderStatus.SUBMITTED, "2026-05-28T00:59:59", instrument=inst))
    snap_order = _order("qubx_BTCUSDT_7", OrderStatus.ACCEPTED, "2026-05-28T00:59:59", vid="V7", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_active_order("qubx_BTCUSDT_7")
    assert order is not None
    assert order.venue_order_id == "V7"
    assert order.status is OrderStatus.ACCEPTED  # no false REJECTED
    assert order.origin is OrderOrigin.FRAMEWORK  # the strategy's order, not a twin
    assert len(state.get_orders()) == 1
    assert state.get_order_by_venue_id("V7") is order


def test_unacked_order_past_grace_matched_by_cid_not_terminalized():
    # F11 regression, past-grace flavor: the cid match must keep the order out of the
    # missing-from-snapshot terminalization even though its venue id never matched.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("qubx_BTCUSDT_8", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", instrument=inst))
    snap_order = _order("qubx_BTCUSDT_8", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V8", instrument=inst)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_active_order("qubx_BTCUSDT_8")
    assert order is not None
    assert order.status is OrderStatus.ACCEPTED
    assert order.venue_order_id == "V8"
    assert len(state.get_orders()) == 1


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
    assert state.get_transition_counts()[OrderStatus.FILLED.value] == 1
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


def test_late_deal_already_counted_by_snapshot_not_double_booked():
    # F31 split-stream regression: the snapshot raises filled_quantity (its position/
    # balance legs already incorporate the execution), then the late DealEvent for the
    # SAME execution arrives. The trade id must be recorded for dedup, but the deal must
    # NOT re-book position/balance; a genuinely-new deal after the snapshot still books.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", free=1000.0, total=1000.0))
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))

    snap_order = _order("cid-1", OrderStatus.PARTIALLY_FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 0.4
    snap_order.avg_fill_price = 50_000.0
    snap_pos = Position(instrument=inst, quantity=0.4, pos_average_price=50_000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order], positions=[snap_pos]))
    assert state.get_position(inst).quantity == 0.4

    late_deal = Deal(
        trade_id="t1",
        order_id="V1",
        time=np.datetime64("2026-05-28T00:59:00"),  # at/before the snapshot as_of -> already counted
        amount=0.4,
        price=50_000.0,
        aggressive=True,
        fee_amount=1.0,
    )
    result = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=late_deal))
    assert result.deal is None and result.position is None
    assert state.get_position(inst).quantity == 0.4  # not double-booked
    assert state.get_balance("USDT").total == 1000.0  # no fee/pnl re-applied
    assert state.get_order("cid-1").filled_quantity == 0.4  # not pushed past the snapshot figure

    # genuinely-new execution after the snapshot books normally
    new_deal = Deal(
        trade_id="t2",
        order_id="V1",
        time=np.datetime64("2026-05-28T01:00:01"),
        amount=0.3,
        price=50_000.0,
        aggressive=True,
    )
    result = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=new_deal))
    assert result.deal is new_deal
    assert state.get_position(inst).quantity == 0.7
    assert state.get_order("cid-1").filled_quantity == 0.7

    # the skipped trade id WAS recorded: a re-delivery embedded on the order stream dedups
    result = am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=late_deal))
    assert result.deal is None
    assert state.get_position(inst).quantity == 0.7
    assert state.get_order("cid-1").filled_quantity == 0.7


def test_late_deal_for_snapshot_materialized_order_not_double_booked():
    # Same window, materialize flavor: an order first seen via snapshot with fills already
    # counted (position leg included) must not re-book when its late DealEvent arrives.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    snap_order = _order("manual-1", OrderStatus.PARTIALLY_FILLED, "2026-05-28T00:00:00", vid="VX", instrument=inst)
    snap_order.filled_quantity = 0.4
    snap_pos = Position(instrument=inst, quantity=0.4, pos_average_price=50_000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order], positions=[snap_pos]))
    assert state.get_order_by_venue_id("VX").filled_quantity == 0.4

    late_deal = Deal(
        trade_id="t1",
        order_id="VX",
        time=np.datetime64("2026-05-28T00:59:00"),
        amount=0.4,
        price=50_000.0,
        aggressive=True,
    )
    result = am.apply(DealEvent(instrument=inst, client_order_id="manual-1", venue_order_id="VX", deal=late_deal))
    assert result.deal is None and result.position is None
    assert state.get_position(inst).quantity == 0.4
    assert state.get_order_by_venue_id("VX").filled_quantity == 0.4


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


def test_identical_size_snapshot_preserves_position_accounting():
    # F1 regression: the periodic snapshot must NOT wipe locally accumulated accounting.
    # Snapshot legs are built from venue data with r_pnl/commissions/funding defaulting to
    # zero — routing them through reset_by_position zeroed the local ledger every ~30s.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0, r_pnl=123.45, cumulative_funding=9.99)
    pos.commissions = 6.78
    state.set_position(inst, pos)

    snap_pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", positions=[snap_pos]))

    assert state.get_position(inst) is pos
    assert pos.r_pnl == 123.45
    assert pos.commissions == 6.78
    assert pos.cumulative_funding == 9.99
    # unchanged size/avg-price -> not reported in the reconcile diff
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.positions == []


def test_size_drift_snapshot_corrects_size_but_preserves_accounting():
    # The snapshot is authoritative for size/avg-price; the local ledger
    # (r_pnl/commissions/funding) is ours and must survive the correction.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0, r_pnl=123.45, cumulative_funding=9.99)
    pos.commissions = 6.78
    state.set_position(inst, pos)

    snap_pos = Position(instrument=inst, quantity=2.0, pos_average_price=49_000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", positions=[snap_pos]))

    assert state.get_position(inst) is pos
    assert pos.quantity == 2.0
    assert pos.position_avg_price == 49_000.0
    assert pos.r_pnl == 123.45
    assert pos.commissions == 6.78
    assert pos.cumulative_funding == 9.99
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.positions == [snap_pos]


def test_snapshot_margin_and_mark_refresh_even_when_size_unchanged():
    # The changed predicate gates only the size/avg-price correction and the diff entry;
    # venue margin and mark keep moving regardless and must always refresh.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0, r_pnl=100.0)
    state.set_position(inst, pos)

    snap_pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    snap_pos.update_market_price(np.datetime64("2026-05-28T01:00:00"), 51_000.0, 1)
    snap_pos.set_external_maint_margin(12.5)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", positions=[snap_pos]))

    assert state.get_position(inst) is pos
    assert pos.maint_margin == 12.5
    assert pos.last_update_price == 51_000.0
    assert pos.pnl == 1_000.0 + 100.0  # re-marked: unrealized at the new mark + preserved r_pnl
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.positions == []


def test_restored_positions_first_snapshot_updates_present_keeps_absent():
    # F63: restoration seeds positions (with persisted accounting) via set_position before
    # the first venue snapshot arrives. The snapshot reconciles surgically: the position it
    # carries is corrected in place (identity + restored accounting preserved per F1); a
    # restored position the snapshot omits is left entirely untouched.
    am = _am()
    state = am._states["binance"]
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    btc_pos = Position(instrument=btc, quantity=1.0, pos_average_price=50_000.0, r_pnl=11.0, cumulative_funding=2.5)
    btc_pos.commissions = 3.0
    eth_pos = Position(instrument=eth, quantity=5.0, pos_average_price=3_000.0, r_pnl=7.0)
    state.set_position(btc, btc_pos)
    state.set_position(eth, eth_pos)

    snap_pos = Position(instrument=btc, quantity=2.0, pos_average_price=49_500.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", positions=[snap_pos]))

    assert state.get_position(btc) is btc_pos
    assert btc_pos.quantity == 2.0
    assert btc_pos.position_avg_price == 49_500.0
    assert btc_pos.r_pnl == 11.0
    assert btc_pos.commissions == 3.0
    assert btc_pos.cumulative_funding == 2.5
    assert state.get_position(eth) is eth_pos
    assert eth_pos.quantity == 5.0
    assert eth_pos.position_avg_price == 3_000.0
    assert eth_pos.r_pnl == 7.0
    assert result.reconcile_diff is not None
    assert result.reconcile_diff.positions == [snap_pos]


def test_reconcile_applies_to_state():
    # The one combined-integration case: every facet (terminalize / materialize / position /
    # balance) is pinned individually above; this pins them landing together in ONE snapshot.
    # Reconcile mutates AccountState silently (no per-event callback) — the PM fires
    # on_position_change per corrected position off the diff. Here we assert the state effects.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # one missing with spent fetch budget (-> terminal), one unknown (-> materialized),
    # plus position and balance
    state.add_order(_order("cid-gone", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="VG", instrument=inst))
    _exhaust_fetch_budget(state, "cid-gone")
    snap_order = _order(
        "manual-9",
        OrderStatus.ACCEPTED,
        "2026-05-28T00:00:00",
        vid="VNEW",
        origin=OrderOrigin.EXTERNAL,
        instrument=inst,
    )
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


def test_reconcile_termination_logs_warning_with_cid():
    # F8: a reconcile that terminalized an order is operator-visible drift — the applied
    # diff must leave a WARNING carrying the cid (message prose deliberately not pinned).
    am = _am()
    state = am._states["binance"]
    state.add_order(
        _order("cid-gone", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="V1", instrument=_instrument())
    )
    _exhaust_fetch_budget(state, "cid-gone")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[]))
    finally:
        logger.remove(sink)
    assert any("cid-gone" in m for m in warnings)


def test_benign_reconcile_diff_logs_info_not_warning():
    # Drift without terminalized/materialized/missing orders (here: a balance correction)
    # logs the applied diff at INFO; nothing escalates to WARNING.
    am = _am()
    snap_bal = Balance(exchange="binance", currency="USDT", free=900.0, locked=100.0, total=1000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    records = []
    sink = logger.add(lambda m: records.append(m.record), level="INFO")
    try:
        am.apply(_snap_event(as_of="2026-05-28T01:00:00", balances=[snap_bal]))
    finally:
        logger.remove(sink)
    assert not [r for r in records if r["level"].name == "WARNING"]
    assert any("reconcile" in r["message"] for r in records if r["level"].name == "INFO")
