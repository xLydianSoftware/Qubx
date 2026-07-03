from unittest.mock import MagicMock

import numpy as np

from qubx.connectors.ccxt.utils import ccxt_convert_order_info
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
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
)


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
        cfg=cfg or AccountManagerConfig(snapshot_grace_ms=5_000),
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
    withdrawable=None,
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
            withdrawable=withdrawable,
        ),
    )


def test_open_orders_none_failed_fetch_leg_skips_missing_handling():
    # F16: open_orders=None means the connector's order-fetch leg FAILED — it must never
    # be treated as "venue has no orders". Live orders past grace stay untouched: no fetch
    # rescue, no missing-resolve task spawned. A regression to `if snapshot.open_orders:`
    # flips this to mass-handling.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-fresh", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="VF", instrument=inst))
    state.add_order(_order("cid-old", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="VS", instrument=inst))
    snap_bal = Balance(exchange="binance", currency="USDT", free=1000.0, total=1000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    result = am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=None, balances=[snap_bal]))

    assert state.get_order("cid-fresh").status is OrderStatus.ACCEPTED
    assert state.get_order("cid-old").status is OrderStatus.SUBMITTED
    am._connectors["binance"].request_order_status.assert_not_called()
    assert am._reconcilers["binance"].active_keys() == set()  # None leg -> no order handling, no task
    # only the orders leg is absent — the rest of the snapshot still applies
    assert state.get_balance("USDT").total == 1000.0


def test_stale_snapshot_does_not_clobber_fresh_state():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    existing = _order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    existing.last_update_time = np.datetime64("2026-05-28T02:00:00")
    existing.filled_quantity = 0.5
    state.add_order(existing)
    # snapshot older than the order's last_update_time must not overwrite
    snap_order = _order("cid-1", OrderStatus.FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 1.0
    snap_order.last_update_time = np.datetime64("2026-05-28T01:00:00")
    am._time.t = np.datetime64("2026-05-28T03:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    assert state.get_order("cid-1").filled_quantity == 0.5
    assert state.get_order("cid-1").status is OrderStatus.ACCEPTED


def test_out_of_order_snapshot_skipped():
    # the as_of ratchet: a snapshot at/before the last applied as_of is dropped wholesale.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    am._time.t = np.datetime64("2026-05-28T03:00:00")
    am.apply(
        _snap_event(
            as_of="2026-05-28T02:00:00", positions=[Position(instrument=inst, quantity=2.0, pos_average_price=50_000.0)]
        )
    )
    assert state.get_position(inst).quantity == 2.0
    # an OLDER snapshot must be skipped entirely (no clobber)
    am.apply(
        _snap_event(
            as_of="2026-05-28T01:00:00", positions=[Position(instrument=inst, quantity=9.0, pos_average_price=50_000.0)]
        )
    )
    assert state.get_position(inst).quantity == 2.0


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


def test_unacked_order_past_grace_matched_by_cid_not_terminalized():
    # F11 regression, past-grace flavor: the cid match must keep the order out of the
    # missing-from-snapshot terminalization even though its venue id never matched.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("qubx_BTCUSDT_8", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", instrument=inst))
    snap_order = _order("qubx_BTCUSDT_8", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V8", instrument=inst)
    snap_order.last_update_time = np.datetime64("2026-05-28T01:00:00")
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
    existing.last_update_time = np.datetime64("2026-05-28T00:30:00")
    state.add_order(existing)
    snap_order = _order("cid-1", OrderStatus.PARTIALLY_FILLED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.filled_quantity = 0.4
    snap_order.avg_fill_price = 49_900.0
    snap_order.last_update_time = np.datetime64("2026-05-28T01:00:00")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))
    updated = state.get_order("cid-1")
    assert updated.status is OrderStatus.PARTIALLY_FILLED
    assert updated.filled_quantity == 0.4
    assert updated.avg_fill_price == 49_900.0
    assert updated.last_update_time == np.datetime64("2026-05-28T01:00:00")


def _raw_ccxt_open_order(cid="qubx_BTCUSDT_9", vid="900001"):
    # Realistic Binance UM fetch_open_orders unified order: partially-filled limit SELL.
    return {
        "info": {
            "orderId": vid,
            "symbol": "BTCUSDT",
            "status": "PARTIALLY_FILLED",
            "clientOrderId": cid,
            "price": "50200.00",
            "avgPrice": "50100.00",
            "origQty": "2.000",
            "executedQty": "0.500",
            "side": "SELL",
            "type": "LIMIT",
            "timeInForce": "GTC",
        },
        "id": vid,
        "clientOrderId": cid,
        "symbol": "BTC/USDT:USDT",
        "timestamp": 1_716_854_400_000,
        "type": "limit",
        "timeInForce": "GTC",
        "side": "sell",
        "price": 50_200.0,
        "amount": 2.0,
        "filled": 0.5,
        "remaining": 1.5,
        "average": 50_100.0,
        "status": "open",
        "reduceOnly": False,
    }


def test_raw_ccxt_snapshot_order_keeps_fill_state_through_reconcile():
    # R2 seam regression: a raw ccxt payload routed converter -> AccountSnapshot ->
    # reconcile must not wipe a tracked order's fill state. Main's converter emitted
    # signed quantity and never mapped filled/average, so every periodic snapshot
    # overwrote SELL quantity with -q and reset filled_quantity/avg_fill_price.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    existing = _order(
        "qubx_BTCUSDT_9", OrderStatus.PARTIALLY_FILLED, "2026-05-28T00:59:00", vid="900001", instrument=inst, qty=2.0
    )
    existing.side = "SELL"
    existing.filled_quantity = 0.5
    existing.avg_fill_price = 50_100.0
    state.add_order(existing)

    snap_order = ccxt_convert_order_info(inst, _raw_ccxt_open_order())
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order("qubx_BTCUSDT_9")
    assert order is existing
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.quantity == 2.0  # unsigned, never sign-flipped for SELL
    assert order.filled_quantity == 0.5
    assert order.avg_fill_price == 50_100.0


def test_raw_ccxt_snapshot_order_materializes_with_fill_state():
    # Same seam, materialize path: an untracked partially-filled order from a raw ccxt
    # payload must arrive in AM with real fill state and unsigned quantity.
    am = _am()
    state = am._states["binance"]
    snap_order = ccxt_convert_order_info(_instrument(), _raw_ccxt_open_order())
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order_by_venue_id("900001")
    assert order is not None
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.quantity == 2.0
    assert order.filled_quantity == 0.5
    assert order.avg_fill_price == 50_100.0


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
    snap_order.last_update_time = np.datetime64("2026-05-28T01:00:00")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.FILLED
    # the stale inflight entry is gone -> no pointless venue polling on the next tick
    assert state.get_inflight_orders() == []
    # audit trail + counters recorded the transition
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


def test_snapshot_terminal_status_resolves_pending_marker():
    # A terminal snapshot status IS the venue's resolution of the outstanding request —
    # it falls through the pending guard and terminalizes via the machinery.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.add_order(_order("cid-1", OrderStatus.ACCEPTED, "2026-05-28T00:00:00", vid="V1", instrument=inst))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    snap_order = _order("cid-1", OrderStatus.CANCELED, "2026-05-28T00:00:00", vid="V1", instrument=inst)
    snap_order.last_update_time = np.datetime64("2026-05-28T01:00:00")
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
    snap_order.last_update_time = np.datetime64("2026-05-28T01:00:00")
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(_snap_event(as_of="2026-05-28T01:00:00", open_orders=[snap_order]))

    order = state.get_order("cid-1")
    assert order.status is OrderStatus.ACCEPTED
    assert state.get_inflight_orders() == []


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


def test_snapshot_balance_leg_skipped_when_push_at_least_as_fresh():
    # F26: a currency whose WS push as_of >= snapshot.as_of keeps the push figure.
    # Tie-break favors the push (snapshot as_of is the local fetch clock, push as_of
    # is venue event time). Per-currency: an uncovered currency still applies.
    am = _am()
    state = am._states["binance"]
    state.apply_balance_push("USDT", 1500.0, np.datetime64("2026-05-28T01:00:00"))
    snap_usdt = Balance(exchange="binance", currency="USDT", free=900.0, locked=100.0, total=1000.0)
    snap_btc = Balance(exchange="binance", currency="BTC", free=0.5, locked=0.0, total=0.5)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    r = am.apply(_snap_event(as_of="2026-05-28T01:00:00", balances=[snap_usdt, snap_btc]))
    assert state.get_balance("USDT").total == 1500.0  # push figure stands
    assert state.get_balance("BTC").total == 0.5


def test_snapshot_multi_currency_wallet_kept_capital_counts_base_only():
    # The ignored-assets contract: every snapshot currency lands in get_balances
    # regardless of base; the DERIVED total_capital counts base cash + position
    # market values only (non-base cash excluded — design.md Capital/margin);
    # venue-reported equity, when present, overrides the derivation entirely.
    am = _am()
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    am.apply(
        _snap_event(
            as_of="2026-05-28T01:00:00",
            balances=[
                Balance(exchange="binance", currency="USDT", free=1_000.0, locked=0.0, total=1_000.0),
                Balance(exchange="binance", currency="BTC", free=0.5, locked=0.0, total=0.5),
                Balance(exchange="binance", currency="ETH", free=10.0, locked=0.0, total=10.0),
            ],
        )
    )
    assert {b.currency for b in am.get_balances("binance")} == {"USDT", "BTC", "ETH"}
    assert am.get_total_capital("binance") == 1_000.0

    am._time.t = np.datetime64("2026-05-28T01:01:00")
    am.apply(_snap_event(as_of="2026-05-28T01:01:00", equity=50_000.0))
    assert am.get_total_capital("binance") == 50_000.0


def test_snapshot_balance_leg_applies_when_newer_than_push():
    am = _am()
    state = am._states["binance"]
    state.apply_balance_push("USDT", 1500.0, np.datetime64("2026-05-28T00:59:00"))
    snap_bal = Balance(exchange="binance", currency="USDT", free=900.0, locked=100.0, total=1000.0)
    am._time.t = np.datetime64("2026-05-28T01:00:00")
    r = am.apply(_snap_event(as_of="2026-05-28T01:00:00", balances=[snap_bal]))
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
    # unchanged size/avg-price -> not reported as changed
    assert result.positions == []


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
    assert result.positions == [pos]


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
    assert result.positions == []


def test_restored_positions_first_snapshot_corrects_present_flattens_absent():
    # F63: restoration seeds positions (with persisted accounting) via set_position before
    # the first venue snapshot arrives. An OBSERVED snapshot (positions list, not None) is
    # venue-authoritative: the position it carries is corrected in place (identity + restored
    # accounting preserved per F1); a restored position the snapshot OMITS is flattened
    # (LocalPositionMissing -> settle), keeping its accumulated accounting.
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
    # eth absent from the observed snapshot -> flattened (venue-authoritative), accounting kept
    assert state.get_position(eth) is eth_pos
    assert eth_pos.quantity == 0.0
    assert eth_pos.r_pnl == 7.0
    assert result.positions == [btc_pos, eth_pos]


def test_reconcile_applies_to_state():
    # The one combined-integration case: every facet (missing-resolve / materialize / position /
    # balance) pinned individually above, here landing together in ONE snapshot through the AM.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # one missing past grace (-> ResolveMissingOrder task, NOT blind-terminalized), one unknown
    # (-> recovered), plus position and balance
    state.add_order(_order("cid-gone", OrderStatus.SUBMITTED, "2026-05-28T00:00:00", vid="VG", instrument=inst))
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
    # missing from snapshot -> a ResolveMissingOrder task (waits/fetches), not blind-terminalized
    assert state.get_order("cid-gone").status is OrderStatus.SUBMITTED
    assert "cid-gone" in am._reconcilers["binance"].active_keys()
    # unknown venue id in the snapshot -> recovered into the cache
    assert state.get_order_by_venue_id("VNEW") is not None
    # position / balance overwritten from the snapshot
    assert state.get_position(inst).quantity == 1.0
    assert state.get_balance("USDT").total == 1000.0


def test_snapshot_sets_venue_figures_and_metrics_prefer_them():
    am = _am()
    state = am._states["binance"]
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))
    am.apply(_snap_event(equity=5000.0, available_margin=4000.0, margin_ratio=42.0, withdrawable=3500.0))
    figures = state.get_venue_figures()
    assert figures is not None
    assert figures.equity == 5000.0
    assert figures.available_margin == 4000.0
    assert figures.margin_ratio == 42.0
    assert figures.withdrawable == 3500.0
    assert figures.as_of == np.datetime64("2026-05-28T01:00:00")
    # per-metric prefer-venue over the derived values
    assert am.get_total_capital("binance") == 5000.0
    assert am.get_available_margin("binance") == 4000.0
    assert am.get_margin_ratio("binance") == 42.0
    assert am.get_withdrawable_balance("binance") == 3500.0


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


def test_sim_capital_getters_derive_consistently():
    # Sim sets no venue figures, so the three wallet getters derive from one another:
    # available = total - initial margin, withdrawable = available (documented
    # simplification), and with no margin used all three coincide.
    am = _am()
    state = am._states["binance"]
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))
    am.apply(_snap_event(open_orders=[]))
    assert am.get_total_capital("binance") == 1000.0
    assert am.get_available_margin("binance") == 1000.0
    assert am.get_withdrawable_balance("binance") == 1000.0

    pos = Position(_instrument())
    pos.initial_margin = 100.0
    state.set_position(pos.instrument, pos)
    assert am.get_total_capital("binance") == 1000.0
    assert am.get_available_margin("binance") == 900.0
    assert am.get_withdrawable_balance("binance") == am.get_available_margin("binance") == 900.0
