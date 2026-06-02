from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig, AccountState
from qubx.core.basics import Order, OrderOrigin, OrderStatus


class _T:
    def __init__(self):
        self.t = np.datetime64("2026-05-28T00:00:00")

    def time(self):
        return self.t

    def adv(self, ms):
        self.t = self.t + np.timedelta64(ms, "ms")


def _am(connectors):
    am = AccountManager.__new__(AccountManager)
    am._states = {ex: AccountState(exchange=ex) for ex in connectors}
    am._connectors = connectors
    am._cfg = AccountManagerConfig(inflight_check_threshold_ms=5_000, inflight_check_retries=3)
    am._time = _T()
    am._strategy = MagicMock()
    am._liveness_unready_since = {}
    am._applied_funding_buckets = {}
    am._ctx = object()
    return am


def test_inflight_tick_calls_request_order_status():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"]._add_order(
        Order(
            client_order_id="cid-1",
            venue_order_id="V1",
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=None,
            time=np.datetime64("2026-05-28T00:00:00"),
            quantity=1.0,
            price=50_000.0,
            side="BUY",
            status=OrderStatus.SUBMITTED,
            time_in_force="gtc",
        )
    )
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    conn.request_order_status.assert_called_once_with(
        client_order_id="cid-1",
        venue_order_id="V1",
    )


def test_inflight_tick_increments_retry_counter():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"]._add_order(
        Order(
            client_order_id="cid-1",
            venue_order_id="V1",
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=None,
            time=np.datetime64("2026-05-28T00:00:00"),
            quantity=1.0,
            price=50_000.0,
            side="BUY",
            status=OrderStatus.SUBMITTED,
            time_in_force="gtc",
        )
    )
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    assert am._states["binance"].get_order("cid-1").retry_count == 1


def test_inflight_tick_no_action_within_threshold():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"]._add_order(
        Order(
            client_order_id="cid-1",
            venue_order_id="V1",
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=None,
            time=np.datetime64("2026-05-28T00:00:00"),
            quantity=1.0,
            price=50_000.0,
            side="BUY",
            status=OrderStatus.SUBMITTED,
            time_in_force="gtc",
        )
    )
    am._time.adv(1_000)
    am._on_inflight_tick(None)
    conn.request_order_status.assert_not_called()


def test_inflight_exhausted_submitted_transitions_rejected():
    conn = MagicMock()
    am = _am({"binance": conn})
    order = Order(
        client_order_id="cid-1",
        venue_order_id=None,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=OrderStatus.SUBMITTED,
        time_in_force="gtc",
        retry_count=3,
    )
    am._states["binance"]._add_order(order)
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    assert am._states["binance"].get_order("cid-1").status is OrderStatus.REJECTED


def test_inflight_exhausted_pending_cancel_reverts_and_fires_callback():
    conn = MagicMock()
    am = _am({"binance": conn})
    order = Order(
        client_order_id="cid-1",
        venue_order_id="V1",
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=OrderStatus.PENDING_CANCEL,
        time_in_force="gtc",
        retry_count=3,
        pre_pending_status=OrderStatus.ACCEPTED,
    )
    am._states["binance"]._add_order(order)
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.pre_pending_status is None
    am._strategy.on_order_cancel_rejected.assert_called_once()
    assert am._strategy.on_order_cancel_rejected.call_args.args[0] is am._ctx


def test_inflight_exhausted_pending_update_reverts_and_fires_callback():
    conn = MagicMock()
    am = _am({"binance": conn})
    order = Order(
        client_order_id="cid-1",
        venue_order_id="V1",
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=OrderStatus.PENDING_UPDATE,
        time_in_force="gtc",
        retry_count=3,
        pre_pending_status=OrderStatus.PARTIALLY_FILLED,
    )
    am._states["binance"]._add_order(order)
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PARTIALLY_FILLED
    am._strategy.on_order_update_rejected.assert_called_once()
    assert am._strategy.on_order_update_rejected.call_args.args[0] is am._ctx


def test_snapshot_tick_calls_request_snapshot_when_stale():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._on_snapshot_tick(None)
    conn.request_snapshot.assert_called_once()


def test_snapshot_tick_skips_when_fresh():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._cfg.snapshot_check_interval_ms = 30_000
    am._states["binance"]._last_snapshot_as_of = am._time.time()
    am._on_snapshot_tick(None)
    conn.request_snapshot.assert_not_called()


def test_liveness_tick_forces_reconnect_after_threshold():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn})
    am._cfg.liveness_check_threshold_ms = 5_000
    am._on_liveness_tick(None)
    conn.force_ws_reconnect_sync.assert_not_called()
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    conn.force_ws_reconnect_sync.assert_called_once()


def test_liveness_tick_resets_when_ws_recovers():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn})
    am._cfg.liveness_check_threshold_ms = 5_000
    am._on_liveness_tick(None)
    # WS recovers before threshold -> unready timer cleared
    conn.is_ws_ready.return_value = True
    am._time.adv(3_000)
    am._on_liveness_tick(None)
    assert "binance" not in am._liveness_unready_since
    conn.force_ws_reconnect_sync.assert_not_called()


def test_init_registers_three_ticks_via_pm_schedule():
    pm = MagicMock()
    conn = MagicMock()
    strategy = MagicMock()
    am = AccountManager(pm=pm, connectors={"binance": conn}, strategy=strategy, time=_T())
    # one schedule call per enabled tick (inflight, snapshot, liveness)
    assert pm.schedule.call_count == 3
    scheduled = {call.args[1] for call in pm.schedule.call_args_list}
    assert am._on_inflight_tick in scheduled
    assert am._on_snapshot_tick in scheduled
    assert am._on_liveness_tick in scheduled


def test_init_skips_disabled_ticks():
    pm = MagicMock()
    conn = MagicMock()
    strategy = MagicMock()
    cfg = AccountManagerConfig(
        inflight_check_interval_ms=0,
        snapshot_check_interval_ms=0,
        liveness_check_interval_ms=5_000,
    )
    AccountManager(pm=pm, connectors={"binance": conn}, strategy=strategy, time=_T(), cfg=cfg)
    assert pm.schedule.call_count == 1


def _mk_order(cid, status, vid):
    return Order(
        client_order_id=cid,
        venue_order_id=vid,
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


def test_inflight_sweep_isolates_raising_callback():
    # A raising strategy callback (or connector error) on one order must not abort the
    # rest of the sweep — design §1260 "one bad callback never blocks the next".
    conn = MagicMock()
    am = _am({"binance": conn})
    am._strategy.on_order_cancel_rejected.side_effect = RuntimeError("boom")
    state = am._states["binance"]

    a = _mk_order("cid-a", OrderStatus.PENDING_CANCEL, "VA")
    a.retry_count = 3  # exhausted -> revert + (raising) callback
    a.pre_pending_status = OrderStatus.ACCEPTED
    state._add_order(a)
    b = _mk_order("cid-b", OrderStatus.SUBMITTED, "VB")  # retries left
    state._add_order(b)

    am._time.adv(6_000)
    am._on_inflight_tick(None)  # must not raise

    # order B still processed despite A's callback raising
    conn.request_order_status.assert_any_call(client_order_id="cid-b", venue_order_id="VB")
    # A reverted out of PENDING_CANCEL to its captured pre-pending status
    assert state.get_order("cid-a").status is OrderStatus.ACCEPTED
