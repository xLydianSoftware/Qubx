from unittest.mock import MagicMock

import numpy as np

from qubx import logger
from qubx.core.account_manager import AccountManager, AccountManagerConfig
from qubx.core.basics import Order, OrderChange, OrderOrigin, OrderStatus
from qubx.core.mixins.processing import ProcessingManager
from tests.qubx.core.conftest import make_pm


class _T:
    def __init__(self):
        self.t = np.datetime64("2026-05-28T00:00:00")

    def time(self):
        return self.t

    def adv(self, ms):
        self.t = self.t + np.timedelta64(ms, "ms")


def _real_pm(am: AccountManager) -> ProcessingManager:
    # A REAL ProcessingManager wired to the real AM — the give-up path must exercise the
    # genuine process_event -> apply -> _safe_call dispatch (error isolation included),
    # not a mock that would skip the apply.
    return make_pm(_account_manager=am)


def _am(connectors, cfg=None):
    am = AccountManager(
        connectors=connectors,
        base_currencies={ex: "USDT" for ex in connectors},
        time=_T(),
        cfg=cfg or AccountManagerConfig(inflight_check_threshold_ms=5_000, inflight_check_retries=3),
        account_id="test",
    )
    # assigned post-construction (not via set_processing_manager): the half-object PM
    # has no scheduler, and these tests drive the ticks directly.
    am._pm = _real_pm(am)
    return am


def _mk_order(cid, status, vid):
    return Order(
        client_order_id=cid,
        venue_order_id=vid,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        submitted_at=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
    )


def test_inflight_tick_requests_status_and_bumps_retry():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"].add_order(_mk_order("cid-1", OrderStatus.SUBMITTED, "V1"))
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    conn.request_order_status.assert_called_once_with(
        client_order_id="cid-1",
        venue_order_id="V1",
        instrument=None,
    )
    assert am._states["binance"].get_retry("cid-1") == 1


def test_inflight_tick_no_action_within_threshold():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"].add_order(_mk_order("cid-1", OrderStatus.SUBMITTED, "V1"))
    am._time.adv(1_000)
    am._on_inflight_tick(None)
    conn.request_order_status.assert_not_called()


def _exhaust_retries(state, cid, n=3):
    for _ in range(n):
        state.bump_retry(cid)


def test_inflight_exhausted_submitted_transitions_rejected():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"].add_order(_mk_order("cid-1", OrderStatus.SUBMITTED, None))
    _exhaust_retries(am._states["binance"], "cid-1")
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.REJECTED
    assert o.rejected_reason == "reconcile: no venue ack after 3 retries"
    # the synthetic reconcile reject carries no venue code -> error_code stays None
    assert o.error_code is None
    # the synthetic reject reaches the strategy through the PM dispatch
    am._pm._strategy.on_order.assert_called_once()
    assert am._pm._strategy.on_order.call_args.args[2] is OrderChange.REJECTED


def test_inflight_exhausted_pending_cancel_reverts_and_fires_callback():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"].add_order(_mk_order("cid-1", OrderStatus.ACCEPTED, "V1"))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    _exhaust_retries(am._states["binance"], "cid-1")
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert am._states["binance"].get_pre_pending("cid-1") is None
    # callback fired through the PM dispatch (ctx-first signature), not by the AM directly;
    # the synthetic give-up reason is PM-log-only on cancel/update rejects
    am._pm._strategy.on_order.assert_called_once()
    assert am._pm._strategy.on_order.call_args.args[0] is am._pm._context
    assert am._pm._strategy.on_order.call_args.args[2] is OrderChange.CANCEL_REJECTED


def test_inflight_exhausted_pending_update_reverts_and_fires_callback():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"].add_order(_mk_order("cid-1", OrderStatus.PARTIALLY_FILLED, "V1"))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    _exhaust_retries(am._states["binance"], "cid-1")
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PARTIALLY_FILLED
    am._pm._strategy.on_order.assert_called_once()
    assert am._pm._strategy.on_order.call_args.args[0] is am._pm._context
    assert am._pm._strategy.on_order.call_args.args[2] is OrderChange.UPDATE_REJECTED


def test_inflight_giveup_logs_warning_with_cid():
    # F8: the give-up is an operator-visible event — it must leave a WARNING carrying the
    # cid (message prose deliberately not pinned).
    conn = MagicMock()
    am = _am({"binance": conn})
    am._states["binance"].add_order(_mk_order("cid-giveup", OrderStatus.SUBMITTED, None))
    _exhaust_retries(am._states["binance"], "cid-giveup")
    am._time.adv(6_000)
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        am._on_inflight_tick(None)
    finally:
        logger.remove(sink)
    assert any("cid-giveup" in m for m in messages)


def test_retry_budget_resets_on_status_change():
    # Retries consumed while SUBMITTED must NOT deplete a later PENDING_CANCEL sweep
    # budget: transition_order resets the counter on every status change.
    conn = MagicMock()
    am = _am({"binance": conn})
    state = am._states["binance"]
    state.add_order(_mk_order("cid-1", OrderStatus.SUBMITTED, "V1"))
    _exhaust_retries(state, "cid-1")
    assert state.get_retry("cid-1") == 3
    am.transition_order("binance", "cid-1", OrderStatus.ACCEPTED)
    assert state.get_retry("cid-1") == 0
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am._time.adv(6_000)
    am._on_inflight_tick(None)
    # fresh budget: the sweep polls the venue instead of giving up and reverting
    conn.request_order_status.assert_called_once_with(client_order_id="cid-1", venue_order_id="V1", instrument=None)
    assert state.get_order("cid-1").status is OrderStatus.PENDING_CANCEL
    assert state.get_retry("cid-1") == 1


def test_snapshot_tick_calls_request_snapshot_when_stale():
    conn = MagicMock()
    am = _am({"binance": conn})
    am._on_snapshot_tick(None)
    conn.request_snapshot.assert_called_once()


def test_snapshot_tick_skips_when_fresh():
    conn = MagicMock()
    am = _am({"binance": conn}, cfg=AccountManagerConfig(snapshot_check_interval_ms=30_000))
    am._states["binance"].mark_snapshot_applied(am._time.time())
    am._on_snapshot_tick(None)
    conn.request_snapshot.assert_not_called()


def test_liveness_tick_forces_reconnect_after_threshold():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    conn.reconnect.assert_not_called()
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    conn.reconnect.assert_called_once()


def test_liveness_tick_resets_when_ws_recovers():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    # WS recovers before threshold -> unready timer cleared
    conn.is_ws_ready.return_value = True
    am._time.adv(3_000)
    am._on_liveness_tick(None)
    assert "binance" not in am._liveness_unready_since
    conn.reconnect.assert_not_called()


def test_liveness_tick_retries_when_reconnect_fails():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    # Failed reconnect keeps the timestamp -> the very next tick retries without
    # waiting out the full threshold again.
    assert "binance" in am._liveness_unready_since
    am._time.adv(1_000)
    am._on_liveness_tick(None)
    assert conn.reconnect.call_count == 2


def test_liveness_tick_retries_when_reconnect_raises():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.side_effect = RuntimeError("boom")
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    assert "binance" in am._liveness_unready_since
    am._time.adv(1_000)
    am._on_liveness_tick(None)
    assert conn.reconnect.call_count == 2


def test_liveness_tick_clears_timestamp_on_successful_reconnect():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = True
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick(None)
    am._time.adv(6_000)
    am._on_liveness_tick(None)
    assert "binance" not in am._liveness_unready_since


def test_init_registers_three_ticks_via_pm_schedule():
    pm = MagicMock()
    conn = MagicMock()
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T())
    # one schedule call per enabled tick (inflight, snapshot, liveness)
    assert pm.schedule.call_count == 3
    scheduled = {call.args[1] for call in pm.schedule.call_args_list}
    assert am._on_inflight_tick in scheduled
    assert am._on_snapshot_tick in scheduled
    assert am._on_liveness_tick in scheduled


def test_init_skips_disabled_ticks():
    pm = MagicMock()
    conn = MagicMock()
    cfg = AccountManagerConfig(
        inflight_check_interval_ms=0,
        snapshot_check_interval_ms=0,
        liveness_check_interval_ms=5_000,
    )
    AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T(), cfg=cfg)
    assert pm.schedule.call_count == 1


def test_am_holds_no_strategy_reference():
    # I2 regression: the AM must never hold a strategy — all callbacks route through the PM.
    am = AccountManager(
        pm=MagicMock(), connectors={"binance": MagicMock()}, base_currencies={"binance": "USDT"}, time=_T()
    )
    assert not hasattr(am, "_strategy")
    assert not hasattr(am, "_ctx")


def test_inflight_sweep_isolates_raising_callback():
    # A raising strategy callback (or connector error) on one order must not abort the
    # rest of the sweep — design §1260 "one bad callback never blocks the next". The
    # isolation now lives in the PM's _safe_call, which the give-up path routes through.
    conn = MagicMock()
    am = _am({"binance": conn})
    am._pm._strategy.on_order.side_effect = RuntimeError("boom")
    state = am._states["binance"]

    a = _mk_order("cid-a", OrderStatus.ACCEPTED, "VA")
    state.add_order(a)
    am.transition_order("binance", "cid-a", OrderStatus.PENDING_CANCEL)
    _exhaust_retries(state, "cid-a")  # exhausted -> revert + (raising) callback
    b = _mk_order("cid-b", OrderStatus.SUBMITTED, "VB")  # retries left
    state.add_order(b)

    am._time.adv(6_000)
    am._on_inflight_tick(None)  # must not raise

    # order B still processed despite A's callback raising
    conn.request_order_status.assert_any_call(client_order_id="cid-b", venue_order_id="VB", instrument=None)
    # A reverted out of PENDING_CANCEL to its captured pre-pending status
    assert state.get_order("cid-a").status is OrderStatus.ACCEPTED
