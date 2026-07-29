import time
from collections import Counter
from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig, SimulatedAccountManager
from qubx.core.account_manager.manager import DEFAULT_ACCOUNT_WATCH_TIMEOUT_S, evaluate_liveness
from qubx.core.interfaces import IHealthMonitor, StreamHealth
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


def _am(connectors, cfg=None, health_monitor=None):
    am = AccountManager(
        connectors=connectors,
        base_currencies={ex: "USDT" for ex in connectors},
        time=_T(),
        cfg=cfg or AccountManagerConfig(missing_order_wait_ms=5_000, missing_order_retries=3),
        account_id="test",
        health_monitor=health_monitor,
    )
    # assigned post-construction (not via set_processing_manager): the half-object PM
    # has no scheduler, and these tests drive the ticks directly. _register_ticks() (and
    # therefore the liveness thread) never runs for these — _on_liveness_tick is called
    # directly instead, so these stay synchronous/deterministic.
    am._pm = _real_pm(am)
    return am


def _fake_health_monitor(ages: dict | None = None, violations: Counter | None = None) -> MagicMock:
    """A MagicMock(spec=IHealthMonitor) so record_stream_violation calls can be asserted
    on, with get_stream_health stubbed to a fixed StreamHealth snapshot (empty ages by
    default -> exempt from the drive-age/violation-burst legs, i.e. ws_ready-only, same as
    the default DummyHealthMonitor the other tests in this file rely on)."""
    hm = MagicMock(spec=IHealthMonitor)
    hm.get_stream_health.return_value = StreamHealth(ages=ages or {}, violations=violations or Counter())
    return hm


# ---- liveness tick: verdict + reconnect ladder (business logic, no thread involved) ---- #


def test_liveness_tick_isolates_raising_ws_check():
    # Same isolation rule for the liveness loop: a raising is_ws_ready on one connector
    # must not skip the health check of the rest.
    bad, good = MagicMock(), MagicMock()
    bad.is_ws_ready.side_effect = RuntimeError("boom")
    good.is_ws_ready.return_value = False
    am = _am({"binance": bad, "kraken": good}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick()  # must not raise
    am._time.adv(6_000)
    am._on_liveness_tick()
    good.reconnect.assert_called_once()


def test_liveness_tick_forces_reconnect_after_threshold():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick()
    conn.reconnect.assert_not_called()
    am._time.adv(6_000)
    am._on_liveness_tick()
    conn.reconnect.assert_called_once()


def test_liveness_tick_resets_when_ws_recovers():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick()
    # WS recovers before threshold -> unready timer cleared
    conn.is_ws_ready.return_value = True
    am._time.adv(3_000)
    am._on_liveness_tick()
    assert "binance" not in am._liveness_unready_since
    conn.reconnect.assert_not_called()


def test_liveness_tick_retries_when_reconnect_fails():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick()
    am._time.adv(6_000)
    am._on_liveness_tick()
    # Failed reconnect keeps the timestamp -> the very next tick retries without
    # waiting out the full threshold again.
    assert "binance" in am._liveness_unready_since
    am._time.adv(1_000)
    am._on_liveness_tick()
    assert conn.reconnect.call_count == 2


def test_liveness_tick_retries_when_reconnect_raises():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.side_effect = RuntimeError("boom")
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick()
    am._time.adv(6_000)
    am._on_liveness_tick()
    assert "binance" in am._liveness_unready_since
    am._time.adv(1_000)
    am._on_liveness_tick()
    assert conn.reconnect.call_count == 2


def test_liveness_tick_clears_timestamp_on_successful_reconnect():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = True
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000))
    am._on_liveness_tick()
    am._time.adv(6_000)
    am._on_liveness_tick()
    assert "binance" not in am._liveness_unready_since


# ---- A.3: verdict driven by StreamHealth (drive-age), not just is_ws_ready ---- #


def test_liveness_tick_reconnects_on_stale_drive_age_even_when_ws_ready():
    conn = MagicMock()
    conn.is_ws_ready.return_value = True
    conn.ACCOUNT_WATCH_TIMEOUT_S = 60.0
    stale_drive_age = 3 * conn.ACCOUNT_WATCH_TIMEOUT_S + 1
    hm = _fake_health_monitor(ages={"executions": (5.0, stale_drive_age)})
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000), health_monitor=hm)
    am._on_liveness_tick()
    conn.reconnect.assert_not_called()  # not yet overdue past the threshold
    am._time.adv(6_000)
    am._on_liveness_tick()
    conn.reconnect.assert_called_once()  # drive_age > 3*T -> unhealthy despite ws_ready=True


def test_liveness_tick_healthy_when_drive_age_within_3x_timeout():
    conn = MagicMock()
    conn.is_ws_ready.return_value = True
    conn.ACCOUNT_WATCH_TIMEOUT_S = 60.0
    hm = _fake_health_monitor(ages={"executions": (5.0, 90.0)})
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000), health_monitor=hm)
    am._time.adv(6_000)
    am._on_liveness_tick()
    conn.reconnect.assert_not_called()


def test_no_registered_streams_is_exempt_from_drive_age_check():
    # ages={} (no connector adoption yet / migration grace) -> judged on ws_ready alone,
    # same as the DummyHealthMonitor-default tests above.
    conn = MagicMock()
    conn.is_ws_ready.return_value = True
    hm = _fake_health_monitor(ages={})
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000), health_monitor=hm)
    am._time.adv(6_000)
    am._on_liveness_tick()
    conn.reconnect.assert_not_called()


def test_missing_account_watch_timeout_attribute_falls_back_to_default():
    # spec=[...] restricts attribute access — MagicMock's usual "any attribute exists"
    # behavior would otherwise mask a missing ACCOUNT_WATCH_TIMEOUT_S entirely.
    conn = MagicMock(spec=["is_ws_ready", "reconnect"])
    conn.is_ws_ready.return_value = True
    conn.reconnect.return_value = True
    stale_drive_age = 3 * DEFAULT_ACCOUNT_WATCH_TIMEOUT_S + 1
    hm = _fake_health_monitor(ages={"executions": (0.0, stale_drive_age)})
    am = _am({"binance": conn}, cfg=AccountManagerConfig(liveness_check_threshold_ms=5_000), health_monitor=hm)
    am._on_liveness_tick()  # seeds _liveness_unready_since
    am._time.adv(6_000)
    am._on_liveness_tick()
    conn.reconnect.assert_called_once()


# ---- A.3: escalation ladder (WARNING+reconnect -> ERROR after N further cycles) ---- #


def test_liveness_does_not_escalate_before_configured_cycles():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    hm = _fake_health_monitor()
    cfg = AccountManagerConfig(liveness_check_threshold_ms=5_000, liveness_escalation_cycles=3)
    am = _am({"binance": conn}, cfg=cfg, health_monitor=hm)
    am._on_liveness_tick()  # seeds _liveness_unready_since (not yet overdue)
    am._time.adv(6_000)
    am._on_liveness_tick()  # cycle 1: overdue -> WARNING + reconnect (fails)
    am._time.adv(1_000)
    am._on_liveness_tick()  # cycle 2
    hm.record_stream_violation.assert_not_called()


def test_liveness_escalates_after_configured_cycles_of_failed_reconnect():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    hm = _fake_health_monitor()
    cfg = AccountManagerConfig(liveness_check_threshold_ms=5_000, liveness_escalation_cycles=2)
    am = _am({"binance": conn}, cfg=cfg, health_monitor=hm)
    am._on_liveness_tick()  # seeds _liveness_unready_since (not yet overdue)
    am._time.adv(6_000)
    am._on_liveness_tick()  # cycle 1: WARNING + reconnect (fails) -> not escalated yet
    hm.record_stream_violation.assert_not_called()
    am._time.adv(1_000)
    am._on_liveness_tick()  # cycle 2 >= 2 -> escalate
    hm.record_stream_violation.assert_called_once_with("binance", "executions", "liveness_escalation")


def test_liveness_escalation_raising_reconnect_also_counts_toward_cycles():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.side_effect = RuntimeError("boom")
    hm = _fake_health_monitor()
    cfg = AccountManagerConfig(liveness_check_threshold_ms=5_000, liveness_escalation_cycles=2)
    am = _am({"binance": conn}, cfg=cfg, health_monitor=hm)
    am._on_liveness_tick()  # seeds _liveness_unready_since (not yet overdue)
    am._time.adv(6_000)
    am._on_liveness_tick()  # cycle 1 (reconnect raises)
    am._time.adv(1_000)
    am._on_liveness_tick()  # cycle 2 (reconnect raises again) -> escalate
    hm.record_stream_violation.assert_called_once_with("binance", "executions", "liveness_escalation")


def test_liveness_escalates_only_once_per_unhealthy_episode():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    hm = _fake_health_monitor()
    cfg = AccountManagerConfig(liveness_check_threshold_ms=5_000, liveness_escalation_cycles=1)
    am = _am({"binance": conn}, cfg=cfg, health_monitor=hm)
    am._on_liveness_tick()  # seeds _liveness_unready_since (not yet overdue)
    am._time.adv(6_000)
    am._on_liveness_tick()  # cycle 1 >= 1 -> escalate
    am._time.adv(1_000)
    am._on_liveness_tick()  # still unhealthy -> must NOT escalate again
    assert hm.record_stream_violation.call_count == 1


def test_liveness_escalation_resets_when_healthy_then_can_fire_again():
    conn = MagicMock()
    conn.is_ws_ready.return_value = False
    conn.reconnect.return_value = False
    hm = _fake_health_monitor()
    cfg = AccountManagerConfig(liveness_check_threshold_ms=5_000, liveness_escalation_cycles=1)
    am = _am({"binance": conn}, cfg=cfg, health_monitor=hm)
    am._on_liveness_tick()  # seeds _liveness_unready_since (not yet overdue)
    am._time.adv(6_000)
    am._on_liveness_tick()  # escalate
    assert hm.record_stream_violation.call_count == 1

    conn.is_ws_ready.return_value = True
    am._on_liveness_tick()  # healthy -> ladder reset
    assert "binance" not in am._liveness_escalated
    assert "binance" not in am._liveness_cycles_unhealthy

    conn.is_ws_ready.return_value = False
    am._on_liveness_tick()  # re-seeds _liveness_unready_since (not yet overdue)
    am._time.adv(6_000)
    am._on_liveness_tick()  # unhealthy again -> escalates again, not suppressed by stale state
    assert hm.record_stream_violation.call_count == 2


# ---- thread lifecycle: start/stop with the manager's lifecycle (A.3) ---- #


def test_init_registers_reconcile_tick_via_pm_schedule_and_starts_liveness_thread():
    pm = MagicMock()
    conn = MagicMock()
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T())
    try:
        # Only the reconcile heartbeat rides the channel/pm now — liveness moved off it (A.3).
        assert pm.schedule.call_count == 1
        assert am._on_reconcile_tick in {call.args[1] for call in pm.schedule.call_args_list}
        assert am._liveness_thread is not None
        assert am._liveness_thread.is_alive()
        assert am._liveness_thread.daemon
    finally:
        am.stop()


def test_init_skips_reconcile_schedule_when_disabled_but_still_starts_liveness_thread():
    pm = MagicMock()
    conn = MagicMock()
    cfg = AccountManagerConfig(reconcile_tick_interval_ms=0, liveness_check_interval_ms=5_000)
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T(), cfg=cfg)
    try:
        assert pm.schedule.call_count == 0
        assert am._liveness_thread is not None
    finally:
        am.stop()


def test_init_skips_liveness_thread_when_disabled():
    pm = MagicMock()
    conn = MagicMock()
    cfg = AccountManagerConfig(liveness_check_interval_ms=0)
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T(), cfg=cfg)
    try:
        assert am._liveness_thread is None
    finally:
        am.stop()


def test_stop_interrupts_wait_immediately_even_with_long_interval():
    # A long interval must not make stop() slow — Event.wait(timeout) returns as soon as
    # the event is set, it doesn't sleep out the full timeout.
    pm = MagicMock()
    conn = MagicMock()
    cfg = AccountManagerConfig(liveness_check_interval_ms=60_000)
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T(), cfg=cfg)
    t0 = time.monotonic()
    am.stop()
    assert time.monotonic() - t0 < 2.0
    assert am._liveness_thread is None


def test_stop_is_noop_when_never_started():
    am = AccountManager(connectors={"binance": MagicMock()}, base_currencies={"binance": "USDT"}, time=_T())
    am.stop()  # must not raise
    assert am._liveness_thread is None


def test_liveness_watchdog_thread_actually_ticks_the_connector():
    pm = MagicMock()
    conn = MagicMock()
    conn.is_ws_ready.return_value = True
    cfg = AccountManagerConfig(liveness_check_interval_ms=10)  # fast, so the test doesn't stall
    am = AccountManager(pm=pm, connectors={"binance": conn}, base_currencies={"binance": "USDT"}, time=_T(), cfg=cfg)
    try:
        deadline = time.monotonic() + 5.0
        while conn.is_ws_ready.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert conn.is_ws_ready.call_count > 0
    finally:
        am.stop()


def test_simulated_account_manager_never_starts_liveness_thread():
    am = SimulatedAccountManager(connectors={"binance": MagicMock()}, base_currencies={"binance": "USDT"}, time=_T())
    am.set_processing_manager(MagicMock())  # no-op override — backtest/paper never ticks
    assert am._liveness_thread is None
    am.stop()  # must not raise even though nothing ever started


# ---- pure verdict math (evaluate_liveness) ---- #


def test_evaluate_liveness_ws_not_ready_is_unhealthy_regardless_of_streams():
    v = evaluate_liveness(
        ws_ready=False,
        stream_health=StreamHealth(ages={}, violations=Counter()),
        account_watch_timeout_s=60.0,
        violations_delta=0,
        violation_burst_threshold=0,
    )
    assert v.unhealthy
    assert v.reason == "ws_not_ready"


def test_evaluate_liveness_no_registered_streams_is_exempt():
    # ws_ready True + empty ages -> healthy even though a violation delta that would
    # otherwise trip the burst check is present — ages={} means "not judged at all".
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=StreamHealth(ages={}, violations=Counter()),
        account_watch_timeout_s=60.0,
        violations_delta=999,
        violation_burst_threshold=1,
    )
    assert not v.unhealthy
    assert v.reason is None


def test_evaluate_liveness_drive_age_over_3x_timeout_is_unhealthy():
    health = StreamHealth(ages={"executions": (10.0, 181.0)}, violations=Counter())
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=health,
        account_watch_timeout_s=60.0,
        violations_delta=0,
        violation_burst_threshold=0,
    )
    assert v.unhealthy
    assert v.reason == "stream_drive_stale"


def test_evaluate_liveness_drive_age_at_exactly_3x_timeout_is_healthy():
    health = StreamHealth(ages={"executions": (10.0, 180.0)}, violations=Counter())
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=health,
        account_watch_timeout_s=60.0,
        violations_delta=0,
        violation_burst_threshold=0,
    )
    assert not v.unhealthy


def test_evaluate_liveness_max_drive_age_taken_across_streams():
    health = StreamHealth(ages={"executions": (0.0, 10.0), "balance": (0.0, 500.0)}, violations=Counter())
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=health,
        account_watch_timeout_s=60.0,
        violations_delta=0,
        violation_burst_threshold=0,
    )
    assert v.unhealthy
    assert v.reason == "stream_drive_stale"


def test_evaluate_liveness_violation_burst_disabled_by_default_never_trips():
    # DEFAULT DISABLED per reviewer decision: the plumbing (violations_delta) is wired,
    # but threshold<=0 means the OR-clause never fires.
    health = StreamHealth(ages={"executions": (0.0, 0.0)}, violations=Counter({"executions": 999}))
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=health,
        account_watch_timeout_s=60.0,
        violations_delta=999,
        violation_burst_threshold=0,
    )
    assert not v.unhealthy


def test_evaluate_liveness_violation_burst_trips_when_armed():
    health = StreamHealth(ages={"executions": (0.0, 0.0)}, violations=Counter({"executions": 5}))
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=health,
        account_watch_timeout_s=60.0,
        violations_delta=5,
        violation_burst_threshold=5,
    )
    assert v.unhealthy
    assert v.reason == "violation_burst"


def test_evaluate_liveness_violation_burst_below_threshold_is_healthy():
    health = StreamHealth(ages={"executions": (0.0, 0.0)}, violations=Counter({"executions": 4}))
    v = evaluate_liveness(
        ws_ready=True,
        stream_health=health,
        account_watch_timeout_s=60.0,
        violations_delta=4,
        violation_burst_threshold=5,
    )
    assert not v.unhealthy


def test_am_holds_no_strategy_reference():
    # I2 regression: the AM must never hold a strategy — all callbacks route through the PM.
    am = AccountManager(
        pm=MagicMock(), connectors={"binance": MagicMock()}, base_currencies={"binance": "USDT"}, time=_T()
    )
    try:
        assert not hasattr(am, "_strategy")
        assert not hasattr(am, "_ctx")
    finally:
        am.stop()
