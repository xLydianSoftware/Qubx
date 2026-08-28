from unittest.mock import MagicMock, call

import numpy as np

from qubx.core.basics import td_64
from qubx.core.boot import BootPhase, BootStateMachine

T0 = np.datetime64("2025-01-01T00:00:00", "ns")
SEC = td_64(1, "s")


def make_machine(**kw) -> tuple[BootStateMachine, MagicMock]:
    health = MagicMock()
    return BootStateMachine(health, **kw), health


def test_initial_phase_and_advance_emits_state_gauge():
    m, health = make_machine()
    assert m.phase == BootPhase.WAIT_READY
    assert not m.is_trading and not m.is_blocked
    m.advance(BootPhase.ON_START)
    assert m.phase == BootPhase.ON_START
    health.record_gauge.assert_called_with("boot.state", float(BootPhase.ON_START))


def test_advance_logs_each_transition_once():
    from qubx import logger

    lines: list[str] = []
    sink_id = logger.add(lambda msg: lines.append(str(msg)), level="INFO")
    try:
        m, _ = make_machine()
        m.advance(BootPhase.ON_START)
        m.advance(BootPhase.ON_START)  # same phase: no log
        m.advance(BootPhase.RESOLVE)
    finally:
        logger.remove(sink_id)
    transitions = [ln for ln in lines if "boot" in ln and "->" in ln]
    assert len(transitions) == 2
    assert "WAIT_READY -> " in transitions[0] and "ON_START" in transitions[0]
    assert "ON_START -> " in transitions[1] and "RESOLVE" in transitions[1]


def test_account_sync_alert_emits_once_and_clears():
    m, health = make_machine()
    m.account_sync_alert()
    m.account_sync_alert()
    assert health.record_gauge.call_args_list.count(call("boot.account_sync_blocked", 1.0)) == 1
    m.account_synced()
    health.record_gauge.assert_called_with("boot.account_sync_blocked", 0.0)
    health.record_gauge.reset_mock()
    m.account_synced()  # no alert pending -> no gauge
    health.record_gauge.assert_not_called()


def test_fit_retry_cadence_and_blocked_after_exhaustion():
    m, health = make_machine(fit_max_attempts=3, fit_retry_delay=td_64(60, "s"))
    m.advance(BootPhase.BOOT_FIT)

    assert m.fit_attempt_allowed(T0)
    m.record_fit_attempt()
    m.record_fit_failure(T0)
    assert m.phase == BootPhase.BOOT_FIT
    assert not m.fit_attempt_allowed(T0 + 59 * SEC)
    assert m.fit_attempt_allowed(T0 + 60 * SEC)

    m.record_fit_attempt()
    m.record_fit_failure(T0 + 60 * SEC)
    m.record_fit_attempt()
    m.record_fit_failure(T0 + 120 * SEC)

    assert m.is_blocked
    assert m.blocked_reason == "boot fit failed"
    assert m.fit_attempts == 3
    health.record_gauge.assert_any_call("boot.fit_failed", 1.0)
    assert not m.fit_attempt_allowed(T0 + 300 * SEC)


def test_fit_success_reaches_trading():
    m, health = make_machine()
    m.advance(BootPhase.BOOT_FIT)
    m.record_fit_attempt()
    m.record_fit_success()
    assert m.is_trading
    health.record_gauge.assert_called_with("boot.state", float(BootPhase.TRADING))


def test_blocked_self_heals_on_later_fit_success():
    m, health = make_machine(fit_max_attempts=1)
    m.advance(BootPhase.BOOT_FIT)
    m.record_fit_attempt()
    m.record_fit_failure(T0)
    assert m.is_blocked
    m.record_fit_success()
    assert m.is_trading
    health.record_gauge.assert_any_call("boot.fit_failed", 0.0)


def test_fit_failure_while_blocked_is_noop():
    m, _ = make_machine(fit_max_attempts=1)
    m.advance(BootPhase.BOOT_FIT)
    m.record_fit_attempt()
    m.record_fit_failure(T0)
    m.record_fit_failure(T0 + SEC)
    assert m.is_blocked and m.fit_attempts == 1


def test_warmup_finished_failure_gauge():
    m, health = make_machine()
    m.record_warmup_finished_failure()
    health.record_gauge.assert_called_with("boot.warmup_finished_failed", 1.0)


def test_fit_attempts_gauge_emitted():
    m, health = make_machine()
    m.advance(BootPhase.BOOT_FIT)
    m.record_fit_attempt()
    health.record_gauge.assert_any_call("boot.fit_attempts", 1.0)
