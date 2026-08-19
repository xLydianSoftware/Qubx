import numpy as np

from qubx.core.basics import ITimeProvider
from qubx.core.status import ContextStatus, DegradeReason
from qubx.health.base import BaseHealthMonitor


class FixedTime(ITimeProvider):
    def __init__(self) -> None:
        self.now = np.datetime64("2026-08-19T00:00:00", "ns")

    def time(self) -> np.datetime64:
        return self.now


class StubSafePersistence:
    staleness_threshold_s = 60.0

    def __init__(self) -> None:
        self.age: float | None = 0.0

    def last_success_age(self) -> float | None:
        return self.age


def _make_monitor():
    monitor = BaseHealthMonitor(FixedTime())
    status = ContextStatus()
    monitor.set_status(status)
    sp = StubSafePersistence()
    monitor.set_state_persistence(sp)
    return monitor, status, sp


def test_degrades_when_stale_and_clears_on_recovery():
    monitor, status, sp = _make_monitor()

    sp.age = 10.0
    monitor.check_state_persistence()
    assert not any(d.reason == DegradeReason.STATE_PERSISTENCE_STALE for d in status.info.degradations)

    sp.age = 120.0
    monitor.check_state_persistence()
    assert any(d.reason == DegradeReason.STATE_PERSISTENCE_STALE for d in status.info.degradations)

    sp.age = 1.0
    monitor.check_state_persistence()
    assert not any(d.reason == DegradeReason.STATE_PERSISTENCE_STALE for d in status.info.degradations)


def test_no_persistence_wired_is_noop():
    monitor = BaseHealthMonitor(FixedTime())
    monitor.set_status(ContextStatus())
    monitor.check_state_persistence()  # must not raise


def test_age_none_before_first_write_is_not_stale():
    monitor, status, sp = _make_monitor()
    sp.age = None
    monitor.check_state_persistence()
    assert not any(d.reason == DegradeReason.STATE_PERSISTENCE_STALE for d in status.info.degradations)
