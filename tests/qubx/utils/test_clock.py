"""Tests for the monotonic, host-disciplined wall clock."""

from itertools import pairwise

import numpy as np

import qubx.utils.clock
from qubx import logger
from qubx.utils.clock import MAX_SLEW, MonotonicClock, time_now, time_now_ns

SEC = 1_000_000_000
MS = 1_000_000
EPOCH = 1_700_000_000 * SEC
TICK = 100 * MS  # sampling granularity: fine enough to observe a sub-second step


class FakeClock:
    """A manually advanced nanosecond clock source."""

    def __init__(self, start_ns: int):
        self.value = start_ns

    def __call__(self) -> int:
        return self.value

    def advance(self, delta_ns: int) -> None:
        self.value += delta_ns


def _run(clock: MonotonicClock, mono: FakeClock, real: FakeClock, seconds: float, host_drift_ppm: float = 0.0):
    """Advance both sources in TICK steps, disciplining each step. Returns the emitted samples."""
    samples = []
    for _ in range(int(seconds * SEC / TICK)):
        mono.advance(TICK)
        real.advance(TICK + int(TICK * host_drift_ppm / 1e6))
        samples.append(clock.now_ns())
        clock.discipline()
    return samples


def _assert_non_decreasing(samples):
    for i, (a, b) in enumerate(pairwise(samples)):
        assert b >= a, f"clock went backwards at sample {i + 1}: {a} -> {b} ({(b - a) / 1e6:.3f}ms)"


def _new(start_mono=42 * SEC, start_real=EPOCH):
    mono, real = FakeClock(start_mono), FakeClock(start_real)
    return MonotonicClock(mono=mono, real=real), mono, real


def test_now_never_decreases_when_host_clock_steps_backward():
    clock, mono, real = _new()

    samples = _run(clock, mono, real, seconds=10)
    real.advance(-50 * MS)  # host NTP yanks the wall clock back 50ms
    samples.append(clock.now_ns())  # the very next read must not regress
    samples += _run(clock, mono, real, seconds=200)

    _assert_non_decreasing(samples)


def test_backward_host_step_is_absorbed_by_slewing():
    clock, mono, real = _new()
    _run(clock, mono, real, seconds=10)

    real.advance(-50 * MS)
    _run(clock, mono, real, seconds=400)

    assert abs(clock.now_ns() - real()) < 1 * MS, "did not converge back onto the host clock"


def test_slew_stays_within_the_rate_bound():
    clock, mono, real = _new()
    _run(clock, mono, real, seconds=10)
    real.advance(-50 * MS)

    samples = _run(clock, mono, real, seconds=100)

    for i, (a, b) in enumerate(pairwise(samples)):
        rate = (b - a) / TICK
        assert abs(rate - 1.0) <= MAX_SLEW * 1.01, f"slew {(rate - 1) * 1e6:.1f}ppm exceeds bound at sample {i + 1}"


def test_large_forward_host_step_is_followed_immediately():
    clock, mono, real = _new()
    _run(clock, mono, real, seconds=10)

    real.advance(3600 * SEC)  # host jumps an hour forward
    clock.discipline()

    assert abs(clock.now_ns() - real()) < 1 * MS, "did not follow a large forward step"


def test_large_backward_host_step_is_never_followed_by_stepping_back():
    clock, mono, real = _new()
    samples = _run(clock, mono, real, seconds=10)

    real.advance(-3600 * SEC)  # host jumps an hour backward
    clock.discipline()
    samples.append(clock.now_ns())
    samples += _run(clock, mono, real, seconds=60)

    _assert_non_decreasing(samples)
    # still slewing toward the host, just slowly — never by jumping
    assert clock.now_ns() > real(), "should still be ahead of the host, absorbing the step gradually"


def test_large_backward_step_warns_once_rather_than_on_every_poll():
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        clock, mono, real = _new()
        _run(clock, mono, real, seconds=10)
        real.advance(-3600 * SEC)
        _run(clock, mono, real, seconds=60)  # 600 servo steps while absorbing the step
    finally:
        logger.remove(sink_id)

    warnings = [m for m in messages if "behind us" in m]
    assert len(warnings) == 1, f"expected one warning for the episode, got {len(warnings)}"


def test_tracks_host_clock_frequency_drift():
    clock, mono, real = _new()

    _run(clock, mono, real, seconds=600, host_drift_ppm=20.0)

    # a proportional servo keeps a bounded lag under a frequency ramp, it must not diverge
    assert abs(clock.now_ns() - real()) < 5 * MS, "diverged from a drifting host clock"


def test_discipline_replaces_state_atomically_rather_than_mutating():
    clock, mono, real = _new()
    before = clock._state

    mono.advance(SEC)
    real.advance(SEC + MS)
    clock.discipline()

    assert clock._state is not before, "state must be rebound, not mutated"
    assert before == (42 * SEC, EPOCH, 1.0), "the previously published state must stay intact for in-flight readers"


def test_discipline_thread_is_started_only_once():
    # runner.py builds LiveTimeProvider more than once; two servo threads would fight
    clock, _, _ = _new()

    clock.start_discipline_thread()
    first = clock._discipline_thread
    clock.start_discipline_thread()

    assert clock._discipline_thread is first, "a second discipline thread was started"


def test_live_time_provider_reads_the_monotonic_clock(monkeypatch):
    from qubx.core.basics import LiveTimeProvider

    mono, real = FakeClock(42 * SEC), FakeClock(EPOCH)
    monkeypatch.setattr(qubx.utils.clock, "_clock", MonotonicClock(mono=mono, real=real))

    provider = LiveTimeProvider()

    assert provider.time() == np.datetime64(EPOCH, "ns")


def test_time_now_returns_nanosecond_datetime64():
    value = time_now()

    assert isinstance(value, np.datetime64)
    assert np.datetime_data(value)[0] == "ns"
    assert abs(value.astype("int64") - time_now_ns()) < SEC
