"""
Monotonic, host-disciplined wall clock.

Qubx timestamps must satisfy two properties at once:

- **monotonic** — `ctx.time()` must never move backwards. Backward steps corrupt OHLC
  aggregation, make the scheduler double-fire or skip cron events, and produce
  out-of-order signals and executions.
- **accurate** — timestamps have to be comparable against exchange timestamps and against
  other services, so the clock must stay pinned to true UTC.

`time.time()` alone gives the second but not the first: NTP can step `CLOCK_REALTIME`.
`time.monotonic()` alone gives the first but not the second: its epoch is an arbitrary
per-boot value, and its offset from UTC drifts without bound over a long-lived process.

This module combines them. Wall time is read as

    anchor_wall + (boot_now - anchor_boot) * rate

which is non-decreasing by construction, while a background thread steers `rate` toward
the host clock by *slewing* rather than stepping. The host clock is the reference because
it is already NTP-disciplined by the node (on the Xlydian platform, Talos syncs every node
to `time.cloudflare.com`, and node-exporter alerts on skew).

`CLOCK_BOOTTIME` is preferred over `CLOCK_MONOTONIC` because it keeps counting across a VM
suspend, which `CLOCK_MONOTONIC` does not.
"""

import threading
import time
from collections.abc import Callable

import numpy as np

from qubx import logger

# BOOTTIME survives VM suspend/live-migration; MONOTONIC is the portable fallback.
_MONO_CLOCK = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)

MAX_SLEW = 500e-6
"""Maximum frequency correction, as a fraction. 500ppm matches chrony's default ceiling."""

CORRECTION_WINDOW_NS = 60 * 1_000_000_000
"""Time constant of the servo: an offset error decays exponentially with this period."""

STEP_FORWARD_NS = 1_000_000_000
"""Forward error above which we re-anchor directly instead of slewing. Still monotonic."""

POLL_INTERVAL_SEC = 16.0
"""How often the discipline thread compares against the host clock."""


def _mono_ns() -> int:
    return time.clock_gettime_ns(_MONO_CLOCK)


def _real_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_REALTIME)


class MonotonicClock:
    """
    A wall clock that is monotonic by construction and disciplined onto the host clock.

    The read path is lock-free: `_state` is an immutable tuple that the discipline thread
    *replaces*, so a reader either sees the whole old state or the whole new one, never a
    mix of the two.
    """

    def __init__(
        self,
        mono: Callable[[], int] = _mono_ns,
        real: Callable[[], int] = _real_ns,
        max_slew: float = MAX_SLEW,
        correction_window_ns: int = CORRECTION_WINDOW_NS,
        step_forward_ns: int = STEP_FORWARD_NS,
    ):
        self._mono = mono
        self._real = real
        self._max_slew = max_slew
        self._correction_window_ns = correction_window_ns
        self._step_forward_ns = step_forward_ns
        # (anchor_mono_ns, anchor_wall_ns, rate) — replaced wholesale, never mutated in place
        self._state: tuple[int, int, float] = (mono(), real(), 1.0)
        self._discipline_thread: threading.Thread | None = None
        self._absorbing_backward_step = False

    def now_ns(self) -> int:
        """Current wall clock in nanoseconds since the Unix epoch. Never decreases."""
        anchor_mono, anchor_wall, rate = self._state  # single atomic read of an immutable tuple
        return anchor_wall + int((self._mono() - anchor_mono) * rate)

    def now(self) -> np.datetime64:
        return np.datetime64(self.now_ns(), "ns")

    def discipline(self) -> None:
        """
        Run one servo step: measure the error against the host clock and adjust the rate.

        Re-anchoring uses the value the clock *would have emitted* at this instant rather
        than the host's value, so the output stays continuous across the adjustment.
        """
        mono = self._mono()
        real = self._real()
        anchor_mono, anchor_wall, rate = self._state
        predicted = anchor_wall + int((mono - anchor_mono) * rate)
        error = real - predicted  # positive => we are behind the host

        if error > self._step_forward_ns:
            # A large forward jump means the host clock was corrected (or was never synced
            # when we anchored). Following it is safe: forward steps preserve monotonicity.
            logger.warning(f"<yellow>clock</yellow>: stepping forward {error / 1e9:.3f}s to match host clock")
            self._state = (mono, real, 1.0)
            return

        if error < -self._step_forward_ns:
            # Matching the host would mean going backwards, which we never do. Slew at the
            # maximum rate instead and let it be absorbed gradually. Absorbing a large step
            # takes a long time (1s at 500ppm is ~33min), so log the episode, not each poll.
            if not self._absorbing_backward_step:
                self._absorbing_backward_step = True
                logger.warning(
                    f"<yellow>clock</yellow>: host clock is {-error / 1e9:.3f}s behind us, "
                    f"slewing at {self._max_slew * 1e6:.0f}ppm (no backward step). "
                    f"Absorbing this will take ~{-error / 1e9 / self._max_slew / 60:.0f}min"
                )
        elif self._absorbing_backward_step:
            self._absorbing_backward_step = False
            logger.info(f"<green>clock</green>: back in sync with the host clock ({error / 1e6:+.3f}ms)")

        adjustment = max(-self._max_slew, min(self._max_slew, error / self._correction_window_ns))
        self._state = (mono, predicted, 1.0 + adjustment)

    def start_discipline_thread(self) -> None:
        """Start the background servo. Idempotent."""
        if self._discipline_thread is not None:
            return

        def _loop() -> None:
            logger.debug("Clock discipline thread is started")
            while True:
                time.sleep(POLL_INTERVAL_SEC)
                try:
                    self.discipline()
                except Exception as e:
                    logger.error(f"Clock discipline step failed: {e}")

        self._discipline_thread = threading.Thread(target=_loop, daemon=True, name="ClockDiscipline")
        self._discipline_thread.start()


_clock = MonotonicClock()


def time_now_ns() -> int:
    """Current wall clock in nanoseconds since the Unix epoch. Monotonic."""
    return _clock.now_ns()


def time_now() -> np.datetime64:
    """Current wall clock as a nanosecond `datetime64`. Monotonic."""
    return _clock.now()


def start_clock_discipline() -> None:
    """Start disciplining the process clock onto the host clock. Idempotent."""
    _clock.start_discipline_thread()
