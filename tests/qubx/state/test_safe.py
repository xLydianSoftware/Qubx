import contextlib
import threading
import time
from typing import Any

import pytest

from qubx import logger as qubx_logger
from qubx.core.exceptions import StatePersistenceUnavailable
from qubx.state.safe import SafeStatePersistence


class FakeBackend:
    """Scriptable IStatePersistence backend: can hang (Event) or fail N times."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.saves: list[tuple[str, Any]] = []
        self.gate = threading.Event()
        self.gate.set()  # open by default
        self.fail_saves_remaining = 0
        self.entered = threading.Event()  # set as soon as save() is invoked, before it can block on the gate

    def save(self, key: str, value: Any) -> None:
        self.entered.set()
        self.gate.wait(10.0)
        if self.fail_saves_remaining > 0:
            self.fail_saves_remaining -= 1
            raise ConnectionError("backend down")
        self.saves.append((key, value))
        self.store[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        self.gate.wait(10.0)
        return self.store.get(key, default)

    def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return key in self.store


def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@contextlib.contextmanager
def _capture_logs():
    """Capture qubx (loguru) log lines for the duration of the block."""
    messages: list[str] = []
    handler_id = qubx_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        yield messages
    finally:
        qubx_logger.remove(handler_id)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def sp(backend):
    p = SafeStatePersistence(backend, retry_backoff_s=(0.01, 0.02), sleep_fn=lambda s: time.sleep(min(s, 0.02)))
    yield p
    backend.gate.set()
    p.stop()


def test_save_returns_instantly_while_backend_hangs(sp, backend):
    backend.gate.clear()  # backend hangs forever
    t0 = time.monotonic()
    sp.save("state", {"a": 1})
    assert time.monotonic() - t0 < 0.05  # THE loop-liveness invariant


def test_unserializable_value_raises_at_call_site(sp):
    with pytest.raises(TypeError):
        sp.save("bad", object())


def test_write_through_and_last_success_age(sp, backend):
    assert sp.last_success_age() is None
    sp.save("k", {"v": 1})
    assert _wait(lambda: backend.store.get("k") == {"v": 1})
    assert sp.last_success_age() is not None and sp.last_success_age() < 1.0


def test_latest_wins_per_key_while_blocked(sp, backend):
    backend.gate.clear()
    sp.save("k", 1)
    sp.save("k", 2)
    sp.save("other", "x")
    backend.gate.set()
    assert _wait(lambda: backend.store.get("k") == 2 and backend.store.get("other") == "x")
    assert ("k", 1) not in backend.saves  # the stale intermediate was never written


def test_read_your_writes_before_flush(sp, backend):
    backend.gate.clear()
    sp.save("k", {"pending": True})
    # deterministically pin the read to the harder in-flight window: wait until the writer has actually
    # dequeued 'k' and entered backend.save() (blocked on the gate) rather than racing on scheduling -
    # without the _inflight fix this falls through to the (gate-blocked) backend.load() call. On a lucky
    # scheduling the two independent 10s gate.wait() timeouts (writer's save, this load) can resolve close
    # enough together that the value still comes back right - but only after blocking the caller for ~10s,
    # which is precisely the loop-liveness violation this class exists to prevent (spec review F2/F9), so
    # we assert on latency too, not just on the returned value.
    assert backend.entered.wait(2.0)
    t0 = time.monotonic()
    assert sp.load("k") == {"pending": True}  # served from _inflight, not a stale/blocked backend read
    assert time.monotonic() - t0 < 0.05  # loop-liveness invariant: never blocks on the network
    assert sp.exists("k") is True

    sp.delete("k")  # a newer write arrives while the stale save is still in flight
    assert sp.load("k", default="D") == "D"  # newer pending tombstone wins over the stale in-flight value
    assert sp.exists("k") is False

    backend.gate.set()


def test_tombstones(sp, backend):
    backend.store["k"] = "old"
    backend.gate.clear()
    sp.delete("k")
    assert sp.load("k", default="D") == "D"
    assert sp.exists("k") is False
    backend.gate.set()
    assert _wait(lambda: "k" not in backend.store)


def test_failed_batch_remerge_does_not_clobber_newer(sp, backend):
    backend.fail_saves_remaining = 1
    sp.save("k", "v1")  # first write attempt fails
    time.sleep(0.005)
    sp.save("k", "v2")  # arrives while retry pending
    assert _wait(lambda: backend.store.get("k") == "v2")
    assert not _wait(lambda: backend.store.get("k") == "v1", timeout=0.2)


def test_stop_flushes_pending(backend):
    p = SafeStatePersistence(backend, sleep_fn=lambda s: time.sleep(min(s, 0.01)))
    p.save("final", 42)
    p.stop()
    assert backend.store.get("final") == 42


def test_load_propagates_backend_errors(sp, backend):
    def boom(key, default=None):
        raise ConnectionError("down")

    backend.load = boom  # type: ignore[assignment]
    with pytest.raises(ConnectionError):
        sp.load("missing")


def test_backoff_uses_round_count_not_failed_key_count(backend):
    """spec review F1: a batch with one failing key and one succeeding key must not reset the
    consecutive-failure counter to 0 (via the succeeding key) and then index backoff[-1] (max tier)
    on the very first partial failure - it must count failing ROUNDS and use backoff[0]."""
    backend.fail_saves_remaining = 1  # exactly the first backend.save() call fails, whichever key
    recorded: list[float] = []

    def recording_sleep(s: float) -> None:
        recorded.append(s)
        time.sleep(min(s, 0.02))

    p = SafeStatePersistence(backend, retry_backoff_s=(0.01, 0.02, 0.03), sleep_fn=recording_sleep)
    try:
        with p._cond:  # hold the lock so the writer can't dequeue until BOTH keys are buffered together
            p.save("a", 1)
            p.save("b", 2)
        assert _wait(lambda: recorded)  # the round-level backoff was recorded
        assert recorded[0] == pytest.approx(0.01)  # retry_backoff_s[0], never backoff[-1] (0.03)
        assert _wait(lambda: backend.store.get("a") == 1 and backend.store.get("b") == 2)
    finally:
        p.stop()


def test_stop_interrupts_backoff_bounded_final_attempt(backend):
    """spec review F3: stop() called while the writer is sleeping in a (large) backoff must not wait
    out the full backoff - it interrupts immediately, the writer makes at most one more bounded
    final-drain attempt (no further backoff/retries), and abandoned keys are logged (spec review F4a)."""
    backend.fail_saves_remaining = 10**6  # keep failing forever
    call_count = {"n": 0}
    first_attempt = threading.Event()
    orig_save = backend.save

    def counting_save(key: str, value: Any) -> None:
        call_count["n"] += 1
        first_attempt.set()
        return orig_save(key, value)

    backend.save = counting_save  # type: ignore[assignment]

    # default sleep_fn=time.sleep -> the writer's backoff wait uses the interruptible _stop_event
    p = SafeStatePersistence(backend, flush_timeout_s=2.0, retry_backoff_s=(30.0,))
    p.save("k", "v1")
    assert first_attempt.wait(1.0)  # first (failing) attempt happened; writer now backing off for 30s
    time.sleep(0.05)  # let the writer settle into the backoff wait
    n_before = call_count["n"]

    with _capture_logs() as messages:
        t0 = time.monotonic()
        p.stop()
        dt = time.monotonic() - t0

    assert dt < 1.0  # returned fast: the 30s backoff was interrupted, not waited out
    assert not p._writer.is_alive()  # writer actually exited - never a zombie thread sleeping in the background
    assert call_count["n"] - n_before <= 1  # at most ONE more attempt after stop was observed
    assert any("abandoning" in m and "'k'" in m for m in messages)  # abandoned key logged, not silently dropped


def test_stop_join_timeout_logs_unflushed_count(backend):
    """spec review F4b: if the writer is still flushing when the join times out, stop() must not
    silently swallow that - it logs how many keys are still unflushed."""
    backend.gate.clear()  # backend call hangs
    p = SafeStatePersistence(backend, flush_timeout_s=0.05, sleep_fn=lambda s: time.sleep(min(s, 0.01)))
    p.save("k", 1)
    assert backend.entered.wait(1.0)  # writer has started the (now-hanging) backend call

    with _capture_logs() as messages:
        t0 = time.monotonic()
        p.stop()
        dt = time.monotonic() - t0

    assert dt < 0.5  # bounded by flush_timeout_s, not an indefinite hang
    assert p._writer.is_alive()  # writer genuinely still stuck - stop() must not claim otherwise
    assert any("timed out" in m and "1 key" in m for m in messages)  # unflushed count logged, not swallowed

    backend.gate.set()  # release so the writer (and the process) isn't left hanging
    assert _wait(lambda: not p._writer.is_alive())


def test_save_after_stop_does_not_raise_and_warns_once(backend):
    """spec review F4c: save()/delete() after stop() must not raise (spec) but also must not silently
    buffer into a dead queue - warn once, and drop."""
    p = SafeStatePersistence(backend, sleep_fn=lambda s: time.sleep(min(s, 0.01)))
    p.stop()

    with _capture_logs() as messages:
        p.save("late", 1)  # must not raise
        assert p.delete("late2") is True  # still optimistic True per contract
        p.save("late3", 3)  # second post-stop write - warning must not repeat

    warn_count = sum(1 for m in messages if "save after stop" in m)
    assert warn_count == 1  # warned once, not once per call
    time.sleep(0.05)
    assert backend.store == {}  # nothing reaches the (dead) backend


def test_validate_startup_succeeds_after_transient_failures(backend):
    calls = {"n": 0}

    def flaky_exists(key: str) -> bool:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("not yet")
        return False

    backend.exists = flaky_exists  # type: ignore[assignment]
    p = SafeStatePersistence(backend, sleep_fn=lambda s: None)
    p.validate_startup(deadline_s=60.0)
    assert calls["n"] == 3
    # - probe success seeds last-success so staleness monitoring is never silently absent
    #   for a backend that dies before the first write
    age = p.last_success_age()
    assert age is not None and age < 1.0
    p.stop()


def test_validate_startup_exhausts_budget_and_raises(backend):
    def always_down(key: str) -> bool:
        raise ConnectionError("down")

    backend.exists = always_down  # type: ignore[assignment]
    fake_now = {"t": 0.0}
    slept: list[float] = []

    def fake_sleep(s: float) -> None:
        slept.append(s)
        fake_now["t"] += s

    p = SafeStatePersistence(backend, sleep_fn=fake_sleep)
    with pytest.raises(StatePersistenceUnavailable):
        p.validate_startup(deadline_s=60.0, clock=lambda: fake_now["t"])
    assert slept[:6] == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]  # spec D3 schedule
    assert sum(slept) >= 60.0
    p.stop()
