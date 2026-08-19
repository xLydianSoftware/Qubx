import threading
import time
from typing import Any

import pytest

from qubx.state.safe import SafeStatePersistence


class FakeBackend:
    """Scriptable IStatePersistence backend: can hang (Event) or fail N times."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.saves: list[tuple[str, Any]] = []
        self.gate = threading.Event()
        self.gate.set()  # open by default
        self.fail_saves_remaining = 0

    def save(self, key: str, value: Any) -> None:
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
    assert sp.load("k") == {"pending": True}  # served from pending, no network
    assert sp.exists("k") is True


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
