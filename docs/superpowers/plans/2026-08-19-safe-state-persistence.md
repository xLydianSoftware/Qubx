# Safe State Persistence & Bounded Outbound I/O — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The event loop never performs network I/O; every network call is bounded in time; every buffer is bounded in space (qubx side of xlydian-platform#375).

**Architecture:** A `SafeStatePersistence` wrapper gives all persistence calls async per-key latest-wins semantics with read-your-writes and fail-fast startup validation; a shared `BoundedWorker` primitive (single thread, drop-oldest deque) replaces the unbounded `ThreadPoolExecutor`s in `RedisStreamsExporter` and `QuestDBMetricEmitter`; all redis/QuestDB clients gain socket timeouts and keepalive. Health monitor gains a `STATE_PERSISTENCE_STALE` degradation.

**Tech Stack:** Python 3.12, redis-py, questdb-client, loguru (`from qubx import logger`), pytest (`just test` = `uv run pytest -m "not integration and not e2e" --ignore=debug -v -n auto`).

**Spec:** `docs/superpowers/specs/2026-08-19-safe-state-persistence-design.md`

## Global Constraints

- Work in worktree `~/devs/Qubx/.worktrees/safe-state-persistence`, branch `feat/safe-state-persistence` (off `dev`). **Never push to `dev` directly** — pushing to `dev` triggers the full release pipeline. The branch is pushed and PR'd at the end.
- Modern typing (`str | None`, `dict`, `tuple`), ruff line length 120.
- Logging only via `from qubx import logger`.
- All commands via `uv run ...` from the worktree root.
- `IStatePersistence` protocol (`src/qubx/core/interfaces.py:59`) must NOT change.
- Client timeout defaults (spec §2): `socket_connect_timeout=2.0, socket_timeout=5.0, socket_keepalive=True, health_check_interval=30` — overridable via constructor kwargs.
- Fail-fast startup (spec D3): probe budget 60s, backoff `0.5,1,2,4,8,10,10,...`; exhaustion raises `StatePersistenceUnavailable`.

---

### Task 1: `BoundedWorker` primitive

**Files:**
- Create: `src/qubx/utils/threading.py`
- Create: `tests/qubx/utils/__init__.py` (empty, if missing)
- Test: `tests/qubx/utils/test_bounded_worker.py`

**Interfaces:**
- Consumes: stdlib only.
- Produces (used by Tasks 3, 7, 8):
  ```python
  class BoundedWorker:
      def __init__(self, name: str, maxlen: int, warn_every_s: float = 30.0) -> None: ...
      def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...  # never blocks, never raises
      def stop(self, flush_timeout_s: float = 5.0) -> None: ...  # drain best-effort, then join
      @property
      def dropped(self) -> int: ...  # total items dropped since construction
      @property
      def queued(self) -> int: ...
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/utils/test_bounded_worker.py
import threading
import time

from qubx.utils.threading import BoundedWorker


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_executes_in_fifo_order_and_stop_flushes():
    out: list[int] = []
    w = BoundedWorker("t1", maxlen=100)
    for i in range(20):
        w.submit(out.append, i)
    w.stop(flush_timeout_s=2.0)
    assert out == list(range(20))


def test_submit_never_blocks_and_drops_oldest_when_full():
    gate = threading.Event()
    out: list[int] = []

    def task(i: int) -> None:
        gate.wait(5.0)
        out.append(i)

    w = BoundedWorker("t2", maxlen=3)
    t0 = time.monotonic()
    for i in range(10):  # 1 in-flight (blocked on gate), 3 queued max, rest dropped-oldest
        w.submit(task, i)
    assert time.monotonic() - t0 < 0.5  # submit never blocked
    assert _wait(lambda: w.dropped >= 6)
    gate.set()
    w.stop(flush_timeout_s=2.0)
    # the in-flight item plus the LAST 3 queued survive; older ones were dropped
    assert out[0] == 0 and out[-3:] == [7, 8, 9]
    assert w.dropped == 6


def test_task_exception_does_not_kill_worker():
    out: list[str] = []
    w = BoundedWorker("t3", maxlen=10)
    w.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    w.submit(out.append, "alive")
    w.stop(flush_timeout_s=2.0)
    assert out == ["alive"]


def test_submit_after_stop_is_noop():
    w = BoundedWorker("t4", maxlen=10)
    w.stop()
    w.submit(lambda: None)  # must not raise
    assert w.queued == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/utils/test_bounded_worker.py -v`
Expected: FAIL (ModuleNotFoundError: `qubx.utils.threading`)

- [ ] **Step 3: Implement `BoundedWorker`**

```python
# src/qubx/utils/threading.py
"""Bounded background-work primitives (spec: 2026-08-19-safe-state-persistence).

Absolute imports mean ``import threading`` below resolves to the stdlib even
though this module shares its name.
"""
import threading
import time
from collections import deque
from typing import Any, Callable

from qubx import logger


class BoundedWorker:
    """Single daemon worker thread over a drop-oldest bounded queue.

    ``submit`` never blocks and never raises: when the queue is full the OLDEST
    pending item is dropped (counted, warned at most once per ``warn_every_s``).
    One worker per instance is deliberate — it preserves FIFO ordering, which
    redis-stream exports rely on.
    """

    def __init__(self, name: str, maxlen: int, warn_every_s: float = 30.0) -> None:
        self._name = name
        self._maxlen = maxlen
        self._warn_every_s = warn_every_s
        self._queue: deque[tuple[Callable[..., Any], tuple, dict]] = deque()
        self._cond = threading.Condition()
        self._dropped = 0
        self._dropped_unreported = 0
        self._last_warn = 0.0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=f"BoundedWorker-{name}", daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def queued(self) -> int:
        with self._cond:
            return len(self._queue)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        with self._cond:
            if self._stopped:
                return
            if len(self._queue) >= self._maxlen:
                self._queue.popleft()
                self._dropped += 1
                self._dropped_unreported += 1
                now = time.monotonic()
                if now - self._last_warn >= self._warn_every_s:
                    logger.warning(
                        f"[BoundedWorker:{self._name}] queue full (maxlen={self._maxlen}) — "
                        f"dropped {self._dropped_unreported} oldest items since last report"
                    )
                    self._last_warn = now
                    self._dropped_unreported = 0
            self._queue.append((fn, args, kwargs))
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopped:
                    self._cond.wait()
                if not self._queue and self._stopped:
                    return
                fn, args, kwargs = self._queue.popleft()
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[BoundedWorker:{self._name}] task failed: {e}")

    def stop(self, flush_timeout_s: float = 5.0) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=flush_timeout_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/utils/test_bounded_worker.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/utils/threading.py tests/qubx/utils/
git commit -m "feat(utils): BoundedWorker — single-thread drop-oldest bounded work queue"
```

---

### Task 2: Client timeouts in `RedisStatePersistence`

**Files:**
- Modify: `src/qubx/state/redis.py:33-60` (constructor only)
- Test: `tests/qubx/state/__init__.py` (create empty), `tests/qubx/state/test_redis_client_options.py`

**Interfaces:**
- Produces: `RedisStatePersistence.__init__` gains keyword-only params
  `socket_connect_timeout: float = 2.0, socket_timeout: float = 5.0, socket_keepalive: bool = True, health_check_interval: int = 30` — forwarded to `redis.from_url`. Everything else unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/qubx/state/test_redis_client_options.py
from unittest.mock import MagicMock, patch

from qubx.state.redis import RedisStatePersistence


def test_client_created_with_bounded_failure_defaults():
    with patch("qubx.state.redis.redis.from_url", return_value=MagicMock()) as from_url:
        RedisStatePersistence(redis_url="redis://localhost:6379/0", strategy_name="s")
    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 2.0
    assert kwargs["socket_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == 30


def test_timeouts_overridable():
    with patch("qubx.state.redis.redis.from_url", return_value=MagicMock()) as from_url:
        RedisStatePersistence(redis_url="redis://localhost:6379/0", strategy_name="s", socket_timeout=1.5)
    assert from_url.call_args.kwargs["socket_timeout"] == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/state/test_redis_client_options.py -v`
Expected: FAIL (KeyError: 'socket_connect_timeout')

- [ ] **Step 3: Implement**

In `RedisStatePersistence.__init__`, extend the signature after `indent: int | None = 2` with:

```python
        *,
        socket_connect_timeout: float = 2.0,
        socket_timeout: float = 5.0,
        socket_keepalive: bool = True,
        health_check_interval: int = 30,
```

and replace `self._redis = redis.from_url(redis_url)` with:

```python
        # - bounded-failure client: a dead peer costs seconds, never TCP-retransmission
        #   minutes (incident 2026-08-19; platform #375). Retry policy lives in the
        #   SafeStatePersistence wrapper, not here.
        self._redis = redis.from_url(
            redis_url,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/state/test_redis_client_options.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/state/redis.py tests/qubx/state/
git commit -m "fix(state): bounded-failure redis client defaults (timeouts, keepalive)"
```

---

### Task 3: `SafeStatePersistence` — async core with read-your-writes

**Files:**
- Create: `src/qubx/state/safe.py`
- Test: `tests/qubx/state/test_safe.py`

**Interfaces:**
- Consumes: `IStatePersistence` protocol (`qubx.core.interfaces`).
- Produces (used by Tasks 4, 5, 6):
  ```python
  class SafeStatePersistence(IStatePersistence):
      def __init__(self, backend: IStatePersistence, *,
                   staleness_threshold_s: float = 60.0,
                   flush_timeout_s: float = 5.0,
                   retry_backoff_s: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0),
                   sleep_fn: Callable[[float], None] = time.sleep) -> None: ...
      def save(self, key: str, value: Any) -> None            # buffers; eager json dry-run; never network-blocks
      def load(self, key: str, default: Any = None) -> Any     # pending-first read-your-writes
      def delete(self, key: str) -> bool
      def exists(self, key: str) -> bool
      def last_success_age(self) -> float | None               # seconds since last successful write; None before first
      def stop(self) -> None                                   # bounded flush + join
      staleness_threshold_s: float                             # threshold the health monitor reads (Task 6)
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/state/test_safe.py
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
    assert sp.load("k") == {"pending": True}   # served from pending, no network
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
    sp.save("k", "v1")            # first write attempt fails
    time.sleep(0.005)
    sp.save("k", "v2")            # arrives while retry pending
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/state/test_safe.py -v`
Expected: FAIL (ModuleNotFoundError: `qubx.state.safe`)

- [ ] **Step 3: Implement `SafeStatePersistence`**

```python
# src/qubx/state/safe.py
"""Async wrapper making any IStatePersistence backend safe for the event loop.

Contract (spec 2026-08-19-safe-state-persistence, D2): all writes are buffered
per-key (latest wins) and flushed by ONE background writer thread; ``load``
consults the pending buffer first (read-your-writes); no caller ever blocks on
the network and network errors never raise from ``save``/``delete``.
"""
import json
import threading
import time
from typing import Any, Callable

from qubx import logger
from qubx.core.interfaces import IStatePersistence

_TOMBSTONE = object()


class SafeStatePersistence(IStatePersistence):
    def __init__(
        self,
        backend: IStatePersistence,
        *,
        staleness_threshold_s: float = 60.0,
        flush_timeout_s: float = 5.0,
        retry_backoff_s: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0),
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._backend = backend
        self.staleness_threshold_s = staleness_threshold_s
        self._flush_timeout_s = flush_timeout_s
        self._retry_backoff_s = retry_backoff_s
        self._sleep = sleep_fn

        self._pending: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._stopped = False
        self._last_success: float | None = None
        self._consecutive_failures = 0
        self._last_fail_log = 0.0
        self._writer = threading.Thread(target=self._run, name="StatePersistenceWriter", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------ writes
    def save(self, key: str, value: Any) -> None:
        json.dumps(value)  # - eager dry-run: programming errors (TypeError/ValueError) stay loud at the call site
        with self._cond:
            self._pending[key] = value
            self._cond.notify()

    def delete(self, key: str) -> bool:
        with self._cond:
            self._pending[key] = _TOMBSTONE
            self._cond.notify()
        return True  # optimistic: actual backend result is async

    # ------------------------------------------------------------------- reads
    def load(self, key: str, default: Any = None) -> Any:
        with self._cond:
            if key in self._pending:
                v = self._pending[key]
                return default if v is _TOMBSTONE else v
        return self._backend.load(key, default)

    def exists(self, key: str) -> bool:
        with self._cond:
            if key in self._pending:
                return self._pending[key] is not _TOMBSTONE
        return self._backend.exists(key)

    # ------------------------------------------------------------------ health
    def last_success_age(self) -> float | None:
        return None if self._last_success is None else time.monotonic() - self._last_success

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # ------------------------------------------------------------------ writer
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopped:
                    self._cond.wait()
                if not self._pending and self._stopped:
                    return
                batch, self._pending = self._pending, {}

            failed: dict[str, Any] = {}
            for key, value in batch.items():
                try:
                    if value is _TOMBSTONE:
                        self._backend.delete(key)
                    else:
                        self._backend.save(key, value)
                    self._last_success = time.monotonic()
                    self._consecutive_failures = 0
                except (TypeError, ValueError):
                    logger.error(f"[SafeStatePersistence] unserializable value for '{key}' reached writer; dropped")
                except Exception as e:
                    failed[key] = value
                    self._consecutive_failures += 1
                    now = time.monotonic()
                    if now - self._last_fail_log >= 30.0:
                        logger.warning(
                            f"[SafeStatePersistence] backend write failed ({self._consecutive_failures} in a row): {e}"
                        )
                        self._last_fail_log = now

            if failed:
                with self._cond:
                    for key, value in failed.items():
                        self._pending.setdefault(key, value)  # - newer pending values win over the failed batch
                    if self._stopped:
                        return  # - do not backoff-sleep during shutdown
                backoff = self._retry_backoff_s[min(self._consecutive_failures, len(self._retry_backoff_s)) - 1]
                self._sleep(backoff)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._writer.join(timeout=self._flush_timeout_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/state/test_safe.py -v`
Expected: 10 PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/state/safe.py tests/qubx/state/test_safe.py
git commit -m "feat(state): SafeStatePersistence — async per-key latest-wins wrapper with read-your-writes"
```

---

### Task 4: Startup validation + `StatePersistenceUnavailable`

**Files:**
- Modify: `src/qubx/core/exceptions.py` (append)
- Modify: `src/qubx/state/safe.py` (add `validate_startup`)
- Test: `tests/qubx/state/test_safe.py` (append)

**Interfaces:**
- Produces (used by Task 5):
  ```python
  class StatePersistenceUnavailable(BaseError): ...   # qubx.core.exceptions

  SafeStatePersistence.validate_startup(
      self, deadline_s: float = 60.0,
      probe_backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 10.0),
      clock: Callable[[], float] = time.monotonic,
  ) -> None   # raises StatePersistenceUnavailable on budget exhaustion
  ```

- [ ] **Step 1: Write the failing tests** (append to `tests/qubx/state/test_safe.py`)

```python
from qubx.core.exceptions import StatePersistenceUnavailable


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/state/test_safe.py -k validate_startup -v`
Expected: FAIL (ImportError: `StatePersistenceUnavailable`)

- [ ] **Step 3: Implement**

Append to `src/qubx/core/exceptions.py`:

```python
class StatePersistenceUnavailable(BaseError):
    """State persistence is enabled but unreachable/unreadable at startup.

    Fatal by design (incident 2026-08-19, platform #375): starting a bot that
    silently lost its persisted state is worse than crash-looping until the
    backend returns.
    """
```

Append to `SafeStatePersistence` (import `StatePersistenceUnavailable` from `qubx.core.exceptions` at top of `safe.py`):

```python
    def validate_startup(
        self,
        deadline_s: float = 60.0,
        probe_backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 10.0),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Probe the backend with backoff for up to ``deadline_s``; raise on exhaustion (spec D3)."""
        start = clock()
        attempt = 0
        last_err: Exception | None = None
        while True:
            try:
                self._backend.exists("__qubx_probe__")
                logger.info(f"[SafeStatePersistence] backend validated after {attempt + 1} attempt(s)")
                return
            except Exception as e:
                last_err = e
                elapsed = clock() - start
                if elapsed >= deadline_s:
                    raise StatePersistenceUnavailable(
                        f"state persistence unreachable after {elapsed:.0f}s ({attempt + 1} attempts): {last_err}"
                    ) from last_err
                backoff = probe_backoff_s[min(attempt, len(probe_backoff_s) - 1)]
                logger.warning(f"[SafeStatePersistence] startup probe failed (attempt {attempt + 1}): {e}; retrying in {backoff}s")
                self._sleep(backoff)
                attempt += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/state/test_safe.py -v`
Expected: 12 PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/exceptions.py src/qubx/state/safe.py tests/qubx/state/test_safe.py
git commit -m "feat(state): fail-fast startup validation with 60s backoff budget"
```

---

### Task 5: Factory wiring

**Files:**
- Modify: `src/qubx/utils/runner/factory.py` (function `create_state_persistence`, currently at :445)
- Test: `tests/qubx/state/test_factory_wrapping.py`

**Interfaces:**
- Consumes: `SafeStatePersistence` (Task 3/4), `StatePersistenceConfig` (has `type: str`, `parameters: dict`, `snapshot_interval: str | None = "5s"`).
- Produces: `create_state_persistence(...)` returns `SafeStatePersistence` for any non-Dummy backend, already startup-validated; staleness threshold = `max(3 × snapshot_interval_seconds, 60.0)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/state/test_factory_wrapping.py
from unittest.mock import patch

from qubx.state.dummy import DummyStatePersistence
from qubx.state.safe import SafeStatePersistence
from qubx.utils.runner.configs import StatePersistenceConfig
from qubx.utils.runner.factory import create_state_persistence


class InMemoryBackend:
    def __init__(self, strategy_name: str = "", **kwargs):
        self.store = {}
        self.probed = False

    def save(self, key, value):
        self.store[key] = value

    def load(self, key, default=None):
        return self.store.get(key, default)

    def delete(self, key):
        return self.store.pop(key, None) is not None

    def exists(self, key):
        self.probed = True
        return key in self.store


def test_real_backend_is_wrapped_and_validated():
    cfg = StatePersistenceConfig(type="RedisStatePersistence", parameters={}, snapshot_interval="5s")
    with patch("qubx.utils.runner.factory.class_import", return_value=InMemoryBackend):
        sp = create_state_persistence(cfg, "strat")
    assert isinstance(sp, SafeStatePersistence)
    assert sp.staleness_threshold_s == 60.0  # max(3*5s, 60s)
    sp.stop()


def test_threshold_scales_with_long_snapshot_interval():
    cfg = StatePersistenceConfig(type="RedisStatePersistence", parameters={}, snapshot_interval="1m")
    with patch("qubx.utils.runner.factory.class_import", return_value=InMemoryBackend):
        sp = create_state_persistence(cfg, "strat")
    assert sp.staleness_threshold_s == 180.0  # max(3*60s, 60s)
    sp.stop()


def test_dummy_backend_is_not_wrapped():
    cfg = StatePersistenceConfig(type="DummyStatePersistence", parameters={})
    with patch("qubx.utils.runner.factory.class_import", return_value=DummyStatePersistence):
        sp = create_state_persistence(cfg, "strat")
    assert isinstance(sp, DummyStatePersistence)


def test_none_config_returns_none():
    assert create_state_persistence(None, "strat") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/state/test_factory_wrapping.py -v`
Expected: `test_real_backend_is_wrapped_and_validated` FAILs (returns `InMemoryBackend`, not `SafeStatePersistence`)

- [ ] **Step 3: Implement**

In `factory.py`, add imports:

```python
import pandas as pd

from qubx.state.dummy import DummyStatePersistence
from qubx.state.safe import SafeStatePersistence
```

and in `create_state_persistence`, replace

```python
        persistence = persistence_class(**params)
        logger.info(f"Created state persistence: {persistence_class_name}")
        return persistence
```

with

```python
        persistence = persistence_class(**params)
        logger.info(f"Created state persistence: {persistence_class_name}")

        if isinstance(persistence, DummyStatePersistence):
            return persistence

        # - wrap every real backend (spec D2/D4): async latest-wins writes, read-your-writes,
        #   fail-fast startup (D3). StatePersistenceUnavailable propagates and kills the runner.
        interval_s = pd.Timedelta(config.snapshot_interval).total_seconds() if config.snapshot_interval else 0.0
        safe = SafeStatePersistence(persistence, staleness_threshold_s=max(3.0 * interval_s, 60.0))
        safe.validate_startup()
        return safe
```

(the surrounding `except Exception ... raise` already makes the failure fatal to the runner — `runner.py:653` performs no fallback.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/state/test_factory_wrapping.py tests/qubx/state -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/utils/runner/factory.py tests/qubx/state/test_factory_wrapping.py
git commit -m "feat(runner): wrap real state backends in SafeStatePersistence with fail-fast startup"
```

---

### Task 6: Health integration — `STATE_PERSISTENCE_STALE`

**Files:**
- Modify: `src/qubx/core/status.py` (`DegradeReason`)
- Modify: `src/qubx/health/base.py` (new setter + check, call in `_monitor_loop` at :533)
- Modify: `src/qubx/core/context.py` (wire after `self._health_monitor.set_status(self._status)` at :216; note `self._state_persistence` is assigned at :217 — insert the wiring after that line)
- Test: `tests/qubx/health/test_state_staleness.py`

**Interfaces:**
- Consumes: `SafeStatePersistence.last_success_age()`, `.staleness_threshold_s` (Task 3), `ContextStatus.add/clear`, `record_gauge(name, value, tags)` (`health/base.py:526`).
- Produces: `DegradeReason.STATE_PERSISTENCE_STALE = "state_persistence_stale"`;
  `BaseHealthMonitor.set_state_persistence(sp: Any) -> None`; `BaseHealthMonitor.check_state_persistence() -> None`; gauge `state_persistence_lag` (seconds).

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/health/test_state_staleness.py
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
```

(If `BaseHealthMonitor(FixedTime())` needs more constructor args, mirror the fixture at `tests/qubx/health/test_base.py:93-97` — it constructs a monitor from a `MockTimeProvider`; reuse its argument pattern verbatim.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/health/test_state_staleness.py -v`
Expected: FAIL (AttributeError: `set_state_persistence`)

- [ ] **Step 3: Implement**

`src/qubx/core/status.py` — extend the enum:

```python
class DegradeReason(StrEnum):
    INTERNAL_QUEUE_OVERFLOW = "internal_queue_overflow"
    EXCHANGE_MAINTENANCE = "exchange_maintenance"
    STATE_PERSISTENCE_STALE = "state_persistence_stale"
```

`src/qubx/health/base.py` — add next to `set_status` (`:252`), following the
`check_queue_drain` degrade/clear pattern (`:257-293`):

```python
    def set_state_persistence(self, persistence: Any) -> None:
        """Wire a SafeStatePersistence (duck-typed: needs last_success_age() and
        staleness_threshold_s) so the monitor can report write staleness."""
        self._state_persistence = persistence
        self._state_stale = False

    def check_state_persistence(self) -> None:
        sp = getattr(self, "_state_persistence", None)
        if sp is None or self._status is None:
            return
        age = sp.last_success_age()
        if age is not None:
            self.record_gauge("state_persistence_lag", age)
        threshold = sp.staleness_threshold_s
        stale = age is not None and age > threshold
        if stale and not getattr(self, "_state_stale", False):
            self._state_stale = True
            logger.warning(
                f"[health] state not persisted for {age:.0f}s (threshold {threshold:.0f}s) — "
                "context DEGRADED (state_persistence_stale)"
            )
            self._status.add(
                DegradeReason.STATE_PERSISTENCE_STALE, self.time_provider.time(),
                message=f"no successful state write for {age:.0f}s",
            )
        elif not stale and getattr(self, "_state_stale", False):
            self._state_stale = False
            logger.info("[health] state persistence recovered — state_persistence_stale cleared")
            self._status.clear(DegradeReason.STATE_PERSISTENCE_STALE)
```

In `_monitor_loop` (`:533`), inside the `try:` after `self.check_queue_drain(current_size)`:

```python
                self.check_state_persistence()
```

`src/qubx/core/context.py` — after line 217 (`self._state_persistence = state_persistence or DummyStatePersistence()`):

```python
        if hasattr(self._state_persistence, "last_success_age"):
            self._health_monitor.set_state_persistence(self._state_persistence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/health/ -v`
Expected: new tests PASS, existing `test_base.py` still PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/status.py src/qubx/health/base.py src/qubx/core/context.py tests/qubx/health/test_state_staleness.py
git commit -m "feat(health): STATE_PERSISTENCE_STALE degradation + state_persistence_lag gauge"
```

---

### Task 7: `RedisStreamsExporter` — BoundedWorker + client timeouts

**Files:**
- Modify: `src/qubx/exporters/redis_streams.py` (`:66` client, `:82` executor, `:102-109` stop, `:145` submit)
- Test: `tests/qubx/exporters/test_redis_streams_bounded.py`

**Interfaces:**
- Consumes: `BoundedWorker` (Task 1).
- Produces: same public exporter API; constructor param `max_workers: int = 2` becomes deprecated-ignored (kept for config compat, log a deprecation debug); new param `max_queue: int = 1000`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/exporters/test_redis_streams_bounded.py
import threading
import time
from unittest.mock import MagicMock, patch

from qubx.exporters.redis_streams import RedisStreamsExporter


def _make_exporter(**kwargs):
    with patch("qubx.exporters.redis_streams.redis.from_url", return_value=MagicMock()) as from_url:
        exp = RedisStreamsExporter(redis_url="redis://localhost:6379/0", strategy_name="s", **kwargs)
    return exp, from_url


def test_client_has_bounded_failure_defaults():
    _, from_url = _make_exporter()
    kwargs = from_url.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == 2.0
    assert kwargs["socket_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True


def test_stream_writes_preserve_fifo_order():
    exp, _ = _make_exporter()
    seen: list[int] = []
    for i in range(50):
        exp._worker.submit(seen.append, i)
    exp._worker.stop(flush_timeout_s=2.0)
    assert seen == list(range(50))


def test_queue_bounded_under_hung_backend():
    exp, _ = _make_exporter(max_queue=5)
    gate = threading.Event()
    exp._worker.submit(gate.wait, 5.0)  # occupy the worker
    t0 = time.monotonic()
    for i in range(50):
        exp._worker.submit(lambda: None)
    assert time.monotonic() - t0 < 0.5
    assert exp._worker.dropped >= 44
    gate.set()
    exp.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/exporters/test_redis_streams_bounded.py -v`
Expected: FAIL (no `socket_connect_timeout` kwarg; no `_worker` attribute)

- [ ] **Step 3: Implement**

In `redis_streams.py`:
- add `from qubx.utils.threading import BoundedWorker`; add constructor param `max_queue: int = 1000`.
- `:66` → same client kwargs block as Task 2 (`socket_connect_timeout=2.0, socket_timeout=5.0, socket_keepalive=True, health_check_interval=30`).
- `:82` → replace `self._executor = ThreadPoolExecutor(...)` with:

```python
        # - single bounded worker: preserves XADD ordering per stream (2 pool workers could
        #   reorder targets) and bounds memory/burst under outages (platform #375).
        if max_workers != 2:
            logger.debug("[RedisStreamsExporter] max_workers is deprecated and ignored (single bounded worker)")
        self._worker = BoundedWorker("redis_exporter", maxlen=max_queue)
```

- `:145` (and any other `self._executor.submit(...)` in the file — grep for them) → `self._worker.submit(...)` with identical arguments.
- `stop()` (`:102-109`): replace `self._executor.shutdown(wait=False, cancel_futures=True)` with `self._worker.stop(flush_timeout_s=5.0)`.
- Remove the now-unused `from concurrent.futures import ThreadPoolExecutor` import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/exporters/ -v`
Expected: new tests PASS, existing exporter tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/exporters/redis_streams.py tests/qubx/exporters/test_redis_streams_bounded.py
git commit -m "fix(exporters): bounded single-worker redis exporter with client timeouts (fixes XADD reordering)"
```

---

### Task 8: `QuestDBMetricEmitter` — BoundedWorker + Sender timeouts

**Files:**
- Modify: `src/qubx/emitters/questdb.py` (`:70` conn str, `:74` executor, `:96/:125/:179/:299/:314/:409` submit/stop sites)
- Test: `tests/qubx/emitters/test_questdb_bounded.py`

**Interfaces:**
- Consumes: `BoundedWorker` (Task 1).
- Produces: same public emitter API; new param `max_queue: int = 10_000`; conn string gains `request_timeout=5000;retry_timeout=5000;`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/emitters/test_questdb_bounded.py
from qubx.emitters.questdb import QuestDBMetricEmitter
from qubx.utils.threading import BoundedWorker


def test_conn_string_has_bounded_timeouts():
    em = QuestDBMetricEmitter(host="qdb.local", port=9000)
    assert "request_timeout=5000;" in em._conn_str
    assert "retry_timeout=5000;" in em._conn_str
    em._worker.stop()


def test_uses_bounded_worker():
    em = QuestDBMetricEmitter(host="qdb.local", port=9000, max_queue=123)
    assert isinstance(em._worker, BoundedWorker)
    assert em._worker._maxlen == 123
    em._worker.stop()
```

(If `QuestDBMetricEmitter.__init__` requires more args, copy the minimal construction used in `tests/qubx/emitters/metric_emitters_test.py` and adapt.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/emitters/test_questdb_bounded.py -v`
Expected: FAIL (`request_timeout` not in conn str)

- [ ] **Step 3: Implement**

In `questdb.py`:
- `:70` → `self._conn_str = f"http::addr={host}:{port};request_timeout=5000;retry_timeout=5000;"`
  (client default `retry_timeout` is 10000 — halved so a dead server costs ≤10s per flush attempt; verified against installed `questdb.ingress.Sender` API.)
- `:74` → `self._worker = BoundedWorker("questdb_emitter", maxlen=max_queue)` with new constructor param `max_queue: int = 10_000`; `max_workers` kept but deprecated-ignored (debug log), as in Task 7.
- Every `self._executor.submit(...)` (`:96, :179, :299, :314, :409`) → `self._worker.submit(...)`.
- `:125` `self._executor.shutdown(wait=False, cancel_futures=True)` → `self._worker.stop(flush_timeout_s=5.0)`.
- Remove unused `ThreadPoolExecutor` import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/emitters/ -v`
Expected: new tests PASS, `metric_emitters_test.py` and `strategy_tables_test.py` PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/emitters/questdb.py tests/qubx/emitters/test_questdb_bounded.py
git commit -m "fix(emitters): bounded questdb emitter queue + ILP request/retry timeouts"
```

---

### Task 9: Integration test (docker redis), full validation, PR

**Files:**
- Create: `tests/qubx/state/test_safe_integration.py`
- Test: full suite + lint

**Interfaces:** consumes everything above; produces the PR.

- [ ] **Step 1: Write the integration test**

```python
# tests/qubx/state/test_safe_integration.py
"""Integration: SafeStatePersistence vs a real redis that gets frozen mid-run.

``docker pause`` freezes the server process WITHOUT closing TCP connections —
the closest reproduction of the 2026-08-19 half-open-socket incident.
Requires docker; runs only under `-m integration`.
"""
import json
import subprocess
import time
import uuid

import pytest

pytestmark = pytest.mark.integration

CONTAINER = f"qubx-test-redis-{uuid.uuid4().hex[:8]}"
PORT = 63790


@pytest.fixture(scope="module")
def redis_container():
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", CONTAINER, "-p", f"{PORT}:6379", "redis:7-alpine"],
        check=True, capture_output=True,
    )
    time.sleep(1.0)
    yield CONTAINER
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def _pause():
    subprocess.run(["docker", "pause", CONTAINER], check=True, capture_output=True)


def _unpause():
    subprocess.run(["docker", "unpause", CONTAINER], check=True, capture_output=True)


def test_event_loop_liveness_through_redis_freeze(redis_container):
    from qubx.state.redis import RedisStatePersistence
    from qubx.state.safe import SafeStatePersistence

    backend = RedisStatePersistence(
        redis_url=f"redis://localhost:{PORT}/0", strategy_name="itest", socket_timeout=1.0, socket_connect_timeout=1.0
    )
    sp = SafeStatePersistence(backend, retry_backoff_s=(0.5, 1.0))
    sp.validate_startup(deadline_s=10.0)

    sp.save("k", {"phase": 1})
    deadline = time.monotonic() + 5.0
    while sp.last_success_age() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sp.last_success_age() is not None

    _pause()
    try:
        t0 = time.monotonic()
        sp.save("k", {"phase": 2})           # must return instantly despite frozen server
        assert time.monotonic() - t0 < 0.05
        assert sp.load("k") == {"phase": 2}  # read-your-writes while frozen
        time.sleep(3.0)
        assert sp.last_success_age() > 2.0   # staleness visibly grows
    finally:
        _unpause()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if json.loads(backend._redis.get("state:itest:k") or "null") == {"phase": 2}:
            break
        time.sleep(0.2)
    else:
        pytest.fail("pending write was not flushed after recovery")
    sp.stop()


def test_cold_start_against_frozen_redis_fails_fast(redis_container):
    from qubx.core.exceptions import StatePersistenceUnavailable
    from qubx.state.redis import RedisStatePersistence
    from qubx.state.safe import SafeStatePersistence

    _pause()
    try:
        backend = RedisStatePersistence(
            redis_url=f"redis://localhost:{PORT}/0", strategy_name="itest2",
            socket_timeout=1.0, socket_connect_timeout=1.0,
        )
        sp = SafeStatePersistence(backend)
        t0 = time.monotonic()
        with pytest.raises(StatePersistenceUnavailable):
            sp.validate_startup(deadline_s=5.0)
        assert time.monotonic() - t0 < 15.0  # bounded, not TCP-retransmission minutes
        sp.stop()
    finally:
        _unpause()
```

- [ ] **Step 2: Run the integration test (requires docker)**

Run: `uv run pytest tests/qubx/state/test_safe_integration.py -m integration -v`
Expected: 2 PASS (skip gracefully and note it in the PR if docker is unavailable on this machine)

- [ ] **Step 3: Full validation**

Run: `uv run pytest -m "not integration and not e2e" --ignore=debug -q -n 4` (NOT `-n auto` — cgroup-unaware CPU count OOMs CI-sized environments; locally 4 is safe) and `just lint` (or `uv run ruff check src tests` if no lint recipe).
Expected: suite green, lint clean.

- [ ] **Step 4: Commit and push the branch (NOT dev)**

```bash
git add tests/qubx/state/test_safe_integration.py
git commit -m "test(state): integration — SafeStatePersistence liveness through docker-paused redis"
git push -u origin feat/safe-state-persistence
```

- [ ] **Step 5: Open the PR**

```bash
gh pr create --base dev --title "feat: safe state persistence & bounded outbound I/O (platform #375)" \
  --body "Implements docs/superpowers/specs/2026-08-19-safe-state-persistence-design.md — see spec for the incident analysis and decisions (D1-D4).

- SafeStatePersistence: async per-key latest-wins writes, read-your-writes, fail-fast 60s startup validation (StatePersistenceUnavailable), bounded flush on stop
- Redis clients (state + streams exporter): socket/connect timeouts, keepalive, health checks
- BoundedWorker replaces unbounded ThreadPoolExecutors in RedisStreamsExporter (also fixes XADD reordering) and QuestDBMetricEmitter; QuestDB Sender gains request/retry timeouts
- Health: DegradeReason.STATE_PERSISTENCE_STALE + state_persistence_lag gauge

Closes the qubx side of xLydianSoftware/xlydian-platform#375 (epic #374, incident 2026-08-19)."
```

---

### Task 10: Resilient rate-limit backend (client timeouts + local fallback + 30s breaker)

Implements spec section "Rate-limit backend resilience (increment 2)". Read that
section first — decisions RL-D1…RL-D4 are binding.

**Files:**
- Modify: `src/qubx/rate_limiting/redis_backend.py` (ctor timeout kwargs → `from_url`)
- Modify: `src/qubx/rate_limiting/backend.py` (add non-abstract `async def close()` no-op to `IRateLimitBackend`)
- Create: `src/qubx/rate_limiting/resilient.py` (`ResilientRateLimitBackend`)
- Modify: `src/qubx/rate_limiting/manager.py` (`_create_backend` wraps redis in the composite)
- Modify: `src/qubx/rate_limiting/__init__.py` (export `ResilientRateLimitBackend`)
- Test: `tests/qubx/rate_limiting/test_resilient.py`
- Test: `tests/qubx/rate_limiting/test_redis_client_options.py`
- Test: `tests/qubx/rate_limiting/test_resilient_integration.py` (docker, `pytest.mark.integration`)

**Interfaces:**
- Consumes: `IRateLimitBackend`, `InMemoryBackend` (backend.py), `RedisBackend` (redis_backend.py), `from qubx import logger`
- Produces: `ResilientRateLimitBackend(primary: IRateLimitBackend, fallback: IRateLimitBackend | None = None, breaker_cooldown_s: float = 30.0)` implementing the full `IRateLimitBackend` surface + `close()`; `RedisBackend(redis_url, *, socket_connect_timeout=2.0, socket_timeout=5.0, socket_keepalive=True, health_check_interval=30)`

- [ ] **Step 1: Failing unit tests for the composite**

`tests/qubx/rate_limiting/test_resilient.py` — a `FakeBackend(IRateLimitBackend)`
records calls and can be set to raise or to block on an `asyncio.Event`:

```python
class FakeBackend(IRateLimitBackend):
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.get_calls = 0
        self.set_calls = 0
        self.closed = False
        self.raise_exc: Exception | None = None
        self.block_on: asyncio.Event | None = None

    async def acquire(self, key, weight, capacity, refill_rate) -> float:
        self.acquire_calls += 1
        if self.block_on is not None:
            await self.block_on.wait()
        if self.raise_exc is not None:
            raise self.raise_exc
        return 0.0

    async def get_remaining(self, key, capacity=0, refill_rate=0):
        self.get_calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return 1.0

    async def set_remaining(self, key, remaining, capacity=0, refill_rate=0) -> None:
        self.set_calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc

    async def close(self) -> None:
        self.closed = True
```

Tests (all `async def` via anyio/asyncio marker used elsewhere in this test dir;
no sleeps — force breaker timing by assigning `backend._broken_until` directly):

1. `test_passthrough_when_primary_healthy` — acquire/get/set hit primary; fallback untouched.
2. `test_error_opens_breaker_and_serves_fallback` — primary raises `ConnectionError`; acquire returns via fallback; a second acquire does not touch primary (`acquire_calls` stays 1).
3. `test_probe_after_cooldown_single_caller` — after the error, set `resilient._broken_until = 0.0`; primary healed (`raise_exc = None`); next acquire probes primary and closes the breaker; the following acquire also hits primary.
4. `test_concurrent_callers_stay_local_while_probe_inflight` — primary healed but blocked on an `Event`; force probe due; start task A (probes, blocks), then await call B → must be served by fallback (primary `acquire_calls == 1`); release the event; A completes; next call hits primary.
5. `test_probe_failure_reopens` — probe raises again → immediate next call goes to fallback without touching primary.
6. `test_set_and_get_follow_policy` — with breaker open, `get_remaining`/`set_remaining` route to fallback.
7. `test_cancelled_error_propagates` — primary raises `asyncio.CancelledError`: it propagates out of acquire and the breaker stays closed.
8. `test_close_closes_both` — `close()` closes primary and fallback.

`tests/qubx/rate_limiting/test_redis_client_options.py` — mirror
`tests/qubx/state/test_redis_client_options.py`: monkeypatch
`redis.asyncio.from_url` with a MagicMock (whose `register_script` returns
MagicMocks), invoke `RedisBackend("redis://x")._scripts_for_current_loop()`
inside a running loop, and assert `from_url` received
`socket_connect_timeout=2.0, socket_timeout=5.0, socket_keepalive=True,
health_check_interval=30` (plus the existing `decode_responses=True,
single_connection_client=True`).

- [ ] **Step 2: Run tests, verify they fail** (`ResilientRateLimitBackend` doesn't exist; `from_url` lacks kwargs)

- [ ] **Step 3: Implement**

`src/qubx/rate_limiting/backend.py` — append to `IRateLimitBackend` (non-abstract):

```python
    async def close(self) -> None:
        """Release backend resources for the current event loop. Default: no-op."""
        return None
```

`src/qubx/rate_limiting/redis_backend.py` — ctor gains keyword-only params with
the RL-D1 defaults, stored as a `self._client_kwargs` dict merged into the
existing `from_url` call in `_scripts_for_current_loop` (keep
`decode_responses=True, single_connection_client=True`).

`src/qubx/rate_limiting/resilient.py`:

```python
import time

from redis.exceptions import RedisError

from qubx import logger

from .backend import InMemoryBackend, IRateLimitBackend

_BACKEND_ERRORS = (RedisError, OSError)


class ResilientRateLimitBackend(IRateLimitBackend):
    """Composite: primary (redis) with local fallback and a circuit breaker.

    Primary healthy → pass-through. Primary error → breaker opens for
    breaker_cooldown_s and calls are served by the local fallback (per-bot
    pacing, no cross-bot coordination). Cooldown expiry → exactly one caller
    probes the primary; concurrent callers stay on the fallback until the
    probe resolves. State is plain monotonic floats shared across event
    loops — races are benign (worst case: two probes).
    """

    def __init__(
        self,
        primary: IRateLimitBackend,
        fallback: IRateLimitBackend | None = None,
        breaker_cooldown_s: float = 30.0,
    ):
        self._primary = primary
        self._fallback = fallback or InMemoryBackend()
        self._breaker_cooldown_s = breaker_cooldown_s
        self._broken = False
        self._broken_until = 0.0
        self._probe_inflight = False

    def _use_primary(self) -> bool:
        """Route decision for this call; claims the probe slot when one is due."""
        if not self._broken:
            return True
        if self._probe_inflight or time.monotonic() < self._broken_until:
            return False
        self._probe_inflight = True
        return True

    def _on_success(self) -> None:
        self._probe_inflight = False
        if self._broken:
            self._broken = False
            logger.info("Rate-limit redis backend recovered — resuming cross-bot coordination")

    def _on_error(self, exc: Exception) -> None:
        self._probe_inflight = False
        if not self._broken:
            logger.warning(
                f"Rate-limit redis backend unavailable ({exc!r}) — "
                f"falling back to local buckets for {self._breaker_cooldown_s:.0f}s"
            )
        self._broken = True
        self._broken_until = time.monotonic() + self._breaker_cooldown_s

    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        if self._use_primary():
            try:
                waited = await self._primary.acquire(key, weight, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e)
            else:
                self._on_success()
                return waited
        return await self._fallback.acquire(key, weight, capacity, refill_rate)

    async def get_remaining(self, key: str, capacity: float = 0, refill_rate: float = 0) -> float | None:
        if self._use_primary():
            try:
                remaining = await self._primary.get_remaining(key, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e)
            else:
                self._on_success()
                return remaining
        return await self._fallback.get_remaining(key, capacity, refill_rate)

    async def set_remaining(self, key: str, remaining: float, capacity: float = 0, refill_rate: float = 0) -> None:
        if self._use_primary():
            try:
                await self._primary.set_remaining(key, remaining, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e)
            else:
                self._on_success()
                return
        await self._fallback.set_remaining(key, remaining, capacity, refill_rate)

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()
```

`src/qubx/rate_limiting/manager.py::_create_backend` — wrap:

```python
        if config.backend == "redis" and config.redis_url:
            try:
                from .redis_backend import RedisBackend
                from .resilient import ResilientRateLimitBackend

                return ResilientRateLimitBackend(RedisBackend(config.redis_url))
            except Exception as e:
                logger.error(f"Failed to create Redis rate limit backend: {e}, falling back to local")
        return InMemoryBackend()
```

`__init__.py`: add `ResilientRateLimitBackend` to the imports and `__all__`.

- [ ] **Step 4: Run unit tests, verify pass** (`uv run pytest tests/qubx/rate_limiting/ -q`)

- [ ] **Step 5: Integration test** — `tests/qubx/rate_limiting/test_resilient_integration.py`,
mirroring `tests/qubx/state/test_safe_integration.py`'s docker fixture (own
container name e.g. `qubx-rl-itest-redis` and own port, `redis:7-alpine`,
`pytestmark = pytest.mark.integration`):
  1. Build `ResilientRateLimitBackend(RedisBackend(url))`; `acquire` succeeds via primary (verify the `ratelimit:` key exists through a separate control client).
  2. `docker pause` → next `acquire` returns in < 8 s (socket timeout → fallback) and the one after in < 0.5 s (breaker open, no timeout tax).
  3. `docker unpause` → force `_broken_until = 0.0` → `acquire` routes to primary again (control client sees the key's `last_refill` advance).

- [ ] **Step 6: Full check + commit**

```bash
uv run ruff check src/qubx/rate_limiting tests/qubx/rate_limiting && uv run ruff format --check src/qubx/rate_limiting tests/qubx/rate_limiting
uv run pytest tests/qubx/rate_limiting -q
git add -A && git commit -m "feat(rate-limiting): resilient redis backend — client timeouts, local fallback, 30s circuit breaker"
```

---

## Self-review notes

- Spec coverage: §1→T3/T4, §2→T2, §3→T1, §4→T7, §5→T8, §6→T5, §7→T6, §8 (no `processing.py` change) → enforced by touching nothing there; failure timeline → T9 integration; testing section → per-task tests + T9; rollout → PR to dev only.
- `delete()` returning optimistic `True` is a documented semantic change (was: actual backend result); acceptable — no caller in qubx/strategies branches on it (verified: only frab/quantkit use save/load).
- Task 6 test constructs `BaseHealthMonitor` minimally; if the constructor needs extra args the task instructs reusing the existing fixture pattern from `tests/qubx/health/test_base.py` — not a placeholder, the pattern exists in-repo.
