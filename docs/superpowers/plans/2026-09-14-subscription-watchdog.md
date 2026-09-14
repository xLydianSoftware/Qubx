# Subscription Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make qubx's stale-data recovery unable to destroy the subscription universe it is trying to repair, so a venue outage can never leave a bot silently blind.

**Architecture:** `SubscriptionManager` becomes the owner of subscription *intent* (`_desired`) instead of reading current state back from the data provider. A new `SubscriptionWatchdog` class owns the monitoring thread and all repair policy, calling `reconcile()` to drive providers toward intent. Because intent survives a failed repair, every failure is retried rather than permanent. Per-exchange facts move into `BaseHealthMonitor`, and a fully dark exchange publishes the existing (never-written) `DegradeReason.EXCHANGE_MAINTENANCE`.

**Tech Stack:** Python 3.12, pytest (`asyncio_mode = "auto"`, `pythonpath = ["src"]`), `unittest.mock`, numpy `datetime64` time providers.

**Spec:** `docs/superpowers/specs/2026-09-14-subscription-watchdog-design.md` — read it before Task 1. Decisions are referenced below as D1–D8.

**Worktree:** `~/devs/Qubx/.worktrees/subscription-watchdog`, branch `fix/subscription-watchdog`. All paths below are relative to that worktree root. Run everything with `uv run pytest ...`.

## Global Constraints

- **No `IDataProvider` interface change** (D1). Do not add, remove, or change the signature of any method on `IDataProvider`. `qubx-lighter` and `qubx-hyperliquid` must build against this branch unmodified.
- **Modern Python types only:** `list`, `dict`, `set`, `| None`, `tuple`. Never `typing.List`/`Optional`.
- **Logging:** `from qubx import logger`.
- **Comments:** only for non-obvious invariants. No narration, no restating the code.
- **Staleness thresholds** are `qubx.health.base.STALE_THRESHOLDS`: `quote` 10min, `orderbook` 10min, `trade` 30min. Never hardcode these values; read them from the dict.
- **Watchdog data types** are `_WATCHDOG_DATA_TYPES = frozenset({DataType.QUOTE, DataType.ORDERBOOK, DataType.TRADE})` in `subscription.py:27`.
- **The repair must never call `IHealthMonitor.unsubscribe`** — it pops `_last_event_time` (`health/base.py:207`), destroying the signal used to verify the repair.
- Conventional commits, no co-authored-by lines. Do not push; commit only.

---

### Task 1: Fix `synchronized` to use a real per-instance lock

`synchronized` creates its lock at decoration time, so every decorated function has its own lock and they do not exclude each other. `SubscriptionManager.subscribe`, `unsubscribe` and `commit` are each `@synchronized` and can therefore run concurrently. This is a precondition for every later task (D8): `reconcile()` racing `commit()` would corrupt `_desired`.

**Files:**
- Modify: `src/qubx/utils/misc.py:507-516`
- Modify: `src/qubx/core/mixins/subscription.py:104,108,116`
- Modify: `src/qubx/connectors/tardis/data.py:349`
- Test: `tests/qubx/utils/test_synchronized.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `qubx.utils.misc.synchronized` — same decorator name, now acquiring `self._instance_lock`, an `threading.RLock` created lazily per instance. Later tasks rely on `SubscriptionManager` methods being mutually exclusive.

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/utils/test_synchronized.py`:

```python
import threading
import time

from qubx.utils.misc import synchronized


class Counter:
    def __init__(self) -> None:
        self.inside = 0
        self.max_concurrent = 0

    @synchronized
    def a(self) -> None:
        self._body()

    @synchronized
    def b(self) -> None:
        self._body()

    def _body(self) -> None:
        self.inside += 1
        self.max_concurrent = max(self.max_concurrent, self.inside)
        time.sleep(0.01)
        self.inside -= 1


def test_different_methods_on_same_instance_are_mutually_exclusive():
    c = Counter()
    threads = [threading.Thread(target=c.a) for _ in range(5)]
    threads += [threading.Thread(target=c.b) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.max_concurrent == 1


def test_different_instances_do_not_block_each_other():
    first, second = Counter(), Counter()
    done = threading.Event()

    def hold() -> None:
        first.a()
        done.set()

    t = threading.Thread(target=hold)
    t.start()
    second.a()
    t.join()
    assert done.is_set()


def test_reentrant_on_same_instance():
    class Nested:
        @synchronized
        def outer(self) -> str:
            return self.inner()

        @synchronized
        def inner(self) -> str:
            return "ok"

    assert Nested().outer() == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/utils/test_synchronized.py -v`
Expected: `test_different_methods_on_same_instance_are_mutually_exclusive` FAILS with `assert 2 == 1` (or a higher number) — the two methods hold separate locks. `test_reentrant_on_same_instance` may also fail or deadlock; if it hangs, that confirms the same defect.

- [ ] **Step 3: Write minimal implementation**

Replace `synchronized` in `src/qubx/utils/misc.py`:

```python
def synchronized(func: Callable):
    """Serialize calls to the decorated methods of one instance.

    The lock is per-instance and shared by every method decorated in that class, so
    subscribe/unsubscribe/commit exclude each other. Re-entrant: commit() -> _apply_swap()
    -> reconcile() is one call chain on one thread.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_synchronized_lock", None)
        if lock is None:
            # - double-checked under a module-level lock: two threads may reach an
            #   instance's first synchronized call at once and must agree on one lock
            with _SYNCHRONIZED_INIT_LOCK:
                lock = getattr(self, "_synchronized_lock", None)
                if lock is None:
                    lock = RLock()
                    object.__setattr__(self, "_synchronized_lock", lock)
        with lock:
            return func(self, *args, **kwargs)

    return wrapper
```

Add near the top of the file, next to the existing imports:

```python
_SYNCHRONIZED_INIT_LOCK = Lock()
```

`Lock` is already imported in `misc.py`; add `RLock` to that same import.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/utils/test_synchronized.py -v`
Expected: 3 passed.

- [ ] **Step 5: Verify no existing caller regressed**

Run: `uv run pytest tests/qubx/core/subscription_test.py tests/qubx/connectors -q`
Expected: same pass/fail set as on `main`. The only other `@synchronized` user is `src/qubx/connectors/tardis/data.py:349`, a single decorated method — per-instance locking is strictly safer there.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/utils/misc.py tests/qubx/utils/test_synchronized.py
git commit -m "fix(utils): synchronized locks per instance, not per function

The lock was built at decoration time, so each decorated function had its
own. SubscriptionManager.subscribe/unsubscribe/commit therefore did not
exclude each other. Now one RLock per instance, shared by every decorated
method and re-entrant for commit() -> _apply_swap() chains."
```

---

### Task 2: `SubscriptionManager` owns `_desired`

Stop reading current subscription state back from the provider (D2). This is the fix: after this task, a provider that loses its registry no longer takes the universe with it.

**Files:**
- Modify: `src/qubx/core/mixins/subscription.py` — `__init__` (~line 63-102), `_apply_swap` (~line 198-243), `has_subscription` (~249), `get_subscribed_instruments` (~263)
- Test: `tests/qubx/core/test_subscription_intent.py` (create)

**Interfaces:**
- Consumes: Task 1's per-instance lock.
- Produces:
  - `SubscriptionManager._desired: dict[str, set[Instrument]]` — subscription key (e.g. `"orderbook(0, 1)"`) to instruments.
  - `SubscriptionManager._unsupported: set[tuple[str, str]]` — `(exchange, subscription_key)` pairs the venue rejected with `NotSupported`.
  - `SubscriptionManager.get_subscribed_instruments(subscription_type: str | None = None) -> list[Instrument]` — now reads `_desired`.
  - `SubscriptionManager.has_subscription(instrument: Instrument, subscription_type: str) -> bool` — now reads `_desired`.

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/core/test_subscription_intent.py`:

```python
from unittest.mock import Mock

import pytest

from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.mixins.subscription import SubscriptionManager
from qubx.health.dummy import DummyHealthMonitor

EXCHANGE = "BINANCE.UM"


def _instrument(symbol: str) -> Instrument:
    instr = lookup.find_symbol(EXCHANGE, symbol)
    assert instr is not None
    return instr


@pytest.fixture
def manager_and_provider():
    provider = Mock()
    provider.is_simulation = False
    provider.exchange.return_value = EXCHANGE
    provider.get_subscribed_instruments.return_value = []
    provider.get_subscriptions.return_value = []
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    manager = SubscriptionManager(
        time_provider, [provider], CtrlChannel("test"), DummyHealthMonitor(), StrategyState()
    )
    return manager, provider


def test_intent_survives_provider_losing_its_registry(manager_and_provider):
    """The 2026-09-13 incident, reduced: the provider forgets everything and the
    manager must still know what the universe is."""
    manager, provider = manager_and_provider
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")

    manager.subscribe(DataType.ORDERBOOK, [btc, eth])
    manager.commit()

    # provider wipes its own bookkeeping, as LighterDataProvider.unsubscribe did
    provider.get_subscribed_instruments.return_value = []
    provider.get_subscriptions.return_value = []

    assert set(manager.get_subscribed_instruments(DataType.ORDERBOOK)) == {btc, eth}
    assert manager.has_subscription(btc, DataType.ORDERBOOK)


def test_desired_tracks_adds_and_removes(manager_and_provider):
    manager, provider = manager_and_provider
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")

    manager.subscribe(DataType.ORDERBOOK, [btc, eth])
    manager.commit()
    assert manager._desired[DataType.ORDERBOOK] == {btc, eth}

    manager.unsubscribe(DataType.ORDERBOOK, eth)
    manager.commit()
    assert manager._desired[DataType.ORDERBOOK] == {btc}
    assert not manager.has_subscription(eth, DataType.ORDERBOOK)


def test_not_supported_is_recorded_once(manager_and_provider):
    from qubx.core.exceptions import NotSupported

    manager, provider = manager_and_provider
    provider.subscribe.side_effect = NotSupported("no orderbook here")

    manager.subscribe(DataType.ORDERBOOK, _instrument("BTCUSDT"))
    manager.commit()

    assert (EXCHANGE, DataType.ORDERBOOK) in manager._unsupported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/test_subscription_intent.py -v`
Expected: `test_intent_survives_provider_losing_its_registry` FAILS (returns `[]` — it reads the provider). `test_desired_tracks_adds_and_removes` and `test_not_supported_is_recorded_once` FAIL with `AttributeError: 'SubscriptionManager' object has no attribute '_desired'` / `'_unsupported'`.

- [ ] **Step 3: Write minimal implementation**

In `__init__`, after `self._pending_stream_unsubscriptions = defaultdict(set)`:

```python
# - the desired universe: authoritative, and deliberately NOT read back from the
#   providers. A provider that loses its registry (a failed repair, a dropped
#   socket) must not take the universe with it.
self._desired: dict[str, set[Instrument]] = defaultdict(set)
self._unsupported: set[tuple[str, str]] = set()
```

In `_apply_swap`, replace line 201:

```python
_current_sub_instruments = set(self.get_subscribed_instruments(_sub))
```

with:

```python
_current_sub_instruments = set(self._desired.get(_sub, set()))
```

and line 209:

```python
_added_instruments.update(self.get_subscribed_instruments())
```

with:

```python
_added_instruments.update({i for instrs in self._desired.values() for i in instrs})
```

After `_updated_instruments` is computed (immediately after the `_updated_instruments = ...` line), record intent before any provider call:

```python
# - intent is recorded BEFORE the provider calls, so a raising provider leaves
#   the universe intact and the next reconcile repairs it
if _updated_instruments:
    self._desired[_sub] = set(_updated_instruments)
else:
    self._desired.pop(_sub, None)
```

In the `except NotSupported` handler, record the pair:

```python
except NotSupported as e:
    self._unsupported.add((_exchange, _sub))
    logger.warning(f"Subscription not supported for {_exchange}: {e}")
```

Replace `has_subscription` and `get_subscribed_instruments`:

```python
def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
    return instrument in self._desired.get(subscription_type, set())

def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
    if subscription_type is not None:
        return list(self._desired.get(subscription_type, set()))
    return list({i for instrs in self._desired.values() for i in instrs})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/test_subscription_intent.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the existing subscription suite**

Run: `uv run pytest tests/qubx/core/subscription_test.py tests/qubx/core/test_funding_payment_subscription.py -v`
Expected: all pass. These assert on the calls made to the mock provider, which this task does not change. If one fails because it asserted on `get_subscribed_instruments` being read from the provider, update the test to the new source of truth and say so in the commit message — that is the intended behaviour change, not a regression.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/mixins/subscription.py tests/qubx/core/test_subscription_intent.py
git commit -m "feat(core): SubscriptionManager owns the desired universe

_apply_swap read current state back from the data provider, making the
provider's registry the only record of the universe anywhere. A provider
that dropped it (a failed repair during a venue outage) destroyed the
universe system-wide, and nothing could rebuild it.

Intent now lives in _desired, recorded before any provider call, so a
raising provider leaves it intact."
```

---

### Task 3: `SubscriptionManager.reconcile()`

Drive providers toward `_desired`. This is the only method the watchdog calls.

**Files:**
- Modify: `src/qubx/core/mixins/subscription.py` (add method after `_apply_swap`)
- Test: `tests/qubx/core/test_subscription_intent.py` (extend)

**Interfaces:**
- Consumes: `_desired`, `_unsupported` from Task 2.
- Produces: `SubscriptionManager.reconcile(refresh: set[Instrument] = frozenset()) -> None` — decorated `@synchronized`. With a non-empty `refresh`, unsubscribes exactly those instruments, sleeps `_repair_settle_seconds`, then re-subscribes the full desired set for that key. With an empty `refresh`, re-asserts the full desired set with no unsubscribe. Exceptions are caught per exchange and logged; `_desired` is never mutated.
- Produces: `SubscriptionManager._repair_settle_seconds: float = 3.0` — overridable in tests to keep them fast.

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/test_subscription_intent.py`:

```python
def test_reconcile_refresh_unsubscribes_then_resubscribes_full_set(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    manager.subscribe(DataType.ORDERBOOK, [btc, eth])
    manager.commit()
    provider.reset_mock()

    manager.reconcile(refresh={btc})

    provider.unsubscribe.assert_called_once_with(DataType.ORDERBOOK, {btc})
    provider.subscribe.assert_called_once_with(DataType.ORDERBOOK, {btc, eth}, reset=True)


def test_reconcile_without_refresh_reasserts_and_never_unsubscribes(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()

    manager.reconcile()

    provider.unsubscribe.assert_not_called()
    provider.subscribe.assert_called_once_with(DataType.ORDERBOOK, {btc}, reset=True)


def test_reconcile_failure_leaves_desired_intact_and_retries(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()
    provider.subscribe.side_effect = TimeoutError("WebSocket connection not ready after 5.0s")

    manager.reconcile(refresh={btc})

    assert manager._desired[DataType.ORDERBOOK] == {btc}

    provider.subscribe.side_effect = None
    manager.reconcile(refresh={btc})
    provider.subscribe.assert_called_with(DataType.ORDERBOOK, {btc}, reset=True)


def test_reconcile_skips_unsupported_pairs(manager_and_provider):
    from qubx.core.exceptions import NotSupported

    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    provider.subscribe.side_effect = NotSupported("nope")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()
    provider.subscribe.side_effect = None

    manager.reconcile(refresh={btc})

    provider.subscribe.assert_not_called()
    provider.unsubscribe.assert_not_called()


def test_reconcile_isolates_a_failing_exchange(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.subscribe(DataType.TRADE, btc)
    manager.commit()
    provider.reset_mock()

    calls: list[str] = []

    def record(sub, instruments, reset=False):
        calls.append(sub)
        if sub == DataType.ORDERBOOK:
            raise TimeoutError("boom")

    provider.subscribe.side_effect = record
    manager.reconcile()

    assert DataType.ORDERBOOK in calls and DataType.TRADE in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/test_subscription_intent.py -v -k reconcile`
Expected: 5 FAIL with `AttributeError: 'SubscriptionManager' object has no attribute 'reconcile'`.

- [ ] **Step 3: Write minimal implementation**

Add `self._repair_settle_seconds: float = 3.0` in `__init__` next to `_desired`, then add after `_apply_swap`:

```python
@synchronized
def reconcile(self, refresh: set[Instrument] = frozenset()) -> None:
    """Drive every provider toward `_desired`.

    `refresh` names instruments whose transport must be re-established even though
    intent has not changed — a provider may believe it is subscribed while delivering
    nothing. Raising is safe: `_desired` is untouched, so the caller retries.
    """
    for _sub, _instruments in list(self._desired.items()):
        if not _instruments:
            continue
        _by_exchange: dict[str, set[Instrument]] = defaultdict(set)
        for instr in _instruments:
            _by_exchange[instr.exchange].add(instr)

        for _exchange, _desired_here in _by_exchange.items():
            if (_exchange, _sub) in self._unsupported:
                continue
            _refresh_here = {i for i in refresh if i.exchange == _exchange} & _desired_here
            try:
                _data_provider = self._get_data_provider(_exchange)
                if _refresh_here:
                    _data_provider.unsubscribe(_sub, _refresh_here)
                    # - the settle delay guards against the venue processing our
                    #   unsubscribe after the resubscribe. It bounds churn; it is NOT
                    #   the safety net - that is the caller's retry (D4).
                    time.sleep(self._repair_settle_seconds)
                _data_provider.subscribe(_sub, set(_desired_here), reset=True)
            except NotSupported as e:
                self._unsupported.add((_exchange, _sub))
                logger.warning(f"[{_exchange}] :: {_sub} not supported: {e}")
            except Exception as e:
                logger.error(f"[{_exchange}] :: reconcile of {_sub} failed: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/test_subscription_intent.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/mixins/subscription.py tests/qubx/core/test_subscription_intent.py
git commit -m "feat(core): add SubscriptionManager.reconcile

Drives providers toward _desired. A refresh set re-establishes transport
for instruments intent already covers; an empty refresh re-asserts the
whole universe after a reconnect. Failures are caught per exchange and
leave _desired intact, so the caller retries."
```

---

### Task 4: Per-exchange facts in the health monitor

Add `subscribed_at` tracking and the `ExchangeDataStatus` snapshot the watchdog classifies on (D5b, §5).

**Files:**
- Create: `src/qubx/health/status.py`
- Modify: `src/qubx/health/base.py` — `__init__` (~line 100-112), `subscribe` (185-193), `unsubscribe` (195-207); add `get_exchange_data_status`
- Modify: `src/qubx/health/dummy.py` — add the same method
- Modify: `src/qubx/core/interfaces.py` — add the method to `IHealthMonitor` near line 2110
- Test: `tests/qubx/health/test_exchange_data_status.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `qubx.health.status.ExchangeDataStatus` — frozen slotted dataclass with fields `exchange: str`, `connected: bool | None`, `subscribed: int`, `stale: int`, `in_grace: int`, `last_event_time: dt_64 | None`.
  - `IHealthMonitor.get_exchange_data_status(exchange: str, subscribed: dict[str, set[Instrument]]) -> ExchangeDataStatus` — `subscribed` maps base data type to instruments. Counts exclude instruments still in their grace window.
  - `BaseHealthMonitor._subscribed_at: dict[tuple[Instrument, str], dt_64]`.

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/health/test_exchange_data_status.py`:

```python
import numpy as np

from qubx.core.basics import DataType, ITimeProvider
from qubx.core.lookups import lookup
from qubx.health.base import BaseHealthMonitor

EXCHANGE = "BINANCE.UM"


class FixedTime(ITimeProvider):
    def __init__(self) -> None:
        self.now = np.datetime64("2026-09-13T00:00:00", "ns")

    def time(self) -> np.datetime64:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + np.timedelta64(minutes, "m")


def _instrument(symbol: str):
    instr = lookup.find_symbol(EXCHANGE, symbol)
    assert instr is not None
    return instr


def _monitor():
    clock = FixedTime()
    return BaseHealthMonitor(clock), clock


def test_fresh_subscription_is_in_grace_not_stale():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.in_grace == 1
    assert status.subscribed == 0
    assert status.stale == 0


def test_becomes_eligible_and_stale_after_the_grace_window():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)  # orderbook threshold is 10min

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.in_grace == 0
    assert status.subscribed == 1
    assert status.stale == 1


def test_delivering_instrument_is_not_stale():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.stale == 0
    assert status.subscribed == 1
    assert status.last_event_time == clock.time()


def test_counts_are_per_instrument_not_max_over_the_exchange():
    """One live instrument must not mask nineteen dead ones."""
    monitor, clock = _monitor()
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc, eth}})

    assert status.subscribed == 2
    assert status.stale == 1


def test_connected_is_none_when_no_callback_registered():
    monitor, _ = _monitor()
    status = monitor.get_exchange_data_status(EXCHANGE, {})
    assert status.connected is None


def test_connected_reflects_the_registered_callback():
    monitor, _ = _monitor()
    monitor.set_is_connected(EXCHANGE, lambda: False)
    assert monitor.get_exchange_data_status(EXCHANGE, {}).connected is False


def test_unsubscribe_clears_subscribed_at():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.unsubscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(5)

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.in_grace == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/health/test_exchange_data_status.py -v`
Expected: all FAIL with `AttributeError: 'BaseHealthMonitor' object has no attribute 'get_exchange_data_status'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qubx/health/status.py`:

```python
from dataclasses import dataclass

from qubx.core.basics import dt_64


@dataclass(frozen=True, slots=True)
class ExchangeDataStatus:
    """Per-exchange data-flow facts, counted per instrument.

    `subscribed` and `stale` cover only instruments past their grace window; a freshly
    subscribed instrument has no last-event time and would otherwise read as stale,
    which after a full universe swap would make the exchange look entirely dark.
    """

    exchange: str
    connected: bool | None
    subscribed: int
    stale: int
    in_grace: int
    last_event_time: dt_64 | None
```

In `BaseHealthMonitor.__init__`, next to `self._active_subscriptions`:

```python
self._subscribed_at: dict[tuple[Instrument, str], dt_64] = {}
```

In `subscribe`, after the `_active_subscriptions.add(...)` line:

```python
self._subscribed_at[(instrument, DataType.from_str(event_type)[0])] = self.time_provider.time()
```

In `unsubscribe`, next to the existing `self._last_event_time.pop(key, None)`:

```python
self._subscribed_at.pop(key, None)
```

Add the new method after `is_exchange_stale`:

```python
def get_exchange_data_status(
    self, exchange: str, subscribed: dict[str, set[Instrument]]
) -> ExchangeDataStatus:
    now = self.time_provider.time()
    n_subscribed = n_stale = n_grace = 0
    last_event: dt_64 | None = None

    for event_type, instruments in subscribed.items():
        base_type = DataType.from_str(event_type)[0]
        threshold = STALE_THRESHOLDS.get(str(base_type))
        if threshold is None:
            continue
        grace = convert_tf_str_td64(threshold)
        for instrument in instruments:
            if instrument.exchange != exchange:
                continue
            key = (instrument, base_type)
            subscribed_at = self._subscribed_at.get(key)
            if subscribed_at is not None and now - subscribed_at < grace:
                n_grace += 1
                continue
            n_subscribed += 1
            if self.is_stale(instrument, str(base_type)):
                n_stale += 1
            event_time = self._last_event_time.get(key)
            if event_time is not None and (last_event is None or event_time > last_event):
                last_event = event_time

    connected: bool | None = None
    if exchange in self._is_connected_callbacks:
        try:
            connected = bool(self._is_connected_callbacks[exchange]())
        except Exception:
            connected = False

    return ExchangeDataStatus(
        exchange=exchange,
        connected=connected,
        subscribed=n_subscribed,
        stale=n_stale,
        in_grace=n_grace,
        last_event_time=last_event,
    )
```

Add `from qubx.health.status import ExchangeDataStatus` to `base.py` imports. `convert_tf_str_td64` and `DataType` are already imported there.

Add to `IHealthMonitor` in `src/qubx/core/interfaces.py`, after `unsubscribe` (~line 2120):

```python
def get_exchange_data_status(
    self, exchange: str, subscribed: dict[str, set[Instrument]]
) -> "ExchangeDataStatus":
    """Per-exchange data-flow facts. `subscribed` maps data type to instruments."""
    ...
```

Import it under `TYPE_CHECKING` at the top of `interfaces.py`, not at runtime — `qubx.health.base` imports `qubx.core.interfaces`, so a runtime import here would close a cycle:

```python
if TYPE_CHECKING:
    from qubx.health.status import ExchangeDataStatus
```

`qubx/health/status.py` deliberately imports only `qubx.core.basics`, so it stays safe to import at runtime from `qubx/health/dummy.py` and from the watchdog in Task 5.

Add to `DummyHealthMonitor` in `src/qubx/health/dummy.py`:

```python
def get_exchange_data_status(
    self, exchange: str, subscribed: dict[str, set[Instrument]]
) -> ExchangeDataStatus:
    return ExchangeDataStatus(
        exchange=exchange, connected=None, subscribed=0, stale=0, in_grace=0, last_event_time=None
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/health/test_exchange_data_status.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the health suite**

Run: `uv run pytest tests/qubx/health -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/health/status.py src/qubx/health/base.py src/qubx/health/dummy.py src/qubx/core/interfaces.py tests/qubx/health/test_exchange_data_status.py
git commit -m "feat(health): per-exchange data-flow facts counted per instrument

get_exchange_data_status reports subscribed/stale/in_grace counts rather
than a max() over the exchange, which one live instrument could clear for
nineteen dead ones. Records subscribed_at so a freshly subscribed
instrument - which has no last-event time and therefore reads as stale -
is held in a grace window instead of counted."
```

---

### Task 5: `SubscriptionWatchdog`

The thread and all policy: classification, grace, repair, verification by advancement, capped backoff, and `EXCHANGE_MAINTENANCE` publication (D5, D5a, D5b, D6, §4, §6).

**Files:**
- Create: `src/qubx/core/subscription_watchdog.py`
- Test: `tests/qubx/core/test_subscription_watchdog.py` (create)

**Interfaces:**
- Consumes: `ExchangeDataStatus` and `get_exchange_data_status` (Task 4); `reconcile` (Task 3).
- Produces:
  - `qubx.core.subscription_watchdog.ExchangeClassification` — `StrEnum` with members `OK`, `DARK`, `PARTIAL`.
  - `SubscriptionWatchdog(data_providers, health_monitor, status, reconcile_fn, subscriptions_fn, strategy_state, interval_seconds=30.0)` where `reconcile_fn: Callable[[set[Instrument]], None]` and `subscriptions_fn: Callable[[], dict[str, set[Instrument]]]` returns the desired universe keyed by subscription type.
  - `SubscriptionWatchdog.classify(status: ExchangeDataStatus) -> ExchangeClassification` — pure, per §4.2.
  - `SubscriptionWatchdog.tick() -> None` — one pass over every live provider. Tests call this directly; the thread only calls it on a timer.
  - `SubscriptionWatchdog.start() -> None` / `stop() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/core/test_subscription_watchdog.py`:

```python
from unittest.mock import Mock

import numpy as np
import pytest

from qubx.core.basics import DataType, ITimeProvider
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.status import ContextStatus, DegradeReason, QubxStatus
from qubx.core.subscription_watchdog import ExchangeClassification, SubscriptionWatchdog
from qubx.health.base import BaseHealthMonitor
from qubx.health.status import ExchangeDataStatus

EXCHANGE = "BINANCE.UM"


class FixedTime(ITimeProvider):
    def __init__(self) -> None:
        self.now = np.datetime64("2026-09-13T00:00:00", "ns")

    def time(self) -> np.datetime64:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + np.timedelta64(minutes, "m")


def _instrument(symbol: str):
    instr = lookup.find_symbol(EXCHANGE, symbol)
    assert instr is not None
    return instr


def _status(**kw) -> ExchangeDataStatus:
    base = dict(
        exchange=EXCHANGE, connected=True, subscribed=0, stale=0, in_grace=0, last_event_time=None
    )
    base.update(kw)
    return ExchangeDataStatus(**base)


@pytest.fixture
def rig():
    clock = FixedTime()
    monitor = BaseHealthMonitor(clock)
    status = ContextStatus()
    monitor.set_status(status)
    provider = Mock()
    provider.is_simulation = False
    provider.exchange.return_value = EXCHANGE
    state = StrategyState()
    state.is_on_warmup_finished_called = True
    universe: dict[str, set] = {}
    reconciled: list[set] = []

    watchdog = SubscriptionWatchdog(
        data_providers=[provider],
        health_monitor=monitor,
        status=status,
        reconcile_fn=lambda refresh=frozenset(): reconciled.append(set(refresh)),
        subscriptions_fn=lambda: universe,
        strategy_state=state,
        interval_seconds=30.0,
    )
    return watchdog, monitor, status, clock, universe, reconciled


# --- classification (spec 4.2) ---

def test_classify_ok():
    assert SubscriptionWatchdog.classify(_status(subscribed=5, stale=0)) is ExchangeClassification.OK


def test_classify_dark_when_disconnected():
    s = _status(connected=False, subscribed=5, stale=0)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.DARK


def test_classify_dark_when_all_stale():
    s = _status(subscribed=5, stale=5)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.DARK


def test_classify_partial_when_some_stale():
    s = _status(subscribed=5, stale=2)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.PARTIAL


def test_single_instrument_all_stale_is_partial_not_dark():
    s = _status(subscribed=1, stale=1)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.PARTIAL


def test_connected_none_does_not_make_it_dark():
    s = _status(connected=None, subscribed=5, stale=0)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.OK


# --- maintenance publication (spec 6) ---

def test_maintenance_needs_two_consecutive_dark_ticks(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    assert status.info.status is QubxStatus.NORMAL

    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)
    assert status.info.degradations[0].reason is DegradeReason.EXCHANGE_MAINTENANCE


def test_maintenance_clears_on_first_delivering_tick(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    watchdog.tick()
    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)

    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL


def test_dark_issues_no_repair(rig):
    """The 11:16 regression: 21 simultaneous subscribes tripped the venue's
    message rate limit and cost the recovery."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    watchdog.tick()
    watchdog.tick()

    assert reconciled == []


def test_partial_never_publishes_maintenance(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    for _ in range(5):
        watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert reconciled and reconciled[0] == {eth}


def test_grace_prevents_false_dark_on_universe_swap(rig):
    """A full universe replacement leaves every instrument with no last-event
    time. Without the grace gate this halts trading on a healthy venue."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)

    watchdog.tick()
    watchdog.tick()
    watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert reconciled == []


# --- verification and backoff (spec 4.3, D5) ---

def test_verification_is_by_advancement_not_by_staleness(rig):
    """A trade feed that delivers once and goes quiet stays stale by threshold
    but must never be repaired again."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.TRADE] = {btc, eth}
    monitor.subscribe(btc, DataType.TRADE)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(31)  # trade threshold is 30min
    monitor.on_data_arrival(btc, DataType.TRADE, clock.time())

    watchdog.tick()
    assert reconciled and reconciled[-1] == {eth}

    # eth delivers once, then nothing for a long time
    monitor.on_data_arrival(eth, DataType.TRADE, clock.time())
    before = len(reconciled)
    for _ in range(20):
        clock.advance(1)
        watchdog.tick()

    assert monitor.is_stale(eth, DataType.TRADE) is True
    assert len(reconciled) == before


def test_unverified_repair_backs_off_and_never_stops(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    attempts = 0
    for _ in range(40):
        before = len(reconciled)
        watchdog.tick()
        attempts += len(reconciled) - before

    assert 2 <= attempts < 40, "repairs must back off but never stop"


def test_backoff_resets_after_recovery(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    for _ in range(6):
        watchdog.tick()

    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()
    assert watchdog._repair_state == {}


def test_repair_never_calls_health_monitor_unsubscribe(rig):
    """Popping _last_event_time would destroy the verification signal."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.unsubscribe = Mock(side_effect=AssertionError("repair must not unsubscribe in health"))

    for _ in range(5):
        watchdog.tick()


# --- gating ---

def test_does_nothing_before_warmup_finished(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    watchdog._strategy_state.is_on_warmup_finished_called = False
    btc = _instrument("BTCUSDT")
    universe[DataType.ORDERBOOK] = {btc}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    watchdog.tick()

    assert reconciled == []
    assert status.info.status is QubxStatus.NORMAL


def test_tick_survives_an_exception(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    watchdog._subscriptions_fn = Mock(side_effect=RuntimeError("boom"))
    watchdog.tick()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/test_subscription_watchdog.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'qubx.core.subscription_watchdog'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qubx/core/subscription_watchdog.py`:

```python
import threading
import time
from collections import defaultdict
from enum import StrEnum
from typing import Callable

from qubx import logger
from qubx.core.basics import DataType, Instrument, dt_64
from qubx.core.interfaces import IDataProvider, IHealthMonitor, StrategyState
from qubx.core.status import ContextStatus, DegradeReason
from qubx.health.status import ExchangeDataStatus
from qubx.utils.time import convert_tf_str_td64

# Data types with a continuous feed and a staleness threshold worth policing.
_WATCHDOG_DATA_TYPES = frozenset({DataType.QUOTE, DataType.ORDERBOOK, DataType.TRADE})

# Consecutive DARK ticks before EXCHANGE_MAINTENANCE is published. Every healthy
# reconnect observed during the 2026-09-13 outage completed in 1-4s, so one tick would
# flap the order path on routine blips.
_DARK_TICKS_BEFORE_MAINTENANCE = 2

# Repair backoff caps at threshold/10 - 1min for orderbook/quote, 3min for trade. Tying
# it to the type's own threshold keeps retries aggressive where feeds tick continuously
# and patient where they legitimately do not.
_BACKOFF_CAP_DIVISOR = 10


class ExchangeClassification(StrEnum):
    OK = "ok"
    DARK = "dark"
    PARTIAL = "partial"


class _RepairRecord:
    __slots__ = ("repaired_at", "interval_ticks", "ticks_until_retry")

    def __init__(self, repaired_at: dt_64, interval_ticks: int) -> None:
        self.repaired_at = repaired_at
        self.interval_ticks = interval_ticks
        self.ticks_until_retry = interval_ticks


class SubscriptionWatchdog:
    """Detects instruments whose feed has stopped and drives repair through reconcile().

    Owns no subscription state: intent lives in SubscriptionManager, so a repair this
    class fails to complete is retried rather than lost.
    """

    def __init__(
        self,
        data_providers: list[IDataProvider],
        health_monitor: IHealthMonitor,
        status: ContextStatus,
        reconcile_fn: Callable[..., None],
        subscriptions_fn: Callable[[], dict[str, set[Instrument]]],
        strategy_state: StrategyState,
        interval_seconds: float = 30.0,
    ) -> None:
        self._data_providers = data_providers
        self._health_monitor = health_monitor
        self._status = status
        self._reconcile_fn = reconcile_fn
        self._subscriptions_fn = subscriptions_fn
        self._strategy_state = strategy_state
        self._interval_seconds = interval_seconds
        self._repair_state: dict[tuple[Instrument, str], _RepairRecord] = {}
        self._dark_ticks: dict[str, int] = defaultdict(int)
        self._maintenance_held: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ----- policy -----

    @staticmethod
    def classify(status: ExchangeDataStatus) -> ExchangeClassification:
        if status.connected is False:
            return ExchangeClassification.DARK
        if status.stale == 0:
            return ExchangeClassification.OK
        if status.stale == status.subscribed and status.subscribed >= 2:
            return ExchangeClassification.DARK
        return ExchangeClassification.PARTIAL

    # ----- loop -----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SubscriptionWatchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.tick()

    def tick(self) -> None:
        try:
            if not self._strategy_state.is_on_warmup_finished_called:
                return
            universe = self._watchdog_subscriptions()
            for provider in self._data_providers:
                if provider.is_simulation:
                    continue
                self._tick_exchange(provider.exchange(), universe)
        except Exception as e:
            logger.error(f"[SubscriptionWatchdog] :: tick failed: {e}")

    def _watchdog_subscriptions(self) -> dict[str, set[Instrument]]:
        return {
            sub: instrs
            for sub, instrs in self._subscriptions_fn().items()
            if DataType.from_str(sub)[0] in _WATCHDOG_DATA_TYPES and instrs
        }

    def _tick_exchange(self, exchange: str, universe: dict[str, set[Instrument]]) -> None:
        status = self._health_monitor.get_exchange_data_status(exchange, universe)
        classification = self.classify(status)

        if classification is ExchangeClassification.DARK:
            self._dark_ticks[exchange] += 1
            if self._dark_ticks[exchange] >= _DARK_TICKS_BEFORE_MAINTENANCE:
                self._hold_maintenance(exchange, status)
            return

        was_dark = self._dark_ticks[exchange] > 0
        self._dark_ticks[exchange] = 0
        self._clear_maintenance(exchange)
        if was_dark:
            # - transport is back: re-assert the whole universe once rather than
            #   repairing instrument by instrument
            self._reconcile_fn()
            return

        if classification is ExchangeClassification.OK:
            self._forget_repairs(exchange)
            return

        self._repair_stale(exchange, universe)

    # ----- repair -----

    def _repair_stale(self, exchange: str, universe: dict[str, set[Instrument]]) -> None:
        now = self._health_monitor.time_provider.time()
        due: set[Instrument] = set()

        for sub, instruments in universe.items():
            base_type = str(DataType.from_str(sub)[0])
            for instrument in instruments:
                if instrument.exchange != exchange:
                    continue
                key = (instrument, base_type)
                if not self._health_monitor.is_stale(instrument, base_type):
                    self._repair_state.pop(key, None)
                    continue
                record = self._repair_state.get(key)
                if record is None:
                    due.add(instrument)
                    continue
                last_event = self._health_monitor.get_last_event_time(instrument, base_type)
                if last_event is not None and last_event > record.repaired_at:
                    # - verified by advancement: a message arrived after the repair, so
                    #   the subscription is live whatever the staleness threshold says
                    self._repair_state.pop(key, None)
                    continue
                record.ticks_until_retry -= 1
                if record.ticks_until_retry <= 0:
                    due.add(instrument)

        if not due:
            return

        self._reconcile_fn(refresh=due)

        for sub, instruments in universe.items():
            base_type = str(DataType.from_str(sub)[0])
            cap = self._backoff_cap_ticks(base_type)
            for instrument in due & instruments:
                key = (instrument, base_type)
                record = self._repair_state.get(key)
                if record is None:
                    self._repair_state[key] = _RepairRecord(now, 1)
                    logger.info(f"[{exchange}] :: repairing {sub} for {instrument.symbol}")
                else:
                    record.repaired_at = now
                    record.interval_ticks = min(record.interval_ticks * 2, cap)
                    record.ticks_until_retry = record.interval_ticks
                    level = logger.error if record.interval_ticks >= cap else logger.warning
                    level(
                        f"[{exchange}] :: {sub} for {instrument.symbol} still not delivering "
                        f"after repair; retrying every {record.interval_ticks} ticks"
                    )

    def _backoff_cap_ticks(self, base_type: str) -> int:
        from qubx.health.base import STALE_THRESHOLDS

        threshold = STALE_THRESHOLDS.get(base_type)
        if threshold is None:
            return 1
        cap_seconds = convert_tf_str_td64(threshold).astype("timedelta64[s]").astype(int)
        cap_seconds = cap_seconds / _BACKOFF_CAP_DIVISOR
        return max(1, int(cap_seconds // self._interval_seconds))

    def _forget_repairs(self, exchange: str) -> None:
        for key in [k for k in self._repair_state if k[0].exchange == exchange]:
            self._repair_state.pop(key, None)

    # ----- status -----

    def _hold_maintenance(self, exchange: str, status: ExchangeDataStatus) -> None:
        if exchange in self._maintenance_held:
            return
        self._maintenance_held.add(exchange)
        reason = "connection down" if status.connected is False else "no instrument delivering"
        logger.error(f"[{exchange}] :: exchange is dark ({reason}) - holding EXCHANGE_MAINTENANCE")
        self._status.add(
            DegradeReason.EXCHANGE_MAINTENANCE,
            self._health_monitor.time_provider.time(),
            scope=exchange,
            message=reason,
        )

    def _clear_maintenance(self, exchange: str) -> None:
        if exchange not in self._maintenance_held:
            return
        self._maintenance_held.discard(exchange)
        logger.info(f"[{exchange}] :: data resumed - clearing EXCHANGE_MAINTENANCE")
        self._status.clear(DegradeReason.EXCHANGE_MAINTENANCE, scope=exchange)
```

If `IHealthMonitor` does not expose `time_provider`, add it to the protocol in `interfaces.py` — `BaseHealthMonitor` already has the attribute.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/test_subscription_watchdog.py -v`
Expected: 16 passed. If `test_unverified_repair_backs_off_and_never_stops` fails at the upper bound, the backoff is not doubling; if it fails at the lower bound, repairs stopped — both are real defects, not test noise.

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/subscription_watchdog.py tests/qubx/core/test_subscription_watchdog.py
git commit -m "feat(core): SubscriptionWatchdog with verification by advancement

Classifies each exchange OK/PARTIAL/DARK over instruments past their grace
window. PARTIAL repairs through reconcile and never degrades - instruments
still delivering prove the venue is up. DARK issues no repair at all and
publishes EXCHANGE_MAINTENANCE after two consecutive ticks.

Repairs are verified by last_event_time advancing past the repair, never
by is_stale, which reads True immediately after a successful repair and
would loop-repair a sparse feed forever."
```

---

### Task 6: Wire the watchdog into `SubscriptionManager`, delete the old loop

**Files:**
- Modify: `src/qubx/core/mixins/subscription.py` — `_init_subscription_monitoring` (~444), delete `_monitor_loop` (~447) and `_monitor_subscription_status` (~456-495), delete `_WATCHDOG_DATA_TYPES` (line 27, now owned by the watchdog)
- Test: `tests/qubx/core/test_subscription_intent.py` (extend)

**Interfaces:**
- Consumes: `SubscriptionWatchdog` (Task 5), `reconcile` (Task 3), `_desired` (Task 2).
- Produces: `SubscriptionManager._watchdog: SubscriptionWatchdog | None` — `None` in simulation.

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/test_subscription_intent.py`:

```python
def test_watchdog_is_started_for_live_and_absent_in_simulation():
    from qubx.core.subscription_watchdog import SubscriptionWatchdog

    live = Mock()
    live.is_simulation = False
    live.exchange.return_value = EXCHANGE
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    manager = SubscriptionManager(
        time_provider, [live], CtrlChannel("test"), DummyHealthMonitor(), StrategyState()
    )
    assert isinstance(manager._watchdog, SubscriptionWatchdog)

    sim = Mock()
    sim.is_simulation = True
    sim.exchange.return_value = EXCHANGE
    sim_manager = SubscriptionManager(
        time_provider, [sim], CtrlChannel("test"), DummyHealthMonitor(), StrategyState()
    )
    assert sim_manager._watchdog is None


def test_watchdog_sees_the_desired_universe():
    live = Mock()
    live.is_simulation = False
    live.exchange.return_value = EXCHANGE
    live.get_subscribed_instruments.return_value = []
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    manager = SubscriptionManager(
        time_provider, [live], CtrlChannel("test"), DummyHealthMonitor(), StrategyState()
    )
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()

    assert manager._watchdog._subscriptions_fn() == {DataType.ORDERBOOK: {btc}}


def test_old_monitor_entry_points_are_gone():
    assert not hasattr(SubscriptionManager, "_monitor_subscription_status")
    assert not hasattr(SubscriptionManager, "_monitor_loop")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/test_subscription_intent.py -v -k "watchdog or old_monitor"`
Expected: `test_watchdog_is_started_for_live_and_absent_in_simulation` FAILS with `AttributeError: '_watchdog'`; `test_old_monitor_entry_points_are_gone` FAILS because both methods still exist.

- [ ] **Step 3: Write minimal implementation**

Replace `_init_subscription_monitoring` and delete both old methods entirely:

```python
def _init_subscription_monitoring(self) -> None:
    self._watchdog: SubscriptionWatchdog | None = None
    if self._is_simulation:
        return
    self._watchdog = SubscriptionWatchdog(
        data_providers=self._data_providers,
        health_monitor=self._health_monitor,
        status=self._status,
        reconcile_fn=self.reconcile,
        subscriptions_fn=lambda: dict(self._desired),
        strategy_state=self._strategy_state,
        interval_seconds=self._monitor_interval_seconds,
    )
    self._watchdog.start()
```

Add the import at the top of `subscription.py`:

```python
from qubx.core.subscription_watchdog import SubscriptionWatchdog
```

Remove `_WATCHDOG_DATA_TYPES` from `subscription.py` (line 27) and remove the now-unused `pprint` import if nothing else uses it.

`SubscriptionManager` needs the `ContextStatus`. It is owned by `StrategyContext` (`context.py:233`). Add a `status: ContextStatus` parameter to `SubscriptionManager.__init__` with default `None`, store it as `self._status = status if status is not None else ContextStatus()`, and pass `self._status` from `context.py` where the manager is constructed. A standalone `ContextStatus` is the correct default for the existing tests that build the manager directly — they have no context to share one with.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/test_subscription_intent.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full core and health suites**

Run: `uv run pytest tests/qubx/core tests/qubx/health -q`
Expected: all pass. Any test referencing `_monitor_subscription_status` must be deleted — its behaviour is now covered by `test_subscription_watchdog.py`.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/mixins/subscription.py src/qubx/core/context.py tests/qubx/core/test_subscription_intent.py
git commit -m "refactor(core): replace the inline stale monitor with SubscriptionWatchdog

The old _monitor_subscription_status unsubscribed stale instruments, slept,
then resubscribed with no rollback and no retry. When the resubscribe failed
during a venue outage it emptied the provider's registry, which was also the
only record of the universe, and left itself nothing to police."
```

---

### Task 7: `QubxDegradedState` must not count as a strategy failure

Without this, publishing `EXCHANGE_MAINTENANCE` turns a 14-minute venue blip into a stopped run for any strategy with `deny_trading_when_degraded=True` (frab defaults it on). D7.

**Files:**
- Modify: `src/qubx/core/mixins/processing.py:658-668`
- Test: `tests/qubx/core/test_degraded_not_a_failure.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: no new API. `ProcessingManager._fails_counter` is no longer incremented by `QubxDegradedState`.

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/core/test_degraded_not_a_failure.py`:

```python
import pytest

from qubx.core.exceptions import QubxDegradedState, StrategyExceededMaxNumberOfRuntimeFailuresError
from qubx.core.mixins.processing import ProcessingManager


class _Counter:
    """Minimal stand-in exercising only the failure-counting branch."""

    MAX_NUMBER_OF_STRATEGY_FAILURES = ProcessingManager.MAX_NUMBER_OF_STRATEGY_FAILURES

    def __init__(self) -> None:
        self._fails_counter = 0

    def run(self, exc: Exception) -> None:
        try:
            raise exc
        except QubxDegradedState as degraded:
            # expected refusal, not a strategy bug
            _ = degraded
        except Exception:
            self._fails_counter += 1
            if self._fails_counter >= self.MAX_NUMBER_OF_STRATEGY_FAILURES:
                raise StrategyExceededMaxNumberOfRuntimeFailuresError()


def test_degraded_state_does_not_stop_the_run():
    c = _Counter()
    for _ in range(20):
        c.run(QubxDegradedState("venue in maintenance", ()))
    assert c._fails_counter == 0


def test_other_exceptions_still_stop_the_run():
    c = _Counter()
    with pytest.raises(StrategyExceededMaxNumberOfRuntimeFailuresError):
        for _ in range(ProcessingManager.MAX_NUMBER_OF_STRATEGY_FAILURES):
            c.run(ValueError("real bug"))
```

Replace `_Counter` with a direct exercise of `ProcessingManager` if the real class can be constructed cheaply in this codebase; check `tests/qubx/core/` for an existing `ProcessingManager` fixture first and prefer it. The assertions above are the contract either way.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/test_degraded_not_a_failure.py -v`
Expected: PASS against the stand-in (it encodes the target behaviour). The real failure is in `processing.py` — confirm it by reading `processing.py:658-668` and verifying `QubxDegradedState` is not caught separately. If you replaced the stand-in with the real `ProcessingManager`, `test_degraded_state_does_not_stop_the_run` FAILS with `StrategyExceededMaxNumberOfRuntimeFailuresError`.

- [ ] **Step 3: Write minimal implementation**

In `processing.py`, add before the existing `except Exception as strat_error:` block:

```python
except QubxDegradedState as degraded:
    # - a refusal the framework generated for an expected condition (venue in
    #   maintenance, stale local view), not a strategy bug. Counting it would let a
    #   venue outage stop the run after 10 events.
    logger.warning(f"Strategy {self._strategy_name} order refused: {degraded}")
```

Add `QubxDegradedState` to the existing `from qubx.core.exceptions import (...)` block in `processing.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/test_degraded_not_a_failure.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/mixins/processing.py tests/qubx/core/test_degraded_not_a_failure.py
git commit -m "fix(core): QubxDegradedState is not a strategy failure

A refused order raised inside on_event counted toward
MAX_NUMBER_OF_STRATEGY_FAILURES, so ten consecutive refusals stopped the
run. With EXCHANGE_MAINTENANCE now having a writer, a short venue outage
would kill any strategy running deny_trading_when_degraded."
```

---

### Task 8: Full verification and docs

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-subscription-watchdog-design.md` (status line only)
- Test: whole suite

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest tests/qubx -q -m "not integration and not e2e"`
Expected: no new failures against `main`. Record the baseline first with `git stash && uv run pytest tests/qubx -q -m "not integration and not e2e"; git stash pop` if anything looks pre-existing.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/qubx tests/qubx && uv run ruff format --check src/qubx tests/qubx`
Expected: clean. Run `uv run ruff format src/qubx tests/qubx` if formatting fails.

- [ ] **Step 3: Verify the connector plugins still build (D1)**

Run:
```bash
cd ~/devs/exchanges && uv run python -c "import qubx_lighter.data, hyperliquid.data; print('ok')"
```
Expected: `ok`. If this fails with an `IDataProvider` signature mismatch, D1 has been violated — stop and report rather than changing the plugins.

- [ ] **Step 4: Update the spec status**

Change the spec header line from:

```markdown
**Status:** approved design, pre-implementation
```

to:

```markdown
**Status:** implemented on `fix/subscription-watchdog`
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-09-14-subscription-watchdog-design.md
git commit -m "docs: mark the subscription watchdog design implemented"
```

---

## Self-Review Notes

Spec coverage: D1 Task 8 step 3 · D2 Task 2 · D3 Tasks 5-6 · D4 Task 3 · D5 Task 5 · D5a Task 5 · D5b Tasks 4-5 · D6 Task 5 · D7 Task 7 · D8 Task 1. Spec §1 Task 1, §2 Task 2, §3 Task 3, §4 Tasks 5-6, §5 Task 4, §6 Task 5, §7 Task 7.

Deliberately **not** in this plan, per the spec's Follow-ups: `IDataProvider.reconnect()` and `set_subscriptions()`, the Lighter silent-drop fix, `_resubscribe_all`'s `contextlib.suppress`, folding in ccxt's `ExchangeManager._stale_monitor_loop`, the platform Prometheus rule, and instrument-level trading protection.

`ExchangeDataStatus` is imported from `qubx.health.status` in Tasks 4 and 5 and constructed with the same six fields in both. `reconcile` is defined in Task 3 with `refresh` keyword and called that way in Tasks 5 and 6. `classify` is a `@staticmethod` in Task 5 and called unbound in its own tests.
