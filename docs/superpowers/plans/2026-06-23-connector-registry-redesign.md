# Connector Registry Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three import-side-effect connector decorators + untyped `**kwargs` factory with a typed two-part build context, a cohesive `ExchangePlugin` ABC, and entry-point discovery — and retire `AsyncThreadLoop` for a proper `BackgroundEventLoop` + `run_sync`.

**Architecture:** A venue is one `ExchangePlugin` (ABC, `None`-returning defaults) that builds its data provider and connector from typed contexts (`BuildContext` → `ConnectorBuildContext`). Plugins are discovered lazily via `importlib.metadata` entry points (group `qubx.exchange_plugins`); built-in `ccxt`/`tardis` self-declare entry points. `read_only` becomes a single trading-mixin gate; `loop` is required and owned externally.

**Tech Stack:** Python 3.12, `asyncio`, `importlib.metadata`, pydantic configs, pytest (`just test`), ruff.

**Spec:** `docs/superpowers/specs/2026-06-23-connector-registry-redesign-design.md` · **Issue:** [#318](https://github.com/xLydianSoftware/Qubx/issues/318)

## Global Constraints

- Python 3.12+ modern types: `list`, `dict`, `| None`, `tuple`. Never `from __future__ import annotations`. Import `Any` from `typing` only if needed.
- Logging: `from qubx import logger`.
- Tooling: `uv run` for Python; run tests with `just test` (parallel) or `uv run pytest <path> -v` for a single test; `just style-check` before each commit.
- Conventional commits (`feat:`/`fix:`/`refactor:`/`docs:`/`test:`); **no** co-authored-by lines; concise messages.
- ruff line length 120.
- Two PRs: **Phase 1** (Tasks 1–4) is behavior-preserving and merges green on its own; **Phase 2** (Tasks 5–13) is the atomic migration. Do not start Phase 2 until Phase 1 is green.

---

## File Structure

**Phase 1 (prep):**
- `src/qubx/utils/misc.py` — add `run_sync()` + `BackgroundEventLoop`; **delete** `AsyncThreadLoop`.
- ccxt stack + tardis + ccxt storage — migrate `AsyncThreadLoop` call sites to `run_coroutine_threadsafe`/`run_sync`.
- `src/qubx/core/mixins/trading.py` — `read_only` gate.
- `src/qubx/connectors/ccxt/connector.py` — delete 5 `read_only` guards + ctor param.
- `src/qubx/connectors/ccxt/factory.py` — `loop` required; delete self-spawn branch.

**Phase 2 (migration):**
- `src/qubx/connectors/plugin.py` — **new**: `BuildContext`, `ConnectorBuildContext`, `ExchangePlugin`.
- `src/qubx/plugins/loader.py` — add `PluginLoader` (entry-point discovery).
- `src/qubx/connectors/registry.py` — rewrite to `ExchangePlugin` registry + convenience methods + tombstone.
- `src/qubx/connectors/ccxt/plugin.py` — **new**: `CcxtPlugin`, `PLUGIN`.
- `src/qubx/connectors/tardis/plugin.py` — **new**: `TardisPlugin`, `PLUGIN`.
- `src/qubx/connectors/ccxt/__init__.py`, `tardis/data.py`, `ccxt/data.py`, `ccxt/rate_limits.py` — remove decorators.
- `src/qubx/utils/runner/runner.py` — two-phase build.
- `pyproject.toml` — entry-point declarations.

---

# Phase 1 — Prep (PR 1, behavior-preserving)

## Task 1: `run_sync` + `BackgroundEventLoop` helpers

**Files:**
- Modify: `src/qubx/utils/misc.py` (add near the existing `AsyncThreadLoop` at `:451`)
- Test: `tests/qubx/utils/test_async_loop.py` (create)

**Interfaces:**
- Produces:
  - `run_sync(loop: asyncio.AbstractEventLoop, coro, *, timeout: float | None = None) -> Any`
  - `class BackgroundEventLoop` with `.loop`, `.submit(coro) -> concurrent.futures.Future`, `.run_sync(coro, *, timeout=None)`, `.stop()`

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/utils/test_async_loop.py
import asyncio
import threading

import pytest

from qubx.utils.misc import BackgroundEventLoop, run_sync


def test_background_loop_run_sync_returns_result():
    bel = BackgroundEventLoop(name="test-loop")
    try:
        async def add(a, b):
            await asyncio.sleep(0)
            return a + b
        assert bel.run_sync(add(2, 3)) == 5
    finally:
        bel.stop()


def test_run_sync_propagates_exception():
    bel = BackgroundEventLoop()
    try:
        async def boom():
            raise ValueError("kaboom")
        with pytest.raises(ValueError, match="kaboom"):
            bel.run_sync(boom())
    finally:
        bel.stop()


def test_run_sync_times_out():
    bel = BackgroundEventLoop()
    try:
        async def slow():
            await asyncio.sleep(5)
        with pytest.raises(TimeoutError):
            bel.run_sync(slow(), timeout=0.05)
    finally:
        bel.stop()


def test_run_sync_reentrancy_guard_raises():
    bel = BackgroundEventLoop()
    try:
        async def reenter():
            # called ON the loop thread → must raise, not deadlock
            return run_sync(bel.loop, asyncio.sleep(0))
        with pytest.raises(RuntimeError, match="own thread"):
            bel.run_sync(reenter())
    finally:
        bel.stop()


def test_stop_joins_thread():
    bel = BackgroundEventLoop(name="join-me")
    bel.stop()
    assert not bel._thread.is_alive()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/utils/test_async_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'BackgroundEventLoop'`.

- [ ] **Step 3: Implement the helpers**

In `src/qubx/utils/misc.py`, ensure imports exist at top (`import asyncio`, `import concurrent.futures`, `from threading import Thread`), then add directly below the existing `AsyncThreadLoop` class (do not remove `AsyncThreadLoop` yet — Task 2 does):

```python
def run_sync(loop: asyncio.AbstractEventLoop, coro, *, timeout: float | None = None):
    """Submit ``coro`` to ``loop`` from another thread, block for the result, propagate its exception.

    Guards the classic deadlock of being called from ``loop``'s own thread.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError("run_sync called from the target loop's own thread — would deadlock")
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


class BackgroundEventLoop:
    """Owns an asyncio loop on a dedicated daemon thread; submit/run coroutines onto it."""

    def __init__(self, name: str = "QubxConnectorLoop"):
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True, name=name)
        self._thread.start()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro, *, timeout: float | None = None):
        return run_sync(self._loop, coro, timeout=timeout)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
```

Note: `concurrent.futures.Future.result(timeout)` raises `concurrent.futures.TimeoutError`, which **is** `TimeoutError` in Python 3.12 — the `test_run_sync_times_out` assertion holds.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/utils/test_async_loop.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Style-check and commit**

```bash
just style-check
git add src/qubx/utils/misc.py tests/qubx/utils/test_async_loop.py
git commit -m "feat(utils): add run_sync + BackgroundEventLoop (owns loop+thread, reentrancy-guarded)"
```

---

## Task 2: Retire `AsyncThreadLoop` (migrate in-tree call sites)

**Files:**
- Modify (call sites): `src/qubx/connectors/ccxt/connector.py:203-211`, `ccxt/data.py:103-105`, `ccxt/connection_manager.py:83-85`, `ccxt/subscription_orchestrator.py:49-51`, `ccxt/warmup_service.py:47-49`, `ccxt/tardis/data.py` (`tardis/data.py:71,76`), `data/storages/ccxt.py:454,503`
- Modify (delete class): `src/qubx/utils/misc.py` (`AsyncThreadLoop` at `:451`)
- Test: existing ccxt/tardis/storage suites must stay green (no new test file).

**Interfaces:**
- Consumes: `run_sync`, `BackgroundEventLoop` (Task 1), stdlib `asyncio.run_coroutine_threadsafe`.

The transform per call site: a `_loop` property returning `AsyncThreadLoop(loop)` and callers doing `self._loop.submit(coro)` become a `_loop` property returning the raw loop and callers doing `asyncio.run_coroutine_threadsafe(coro, self._loop)`; any blocking `.submit(coro).result()` becomes `run_sync(self._loop, coro, timeout=...)`.

- [ ] **Step 1: Read each call site to capture exact usage**

Run: `grep -rn "AsyncThreadLoop\|\._loop\b\|\.submit(" src/qubx/connectors/ccxt/ src/qubx/connectors/tardis/ src/qubx/data/storages/ccxt.py`
Record, per file: whether `.submit()` results are awaited/`.result()`-ed (→ `run_sync`) or fire-and-forget (→ `run_coroutine_threadsafe`).

- [ ] **Step 2: Migrate one file (representative: `ccxt/connector.py`)**

Replace the `_loop` property (`connector.py:203-211`):

```python
# BEFORE
from qubx.utils.misc import AsyncThreadLoop
@property
def _loop(self) -> AsyncThreadLoop:
    loop = self._exchange_manager.exchange.asyncio_loop
    return AsyncThreadLoop(loop)

# AFTER
import asyncio
from qubx.utils.misc import run_sync  # if any blocking submit exists in this file
@property
def _loop(self) -> asyncio.AbstractEventLoop:
    return self._exchange_manager.exchange.asyncio_loop
```

Then update this file's callers: `self._loop.submit(coro)` → `asyncio.run_coroutine_threadsafe(coro, self._loop)`; `self._loop.submit(coro).result(t)` → `run_sync(self._loop, coro, timeout=t)`.

- [ ] **Step 3: Run that file's tests**

Run: `uv run pytest tests/qubx/connectors/ccxt/ -k connector -v`
Expected: PASS (unchanged behavior).

- [ ] **Step 4: Commit the file, then repeat Steps 2–3 for each remaining call site**

Repeat for `ccxt/data.py`, `ccxt/connection_manager.py`, `ccxt/subscription_orchestrator.py`, `ccxt/warmup_service.py`, `tardis/data.py`, `data/storages/ccxt.py`. Also delete the now-stale comment at `ccxt/data.py:275` ("AsyncThreadLoop stop is handled by its own lifecycle"). Commit per file:

```bash
git add src/qubx/connectors/ccxt/connector.py
git commit -m "refactor(ccxt): drop AsyncThreadLoop in connector for run_coroutine_threadsafe/run_sync"
```

- [ ] **Step 5: Delete `AsyncThreadLoop` and verify nothing references it**

Run: `grep -rn "AsyncThreadLoop" src/` → expect **0** matches. Remove the `class AsyncThreadLoop` block from `src/qubx/utils/misc.py`.

- [ ] **Step 6: Full suite + commit**

Run: `just test`
Expected: PASS.

```bash
just style-check
git add src/qubx/utils/misc.py
git commit -m "refactor(utils): remove AsyncThreadLoop (superseded by run_sync/BackgroundEventLoop)"
```

---

## Task 3: Move `read_only` to a single trading-mixin gate

**Files:**
- Read first: `src/qubx/core/mixins/trading.py:100-130,320-340,380-395,470-480` (submit/cancel/update dispatch + `_get_connector`)
- Modify: `src/qubx/core/mixins/trading.py` (add `self._read_only` + 3 guards), `src/qubx/core/context.py` (thread `config.live.read_only` into the trading mixin at construction)
- Modify: `src/qubx/connectors/ccxt/connector.py` (delete guards at `:302,404,572,701,713` + the `read_only` ctor param at `:161,174`), `src/qubx/connectors/ccxt/factory.py` (drop `read_only=` passthrough at `:211`)
- Test: `tests/qubx/core/mixins/trading_test.py` (add gate tests); delete `tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py` read_only tests (`:581-617`)

**Interfaces:**
- Produces: trading mixin raises `qubx.core.exceptions.ReadOnlyConnector` from `trade`/cancel/update when `self._read_only` is set.

- [ ] **Step 1: Read the dispatch points and the connector constructor**

Run: `sed -n '100,130p;320,340p;380,395p;470,480p' src/qubx/core/mixins/trading.py` and `sed -n '153,180p;295,310p' src/qubx/connectors/ccxt/connector.py`. Note the exact method names that call `self._get_connector(...).submit_order/cancel_order/update_order` and how the mixin is constructed.

- [ ] **Step 2: Write the failing gate test**

```python
# tests/qubx/core/mixins/trading_test.py  (add)
import pytest
from qubx.core.exceptions import ReadOnlyConnector

def test_read_only_blocks_trade(read_only_trading_ctx):  # fixture: context built with read_only=True
    with pytest.raises(ReadOnlyConnector):
        read_only_trading_ctx.trade(read_only_trading_ctx.instruments[0], 1.0)
```

Add a `read_only_trading_ctx` fixture mirroring the existing trading-test context fixture but constructing the context with `read_only=True`. (Use the existing trading_test context builder; pass the flag through.)

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py::test_read_only_blocks_trade -v`
Expected: FAIL (no gate yet → order attempts to dispatch).

- [ ] **Step 4: Implement the gate**

In `trading.py`, set `self._read_only` at construction (sourced from `config.live.read_only`, threaded via `context.py`), and at the **top** of each of the three public entry points (`trade`/cancel/update — exact names from Step 1) add:

```python
if self._read_only:
    raise ReadOnlyConnector(f"{instrument.exchange} is read-only — order rejected")
```

In `context.py`, where the trading manager/mixin is constructed, pass `read_only=config.live.read_only` and store it as `self._read_only` (default `False`).

- [ ] **Step 5: Delete the connector-level guards + ctor param**

In `ccxt/connector.py` remove the five `if self._read_only: raise ReadOnlyConnector(...)` blocks (`:302,404,572,701,713`), the `self._read_only = read_only` assignment (`:174`), and the `read_only: bool = False` ctor param (`:161`). In `ccxt/factory.py` drop `read_only=read_only` from `get_ccxt_connector(...)` (`:211`) and the `read_only` factory param if now unused. Delete the read_only tests in `test_ccxt_connector_writes.py:581-617`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py tests/qubx/connectors/ccxt/test_ccxt_connector_writes.py -v`
Expected: PASS (new gate test passes; connector write tests pass without read_only cases).

- [ ] **Step 7: Style-check + commit**

```bash
just style-check
git add src/qubx/core/mixins/trading.py src/qubx/core/context.py src/qubx/connectors/ccxt/connector.py src/qubx/connectors/ccxt/factory.py tests/
git commit -m "refactor(core): enforce read_only once at the trading-mixin write boundary"
```

---

## Task 4: Make `loop` required in the ccxt factory

**Files:**
- Modify: `src/qubx/connectors/ccxt/factory.py:42-62` (`get_ccxt_exchange`), `:100-110` (`get_ccxt_exchange_manager`)
- Test: `tests/qubx/connectors/ccxt/` factory/connector construction tests (ensure they pass a loop)

**Interfaces:**
- Consumes: callers already pass `loop` (runner). Test helpers that constructed connectors with `loop=None` must now pass `BackgroundEventLoop().loop` (Task 1).

- [ ] **Step 1: Make the parameter required and delete the self-spawn branch**

In `get_ccxt_exchange` (`factory.py:42`), change `loop: asyncio.AbstractEventLoop | None = None` → `loop: asyncio.AbstractEventLoop` and replace `:55-62`:

```python
# BEFORE
if loop is not None:
    options["asyncio_loop"] = loop
else:
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    options["thread_asyncio_loop"] = thread
    options["asyncio_loop"] = loop

# AFTER
options["asyncio_loop"] = loop
```

Make `loop` required in `get_ccxt_exchange_manager` (`:106`) too.

- [ ] **Step 2: Update any test helper passing `loop=None`**

Run: `grep -rn "get_ccxt_exchange\|get_ccxt_exchange_manager\|_make_connector" tests/qubx/connectors/ccxt/`. For each that omits `loop`, construct one via `BackgroundEventLoop` in a fixture and pass `.loop` (stop it in teardown).

- [ ] **Step 3: Run + commit**

Run: `uv run pytest tests/qubx/connectors/ccxt/ -v` → PASS.

```bash
just style-check
git add src/qubx/connectors/ccxt/factory.py tests/qubx/connectors/ccxt/
git commit -m "refactor(ccxt): require an externally-owned loop in the factory (drop self-spawn)"
```

**Phase 1 gate:** `just test` green → open PR 1.

---

# Phase 2 — Migration (PR 2, atomic)

## Task 5: Core types — `connectors/plugin.py`

**Files:**
- Create: `src/qubx/connectors/plugin.py`
- Test: `tests/qubx/connectors/test_plugin_types.py`

**Interfaces:**
- Produces:
  - `BuildContext(exchange_name, time_provider, channel, credentials, health_monitor, loop, rate_limiter=None)`
  - `ConnectorBuildContext(BuildContext, data_provider)`
  - `ExchangePlugin` ABC: `name: str`; `create_data_provider(ctx) -> IDataProvider | None`, `create_connector(ctx) -> IConnector | None`, `rate_limits(exchange_name) -> ExchangeRateLimitConfig | None` — all default `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/qubx/connectors/test_plugin_types.py
from qubx.connectors.plugin import BuildContext, ConnectorBuildContext, ExchangePlugin


def test_connector_ctx_is_a_build_ctx():
    assert issubclass(ConnectorBuildContext, BuildContext)


def test_plugin_defaults_return_none():
    class P(ExchangePlugin):
        name = "p"
    p = P()
    assert p.create_connector(object()) is None
    assert p.create_data_provider(object()) is None
    assert p.rate_limits("X") is None


def test_partial_plugin_overrides_only_data_provider():
    sentinel = object()
    class DataOnly(ExchangePlugin):
        name = "dataonly"
        def create_data_provider(self, ctx):
            return sentinel
    p = DataOnly()
    assert p.create_data_provider(object()) is sentinel
    assert p.create_connector(object()) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/qubx/connectors/test_plugin_types.py -v`
Expected: FAIL (`ModuleNotFoundError: qubx.connectors.plugin`).

- [ ] **Step 3: Implement**

```python
# src/qubx/connectors/plugin.py
import asyncio
from abc import ABC
from dataclasses import dataclass

from qubx.core.basics import CtrlChannel
from qubx.core.interfaces import IConnector, IDataProvider, ITimeProvider
from qubx.connectors.registry import CredentialsProvider
from qubx.health import IHealthMonitor  # adjust import to the real IHealthMonitor location (see Step 3a)
from qubx.rate_limiting import ExchangeRateLimitConfig, ExchangeRateLimiter  # adjust to real path


@dataclass(frozen=True, kw_only=True)
class BuildContext:
    exchange_name: str
    time_provider: ITimeProvider
    channel: CtrlChannel
    credentials: CredentialsProvider
    health_monitor: IHealthMonitor
    loop: asyncio.AbstractEventLoop
    rate_limiter: ExchangeRateLimiter | None = None


@dataclass(frozen=True, kw_only=True)
class ConnectorBuildContext(BuildContext):
    data_provider: IDataProvider


class ExchangePlugin(ABC):
    """One venue: connector + data provider + rate-limit declaration. Override only what you provide."""

    name: str

    def create_data_provider(self, ctx: BuildContext) -> IDataProvider | None:
        return None

    def create_connector(self, ctx: ConnectorBuildContext) -> IConnector | None:
        return None

    def rate_limits(self, exchange_name: str) -> "ExchangeRateLimitConfig | None":
        return None
```

- [ ] **Step 3a: Fix imports to real module paths**

Run: `grep -rn "class IHealthMonitor\|class ExchangeRateLimiter\|class ExchangeRateLimitConfig\|class ITimeProvider\|class CredentialsProvider" src/qubx/` and correct each import above to the actual module. (Contexts are **not** `slots=True`, so the runner can use `ConnectorBuildContext(**vars(base), data_provider=dp)`.)

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/qubx/connectors/test_plugin_types.py -v` → PASS.

```bash
just style-check
git add src/qubx/connectors/plugin.py tests/qubx/connectors/test_plugin_types.py
git commit -m "feat(connectors): add ExchangePlugin ABC + BuildContext/ConnectorBuildContext"
```

---

## Task 6: `PluginLoader` — entry-point discovery

**Files:**
- Modify: `src/qubx/plugins/loader.py` (add `PluginLoader`; leave `load_plugins` untouched)
- Test: `tests/qubx/plugins/test_plugin_loader.py`

**Interfaces:**
- Consumes: `ExchangePlugin` (Task 5).
- Produces: `PluginLoader.available() -> set[str]`; `PluginLoader.load(name: str) -> ExchangePlugin | None`.

- [ ] **Step 1: Write failing tests (monkeypatch entry points)**

```python
# tests/qubx/plugins/test_plugin_loader.py
import pytest
from qubx.connectors.plugin import ExchangePlugin
from qubx.plugins.loader import PluginLoader


class _FakePlugin(ExchangePlugin):
    name = "fake"


class _FakeEP:
    name = "fake"
    def load(self):
        return _FakePlugin()


def test_available_lists_names_without_loading(monkeypatch):
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_FakeEP()])
    assert "fake" in PluginLoader.available()


def test_load_returns_plugin(monkeypatch):
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_FakeEP()])
    assert isinstance(PluginLoader.load("fake"), _FakePlugin)


def test_load_unknown_returns_none(monkeypatch):
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_FakeEP()])
    assert PluginLoader.load("nope") is None


def test_load_rejects_non_plugin(monkeypatch):
    class _BadEP:
        name = "bad"
        def load(self): return object()
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_BadEP()])
    with pytest.raises(TypeError):
        PluginLoader.load("bad")


def test_load_asserts_name_matches(monkeypatch):
    class _MismatchPlugin(ExchangePlugin):
        name = "other"
    class _EP:
        name = "fake"
        def load(self): return _MismatchPlugin()
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_EP()])
    with pytest.raises(AssertionError):
        PluginLoader.load("fake")
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/qubx/plugins/test_plugin_loader.py -v`
Expected: FAIL (`ImportError: cannot import name 'PluginLoader'`).

- [ ] **Step 3: Implement**

Add to `src/qubx/plugins/loader.py`:

```python
from importlib.metadata import entry_points

from qubx.connectors.plugin import ExchangePlugin

_GROUP = "qubx.exchange_plugins"


class PluginLoader:
    @staticmethod
    def available() -> set[str]:
        return {ep.name for ep in entry_points(group=_GROUP)}

    @staticmethod
    def load(name: str) -> ExchangePlugin | None:
        for ep in entry_points(group=_GROUP):
            if ep.name != name:
                continue
            try:
                obj = ep.load()
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"connector plugin '{name}' is installed but its module is unavailable "
                    f"(missing optional dependency? for built-ins try `qubx[connectors]`): {e}"
                ) from e
            if not isinstance(obj, ExchangePlugin):
                raise TypeError(f"entry point '{name}' did not resolve to an ExchangePlugin (got {type(obj)!r})")
            assert obj.name == name, f"plugin.name {obj.name!r} != entry-point name {name!r}"
            return obj
        return None
```

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/qubx/plugins/test_plugin_loader.py -v` → PASS.

```bash
just style-check
git add src/qubx/plugins/loader.py tests/qubx/plugins/test_plugin_loader.py
git commit -m "feat(plugins): add entry-point PluginLoader for qubx.exchange_plugins"
```

---

## Task 7: Rewrite `ConnectorRegistry` + tombstone

**Files:**
- Modify: `src/qubx/connectors/registry.py` (replace the three-decorator API with the plugin registry; keep `CredentialsProvider`; extend the tombstone)
- Test: `tests/qubx/connectors/test_registry.py` (rewrite from the old decorator tests)

**Interfaces:**
- Consumes: `ExchangePlugin` (Task 5), `PluginLoader` (Task 6), `BuildContext`/`ConnectorBuildContext` (Task 5).
- Produces: `ConnectorRegistry.register(plugin)`, `.get_plugin(name)`, `.get_data_provider(name, ctx)`, `.get_connector(name, ctx)`, `.get_rate_limit_config(name, exchange_name)`. Module-level tombstone `__getattr__` raises on `connector`/`data_provider`/`rate_limit_config`/`register_connector`/`register_data_provider`/`register_rate_limit_config`/`broker`/`account_processor`.

- [ ] **Step 1: Write failing tests**

```python
# tests/qubx/connectors/test_registry.py
import pytest
from qubx.connectors.plugin import ExchangePlugin
from qubx.connectors.registry import ConnectorRegistry


class _DataOnly(ExchangePlugin):
    name = "dataonly"
    def create_data_provider(self, ctx): return "DP"


def test_register_and_get_plugin():
    ConnectorRegistry._plugins.clear()
    p = _DataOnly()
    ConnectorRegistry.register(p)
    assert ConnectorRegistry.get_plugin("dataonly") is p


def test_get_connector_missing_capability_raises():
    ConnectorRegistry._plugins.clear()
    ConnectorRegistry.register(_DataOnly())
    with pytest.raises(ValueError, match="no execution connector"):
        ConnectorRegistry.get_connector("dataonly", object())


def test_get_plugin_unknown_raises():
    ConnectorRegistry._plugins.clear()
    with pytest.raises(ValueError, match="No connector plugin"):
        ConnectorRegistry.get_plugin("ghost")


def test_old_decorator_imports_tombstoned():
    import qubx.connectors.registry as reg
    for name in ("connector", "data_provider", "rate_limit_config"):
        with pytest.raises(ImportError):
            getattr(reg, name)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/qubx/connectors/test_registry.py -v`
Expected: FAIL (old API still present / new API absent).

- [ ] **Step 3: Implement the new registry**

Replace the body of `ConnectorRegistry` (keep `CredentialsProvider` Protocol). New methods:

```python
class ConnectorRegistry:
    _plugins: dict[str, "ExchangePlugin"] = {}

    @classmethod
    def register(cls, plugin: "ExchangePlugin") -> None:
        cls._plugins[plugin.name.lower()] = plugin

    @classmethod
    def get_plugin(cls, name: str) -> "ExchangePlugin":
        key = name.lower()
        if key not in cls._plugins:
            from qubx.plugins.loader import PluginLoader
            plugin = PluginLoader.load(key)
            if plugin is None:
                raise ValueError(f"No connector plugin '{name}'. Available: {sorted(PluginLoader.available())}")
            cls._plugins[key] = plugin
        return cls._plugins[key]

    @classmethod
    def get_data_provider(cls, name: str, ctx) -> "IDataProvider":
        dp = cls.get_plugin(name).create_data_provider(ctx)
        if dp is None:
            raise ValueError(f"Venue plugin '{name}' provides no data provider")
        return dp

    @classmethod
    def get_connector(cls, name: str, ctx) -> "IConnector":
        conn = cls.get_plugin(name).create_connector(ctx)
        if conn is None:
            raise ValueError(f"Venue plugin '{name}' provides no execution connector")
        return conn

    @classmethod
    def get_rate_limit_config(cls, name: str, exchange_name: str):
        return cls.get_plugin(name).rate_limits(exchange_name)
```

Extend `_REMOVED_NAMES` to include `"connector", "data_provider", "rate_limit_config", "register_connector", "register_data_provider", "register_rate_limit_config"` and update the tombstone `__getattr__` message to point at the `ExchangePlugin` + entry-point model. Delete `register_connector`/`register_data_provider`/`register_rate_limit_config` and the three `connector`/`data_provider`/`rate_limit_config` decorator functions and their `_connectors`/`_data_providers`/`_rate_limit_configs` dicts.

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/qubx/connectors/test_registry.py -v` → PASS.

```bash
just style-check
git add src/qubx/connectors/registry.py tests/qubx/connectors/test_registry.py
git commit -m "refactor(connectors): registry holds ExchangePlugins; tombstone the old decorators"
```

---

## Task 8: `CcxtPlugin`

**Files:**
- Read first: `src/qubx/connectors/ccxt/data.py:24-61` (`CcxtDataProvider.__init__` + rate_limiter), `ccxt/factory.py:171-213` (`create_ccxt_connector`), `ccxt/rate_limits.py:20-21`
- Create: `src/qubx/connectors/ccxt/plugin.py`
- Modify: `ccxt/data.py` (drop `@data_provider`), `ccxt/rate_limits.py` (drop `@rate_limit_config`), `ccxt/factory.py` (drop `@connector`), `src/qubx/connectors/__init__.py` (drop the import-for-side-effect at `:12-20`)
- Test: `tests/qubx/connectors/ccxt/test_ccxt_plugin.py`

**Interfaces:**
- Consumes: `ExchangePlugin`, `BuildContext`, `ConnectorBuildContext`.
- Produces: `CcxtPlugin` (name `"ccxt"`) + module-level `PLUGIN = CcxtPlugin()`. `create_data_provider`/`create_connector` attach `ctx.rate_limiter` to **both** exchange managers.

- [ ] **Step 1: Read the current constructors to match signatures**

Run: `sed -n '24,61p' src/qubx/connectors/ccxt/data.py; sed -n '171,213p' src/qubx/connectors/ccxt/factory.py`. Note how `CcxtDataProvider(...)` and `create_ccxt_connector(...)` build their objects so the plugin can call the same builders.

- [ ] **Step 2: Write the failing test (shared-limiter identity)**

```python
# tests/qubx/connectors/ccxt/test_ccxt_plugin.py
from unittest.mock import MagicMock
from qubx.connectors.ccxt.plugin import PLUGIN, CcxtPlugin


def test_plugin_name():
    assert PLUGIN.name == "ccxt"
    assert isinstance(PLUGIN, CcxtPlugin)


def test_rate_limits_delegates(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("qubx.connectors.ccxt.plugin.create_ccxt_rate_limit_config", lambda e: sentinel)
    assert PLUGIN.rate_limits("BINANCE.UM") is sentinel
```

(A full shared-limiter-identity integration assertion lands in Task 11 against the runner. Here we cover name + rate_limits delegation; deeper construction is exercised by the existing ccxt suites after the decorators are removed.)

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ccxt_plugin.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement `CcxtPlugin`**

```python
# src/qubx/connectors/ccxt/plugin.py
from qubx.connectors.plugin import BuildContext, ConnectorBuildContext, ExchangePlugin
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import create_ccxt_connector
from qubx.connectors.ccxt.rate_limits import create_ccxt_rate_limit_config


class CcxtPlugin(ExchangePlugin):
    name = "ccxt"

    def create_data_provider(self, ctx: BuildContext) -> CcxtDataProvider:
        # build exactly as today (see data.py constructor), passing ctx.rate_limiter through
        return CcxtDataProvider(
            exchange_name=ctx.exchange_name,
            time_provider=ctx.time_provider,
            channel=ctx.channel,
            health_monitor=ctx.health_monitor,
            credentials=ctx.credentials,
            loop=ctx.loop,
            rate_limiter=ctx.rate_limiter,
        )

    def create_connector(self, ctx: ConnectorBuildContext):
        return create_ccxt_connector(
            exchange_name=ctx.exchange_name,
            time_provider=ctx.time_provider,
            channel=ctx.channel,
            credentials=ctx.credentials,
            data_provider=ctx.data_provider,
            health_monitor=ctx.health_monitor,
            loop=ctx.loop,
            rate_limiter=ctx.rate_limiter,
        )

    def rate_limits(self, exchange_name: str):
        return create_ccxt_rate_limit_config(exchange_name)


PLUGIN = CcxtPlugin()
```

- [ ] **Step 4a: Wire `rate_limiter` into the connector's exchange manager**

`create_ccxt_connector` (`factory.py:171`) does not take `rate_limiter` today. Add a `rate_limiter: ExchangeRateLimiter | None = None` param and, after building `exchange_manager` (`factory.py:191`), call `exchange_manager.attach_rate_limiter(rate_limiter)` when not None (mirroring `data.py:60-61`). Remove `@connector("ccxt")` from `create_ccxt_connector`, `@data_provider("ccxt")` from `CcxtDataProvider`, `@rate_limit_config("ccxt")` from `create_ccxt_rate_limit_config`, and the registration imports in `connectors/__init__.py:12-20`.

- [ ] **Step 5: Run + commit**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ccxt_plugin.py -v` → PASS.

```bash
just style-check
git add src/qubx/connectors/ccxt/plugin.py src/qubx/connectors/ccxt/{data.py,factory.py,rate_limits.py} src/qubx/connectors/__init__.py tests/qubx/connectors/ccxt/test_ccxt_plugin.py
git commit -m "feat(ccxt): CcxtPlugin (connector+data+rate_limits); attach shared limiter to connector"
```

---

## Task 9: `TardisPlugin`

**Files:**
- Read first: `src/qubx/connectors/tardis/data.py:34` (`@data_provider("tardis")` + the data provider ctor)
- Create: `src/qubx/connectors/tardis/plugin.py`
- Modify: `tardis/data.py` (drop `@data_provider`)
- Test: `tests/qubx/connectors/tardis/test_tardis_plugin.py`

**Interfaces:**
- Produces: `TardisPlugin` (name `"tardis"`, data-only) + `PLUGIN = TardisPlugin()`.

- [ ] **Step 1: Failing test**

```python
# tests/qubx/connectors/tardis/test_tardis_plugin.py
from qubx.connectors.tardis.plugin import PLUGIN

def test_tardis_is_data_only():
    assert PLUGIN.name == "tardis"
    assert PLUGIN.create_connector(object()) is None  # default → no execution connector
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/qubx/connectors/tardis/test_tardis_plugin.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/qubx/connectors/tardis/plugin.py
from qubx.connectors.plugin import BuildContext, ExchangePlugin
from qubx.connectors.tardis.data import TardisDataProvider  # use the real class name from data.py


class TardisPlugin(ExchangePlugin):
    name = "tardis"

    def create_data_provider(self, ctx: BuildContext):
        return TardisDataProvider(  # match the real constructor signature
            exchange_name=ctx.exchange_name,
            time_provider=ctx.time_provider,
            channel=ctx.channel,
            health_monitor=ctx.health_monitor,
            credentials=ctx.credentials,
            loop=ctx.loop,
        )


PLUGIN = TardisPlugin()
```

Remove `@data_provider("tardis")` from `tardis/data.py:34`.

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/qubx/connectors/tardis/test_tardis_plugin.py -v` → PASS.

```bash
just style-check
git add src/qubx/connectors/tardis/plugin.py src/qubx/connectors/tardis/data.py tests/qubx/connectors/tardis/test_tardis_plugin.py
git commit -m "feat(tardis): TardisPlugin (data-only)"
```

---

## Task 10: Declare built-in entry points

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/qubx/connectors/test_builtin_discovery.py`

- [ ] **Step 1: Failing discovery test**

```python
# tests/qubx/connectors/test_builtin_discovery.py
from qubx.plugins.loader import PluginLoader

def test_builtin_plugins_discoverable():
    avail = PluginLoader.available()
    assert {"ccxt", "tardis"} <= avail

def test_ccxt_loads_to_plugin():
    p = PluginLoader.load("ccxt")
    assert p is not None and p.name == "ccxt"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/qubx/connectors/test_builtin_discovery.py -v`
Expected: FAIL (no entry points yet).

- [ ] **Step 3: Add entry points and reinstall**

In `pyproject.toml`:

```toml
[project.entry-points."qubx.exchange_plugins"]
ccxt   = "qubx.connectors.ccxt.plugin:PLUGIN"
tardis = "qubx.connectors.tardis.plugin:PLUGIN"
```

Entry points are read from installed metadata, so re-sync the editable install: `uv sync` (or `uv pip install -e .`).

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/qubx/connectors/test_builtin_discovery.py -v` → PASS.

```bash
just style-check
git add pyproject.toml tests/qubx/connectors/test_builtin_discovery.py
git commit -m "feat(connectors): declare built-in ccxt/tardis entry points"
```

---

## Task 11: Runner two-phase build

**Files:**
- Read first: `src/qubx/utils/runner/runner.py:441-590,755-775` (`create_strategy_context`, the data-provider + connector construction + `_rl_manager` usage)
- Modify: `src/qubx/utils/runner/runner.py`
- Test: `tests/qubx/utils/runner/test_runner.py` (extend) + a shared-limiter assertion

**Interfaces:**
- Consumes: `ConnectorRegistry.get_data_provider/get_connector` (Task 7), `BuildContext`/`ConnectorBuildContext` (Task 5), `_rl_manager.get_or_create` (unchanged).

- [ ] **Step 1: Read the exact construction block**

Run: `sed -n '535,590p;755,775p' src/qubx/utils/runner/runner.py`. Identify where the data provider and connector are built and where `exchange_config.params["rate_limiter"]` (`:543`) and `read_only=` (`:576`) are set.

- [ ] **Step 2: Replace the construction with the two-phase build**

```python
limiter = _rl_manager.get_or_create(venue_name, exchange_config.connector)
base = BuildContext(
    exchange_name=venue_name,
    time_provider=_time,
    channel=_chan,
    credentials=account_manager,
    health_monitor=_health_monitor,
    loop=loop,
    rate_limiter=limiter,
)
_data_provider = ConnectorRegistry.get_data_provider(exchange_config.connector.lower(), base)
cctx = ConnectorBuildContext(**vars(base), data_provider=_data_provider)
_connectors[exchange_name] = ConnectorRegistry.get_connector(exchange_config.connector.lower(), cctx)
```

Delete the old `get_data_provider(name, **kwargs)` / `get_connector(name, **kwargs)` calls, the `exchange_config.params["rate_limiter"] = rate_limiter` injection (`:541-543`), and the `read_only=` arg (`:576`). Keep `load_plugins(stg_config.plugins)` (`:161`) for `@storage`/`@reader`.

- [ ] **Step 3: Write the shared-limiter integration test**

```python
# tests/qubx/utils/runner/test_runner.py  (add)
def test_data_provider_and_connector_share_one_rate_limiter(built_paper_context):
    dp_rl = built_paper_context._data_provider._exchange_manager._rate_limiter  # adjust attr to real
    conn_rl = built_paper_context._connector._exchange_manager._rate_limiter
    assert dp_rl is conn_rl is not None
```

Adjust the attribute paths after Step 1's read; use the existing runner test fixture that builds a paper/live context with rate limiting enabled.

- [ ] **Step 4: Run + commit**

Run: `uv run pytest tests/qubx/utils/runner/test_runner.py -v` → PASS.

```bash
just style-check
git add src/qubx/utils/runner/runner.py tests/qubx/utils/runner/test_runner.py
git commit -m "refactor(runner): two-phase plugin build with typed contexts + shared rate limiter"
```

---

## Task 12: Migrate the old registry/plugin-loader tests + full regression

**Files:**
- Modify/delete: any remaining tests asserting the old decorator API (`tests/qubx/plugins/test_loader.py`, `tests/qubx/connectors/*`); update fixtures that constructed providers/connectors via the old `get_*` `**kwargs` calls.

- [ ] **Step 1: Find stragglers**

Run: `grep -rn "register_connector\|@connector\|@data_provider\|@rate_limit_config\|get_connector(.*=.*)\|AsyncThreadLoop" tests/ src/` → expect only legitimate new-API usages.

- [ ] **Step 2: Fix each failing test to the new API**

For each, replace decorator/`**kwargs` usage with `ConnectorRegistry.register(FakePlugin())` + context-based `get_*`. Commit per logical group.

- [ ] **Step 3: Full suite**

Run: `just test`
Expected: PASS.

```bash
just style-check
git add tests/
git commit -m "test: migrate connector-registry tests to the ExchangePlugin model"
```

---

## Task 13: Smoke-run a config end-to-end

**Files:** none (validation only)

- [ ] **Step 1: Validate a real config resolves via entry points**

Run: `uv run qubx validate examples/<a ccxt config>.yml` (pick one that uses `connector: ccxt`).
Expected: validates; no registry/loader errors.

- [ ] **Step 2: Paper smoke (no creds)**

Run the existing paper/e2e smoke that builds a ccxt context (`uv run pytest tests/e2e -k paper -v` or the project's designated smoke). Expected: PASS — connector + data provider built through the plugin path, limiter shared.

**Phase 2 gate:** `just test` + smoke green → open PR 2.

---

## Self-Review

**Spec coverage:** ExchangePlugin/contexts (T5) ✓; entry-point discovery (T6) + built-ins (T10) ✓; registry + convenience + tombstone (T7) ✓; runner two-phase + rate_limiter-in-context (T11) ✓; rate-limit ownership unchanged in manager, shared via context (T8/T11) ✓; read_only single gate (T3) ✓; loop required (T4) + run_sync/BackgroundEventLoop + AsyncThreadLoop retirement (T1/T2) ✓; ccxt/tardis migration (T8/T9) ✓; config/release `plugins.modules` kept (no task needed — unchanged) ✓; testing + sequencing (whole plan) ✓.

**Type consistency:** `BuildContext`/`ConnectorBuildContext` field names match across T5/T8/T11; `create_data_provider(ctx)`/`create_connector(ctx)`/`rate_limits(exchange_name)` consistent T5↔T7↔T8↔T9; `PluginLoader.available()/load()` consistent T6↔T7↔T10; `run_sync(loop, coro, timeout=)` consistent T1↔T2.

**Known read-before-write steps (not placeholders):** T3/T8/T9/T11 begin by reading exact existing constructor/dispatch signatures, because the plugin must call the *current* builders unchanged. Each such step names the exact file:line to read.
