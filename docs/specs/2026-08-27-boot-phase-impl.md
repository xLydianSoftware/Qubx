# Boot Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live-strategy boot a first-class phase: strict account-sync gate, unconditional state resolution, an opt-in live boot fit (`set_fit_on_start`), and boot-failure retry/observability.

**Architecture:** A new pure `BootStateMachine` (`src/qubx/core/boot.py`) holds phase, retry bookkeeping, and health-gauge emission. `ProcessingManager._run_strategy_pipeline` drives it via a new `_advance_boot()` that replaces the flag-soup block at `processing.py:502-527`, calling the existing handlers in the same order. Stock resolvers gain empty-output guards plus a new `HOLD` resolver.

**Tech Stack:** Python 3.12, pytest (`uv run pytest`), MagicMock-based unit harnesses (see `tests/qubx/core/mixins/fit_executor_test.py`), ruff (120 cols).

**Spec:** `docs/specs/2026-08-27-boot-phase-design.md` — read it first; it contains the sequencing diagrams, the four-case `on_fit` table, and the rationale each task implements.

## Global Constraints

- Work in the worktree `/home/yuriy/devs/Qubx/.worktrees/boot-phase-spec`. Before Task 1: `git checkout -b feat/boot-phase` (branches off `docs/boot-phase-design-spec`, which contains the spec).
- Run all Python through `uv run` (e.g. `uv run pytest ...`). Full suite: `just test` from the worktree root.
- Modern types only: `list`, `dict`, `| None`, `tuple`. Never `from __future__ import annotations`.
- Logging: `from qubx import logger`.
- Comments terse, only non-obvious constraints. NO banner/section-divider comments. Do not comment what a change fixes.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`). No co-authored-by lines.
- Simulation behavior must not change (regression-checked in Task 8). The warmup-sim context runs the same pipeline with `is_warmup_in_progress=True` — every task touching the pipeline must preserve that path's behavior as documented in the spec diagrams.
- Line references (`processing.py:510` etc.) are as of branch point `57397d47`; re-locate by content if drifted.

---

### Task 1: Resolver empty-output guards, `StateResolver.HOLD`, dead-flag cleanup

**Files:**
- Modify: `src/qubx/restarts/state_resolvers.py`
- Test: `tests/qubx/restarts/test_state_resolvers.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `StateResolver.HOLD(ctx, sim_positions, sim_orders, sim_active_targets) -> None`; module-level `_no_warmup_output(sim_positions, sim_orders, sim_active_targets) -> bool`. `REDUCE_ONLY`/`SYNC_STATE` now return early (no venue reads, no signals) when all three sim args are empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/restarts/test_state_resolvers.py` (follow the file's existing `TestStateResolverBase` pattern — MagicMock ctx, `self._find_instrument("BINANCE.UM", "BTCUSDT")`):

```python
class TestStateResolverEmptyGuards(TestStateResolverBase):
    """All-empty sim args mean 'no warmup output' -> hold the live book, touch nothing."""

    def test_reduce_only_all_empty_holds(self):
        StateResolver.REDUCE_ONLY(self.ctx, {}, {}, {})
        self.ctx.get_positions.assert_not_called()
        self.ctx.emit_signal.assert_not_called()

    def test_sync_state_all_empty_holds(self):
        StateResolver.SYNC_STATE(self.ctx, {}, {}, {})
        self.ctx.get_positions.assert_not_called()
        self.ctx.emit_signal.assert_not_called()
        self.ctx.cancel_orders.assert_not_called()

    def test_close_all_ignores_empty_sim_args(self):
        instr = self._find_instrument("BINANCE.UM", "BTCUSDT")
        pos = MagicMock()
        pos.quantity = 1.0
        self.ctx.get_orders.return_value = {}
        self.ctx.get_positions.return_value = {instr: pos}
        StateResolver.CLOSE_ALL(self.ctx, {}, {}, {})
        self.ctx.emit_signal.assert_called_once()


class TestStateResolverHold(TestStateResolverBase):
    def test_hold_cancels_orders_keeps_positions(self):
        order = MagicMock()
        order.venue_order_id = "v-1"
        order.client_order_id = "c-1"
        self.ctx.get_orders.return_value = {"v-1": order}
        StateResolver.HOLD(self.ctx, {}, {}, {})
        self.ctx.cancel_order.assert_called_once_with(order_id="v-1")
        self.ctx.emit_signal.assert_not_called()

    def test_hold_no_orders_does_nothing(self):
        self.ctx.get_orders.return_value = {}
        StateResolver.HOLD(self.ctx, {}, {}, {})
        self.ctx.cancel_order.assert_not_called()
        self.ctx.emit_signal.assert_not_called()
```

Also UPDATE the existing `test_reduce_only_with_empty_positions` in `TestStateResolverReduceOnly`: it currently asserts `self.ctx.get_positions.assert_called_once()` with all-empty sim args — under the guard the resolver returns before reading positions. Change those assertions to:

```python
        self.ctx.get_positions.assert_not_called()
        self.ctx.emit_signal.assert_not_called()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/qubx/restarts/test_state_resolvers.py -v -x`
Expected: `TestStateResolverHold` fails with `AttributeError: ... has no attribute 'HOLD'`; guard tests fail on `get_positions.assert_not_called()`.

- [ ] **Step 3: Implement guards, HOLD, and cleanup**

In `src/qubx/restarts/state_resolvers.py`:

Add a module-level helper above the class:

```python
def _no_warmup_output(
    sim_positions: dict[Instrument, Position],
    sim_orders: dict[Instrument, list[Order]],
    sim_active_targets: dict[Instrument, TargetPosition],
) -> bool:
    """All-empty sim args ⇔ no warmup sim ran (a sim that ran captures a Position per
    instrument, flat included). Resolvers that steer toward sim state must hold then."""
    if sim_positions or sim_orders or sim_active_targets:
        return False
    logger.warning(
        "<yellow>State resolver received no warmup output — holding the live book as-is. "
        "Register a custom resolver (or StateResolver.HOLD) to silence this.</yellow>"
    )
    return True
```

At the top of `REDUCE_ONLY` (before `live_positions = ctx.get_positions()`):

```python
        if _no_warmup_output(sim_positions, sim_orders, sim_active_targets):
            return
```

Same two lines at the top of `SYNC_STATE`. Do NOT add the guard to `CLOSE_ALL` or `NONE`.

In `SYNC_STATE`, delete the dead `use_limit_order` computation and its kwarg (the field is read nowhere):

```python
        for instrument, a_tgt in sim_active_targets.items():
            s = InitializingSignal(
                time=ctx.time(),
                instrument=instrument,
                signal=a_tgt.target_position_size,
                price=a_tgt.price,
                stop=a_tgt.stop,
                take=a_tgt.take,
            )
            ctx.emit_signal(s)
```

(also remove the now-stale comment block about limit orders above it, and the commented-out legacy code at the bottom of `SYNC_STATE`).

Add `HOLD` after `NONE`:

```python
    @staticmethod
    def HOLD(
        ctx: IStrategyContext,
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Cancel all open live orders, keep all live positions untouched, emit nothing.
        The recommended partner of initializer.set_fit_on_start(True): let the first
        live fit reconcile positions through the strategy's own tracker.
        """
        orders = ctx.get_orders()
        if not orders:
            return
        logger.info(f"HOLD resolver: cancelling {len(orders)} live orders, keeping positions")
        for order in orders.values():
            oid = order.venue_order_id or order.client_order_id
            try:
                ctx.cancel_order(order_id=oid)
            except OrderNotFound:
                logger.debug(f"Order {oid} already cancelled or doesn't exist")
```

Also document in `sim_orders`'s docstring mention (one line in the class docstring): `sim_orders is unused by all stock resolvers; it remains in the signature for custom resolvers.`

- [ ] **Step 4: Run the file's full test suite**

Run: `uv run pytest tests/qubx/restarts/test_state_resolvers.py -v`
Expected: all PASS (including pre-existing REDUCE_ONLY/SYNC_STATE/CLOSE_ALL tests — if any pre-existing SYNC_STATE test asserts `use_limit_order`, update it to not expect the kwarg).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/restarts/state_resolvers.py tests/qubx/restarts/test_state_resolvers.py
git commit -m "feat(restarts): empty-warmup guards for stock resolvers, StateResolver.HOLD, drop dead use_limit_order"
```

---

### Task 2: `set_fit_on_start` initializer API

**Files:**
- Modify: `src/qubx/core/interfaces.py` (in `IStrategyInitializer`, next to `set_state_resolver`/`get_state_resolver` ~:2263-2286)
- Modify: `src/qubx/core/initializer.py` (in `BasicStrategyInitializer`, next to `set_state_resolver` ~:112-118)
- Test: `tests/qubx/core/initializer_test.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `IStrategyInitializer.set_fit_on_start(enabled: bool = True) -> None` and `IStrategyInitializer.get_fit_on_start() -> bool` (default `False`). Task 6 reads `ctx.initializer.get_fit_on_start()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/initializer_test.py` (match its existing construction of `BasicStrategyInitializer` — copy the fixture/instantiation pattern already used there):

```python
def test_fit_on_start_defaults_false_and_toggles():
    init = BasicStrategyInitializer()
    assert init.get_fit_on_start() is False
    init.set_fit_on_start(True)
    assert init.get_fit_on_start() is True
    init.set_fit_on_start(False)
    assert init.get_fit_on_start() is False
```

If `BasicStrategyInitializer()` requires constructor args in this repo, reuse whatever the existing tests in the file pass.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/initializer_test.py -v -k fit_on_start`
Expected: FAIL with `AttributeError: ... 'get_fit_on_start'`.

- [ ] **Step 3: Implement**

`src/qubx/core/interfaces.py`, inside `IStrategyInitializer`, following the exact style of the neighboring `set_state_resolver`/`get_state_resolver` (same body convention those use — `...` or `raise NotImplementedError`, copy it):

```python
    def set_fit_on_start(self, enabled: bool = True) -> None:
        """
        Force the first on_fit to run in the live context even when a warmup-sim fit ran.

        The warmup simulation fits against sim state; strategies that reconcile positions
        through their tracker on fit (e.g. buffered trackers) need the first fit to run
        live, against live positions. Opting in guarantees exactly one live fit after
        on_warmup_finished and replaces calling ctx.trigger_fit() there (doing both
        double-fits). No-op when no warmup sim runs — the first fit is live anyway.
        """
        ...

    def get_fit_on_start(self) -> bool:
        """Whether the first on_fit is forced to run in the live context (default False)."""
        ...
```

`src/qubx/core/initializer.py`, inside `BasicStrategyInitializer`: add a backing field where the class initializes its other fields (e.g. next to the state-resolver field — match how `_state_resolver` is stored):

```python
    def set_fit_on_start(self, enabled: bool = True) -> None:
        self._fit_on_start = enabled

    def get_fit_on_start(self) -> bool:
        return getattr(self, "_fit_on_start", False)
```

(The `getattr` default makes the field optional in `__init__`; if the class body declares fields as annotations, add `_fit_on_start: bool = False` there instead and drop the `getattr`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/initializer_test.py tests/qubx/core/strategy_initializer_test.py -v`
Expected: PASS (both files, no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/interfaces.py src/qubx/core/initializer.py tests/qubx/core/initializer_test.py
git commit -m "feat(core): IStrategyInitializer.set_fit_on_start opt-in live boot fit knob"
```

---

### Task 3: `BootPhase` + `BootStateMachine`

**Files:**
- Create: `src/qubx/core/boot.py`
- Test: `tests/qubx/core/boot_test.py`

**Interfaces:**
- Consumes: `IHealthMonitor.record_gauge(name: str, value: float, tags: dict[str, str] | None = None)`; `dt_64`, `td_64` from `qubx.core.basics`.
- Produces (used verbatim by Tasks 4-7):

```python
class BootPhase(IntEnum):
    BLOCKED = -1
    WAIT_READY = 0
    ON_START = 1
    RESOLVE = 2
    RESTORE = 3
    WARMUP_FINISHED = 4
    BOOT_FIT = 5
    TRADING = 6

class BootStateMachine:
    def __init__(self, health_monitor, *, fit_max_attempts: int = 3, fit_retry_delay: td_64 = ...) -> None
    phase: BootPhase
    blocked_reason: str | None
    fit_attempts: int
    @property is_trading: bool
    @property is_blocked: bool
    def advance(self, phase: BootPhase) -> None
    def account_sync_alert(self) -> None
    def account_synced(self) -> None
    def fit_attempt_allowed(self, now: dt_64) -> bool
    def record_fit_attempt(self) -> None
    def record_fit_failure(self, now: dt_64) -> None
    def record_fit_success(self) -> None
    def record_warmup_finished_failure(self) -> None
```

- [ ] **Step 1: Write the failing tests**

Create `tests/qubx/core/boot_test.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/boot_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qubx.core.boot'`.

- [ ] **Step 3: Implement `src/qubx/core/boot.py`**

```python
from enum import IntEnum

from qubx import logger
from qubx.core.basics import dt_64, td_64


class BootPhase(IntEnum):
    BLOCKED = -1
    WAIT_READY = 0
    ON_START = 1
    RESOLVE = 2
    RESTORE = 3
    WARMUP_FINISHED = 4
    BOOT_FIT = 5
    TRADING = 6


class BootStateMachine:
    """Live-boot phase bookkeeping: current phase, boot-fit retry policy, health gauges.

    Pure state — the ProcessingManager drives transitions and executes side effects
    (see _advance_boot). BLOCKED is sticky for trading; a later successful fit releases
    it (record_fit_success), a late account snapshot releases WAIT_READY (caller-side).
    """

    def __init__(self, health_monitor, *, fit_max_attempts: int = 3, fit_retry_delay: td_64 = td_64(60, "s")) -> None:
        self.phase = BootPhase.WAIT_READY
        self.blocked_reason: str | None = None
        self.fit_attempts = 0
        self._health = health_monitor
        self._fit_max_attempts = fit_max_attempts
        self._fit_retry_delay = fit_retry_delay
        self._next_fit_attempt: dt_64 | None = None
        self._sync_alerted = False

    @property
    def is_trading(self) -> bool:
        return self.phase == BootPhase.TRADING

    @property
    def is_blocked(self) -> bool:
        return self.phase == BootPhase.BLOCKED

    def advance(self, phase: BootPhase) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self._gauge("boot.state", float(phase))

    def account_sync_alert(self) -> None:
        if self._sync_alerted:
            return
        self._sync_alerted = True
        self._gauge("boot.account_sync_blocked", 1.0)
        logger.warning("<yellow>Boot blocked: initial account snapshot not applied — holding until synced</yellow>")

    def account_synced(self) -> None:
        if not self._sync_alerted:
            return
        self._sync_alerted = False
        self._gauge("boot.account_sync_blocked", 0.0)

    def fit_attempt_allowed(self, now: dt_64) -> bool:
        if self.phase != BootPhase.BOOT_FIT:
            return False
        return self._next_fit_attempt is None or now >= self._next_fit_attempt

    def record_fit_attempt(self) -> None:
        self.fit_attempts += 1
        self._gauge("boot.fit_attempts", float(self.fit_attempts))

    def record_fit_failure(self, now: dt_64) -> None:
        if self.is_blocked:
            return
        if self.fit_attempts >= self._fit_max_attempts:
            self.blocked_reason = "boot fit failed"
            self.advance(BootPhase.BLOCKED)
            self._gauge("boot.fit_failed", 1.0)
            logger.error(
                f"<red>Boot fit failed after {self.fit_attempts} attempts — trading blocked, "
                "book unreconciled; a later successful fit will unblock</red>"
            )
        else:
            self._next_fit_attempt = now + self._fit_retry_delay
            logger.warning(f"<yellow>Boot fit failed (attempt {self.fit_attempts}/{self._fit_max_attempts}) — retrying</yellow>")

    def record_fit_success(self) -> None:
        if self.is_blocked:
            self._gauge("boot.fit_failed", 0.0)
            self.blocked_reason = None
        self.advance(BootPhase.TRADING)

    def record_warmup_finished_failure(self) -> None:
        self._gauge("boot.warmup_finished_failed", 1.0)

    def _gauge(self, name: str, value: float) -> None:
        try:
            self._health.record_gauge(name, value)
        except Exception:
            logger.exception(f"failed to record {name} gauge")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/core/boot_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/boot.py tests/qubx/core/boot_test.py
git commit -m "feat(core): BootStateMachine — boot phase, fit retry policy, health gauges"
```

---

### Task 4: Wire the machine into the pipeline (unconditional resolution, parity)

**Files:**
- Modify: `src/qubx/core/mixins/processing.py` (block `:502-527` inside `_run_strategy_pipeline`; `_handle_state_resolution:1096`; `_handle_start:1090`; init section near `:242`)
- Modify: `src/qubx/utils/runner/runner.py:1019-1020` (remove the warmup-only default-resolver install)
- Test: `tests/qubx/core/mixins/boot_pipeline_test.py` (new)

**Interfaces:**
- Consumes: `BootPhase`, `BootStateMachine` (Task 3, exact API above).
- Produces: `ProcessingManager._boot: BootStateMachine`; `ProcessingManager._advance_boot() -> bool` (True ⇔ trading may proceed this pass). `_handle_state_resolution` now falls back to `StateResolver.REDUCE_ONLY` when no resolver is registered, and no longer checks `_is_ready` itself (the machine already did). Tasks 5-7 build on `_boot` and `_advance_boot`.

**Behavioral contract (from the spec — the test list below encodes it):**
1. Live, no warmup output: RESOLVE runs anyway; default resolver = `REDUCE_ONLY` (whose Task-1 guard makes it hold + warn).
2. Live, warmup output present: resolver called with the warmup dicts (as today).
3. Warmup-sim context (`is_simulation=True, is_warmup_in_progress=True`): on_start + fit fire; resolution, restore, and `on_warmup_finished` do NOT (pass-through).
4. Plain simulation: on_start, `on_warmup_finished`, fit fire; resolution and restore do NOT.
5. The pass that fires the boot fit returns `False` (no event processing on the fit pass — parity with old `:526-527`).
6. Warmup fit already latched `is_on_fit_called` (knob off): BOOT_FIT passes through without firing a fit; `on_event` flows.

- [ ] **Step 1: Write the failing tests**

Create `tests/qubx/core/mixins/boot_pipeline_test.py`. Build the harness by adapting `make_thread_pm` from `tests/qubx/core/mixins/fit_executor_test.py` (copy its collaborator mocks verbatim; differences noted inline):

```python
from unittest.mock import MagicMock

import numpy as np

from qubx.core.basics import CtrlChannel
from qubx.core.boot import BootPhase
from qubx.core.interfaces import StrategyState
from qubx.core.mixins.processing import ProcessingManager

T0 = np.datetime64("2025-01-01T00:00:00", "ns")


def make_pm(
    *,
    is_simulation: bool = False,
    is_warmup_in_progress: bool = False,
    is_on_fit_called: bool = False,
    warmup_positions: dict | None = None,
    restored_state=None,
    resolver=None,
    fit_on_start: bool = False,
):
    channel = CtrlChannel("test-databus")
    context = MagicMock()
    context.is_simulation = is_simulation
    context.is_paper_trading = False
    context.instruments = []
    context._strategy_state = StrategyState(
        is_on_init_called=True,
        is_on_start_called=False,
        is_on_warmup_finished_called=False,
        is_on_fit_called=is_on_fit_called,
        is_warmup_in_progress=is_warmup_in_progress,
    )
    context._data_providers = [MagicMock(channel=channel)]
    context.emitter = None
    context._market_data_provider = MagicMock()
    context.get_warmup_positions.return_value = warmup_positions or {}
    context.get_warmup_orders.return_value = {}
    context.get_warmup_active_targets.return_value = {}
    context.get_restored_state.return_value = restored_state
    context.initializer.get_state_resolver.return_value = resolver
    context.initializer.get_fit_on_start.return_value = fit_on_start

    strategy = MagicMock()
    strategy.__class__.__name__ = "TestStrategy"
    strategy.on_fit.return_value = None
    strategy.on_event.return_value = []
    strategy.on_market_data.return_value = []

    cache = MagicMock()
    cache.default_timeframe = "1h"
    market_data = MagicMock()
    market_data.get_market_data_cache.return_value = cache

    subscription_manager = MagicMock()
    subscription_manager.get_base_subscription.return_value = "quote"
    time_provider = MagicMock()
    time_provider.time.return_value = T0
    position_tracker = MagicMock()
    position_tracker.process_signals.return_value = []
    position_tracker.update.return_value = []
    universe_manager = MagicMock()
    universe_manager.is_trading_allowed.return_value = True
    health_monitor = MagicMock()
    health_monitor.return_value.__enter__ = MagicMock(return_value=None)
    health_monitor.return_value.__exit__ = MagicMock(return_value=False)
    account_manager = MagicMock()
    account_manager.is_synced.return_value = True
    account_manager.get_positions.return_value = {}
    account_manager.get_orders.return_value = {}  # _log_state_mismatch iterates these

    pm = ProcessingManager(
        context=context,
        strategy=strategy,
        logging=MagicMock(),
        market_data=market_data,
        subscription_manager=subscription_manager,
        time_provider=time_provider,
        account_manager=account_manager,
        connectors={},
        position_tracker=position_tracker,
        position_gathering=MagicMock(),
        universe_manager=universe_manager,
        scheduler=MagicMock(),
        is_simulation=is_simulation,
        health_monitor=health_monitor,
        delisting_detector=MagicMock(),
        fit_executor="inline",
    )
    pm._is_data_ready = lambda: True  # type: ignore[method-assign]
    pm._init_stage_position_tracker = MagicMock()
    pm._init_stage_position_tracker.process_signals.return_value = []
    pm._init_stage_position_tracker.update.return_value = []
    context._processing_manager = pm
    return pm, context, strategy


def drive(pm, passes: int = 1):
    for _ in range(passes):
        pm._run_strategy_pipeline(None)


class TestLiveBoot:
    def test_resolution_runs_without_warmup_output(self):
        resolver = MagicMock()
        resolver.__name__ = "custom"
        pm, ctx, _ = make_pm(resolver=resolver)
        drive(pm)
        resolver.assert_called_once_with(ctx, {}, {}, {})

    def test_default_resolver_installed_when_none_registered(self):
        pm, ctx, _ = make_pm(resolver=None)
        drive(pm)  # REDUCE_ONLY with empty args -> guard: must not read live positions
        ctx.get_positions.assert_not_called()
        assert pm._boot.phase in (BootPhase.BOOT_FIT, BootPhase.TRADING)

    def test_resolution_receives_warmup_output(self):
        resolver = MagicMock()
        resolver.__name__ = "custom"
        instr = MagicMock()
        instr.symbol = "BTCUSDT"
        pos = MagicMock()
        pos.quantity = 1.0  # real float: _log_state_mismatch formats it with :.6f
        wp = {instr: pos}
        pm, ctx, _ = make_pm(resolver=resolver, warmup_positions=wp)
        drive(pm)
        resolver.assert_called_once_with(ctx, wp, {}, {})

    def test_boot_sequence_reaches_trading_and_fit_pass_returns_false(self):
        pm, ctx, strategy = make_pm()
        assert pm._run_strategy_pipeline(None) is False  # boot pass fires on_start..fit
        strategy.on_start.assert_called_once()
        strategy.on_warmup_finished.assert_called_once()
        strategy.on_fit.assert_called_once()
        assert ctx._strategy_state.is_on_fit_called
        assert pm._boot.is_trading

    def test_warmup_fit_satisfies_boot_fit_when_knob_off(self):
        pm, ctx, strategy = make_pm(is_on_fit_called=True)
        drive(pm)
        strategy.on_fit.assert_not_called()
        assert pm._boot.is_trading

    def test_restore_called_with_restored_state(self):
        restored = MagicMock()
        restored.instrument_to_signal_positions = {}
        restored.instrument_to_target_positions = {}
        pm, ctx, _ = make_pm(restored_state=restored)
        drive(pm)
        assert pm._boot.is_trading


class TestSimulationParity:
    def test_plain_simulation_boots_without_resolution(self):
        resolver = MagicMock()
        pm, ctx, strategy = make_pm(is_simulation=True, resolver=resolver)
        drive(pm)
        resolver.assert_not_called()
        strategy.on_start.assert_called_once()
        strategy.on_warmup_finished.assert_called_once()
        strategy.on_fit.assert_called_once()
        assert pm._boot.is_trading

    def test_warmup_sim_context_skips_warmup_finished_and_restore(self):
        restored = MagicMock()
        pm, ctx, strategy = make_pm(
            is_simulation=True, is_warmup_in_progress=True, restored_state=restored
        )
        drive(pm)
        strategy.on_start.assert_called_once()
        strategy.on_warmup_finished.assert_not_called()
        strategy.on_fit.assert_called_once()
        ctx.get_restored_state.assert_not_called()
        assert pm._boot.is_trading
        assert not ctx._strategy_state.is_on_warmup_finished_called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/mixins/boot_pipeline_test.py -v`
Expected: FAIL — `ProcessingManager` has no `_boot` attribute; the old gate skips resolution with empty warmup output.

- [ ] **Step 3: Implement**

In `src/qubx/core/mixins/processing.py`:

**(a)** Import and construct the machine. Add `from qubx.core.boot import BootPhase, BootStateMachine` to imports. Where the manager initializes instance state (the section around `:242`, next to `self._account_sync_deadline = None`), add:

```python
        self._boot = BootStateMachine(self._health_monitor)
```

**(b)** Replace the boot block in `_run_strategy_pipeline` (`:502-527`, the code from `if not self._context._strategy_state.is_on_start_called:` through the `_handle_fit(...)` / `return False` block) with:

```python
        if not self._advance_boot():
            return False
```

Then DELETE the now-redundant later checks that `_advance_boot` subsumes: the block at `:533-535` (`if not self._context._strategy_state.is_on_fit_called or (not self._is_simulation and not ...is_on_warmup_finished_called): return False`) and the `:539-540` `_warmup_finished_is_running` check. KEEP the `_fit_is_running`/executor-mode check (`:546-547`) — a scheduled threaded fit runs concurrently with trading.

**(c)** Add `_advance_boot` next to `_handle_state_resolution`:

```python
    def _advance_boot(self) -> bool:
        """Drive the boot machine one pipeline pass. True ⇔ the strategy may trade.

        Linearizes the old flag checks: WAIT_READY → ON_START → RESOLVE → RESTORE →
        WARMUP_FINISHED → BOOT_FIT → TRADING. RESOLVE/RESTORE/WARMUP_FINISHED are
        live-boot steps: skipped in simulation (RESOLVE) and while the warmup sim is
        running this pipeline (is_warmup_in_progress). The boot-fit pass returns False
        so no event is processed on it (old :526-527 behavior).
        """
        boot = self._boot
        if boot.is_trading:
            return True
        if boot.is_blocked:
            return False
        state = self._context._strategy_state

        if boot.phase == BootPhase.WAIT_READY:
            if not self._is_ready():
                return False
            boot.advance(BootPhase.ON_START)

        if boot.phase == BootPhase.ON_START:
            if not state.is_on_start_called:
                self._strategy.on_start(self._context)
                state.is_on_start_called = True
            boot.advance(BootPhase.RESOLVE)

        if boot.phase == BootPhase.RESOLVE:
            if not self._is_simulation and not state.is_on_warmup_finished_called:
                self._handle_state_resolution()
            boot.advance(BootPhase.RESTORE)

        if boot.phase == BootPhase.RESTORE:
            if not state.is_warmup_in_progress and not state.is_on_warmup_finished_called:
                restored_state = self._context.get_restored_state()
                if restored_state is not None:
                    self._restore_tracker_and_gatherer_state(restored_state)
            boot.advance(BootPhase.WARMUP_FINISHED)

        if boot.phase == BootPhase.WARMUP_FINISHED:
            if state.is_warmup_in_progress:
                boot.advance(BootPhase.BOOT_FIT)  # warmup-sim ctx: the hook belongs to the live boot
            else:
                if not state.is_on_warmup_finished_called and not self._warmup_finished_is_running:
                    self._handle_warmup_finished()
                if not state.is_on_warmup_finished_called:
                    return False
                if not self._is_simulation and state.is_on_fit_called and self._context.initializer.get_fit_on_start():
                    state.is_on_fit_called = False  # warmup fit doesn't count: force one live fit
                boot.advance(BootPhase.BOOT_FIT)

        if boot.phase == BootPhase.BOOT_FIT:
            if state.is_on_fit_called:
                boot.record_fit_success()
                return True
            if self._fit_is_running or self._warmup_finished_is_running:
                return False
            if boot.fit_attempt_allowed(self._time_provider.time()):
                boot.record_fit_attempt()
                self._handle_fit(None, "fit", (None, self._time_provider.time()))
                if state.is_on_fit_called:
                    boot.record_fit_success()
            return False

        return boot.is_trading
```

Notes on BOOT_FIT: the `state.is_on_fit_called → record_fit_success → return True` head covers the warmup-fit-satisfied case — trading proceeds on that same pass (old behavior, no lost pass). The `if state.is_on_fit_called` check after `_handle_fit(...)` covers the inline mode, which latches synchronously (threaded mode leaves the flag unset here; its outcome arrives via the FitCommit in Task 7). The fit-firing pass always returns `False` — parity with old `:526-527`. Task 7 moves outcome reporting into `_handle_fit` itself and deletes the two post-`_handle_fit` lines.

**(d)** `_handle_start` (`:1090-1094`) is no longer called (its body moved into ON_START). Delete the method. `_handle_state_resolution` (`:1096`): remove its `if not self._is_ready(): return` (the machine gates), and replace the resolver-missing warning with a default:

```python
        resolver = _ctx.initializer.get_state_resolver()
        if resolver is None:
            resolver = StateResolver.REDUCE_ONLY
```

Add `from qubx.restarts.state_resolvers import StateResolver` to the module imports (check for import cycles: `restarts` imports only `qubx.core.*`, so importing it from a mixin is safe; if a cycle does appear, import inside the method). Also remove the `if not self._is_ready(): return` from `_restore_tracker_and_gatherer_state` and `_handle_warmup_finished` (same reason). KEEP the `_is_ready` check in `_handle_fit` — scheduled fits arrive outside `_advance_boot`.

**(e)** `src/qubx/utils/runner/runner.py:1019-1020`: delete the two lines

```python
    if initializer.get_state_resolver() is None:
        initializer.set_state_resolver(StateResolver.REDUCE_ONLY)
```

(the `_handle_state_resolution` fallback replaces them; drop the now-unused `StateResolver` import if nothing else in the file uses it).

- [ ] **Step 4: Run the new tests plus the neighboring pipeline suites**

Run: `uv run pytest tests/qubx/core/mixins/ tests/qubx/core/boot_test.py -v`
Expected: all PASS — including `fit_executor_test.py`, `processing_fit_refresh_test.py`, `processing_pending_signals_test.py`, `test_processing_dispatch.py`. These construct `ProcessingManager` with `is_on_warmup_finished_called=True`-style states; two known hazards when fixing them:

1. If one drives `_run_strategy_pipeline` and stalls in WAIT_READY, patch that test's pm with `pm._is_data_ready = lambda: True` (not by weakening `_advance_boot`).
2. **MagicMock contexts make `context.initializer.get_fit_on_start()` return a truthy MagicMock**, accidentally enabling the knob and resetting `is_on_fit_called`. In any pre-existing harness whose tests reach the pipeline, add `context.initializer.get_fit_on_start.return_value = False` (mirror how `make_pm` does it).

- [ ] **Step 5: Run the broader core + backtester suites (simulation parity)**

Run: `uv run pytest tests/qubx/core/ tests/qubx/backtester/ -x -q --disable-warnings -n 2`
Expected: PASS. Any failure here is a parity break — fix `_advance_boot`, do not adjust simulation expectations.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/mixins/processing.py src/qubx/utils/runner/runner.py tests/qubx/core/mixins/boot_pipeline_test.py
git commit -m "feat(core): boot state machine drives the live boot; state resolution runs unconditionally (#363)"
```

---

### Task 5: Strict account-sync gate

**Files:**
- Modify: `src/qubx/core/mixins/processing.py` (`_is_ready`, `:918-945`)
- Test: `tests/qubx/core/mixins/boot_pipeline_test.py` (extend)

**Interfaces:**
- Consumes: `_boot.account_sync_alert()` / `_boot.account_synced()` (Task 3).
- Produces: `_is_ready()` never returns True in live non-paper while `account_manager.is_synced()` is False. `ACCOUNT_SYNC_TIMEOUT` (15s, unchanged) is now the alert threshold, not a fall-through.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/core/mixins/boot_pipeline_test.py`:

```python
class TestStrictAccountSyncGate:
    def test_boot_holds_until_synced_and_alerts_after_timeout(self):
        pm, ctx, strategy = make_pm()
        pm._account_manager.is_synced.return_value = False
        tp = pm._time_provider

        drive(pm)  # arms the deadline
        assert pm._boot.phase == BootPhase.WAIT_READY
        strategy.on_start.assert_not_called()

        tp.time.return_value = T0 + np.timedelta64(20, "s")  # past ACCOUNT_SYNC_TIMEOUT (15s)
        drive(pm)
        assert pm._boot.phase == BootPhase.WAIT_READY
        strategy.on_start.assert_not_called()
        pm._health_monitor.record_gauge.assert_any_call("boot.account_sync_blocked", 1.0)

    def test_boot_proceeds_when_snapshot_lands_late(self):
        pm, ctx, strategy = make_pm()
        pm._account_manager.is_synced.return_value = False
        tp = pm._time_provider
        drive(pm)
        tp.time.return_value = T0 + np.timedelta64(30, "s")
        drive(pm)
        pm._account_manager.is_synced.return_value = True
        drive(pm)
        strategy.on_start.assert_called_once()
        pm._health_monitor.record_gauge.assert_any_call("boot.account_sync_blocked", 0.0)
```

If `pm._account_manager` / `pm._time_provider` / `pm._health_monitor` are not the attribute names on the manager, find the actual private names in `ProcessingManager`'s field list and use those (they are the constructor kwargs with a leading underscore).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/mixins/boot_pipeline_test.py -v -k StrictAccountSync`
Expected: FAIL — old fall-through lets `on_start` fire after the timeout.

- [ ] **Step 3: Implement**

Replace `_is_ready`'s account section (`:929-945`) with:

```python
        if self._context.is_simulation or self._context.is_paper_trading:
            return True
        if self._account_manager.is_synced():
            self._boot.account_synced()
            return True
        # - data ready but the initial venue snapshot hasn't applied: hold the boot.
        #   ACCOUNT_SYNC_TIMEOUT is the alert threshold, not a fall-through — trading
        #   against phantom-zero positions can double-open the book.
        now = self._time_provider.time()
        if self._account_sync_deadline is None:
            self._account_sync_deadline = now + self.ACCOUNT_SYNC_TIMEOUT
        elif now >= self._account_sync_deadline:
            self._boot.account_sync_alert()
        return False
```

Update the method docstring accordingly (remove the "falls through" paragraph). Delete the now-unused `_account_sync_timeout_logged` field and its reset.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/core/mixins/boot_pipeline_test.py tests/qubx/core/mixins/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/mixins/processing.py tests/qubx/core/mixins/boot_pipeline_test.py
git commit -m "feat(core): strict account-sync gate for the boot phase — alert and hold, no fall-through"
```

---

### Task 6: `set_fit_on_start` end-to-end behavior

**Files:**
- Test: `tests/qubx/core/mixins/boot_pipeline_test.py` (extend; the production hook was wired in Task 4's WARMUP_FINISHED step)

**Interfaces:**
- Consumes: `context.initializer.get_fit_on_start()` (Task 2), `_advance_boot` (Task 4).
- Produces: verified four-case behavior table from the spec.

- [ ] **Step 1: Write the tests (some may already pass — that's expected; they pin the contract)**

```python
class TestFitOnStart:
    def test_knob_forces_exactly_one_live_fit_after_warmup(self):
        pm, ctx, strategy = make_pm(is_on_fit_called=True, fit_on_start=True)
        drive(pm, passes=3)
        strategy.on_fit.assert_called_once()
        assert pm._boot.is_trading

    def test_knob_off_no_live_fit_when_warmup_fit_ran(self):
        pm, ctx, strategy = make_pm(is_on_fit_called=True, fit_on_start=False)
        drive(pm, passes=3)
        strategy.on_fit.assert_not_called()
        assert pm._boot.is_trading

    def test_no_double_fit_when_warmup_ran_no_fit(self):
        # LIGHTER case: flag unset after warmup; knob on must still yield exactly one fit
        pm, ctx, strategy = make_pm(is_on_fit_called=False, fit_on_start=True)
        drive(pm, passes=3)
        strategy.on_fit.assert_called_once()
        assert pm._boot.is_trading

    def test_knob_ignored_in_simulation(self):
        pm, ctx, strategy = make_pm(is_simulation=True, is_on_fit_called=True, fit_on_start=True)
        drive(pm, passes=3)
        strategy.on_fit.assert_not_called()
        assert pm._boot.is_trading
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/qubx/core/mixins/boot_pipeline_test.py -v -k FitOnStart`
Expected: PASS if Task 4's reset line is correct; any FAIL means the reset fires in the wrong phase or mode — fix in `_advance_boot`'s WARMUP_FINISHED step (the reset must be: live only, once, only when the flag was set by a warmup fit).

- [ ] **Step 3: Commit**

```bash
git add tests/qubx/core/mixins/boot_pipeline_test.py
git commit -m "test(core): pin set_fit_on_start four-case boot-fit contract"
```

---

### Task 7: Boot-fit failure handling (latch-on-success, retry, BLOCKED, self-heal)

**Files:**
- Modify: `src/qubx/core/mixins/processing.py` (`__invoke_on_fit:733-752`, `_handle_fit:1174-1196`, `_run_fit_off_thread:1226-1280`, `_handle_fit_commit:1288-1318`, `_handle_warmup_finished:1161` + `__invoke_on_warmup_finished:754-771`)
- Modify: `src/qubx/core/fit_executor.py` (`FitCommitData`)
- Test: `tests/qubx/core/mixins/boot_pipeline_test.py` (extend) and `tests/qubx/core/mixins/fit_executor_test.py` (extend)

**Interfaces:**
- Consumes: `_boot.record_fit_failure(now)`, `_boot.record_fit_success()`, `_boot.record_warmup_finished_failure()`, `_boot.is_trading` (Task 3).
- Produces: `FitCommitData.error: BaseException | None = None`; `__invoke_on_fit() -> bool` (success); `__invoke_on_warmup_finished() -> bool` (success). Live latch rule: `is_on_fit_called` set only on success; simulation keeps latch-always.

- [ ] **Step 1: Write the failing tests (inline mode + hook failure)**

Append to `tests/qubx/core/mixins/boot_pipeline_test.py`:

```python
class TestBootFitFailure:
    def test_failed_boot_fit_retries_then_blocks(self):
        pm, ctx, strategy = make_pm()
        strategy.on_fit.side_effect = RuntimeError("boom")
        tp = pm._time_provider

        drive(pm)  # attempt 1
        assert not ctx._strategy_state.is_on_fit_called
        assert pm._boot.phase == BootPhase.BOOT_FIT

        drive(pm)  # before the retry deadline: no new attempt
        assert strategy.on_fit.call_count == 1

        tp.time.return_value = T0 + np.timedelta64(61, "s")
        drive(pm)  # attempt 2
        tp.time.return_value = T0 + np.timedelta64(122, "s")
        drive(pm)  # attempt 3 -> exhausted
        assert strategy.on_fit.call_count == 3
        assert pm._boot.is_blocked
        pm._health_monitor.record_gauge.assert_any_call("boot.fit_failed", 1.0)

        drive(pm)  # blocked: no on_event, no more attempts
        assert strategy.on_fit.call_count == 3
        strategy.on_event.assert_not_called()

    def test_blocked_self_heals_on_later_successful_fit(self):
        pm, ctx, strategy = make_pm()
        strategy.on_fit.side_effect = RuntimeError("boom")
        tp = pm._time_provider
        drive(pm)
        tp.time.return_value = T0 + np.timedelta64(61, "s")
        drive(pm)
        tp.time.return_value = T0 + np.timedelta64(122, "s")
        drive(pm)
        assert pm._boot.is_blocked

        strategy.on_fit.side_effect = None  # a scheduled fit arrives via _handle_fit
        pm._handle_fit(None, "fit", (None, tp.time.return_value))
        assert ctx._strategy_state.is_on_fit_called
        assert pm._boot.is_trading
        pm._health_monitor.record_gauge.assert_any_call("boot.fit_failed", 0.0)

    def test_warmup_finished_failure_latches_and_gauges(self):
        pm, ctx, strategy = make_pm()
        strategy.on_warmup_finished.side_effect = RuntimeError("hook boom")
        drive(pm, passes=2)
        assert ctx._strategy_state.is_on_warmup_finished_called  # latched as before
        pm._health_monitor.record_gauge.assert_any_call("boot.warmup_finished_failed", 1.0)
        assert pm._boot.is_trading  # boot continued to the fit

    def test_simulation_keeps_latch_on_failure(self):
        pm, ctx, strategy = make_pm(is_simulation=True)
        strategy.on_fit.side_effect = RuntimeError("boom")
        drive(pm, passes=2)
        assert ctx._strategy_state.is_on_fit_called  # sim: latch-in-finally preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/mixins/boot_pipeline_test.py -v -k "BootFitFailure"`
Expected: FAIL — the old `finally` latches `is_on_fit_called` even on error, so no retry happens.

- [ ] **Step 3: Implement inline-path + hook changes**

**(a)** `__invoke_on_fit` — return success, latch only on success in live (`:733-752`):

```python
    def __invoke_on_fit(self) -> bool:
        with self._health_monitor("ctx.on_fit"):
            try:
                self._context._instrument_service_manager.enforce_at_fit()
                logger.debug(f"[<y>{self.__class__.__name__}</y>] :: Invoking <g>{self._strategy_name}</g> on_fit")
                self._strategy.on_fit(self._context)
                self._subscription_manager.commit()  # apply pending operations
                logger.debug(f"[<y>{self.__class__.__name__}</y>] :: <g>{self._strategy_name}</g> is fitted")
                self._context._strategy_state.is_on_fit_called = True
                return True
            except Exception as strat_error:
                logger.error(
                    f"[{self.__class__.__name__}] :: Strategy {self._strategy_name} on_fit raised an exception: {strat_error}"
                )
                logger.opt(colors=False).error(traceback.format_exc())
                # - simulation keeps the latch: no retry machinery there, and a raising
                #   fit must not re-fire every tick (would change backtest behavior)
                if self._is_simulation:
                    self._context._strategy_state.is_on_fit_called = True
                return False
```

(preserve the existing comment above `enforce_at_fit` verbatim.)

**(b)** `_handle_fit` inline branch (`:1188-1196`) — report to the machine when boot is not done:

```python
        self._fit_is_running = True
        try:
            current_time = data[1]
            self._cache.finalize_ohlc_for_instruments(current_time, self._context.instruments)
            ok = self.__invoke_on_fit()
        finally:
            self._fit_is_running = False
        if not self._is_simulation and not self._boot.is_trading:
            if ok:
                self._boot.record_fit_success()
            else:
                self._boot.record_fit_failure(self._time_provider.time())
```

Then in Task 4's `_advance_boot` BOOT_FIT step, DELETE the post-`_handle_fit` `if state.is_on_fit_called: boot.record_fit_success()` correction lines — `_handle_fit` now reports both outcomes itself.

**(c)** `_handle_warmup_finished` / `__invoke_on_warmup_finished` (`:754-771`, `:1161-1172`): make the invoker return success (True on no exception; keep the latch in `finally` exactly as-is), and in `_handle_warmup_finished` after the call:

```python
            if not self.__invoke_on_warmup_finished():
                self._boot.record_warmup_finished_failure()
```

(adjust: the current code calls `self.__invoke_on_warmup_finished()` inside the try — capture its return there.)

- [ ] **Step 4: Run inline tests to verify they pass**

Run: `uv run pytest tests/qubx/core/mixins/boot_pipeline_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the failing threaded-path test**

Append to `tests/qubx/core/mixins/fit_executor_test.py` (reuse its `make_thread_pm` + `drain_until_committed` helpers):

```python
def test_threaded_fit_failure_does_not_latch_and_reports_boot_failure():
    pm, context, channel = make_thread_pm()
    pm._strategy.on_fit.side_effect = RuntimeError("thread boom")
    pm._boot.advance(BootPhase.BOOT_FIT)
    pm._boot.record_fit_attempt()

    pm._handle_fit(None, "fit", (None, T0))
    drain_until_committed(pm, channel)

    assert not context._strategy_state.is_on_fit_called
    assert pm._boot.phase == BootPhase.BOOT_FIT  # retry pending, not blocked (attempt 1/3)


def test_threaded_fit_success_latches_and_reaches_trading():
    pm, context, channel = make_thread_pm()
    pm._strategy.on_fit.side_effect = None
    pm._boot.advance(BootPhase.BOOT_FIT)
    pm._boot.record_fit_attempt()

    pm._handle_fit(None, "fit", (None, T0))
    drain_until_committed(pm, channel)

    assert context._strategy_state.is_on_fit_called
    assert pm._boot.is_trading
```

Add `from qubx.core.boot import BootPhase` to that file's imports. If `make_thread_pm`'s context state has `is_on_warmup_finished_called=True` and the machine starts in WAIT_READY, that is fine — these tests drive the machine explicitly.

- [ ] **Step 6: Run to verify the failure test fails**

Run: `uv run pytest tests/qubx/core/mixins/fit_executor_test.py -v -k threaded_fit_failure`
Expected: FAIL — `_handle_fit_commit` latches unconditionally today.

- [ ] **Step 7: Implement the threaded path**

`src/qubx/core/fit_executor.py`, `FitCommitData`:

```python
    ops: tuple[Callable[[], None], ...] = ()
    signals: tuple[Signal, ...] = ()
    duration_s: float = 0.0
    error: BaseException | None = None
```

`_run_fit_off_thread` (`:1226-1280`): capture the strategy error — in the `except Exception as strat_error:` branch add `_fit_error = strat_error` (initialize `_fit_error: BaseException | None = None` before the try), and pass it in the finally's send:

```python
            channel.send(
                (None, FIT_COMMIT_EVENT, FitCommitData(ops=_ops, signals=_signals, duration_s=_duration_s, error=_fit_error), False)
            )
```

`_handle_fit_commit` finally block (`:1314-1318`):

```python
        finally:
            if commit.signals:
                self._emitted_signals.extend(commit.signals)
            if commit.error is None:
                self._context._strategy_state.is_on_fit_called = True
            self._fit_is_running = False
            if not self._boot.is_trading:
                if commit.error is None:
                    self._boot.record_fit_success()
                else:
                    self._boot.record_fit_failure(self._time_provider.time())
```

- [ ] **Step 8: Run the full mixin suite**

Run: `uv run pytest tests/qubx/core/mixins/ tests/qubx/core/boot_test.py -v`
Expected: all PASS (existing fit_executor tests included — none of them raise in `on_fit`, so the latch behavior they see is unchanged).

- [ ] **Step 9: Commit**

```bash
git add src/qubx/core/mixins/processing.py src/qubx/core/fit_executor.py tests/qubx/core/mixins/boot_pipeline_test.py tests/qubx/core/mixins/fit_executor_test.py
git commit -m "feat(core): boot-fit failure handling — latch on success only, bounded retry, BLOCKED with self-heal"
```

---

### Task 8: Full-suite verification and style

**Files:** none new.

- [ ] **Step 1: Style check**

Run: `just style-check`
Expected: clean. Fix any ruff findings in files this branch touched.

- [ ] **Step 2: Full test suite**

Run: `just test`
Expected: PASS. Simulation-behavior failures (backtester suites) are parity breaks — fix in `_advance_boot`/`__invoke_on_fit`, never by changing simulation test expectations. Live-runner suites (`tests/qubx/utils/runner/` if present) may assert the old default-resolver install in `_run_warmup` — update those to the new contract (fallback lives in `_handle_state_resolution`).

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix(core): full-suite fixes for boot phase wiring"
```

(skip if nothing changed)

---

### Task 9: Boot lifecycle documentation (#388)

**Files:**
- Create: `docs/trading/boot-lifecycle.md`
- Modify: `mkdocs.yml` (nav — add the page next to the other `trading/` entries; if `docs/trading/` pages are not in nav, mirror however its siblings are registered)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Write the page**

Create `docs/trading/boot-lifecycle.md` with these sections, sourcing content from `docs/specs/2026-08-27-boot-phase-design.md` (adapt, don't copy verbatim — the audience is strategy authors, not framework reviewers):

1. **Two warmups** — warmup simulation vs subscription warmup (spec "Terminology", one paragraph each).
2. **Boot sequence** — both ASCII diagrams from the spec ("Boot sequencing (reference)"), unchanged.
3. **When `on_fit` runs** — the four-case table from the spec, plus the invariant sentence ("`on_event` is never delivered before a successful fit").
4. **`on_start` fires in the warmup context** — the #388 asymmetry: with warmup, `on_start` ran in the sim against the sim universe; per-instrument state must be built lazily on first use or rebuilt in `on_warmup_finished`, never only in `on_start`; `ctx.instruments` at live start can contain held instruments the strategy never selected.
5. **State resolvers** — the contract (runs at every live boot; all-empty sim args ⇔ no warmup output; stock resolvers hold + warn then), and the recommended-patterns table from the spec ("Documentation deliverable" section: HOLD+`set_fit_on_start` / SYNC_STATE / REDUCE_ONLY / custom).
6. **`set_fit_on_start`** — what it guarantees, that it replaces `trigger_fit()` in `on_warmup_finished`, no-op without warmup.
7. **Boot health** — the five `boot.*` gauges table from the spec and what BLOCKED means operationally (self-heals on a later successful fit or account snapshot).

- [ ] **Step 2: Verify docs build (if mkdocs is wired locally)**

Run: `uv run mkdocs build --strict 2>&1 | tail -5` (skip without failing the task if mkdocs isn't in the dev deps — check `just update-docs` in the justfile as the alternative).
Expected: no errors/warnings about the new page.

- [ ] **Step 3: Commit**

```bash
git add docs/trading/boot-lifecycle.md mkdocs.yml
git commit -m "docs: strategy boot lifecycle — phases, resolvers, set_fit_on_start, boot health (#388)"
```

---

## Self-Review Notes (already applied)

- Spec coverage: §1 machine → Tasks 3-4; §2 strict gate → Task 5; §3 unconditional resolution + guards + HOLD + default move + cleanup → Tasks 1, 4; §4 knob → Tasks 2, 4, 6; §5 failure handling incl. threaded path and sim latch carve-out → Task 7; §6 gauges → Tasks 3, 5, 7; docs deliverable → Task 9; testing §each → per-task tests + Task 8 parity run; rollout/changelog → release notes ride the conventional-commit messages (Qubx changelog is generated).
- The old `:521` fallback, the `:533-535` and `:539-540` gates are subsumed by `_advance_boot`; the `:546-547` threaded-fit re-entrancy guard stays (documented in Task 4b).
- Type consistency: `BootPhase`/`BootStateMachine` names and signatures are identical in Tasks 3 (definition), 4-7 (use). `FitCommitData.error: BaseException | None` defined in Task 7 and used only there.
