# Dynamic-universe delisting resilience + trigger_fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dynamically-added (`add_instruments`) delisted instruments be settled/excluded like `set_universe` does, let the aggregator skip gone instruments at the source, and add an on-demand `trigger_fit` control action — shipping as qubx 1.6.1 + a quantkit tag.

**Architecture:** Qubx components — (1) delegate `is_instrument_listed` on `StrategyContext`; (2) run the existing `_drop_gone` in `UniverseManager.add_instruments`; (3) a `trigger_fit` control action backed by `ctx.trigger_fit()` that injects a one-off fit via `ctx.delay` → `_handle_fit`. QuantKit component — (4) `AggregatorStrategy.on_event` filters out gone instruments before universe-update and target processing.

**Tech Stack:** Python 3.12, pytest + pytest-mock. Qubx repo: `~/devs/Qubx` (branch `feat/dynamic-universe-delisting-and-trigger-fit`). QuantKit repo: `~/devs/quantkit`.

**Spec:** `docs/superpowers/specs/2026-06-16-dynamic-universe-delisting-and-trigger-fit-design.md`

**Run a single test:** `uv run pytest <path>::<test> -v`. Conventional commits, NO `Co-Authored-By`.

---

### Task 1: Delegate `is_instrument_listed` on `StrategyContext` (qubx)

`StrategyContext` inherits `IMarketManager` but doesn't delegate `is_instrument_listed` to `self._market_data_provider`, so `ctx.is_instrument_listed(...)` returns the interface stub's `None`. Add the delegation (prerequisite for Task 4).

**Files:**
- Modify: `src/qubx/core/context.py` (the IMarketDataProvider delegation block, after `get_market_data_cache`/`get_aux_data_storage`, ~line 686)
- Test: `tests/qubx/core/context_delegation_test.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/core/context_delegation_test.py`:

```python
from pytest_mock import MockerFixture

from qubx.core.basics import Instrument
from qubx.core.context import StrategyContext


def test_is_instrument_listed_delegates_to_market_manager(mocker: MockerFixture):
    ctx = StrategyContext.__new__(StrategyContext)  # bypass heavy __init__
    mm = mocker.Mock()
    mm.is_instrument_listed.return_value = False
    ctx._market_data_provider = mm
    instr = mocker.Mock(spec=Instrument)

    result = ctx.is_instrument_listed(instr)

    assert result is False
    mm.is_instrument_listed.assert_called_once_with(instr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/context_delegation_test.py -v`
Expected: FAIL — returns `None` (the `IMarketManager` stub), not `False`.

- [ ] **Step 3: Add the delegation**

In `src/qubx/core/context.py`, in the `# :: IMarketDataProvider delegation ::` block, add (next to `get_market_data_cache`):

```python
    def is_instrument_listed(self, instrument: Instrument) -> bool:
        return self._market_data_provider.is_instrument_listed(instrument)
```

(`Instrument` is already imported in context.py.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/context_delegation_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/context.py tests/qubx/core/context_delegation_test.py
git commit -m "fix(context): delegate is_instrument_listed to the market manager"
```

---

### Task 2: `add_instruments` runs the gone-filter (qubx)

`UniverseManager.add_instruments` (`core/mixins/universe.py:125`) doesn't run `_drop_gone`. Add it at the top, mirroring `set_universe`, so dynamically-added delisted instruments are settled + excluded.

**Files:**
- Modify: `src/qubx/core/mixins/universe.py` (`add_instruments`, ~line 125)
- Test: `tests/qubx/core/universe_manager_test.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/universe_manager_test.py` (reuses the existing `_gone_instr` helper and `mock_dependencies`/`universe_manager` fixtures):

```python
def test_add_instruments_excludes_gone_and_settles(universe_manager, mock_dependencies, mocker):
    mock_dependencies["subscription_manager"].auto_subscribe = True
    live = mocker.Mock(spec=Instrument, symbol="BTCUSDT")
    live.exchange = "OKX.F"
    live.delist_date = None
    live.min_size = 0.001
    gone = _gone_instr(mocker)

    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = lambda i: i is not gone
    pos = mocker.Mock()
    pos.quantity = 3175.0
    mock_dependencies["account"].positions = {gone: pos}

    universe_manager.add_instruments([live, gone])

    assert live in universe_manager.instruments
    assert gone not in universe_manager.instruments
    mock_dependencies["account"].settle_position.assert_called_once_with(gone)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/universe_manager_test.py::test_add_instruments_excludes_gone_and_settles -v`
Expected: FAIL — `gone` is added (and `settle_position` not called) because `add_instruments` skips `_drop_gone`.

- [ ] **Step 3: Add `_drop_gone` to `add_instruments`**

In `src/qubx/core/mixins/universe.py`, read the current `add_instruments` body, then insert `_drop_gone` as the first line:

```python
    def add_instruments(self, instruments: list[Instrument]):
        # Settle & exclude already-gone markets (same gone-filter as set_universe).
        # Only _drop_gone (already-gone), NOT filter_delistings (future/scheduled),
        # so an explicitly-added still-listed instrument with a future delist_date
        # stays addable and is handled by the existing scheduled-delist path.
        instruments = self._drop_gone(instruments)
        to_add = list(set([instr for instr in instruments if instr not in self._instruments]))
        self.__do_add_instruments(to_add)
        self.__cleanup_removal_queue(instruments)
        self._strategy.on_universe_change(self._context, to_add, [])
        self._subscription_manager.commit()
        self._instruments.update(to_add)
```

IMPORTANT: read the real current `add_instruments` first and preserve every existing line — only prepend the `instruments = self._drop_gone(instruments)` line. (Note `__do_add_instruments`/`__cleanup_removal_queue` are name-mangled; keep the exact existing calls.)

- [ ] **Step 4: Run the new test + full universe suite**

Run: `uv run pytest tests/qubx/core/universe_manager_test.py -v`
Expected: PASS (new test + all existing).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/mixins/universe.py tests/qubx/core/universe_manager_test.py
git commit -m "fix(universe): run gone-filter in add_instruments (dynamic universes)"
```

---

### Task 3: `trigger_fit` — context method + control action (qubx)

Add `ctx.trigger_fit()` (one-off fit on the strategy thread via `delay` → `_handle_fit`) and a ⚠ dangerous `trigger_fit` control action.

**Files:**
- Modify: `src/qubx/core/mixins/processing.py` (add `trigger_fit` near `delay`)
- Modify: `src/qubx/core/interfaces.py` (declare `trigger_fit` on `IStrategyContext`, near `delay` at ~1769)
- Modify: `src/qubx/control/builtin.py` (add `_trigger_fit` handler + `BUILTIN_ACTIONS["trigger_fit"]`)
- Test: `tests/qubx/control/test_trigger_fit_action.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/control/test_trigger_fit_action.py`:

```python
from pytest_mock import MockerFixture

from qubx.control.builtin import BUILTIN_ACTIONS


def test_trigger_fit_action_registered_and_dangerous():
    assert "trigger_fit" in BUILTIN_ACTIONS
    action_def, handler = BUILTIN_ACTIONS["trigger_fit"]
    assert action_def.dangerous is True
    assert action_def.read_only is False


def test_trigger_fit_handler_calls_ctx_trigger_fit(mocker: MockerFixture):
    _, handler = BUILTIN_ACTIONS["trigger_fit"]
    ctx = mocker.Mock()
    result = handler(ctx)
    ctx.trigger_fit.assert_called_once_with()
    assert result.status == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/control/test_trigger_fit_action.py -v`
Expected: FAIL — `"trigger_fit"` not in `BUILTIN_ACTIONS`.

- [ ] **Step 3: Add `trigger_fit` to the processing mixin**

In `src/qubx/core/mixins/processing.py`, add a method next to `delay` (find `def delay(self, duration: str, method`):

```python
    def trigger_fit(self) -> None:
        """Run on_fit once, on demand. Schedules a one-off event that invokes the
        fit handler on the strategy thread (via delay → _handle_fit). Uses a unique
        delay event id, so the recurring fit schedule is untouched."""
        self.delay("1s", lambda c: c._handle_fit(None, "fit", (None, c.time())))
```

- [ ] **Step 4: Declare `trigger_fit` on the interface**

In `src/qubx/core/interfaces.py`, in `class IStrategyContext` (near the `delay` declaration, ~line 1769), add:

```python
    def trigger_fit(self) -> None:
        """Run on_fit once, on demand (on the strategy thread)."""
        ...
```

- [ ] **Step 5: Add the control action**

In `src/qubx/control/builtin.py`, add the handler (near `_emit_signal`, ~line 616):

```python
def _trigger_fit(ctx: IStrategyContext, **kwargs) -> ActionResult:
    ctx.trigger_fit()
    return ActionResult(status="ok", data={"status": "fit scheduled"})
```

And register it in the `BUILTIN_ACTIONS` dict (add an entry, e.g. after the trading block):

```python
    "trigger_fit": (
        ActionDef(
            name="trigger_fit",
            description="Trigger a strategy fit (recompute targets / rebalance) on demand",
            category="strategy",
            dangerous=True,
        ),
        _trigger_fit,
    ),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/qubx/control/test_trigger_fit_action.py -v`
Expected: PASS (2)

- [ ] **Step 7: Commit**

```bash
git add src/qubx/core/mixins/processing.py src/qubx/core/interfaces.py src/qubx/control/builtin.py tests/qubx/control/test_trigger_fit_action.py
git commit -m "feat(control): add trigger_fit action + ctx.trigger_fit() (on-demand fit)"
```

---

### Task 4: Aggregator skips gone instruments at the source (quantkit)

In `~/devs/quantkit`, make `AggregatorStrategy.on_event` filter out targets whose instrument is not listed, before `_update_universe` and `process_targets`.

**Files:**
- Modify: `~/devs/quantkit/src/quantkit/aggregation/strategy.py` (`on_event`, ~line 154)
- Test: `~/devs/quantkit/tests/quantkit/aggregation/test_strategy_gone_filter.py` (new)

- [ ] **Step 1: Create a branch in quantkit**

```bash
cd ~/devs/quantkit && git checkout main && git pull --ff-only && git checkout -b feat/aggregator-skip-gone-instruments
```

- [ ] **Step 2: Write the failing test**

Create `~/devs/quantkit/tests/quantkit/aggregation/test_strategy_gone_filter.py`:

```python
from unittest.mock import Mock

import pandas as pd

from quantkit.aggregation import AggregatorStrategy
from quantkit.aggregation.config import Target


def _instr(symbol):
    i = Mock()
    i.symbol = symbol
    i.__hash__ = lambda self=i: hash(symbol)
    i.__eq__ = lambda other, self=i: getattr(other, "symbol", None) == symbol
    return i


def test_on_event_filters_gone_instruments(mocker):
    strat = AggregatorStrategy.__new__(AggregatorStrategy)  # bypass __init__
    btc, ton = _instr("BTCUSDT"), _instr("TONUSDT")

    src = Mock()
    src.get_targets.return_value = [
        Target(time=pd.Timestamp("2026-06-16").asm8, source="s", instrument=btc, quantity=1.0),
        Target(time=pd.Timestamp("2026-06-16").asm8, source="s", instrument=ton, quantity=5.0),
    ]
    strat._sources = {"s": src}
    strat._state_manager = Mock()
    strat._state_manager.process_targets.return_value = []
    strat._track_signals = Mock()
    strat._update_universe = Mock()

    ctx = Mock()
    ctx.is_instrument_listed.side_effect = lambda i: i.symbol != "TONUSDT"
    ctx.is_live = False

    strat.on_event(ctx, Mock())

    # gone instrument excluded from the universe update
    universe_arg = strat._update_universe.call_args.args[1]
    assert btc in universe_arg and ton not in universe_arg
    # and excluded from target processing
    targets_arg = strat._state_manager.process_targets.call_args.args[1]
    assert all(t.instrument.symbol != "TONUSDT" for t in targets_arg)
    assert any(t.instrument.symbol == "BTCUSDT" for t in targets_arg)
```

(If `Target`'s import path or fields differ, adjust to the real `quantkit.aggregation` definitions — read `config.py`/`sources.py` first.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/devs/quantkit && uv run pytest tests/quantkit/aggregation/test_strategy_gone_filter.py -v`
Expected: FAIL — TON is included in the universe and targets.

- [ ] **Step 4: Implement the filter in `on_event`**

In `~/devs/quantkit/src/quantkit/aggregation/strategy.py`, change `on_event` (preserve everything else):

```python
    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        all_targets: list[Target] = []
        for source in self._sources.values():
            all_targets.extend(source.get_targets(ctx))

        # Skip instruments whose market is gone (delisted/removed). Fail-open:
        # is_instrument_listed returns True when markets aren't loaded yet.
        live_targets = [t for t in all_targets if ctx.is_instrument_listed(t.instrument)]
        if len(live_targets) != len(all_targets):
            dropped = sorted({t.instrument.symbol for t in all_targets} - {t.instrument.symbol for t in live_targets})
            logger.warning(f"[aggregator] skipping delisted/gone instruments: {dropped}")

        instruments = set(target.instrument for target in live_targets)
        self._update_universe(ctx, instruments)
        signals = self._state_manager.process_targets(ctx, live_targets)
        self._track_signals(signals)
        if ctx.is_live:
            ctx.persistence.save(MANAGER_STATE_KEY, self._state_manager.get_state(ctx).model_dump())
        return signals
```

Confirm `logger` is imported in `strategy.py` (it is, used elsewhere).

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/devs/quantkit && uv run pytest tests/quantkit/aggregation/test_strategy_gone_filter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd ~/devs/quantkit
git add src/quantkit/aggregation/strategy.py tests/quantkit/aggregation/test_strategy_gone_filter.py
git commit -m "fix(aggregation): skip gone (delisted) instruments in on_event"
```

---

### Task 5: Full suites + lint (both repos)

- [ ] **Step 1: Qubx unit suite**

Run (in `~/devs/Qubx`): `just test` (and ccxt-touching tests aren't involved here, but run `uv run --extra connectors pytest tests/qubx/core tests/qubx/control -q` to be safe).
Expected: all pass. Note any pre-existing `test_ohlc_pagination.py` failures (unrelated, documented).

- [ ] **Step 2: Qubx lint**

Run: `uv run ruff check src/qubx/core/context.py src/qubx/core/mixins/universe.py src/qubx/core/mixins/processing.py src/qubx/core/interfaces.py src/qubx/control/builtin.py tests/qubx/core tests/qubx/control`
Fix any new issues.

- [ ] **Step 3: QuantKit suite + lint**

Run (in `~/devs/quantkit`): `uv run pytest tests/quantkit/aggregation -q` and `uv run ruff check src/quantkit/aggregation/strategy.py tests/quantkit/aggregation/test_strategy_gone_filter.py`
Expected: pass / clean.

- [ ] **Step 4: Commit any lint fixes** (only if needed)

```bash
git add -A && git commit -m "chore: lint fixes"
```

---

## Self-Review

**Spec coverage:**
- Component 1 (delegate `is_instrument_listed`) → Task 1. ✅
- Component 2 (`add_instruments` gone-filter, gone-only not scheduled) → Task 2 (comment encodes the gone-only intent). ✅
- Component 3 (aggregator source-filter) → Task 4. ✅
- Component 4 (`trigger_fit` action + `ctx.trigger_fit()`) → Task 3. ✅
- Fail-open → relies on existing `is_instrument_listed` semantics (Tasks 2/4 use it). ✅
- Testing per component → Tasks 1–4 each include a TDD test. ✅
- Rollout → operational, post-implementation (not a code task); covered in the spec.

**Placeholder scan:** No TBD/TODO. Interface stubs use `...` intentionally (matches existing `IStrategyContext`/`IMarketManager` style). Test files note "adjust to real fixtures" only where the implementer must reconcile mock shapes with real `Target`/fixtures — the code to write is fully given.

**Type/signature consistency:** `is_instrument_listed(instrument) -> bool` consistent across Task 1 (context) and its use in Task 2/4. `trigger_fit(self) -> None` consistent across processing impl (Task 3 step 3), interface decl (step 4), and the action handler (step 5). `ActionResult(status="ok", data=...)` matches the existing builtin.py convention. `_drop_gone` reused from the 1.6.0 code (unchanged signature: `list[Instrument] -> list[Instrument]`).

**Cross-repo note:** Tasks 1–3 + 5(1,2) are qubx; Task 4 + 5(3) are quantkit. Independent test suites; Task 4 depends on Task 1 at *runtime* (ctx.is_instrument_listed) but not for its unit test (ctx mocked). Order: qubx first (publish 1.6.1), then quantkit.
