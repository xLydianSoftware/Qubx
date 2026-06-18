# Blacklist Position Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee a bot never holds an open position in a blacklisted instrument, by force-closing held blacklisted positions on every Apply and every fit, and blocking any order that would open/increase blacklisted exposure.

**Architecture:** Two complementary mechanisms in Qubx core. (#1) `InstrumentServiceManager` gains a full-set force-close helper used by `run_cycle` (Apply) and a new `enforce_at_fit()` (fit-time, callback-free) that replaces `refresh_only()`. (#2) `TradingManager.trade`/`trade_async` gain a reduce-only clamp for blacklisted instruments. Both are no-ops when no instrument service is configured.

**Tech Stack:** Python 3.12, qubx core mixins, pytest. Run tests with `uv run pytest` (env already synced with `--extra connectors`).

**Spec:** `docs/superpowers/specs/2026-06-18-blacklist-position-enforcement-design.md`

**Worktree:** `~/devs/Qubx/.worktrees/blacklist-enforcement`, branch `fix/blacklist-position-enforcement` (off `main`, post-1.9.1). All `cd` paths below are relative to this worktree root.

---

### Task 1: #1 — Full-set force-close in `run_cycle`

Replace the delta-based (`diff.blacklisted_added`) force-close with a sweep over **all** held blacklisted positions, so an already-blacklisted holding (the WLFI case) is closed even when it isn't in the current change delta.

**Files:**
- Modify: `src/qubx/core/mixins/instrument_service.py` (`run_cycle`, ~lines 36-57; add helper)
- Test: `tests/qubx/core/instrument_service_manager_test.py`

- [ ] **Step 1: Write the failing regression test** (WLFI: held + blacklisted but NOT in `diff.blacklisted_added`)

Append to `tests/qubx/core/instrument_service_manager_test.py`:

```python
def test_run_cycle_force_closes_all_held_blacklisted_not_just_delta():
    # WLFI regression: an already-blacklisted holding (absent from the change delta)
    # must still be force-closed.
    ondo, wlfi, btc = MagicMock(), MagicMock(), MagicMock()
    pos = MagicMock(); pos.quantity = -34767.0
    btc_pos = MagicMock(); btc_pos.quantity = 1.0
    svc = MagicMock()
    # Only ONDO is newly-added this cycle; WLFI was blacklisted earlier.
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[ondo], blacklisted_removed=[])
    svc.is_blacklisted.side_effect = lambda i: i in (ondo, wlfi)
    m, ctx = _mgr(svc, instruments=[wlfi, btc], positions={wlfi: pos, btc: btc_pos})
    summary = m.run_cycle()
    ctx.remove_instruments.assert_called_once_with([wlfi], if_has_position_then="close")
    assert summary["force_closed"] == 1
    assert summary["force_closed_instruments"] == [str(wlfi)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/qubx/core/instrument_service_manager_test.py::test_run_cycle_force_closes_all_held_blacklisted_not_just_delta -q`
Expected: FAIL — current code only closes `diff.blacklisted_added` (ONDO, not held) so `remove_instruments` is never called.

- [ ] **Step 3: Add the helper and rewrite `run_cycle` to use it**

In `src/qubx/core/mixins/instrument_service.py`, replace the body of `run_cycle` (the `positions = ...` / `still_held = ...` / `if still_held:` block and the `return {...}`) and add a helper just above `refresh_only`:

```python
    def _force_close_held_blacklisted(self) -> list[Instrument]:
        """Force-close every currently-held position whose instrument is blacklisted.
        Idempotent and full-set (not the change delta), so already-blacklisted holdings
        are closed too. Reduce-only by construction (closes to 0), so it is allowed by the
        trade-layer blacklist gate. No-op for the Null service (is_blacklisted is False)."""
        positions = self._context.get_positions()
        held = [i for i, p in positions.items() if p.quantity != 0 and self._service.is_blacklisted(i)]
        if held:
            self._context.remove_instruments(held, if_has_position_then="close")
        return held
```

And `run_cycle` becomes (keep the refresh + callback block unchanged; swap the close block):

```python
    def run_cycle(self, _ctx: "IStrategyContext | None" = None) -> dict:
        """Refresh the blacklist, fire change callbacks, then force-close ALL held
        blacklisted instruments (full set, not just the change delta). Shared by the
        control action and the startup one-shot. Runs on the strategy thread."""
        diff = self._service.refresh(self._context.instruments)
        if diff.blacklisted_added or diff.blacklisted_removed:
            for cb in self._callbacks:
                try:
                    cb(self._context, diff.blacklisted_added, diff.blacklisted_removed)
                except Exception as e:
                    logger.error(f"[InstrumentService] :: change callback error: {e}")
        closed = self._force_close_held_blacklisted()
        return {
            "blacklisted_added": len(diff.blacklisted_added),
            "blacklisted_removed": len(diff.blacklisted_removed),
            "force_closed": len(closed),
            "force_closed_instruments": [str(i) for i in closed],
        }
```

- [ ] **Step 4: Run the new test + the whole file to verify no regressions**

Run: `uv run pytest tests/qubx/core/instrument_service_manager_test.py -q`
Expected: PASS. The existing `test_run_cycle_fires_callbacks_before_force_close` still passes (its `svc.is_blacklisted` MagicMock is truthy, so the held `btc` is closed in the same order); `test_run_cycle_no_backstop_when_not_held` and `test_run_cycle_empty_diff_is_noop` pass (empty positions → `held == []`).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/mixins/instrument_service.py tests/qubx/core/instrument_service_manager_test.py
git commit -m "fix(core): force-close all held blacklisted instruments in run_cycle (not just delta)"
```

---

### Task 2: #1 — `enforce_at_fit()` replaces `refresh_only()`

At fit time the blacklist must not only refresh the cache but also close held blacklisted positions (the orphan case `set_universe` misses). Replace the cache-only `refresh_only()` with `enforce_at_fit()` (refresh + full-set force-close, **no** change callbacks — the fit is already running).

**Files:**
- Modify: `src/qubx/core/mixins/instrument_service.py` (replace `refresh_only`)
- Modify: `src/qubx/core/interfaces.py:1364` (`IInstrumentServiceManager.refresh_only` → `enforce_at_fit`)
- Modify: `src/qubx/core/mixins/processing.py:621` (call site in `__invoke_on_fit`)
- Test: `tests/qubx/core/instrument_service_manager_test.py`, `tests/qubx/core/mixins/processing_fit_refresh_test.py`

- [ ] **Step 1: Write the failing tests** (mixin behavior + processing call site)

Replace the two `refresh_only` tests in `tests/qubx/core/instrument_service_manager_test.py` (`test_refresh_only_refreshes_cache_without_callbacks_or_force_close` and `test_refresh_only_noop_with_null_service`) with:

```python
def test_enforce_at_fit_refreshes_and_force_closes_without_callbacks():
    ondo = MagicMock()
    pos = MagicMock(); pos.quantity = -891.4
    svc = MagicMock()
    svc.refresh.return_value = InstrumentServiceDiff(blacklisted_added=[], blacklisted_removed=[])
    svc.is_blacklisted.side_effect = lambda i: i is ondo
    calls = []
    m, ctx = _mgr(svc, instruments=[ondo], positions={ondo: pos},
                  callbacks=[lambda c, a, r: calls.append("cb")])
    assert m.enforce_at_fit() is None
    svc.refresh.assert_called_once_with([ondo])
    assert calls == []  # no change callbacks at fit time
    ctx.remove_instruments.assert_called_once_with([ondo], if_has_position_then="close")


def test_enforce_at_fit_noop_with_null_service():
    m, ctx = _mgr(NullInstrumentService())
    assert m.enforce_at_fit() is None
    ctx.remove_instruments.assert_not_called()
```

In `tests/qubx/core/mixins/processing_fit_refresh_test.py`, rename the mock references in both tests from `refresh_only` to `enforce_at_fit`:
- `test_invoke_on_fit_refreshes_instrument_service_before_on_fit`: replace every `refresh_only` token with `enforce_at_fit` (the `attach_mock(..., "enforce_at_fit")`, the `assert_called_once_with()`, and the `called.index("enforce_at_fit") < called.index("on_fit")`).
- `test_invoke_on_fit_marks_fit_called_even_if_refresh_raises`: replace `context._instrument_service_manager.refresh_only.side_effect` with `...enforce_at_fit.side_effect`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/qubx/core/instrument_service_manager_test.py -k enforce_at_fit tests/qubx/core/mixins/processing_fit_refresh_test.py -q`
Expected: FAIL — `enforce_at_fit` does not exist yet (AttributeError on the mixin / call site still uses `refresh_only`).

- [ ] **Step 3: Implement `enforce_at_fit`, update interface + call site**

In `src/qubx/core/mixins/instrument_service.py`, replace the entire `refresh_only` method with:

```python
    def enforce_at_fit(self) -> None:
        """Fit-time enforcement: refresh the cached blacklist AND force-close any held
        blacklisted positions, WITHOUT firing change callbacks (the fit is already
        running; firing re-fit callbacks here would loop). Called immediately before
        `on_fit` so the rebalance selects over current data and never holds a blacklisted
        instrument. No-op for the Null service."""
        self._service.refresh(self._context.instruments)
        self._force_close_held_blacklisted()
```

In `src/qubx/core/interfaces.py`, replace the `refresh_only` declaration (the `def refresh_only(self) -> None:` block at ~line 1364 and its docstring) with:

```python
    def enforce_at_fit(self) -> None:
        """Fit-time enforcement: refresh the cached blacklist and force-close any held
        blacklisted positions, without firing change callbacks. No-op for the Null service."""
        ...
```

In `src/qubx/core/mixins/processing.py` line 621, change:

```python
                self._context._instrument_service_manager.refresh_only()
```
to:
```python
                self._context._instrument_service_manager.enforce_at_fit()
```

Also update the adjacent comment (lines ~616-618) referencing "refresh the blacklist cache (cache-only...)" to read "refresh the blacklist cache and force-close held blacklisted positions" so the comment matches the new behavior.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/qubx/core/instrument_service_manager_test.py tests/qubx/core/mixins/processing_fit_refresh_test.py -q`
Expected: PASS.

- [ ] **Step 5: Grep to confirm no stale `refresh_only` references remain**

Run: `grep -rn 'refresh_only' src/ tests/`
Expected: no matches. If any remain, update them to `enforce_at_fit`.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/mixins/instrument_service.py src/qubx/core/interfaces.py src/qubx/core/mixins/processing.py tests/qubx/core/instrument_service_manager_test.py tests/qubx/core/mixins/processing_fit_refresh_test.py
git commit -m "fix(core): enforce blacklist (refresh + force-close) at fit time, replacing refresh_only"
```

---

### Task 3: #2 — Reduce-only clamp for blacklisted instruments in the trade layer

Block any order that opens or increases exposure on a blacklisted instrument; allow reducing orders; clamp a flip-through-zero to an exact close. Applied at `trade()` and `trade_async()` — the universal order choke.

**Files:**
- Modify: `src/qubx/core/mixins/trading.py` (add `_blacklist_clamp`; call it in `trade` ~line 95 and `trade_async` ~line 142)
- Test: `tests/qubx/core/mixins/trading_test.py`

- [ ] **Step 1: Write the failing unit tests for the clamp logic**

First, give the existing `MockStrategyContext` a default `is_blacklisted` (so every existing `trade()` test takes the no-op path once the gate is added). In `tests/qubx/core/mixins/trading_test.py`, add to the `MockStrategyContext` class body:

```python
    def is_blacklisted(self, instrument):
        return False
```

Then append the clamp unit tests (the `trading_manager`, `mock_account`, `strategy_context` fixtures already exist):

```python
class TestBlacklistReduceOnly:
    def _pos(self, qty):
        p = Mock(spec=Position); p.quantity = qty
        return p

    def test_clamp_blocks_opening_from_flat(self, trading_manager, mock_account, strategy_context):
        strategy_context.is_blacklisted = lambda inst: True
        mock_account.get_position.return_value = self._pos(0.0)
        assert trading_manager._blacklist_clamp(Mock(symbol="ONDOUSDT"), -891.4) == 0.0

    def test_clamp_blocks_increasing_short(self, trading_manager, mock_account, strategy_context):
        strategy_context.is_blacklisted = lambda inst: True
        mock_account.get_position.return_value = self._pos(-500.0)
        assert trading_manager._blacklist_clamp(Mock(symbol="ONDOUSDT"), -100.0) == 0.0  # -500 -> -600

    def test_clamp_allows_reducing_short(self, trading_manager, mock_account, strategy_context):
        strategy_context.is_blacklisted = lambda inst: True
        mock_account.get_position.return_value = self._pos(-500.0)
        assert trading_manager._blacklist_clamp(Mock(symbol="ONDOUSDT"), 200.0) == 200.0  # -500 -> -300

    def test_clamp_flip_through_zero_is_clamped_to_close(self, trading_manager, mock_account, strategy_context):
        strategy_context.is_blacklisted = lambda inst: True
        mock_account.get_position.return_value = self._pos(-500.0)
        # request would go -500 -> +300 (flip); clamp to exact close (+500)
        assert trading_manager._blacklist_clamp(Mock(symbol="ONDOUSDT"), 800.0) == 500.0

    def test_clamp_noop_when_not_blacklisted(self, trading_manager, mock_account, strategy_context):
        strategy_context.is_blacklisted = lambda inst: False
        assert trading_manager._blacklist_clamp(Mock(symbol="BTCUSDT"), -891.4) == -891.4
        mock_account.get_position.assert_not_called()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py::TestBlacklistReduceOnly -q`
Expected: FAIL — `_blacklist_clamp` does not exist (AttributeError).

- [ ] **Step 3: Implement `_blacklist_clamp` and gate both entry points**

In `src/qubx/core/mixins/trading.py`, add the helper method to `TradingManager` (place it just above `def trade(`):

```python
    def _blacklist_clamp(self, instrument: Instrument, amount: float) -> float:
        """Reduce-only for blacklisted instruments: an order may only move the position
        toward zero, never open or increase exposure. No-op when not blacklisted (always,
        without an instrument service). Returns the (possibly clamped) amount; 0.0 means
        'do not send'. A flip through zero is clamped to an exact close."""
        if not self._context.is_blacklisted(instrument):
            return amount
        current = self._account.get_position(instrument).quantity
        new = current + amount
        if current == 0 or abs(new) > abs(current):
            logger.warning(
                f"[Blacklist] :: blocked order increasing exposure on {instrument.symbol} "
                f"(current={current}, amount={amount})"
            )
            return 0.0
        if (current > 0) != (new > 0) and new != 0:  # would flip through zero
            logger.warning(
                f"[Blacklist] :: clamping {instrument.symbol} order to close "
                f"(current={current}, requested_new={new})"
            )
            return -current
        return amount
```

Then at the **first line** of the body of both `trade` (after the signature, before `size_adj = self._adjust_size(...)`) and `trade_async`, insert:

```python
        amount = self._blacklist_clamp(instrument, amount)
        if amount == 0.0:
            return None
```

(`trade` returns `Order | None`; `trade_async` returns `str | None` — `None` is valid for both.)

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py::TestBlacklistReduceOnly -q`
Expected: PASS.

- [ ] **Step 5: Write + run an integration test that `trade()` sends no order when blocked**

Append to `tests/qubx/core/mixins/trading_test.py`:

```python
def test_trade_sends_no_order_when_blacklisted_and_opening(trading_manager, mock_account, mock_broker, strategy_context):
    strategy_context.is_blacklisted = lambda inst: True
    flat = Mock(spec=Position); flat.quantity = 0.0
    mock_account.get_position.return_value = flat
    inst = Mock(symbol="ONDOUSDT"); inst.exchange = "BINANCE.UM"
    assert trading_manager.trade(inst, -891.4) is None
    mock_broker.send_order.assert_not_called()
```

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py::test_trade_sends_no_order_when_blacklisted_and_opening -q`
Expected: PASS (the early `return None` fires before any broker call).

- [ ] **Step 6: Run the whole trading test file for regressions**

Run: `uv run pytest tests/qubx/core/mixins/trading_test.py -q`
Expected: PASS. Existing `trade()`/`trade_async()` tests take the no-op path via the `MockStrategyContext.is_blacklisted` default added in Step 1. If any test fails with `AttributeError: is_blacklisted`, confirm that default was added to the class.

- [ ] **Step 7: Commit**

```bash
git add src/qubx/core/mixins/trading.py tests/qubx/core/mixins/trading_test.py
git commit -m "fix(core): reduce-only trade gate for blacklisted instruments"
```

---

### Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the affected suites together**

Run: `uv run pytest tests/qubx/core/ tests/qubx/control/ -q`
Expected: PASS (all). These cover instrument-service, processing, trading, and control-action behavior.

- [ ] **Step 2: Confirm no `refresh_only` remnants and no broken interface impls**

Run: `grep -rn 'refresh_only' src/ tests/ ; uv run python -c "from qubx.core.context import StrategyContext; print('imports OK')"`
Expected: no `refresh_only` matches; `imports OK` printed (confirms `IInstrumentServiceManager` is still satisfiable after the interface rename).

- [ ] **Step 3: Final commit if any fixups were needed**

```bash
git add -A && git commit -m "test: verify blacklist enforcement across instrument-service, processing, trading" || echo "nothing to commit"
```
