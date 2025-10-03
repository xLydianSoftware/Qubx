# Task 011: Add is_live Tag to All Emitted Metrics

## Status: ✅ Completed

## Overview
Added an `is_live` tag to all metrics emitted by the emitters module. The tag value is based on `ctx.is_simulation`: `is_live=false` when in simulation mode, `is_live=true` for live trading. This enables filtering in Grafana to show only live data.

## Key Design Decision
Replaced the `set_time_provider()` method with `set_context()` since the context provides both time AND `is_simulation` state. The emitter itself handles adding the `is_live` tag automatically - no other components need to know about it.

## Changes Made

### 1. Interface Update (`src/qubx/core/interfaces.py`)
- Replaced `set_time_provider(time_provider: ITimeProvider)` with `set_context(context: IStrategyContext)` in IMetricEmitter interface
- Updated docstring to reflect the new method purpose

### 2. BaseMetricEmitter (`src/qubx/emitters/base.py`)
- Replaced `_time_provider` attribute with `_context`
- Replaced `set_time_provider()` with `set_context(context: IStrategyContext)` method
- Updated `_merge_tags()` to automatically add `is_live` **boolean** tag based on `self._context.is_simulation`
- Updated `emit()` to use `self._context.time()` instead of `self._time_provider.time()`
- Updated `emit_strategy_stats()` to store context reference before processing
- Removed unused `ITimeProvider` import
- **Updated type hints**: Changed `dict[str, str]` to `dict[str, Any]` to support boolean `is_live` tag

### 3. CompositeMetricEmitter (`src/qubx/emitters/composite.py`)
- Added `set_context()` method that propagates context to all child emitters
- Ensures all emitters in the composite receive the context

### 4. Updated All Emitter Implementations
- `src/qubx/emitters/inmemory.py` - Updated type hints to `dict[str, Any]`, added `Any` import
- `src/qubx/emitters/indicator.py` - Updated type hints to `dict[str, Any]`, added `Any` import
- `src/qubx/emitters/questdb.py` - Updated type hints to `dict[str, Any]`, added `Any` import
- `src/qubx/emitters/csv.py` - Updated type hints to `dict[str, Any]`, added `Any` import
- `src/qubx/emitters/prometheus.py` - Updated type hints to `dict[str, Any]`, added `Any` import

### 5. Runner Call Sites
- `src/qubx/utils/runner/runner.py:309-310` - Removed old `set_time_provider()` call
- `src/qubx/utils/runner/runner.py:390-391` - Added `set_context(ctx)` after context creation
- `src/qubx/utils/runner/runner.py:714` - Changed warmup restoration to use `set_context(ctx)`
- `src/qubx/backtester/runner.py:481` - Changed to use `set_context(ctx)`

## Result
All emitted metrics now automatically include an `is_live` tag (as **boolean**, not string) based on the context's simulation state. This tag can be used for filtering in Grafana to show only live trading data.

## Tag Details
- **Type**: Boolean (`True`/`False`, not strings)
- **Value**: `is_live = not ctx.is_simulation`
  - `True` for live/paper trading (`ctx.is_simulation = False`)
  - `False` for backtests/simulations (`ctx.is_simulation = True`)

## Testing Notes
- The `is_live` tag is added automatically by `_merge_tags()` in BaseMetricEmitter
- All emitter implementations inherit this behavior from BaseMetricEmitter
- Type hints updated to `dict[str, Any]` to properly support boolean and other non-string tag values
- All 752 tests pass successfully
