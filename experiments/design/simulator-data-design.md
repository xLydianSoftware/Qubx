# Simulator Data Layer: Adopting New SimulatedDataIterator

## Status: IN PROGRESS

## Context

Old architecture used `DataReader` + `DataFetcher` (one SQL per symbol per read).
New architecture uses `IReader/IStorage` + `DataPump` + `RawSymbolBuffer` + `MemReader` (one batched read for all symbols).

The new `SimulatedDataIterator` is in `simulated_data.py` (replaced old `IterableSimulationData`).
Need to update consumers: `SimulatedDataProvider` and `SimulationRunner`.

---

## Design Principles

1. **Single SimulatedDataIterator per simulation** — created in `SimulationRunner`, shared across all data providers. It owns the slicer, iteration loop, and reader resolution.

2. **One SimulatedDataProvider per exchange** — created in the per-exchange loop in `SimulationRunner._create_backtest_context()`. Each wraps the shared `SimulatedDataIterator` and handles exchange-specific concerns (quotes, account notifications).

3. **Exchange list comes from SimulationSetup** — `setup.exchanges` defines what exchanges exist in the sim.
   - Open question: Can a strategy subscribe to an exchange not in the original config?

4. **SimulationDataConfig updated**:
   - `data_storage: IStorage` — main data storage
   - `customized_data_storages: dict[str, IStorage]` — overrides per subscription type

---

## Decided: SimulatedDataIterator owns IStorage, resolves readers lazily

**SimulatedDataIterator receives IStorage (+ custom storages) directly.**
It resolves IReader internally on demand from instrument data.

### Constructor
```python
def __init__(self, storage: IStorage, custom_types_storages: dict[str, IStorage] | None = None, ...):
    self._storage = storage
    self._custom_storages = dict(custom_types_storages or {})
    self._readers = {}   # cache: reader_key -> IReader (lazy)
```

### Reader resolution: `_get_or_create_reader(data_type, exchange, market_type)`
- Custom storage checked first (`_custom_storages.get(data_type)`), fallback to `_storage`
- Custom readers cached as `"{data_type}:{exchange}:{market_type}"`
- Main readers cached as `"{exchange}:{market_type}"` (shared across data types)

### Pump granularity: one per (subscription, exchange, market_type)
- Pump key: `"{access_key}.{exchange}:{market_type}"` e.g. `"ohlc.1h.BINANCE.UM:SWAP"`
- `add_instruments_for_subscription()` groups instruments by `(exchange, market_type)`
- Each group gets its own pump with its own IReader

### Slicer key: includes exchange scope
- Format: `"{requested_data_type}.{exchange}:{market_type}:{symbol}"`
- e.g. `"ohlc(1h).BINANCE.UM:SWAP:BTCUSDT"` — prevents cross-exchange collisions
- DataPump stores `_exchange` + `_market_type` and uses them in `_make_slicer_key()`

---

## Changes Summary

### `DataPump` (simulated_data.py)
- [x] Add `exchange: str` and `market_type: str` to constructor, stored as `_exchange`, `_market_type`
- [x] `_make_slicer_key()` now returns `f"{requested_data_type}.{exchange}:{market_type}:{symbol}"`
- [x] `__repr__` includes exchange scope

### `SimulatedDataIterator` (simulated_data.py) — formerly `IterableSimulationData`
- [x] Constructor takes `storage: IStorage` + `custom_types_storages: dict[str, IStorage] | None`
- [x] `_get_or_create_reader(data_type, exchange, market_type)` — lazy IReader resolution with caching
- [x] `_get_or_create_pump()` — takes exchange/market_type, pump key includes exchange scope
- [x] `add_instruments_for_subscription()` — groups by (exchange, market_type), per-group pump creation
- [x] `remove_instruments_from_subscription()` — groups by (exchange, market_type) to find pump; ALL iterates all pumps
- [x] `get_instruments_for_subscription()` — iterates all pumps matching access_key prefix
- [x] `peek_historical_data()` — constructs pump_key from instrument's exchange + market_type
- [x] `__iter__` — iterates all pumps (no change needed, already works)
- [x] `get_ohlc(instrument, timeframe, start, end)` — **moved from SimulatedDataProvider**, uses `_get_or_create_reader()` + `TypedRecords` transformer
- [x] `_process_bar_records(records, cut_time_ns, timeframe_ns)` — **moved from SimulatedDataProvider** (was `_convert_records_to_bars`), now works with `list[Bar]` instead of `list[TimestampedDict]`, reads indent from `self.emulation_time_indent_seconds`

### `SimulatedDataProvider` (data.py)
- [x] Dropped `_readers: dict[str, DataReader]` from `__init__` — no longer needed
- [x] Constructor takes `data_source: SimulatedDataIterator` (was `IterableSimulationData`)
- [x] `get_ohlc()` — now delegates to `self._data_source.get_ohlc(instrument, timeframe, start, end)`, uses `time_provider.time()` for current sim time
- [x] `_convert_records_to_bars()` — kept as fallback, reads indent from `self._data_source.emulation_time_indent_seconds` (single source of truth, no longer stored on provider)
- [x] `_open_close_time_indent_ns` removed from provider — indent owned by `SimulatedDataIterator`

### Tests (simulated_data_test.py)
- [x] `TestSimulatedDataIterator`: 7 tests updated to use `storage=CsvStorage(...)` / `storage=HandyStorage(...)`
- [x] `TestSimulatedDataProvider`: 2 tests for `get_ohlc()` via provider:
  - `test_get_ohlc_with_csv_storage` — verifies correct bar count, time range, indent cut at hour boundary (sim_time=12:00:00)
  - `test_get_ohlc_just_before_hour_boundary` — verifies 11:00 bar is fully closed at sim_time=11:59:59 (past effective close 11:59:55), 12:00 bar not visible
- [x] All 9 tests pass

### `SimulationRunner` (runner.py)
- [ ] Update `SimulatedDataIterator(...)` construction — pass `data_config.data_storage`
- [ ] Update `SimulatedDataProvider(...)` construction — drop old `readers` param
- [ ] Fix custom subscription handling (line ~166 references old `data_config.data_providers`)

### `SimulationDataConfig` (utils.py)
- [x] Fields: `data_storage: IStorage`, `customized_data_storages: dict[str, IStorage]`
- [ ] Fix `get_timeguarded_aux_reader()` — still references `self.aux_data_provider` (old field)

---

## Key Concept: `open_close_time_indent`

### Problem

The simulator is driven **only by market data updates** — there is no real wall-clock. The simulator's notion of "current time" is the timestamp of the last emitted market data update. Strategies typically use a **scheduler** to arm `on_event()` triggers at specific times (e.g. "every day at 23:59").

In **live trading** this works perfectly — the OS clock guarantees the trigger fires at the requested wall-clock time, and the latest market data is already available.

In **simulation** the situation breaks down. Consider:
- Base subscription: **1D OHLC bars**
- Scheduler armed: **trigger on_event at 23:59 daily**

What happens without the indent:
1. Day N's bar arrives with timestamp `00:00` of day N (bar open).
2. Day N+1's bar arrives with timestamp `00:00` of day N+1.
3. Simulator sees time jumped from day N to day N+1, checks scheduler — 23:59 has passed!
4. Simulator calls `on_event()` **first**, then emits the new bar update.
5. But inside `on_event()` the strategy sees the **previous day's data** as the latest — which is correct for the trigger time (23:59 of day N). ✅
6. **Problem**: the new bar's open arrives at exactly `00:00` — if the strategy armed at `00:00` it would collide. And for shorter timeframes, the timing gap between "last update" and "trigger time" can cause the scheduler to never fire or fire with stale data.

### Solution: Adjusted Time Indents

We shift emulated update timestamps slightly to create a gap that allows scheduler triggers to land correctly between the last bar's close update and the next bar's open update.

**The indent is applied to emulated quote timestamps produced from OHLC bars:**
- Last bar's **close** update is shifted **backward** (e.g. 23:55 instead of 00:00)
- Next bar's **open** update is shifted **forward** (e.g. 00:05 instead of 00:00)

This creates a time window where the scheduler trigger (e.g. 23:59) naturally falls **after** the close update (23:55) and **before** the next open update (00:05).

**Execution flow with indent (1D bars, trigger at 23:59):**
1. Bar N close emulated at **23:55** — strategy sees latest price ✅
2. Simulator advances to next bar, timestamp **00:05** of day N+1
3. Simulator checks scheduler: 23:59 has passed → calls `on_event()` with time=23:59
4. Inside `on_event()`, last market data is from 23:55 — correct and usable as close price ✅
5. Only then simulator emits the new bar's open update at 00:05

### Indent Values by Timeframe

Implementation: `_adjust_open_close_time_indent_secs()` in `src/qubx/backtester/utils.py:677`

```python
def _adjust_open_close_time_indent_secs(timeframe: pd.Timedelta, original_indent_secs: int) -> int:
    if timeframe >= pd.Timedelta("1d"):
        return max(original_indent_secs, 5 * 60)    # 5 min
    if timeframe >= pd.Timedelta("1min"):
        return max(original_indent_secs, 5)          # 5 sec
    if timeframe > pd.Timedelta("1s"):
        return max(original_indent_secs, 1)          # 1 sec
    return original_indent_secs                       # keep original
```

The indent scales with the base subscription timeframe:

| Base subscription | Indent | Rationale |
|-------------------|--------|-----------|
| `> 1 sec` (sub-minute) | **1 sec** | Tight enough for second-level triggers |
| `>= 1 min` (minute bars) | **5 sec** | Allows arming up to 1 sec before minute boundary |
| `>= 1 day` (daily bars) | **5 min** | Allows arming up to 1 min before daily close |

Note: `max(original_indent_secs, ...)` ensures we never shrink a user-provided indent — only enforce a minimum per timeframe.

### Usage in `get_ohlc()` / `_process_bar_records()`

The indent is used in `SimulatedDataIterator._process_bar_records()` (`src/qubx/backtester/simulated_data.py`) when the strategy requests historical OHLC bars via `ctx.ohlc()` → `SimulatedDataProvider.get_ohlc()` → `SimulatedDataIterator.get_ohlc()` during `on_event()`.

**Call chain:**
```
Strategy.on_event()
  → ctx.ohlc(instrument, "1h", 10)
    → SimulatedDataProvider.get_ohlc(instrument, "1h", nbarsback=10)
      → start = time_provider.time()          # current simulated time
      → end = start - nbarsback * Timedelta(timeframe)
      → SimulatedDataIterator.get_ohlc(instrument, "1h", start, end)
        → _get_or_create_reader()             # reuses cached IReader from pump path
        → reader.read(symbol, dtype, start, end)  # handle_start_stop sorts if reversed
        → TypedRecords transforms RawData → list[Bar]
        → _process_bar_records(bars, cut_time_ns, timeframe_ns)
```

The problem: when `on_event()` fires (e.g. at 23:59), the current bar's raw timestamp is `00:00` of that day — its close hasn't happened yet in simulation time. Without the indent, the method wouldn't know whether this bar is "complete" or still forming.

The solution — use the indent to compute the bar's effective close time:

```python
def _process_bar_records(self, records: list[Bar], cut_time_ns: int, timeframe_ns: int) -> list[Bar]:
    _open_close_time_indent_ns = int(self.emulation_time_indent_seconds * 1_000_000_000)
    ...
    for r in records:
        _b_ts_0 = r.time                                                     # bar open timestamp
        _b_ts_1 = _b_ts_0 + timeframe_ns - _open_close_time_indent_ns        # effective close time

        if _b_ts_0 <= cut_time_ns and cut_time_ns < _b_ts_1:
            break  # this bar is still "open" at current sim time — exclude it
        bars.append(r)
```

Example with 1D bars, indent = 5min, `on_event` at 23:59:
- Bar open: `00:00` → effective close: `00:00 + 24h - 5min = 23:55`
- `cut_time_ns` = 23:59 → `23:59 >= 23:55` → bar is considered **complete** ✅
- Next bar open: `00:00` next day → effective close: `23:55` next day
- `cut_time_ns` = 23:59 → `00:00 <= 23:59 < 23:55` → **still open**, break ✅

Example with 1H bars, indent = 5s, strategy at 11:59:59:
- Bar at 11:00 → effective close: `11:00 + 1h - 5s = 11:59:55`
- `cut_time_ns` = 11:59:59 → `11:59:59 >= 11:59:55` → bar is **complete** ✅
- Bar at 12:00 → not even in reader range (stop < 12:00) ✅

This ensures `get_ohlc()` returns only fully "closed" bars relative to the simulated trigger time, preventing look-ahead into a bar that hasn't completed yet.

**Indent ownership**: `emulation_time_indent_seconds` is stored on `SimulatedDataIterator` (single source of truth). `SimulatedDataProvider` reads it via `self._data_source.emulation_time_indent_seconds` — no longer stores its own copy.

### Why This Matters for the Refactor

Since different data subscriptions can use different base timeframes, the indent must be resolved per subscription type. When `SimulatedDataIterator` creates emulated updates from OHLC bars, the correct indent must be applied based on the bar timeframe. This is part of the data transformation pipeline that produces emulated quotes from raw OHLC data.

The `get_ohlc()` → `_process_bar_records()` path now lives entirely on `SimulatedDataIterator`, reusing the same `_get_or_create_reader()` and cached IReader instances as the streaming pump path. This eliminates the old dual-reader problem where `SimulatedDataProvider` had its own `_readers` dict separate from the iteration pipeline.

---

## Resolved Questions

1. **get_ohlc() — still needed?** ✅ **Yes, kept.** Moved from `SimulatedDataProvider` to `SimulatedDataIterator`. Still used by `ctx.ohlc()` for lookback. Now uses the same `_get_or_create_reader()` + `TypedRecords` pipeline as the streaming path. Old strategies that rely on `ctx.ohlc()` continue to work; new strategies can use streaming OHLCV series instead.
2. **Can strategy subscribe to exchange not in config?** ✅ **Fine as-is.** `_get_or_create_reader()` calls `storage.get_reader(exchange, market_type)` — will raise if storage doesn't have that exchange. This is the expected behavior.
3. **Where does indent live after refactor?** ✅ **On `SimulatedDataIterator`.** Single source of truth via `emulation_time_indent_seconds` property. Provider reads it via `self._data_source.emulation_time_indent_seconds`.

## Open Questions

1. **`SimulatedDataProvider._convert_records_to_bars()` — remove?** The old method is still present in `data.py` but the main `get_ohlc()` path now goes through `SimulatedDataIterator._process_bar_records()`. Can be removed once we're confident the new path covers all cases.
2. **`SimulationRunner` wiring** — still needs update to construct `SimulatedDataIterator(storage=...)` and pass to `SimulatedDataProvider(data_source=...)`. This is the main remaining integration work.
