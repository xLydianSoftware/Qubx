---
name: qubx-indicators
description: Full cycle of implementing and testing streaming technical indicators in the Qubx quantitative trading framework.
---
# Implementing Streaming Indicators in Qubx

This skills file documents best practices and patterns for implementing technical indicators in the Qubx quantitative trading framework. 

## Table of Contents

1. [Overview](#overview)
2. [Architecture & File Locations](#architecture--file-locations)
3. [Core Implementation Patterns](#core-implementation-patterns)
4. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
5. [Testing Strategy](#testing-strategy)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
7. [Real-World Examples](#real-world-examples)

---

## Overview

### What are Streaming Indicators?

Streaming indicators in Qubx process data incrementally as it arrives, rather than requiring the full dataset upfront. This makes them suitable for:
- **Live trading**: Real-time calculation as new bars arrive
- **Backtesting**: Efficient simulation of historical data
- **Memory efficiency**: O(1) memory usage regardless of history length

### Key Characteristics

- **Cython Implementation**: Written in `.pyx` files for performance
- **Incremental Calculation**: Each new data point updates the indicator in O(1) time
- **Bar Update Handling**: Must correctly handle both new bars and updates to current bar
- **Pandas Compatibility**: Results should match pandas reference implementations

---

## Architecture & File Locations

### Core Files

1. **`src/qubx/ta/indicators.pyx`**
   - Main implementation file (Cython)
   - Contains all indicator classes and helper functions
   - Line count: ~1600+ lines

2. **`src/qubx/pandaz/ta.py`**
   - Reference pandas implementations
   - Used as specification for streaming versions
   - Useful for understanding algorithm logic

3. **`tests/qubx/ta/indicators_test.py`**
   - Test suite for all indicators
   - Each indicator has a corresponding test method
   - Tests compare streaming vs pandas results

4. **`src/qubx/core/series.pyx`**
   - Base `TimeSeries` class
   - Indicator base classes: `Indicator`, `IndicatorOHLC`

### Build System

```bash
# - rebuild after changes to .pyx files
just build

# - run indicator tests
just test tests/qubx/ta/indicators_test.py

# - run specific test
poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_macd -v
```

---

## Core Implementation Patterns

### Pattern 1: Simple Indicator with Internal Series

**Use when**: Your indicator needs to transform input before applying calculations.

**Example**: ATR (Average True Range)

```python
cdef class Atr(IndicatorOHLC):

    def __init__(self, str name, OHLCV series, int period, str smoother, short percentage):
        self.percentage = percentage
        # - create internal series for true range values
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        # - apply moving average to the internal series
        self.ma = smooth(self.tr, smoother, period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        if len(self.series) <= 1:
            return np.nan

        # - calculate true range
        cdef double c1 = self.series[1].close
        cdef double h_l = abs(bar.high - bar.low)
        cdef double h_pc = abs(bar.high - c1)
        cdef double l_pc = abs(bar.low - c1)

        # - update internal series first
        self.tr.update(time, max(h_l, h_pc, l_pc))

        # - return smoothed value
        return (100 * self.ma[0] / c1) if self.percentage else self.ma[0]
```

**Key Points**:
- Create internal `TimeSeries` to hold intermediate values
- Update internal series in `calculate()` method
- Attached indicators (like `smooth()`) automatically recalculate

### Pattern 2: Composite Indicators (Most Important!)

**Use when**: Your indicator depends on other indicators (like MACD = fast_ma - slow_ma).

**Critical Rule**: When building composite indicators, NEVER attach dependent indicators directly to the input series. Instead, create an internal series and update it first.

**Why**: When data is already loaded (not streaming), calculation order isn't guaranteed if indicators are attached to the main input series.

**Example**: MACD (correct implementation)

```python
cdef class Macd(Indicator):

    def __init__(self, str name, TimeSeries series, fast=12, slow=26, signal=9,
                 method="ema", signal_method="ema"):
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self.method = method
        self.signal_method = signal_method

        # - CRITICAL: create internal copy of input series
        self.input_series = TimeSeries("input", series.timeframe, series.max_series_length)

        # - attach fast and slow MAs to the INTERNAL series, not the input series!
        self.fast_ma = smooth(self.input_series, method, fast)
        self.slow_ma = smooth(self.input_series, method, slow)

        # - create internal series for MACD line
        self.macd_line_series = TimeSeries("macd_line", series.timeframe,
                                           series.max_series_length)

        # - create signal line (smoothed MACD line)
        self.signal_line = smooth(self.macd_line_series, signal_method, signal)

        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double fast_value, slow_value, macd_value

        # - STEP 1: update internal input series FIRST
        self.input_series.update(time, value)

        # - STEP 2: now safe to access dependent indicators
        fast_value = self.fast_ma[0] if len(self.fast_ma) > 0 else np.nan
        slow_value = self.slow_ma[0] if len(self.slow_ma) > 0 else np.nan

        # - STEP 3: calculate composite value
        if np.isnan(fast_value) or np.isnan(slow_value):
            macd_value = np.nan
        else:
            macd_value = fast_value - slow_value

        # - STEP 4: update intermediate series
        self.macd_line_series.update(time, macd_value)

        # - STEP 5: return final indicator value
        return self.signal_line[0] if len(self.signal_line) > 0 else np.nan
```

**Wrong Implementation** (will return constant values):
```python
# - DON'T DO THIS!
def __init__(self, str name, TimeSeries series, ...):
    # - attaching directly to input series doesn't guarantee calculation order
    self.fast_ma = smooth(series, method, fast)  # ❌ Wrong!
    self.slow_ma = smooth(series, method, slow)  # ❌ Wrong!
    super().__init__(name, series)

cpdef double calculate(self, long long time, double value, short new_item_started):
    # - accessing self.fast_ma[0] may not reflect the current value
    fast_value = self.fast_ma[0]  # ❌ May return stale data!
    slow_value = self.slow_ma[0]  # ❌ May return stale data!
    return fast_value - slow_value
```

### Pattern 3: Indicators with State (Bar Updates)

**Use when**: Your indicator needs to maintain state that must be restored when bars update.

**Example**: CUSUM Filter (state management portion)

```python
cdef class CusumFilter(Indicator):

    def __init__(self, str name, TimeSeries series, TimeSeries target):
        # - state variables
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.prev_value = np.nan

        # - cached state (for bar updates)
        self.saved_s_pos = 0.0
        self.saved_s_neg = 0.0
        self.saved_prev_value = np.nan

        # - for cross-timeframe access, use SeriesCachedValue (see Pattern 4)
        self.target_cache = SeriesCachedValue(target)

        super().__init__(name, series)

    cdef void _store(self):
        """Store state when new bar starts"""
        self.saved_s_pos = self.s_pos
        self.saved_s_neg = self.s_neg
        self.saved_prev_value = self.prev_value

    cdef void _restore(self):
        """Restore state when bar is updated (not new)"""
        self.s_pos = self.saved_s_pos
        self.s_neg = self.saved_s_neg
        self.prev_value = self.saved_prev_value

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - handle first value
        if np.isnan(self.prev_value):
            self.prev_value = value
            self._store()
            return 0.0

        # - restore state if updating existing bar
        if not new_item_started:
            self._restore()

        # - perform calculations...
        diff = value - self.prev_value
        self.s_pos = max(0.0, self.s_pos + diff)
        self.s_neg = min(0.0, self.s_neg + diff)
        # - (more calculation logic here)

        # - store state for next bar
        if new_item_started:
            self.prev_value = value
            self._store()

        return result
```

**Key Points**:
- Use `_store()` and `_restore()` methods for state management
- Check `new_item_started` flag to determine bar state
- Always restore state before recalculating on bar updates
- State management is separate from cross-timeframe caching (see Pattern 4)

### Pattern 4: Cross-Timeframe Access with SeriesCachedValue

**Use when**: Your indicator needs to lookup values from another series, especially from a **higher timeframe** (e.g., using daily volatility in a 1-hour indicator).

**Problem**: `self.target.times.lookup_idx(time, 'ffill')` is expensive when called repeatedly. Manual caching is verbose and error-prone.

**Solution**: Use the `SeriesCachedValue` helper class which handles period-based caching automatically.

**Why SeriesCachedValue?**
- Encapsulates caching logic in a reusable component
- Reduces code by ~30 lines per indicator
- Handles edge cases (empty series, NaN values)
- Uses period-based caching (only lookups when period changes)

**Example**: Getting volatility from daily series in hourly indicator

```python
from qubx.core.series cimport SeriesCachedValue

cdef class MyIndicator(Indicator):

    def __init__(self, str name, TimeSeries series, TimeSeries daily_volatility):
        # - create cached accessor for the higher timeframe series
        self.vol_cache = SeriesCachedValue(daily_volatility)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - get volatility value with automatic caching
        # - SeriesCachedValue will only do lookup when the period changes
        cdef double vol = self.vol_cache.value(time)

        if np.isnan(vol):
            return np.nan

        # - use the volatility value in calculations
        threshold = vol * value
        # - continue with calculation...
        return result
```

**How SeriesCachedValue works internally**:
- Calculates period start time: `floor_t64(time, series.timeframe)`
- Caches result for the entire period
- Only performs expensive `lookup_idx()` when period changes
- Returns `np.nan` if series is empty or lookup fails

**Performance Impact**: This optimization can reduce execution time by 10-100x for indicators that reference higher timeframe data.

**Before refactoring** (manual caching - ~30 lines):
```python
def __init__(self, ...):
    self.target = target
    self.cached_target_value = np.nan
    self.cached_target_time = -1
    self.cached_target_idx = -1

cpdef double calculate(self, long long time, double value, short new_item_started):
    cdef long long target_period_start = floor_t64(time, self.target.timeframe)

    if target_period_start != self.cached_target_time:
        idx = self.target.times.lookup_idx(time, 'ffill')
        if idx >= 0:
            self.cached_target_value = self.target.values.values[idx]
            self.cached_target_idx = idx
        else:
            self.cached_target_value = np.nan
            self.cached_target_idx = -1
        self.cached_target_time = target_period_start

    target_value = self.cached_target_value
```

**After refactoring** (SeriesCachedValue - 2 lines):
```python
def __init__(self, ...):
    self.target_cache = SeriesCachedValue(target)

cpdef double calculate(self, long long time, double value, short new_item_started):
    target_value = self.target_cache.value(time)
```

**Required imports and declarations**:
```python
# - in .pyx file
from qubx.core.series cimport SeriesCachedValue

# - in .pxd file
from qubx.core.series cimport SeriesCachedValue

cdef class MyIndicator(Indicator):
    cdef SeriesCachedValue target_cache  # - cached accessor
```

### Pattern 5: Helper Function Convention

Every indicator class should have a corresponding helper function:

```python
def macd(series: TimeSeries, fast=12, slow=26, signal=9,
         method="ema", signal_method="ema"):
    """
    Moving average convergence divergence (MACD) indicator.

    :param series: input data
    :param fast: fast MA period
    :param slow: slow MA period
    :param signal: signal MA period
    :param method: moving averaging method (sma, ema, tema, dema, kama)
    :param signal_method: method for averaging signal line
    :return: macd indicator
    """
    return Macd.wrap(series, fast, slow, signal, method, signal_method) # type: ignore
```

**Key Points**:
- Use `ClassName.wrap()` to create and register the indicator
- Provide comprehensive docstring
- Include parameter descriptions
- Add `# type: ignore` comment to suppress type checker warnings

---

## Common Pitfalls & Solutions

### Pitfall 1: Test Comparison Syntax Error

**Problem**: Incorrect comparison between pandas Series and streaming indicator.

```python
# - ❌ Wrong: r1 is already a pandas Series, can't call .pd() on it
diff_stream = abs(r1.pd() - r0).dropna()
```

**Solution**:
```python
# - ✅ Correct: convert streaming indicator to pandas, then compare
diff_stream = abs(r1 - r0.pd()).dropna()
```

**Rule**:
- Streaming indicators have `.pd()` method to convert to pandas Series
- Pandas Series don't have `.pd()` method
- Always convert streaming → pandas for comparison

### Pitfall 2: Calculation Order Issues

**Problem**: Accessing dependent indicators without ensuring they're calculated first.

**Symptom**: Indicator returns constant values or incorrect results after initial values.

**Solution**: Use the internal series pattern (Pattern 2 above).

**Debug Approach**:
```python
# - add debug output in calculate() to see what values are being used
print(f"time={time}, fast={fast_value}, slow={slow_value}, result={macd_value}")
```

### Pitfall 3: Forgetting to Handle NaN Values

**Problem**: Not checking for NaN in intermediate calculations.

```python
# - ❌ Wrong: may cause division by zero or invalid operations
return smooth_u / (smooth_u + smooth_d)
```

**Solution**:
```python
# - ✅ Correct: check for NaN and handle edge cases
if np.isnan(smooth_u) or np.isnan(smooth_d):
    return np.nan

# - avoid division by zero
if smooth_u + smooth_d == 0:
    return 50.0  # - neutral value for RSI

# - safe calculation
return 100.0 * smooth_u / (smooth_u + smooth_d)
```

### Pitfall 4: Not Importing Required Functions

**Problem**: Using functions that aren't imported in the Cython file.

```python
# - ❌ Will fail: floor_t64 not imported
target_period_start = floor_t64(time, self.target.timeframe)
```

**Solution**: Check imports at top of file:
```python
from qubx.utils.time cimport floor_t64
```

### Pitfall 5: Incorrect cdef Types

**Problem**: Using wrong Cython types causes compilation errors or performance issues.

```python
# - ❌ Wrong: int can't store nanosecond timestamps
cdef int time_value = time

# - ✅ Correct: use long long for timestamps
cdef long long time_value = time

# - ✅ Correct: use double for price values
cdef double price = value

# - ✅ Correct: use short for boolean flags
cdef short is_new = new_item_started
```

---

## Testing Strategy

### Test Structure

```python
def test_macd(self):
    # - STEP 1: load test data using Storage
    r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
    c1h = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

    # - STEP 2: calculate indicator on streaming data
    r0 = macd(c1h.close, 12, 26, 9, "sma", "sma")

    # - STEP 3: calculate reference using pandas
    r1 = pta.macd(c1h.close.pd(), 12, 26, 9, "sma", "sma")

    # - STEP 4: compare results
    diff_stream = abs(r1 - r0.pd()).dropna()
    assert diff_stream.sum() < 1e-6, f"macd differs from pandas: sum diff = {diff_stream.sum()}"
```

### Test Data Source

**Old Method** (deprecated):
```python
# - ❌ Don't use loader anymore
from qubx.data.loader import load_ohlcv
c1h = load_ohlcv("BINANCE.UM", "BTCUSDT", "1h", "2023-06-01", "2023-08-01")
```

**New Method** (Storage approach):
```python
# - ✅ Use Storage API
from qubx.data.storage import StorageRegistry

r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
c1h = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()
```

### Pandas Reference Import

```python
# - import pandas ta module
import qubx.pandaz.ta as pta

# - call corresponding pandas function
pandas_result = pta.macd(data.pd(), fast, slow, signal, method, signal_method)
```

### Acceptable Error Threshold

```python
# - for most indicators, sum of absolute differences should be < 1e-6
assert diff_stream.sum() < 1e-6

# - for some indicators with more floating point operations, may need < 1e-4
assert diff_stream.sum() < 1e-4
```

### Cross-Timeframe Testing (1h → 4h)

**Purpose**: Verify indicators correctly handle partial bar updates when lower timeframe data builds higher timeframe bars.

**Critical Pattern**: Indicators with state MUST implement store/restore pattern to handle this correctly.

#### The Problem

When feeding 1h data to build 4h bars:
- Each 4h bar receives 4 updates (one per hour)
- The indicator's `calculate()` is called 4 times per 4h bar
- Without store/restore, internal state gets corrupted by intermediate updates
- Results diverge from batch calculations on final 4h bars

**Symptom**: Streaming indicator returns different values than pandas on the same final data.

#### The Solution: Store/Restore Pattern

**For OHLC indicators with state** (like PSAR, SuperTrend):

```python
cdef class MyIndicator(IndicatorOHLC):
    # - working state variables (updated during calculation)
    cdef double _state_var1
    cdef double _state_var2

    # - saved state variables (for restoring on partial updates)
    cdef double state_var1
    cdef double state_var2

    def __init__(self, ...):
        # - initialize both working and saved state
        self._state_var1 = initial_value
        self._state_var2 = initial_value
        self.state_var1 = initial_value
        self.state_var2 = initial_value
        super().__init__(name, series)

    cdef _store(self):
        """Store working state to saved state"""
        self.state_var1 = self._state_var1
        self.state_var2 = self._state_var2

    cdef _restore(self):
        """Restore saved state to working state"""
        self._state_var1 = self.state_var1
        self._state_var2 = self.state_var2

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        # - handle initialization
        if len(self.series) < 2:
            self._state_var1 = initial_value
            self._state_var2 = initial_value
            self._store()
            return np.nan

        # - CRITICAL: store/restore based on new_item_started flag
        if new_item_started:
            self._store()  # - save final state from previous bar
        else:
            self._restore()  # - restore state from when this bar started

        # - perform calculations using _working state variables
        # - update _working state
        self._state_var1 = new_value1
        self._state_var2 = new_value2

        return result
```

**Key Points**:
- **Two sets of variables**: Working (`_var`) and saved (`var`)
- **Store on new bar**: `if new_item_started: self._store()`
- **Restore on update**: `else: self._restore()`
- This ensures each update to a bar starts from the same initial state

#### Test Pattern

```python
def test_my_indicator(self):
    # - STEP 1: load 1h data
    r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
    c1h = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

    # - STEP 2: test on same timeframe (1h → 1h)
    ind_1h = my_indicator(c1h, length=22)
    ind_1h_pd = pta.my_indicator(c1h.pd(), length=22)
    diff_1h = abs(ind_1h.pd() - ind_1h_pd["trend"]).dropna()
    assert diff_1h.sum() < 1e-6

    # - STEP 3: test streaming (bar-by-bar)
    ohlc_stream = OHLCV("test", "1h")
    ind_stream = my_indicator(ohlc_stream, length=22)

    c1h_pd = c1h.pd()
    for idx in c1h_pd.index:
        bar = c1h_pd.loc[idx]
        ohlc_stream.update_by_bar(
            int(idx.value),
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar.get("volume", 0)
        )

    ind_stream_pd = pta.my_indicator(ohlc_stream.pd(), length=22)
    diff_stream = abs(ind_stream.pd() - ind_stream_pd["trend"]).dropna()
    assert diff_stream.sum() < 1e-6

    # - STEP 4: test cross-timeframe (1h → 4h)
    # - this is the CRITICAL test for store/restore pattern
    ohlc_4h_stream = OHLCV("test_4h", "4h")
    ind_4h_stream = my_indicator(ohlc_4h_stream, length=22)

    # - feed 1h data to build 4h bars
    for idx in c1h_pd.index:
        bar = c1h_pd.loc[idx]
        ohlc_4h_stream.update_by_bar(
            int(idx.value),
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar.get("volume", 0)
        )

    # - calculate on final 4h bars
    ind_4h_pd = pta.my_indicator(ohlc_4h_stream.pd(), length=22)

    # - compare results
    diff_4h_trend = abs(ind_4h_stream.pd() - ind_4h_pd["trend"]).dropna()
    assert diff_4h_trend.sum() < 1e-6, f"4h streaming differs: {diff_4h_trend.sum()}"
```

#### When Store/Restore is Required

**Required for**:
- ✅ Indicators maintaining trend state (PSAR, SuperTrend)
- ✅ Indicators with cumulative calculations (CUSUM Filter)
- ✅ Indicators tracking previous bar values

**Not required for**:
- ❌ Simple indicators without state (SMA, EMA)
- ❌ Indicators that only use internal series (ATR, Bollinger Bands)
- ❌ Stateless transformations

#### Debugging Cross-Timeframe Issues

If 4h test fails but 1h tests pass:

1. **Check for missing store/restore**:
```python
# - add to calculate()
if new_item_started:
    print(f"NEW BAR: time={time}, state={self._state_var}")
    self._store()
else:
    print(f"UPDATE: time={time}, restoring state")
    self._restore()
```

2. **Compare values at divergence point**:
```python
# - in test
print(f"\n4h streaming first 30:\n{ind_4h_stream.pd().head(30)}")
print(f"\n4h pandas first 30:\n{ind_4h_pd['trend'].head(30)}")
diff = ind_4h_stream.pd() - ind_4h_pd['trend']
print(f"\nFirst difference at:\n{diff[diff != 0].head(10)}")
```

3. **Verify OHLC bar aggregation**:
```python
# - check that 4h bars are formed correctly
print(f"4h bars shape: {ohlc_4h_stream.pd().shape}")
print(f"Expected: {len(c1h_pd) / 4} bars")
```

#### Real Example: SuperTrend Store/Restore

**File**: `src/qubx/ta/indicators.pyx:1559-1750`

```python
cdef class SuperTrend(IndicatorOHLC):
    def __init__(self, str name, OHLCV series, ...):
        # - working state (updated during calculation)
        self._prev_longstop = np.nan
        self._prev_shortstop = np.nan
        self._prev_direction = np.nan

        # - saved state (for partial bar updates)
        self.prev_longstop = np.nan
        self.prev_shortstop = np.nan
        self.prev_direction = np.nan

        # - ... rest of initialization
        super().__init__(name, series)

    cdef _store(self):
        self.prev_longstop = self._prev_longstop
        self.prev_shortstop = self._prev_shortstop
        self.prev_direction = self._prev_direction

    cdef _restore(self):
        self._prev_longstop = self.prev_longstop
        self._prev_shortstop = self.prev_shortstop
        self._prev_direction = self.prev_direction

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        if len(self.series) < 2:
            self._prev_longstop = np.nan
            self._prev_shortstop = np.nan
            self._prev_direction = np.nan
            self._store()
            return np.nan

        # - CRITICAL: store/restore for partial bar updates
        if new_item_started:
            self._store()  # - save previous bar's final state
        else:
            self._restore()  # - restore this bar's initial state

        # - calculate using _prev_* working variables
        # - save previous stops for comparison
        saved_prev_longstop = self._prev_longstop
        saved_prev_shortstop = self._prev_shortstop

        # - calculate new stops
        longstop = calc_longstop(...)
        shortstop = calc_shortstop(...)

        # - determine direction by comparing against saved_prev_* values
        if low < saved_prev_longstop:
            direction = -1.0
        elif high > saved_prev_shortstop:
            direction = 1.0
        else:
            direction = self._prev_direction

        # - update working state
        self._prev_longstop = longstop
        self._prev_shortstop = shortstop
        self._prev_direction = direction

        return direction
```

**Declaration in .pxd**:
```python
cdef class SuperTrend(IndicatorOHLC):
    # - working state
    cdef double _prev_longstop
    cdef double _prev_shortstop
    cdef double _prev_direction

    # - saved state
    cdef double prev_longstop
    cdef double prev_shortstop
    cdef double prev_direction

    cdef _store(self)
    cdef _restore(self)
```

**Test**: `tests/qubx/ta/indicators_test.py::TestIndicators::test_super_trend` (lines 815-842)

#### Summary

Cross-timeframe testing is **essential** for validating indicators work correctly in live trading scenarios where:
- Higher timeframe bars are built from lower timeframe ticks/bars
- Bars receive multiple updates before being finalized
- State management must be correct to avoid drift

The store/restore pattern ensures that repeated calculations on the same bar (during partial updates) always start from the correct initial state, preventing state corruption and ensuring results match batch calculations.

### Store/Restore for Value-Based Indicators

While the previous examples focused on OHLC indicators (PSAR, SuperTrend), the store/restore pattern is equally critical for **value-based indicators** that maintain state across bars, such as PctChange and StdEma.

#### The Critical Insight: Store BEFORE Adding New Bar

The most important discovery when implementing store/restore is **when** to call `_store()`:

**❌ Wrong Pattern** (store after adding bar):
```python
if new_item_started:
    # - add new bar first
    self.state_var += new_value
    self.count += 1

    # - then store
    self._store()  # ❌ This stores state WITH current bar included
```

**✅ Correct Pattern** (store before adding bar):
```python
if new_item_started:
    # - store BEFORE adding new bar
    self._store()  # ✅ This stores state WITHOUT current bar

    # - then add new bar
    self.state_var += new_value
    self.count += 1
```

**Why this matters**:
- Storing before means stored state doesn't include current bar
- When restore() is called, you're back to the state before current bar was added
- The else branch can then apply the exact same logic as new_item branch
- This makes the implementation simpler and eliminates subtle bugs

#### The Second Critical Insight: Don't Restore Count

When implementing `_restore()`, you must **NOT restore count**:

```python
cdef void _store(self):
    self.stored_count = self.count  # - store count for reference
    self.stored_state_var = self.state_var
    # - ... store other state

cdef void _restore(self):
    # - DON'T restore count! It should only increment on new bars
    # self.count = self.stored_count  # ❌ NEVER DO THIS

    self.state_var = self.stored_state_var  # ✅ restore other state
    # - ... restore other state
```

**Why**:
- Count should only increment when `new_item_started=True`
- If you restore count, every bar appears as the first bar
- Results will be completely wrong (e.g., all NaN or zeroes)

#### Example 1: PctChange (Simple Value-Based Indicator)

**Complexity**: Low (simple deque state)

**Implementation**:
```python
cdef class PctChange(Indicator):
    """Percentage change over N periods"""

    cdef int period
    cdef object past_values  # - deque to hold historical values
    cdef int _count

    # - store/restore variables
    cdef object stored_past_values
    cdef int stored_count

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.past_values = deque(maxlen=period + 1)
        self._count = 0
        self.stored_past_values = None
        self.stored_count = 0
        super().__init__(name, series)

    cdef void _store(self):
        """Store current state"""
        # - create a copy of the deque
        self.stored_past_values = deque(self.past_values, maxlen=self.period + 1)
        self.stored_count = self._count

    cdef void _restore(self):
        """Restore state for bar update"""
        if self.stored_past_values is not None:
            # - restore deque from copy
            self.past_values = deque(self.stored_past_values, maxlen=self.period + 1)
            self._count = self.stored_count

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - force first value to be treated as new item
        if len(self.past_values) == 0:
            new_item_started = True

        # - restore state if updating existing bar
        if not new_item_started:
            self._restore()

        if new_item_started:
            # - store state BEFORE adding new bar
            self._store()

            # - add new value to history
            self.past_values.append(value)
            self._count += 1
        else:
            # - bar update: just update the last value
            if len(self.past_values) > 0:
                self.past_values[-1] = value

        # - calculate percentage change
        if len(self.past_values) < self.period + 1:
            return np.nan

        old_value = self.past_values[0]
        if old_value == 0:
            return np.nan

        return (value - old_value) / abs(old_value)
```

**Key Points**:
- Simple state: just a deque of past values
- Store creates a copy of the deque using `deque(self.past_values, maxlen=...)`
- Else branch just updates last value (doesn't append a new one)
- Count is used but not restored in `_restore()`

#### Example 2: StdEma (Complex EWM-Based Indicator)

**Complexity**: High (exponential weighted moving average with variance tracking)

**Challenge**: StdEma maintains 5 state variables that form an exponentially weighted moving standard deviation. Without store/restore, streaming values diverge dramatically from non-streaming (2.6x difference!).

**The Problem**:
- Daily bar built from 24 hourly updates
- Each update applies exponential decay
- Without store/restore: 24 decays accumulate, causing wrong results
- With store/restore: each update starts from same state, correct results

**Implementation**:
```python
cdef class StdEma(Indicator):
    """Exponentially weighted moving standard deviation"""

    cdef int period
    cdef double alpha

    # - state variables (working state)
    cdef int count
    cdef double ewm_mean_numer
    cdef double ewm_mean_denom
    cdef double ewm_var_numer
    cdef double ewm_var_denom
    cdef double prev_mean

    # - store/restore variables (saved state)
    cdef int stored_count
    cdef double stored_ewm_mean_numer
    cdef double stored_ewm_mean_denom
    cdef double stored_ewm_var_numer
    cdef double stored_ewm_var_denom
    cdef double stored_prev_mean

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)

        # - initialize state
        self.count = 0
        self.ewm_mean_numer = 0.0
        self.ewm_mean_denom = 0.0
        self.ewm_var_numer = 0.0
        self.ewm_var_denom = 0.0
        self.prev_mean = 0.0

        # - initialize stored state
        self.stored_count = 0
        self.stored_ewm_mean_numer = 0.0
        self.stored_ewm_mean_denom = 0.0
        self.stored_ewm_var_numer = 0.0
        self.stored_ewm_var_denom = 0.0
        self.stored_prev_mean = 0.0

        super().__init__(name, series)

    cdef void _store(self):
        """Store current state"""
        self.stored_count = self.count
        self.stored_ewm_mean_numer = self.ewm_mean_numer
        self.stored_ewm_mean_denom = self.ewm_mean_denom
        self.stored_ewm_var_numer = self.ewm_var_numer
        self.stored_ewm_var_denom = self.ewm_var_denom
        self.stored_prev_mean = self.prev_mean

    cdef void _restore(self):
        """Restore state for bar update - DON'T restore count!"""
        # - restore all state EXCEPT count
        # - count should only increment on new bars
        self.ewm_mean_numer = self.stored_ewm_mean_numer
        self.ewm_mean_denom = self.stored_ewm_mean_denom
        self.ewm_var_numer = self.stored_ewm_var_numer
        self.ewm_var_denom = self.stored_ewm_var_denom
        self.prev_mean = self.stored_prev_mean

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - force first value to be treated as new item
        if self.count == 0:
            new_item_started = True

        # - restore state if updating existing bar
        if not new_item_started:
            self._restore()

        # - EWM calculation parameters
        cdef double w_decay = 1.0 - self.alpha
        cdef double new_weight = 1.0
        cdef double current_mean, delta, delta_new

        if new_item_started:
            # - CRITICAL: store BEFORE adding new bar
            self._store()

            # - decay previous weights and add new value
            self.ewm_mean_numer = w_decay * self.ewm_mean_numer + new_weight * value
            self.ewm_mean_denom = w_decay * self.ewm_mean_denom + new_weight

            current_mean = self.ewm_mean_numer / self.ewm_mean_denom

            # - update variance using Welford's online algorithm
            delta = value - self.prev_mean
            delta_new = value - current_mean

            self.ewm_var_numer = w_decay * self.ewm_var_numer + new_weight * delta * delta_new
            self.ewm_var_denom = w_decay * self.ewm_var_denom + new_weight

            self.prev_mean = current_mean
            self.count += 1
        else:
            # - bar update: apply SAME logic as new_item, but don't increment count
            # - after restore, we're back to state before this bar
            # - so we can apply the exact same update logic
            self.ewm_mean_numer = w_decay * self.ewm_mean_numer + new_weight * value
            self.ewm_mean_denom = w_decay * self.ewm_mean_denom + new_weight

            current_mean = self.ewm_mean_numer / self.ewm_mean_denom

            delta = value - self.prev_mean
            delta_new = value - current_mean

            self.ewm_var_numer = w_decay * self.ewm_var_numer + new_weight * delta * delta_new
            self.ewm_var_denom = w_decay * self.ewm_var_denom + new_weight

            self.prev_mean = current_mean
            # - NOTE: don't increment count here!

        # - calculate standard deviation
        if self.count < self.period:
            return np.nan

        if self.ewm_var_denom == 0:
            return np.nan

        variance = self.ewm_var_numer / self.ewm_var_denom
        return sqrt(max(0.0, variance))
```

**Key Points**:
- Complex state: 5 accumulators for EWM mean and variance
- Both branches (new_item and else) have **identical calculation logic**
- Only difference: new_item increments count, else doesn't
- This works because `_store()` is called BEFORE adding bar
- Not restoring count is critical to avoid all NaN results

**Real Results**:
- Without store/restore: streaming vol = 0.100952 vs non-streaming = 0.038469 (2.6x error!)
- With store/restore: perfect match (diff < 1e-10)

#### Example 3: Cross-Timeframe Timestamp Lookback

When indicators access data from a **higher timeframe** (e.g., daily volatility in hourly indicator), timestamp conventions matter:

**Problem**:
- Qubx uses **start-of-bar** timestamps (2023-07-24 00:00 = data FOR July 24)
- Pandas uses **end-of-bar** timestamps (2023-07-24 23:59 = data FROM July 23)

**Solution**: Look back one timeframe period when accessing higher timeframe data:

```python
cdef class CusumFilter(Indicator):
    """Uses daily volatility threshold for hourly price movements"""

    cdef SeriesCachedValue target_cache  # - daily volatility series

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - CRITICAL: look back one period to get previous completed bar
        # - this matches pandas .shift(1) behavior
        cdef long long lookup_time = time - self.target_cache.ser.timeframe
        target_value = self.target_cache.value(lookup_time)

        # - use target_value as threshold
        # - ... rest of calculation
```

**Why this works**:
- On July 24 at 00:00 (start of day), we look back 1 day
- This gives us July 23's completed volatility
- Matches pandas: `vol.shift(1)` on July 24 also gives July 23's value

**Test verification**:
- Without lookback: 20 events detected
- With lookback: 67 events detected (matches pandas exactly)

#### Testing Value-Based Store/Restore Indicators

**Test structure** (comparing hourly → daily streaming vs direct daily):

```python
def test_stdema_streaming(self):
    # - STEP 1: load hourly data
    reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]
    raw = reader.read("ETHUSDT", "ohlc(1h)", "2021-10-01", "2022-03-01")

    # - STEP 2: non-streaming (resample then calculate)
    ohlc = raw.to_ohlc()
    daily = ohlc.resample("1d")
    vol = stdema(pct_change(daily.close), 30)
    result_ns = vol.pd().dropna()

    # - STEP 3: streaming (calculate on resampled series built from hourly bars)
    H1 = OHLCV("streaming", "1h")
    D1 = H1.resample("1d")  # - creates resampled series
    vol_streaming = stdema(pct_change(D1.close), 30)

    # - feed hourly bars one by one
    bars = ohlc.pd()
    for idx in bars.index:
        bar = bars.loc[idx]
        H1.update_by_bar(
            int(idx.value),
            bar["open"], bar["high"], bar["low"], bar["close"],
            bar.get("volume", 0)
        )

    result_s = vol_streaming.pd().dropna()

    # - STEP 4: compare
    diff = (result_s - result_ns).dropna()
    assert diff.abs().max() < 1e-10, f"Streaming differs: max={diff.abs().max()}"
    print(f"✅ Perfect match: max diff = {diff.abs().max():.15f}")
```

**Debug technique** (add to indicator during development):

```python
cpdef double calculate(self, long long time, double value, short new_item_started):
    # - debug first 10 bars
    if self.count < 10:
        print(f"[StdEma] count={self.count} value={value:.6f} new={new_item_started} "
              f"mean_num={self.ewm_mean_numer:.6f} result={result:.6f}")

    return result
```

#### Common Mistakes and Iterations

When implementing StdEma store/restore, these mistakes were made:

**Iteration 1**: Stored after adding bar
- Result: vol = 0.100952 vs expected 0.038469 (2.6x error)
- Problem: Else branch became complex, hard to replicate logic

**Iteration 2**: Removed weight decay in else branch
- Result: vol = 0.027553 vs expected 0.038469
- Problem: EWM algorithm requires decay on every update

**Iteration 3**: Applied decay but wrong logic
- Result: vol = 0.029889 vs expected 0.038469
- Problem: Double-decaying due to restore then decay

**Iteration 4**: Store before, don't restore count
- Result: All NaN
- Problem: Restoring count broke the `if self.count < self.period` check

**Final solution**: Store before, don't restore count, identical logic in both branches
- Result: Perfect match (diff < 1e-10) ✅

#### Implementation Complexity Rating

From the session work:
- **Overall complexity**: 8/10
- **PctChange store/restore**: 4/10 (simple deque state)
- **StdEma store/restore**: 9/10 (5 state variables, EWM algorithm, multiple iterations needed)
- **CUSUM timestamp lookback**: 6/10 (requires understanding of timestamp conventions)

#### Summary: Value-Based Indicator Store/Restore

The store/restore pattern for value-based indicators follows these principles:

1. **Store BEFORE adding new bar** - makes else branch simple
2. **Don't restore count** - count only increments on new bars
3. **Else branch = new_item branch minus count increment** - exact same logic
4. **All state variables need stored copies** - except count
5. **Test with streaming (hourly → daily)** - only way to catch issues
6. **Cross-timeframe access needs lookback** - account for timestamp conventions

These patterns ensure perfect streaming/non-streaming match, which is essential for backtesting accuracy and live trading confidence.

### Pattern 6: Handling Intrabar Updates from Ticks

**Use when**: Your indicator needs to process tick-by-tick data where each bar is built from multiple ticks (OPEN, HIGH, LOW, CLOSE).

**Complexity**: Very High (9/10) - requires understanding tick semantics, event timing, and careful state management.

**The Problem**: Static OHLC vs Tick-Based Sequential Updates

When testing indicators, two different data flows can produce different results:

1. **Static OHLC**: Each bar processed ONCE with its final CLOSE value
   - Bar at 00:00 with value=3721.67 (CLOSE)
   - Indicator calculates once per bar

2. **Tick-Based**: Each bar built from MULTIPLE ticks (O, H, L, C)
   - Bar at 00:00 receives 4 updates:
     - OPEN tick: value=3676.01 (new_item=1)
     - HIGH tick: value=3730.00 (new_item=0)
     - LOW tick: value=3676.01 (new_item=0)
     - CLOSE tick: value=3721.67 (new_item=0)
   - Indicator calculates 4 times per bar

**The Core Issue**: When `new_item=1` (bar start), tick-based data starts with OPEN value, while static OHLC starts with CLOSE value. This causes different event timestamps and counts.

**Symptoms**:
- Events occur at different timestamps between static and tick-based
- Event counts differ (e.g., 67 vs 25 events)
- Overlap between methods is poor (<100%)

#### The Solution: OPEN Tick Detection Pattern

**Strategy**: Detect when a new bar starts with an OPEN tick (value ≈ previous CLOSE) and skip processing it, only process the CLOSE tick.

**Key Components**:

1. **Last Value Tracking**: Track the final processed value separately from the baseline value
2. **OPEN Tick Detection**: Check if new bar value is close to previous bar's final value
3. **Event Lag Management**: Use current/previous bar event tracking for 1-bar lag
4. **Conditional Processing**: Skip cumulative calculations for OPEN ticks

**Implementation Pattern**:

```python
cdef class MyIndicator(Indicator):
    # - baseline value for diff calculations (previous bar's FINAL value)
    cdef double prev_value

    # - last processed value (updated on every tick)
    cdef double last_value

    # - event tracking (1-bar lag)
    cdef double prev_bar_event  # - event from previous completed bar (what we return)
    cdef double current_bar_event  # - event being calculated for current bar

    # - state for restore
    cdef double saved_state_var

    def __init__(self, str name, TimeSeries series, ...):
        self.prev_value = np.nan
        self.last_value = np.nan
        self.prev_bar_event = 0.0
        self.current_bar_event = 0.0
        super().__init__(name, series)

    cdef void _store(self):
        """Store state for intrabar restore"""
        self.saved_state_var = self.state_var
        # - DON'T store prev_value - it should remain as previous bar's final value

    cdef void _restore(self):
        """Restore state for intrabar updates"""
        self.state_var = self.saved_state_var

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double threshold
        cdef int event = 0

        # - STEP 1: handle first value
        if np.isnan(self.prev_value):
            self.prev_value = value
            self.last_value = value
            self._store()
            self.current_bar_event = 0.0
            return 0.0

        # - STEP 2: update prev_value to last bar's final value
        if new_item_started:
            self.prev_value = self.last_value

        # - STEP 3: restore state for intrabar updates
        if not new_item_started:
            self._restore()
        else:
            # - STEP 4: detect OPEN tick and skip processing
            # - use relative threshold (0.01% + small absolute) for different price ranges
            threshold = abs(self.last_value * 0.0001) + 0.001
            if abs(value - self.last_value) < threshold:
                # - this is an OPEN tick (value ≈ previous CLOSE)
                # - store state without processing
                self._store()
                self.last_value = value
                # - move current bar's event to previous (1-bar lag)
                self.prev_bar_event = self.current_bar_event
                self.current_bar_event = 0.0
                return self.prev_bar_event

        # - STEP 5: calculate using prev_value as baseline
        # - all ticks (H, L, C) and static OHLC will execute this
        diff = value - self.prev_value
        self.state_var += diff

        # - perform event detection logic
        if self.state_var > threshold:
            event = 1
            self.state_var = 0.0

        # - STEP 6: manage event tracking and state
        if new_item_started:
            # - new bar (CLOSE tick for tick-based, or direct CLOSE for static OHLC)
            self.prev_bar_event = self.current_bar_event
            self.current_bar_event = float(event)
            self.last_value = value
            self._store()
            return self.prev_bar_event
        else:
            # - intrabar update (H, L, or C ticks)
            # - update current bar's event (CLOSE tick overwrites OPEN tick event)
            self.current_bar_event = float(event)
            self.last_value = value
            # - DON'T store: let final cumulative sums stay in memory
            return self.prev_bar_event
```

**Critical Implementation Details**:

1. **Separate Value Tracking**:
   - `prev_value`: Baseline for diff calculations (previous bar's FINAL value)
   - `last_value`: Last processed value (updated every tick)
   - On new bar start: `prev_value = last_value`

2. **OPEN Tick Detection**:
   - Threshold: `abs(last_value * 0.0001) + 0.001` (0.01% + small absolute)
   - If `abs(value - last_value) < threshold` → OPEN tick
   - Skip all cumulative calculations for OPEN ticks

3. **Event Lag Management**:
   - `current_bar_event`: Event calculated for current bar (updated by all ticks)
   - `prev_bar_event`: Event from previous completed bar (what we return)
   - On new bar: `prev_bar_event = current_bar_event`
   - Ensures events are based on FINAL bar values (CLOSE), not initial (OPEN)

4. **State Storage**:
   - Store AFTER calculating event (so resets are captured)
   - DON'T store during intrabar updates (let final state carry forward)
   - DON'T restore `prev_value` (should remain as previous bar's final value)

#### Real Example: CUSUM Filter with Intrabar Handling

**File**: `src/qubx/ta/indicators.pyx:1459-1562`

**Declaration in .pxd**:
```python
cdef class CusumFilter(Indicator):
    cdef double s_pos, s_neg
    cdef double prev_value  # - baseline for diff calculations
    cdef double last_value  # - last processed value
    cdef double saved_s_pos, saved_s_neg, saved_prev_value
    cdef double prev_bar_event  # - event from previous completed bar
    cdef double current_bar_event  # - event for current bar being calculated
    cdef SeriesCachedValue target_cache
```

**Implementation Highlights**:
```python
cdef class CusumFilter(Indicator):
    def __init__(self, str name, TimeSeries series, TimeSeries target):
        self.target_cache = SeriesCachedValue(target)
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.prev_value = np.nan
        self.last_value = np.nan
        self.prev_bar_event = 0.0
        self.current_bar_event = 0.0
        super().__init__(name, series)

    cdef void _store(self):
        """Store cumulative sums for intrabar restore"""
        self.saved_s_pos = self.s_pos
        self.saved_s_neg = self.s_neg
        # - DON'T store prev_value - keep it as previous bar's final value

    cdef void _restore(self):
        """Restore cumulative sums for intrabar updates"""
        self.s_pos = self.saved_s_pos
        self.s_neg = self.saved_s_neg

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double diff, threshold, target_value
        cdef int event = 0

        # - first value
        if np.isnan(self.prev_value):
            self.prev_value = value
            self.last_value = value
            self._store()
            self.current_bar_event = 0.0
            return 0.0

        # - update prev_value to last bar's final value
        if new_item_started:
            self.prev_value = self.last_value

        # - restore state for intrabar updates
        if not new_item_started:
            self._restore()
        else:
            # - detect OPEN tick
            threshold = abs(self.last_value * 0.0001) + 0.001
            if abs(value - self.last_value) < threshold:
                # - OPEN tick: skip processing
                self._store()
                self.last_value = value
                self.prev_bar_event = self.current_bar_event
                self.current_bar_event = 0.0
                return self.prev_bar_event

        # - calculate diff from previous bar's final value
        diff = value - self.prev_value

        # - update cumulative sums
        self.s_pos = max(0.0, self.s_pos + diff)
        self.s_neg = min(0.0, self.s_neg + diff)

        # - get threshold from target series
        target_value = self.target_cache.value(time)

        # - check for events
        if not np.isnan(target_value):
            threshold = abs(target_value * value)
            if self.s_neg < -threshold:
                self.s_neg = 0.0
                event = 1
            elif self.s_pos > threshold:
                self.s_pos = 0.0
                event = 1

        # - manage event tracking with 1-bar lag
        if new_item_started:
            self.prev_bar_event = self.current_bar_event
            self.current_bar_event = float(event)
            self.last_value = value
            self._store()
            return self.prev_bar_event
        else:
            self.current_bar_event = float(event)
            self.last_value = value
            return self.prev_bar_event
```

#### TickSeries Transformer Requirements

For tick-based testing to work correctly, the `TickSeries` transformer MUST generate a CLOSE trade as the final tick for each bar.

**Problem**: Original transformer only generated trades for O, H, L, missing the CLOSE.

**Solution**: Add CLOSE trade generation (already fixed in transformers.py).

**Test file**: `tests/qubx/ta/indicators_test.py::TestIndicators::test_cusum_filter_on_events`

#### Testing Pattern for Intrabar Updates

```python
def test_cusum_filter_on_events(self):
    from qubx.data.transformers import TickSeries

    reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]

    # - STEP 1: static OHLC (each bar processed once with final close value)
    c1h = reader.read("ETHUSDT", "ohlc(1h)", "2021-12-01", "2022-02-01").to_ohlc()
    c1d = c1h.resample("1d")
    vol = stdema(pct_change(c1d.close), 30)
    r = cusum_filter(c1h.close, vol * 0.3)
    r_pd = r.pd()

    # - STEP 2: tick-based (each bar built from 4 trades: O, H, L, C)
    ticks = reader.read("ETHUSDT", "ohlc(1h)", "2021-12-01", "2022-02-01").transform(
        TickSeries(quotes=False, trades=True)
    )

    s1h = OHLCV("s1", "1h")
    s1d = s1h.resample("1d")
    vol1 = stdema(pct_change(s1d.close), 30)
    r1 = cusum_filter(s1h.close, vol1 * 0.3)

    # - feed ticks one by one
    for t in ticks:
        s1h.update(t.time, t.price, t.size)

    r1_pd = r1.pd()

    # - STEP 3: compare - events MUST match exactly
    assert all(r_pd[r_pd == 1].head(25) == r1_pd[r1_pd == 1].head(25))
```

#### Test Results Timeline

**Before fixes**:
- Static OHLC: 67 events
- Tick-based: 25 events
- Overlap: 0% (completely different timestamps)

**After implementing OPEN tick detection**:
- Static OHLC: 25 events (first 25)
- Tick-based: 25 events (first 25)
- Overlap: **100%** ✅ (all timestamps match exactly)

**Full test suite results**:
- `test_cusum_filter_on_events`: **100% match** (25/25 events) ✅
- `test_cusum_in_strategy`: **100% overlap** (67/67 events) ✅
- `test_cusum_streaming_vs_static`: PASSED ✅
- `test_cusum_with_preloaded_data`: PASSED ✅

#### When This Pattern is Required

**Required for**:
- ✅ Indicators with cumulative calculations (CUSUM, running sums)
- ✅ Event detection indicators (signal when threshold crossed)
- ✅ Indicators that compare current vs previous bar values
- ✅ Any indicator that processes tick-by-tick data in production

**Not required for**:
- ❌ Simple moving averages (process each tick independently)
- ❌ Indicators that only use final bar values
- ❌ Indicators tested only with static OHLC data

#### Summary: Intrabar Updates Pattern

The intrabar updates pattern ensures perfect match between static OHLC and tick-based processing:

1. **Track last_value separately** - final processed value vs baseline value
2. **Detect OPEN ticks** - skip processing when value ≈ previous close
3. **Use event lag** - return previous bar's event, calculate current bar's event
4. **Don't store during intrabar updates** - let final state carry forward in memory
5. **Test with tick-based data** - only way to catch these issues

This pattern is essential for indicators that will process live tick data, where bars are built incrementally from trades. Without it, event timing and counts will be incorrect, leading to poor backtest/live performance correlation.

**Complexity**: 9/10
**Impact**: Critical for production use with tick data
**Test Coverage**: Requires both static OHLC and tick-based tests

---

## Step-by-Step Implementation Guide

### Step 1: Read the Pandas Reference

**Location**: `src/qubx/pandaz/ta.py`

**Goal**: Understand the algorithm logic.

Example for MACD:
```python
def macd(x: pd.Series, fast=12, slow=26, signal=9,
         method="ema", signal_method="ema") -> pd.Series:
    x_diff = smooth(x, method, fast) - smooth(x, method, slow)
    return smooth(x_diff, signal_method, signal).rename("macd")
```

**Key Questions**:
- What inputs does it take?
- What intermediate values are calculated?
- What is the return value?
- Are there any edge cases (division by zero, NaN handling)?

### Step 2: Locate the Stub in indicators.pyx

**Search for**: Class name and helper function

```bash
# - find the class stub
grep -n "^cdef class Macd" src/qubx/ta/indicators.pyx

# - find the helper function stub
grep -n "^def macd" src/qubx/ta/indicators.pyx
```

### Step 3: Design the Indicator Structure

**Decide**:
1. What type of indicator? (`Indicator` or `IndicatorOHLC`)
2. What state variables are needed?
3. Does it need internal series? (Almost always yes)
4. Does it depend on other indicators? (Use internal series pattern)
5. Does it need caching? (If accessing other series frequently)

**Sketch the structure**:
```python
cdef class MyIndicator(Indicator):
    # - configuration
    cdef int period
    cdef str method

    # - internal series
    cdef object internal_series

    # - dependent indicators
    cdef object ma
    cdef object std

    # - state (if needed)
    cdef double prev_value

    # - cached values (if needed)
    cdef long long cached_time
    cdef double cached_value
```

### Step 4: Implement __init__

**Pattern**:
```python
def __init__(self, str name, TimeSeries series, [parameters]):
    # - store parameters
    self.period = period
    self.method = method

    # - create internal series (if needed)
    self.internal_series = TimeSeries("internal", series.timeframe,
                                      series.max_series_length)

    # - create dependent indicators
    self.ma = smooth(self.internal_series, method, period)

    # - initialize state
    self.prev_value = np.nan

    # - initialize cache
    self.cached_time = -1
    self.cached_value = np.nan

    # - MUST call super().__init__ last
    super().__init__(name, series)
```

### Step 5: Implement calculate()

**Pattern**:
```python
cpdef double calculate(self, long long time, double value, short new_item_started):
    # - STEP 1: handle edge cases
    if np.isnan(value):
        return np.nan

    if len(self.series) < self.period:
        return np.nan

    # - STEP 2: update internal series (if using internal series pattern)
    self.internal_series.update(time, value)

    # - STEP 3: access dependent indicators
    ma_value = self.ma[0] if len(self.ma) > 0 else np.nan

    # - STEP 4: perform calculations
    if np.isnan(ma_value):
        return np.nan

    result = (value - ma_value) / ma_value

    # - STEP 5: update state (if needed)
    if new_item_started:
        self.prev_value = value

    # - STEP 6: return result
    return result
```

### Step 6: Implement Helper Function

```python
def my_indicator(series: TimeSeries, period: int = 14, method: str = "ema"):
    """
    Brief description of what the indicator does.

    Longer description with algorithm details if needed.

    :param series: input time series
    :param period: calculation period
    :param method: smoothing method (sma, ema, tema, dema, kama)
    :return: indicator time series
    """
    return MyIndicator.wrap(series, period, method) # type: ignore
```

### Step 7: Build and Test

```bash
# - build the project
just build

# - run the specific test
poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_my_indicator -v

# - if test fails, add debug output and rebuild
just build && poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_my_indicator -v -s
```

### Step 8: Debug if Needed

**Common debugging techniques**:

1. **Print values in calculate()**:
```python
cpdef double calculate(self, long long time, double value, short new_item_started):
    self.internal_series.update(time, value)
    fast_value = self.fast_ma[0]
    slow_value = self.slow_ma[0]

    # - temporary debug output
    print(f"t={time}, v={value}, fast={fast_value}, slow={slow_value}")

    return fast_value - slow_value
```

2. **Compare intermediate values**:
```python
# - in test, extract intermediate series and compare with pandas
streaming_ma = my_indicator.fast_ma.pd()
pandas_ma = df['close'].ewm(span=12).mean()
print(f"MA diff: {abs(streaming_ma - pandas_ma).sum()}")
```

3. **Check first N values**:
```python
# - see where divergence starts
print(r0.pd().head(20))
print(r1.head(20))
print((r0.pd() - r1).head(20))
```

### Step 9: Verify All Tests Pass

```bash
# - run all indicator tests
poetry run pytest tests/qubx/ta/indicators_test.py -v

# - ensure nothing broke
```

---

## Real-World Examples

### Example 1: RSI (Relative Strength Index)

**Complexity**: Medium (needs separate smoothing for ups and downs)

**Key Learnings**:
- Separate series for gains (ups) and losses (downs)
- Smooth each independently
- Handle division by zero (when no movement)
- Return value in 0-100 range

**Implementation highlights**:
```python
cdef class Rsi(Indicator):
    def __init__(self, str name, TimeSeries series, int period, str smoother):
        # - create series for gains and losses
        self.ups = TimeSeries("ups", series.timeframe, series.max_series_length)
        self.downs = TimeSeries("downs", series.timeframe, series.max_series_length)

        # - smooth each independently
        self.smooth_up = smooth(self.ups, smoother, period)
        self.smooth_down = smooth(self.downs, smoother, period)

        self.prev_value = np.nan
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if np.isnan(self.prev_value):
            self.prev_value = value
            return np.nan

        # - calculate change
        change = value - self.prev_value

        # - split into gains and losses
        up = max(change, 0.0)
        down = abs(min(change, 0.0))

        # - update separate series
        self.ups.update(time, up)
        self.downs.update(time, down)

        # - update previous value
        if new_item_started:
            self.prev_value = value

        # - get smoothed values
        smooth_u = self.smooth_up[0]
        smooth_d = self.smooth_down[0]

        # - handle edge cases
        if np.isnan(smooth_u) or np.isnan(smooth_d):
            return np.nan
        if smooth_u + smooth_d == 0:
            return 50.0

        # - calculate RSI
        return 100.0 * smooth_u / (smooth_u + smooth_d)
```

**Test file**: `tests/qubx/ta/indicators_test.py::TestIndicators::test_rsi`

### Example 2: CUSUM Filter

**Complexity**: High (state management, cross-timeframe access with SeriesCachedValue)

**Key Learnings**:
- State must be saved and restored for bar updates (store/restore pattern)
- Use `SeriesCachedValue` for efficient cross-timeframe lookups
- Event detection (returns 0 or 1, not continuous values)
- Perfect use case for both SeriesCachedValue and store/restore patterns
- Demonstrates handling both performance (caching) and correctness (state management)

**Performance optimization**:
- Without caching: ~2-10 seconds
- With SeriesCachedValue: ~0.2 seconds (10-50x speedup)

**Use Case**: The CUSUM filter monitors price movements and triggers events when cumulative changes exceed a threshold. The threshold is based on volatility from a **higher timeframe** (e.g., daily volatility for hourly prices), making it a perfect candidate for `SeriesCachedValue`.

**Implementation highlights** (refactored with SeriesCachedValue):
```python
cdef class CusumFilter(Indicator):
    def __init__(self, str name, TimeSeries series, TimeSeries target):
        # - state variables
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.prev_value = np.nan

        # - saved state for bar updates
        self.saved_s_pos = 0.0
        self.saved_s_neg = 0.0
        self.saved_prev_value = np.nan

        # - use SeriesCachedValue for efficient cross-timeframe access
        self.target_cache = SeriesCachedValue(target)

        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - first value - just store it
        if np.isnan(self.prev_value):
            self.prev_value = value
            self._store()
            return 0.0

        # - restore state if updating bar
        if not new_item_started:
            self._restore()

        # - calculate diff
        diff = value - self.prev_value

        # - update cumulative sums
        self.s_pos = max(0.0, self.s_pos + diff)
        self.s_neg = min(0.0, self.s_neg + diff)

        # - get threshold from target series using cached accessor
        # - SeriesCachedValue handles all the caching logic automatically
        target_value = self.target_cache.value(time)

        # - only check for events if threshold is available
        event = 0
        if not np.isnan(target_value):
            threshold = abs(target_value * value)

            # - check for events
            if self.s_neg < -threshold:
                self.s_neg = 0.0
                event = 1
            elif self.s_pos > threshold:
                self.s_pos = 0.0
                event = 1

        # - save state for new bar
        if new_item_started:
            self.prev_value = value
            self._store()

        return float(event)
```

**What changed in refactoring**:
1. **Removed manual caching variables** (4 lines):
   - `self.target`, `self.cached_target_value`, `self.cached_target_time`, `self.cached_target_idx`
2. **Added SeriesCachedValue** (1 line):
   - `self.target_cache = SeriesCachedValue(target)`
3. **Simplified lookup** (23 lines → 1 line):
   - From: manual floor_t64 calculation + conditional lookup + cache management
   - To: `target_value = self.target_cache.value(time)`

**Required declarations** (in `.pxd` file):
```python
cdef class CusumFilter(Indicator):
    cdef double s_pos, s_neg
    cdef double prev_value
    cdef double saved_s_pos, saved_s_neg, saved_prev_value
    cdef SeriesCachedValue target_cache  # - replaces 4 manual cache variables
```

**Test file**: `tests/qubx/ta/indicators_test.py::TestIndicators::test_cusum_filter`

### Example 3: MACD (Moving Average Convergence Divergence)

**Complexity**: High (composite indicator with multiple dependent indicators)

**Key Learnings**:
- MUST use internal series pattern for composite indicators
- Multiple levels of dependency: input → fast/slow MA → MACD line → signal line
- Classic example of why calculation order matters

**Common mistake**: Attaching fast/slow MAs directly to input series causes incorrect results.

**Implementation highlights**:
```python
cdef class Macd(Indicator):
    def __init__(self, str name, TimeSeries series, fast=12, slow=26, signal=9,
                 method="ema", signal_method="ema"):
        # - CRITICAL: create internal series for input
        self.input_series = TimeSeries("input", series.timeframe,
                                       series.max_series_length)

        # - attach MAs to INTERNAL series
        self.fast_ma = smooth(self.input_series, method, fast)
        self.slow_ma = smooth(self.input_series, method, slow)

        # - create series for MACD line
        self.macd_line_series = TimeSeries("macd_line", series.timeframe,
                                           series.max_series_length)

        # - signal line smooths the MACD line
        self.signal_line = smooth(self.macd_line_series, signal_method, signal)

        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # - update input series FIRST
        self.input_series.update(time, value)

        # - now safe to access fast/slow MAs
        fast_value = self.fast_ma[0] if len(self.fast_ma) > 0 else np.nan
        slow_value = self.slow_ma[0] if len(self.slow_ma) > 0 else np.nan

        # - calculate MACD line
        if np.isnan(fast_value) or np.isnan(slow_value):
            macd_value = np.nan
        else:
            macd_value = fast_value - slow_value

        # - update MACD line series
        self.macd_line_series.update(time, macd_value)

        # - return signal line (smoothed MACD)
        return self.signal_line[0] if len(self.signal_line) > 0 else np.nan
```

**Test file**: `tests/qubx/ta/indicators_test.py::TestIndicators::test_macd`

### Example 4: SuperTrend

**Complexity**: High (OHLC indicator, state management with store/restore, composite calculation with ATR)

**Key Learnings**:
- Trend-following indicator that maintains direction state across bars
- MUST use store/restore pattern for cross-timeframe correctness
- Calculate ATR inline using internal series (follows composite indicator pattern)
- Compare current bar against *previous bar's* stops to determine trend changes
- Separate series for upper trend line (utl) and down trend line (dtl)

**Why store/restore is critical**:
When building 4h bars from 1h data, each 4h bar receives 4 updates. Without store/restore:
- Internal state (previous stops, direction) gets corrupted by intermediate updates
- Trend changes trigger at wrong times
- Results diverge significantly from batch calculations (diff > 60.0)

With store/restore:
- Each update starts from the same initial state
- Results match batch calculations perfectly (diff < 1e-6)

**Implementation highlights**:
```python
cdef class SuperTrend(IndicatorOHLC):
    def __init__(self, str name, OHLCV series, int length, double mult, ...):
        # - working state (modified during calculation)
        self._prev_longstop = np.nan
        self._prev_shortstop = np.nan
        self._prev_direction = np.nan

        # - saved state (preserved across bar updates)
        self.prev_longstop = np.nan
        self.prev_shortstop = np.nan
        self.prev_direction = np.nan

        # - calculate ATR inline (composite indicator pattern)
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        self.atr_ma = smooth(self.tr, atr_smoother, length)

        # - output series
        self.utl = TimeSeries("utl", series.timeframe, series.max_series_length)
        self.dtl = TimeSeries("dtl", series.timeframe, series.max_series_length)

        super().__init__(name, series)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        if len(self.series) < 2:
            self._prev_longstop = np.nan
            self._prev_shortstop = np.nan
            self._prev_direction = np.nan
            self._store()
            return np.nan

        # - CRITICAL: store/restore for partial bar updates
        if new_item_started:
            self._store()  # - save final state from previous bar
        else:
            self._restore()  # - restore state from this bar's start

        # - calculate TR and update internal series
        tr_value = max(abs(bar.high - bar.low),
                      abs(bar.high - prev_bar.close),
                      abs(bar.low - prev_bar.close))
        self.tr.update(time, tr_value)

        # - get ATR (automatically recalculated via smooth)
        atr_value = self.atr_ma[0]
        if np.isnan(atr_value):
            return np.nan

        # - save previous stops for comparison (before updating them)
        saved_prev_longstop = self._prev_longstop
        saved_prev_shortstop = self._prev_shortstop

        # - calculate new stops
        src_value = (bar.high + bar.low) / 2.0
        longstop = src_value - (mult * atr_value)
        shortstop = src_value + (mult * atr_value)

        # - adjust stops based on previous values
        # - (logic to ensure stops don't move against the trend)
        # - ...

        # - determine direction by comparing against SAVED previous stops
        if bar.low < saved_prev_longstop:
            direction = -1.0  # - downtrend
        elif bar.high > saved_prev_shortstop:
            direction = 1.0  # - uptrend
        else:
            direction = self._prev_direction  # - continue current trend

        # - update working state
        self._prev_longstop = longstop
        self._prev_shortstop = shortstop
        self._prev_direction = direction

        # - update output series
        if direction == 1.0:
            self.utl.update(time, longstop)
        elif direction == -1.0:
            self.dtl.update(time, shortstop)

        return direction
```

**Cross-timeframe test results**:
- Without store/restore: `diff_4h_trend.sum() = 64.0` ❌
- With store/restore: `diff_4h_trend.sum() < 1e-6` ✅

**Test file**: `tests/qubx/ta/indicators_test.py::TestIndicators::test_super_trend` (includes 1h → 4h test)

---

## Quick Reference Checklist

When implementing a new indicator, use this checklist:

- [ ] Read pandas reference implementation in `src/qubx/pandaz/ta.py`
- [ ] Locate class and helper function stubs in `src/qubx/ta/indicators.pyx`
- [ ] Decide indicator type: `Indicator` or `IndicatorOHLC`
- [ ] Determine if internal series needed (almost always yes for composite indicators)
- [ ] Implement `__init__`:
  - [ ] Store parameters
  - [ ] Create internal series if needed
  - [ ] Create dependent indicators
  - [ ] Initialize state variables
  - [ ] Call `super().__init__(name, series)` last
- [ ] Implement `calculate()`:
  - [ ] Handle NaN and edge cases first
  - [ ] Update internal series before accessing dependent indicators
  - [ ] Perform calculations
  - [ ] Handle state management if needed
  - [ ] Return result
- [ ] Implement helper function with proper docstring
- [ ] Implement store/restore methods if indicator has state:
  - [ ] Dual state variables (working `_var` and saved `var`)
  - [ ] `_store()` method to save working state
  - [ ] `_restore()` method to restore saved state
  - [ ] Handle in `calculate()`: store on new bar, restore on update
- [ ] Build: `just build`
- [ ] Run test: `poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_name -v`
- [ ] Debug if needed (add print statements, compare intermediate values)
- [ ] Add cross-timeframe test (1h → 4h) if indicator has state
- [ ] Verify all tests pass: `poetry run pytest tests/qubx/ta/indicators_test.py -v`
- [ ] Remove debug code
- [ ] Final build: `just build`

---

## Common Cython Types Reference

```python
# - timestamps (nanoseconds)
cdef long long time

# - prices, values, calculations
cdef double value, price, result

# - counts, periods, indices
cdef int count, period, idx

# - boolean flags
cdef short is_new, has_value

# - indicator references (Python objects)
cdef object ma, std, series

# - strings
cdef str method, name
```

---

## Useful Commands

```bash
# - build after changes
just build

# - run all indicator tests
just test tests/qubx/ta/indicators_test.py

# - run specific test with verbose output
poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_macd -v

# - run test with print output
poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_macd -v -s

# - run test with full error traceback
poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_macd -v --tb=long

# - search for pattern in indicators.pyx
grep -n "^cdef class" src/qubx/ta/indicators.pyx

# - count lines in indicators.pyx
wc -l src/qubx/ta/indicators.pyx
```

---

## Additional Resources

- **Qubx Documentation**: Check project README and docs for framework details
- **CCXT Documentation**: For understanding exchange data formats
- **Pandas TA Documentation**: For reference implementations
- **Cython Documentation**: For advanced Cython features

---

## Conclusion

Implementing streaming indicators in Qubx requires understanding:
1. **The algorithm**: Study the pandas reference first
2. **The pattern**: Choose the right implementation pattern (simple, composite, stateful, cached)
3. **The pitfalls**: Avoid common mistakes (calculation order, NaN handling, test syntax)
4. **The testing**: Always compare against pandas reference

The most important patterns are:
1. **Internal series pattern for composite indicators** - Critical for correctness
2. **SeriesCachedValue for cross-timeframe access** - Critical for performance
3. **Store/restore pattern for stateful indicators** - Critical for cross-timeframe correctness
4. **Intrabar updates pattern for tick data** - Critical for production tick-by-tick processing

With these patterns and guidelines, you can confidently implement any technical indicator for the Qubx platform.

---

**Document Version**: 1.4
**Last Updated**: 2025-11-12
**Indicators Covered**: RSI, CUSUM Filter (SeriesCachedValue + timestamp lookback + intrabar updates), MACD, SuperTrend (store/restore), PctChange (store/restore), StdEma (store/restore)
**Key Addition in v1.1**: SeriesCachedValue pattern for cross-timeframe access
**Key Addition in v1.2**: Cross-timeframe testing (1h → 4h) and store/restore pattern for stateful OHLC indicators
**Key Addition in v1.3**: Store/restore for value-based indicators (PctChange, StdEma), critical insights (store before, don't restore count), cross-timeframe timestamp lookback fix
**Key Addition in v1.4**: Pattern 6 - Intrabar updates from tick data (OPEN tick detection, last_value tracking, event lag management, tick-based vs static OHLC matching)
**Total Lines in indicators.pyx**: ~1750+
