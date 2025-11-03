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
- State must be saved and restored for bar updates
- Use `SeriesCachedValue` for efficient cross-timeframe lookups
- Event detection (returns 0 or 1, not continuous values)
- Perfect use case for SeriesCachedValue pattern

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
- [ ] Build: `just build`
- [ ] Run test: `poetry run pytest tests/qubx/ta/indicators_test.py::TestIndicators::test_name -v`
- [ ] Debug if needed (add print statements, compare intermediate values)
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

With these patterns and guidelines, you can confidently implement any technical indicator for the Qubx platform.

---

**Document Version**: 1.1
**Last Updated**: 2025-11-03
**Indicators Covered**: RSI, CUSUM Filter (refactored with SeriesCachedValue), MACD
**Key Addition in v1.1**: SeriesCachedValue pattern for cross-timeframe access
**Total Lines in indicators.pyx**: ~1600+
