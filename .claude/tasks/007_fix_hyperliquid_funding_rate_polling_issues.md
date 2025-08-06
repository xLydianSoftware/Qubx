# Task 007: Fix Hyperliquid Funding Rate Polling Issues

## Summary
Fix multiple issues with the Hyperliquid watch_funding_rates method and the PollingToWebSocketAdapter that cause incorrect behavior when subscribing to funding rates.

## Problems Identified

### ~~1. Symbol Update Issue~~ ✅ RESOLVED
~~When calling `watch_funding_rates(symbols[:20])` after `watch_funding_rates(symbols[:10])`, the adapter doesn't recognize that symbols changed from 10 to 20.~~

**Status**: This issue has been resolved. Symbol expansion from 10 to 20 now works correctly.

### 1. Polling Should Block, Not Fallback to Cached Data
The `get_next_data()` method falls back to cached data on timeout, but for polling this shouldn't happen. Polling is configured with intervals, so it should block until new data is available rather than serving stale cached data.

**Root Cause**: Lines 196-213 in `polling_adapter.py` implement fallback logic that serves cached data when no new data arrives within timeout.

### 2. Funding Rate Parsing Issues
All funding rates show identical timestamps and `next_funding_time=numpy.datetime64('NaT')`. The user's logs show:
```
FundingRate(time=numpy.datetime64('2025-08-05T09:00:00.000'), rate=1.25e-05, interval='1h', next_funding_time=numpy.datetime64('NaT'), ...)
```

**Root Cause**: The base CCXT implementation returns:
- `timestamp: None` (not set by Hyperliquid)
- `nextFundingTimestamp: None` (not provided by API)
- `fundingTimestamp`: calculated as next hour boundary

The transformation logic in `hyperliquid.py` lines 106-118 tries to fix these fields but incorrectly maps them, leading to data corruption.

## Technical Analysis

### Base CCXT Implementation
The base `hyperliquid.py` `parse_funding_rate()` method returns:
```python
{
    'timestamp': None,  # Not set
    'fundingTimestamp': (next_hour_boundary_ms),  # Calculated value
    'nextFundingTimestamp': None,  # Not provided by API
    'fundingRate': funding,
    'markPrice': markPx,
    'indexPrice': oraclePx,
    'interval': '1h'
}
```

### Current Problematic Transformation
Lines 106-118 in `hyperliquid.py`:
```python
if transformed_info.get("timestamp") is None:
    transformed_info["timestamp"] = transformed_info.get("fundingTimestamp")

if "nextFundingTimestamp" in transformed_info:
    transformed_info["nextFundingTime"] = transformed_info["nextFundingTimestamp"]
elif "nextFundingTime" not in transformed_info:
    current_funding = transformed_info.get("fundingTimestamp")
    if current_funding:
        transformed_info["nextFundingTime"] = current_funding + FUNDING_RATE_HOUR_MS
```

This creates incorrect mappings and duplicated timestamps.

## Solution Plan

### ~~Phase 1: Fix Symbol Update Logic~~ ✅ RESOLVED
~~This issue has been resolved - symbol expansion now works correctly.~~

### Phase 1: Fix Polling Behavior 
1. **Remove fallback to cached data** in `get_next_data()` method
2. **Ensure blocking behavior** until new data is available
3. **Respect polling interval** configuration
4. **Handle timeouts appropriately** without serving stale data

### Phase 2: Fix Funding Rate Data Transformation
1. **Analyze correct field mapping** between Hyperliquid API and CCXT format
2. **Fix timestamp handling** to use proper current timestamp
3. **Calculate next funding time** correctly (current time + 1 hour)
4. **Ensure unique timestamps** for each symbol
5. **Test with actual API responses** to verify correctness

### Phase 3: Create Comprehensive Tests
1. **Unit tests** for symbol update behavior
2. **Integration tests** for polling without cached fallback
3. **Funding rate parsing tests** with real API data
4. **End-to-end test script** to reproduce the user's issues

## Testing Strategy

### Test Script Structure
Create a standalone test script that:
1. Creates fresh Hyperliquid exchange instance
2. Tests expanding symbols from 10 to 20
3. Verifies all 20 symbols are returned with unique timestamps
4. Tests polling behavior without cached data fallback
5. Validates funding rate data structure correctness

### Verification Criteria
- [x] Symbol expansion works correctly (10 → 20 symbols) ✅ RESOLVED
- [ ] No fallback to cached data during normal polling
- [ ] Each symbol has unique, current timestamp
- [ ] Next funding time is properly calculated
- [ ] All funding rate fields are populated correctly

## Implementation Priority
1. **High**: Fix funding rate data parsing (affects data quality)
2. **High**: Fix polling cached data fallback (affects real-time behavior)
3. **Medium**: Add comprehensive tests (prevents regressions)

## Files to Modify
- `src/qubx/connectors/ccxt/exchanges/hyperliquid/hyperliquid.py`
- `src/qubx/connectors/ccxt/adapters/polling_adapter.py`
- Add new test files for verification