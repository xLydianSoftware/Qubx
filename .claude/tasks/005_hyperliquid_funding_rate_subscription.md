# Task 005: Generic Polling-to-WebSocket Adapter + Hyperliquid Funding Rates

## Overview
Create a reusable **PollingToWebSocketAdapter** class that can convert any `fetch_*` method into a `watch_*` method using intelligent polling. Then use this adapter to implement missing `watch_*` methods for Hyperliquid (funding rates) and Binance (various data types).

## Problem Statement
- Multiple exchanges lack WebSocket support for certain data types (funding rates, open interest, etc.)
- Current solution is to duplicate complex polling logic in each handler (see `open_interest.py`)
- Need a clean, reusable way to adapt polling-based `fetch_*` methods into `watch_*` methods
- Specific need: Hyperliquid `watch_funding_rates()` and Binance `watch_open_interest()`

## Current State Analysis

### ✅ Existing Infrastructure
- `src/qubx/connectors/ccxt/handlers/funding_rate.py` - Complete handler expecting `watch_funding_rates()`
- `src/qubx/connectors/ccxt/handlers/open_interest.py` - Complex polling logic (190+ lines)
- `src/qubx/connectors/ccxt/utils.py` - Conversion functions for various data types
- CCXT has `fetch_*` methods but missing `watch_*` for many data types

### ❌ Current Problems
- **Code Duplication**: Complex polling logic repeated in each handler
- **Maintenance Burden**: 190+ lines of polling code in `open_interest.py` 
- **Missing Methods**: Hyperliquid lacks `watch_funding_rates()`, Binance lacks `watch_open_interest()`
- **Inconsistent Patterns**: Each handler implements polling differently

## Proposed Solution: Generic Polling Adapter Pattern

### Phase 1: Create PollingToWebSocketAdapter
1. **Create `src/qubx/connectors/ccxt/adapters/polling_adapter.py`**
   - Extract all polling logic from `open_interest.py` 
   - Make it generic to work with any `fetch_*` method
   - Support configurable intervals, boundaries, error handling
   - Handle dynamic symbol management (add/remove instruments)

2. **Design Pattern**:
   ```python
   # Simple usage in exchange classes:
   async def watch_funding_rates(self, symbols=None, params={}):
       adapter = PollingToWebSocketAdapter(
           fetch_method=self.fetch_funding_rates,
           poll_interval_minutes=5,
           symbols=symbols,
           params=params
       )
       return await adapter.start_watching()
   ```

### Phase 2: Implement Missing Watch Methods
1. **Hyperliquid**: Add `watch_funding_rates()` using the adapter
2. **Binance**: Add `watch_open_interest()` using the adapter  
3. **Future**: Easy to add any missing `watch_*` methods

### Phase 3: Refactor Existing Handler
1. **Simplify `OpenInterestDataHandler`** to use the adapter
2. **Remove 190+ lines of polling code**, replace with simple adapter usage
3. **Maintain backward compatibility**

### Phase 4: Testing & Validation
1. **Test all adapted methods** work identically to original implementations
2. **Validate performance** is equivalent or better
3. **Test edge cases** like rapid symbol changes, errors, cancellation

## Implementation Details

### PollingToWebSocketAdapter Architecture

```python
class PollingToWebSocketAdapter:
    def __init__(self, 
                 fetch_method: Callable,
                 poll_interval_minutes: int = 5,
                 boundary_tolerance_seconds: int = 30,
                 symbols: List[str] = None,
                 params: dict = None,
                 adapter_id: str = None):
        """
        Generic adapter to convert fetch_* methods to watch_* behavior.
        
        Args:
            fetch_method: The CCXT fetch_* method to call (e.g., self.fetch_funding_rates)
            poll_interval_minutes: How often to poll (default: 5 minutes)
            boundary_tolerance_seconds: Polling boundary tolerance (default: 30s)
            symbols: Initial list of symbols to watch
            params: Additional parameters for fetch_method
            adapter_id: Unique identifier for this adapter instance
        """
        
    async def start_watching(self) -> AsyncGenerator[dict, None]:
        """Start polling and yield results as they come in."""
        
    async def add_symbols(self, new_symbols: List[str]):
        """Add symbols to existing watch list."""
        
    async def remove_symbols(self, symbols_to_remove: List[str]):
        """Remove specific symbols from watch list."""
        
    async def update_symbols(self, new_symbols: List[str]):
        """Replace entire symbol list (combination of remove_all + add)."""
        
    async def stop(self):
        """Stop polling completely and cleanup all resources."""
        
    def is_watching(self, symbol: str = None) -> bool:
        """Check if adapter is actively watching (specific symbol or any)."""
```

### Usage Pattern in Exchange Classes

```python
# In HyperliquidEnhanced class
class HyperliquidEnhanced:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._funding_rate_adapter = None
        
    async def watch_funding_rates(self, symbols=None, params={}):
        """Watch funding rates with proper unwatch support."""
        self._funding_rate_adapter = PollingToWebSocketAdapter(
            fetch_method=self.fetch_funding_rates,
            poll_interval_minutes=5,  # Hyperliquid updates hourly
            symbols=symbols,
            params=params,
            adapter_id="hyperliquid_funding_rates"
        )
        async for funding_data in self._funding_rate_adapter.start_watching():
            yield funding_data
    
    async def un_watch_funding_rates(self, symbols=None):
        """Unwatch funding rates (specific symbols or all)."""
        if self._funding_rate_adapter:
            if symbols:
                await self._funding_rate_adapter.remove_symbols(symbols)
            else:
                await self._funding_rate_adapter.stop()
                self._funding_rate_adapter = None

# In BinanceQV class  
class BinanceQV:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._open_interest_adapter = None
        
    async def watch_open_interest(self, symbols=None, params={}):
        """Watch open interest with proper unwatch support."""
        self._open_interest_adapter = PollingToWebSocketAdapter(
            fetch_method=self.fetch_open_interest,
            poll_interval_minutes=5,
            symbols=symbols,
            params=params,
            adapter_id="binance_open_interest"
        )
        async for oi_data in self._open_interest_adapter.start_watching():
            yield oi_data
            
    async def un_watch_open_interest(self, symbols=None):
        """Unwatch open interest (specific symbols or all)."""
        if self._open_interest_adapter:
            if symbols:
                await self._open_interest_adapter.remove_symbols(symbols)
            else:
                await self._open_interest_adapter.stop()
                self._open_interest_adapter = None
```

### Adapter Features
- **Smart Polling**: Uses time boundaries like existing `open_interest.py`
- **Dynamic Symbols**: Add/remove symbols during runtime without restarting polling
- **Granular Unwatch**: Unwatch specific symbols or stop entire adapter
- **Resource Management**: Proper cleanup of tasks, memory, and connections
- **Error Handling**: Robust error recovery and retry logic
- **Cancellation**: Graceful shutdown with proper resource cleanup
- **Configurable**: Different intervals for different data types

### Unwatch Functionality Details

#### Symbol-Level Management
```python
# Add new symbols to existing subscription
await adapter.add_symbols(["BTCUSDT", "ETHUSDT"])

# Remove specific symbols 
await adapter.remove_symbols(["ADAUSDT"])

# Replace entire symbol list (atomic operation)
await adapter.update_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
```

#### Complete Shutdown
```python
# Stop all polling and cleanup resources
await adapter.stop()

# Check if adapter is still active
if adapter.is_watching():
    print("Still watching symbols")
```

#### Integration with Existing Patterns
```python
# Standard CCXT unwatch pattern - works seamlessly
async def un_watch_funding_rates(self, symbols=None):
    if self._funding_rate_adapter:
        if symbols:
            # Partial unwatch - remove specific symbols
            await self._funding_rate_adapter.remove_symbols(symbols)
            if not self._funding_rate_adapter.is_watching():
                # No symbols left, cleanup adapter
                self._funding_rate_adapter = None
        else:
            # Complete unwatch - stop everything
            await self._funding_rate_adapter.stop()
            self._funding_rate_adapter = None
```

#### Edge Cases Handled
- **Empty Symbol List**: Adapter gracefully handles empty symbol list (stops polling)
- **Duplicate Symbols**: Automatically deduplicates symbol lists
- **Invalid Symbols**: Continues polling valid symbols, logs invalid ones
- **Resubscription**: Can restart adapter or modify existing one seamlessly
- **Concurrent Access**: Thread-safe symbol list modifications
- **Resource Leaks**: Ensures all asyncio tasks are properly cancelled

## Files to Create/Modify

### New Files
1. **`src/qubx/connectors/ccxt/adapters/__init__.py`** (new directory)
2. **`src/qubx/connectors/ccxt/adapters/polling_adapter.py`** (new file)
   - Generic `PollingToWebSocketAdapter` class
   - Extract all smart polling logic from `open_interest.py`
   - Make it reusable for any `fetch_*` method

### Exchange Enhancements  
3. **`src/qubx/connectors/ccxt/exchanges/hyperliquid/hyperliquid.py`**
   - Add `watch_funding_rates()` using the adapter
   - Add `un_watch_funding_rates()` for cleanup

4. **`src/qubx/connectors/ccxt/exchanges/binance/exchange.py`**
   - Add `watch_open_interest()` using the adapter
   - Add any other missing `watch_*` methods

### Handler Simplification
5. **`src/qubx/connectors/ccxt/handlers/open_interest.py`**
   - Refactor to use `PollingToWebSocketAdapter`
   - Remove 150+ lines of duplicate polling code
   - Maintain backward compatibility

### Testing
6. **Create `debug/adapter_test/`** (new directory)
   - Test strategy for adapter functionality
   - Test multiple exchanges and data types
   - Validate equivalent behavior to original implementations

## Success Criteria

### Must Have - Adapter Creation
- ✅ Generic `PollingToWebSocketAdapter` class works with any `fetch_*` method
- ✅ Comprehensive unwatch functionality (symbol-level and complete shutdown)
- ✅ Dynamic symbol management during runtime (add/remove instruments)
- ✅ Smart polling with time boundaries (like existing `open_interest.py`)
- ✅ Proper error handling, retry logic, and resource cleanup
- ✅ Thread-safe operations for concurrent symbol modifications

### Must Have - Exchange Methods
- ✅ Hyperliquid `watch_funding_rates()` works using the adapter
- ✅ Binance `watch_open_interest()` works using the adapter
- ✅ Both integrate seamlessly with existing handlers
- ✅ No breaking changes to existing code

### Should Have - Code Quality
- ✅ `OpenInterestDataHandler` simplified by 150+ lines using the adapter
- ✅ Consistent polling patterns across all data types
- ✅ Comprehensive test coverage for adapter and new methods
- ✅ Configurable polling intervals per data type

### Nice to Have - Future Extensions
- ✅ Easy to add more missing `watch_*` methods for any exchange
- ✅ Performance optimizations (batching, caching, etc.)
- ✅ Detailed logging and monitoring

## Benefits of This Approach

### Immediate Benefits
- **🎯 Solves original problem**: Hyperliquid funding rates work
- **🔧 Reduces maintenance burden**: 150+ lines of polling code → reusable adapter
- **📈 Scales easily**: Any missing `watch_*` method can be added in 5 lines

### Long-term Benefits  
- **🏗️ Architectural improvement**: Clean separation of concerns
- **🔄 Consistency**: All polling-based methods use same patterns
- **🚀 Future-proof**: Easy to extend for new exchanges/data types

## Risk Assessment

### Low Risk ✅
- Building on proven polling logic from `open_interest.py`
- Non-breaking changes (purely additive)
- Adapter pattern is well-established architectural pattern

### Mitigation Strategies
- Extract adapter logic gradually from existing working code
- Test adapter with existing `open_interest.py` before adding new methods
- Maintain backward compatibility throughout

## Timeline Estimate
- **Phase 1**: 4-5 hours (create adapter with comprehensive unwatch functionality)
- **Phase 2**: 2-3 hours (add missing `watch_*` and `un_watch_*` methods) 
- **Phase 3**: 2 hours (refactor existing handler)
- **Phase 4**: 2-3 hours (comprehensive testing including unwatch scenarios)
- **Total**: 10-13 hours

### Unwatch Implementation Adds:
- **Symbol-level management**: Dynamic add/remove during polling
- **Resource cleanup**: Proper asyncio task cancellation
- **Thread safety**: Concurrent symbol modifications
- **Edge case handling**: Empty lists, invalid symbols, etc.
- **Integration testing**: Verify compatibility with existing handlers

## Questions for Approval

1. **Adapter API**: Does the proposed `PollingToWebSocketAdapter` interface look good with the unwatch functionality?
2. **Unwatch Granularity**: Should we support symbol-level unwatch, or is complete adapter shutdown sufficient?
3. **Resource Management**: Should adapters be per-exchange-instance or shared across multiple data types?
4. **Scope**: Should we start with just funding rates, or implement the full adapter solution?
5. **Refactoring**: Should we refactor `open_interest.py` immediately, or keep it as-is initially?
6. **Testing**: Create unified test for adapter, or test each usage individually?

This approach with comprehensive unwatch functionality solves your immediate need while creating lasting architectural value! 🚀

## Key Unwatch Benefits

### For Dynamic Trading
- **✅ Symbol Management**: Add XRPUSDT, remove ETHUSDT without restarting
- **✅ Resource Efficiency**: Only poll for symbols actually needed
- **✅ Memory Management**: Proper cleanup prevents memory leaks

### For System Integration
- **✅ CCXT Compatibility**: Standard `un_watch_*` methods work as expected
- **✅ Handler Integration**: Existing subscription handlers work unchanged
- **✅ Cancellation Support**: Graceful shutdown during system restart

## References
- Your existing implementation: `/home/yuriy/projects/xfundarb/quantpylib/wrappers/hyperliquid.py` (lines 226-261)
- CCXT Hyperliquid: `/home/yuriy/devs/ccxt/python/ccxt/hyperliquid.py` (lines 954-1058)
- Current handler: `src/qubx/connectors/ccxt/handlers/funding_rate.py`