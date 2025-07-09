# Funding Payment Subscription Validation

## Issue Description

When subscribing to `DataType.FUNDING_PAYMENT` for non-SWAP instruments, the system currently:
1. Allows subscription to funding payments for any instrument type (SPOT, MARGIN, FUTURE, OPTION, SWAP)
2. In live trading: Fails with "subscription type not supported" error (no `_subscribe_funding_payment` method in ccxt)
3. In backtesting: Attempts to process funding payments for non-perpetual instruments if data exists

## Root Cause Analysis

- **No Market Type Validation**: System doesn't validate that funding payments only apply to SWAP instruments
- **Missing CCXT Implementation**: No `_subscribe_funding_payment` method in ccxt connector
- **Conceptual Issue**: Funding payments are specific to perpetual swaps and shouldn't apply to spot/future/option instruments

## Solution Approach

### Phase 1: Fix Simulation (Silent Filtering) ✅ CURRENT PHASE
- Modify `src/qubx/backtester/simulated_data.py` to filter out non-SWAP instruments
- Add validation in `add_instruments_for_subscription()` method
- Silently ignore non-SWAP instruments for funding payment subscriptions
- Add comprehensive test coverage

### Phase 2: Fix Live Trading (Future)
- Add `_subscribe_funding_payment` method to ccxt connector
- Implement similar filtering logic for live trading
- Handle exchange-specific funding payment subscriptions

## Implementation Details

### Files Modified
- `src/qubx/backtester/simulated_data.py` - Silent filtering logic
- `tests/qubx/core/test_funding_payment_subscription.py` - Additional test cases

### Key Changes
1. **Silent Filtering**: Filter instruments to only SWAP types for funding payment subscriptions
2. **Graceful Handling**: No exceptions raised, just skip non-applicable instruments
3. **Test Coverage**: Comprehensive tests for various instrument type scenarios

### Expected Behavior
- Funding payment subscription with SWAP instruments: ✅ Works normally
- Funding payment subscription with non-SWAP instruments: ✅ Silently ignored
- Funding payment subscription with mixed types: ✅ Only SWAP instruments subscribed
- Other subscription types: ✅ Unaffected

## Testing Strategy

### Test Cases Added
1. `test_funding_payment_non_swap_instruments_filtered()` - Verify non-SWAP instruments are filtered
2. `test_funding_payment_mixed_instrument_types()` - Verify mixed types are handled correctly
3. `test_funding_payment_only_swap_instruments()` - Verify SWAP instruments work normally
4. `test_funding_payment_empty_after_filtering()` - Verify behavior when no SWAP instruments remain

## Progress Log

### 2025-01-09 - Phase 1 Implementation
- [x] Created planning document
- [x] Added silent filtering logic to simulated_data.py
- [x] Added comprehensive test cases
- [x] Validated implementation with test runs

### Implementation Details
- **Silent Filtering**: Added private method `_filter_instruments_for_subscription()` to filter out non-SWAP instruments when subscribing to funding payments
- **Clean Architecture**: Uses DataType enum comparison instead of string matching for type safety
- **Extensible Design**: Private method can easily be extended for future subscription type filtering requirements
- **Test Coverage**: Added 4 new test cases covering various scenarios (non-SWAP filtering, mixed types, SWAP-only, other subscription types)
- **Validation**: All 17 funding payment subscription tests pass, plus 11 position funding tests (no regression)

### Code Quality Improvements
- **Refactored**: Moved filtering logic to dedicated private method `_filter_instruments_for_subscription()`
- **Type Safety**: Changed from string comparison to `DataType.FUNDING_PAYMENT` enum comparison
- **Documentation**: Added comprehensive docstring for the filtering method
- **Maintainability**: Cleaner separation of concerns and easier to extend for future requirements

### Test Results
- **Total Tests**: 17 funding payment subscription tests (13 original + 4 new)
- **Status**: ✅ All tests passing
- **Coverage**: Non-SWAP filtering, mixed instrument types, SWAP-only subscriptions, other subscription types

## Future Enhancements

1. **Live Trading Support**: Add ccxt connector funding payment subscription
2. **Logging**: Add debug logging for filtered instruments
3. **Configuration**: Allow configuration of filtering behavior
4. **Documentation**: Update user documentation about funding payment limitations

## Related Files
- `src/qubx/core/basics.py` - FundingPayment dataclass and MarketType enum
- `src/qubx/connectors/ccxt/data.py` - Live trading connector (future enhancement)
- `tests/qubx/core/test_funding_payment_subscription.py` - Test suite