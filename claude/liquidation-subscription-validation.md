# Liquidation Subscription Validation

## Issue Description

When subscribing to `DataType.LIQUIDATION` for non-SWAP instruments, the system currently:
1. Allows subscription to liquidations for any instrument type (SPOT, MARGIN, FUTURE, OPTION, SWAP)
2. In live trading: Would fail with "subscription type not supported" error (no liquidation subscription method in ccxt)
3. In backtesting: Attempts to process liquidations for non-perpetual instruments if data exists

## Root Cause Analysis

- **No Market Type Validation**: System doesn't validate that liquidations only apply to SWAP instruments
- **Missing CCXT Implementation**: No liquidation subscription method in ccxt connector
- **Conceptual Issue**: Liquidations are specific to perpetual swaps with leverage and shouldn't apply to spot/future/option instruments
- **Data Availability**: Database analysis shows liquidation data only exists for `binance.umswap` tables

## Solution Approach

### Phase 1: Fix Simulation (Silent Filtering) ✅ COMPLETED
- Modify `src/qubx/backtester/simulated_data.py` to filter out non-SWAP instruments
- Add validation in `add_instruments_for_subscription()` method
- Silently ignore non-SWAP instruments for liquidation subscriptions
- Add comprehensive test coverage

### Phase 2: Fix Live Trading (Future)
- Add liquidation subscription method to ccxt connector
- Implement similar filtering logic for live trading
- Handle exchange-specific liquidation subscriptions

## Implementation Details

### Files Modified
- `src/qubx/backtester/simulated_data.py` - Silent filtering logic and consistency fix
- `tests/qubx/core/test_liquidation_subscription.py` - Comprehensive test cases

### Key Changes
1. **Silent Filtering**: Filter instruments to only SWAP types for liquidation subscriptions
2. **Consistency Fix**: Changed `_requested_data_type` from "liquidations" to "liquidation" for consistency with funding payments
3. **Enhanced Filtering Logic**: Extended `_filter_instruments_for_subscription()` to include `DataType.LIQUIDATION`
4. **Graceful Handling**: No exceptions raised, just skip non-applicable instruments
5. **Test Coverage**: Comprehensive tests for various instrument type scenarios

### Expected Behavior
- Liquidation subscription with SWAP instruments: ✅ Works normally
- Liquidation subscription with non-SWAP instruments: ✅ Silently ignored
- Liquidation subscription with mixed types: ✅ Only SWAP instruments subscribed
- Other subscription types: ✅ Unaffected

## Testing Strategy

### Test Cases Added
1. `test_liquidation_non_swap_instruments_filtered()` - Verify non-SWAP instruments are filtered
2. `test_liquidation_mixed_instrument_types()` - Verify mixed types are handled correctly
3. `test_liquidation_only_swap_instruments()` - Verify SWAP instruments work normally
4. `test_liquidation_empty_after_filtering()` - Verify behavior when no SWAP instruments remain
5. `test_other_subscription_types_unaffected()` - Verify other subscriptions are unaffected

## Progress Log

### 2025-07-29 - Phase 1 Implementation
- [x] Created planning document
- [x] Analyzed database schema to confirm liquidation data availability
- [x] Added silent filtering logic to simulated_data.py
- [x] Fixed consistency issue with _requested_data_type
- [x] Added comprehensive test cases
- [x] Validated implementation with unit and integration tests
- [x] Ensured no regression in existing functionality

### Implementation Details
- **Silent Filtering**: Enhanced existing `_filter_instruments_for_subscription()` method to include `DataType.LIQUIDATION` alongside `DataType.FUNDING_PAYMENT`
- **Clean Architecture**: Uses DataType enum comparison for type safety and maintainability
- **Consistency**: Both funding payments and liquidations now use singular form for `_requested_data_type`
- **Extensible Design**: Private method can easily be extended for future subscription type filtering requirements
- **Test Coverage**: Added 5 new test cases covering various scenarios (non-SWAP filtering, mixed types, SWAP-only, empty results, other subscription types)
- **Validation**: All 5 liquidation subscription tests pass, plus 17 funding payment tests (no regression)

### Code Quality Improvements
- **Enhanced Logic**: Extended filtering method to handle both funding payments and liquidations consistently
- **Type Safety**: Maintains DataType enum comparison approach
- **Documentation**: Added comprehensive docstring updates for the enhanced filtering method
- **Maintainability**: Cleaner code with consistent patterns between funding payments and liquidations

### Database Analysis Results
- **Schema Verification**: Confirmed liquidation data only exists in `binance.umswap.liquidations_1m`
- **Market Type Coverage**: No liquidation tables found for spot, margin, future, or other market types
- **Conceptual Validation**: Liquidations only applicable to leveraged perpetual contracts (SWAP instruments)

### Test Results
- **Total Tests**: 5 liquidation subscription tests (all new)
- **Status**: ✅ All tests passing
- **Coverage**: Non-SWAP filtering, mixed instrument types, SWAP-only subscriptions, empty results, other subscription types
- **Integration**: ✅ Real-world filtering test with mixed instruments works correctly
- **Regression**: ✅ All 17 existing funding payment tests still pass

## Future Enhancements

1. **Live Trading Support**: Add ccxt connector liquidation subscription (similar to funding payments)
2. **Logging**: Add debug logging for filtered instruments (consistent with funding payments)
3. **Configuration**: Allow configuration of filtering behavior
4. **Documentation**: Update user documentation about liquidation limitations
5. **Extended Market Types**: Consider support for other leveraged instruments if data becomes available

## Related Files
- `src/qubx/core/basics.py` - Liquidation dataclass and MarketType enum
- `src/qubx/connectors/ccxt/data.py` - Live trading connector (future enhancement)
- `tests/qubx/core/test_liquidation_subscription.py` - Test suite
- `src/qubx/data/readers.py` - AsLiquidations transformer and QuestDB integration

## Comparison with Funding Payments

| Aspect | Funding Payments | Liquidations |
|--------|------------------|--------------|
| **Market Type** | SWAP only | SWAP only |
| **Data Availability** | `binance.umswap.funding_payment` | `binance.umswap.liquidations_1m` |
| **Filtering** | ✅ Implemented | ✅ Implemented |
| **Test Coverage** | 17 tests | 5 tests |
| **Live Trading** | ❌ Not implemented | ❌ Not implemented |
| **Conceptual Basis** | Perpetual swap funding mechanism | Leveraged position liquidations |

Both implementations follow identical patterns for consistency and maintainability.