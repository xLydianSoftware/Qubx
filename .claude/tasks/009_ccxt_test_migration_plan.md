# CCXT Test Migration Plan - Task 009

## Overview

After reviewing the new CCXT implementation (`src/qubx/connectors/ccxt`), current tests (`tests/qubx/connectors/ccxt`), and old tests (`debug/old_ccxt`), I've identified significant test coverage gaps that need to be addressed by migrating and updating valuable old tests.

## Current Test Coverage Analysis

### ✅ Tests Already Present (Well Covered)
- `test_data_provider.py` - CcxtDataProvider functionality
- `test_funding_rate_handler.py` - FundingRateDataHandler
- `test_orderbook_handler.py` - OrderBookDataHandler  
- `test_subscription_manager.py` - SubscriptionManager
- `test_subscription_orchestrator.py` - SubscriptionOrchestrator

### ❌ Critical Missing Test Coverage

#### 1. Core Component Tests (High Priority)
- **DataTypeHandlerFactory** (`test_handler_factory.py`) - Handler registration, retrieval, caching
- **ConnectionManager** (`test_connection_manager.py`) - WebSocket connections, retry logic, error handling
- **WarmupService** (`test_warmup_service.py`) - Historical data warmup operations
- **Utils/Converters** (`utils_test.py`) - Data conversion utilities (orderbook, liquidation, symbol mapping)
- **Position Handling** (`position_test.py`) - Position restoration from deals, CCXT conversions

#### 2. Adapter Tests (Medium Priority) 
- **PollingAdapter** (`adapters/test_polling_adapter.py`) - Polling-to-WebSocket adapter functionality

#### 3. Handler Tests (Medium Priority)
Missing handlers that should have test coverage:
- OhlcDataHandler
- TradeDataHandler  
- QuoteDataHandler
- LiquidationDataHandler
- OpenInterestDataHandler

## Migration Plan (Updated)

### Phase 1: Critical Core Components (Immediate)

#### 1.1 ✅ Migrate Handler Factory Tests - PRIORITY 1
**File**: `debug/old_ccxt/test_handler_factory.py` → `tests/qubx/connectors/ccxt/test_handler_factory.py`
- Direct migration with import updates
- Verify handler class names match new implementation
- Update fixtures to use new architecture

#### 1.2 Migrate Connection Manager Tests - PRIORITY 2  
**File**: `debug/old_ccxt/test_connection_manager.py` → `tests/qubx/connectors/ccxt/test_connection_manager.py`
- Verify ConnectionManager API compatibility
- Update error handling tests for new exception types

#### 1.3 Migrate Warmup Service Tests - PRIORITY 3
**File**: `debug/old_ccxt/test_warmup_service.py` → `tests/qubx/connectors/ccxt/test_warmup_service.py`  
- Verify WarmupService API compatibility
- Update timeout and error handling tests

#### 1.4 Migrate Utils/Converter Tests - PRIORITY 4
**File**: `debug/old_ccxt/utils_test.py` → `tests/qubx/connectors/ccxt/test_utils.py`
- Pure utility functions should be identical
- Check if test data files still exist

#### 1.5 Migrate Position Handling Tests - PRIORITY 5
**File**: `debug/old_ccxt/position_test.py` → `tests/qubx/connectors/ccxt/test_position.py`
- May need significant updates if position handling changed

### Phase 2: Adapter Tests (Optional)

#### 2.1 Migrate Polling Adapter Tests (If Still Relevant)
**File**: `debug/old_ccxt/adapters/test_polling_adapter.py` → `tests/qubx/connectors/ccxt/adapters/test_polling_adapter.py`

### SKIPPED: Integration Tests
- `test_integration.py` - Architecture changed too significantly, not worth migrating

## Implementation Priority

### ✅ COMPLETED - All Migrations Successful:
1. **test_handler_factory.py** - Essential for handler registration/retrieval ⭐ - ✅ 25/25 tests passed
2. **test_connection_manager.py** - Critical for WebSocket functionality ⭐ - ✅ 19/19 tests passed  
3. **test_warmup_service.py** - Important for data initialization ⭐ - ✅ 13/13 tests passed
4. **test_utils.py** - Core data conversion functions ⭐ - ✅ 6/6 tests passed
5. **test_position.py** - Position handling logic ⭐ - ✅ 4/4 tests passed

**TOTAL: 67/67 tests migrated successfully** 🎉

## Risk Assessment

### Low Risk (Direct Migration)
- Handler Factory tests
- Utils/converter tests  
- Warmup Service tests
- Connection Manager tests

### Medium Risk (API Changes)
- Position handling tests (may have changed significantly)

## Execution Notes

1. **Gradual approach**: Migrate one test file at a time
2. **Verify each step**: Run tests after each migration
3. **Clean approach**: Keep migrations minimal and focused
4. **Update tests, not implementation**: If bugs found in implementation, ask before changing
5. **Import path updates**: All imports need updating for new package structure