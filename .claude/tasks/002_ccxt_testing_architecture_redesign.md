# Task 002: CCXT Testing Architecture Redesign

## Task Overview

Redesign the CCXT data provider testing architecture to match the new component separation. Replace the old monolithic `subscription_lifecycle_test.py` with focused component tests and proper integration tests.

## Current Problem

- Old monolithic `subscription_lifecycle_test.py` tests everything through the facade
- Tests are brittle and use compatibility properties that were removed
- No separation between unit tests and integration tests
- Hard to test individual components in isolation

## New Test Architecture

### 1. Component Unit Tests (Fast, Isolated)

Tests go in `tests/qubx/connectors/ccxt/` (existing structure, no unit subfolder):

#### A. Core Components

- `test_subscription_manager.py`
  - Test subscription state transitions
  - Test subscription grouping and management
  - Test resubscription cleanup logic
  - Mock dependencies, focus on state management logic

- `test_connection_manager.py`
  - Test WebSocket connection handling
  - Test retry logic and error handling  
  - Test stream lifecycle management
  - Mock WebSocket connections and exchanges

- `test_subscription_orchestrator.py`
  - Test coordination between subscription manager and connection manager
  - Test complex resubscription scenarios
  - Test cleanup and cancellation logic
  - Mock both dependencies

- `test_warmup_service.py`
  - Test warmup task coordination
  - Test error handling and timeouts
  - Test grouping by data type and period
  - Mock handlers and async execution

#### B. Data Type Handlers

- `test_handler_factory.py`
  - Test handler registration and retrieval
  - Test handler caching
  - Test error cases (unknown data types)

- Individual handler tests:
  - `test_ohlc_handler.py`
  - `test_trade_handler.py`  
  - `test_orderbook_handler.py`
  - `test_quote_handler.py`
  - `test_liquidation_handler.py`
  - `test_funding_rate_handler.py`
  - `test_open_interest_handler.py`

Each handler test:

- Test `prepare_subscription()` returns correct SubscriptionConfiguration
- Test `warmup()` with various parameters
- Mock exchange and data provider dependencies
- Test data conversion and processing logic
- Test error handling (parsing errors, missing data)

### 2. Integration Tests (Slower, Real Dependencies)

#### A. Component Integration

- `test_ccxt_data_provider_integration.py`
  - Test facade coordination between all components
  - Test end-to-end subscription flows
  - Test complex scenarios (multiple data types, resubscriptions)
  - Use mock exchange but real component integration
  - Mark with `@pytest.mark.integration`

#### B. Real Exchange Tests

- `test_binance_um_swap_integration.py`
  - Test with real Binance UM swap exchange
  - Test all data types: OHLC, trades, orderbook, quotes, funding rates, open interest
  - Test subscription lifecycle: subscribe → data flow → unsubscribe
  - Test warmup functionality
  - Test error handling (network issues, invalid symbols)
  - Test edge cases: rapid resubscriptions, instrument changes
  - Mark as `@pytest.mark.integration` and require API credentials

### 3. Test Utilities

#### Common Fixtures

- Update `tests/qubx/connectors/ccxt/conftest.py`
  - Shared fixtures for mock exchanges
  - Mock instruments and data
  - Common test utilities
  - Component fixtures for isolated testing

#### Test Data

- Create `tests/qubx/connectors/ccxt/fixtures/`
  - Sample CCXT data responses
  - Common test instruments
  - Error scenarios data

## Implementation Plan

### Phase 0: Architectural Cleanup ✅ COMPLETED

**Fixed CcxtDataProvider architectural issues before implementing tests:**

1. **✅ COMPLETED** Removed redundant `_listen_to_stream()` delegation wrapper
   - Method was just forwarding calls to ConnectionManager
   - Handlers can use ConnectionManager directly through orchestrator

2. **✅ COMPLETED** Moved `_call_by_market_type()` logic to utility function
   - Created `create_market_type_batched_subscriber()` in utils.py
   - Updated all 5 handlers (trade, liquidation, orderbook, ohlc, quote) to use utility
   - Added convenience wrapper in SubscriptionOrchestrator for backward compatibility

3. **✅ COMPLETED** Removed confusing stop methods
   - Eliminated `_stop_subscriber()` and `_stop_old_subscriber()` delegation wrappers
   - These were redundant facades that just called orchestrator/connection manager
   - Simplified the interface and removed unnecessary indirection

4. **✅ COMPLETED** Removed `_mark_subscription_active()` delegation wrapper
   - ConnectionManager already calls SubscriptionManager directly
   - Another redundant facade method eliminated

5. **✅ COMPLETED** Cleaned up imports and reduced CcxtDataProvider complexity
   - Removed ~50 lines of redundant delegation code
   - Cleaner separation of concerns between facade and components

### Phase 1: Core Component Tests ✅ IN PROGRESS

1. **✅ IN PROGRESS** Create `test_subscription_manager.py` - Test state management in isolation
   - Test subscription state transitions (pending → active)
   - Test subscription grouping and management
   - Test resubscription cleanup logic
   - Mock dependencies, focus on state management logic

2. **✅ PENDING** Create `test_connection_manager.py` - Test WebSocket connection handling
   - Test stream lifecycle management
   - Test retry logic and error handling
   - Mock WebSocket connections and exchanges

3. **✅ PENDING** Create `test_subscription_orchestrator.py` - Test component coordination
   - Test coordination between subscription manager and connection manager
   - Test complex resubscription scenarios
   - Test cleanup and cancellation logic

4. **✅ PENDING** Create `test_warmup_service.py` - Test warmup orchestration
   - Test warmup task coordination
   - Test error handling and timeouts
   - Test grouping by data type and period

### Phase 2: Handler Tests ✅ PENDING  

1. **✅ PENDING** Create `test_handler_factory.py` - Test handler registration/creation
   - Test handler registry and retrieval
   - Test handler caching mechanism
   - Test error cases (unknown data types)

2. **✅ PENDING** Create individual handler tests for all 7 data types
   - Each handler test should validate `prepare_subscription()` and `warmup()` methods
   - Test SubscriptionConfiguration creation
   - Test data conversion and processing logic
   - Test error handling (parsing errors, missing data)
   - Mock exchange and data provider dependencies

### Phase 3: Integration Tests ✅ PENDING

1. **✅ PENDING** Create facade integration tests with mock exchange
   - Test end-to-end subscription flows
   - Test complex scenarios (multiple data types, resubscriptions)
   - Use mock exchange but real component integration

2. **✅ PENDING** Create real Binance UM swap tests (require API credentials)
   - Test all data types with real exchange
   - Test subscription lifecycle and error handling
   - Mark as `@pytest.mark.integration`

### Phase 4: Cleanup ✅ PENDING

1. **✅ PENDING** Delete old `subscription_lifecycle_test.py`
2. **✅ PENDING** Update any remaining tests that use old patterns
3. **✅ PENDING** Update CI/CD to run new test structure

## Benefits of New Architecture

1. **Fast Feedback**: Component tests run quickly, identify issues immediately
2. **Isolation**: Each component tested independently  
3. **Reliability**: Less flaky than large integration tests
4. **Coverage**: Better test coverage of edge cases
5. **Maintainability**: Tests match code architecture
6. **Real-world Validation**: Integration tests ensure everything works together

## Testing Strategy Notes

- **Unit Tests**: Focus on testing individual components with mocked dependencies
- **Integration Tests**: Test component interaction and real exchange functionality
- **Use existing test structure**: `tests/qubx/` is already the unit test location
- **Mark integration tests**: Use `@pytest.mark.integration` for slower tests
- **Test isolation**: Each test should be independent and able to run in any order
- **Mock external dependencies**: Unit tests should not depend on real exchanges or network

## Progress Tracking

- [ ] Phase 1: Core component tests
- [ ] Phase 2: Handler tests  
- [ ] Phase 3: Integration tests
- [ ] Phase 4: Cleanup old tests

## Success Criteria

1. All components have comprehensive unit tests
2. Integration tests validate end-to-end functionality
3. Real exchange tests work with Binance UM swap
4. Test suite runs faster than before
5. Better test coverage and maintainability
6. Easy to add new components and handlers
