# Task 021: Order Processing Flow Refactoring

## Overview

Refactor the order processing flow to fix architectural inconsistencies in async order submission, client_id handling, and order tracking. This task implements a clean pass-through pattern where `OrderRequest` objects are created by TradingManager, enriched by brokers, and tracked by account processors.

## Problem Statement

### Current Issues

1. **Inconsistent client_id handling**: Lighter broker mutates client_id, breaking health monitoring correlation
2. **Async orders not tracked**: `trade_async()` fires orders but doesn't track them in active orders
3. **Order ID migration complexity**: Lighter-specific migration logic in account processor
4. **Interface contract violations**: Brokers return different types from `send_order_async()`
5. **No clear request/order separation**: Async submissions don't have a "pending" state

### Root Causes

- Lighter broker overwrites client_id with `str(client_order_index)`, losing original tracking ID
- TradingManager doesn't add async orders to account tracking
- Account processor has exchange-specific migration logic (should be generic)
- `send_order_async()` interface returns `None`, but implementations return futures
- No distinction between order request (intent) and order (confirmed)

## Solution Architecture

### Pass-Through Pattern

```
TradingManager creates OrderRequest
         ↓
Broker enriches request.options (adds exchange-specific metadata)
         ↓
TradingManager passes to Account.process_order_request()
         ↓
Account tracks pending request
         ↓
Order update arrives via WebSocket → Account matches and moves to active
```

### Key Principles

1. **OrderRequest = Intent**: Created by caller with trading intent
2. **Broker enriches**: Adds exchange-specific metadata to `request.options`
3. **client_id never mutated**: Always preserved for health monitoring
4. **Pending tracking**: Account processor tracks requests until confirmed
5. **Generic matching**: Base class handles matching by client_id or exchange metadata

## Implementation Plan

### Phase 1: Core Data Structures

**1.1 Enhance OrderRequest**
- **File**: `src/qubx/core/basics.py`
- **Changes**:
  - Add `order_type`, `side`, `time_in_force` fields to `OrderRequest`
  - Ensure `client_id` is always required (not optional)
  - Add docstring explaining usage

**1.2 Add process_order_request to IAccountProcessor**
- **File**: `src/qubx/core/interfaces.py`
- **Changes**:
  - Add `process_order_request(request: OrderRequest) -> None` method
  - Update interface docstring

**1.3 Update IBroker.send_order_async signature**
- **File**: `src/qubx/core/interfaces.py`
- **Changes**:
  - Change signature to `send_order_async(self, request: OrderRequest) -> None`
  - Update docstring to clarify broker enrichment contract

### Phase 2: Account Processor Implementation

**2.1 BasicAccountProcessor - Add pending request tracking**
- **File**: `src/qubx/core/account.py`
- **Changes**:
  - Add `_pending_order_requests: dict[str, OrderRequest]` field
  - Add `_lighter_index_to_client_id: dict[int, str]` for Lighter matching
  - Initialize in `__init__`

**2.2 Implement process_order_request**
- **File**: `src/qubx/core/account.py`
- **Changes**:
  - Store request by client_id
  - If `lighter_client_order_index` in options, create reverse mapping
  - Add debug logging

**2.3 Implement _match_pending_request**
- **File**: `src/qubx/core/account.py`
- **Changes**:
  - Try direct client_id match first (CCXT, most exchanges)
  - Try Lighter index match (numeric client_id → original client_id)
  - Restore original client_id in order object
  - Remove from pending and cleanup mappings
  - Add debug logging

**2.4 Update process_order to use matching**
- **File**: `src/qubx/core/account.py`
- **Changes**:
  - Call `_match_pending_request(order)` for NEW/OPEN orders
  - Remove existing order if matched to pending request
  - Continue with existing merge/add logic
  - Keep health monitoring calls

**2.5 CompositeAccountProcessor - Delegate**
- **File**: `src/qubx/core/account.py`
- **Changes**:
  - Implement `process_order_request` to delegate to exchange processor

### Phase 3: TradingManager Updates

**3.1 Update trade_async to use OrderRequest**
- **File**: `src/qubx/core/mixins/trading.py`
- **Changes**:
  - Create `OrderRequest` with all fields
  - Create request before broker call (not after)
  - Pass request to `broker.send_order_async(request)`
  - Call `self._account.process_order_request(request)` after broker enriches
  - Keep health monitoring

**3.2 Implement submit_orders (future-proofing)**
- **File**: `src/qubx/core/mixins/trading.py`
- **Changes**:
  - Group requests by exchange
  - Call `broker.submit_orders(requests)` per exchange
  - Track each request via `process_order_request`
  - Return aggregated orders

### Phase 4: CCXT Broker Updates

**4.1 Update send_order_async signature**
- **File**: `src/qubx/connectors/ccxt/broker.py`
- **Changes**:
  - Change signature to `send_order_async(self, request: OrderRequest) -> None`
  - Extract parameters from request object
  - Keep existing error handling via channel
  - Remove OrderRequest return (void now)

**4.2 Keep send_order unchanged**
- **File**: `src/qubx/connectors/ccxt/broker.py`
- **Changes**: None (already works correctly)

### Phase 5: Lighter Broker Updates

**5.1 Update send_order_async signature**
- **File**: `src/qubx/connectors/xlighter/broker.py`
- **Changes**:
  - Change signature to `send_order_async(self, request: OrderRequest) -> None`
  - Extract parameters from request object
  - Keep existing async execution pattern

**5.2 Enrich request with lighter_client_order_index**
- **File**: `src/qubx/connectors/xlighter/broker.py`
- **Changes**:
  - Compute: `client_order_index = abs(hash(request.client_id)) % (10**9)`
  - **Add to request**: `request.options["lighter_client_order_index"] = client_order_index`
  - Track in broker: `self._client_order_indices[request.client_id] = client_order_index`
  - **Never mutate request.client_id**

**5.3 Update _create_order to not return Order**
- **File**: `src/qubx/connectors/xlighter/broker.py`
- **Changes**:
  - Remove Order object creation
  - Just submit transaction and return void
  - Order updates come via WebSocket (existing flow)

**5.4 Update send_order (sync) if needed**
- **File**: `src/qubx/connectors/xlighter/broker.py`
- **Changes**:
  - Keep existing pattern (returns Order)
  - Ensure client_id preserved in returned Order

**5.5 Implement submit_orders (batch)**
- **File**: `src/qubx/connectors/xlighter/broker.py`
- **Changes**:
  - Iterate requests, enrich each with `lighter_client_order_index`
  - Build batch transaction
  - Submit via `send_batch_tx`
  - Return empty list (orders come via WebSocket)

### Phase 6: Lighter Account Processor Cleanup

**6.1 Remove process_order override**
- **File**: `src/qubx/connectors/xlighter/account.py`
- **Changes**:
  - Delete entire `process_order()` override method
  - Use base class implementation (now has proper matching)

**6.2 Verify existing flows**
- **File**: `src/qubx/connectors/xlighter/account.py`
- **Changes**: None (just verification)
  - Ensure order updates via WebSocket still work
  - Ensure parsers extract order fields correctly

### Phase 7: Testing & Validation

**7.1 Unit tests for OrderRequest tracking**
- **File**: `tests/qubx/core/test_account.py`
- **Tests**:
  - `test_process_order_request_stores_pending`
  - `test_match_pending_request_by_client_id`
  - `test_match_pending_request_by_lighter_index`
  - `test_order_confirmation_removes_pending`

**7.2 Unit tests for TradingManager async flow**
- **File**: `tests/qubx/core/test_trading_manager.py`
- **Tests**:
  - `test_trade_async_creates_and_tracks_request`
  - `test_trade_async_preserves_client_id`
  - `test_submit_orders_batch`

**7.3 Integration test for Lighter flow**
- **File**: `tests/integration/test_lighter_order_flow.py`
- **Tests**:
  - `test_lighter_async_order_submission`
  - `test_lighter_client_order_index_matching`
  - `test_lighter_preserves_client_id`

**7.4 Integration test for CCXT flow**
- **File**: `tests/integration/test_ccxt_order_flow.py`
- **Tests**:
  - `test_ccxt_async_order_submission`
  - `test_ccxt_client_id_preserved`

**7.5 Manual testing**
- Test async orders on Lighter testnet
- Test async orders on CCXT sandbox
- Verify health monitoring metrics
- Verify order cancellation works with pending requests

## Benefits

1. ✅ **Client ID sanctity**: Never mutated, always traceable
2. ✅ **Consistent tracking**: All orders tracked from submission
3. ✅ **Health monitoring**: Proper correlation via preserved client_id
4. ✅ **Generic matching**: Base class handles all exchanges
5. ✅ **Clear separation**: OrderRequest (intent) vs Order (confirmed)
6. ✅ **Future-proof**: Easy to add batch async submissions
7. ✅ **Simplified code**: Remove exchange-specific overrides

## Success Criteria

- [ ] All async orders tracked in pending requests
- [ ] Lighter orders match correctly via client_order_index
- [ ] CCXT orders match correctly via client_id
- [ ] Health monitoring shows correct request/response correlation
- [ ] No client_id mutation anywhere in codebase
- [ ] Order cancellation works for both pending and active orders
- [ ] All tests pass
- [ ] Manual testing on testnet successful

## Rollback Plan

If issues arise:
1. Revert interface changes to `IBroker.send_order_async`
2. Restore Lighter account processor override
3. Keep current async flow (fire-and-forget)
4. Document issues for future attempt

## Notes

- **Breaking change**: `IBroker.send_order_async` signature changes from parameters to `OrderRequest`
- **Internal API**: Only TradingManager calls brokers, so impact is contained
- **Lighter-specific**: The `lighter_client_order_index` matching is generic enough (base class handles it)
- **Order status**: Use existing "NEW" status for pending exchange confirmation (no new status needed)

## Related Files

### Core Framework
- `src/qubx/core/basics.py` - OrderRequest, Order dataclasses
- `src/qubx/core/interfaces.py` - IBroker, IAccountProcessor interfaces
- `src/qubx/core/account.py` - BasicAccountProcessor, CompositeAccountProcessor
- `src/qubx/core/mixins/trading.py` - TradingManager

### Connectors
- `src/qubx/connectors/ccxt/broker.py` - CCXT broker implementation
- `src/qubx/connectors/ccxt/account.py` - CCXT account processor
- `src/qubx/connectors/xlighter/broker.py` - Lighter broker implementation
- `src/qubx/connectors/xlighter/account.py` - Lighter account processor
- `src/qubx/connectors/xlighter/parsers.py` - Lighter order parsing

### Tests
- `tests/qubx/core/test_account.py`
- `tests/qubx/core/test_trading_manager.py`
- `tests/integration/test_lighter_order_flow.py`
- `tests/integration/test_ccxt_order_flow.py`

## Implementation Order

Follow phases sequentially:
1. Phase 1: Data structures (safe, no behavior change)
2. Phase 2: Account processor (core logic)
3. Phase 3: TradingManager (connects the pieces)
4. Phase 4: CCXT broker (simpler exchange)
5. Phase 5: Lighter broker (complex exchange)
6. Phase 6: Cleanup (remove old code)
7. Phase 7: Testing (validation)

Each phase should be completed and tested before moving to the next.
