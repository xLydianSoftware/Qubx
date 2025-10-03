# Task 010: Fix Binance.UM Bulk Mode Resubscription Issue

## Problem Description
After recent CCXT connector refactoring, Binance.UM bulk subscription mode has an issue with resubscriptions. When changing universe and subscribing to new instruments, OHLC(1h) data stops being received after resubscription.

## Key Differences
- **Binance.UM**: Supports bulk subscription mode (multiple data types in one subscription)
- **Hyperliquid.F**: Does NOT support bulk mode (individual streams only)

## Investigation Plan

### 1. Reproduce the Issue
- [ ] Run the notebook `examples/notebooks/1.1 Run binance um paper mode.ipynb`
- [ ] Observe behavior when changing universe and resubscribing
- [ ] Verify that OHLC(1h) data stops after resubscription

### 2. Create Integration Test
- [ ] Create `tests/integration/connectors/ccxt/test_binance_um_swap_integration.py`
- [ ] Test bulk subscription mode
- [ ] Test resubscription scenarios
- [ ] Verify data continues flowing after resubscription

### 3. Debug the Issue
- [ ] Examine subscription orchestrator behavior in bulk mode
- [ ] Check how existing subscriptions are handled during resubscription
- [ ] Verify websocket connection state management
- [ ] Check if handlers are properly maintained after resubscription

### 4. Fix Implementation
- [ ] Identify root cause in bulk mode handling
- [ ] Implement fix
- [ ] Verify fix with integration test

## Files to Examine
- `src/qubx/connectors/ccxt/subscription_orchestrator.py`
- `src/qubx/connectors/ccxt/connection_manager.py`
- `src/qubx/connectors/ccxt/data.py`
- `examples/notebooks/1.1 Run binance um paper mode.ipynb`

## Progress Log

### 2025-08-07 - Initial Investigation
- Created task plan
- Starting investigation of bulk mode resubscription issue
- Created integration test `test_bulk_mode_resubscription_ohlc_continues` in `test_binance_um_swap_integration.py`
- Test PASSES - OHLC data continues after resubscription in the test environment
- However, notebook shows an issue with OHLC data stopping after resubscription

### Findings
1. When universe changes in bulk mode:
   - Old stream is cancelled (e.g., `ohlc(1h):2606c9`)
   - New stream is created (e.g., `ohlc(1h):cce8f5`)
   - Stream hash changes because instruments are different

2. The integration test shows data continues flowing
3. The notebook example shows data stops

### Hypothesis - CONFIRMED
The issue is related to:
1. **CCXT stream reuse**: CCXT uses a single stream for all OHLCV subscriptions (`'multipleOHLCV'`)
2. **State conflicts**: During resubscription, CCXT sends UNSUBSCRIBE then SUBSCRIBE, but there are timing/state issues
3. **Stream cleanup**: The old stream state is not properly cleaned up, causing UnsubscribeError

### Root Cause - CONFIRMED
- CCXT uses `self.stream(type, 'multipleOHLCV')` - same stream hash for all OHLCV subscriptions
- When resubscribing, the old symbols are unsubscribed and new symbols subscribed on the same WebSocket
- But there are race conditions/state conflicts that prevent the new subscription from working

### Solution Implemented
- Added enhanced `watch_ohlcv_for_symbols` override in BinanceQV exchange
- Catches UnsubscribeError during resubscription
- Forces cleanup of problematic stream state
- Retries subscription with clean state
- Includes small delay to allow state to settle

### Status
- Integration test passes: OHLC data continues after resubscription ✅
- Simple script test still fails: No OHLC data received at all ❌
- Need to investigate why simple script differs from pytest environment