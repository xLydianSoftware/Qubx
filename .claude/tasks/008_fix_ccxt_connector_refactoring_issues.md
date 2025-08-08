# Task 008: Fix CCXT Connector Refactoring Issues

## Overview
Recent refactoring of the CCXT connector introduced several critical bugs and architectural issues around individual vs bulk subscription handling. This task outlines the systematic fixes needed to restore functionality while maintaining the improved design patterns.

## Phase 1: Critical Bug Fixes (Immediate)

### 1.1 Fix Connection Manager stop_stream Signature Mismatch
**Issue**: `stop_stream` method signature changed but callers weren't updated
**Files**: `connection_manager.py`, `subscription_orchestrator.py`

```python
# Current broken call in _cleanup_removed_instrument_streams:
self._loop.submit(self._connection_manager.stop_stream(stream_name, future))

# Fix: Update to match new signature
self._loop.submit(self._connection_manager.stop_stream(stream_name, subscription_type))
```

### 1.2 Implement Missing _wait Method
**Issue**: `connection_manager.py` calls undefined `_wait` method
**File**: `connection_manager.py`

```python
# Add to ConnectionManager class:
def _wait(self, future: concurrent.futures.Future, description: str, timeout: float = 5.0) -> None:
    """Wait for future completion with timeout and error handling"""
    try:
        future.result(timeout=timeout)
        logger.debug(f"[{self._exchange_id}] Completed: {description}")
    except concurrent.futures.TimeoutError:
        logger.warning(f"[{self._exchange_id}] Timeout waiting for {description}")
    except Exception as e:
        logger.error(f"[{self._exchange_id}] Error in {description}: {e}")
```

### 1.3 Fix Individual Stream Cleanup Logic
**Issue**: `execute_unsubscription` only stops main stream, not individual instrument streams
**File**: `subscription_orchestrator.py`

```python
def execute_unsubscription(self, subscription_type: str):
    """Clean up existing subscription including all individual streams."""
    stream_name = self._subscription_manager.get_subscription_stream(subscription_type)
    if not stream_name:
        return
    
    # Stop main/bulk stream
    logger.debug(f"[{self._exchange_id}] Stopping main stream for {subscription_type}")
    self._connection_manager.stop_stream(stream_name, subscription_type)
    
    # Stop individual instrument streams if they exist
    individual_streams = self._get_existing_individual_streams(subscription_type)
    for instrument_stream_name in individual_streams:
        logger.debug(f"[{self._exchange_id}] Stopping individual stream: {instrument_stream_name}")
        self._connection_manager.stop_stream(instrument_stream_name, subscription_type)
```

## Phase 2: Individual Stream State Management

### 2.1 Implement Individual Stream Tracking
**Issue**: `_get_existing_individual_streams` returns empty dict
**Files**: `subscription_orchestrator.py`, `subscription_manager.py`

**Add to SubscriptionManager:**
```python
def __init__(self):
    # ... existing code ...
    # Track individual streams per subscription type
    self._individual_streams: dict[str, dict[str, str]] = {}  # sub_type -> {instrument_id: stream_name}

def store_individual_streams(self, subscription_type: str, instrument_streams: dict[Instrument, str]) -> None:
    """Store individual stream names for instruments"""
    self._individual_streams[subscription_type] = {
        f"{inst.symbol}:{inst.exchange}": stream_name 
        for inst, stream_name in instrument_streams.items()
    }

def get_individual_streams(self, subscription_type: str) -> dict[str, str]:
    """Get individual stream names for a subscription type"""
    return self._individual_streams.get(subscription_type, {}).copy()

def clear_individual_streams(self, subscription_type: str) -> None:
    """Clear individual stream mapping"""
    self._individual_streams.pop(subscription_type, None)
```

**Update SubscriptionOrchestrator:**
```python
def _get_existing_individual_streams(self, subscription_type: str) -> dict[str, concurrent.futures.Future]:
    """Get existing individual streams with their futures"""
    stream_names = self._subscription_manager.get_individual_streams(subscription_type)
    return {
        name: self._connection_manager.get_stream_future(name) 
        for name in stream_names.values() 
        if self._connection_manager.get_stream_future(name)
    }

def _store_individual_stream_mapping(self, subscription_type: str, stream_mapping: dict[str, Any]) -> None:
    """Store individual stream mapping"""
    # Extract instrument->stream_name mapping from stream_mapping
    instrument_streams = {}
    for stream_name, future in stream_mapping.items():
        # Parse instrument from stream name (reverse engineering)
        if "_" in stream_name:
            symbol_part = stream_name.split("_")[-1].replace("_", "/")
            # Find matching instrument from subscription
            instruments = self._subscription_manager.get_subscribed_instruments(subscription_type)
            for inst in instruments:
                if inst.symbol.replace("/", "_") in stream_name:
                    instrument_streams[inst] = stream_name
                    break
    
    self._subscription_manager.store_individual_streams(subscription_type, instrument_streams)
```

### 2.2 Fix Subscription State Tracking for Individual Streams
**Issue**: Individual streams all mark same subscription as active
**File**: `connection_manager.py`

```python
async def listen_to_stream(self, ...):
    # ... existing code ...
    
    # Mark subscription as active on first successful data reception
    if not connection_established and self._subscription_manager:
        # For individual streams, track per-instrument connection state
        if "_" in stream_name and stream_name != subscription_type:
            # This is an individual instrument stream
            # Don't mark whole subscription active until we have enough instruments connected
            pass  # TODO: Implement partial connection tracking
        else:
            # Bulk stream - mark subscription active immediately
            self._subscription_manager.mark_subscription_active(subscription_type)
        connection_established = True
```

## Phase 3: Architectural Improvements

### 3.1 Make ConnectionManager.stop_stream Async and Pure
**Issue**: `stop_stream` should be async and only handle stream lifecycle, not subscription state
**File**: `connection_manager.py`

```python
async def stop_stream(self, stream_name: str) -> None:
    """Stop a stream - pure stream lifecycle management"""
    # Disable stream
    self._is_stream_enabled[stream_name] = False
    
    # Run unsubscriber if available (async)
    unsubscriber = self._stream_to_unsubscriber.get(stream_name)
    if unsubscriber:
        try:
            await unsubscriber()
        except Exception as e:
            logger.error(f"Error in unsubscriber for {stream_name}: {e}")
    
    # Clean up stream state only
    self._stream_to_unsubscriber.pop(stream_name, None)
    self._stream_to_coro.pop(stream_name, None)
    
    # DO NOT touch subscription manager state - that's orchestrator's job
```

**Update SubscriptionOrchestrator:**
```python
async def execute_unsubscription(self, subscription_type: str):
    # Stop streams first (async)
    await self._connection_manager.stop_stream(stream_name)
    
    # Then clean up subscription state (orchestrator responsibility)  
    self._subscription_manager.clear_subscription_state(subscription_type)
```

### 3.2 Handle Async Unsubscribers Properly
**Issue**: Blocking wait for unsubscribers can cause delays
**File**: `connection_manager.py`

```python
def stop_stream(self, stream_name: str, subscription_type: str) -> None:
    """Stop a stream gracefully"""
    self._is_stream_enabled[stream_name] = False
    
    # Get unsubscriber before cleanup
    unsubscriber = self._stream_to_unsubscriber.get(stream_name)
    
    # Clean up stream state
    self._is_stream_enabled.pop(stream_name, None)
    self._stream_to_coro.pop(stream_name, None)
    self._stream_to_unsubscriber.pop(stream_name, None)
    
    # Run unsubscriber asynchronously (don't block)
    if unsubscriber:
        unsub_task = self._loop.submit(unsubscriber())
        # Don't wait - let it complete in background
        logger.debug(f"[{self._exchange_id}] Scheduled unsubscriber for {stream_name}")
```

## Phase 4: Testing Strategy

### 4.1 Core Test Requirements
- **Bulk subscription lifecycle**: subscribe → receive data → unsubscribe → cleanup
- **Individual subscription lifecycle**: multi-instrument subscribe → partial failures → cleanup
- **Mixed subscription scenarios**: bulk + individual for same exchange
- **Resubscription handling**: change instruments → proper cleanup → new subscription
- **Error handling**: network failures → retry → recovery

### 4.2 Test Cleanup Plan
1. **Remove complex integration tests** that test multiple components together
2. **Keep unit tests** for each component (SubscriptionManager, ConnectionManager, Orchestrator)
3. **Keep basic end-to-end tests** for subscription lifecycle
4. **Remove tests for deprecated/removed functionality**

## Implementation Order

1. **Phase 1**: Fix immediate bugs (stop_stream, _wait method, cleanup logic)
2. **Phase 2**: Implement individual stream tracking properly
3. **Phase 3**: Clean up architectural issues (separation of concerns)
4. **Phase 4**: Update tests to match new design

## Success Criteria

- [ ] All existing tests pass after fixes
- [ ] No more method signature mismatches
- [ ] Individual instrument subscriptions clean up properly
- [ ] No blocking operations in stream lifecycle
- [ ] Clear separation between stream and subscription state
- [ ] Resubscription works for both bulk and individual streams

## Risk Assessment

**High Risk**: Individual stream resubscription logic is complex
**Medium Risk**: Async unsubscriber execution might leave resources
**Low Risk**: Method signature fixes are straightforward

## Dependencies

- Requires understanding of exchange-specific subscription patterns
- May need coordination with exchange-specific handler implementations
- Test refactoring depends on fixing core issues first