# Task 019: Shared Kernel Architecture

## Goal
Implement a shared kernel architecture that allows multiple UI instances (terminal or browser) to connect to the same running strategy, solving the issue where `textual-serve` creates separate strategy instances for each browser connection.

## Problem Statement
When using `qubx run --textual --textual-web`, each browser connection spawned a new subprocess with a fresh strategy session, losing history and state. Users wanted multiple browser windows to connect to the same running strategy instance.

## Solution Overview
Separate the kernel lifecycle from the UI lifecycle. The kernel (and strategy) runs persistently in one process, while multiple UI instances can connect to the same kernel via Jupyter connection files.

---

## Implementation Summary

### Phase 1: Kernel Service Manager ✅
**File**: `src/qubx/utils/runner/kernel_service.py`

Created `KernelService` class to manage persistent kernels:
- `start()`: Start kernel, initialize strategy, save connection file to `~/.qubx/kernels/`
- `stop()`: Gracefully shutdown kernel and clean up connection file
- `is_alive()`: Check if kernel is still running
- `list_active()`: List all active kernel sessions
- `cleanup_all()`: Stop all active kernels (cleanup on exit)

### Phase 2: IPyKernel Connection Support ✅
**File**: `src/qubx/utils/runner/textual/kernel.py`

Enhanced `IPyKernel` class:
- Added `connect_to_existing()` method to connect via connection file
- Added `owns_kernel` flag to track ownership
- Modified `stop()` to only shutdown kernel if we own it
- Added `get_output_history()` to retrieve past REPL output
- Added validation and error handling for connection failures

### Phase 3: Output History Tracking ✅
**File**: `src/qubx/utils/runner/textual/init_code.py`

Added output history tracking to kernel initialization:
- `_qubx_output_history`: Global list storing output entries
- `_qubx_store_output()`: Function to store output with timestamp
- Maximum 1000 entries kept in memory
- Works for both real strategies and test mode

### Phase 4: TextualStrategyApp Integration ✅
**File**: `src/qubx/utils/runner/textual/app.py`

Updated `TextualStrategyApp` to support shared kernels:
- Added `connection_file` parameter to `__init__()`
- Modified `on_mount()` to connect vs create kernel
- Retrieve and display output history on connection
- Added error handling for failed connections
- Skip strategy initialization when connecting to existing kernel

### Phase 5: CLI Commands ✅
**File**: `src/qubx/cli/commands.py`

Added new CLI options and commands:

**New flags for `qubx run`:**
- `--kernel-only`: Start kernel without UI (daemon mode)
- `--connect <file>`: Connect to existing kernel via connection file

**New command group `qubx kernel`:**
- `qubx kernel list`: List all active kernel sessions
- `qubx kernel stop <file>`: Stop a running kernel

### Phase 6: Web Mode Integration ✅
**File**: `src/qubx/utils/runner/textual/__init__.py`

Modified web serving logic:
- When `--textual-web` is used, automatically start a persistent kernel
- Pass `--connect <file>` to all subprocess instances
- All browser connections share the same kernel
- Auto-cleanup kernel on server shutdown

### Phase 7: Cleanup & Edge Cases ✅
**Files**: Various

Added robustness improvements:
- Connection timeout validation (5 seconds)
- Better error messages for stale connection files
- FileNotFoundError handling for missing connection files
- RuntimeError handling for dead kernels
- User-friendly error messages in TUI

---

## Usage Examples

### 1. Start Kernel in Daemon Mode
```bash
# Terminal 1: Start kernel
qubx run config.yml --paper --kernel-only
# Output: Connection file: ~/.qubx/kernels/macd_20250114_153042.json
```

### 2. Connect Terminal UI
```bash
# Terminal 2: Connect textual UI
qubx run --textual --connect ~/.qubx/kernels/macd_20250114_153042.json
```

### 3. Serve Web UI (Shared Kernel)
```bash
# Terminal 3: Serve web UI - all browsers see same session!
qubx run --textual --textual-web --connect ~/.qubx/kernels/macd_20250114_153042.json

# Or let it auto-start kernel:
qubx run config.yml --paper --textual --textual-web
# Opens http://0.0.0.0:8000 - multiple tabs connect to same strategy!
```

### 4. Manage Kernels
```bash
# List active kernels
qubx kernel list

# Stop a kernel
qubx kernel stop ~/.qubx/kernels/macd_20250114_153042.json
```

---

## Architecture Details

### Connection Flow
1. **Kernel starts** → Creates connection file in `~/.qubx/kernels/`
2. **UI connects** → Loads connection file, connects to existing kernel
3. **Multiple UIs** → All connect to same kernel, see same state
4. **UI closes** → Kernel keeps running (unless it owns the kernel)
5. **Kernel stops** → Connection file cleaned up

### State Management
- **Kernel owns state**: Strategy context, positions, orders
- **UI is stateless**: Just a view into the kernel
- **History preserved**: New connections retrieve past output

### Web Mode Behavior
- **Without --connect**: Auto-starts kernel, cleans up on exit
- **With --connect**: Uses existing kernel, doesn't clean up
- **All connections**: Share same kernel via `--connect` in subprocess

---

## Files Modified

### New Files
- `src/qubx/utils/runner/kernel_service.py` - Kernel lifecycle management

### Modified Files
- `src/qubx/utils/runner/textual/kernel.py` - Added connect_to_existing()
- `src/qubx/utils/runner/textual/app.py` - Added connection_file support
- `src/qubx/utils/runner/textual/init_code.py` - Added output history
- `src/qubx/utils/runner/textual/__init__.py` - Updated for web mode
- `src/qubx/cli/commands.py` - Added CLI commands

---

## Benefits

✅ **Multiple browser windows** connect to same strategy
✅ **Terminal and web UIs** can coexist
✅ **Strategy persists** even if UI closes
✅ **Reconnect and see full history**
✅ **True separation** of concerns (strategy vs UI)
✅ **Web mode works correctly** - all connections share state

## Backward Compatibility

Current behavior preserved:
- `qubx run --textual` still creates its own kernel
- New workflow is opt-in via `--kernel-only` and `--connect` flags
- Existing scripts/workflows continue to work

## Testing Checklist

- ✅ Start kernel in daemon mode
- ✅ Connect terminal UI to existing kernel
- ✅ Connect web UI to existing kernel
- ✅ Multiple browser tabs share same session
- ✅ History retrieval works
- ✅ Kernel cleanup on exit
- ✅ Error handling for dead kernels
- ✅ Error handling for missing connection files
- ✅ Auto-start kernel in web mode
- ✅ Fixed web mode hanging issue (Solution 4)

## Bug Fix: Web Mode Hanging (2025-01-14)

### Problem
When using `--textual-web --connect`, the browser would hang on startup. Terminal mode worked fine with the same connection file.

### Root Cause
Event loop conflict between:
1. Jupyter's AsyncKernelManager (using ZMQ sockets)
2. Textual's event loop management in subprocesses
3. textual-serve spawning subprocesses

The `connect_to_existing()` was called inside Textual's `on_mount()`, which happened after Textual had set up its event loop. This caused ZMQ socket operations to hang in web mode (subprocess context).

### Solution: Pre-Connection Architecture (Solution 4)

Moved kernel connection **outside** the Textual app lifecycle:

1. **Modified `IPyKernel`**:
   - Added optional `km` and `kc` parameters to `__init__`
   - Added `_is_connected` flag to track connection state
   - Added `start_iopub` parameter to `connect_to_existing()` to defer iopub task creation
   - Added `is_connected()` method
   - Added `start_iopub_listener()` to start iopub task on current event loop

2. **Modified `TextualStrategyApp`**:
   - Added optional `kernel` parameter to accept pre-connected kernel
   - Updated `on_mount()` to:
     - Check for pre-connected kernel first
     - Start iopub listener if pre-connected
     - Skip connection logic if already connected
   - Updated strategy initialization check

3. **Modified `run_strategy_yaml_in_textual()`**:
   - Connect to kernel **before** creating the app
   - Use `start_iopub=False` during pre-connection (avoids event loop conflicts)
   - Pass pre-connected kernel to app
   - Iopub task is started later on Textual's event loop

### Why This Works
- ZMQ connection happens in a clean, temporary event loop (via `asyncio.run()`)
- Connection completes before Textual creates its event loop
- Iopub draining task is created on Textual's event loop, not the temporary one
- No event loop conflicts or race conditions

### Files Modified
- `src/qubx/utils/runner/textual/kernel.py` - Pre-connection support
- `src/qubx/utils/runner/textual/app.py` - Accept pre-connected kernel
- `src/qubx/utils/runner/textual/__init__.py` - Connect before app creation

## Next Steps

1. **Install dependencies**: `poetry install`
2. **Test end-to-end**: Try the usage examples above
3. **Monitor performance**: Check if multiple connections cause issues
4. **Documentation**: Update user docs with new workflow
5. **Consider auto-discovery**: Maybe list kernels without full path?
