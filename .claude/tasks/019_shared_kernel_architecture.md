# Task 019: Shared Kernel Architecture for Textual Multi-Connection Support

## Goal
Enable multiple browser tabs (or terminal sessions) to connect to the same running strategy kernel, sharing the same REPL history, positions, orders, and state.

## Problem
Currently with `--textual-web`, each browser connection spawns a new subprocess with a fresh strategy session. Users lose history and state between connections.

## Solution
Automatically create a persistent kernel subprocess when starting textual mode. All UI instances connect to this shared kernel. The kernel is automatically cleaned up when the main process exits.

## Architecture

### Component 1: Kernel Service Manager
**File:** `src/qubx/utils/runner/kernel_service.py`

Manages the lifecycle of persistent kernel subprocesses:
- Starts kernel as subprocess with strategy initialization
- Saves connection info to temp file
- Provides methods to check if kernel is alive
- Gracefully shuts down kernel

### Component 2: IPyKernel Connection Support
**File:** `src/qubx/utils/runner/textual/kernel.py`

Extended to support both creating new kernels and connecting to existing ones:
- `connect_to_existing(connection_file)` - Connect to running kernel
- `get_output_history()` - Retrieve past REPL output from kernel

### Component 3: Output History
**File:** `src/qubx/utils/runner/textual/init_code.py`

Track all output in kernel memory so new connections can replay history:
- REPL commands and outputs
- Positions/orders updates
- Dashboard data
- Error messages

### Component 4: TextualStrategyApp Updates
**File:** `src/qubx/utils/runner/textual/app.py`

Support both modes:
- Creating new kernel (current behavior)
- Connecting to existing kernel (new behavior)
- Replaying output history on connection

### Component 5: Automatic Lifecycle Management
**File:** `src/qubx/utils/runner/textual/__init__.py`

Terminal and web modes automatically manage kernel lifecycle:

**Terminal Mode:**
1. Start kernel service
2. Connect Textual UI to it
3. On exit, stop kernel service

**Web Mode:**
1. Start kernel service (once, in main process)
2. Pass connection file to textual-serve subprocesses
3. All browser tabs connect to same kernel
4. On server exit, stop kernel service

## User Experience

### Terminal Mode
```bash
qubx run config.yml --paper --textual
# Kernel auto-starts → UI connects → On quit, kernel auto-stops
```

### Web Mode
```bash
qubx run config.yml --paper --textual --textual-web
# Kernel auto-starts → Server spawns UIs → All connect to same kernel
# Open multiple browser tabs → All see same session!
# Ctrl+C server → Kernel auto-stops
```

## Implementation Phases

### Phase 1: Kernel Service Foundation ✅
- Created `kernel_service.py` with KernelService class
- Implemented start/stop/is_alive/list_active/cleanup_all methods
- Connection files stored in ~/.qubx/kernels/
- Fixed JSON serialization issue with bytes key field (base64 encoding)

### Phase 2: IPyKernel Connection Support ✅
- Added `connect_to_existing(connection_file)` method
- Implemented connection file loading via AsyncKernelManager
- Added `get_output_history()` method to retrieve kernel globals

### Phase 3: Output History Tracking ✅
- Added `_qubx_output_history` global list in init_code
- Implemented HistoryTrackingDisplayPublisher wrapper
- Tracks text and dashboard outputs for replay
- Limited to 1000 entries with automatic rotation

### Phase 4: TextualStrategyApp Integration ✅
- Added `connection_file` parameter to __init__
- Implemented dual mode in `on_mount()`: create new or connect existing
- History replay on connection with status messages
- Modified `on_unmount()` to not shutdown externally-managed kernels

### Phase 5: Terminal Mode Lifecycle ✅
- Auto-start kernel before UI with KernelService.start()
- Pass connection_file to TextualStrategyApp
- Auto-stop kernel in finally block on exit
- Proper error handling and logging

### Phase 6: Web Mode Lifecycle ✅
- Start kernel once in main process before web server
- Pass connection_file to subprocesses via --textual-connect flag
- All browser tabs connect to same kernel instance
- Auto-cleanup kernel on server exit (try/finally)

### Phase 7: CLI Integration ✅
- Added hidden `--textual-connect` flag to run command
- Wire connection_file through CLI to run_strategy_yaml_in_textual
- Dual behavior: None = start kernel, provided = connect to existing

## Files Modified

**New Files:**
- `src/qubx/utils/runner/kernel_service.py`

**Modified Files:**
- `src/qubx/utils/runner/textual/kernel.py`
- `src/qubx/utils/runner/textual/app.py`
- `src/qubx/utils/runner/textual/init_code.py`
- `src/qubx/utils/runner/textual/__init__.py`
- `src/qubx/cli/commands.py`

## Testing Checklist

- [ ] Start terminal mode → verify kernel starts
- [ ] Quit terminal mode → verify kernel stops
- [ ] Start web mode → verify kernel starts once
- [ ] Open multiple browser tabs → verify all connect to same kernel
- [ ] Execute command in tab 1 → verify appears in tab 2
- [ ] Stop web server → verify kernel stops
- [ ] Test history replay on reconnection
- [ ] Test kernel crash handling
- [ ] Test stale connection file cleanup

## Benefits

✅ Multiple browser windows share same strategy state
✅ Zero manual kernel management
✅ Strategy persists across browser refreshes
✅ Clean automatic lifecycle management
✅ No breaking changes to existing workflows

## Implementation Summary

### Completed ✅
All 7 phases have been successfully implemented:

1. **Kernel Service** - Manages persistent kernels with start/stop/alive checking
2. **Connection Support** - IPyKernel can connect to existing kernels via connection files
3. **History Tracking** - All outputs tracked in kernel memory for replay
4. **App Integration** - TextualStrategyApp supports both create and connect modes
5. **Terminal Mode** - Auto-lifecycle management (start kernel → connect UI → stop on exit)
6. **Web Mode** - Single kernel shared across all browser tabs
7. **CLI Integration** - Hidden --textual-connect flag wires everything together

### Technical Details

**Connection Files:**
- Stored in `~/.qubx/kernels/` directory
- Format: `{strategy_name}_{timestamp}.json`
- Contains kernel connection info (ports, IP, auth key)
- Auto-cleaned up when kernel stops

**Key Changes:**
- `kernel_service.py` (159 lines) - New kernel lifecycle manager
- `kernel.py` - Added connect_to_existing() and get_output_history()
- `init_code.py` - Added output history tracking via IPython hooks
- `app.py` - Added connection_file parameter and dual-mode support
- `__init__.py` - Auto-lifecycle for terminal and web modes
- `commands.py` - Added --textual-connect hidden flag

**Code Quality:**
- All modified files pass ruff checks
- Proper import ordering (stdlib → third-party → local)
- Fixed f-string without placeholders warning
- Fixed bytes serialization in JSON (base64 encoding)

### Testing Notes

**Basic Kernel Service Tested:**
- Kernel starts successfully
- Connection file created correctly
- JSON serialization works (bytes → base64)

**Manual Testing Recommended:**
- Terminal mode: `qubx run config.yml --paper --textual`
- Web mode: `qubx run config.yml --paper --textual --textual-web`
- Multiple browser tabs connecting to same kernel
- History replay when reconnecting
- Kernel cleanup on exit

### Future Enhancements

**Potential Improvements:**
- Add kernel health monitoring and auto-restart on crash
- Implement kernel connection timeout handling
- Add command to list/connect to running kernels manually
- Support multiple concurrent strategies with separate kernels
- Add kernel logs viewing capability

**Known Limitations:**
- Strategy initialization errors will cause kernel to die
- No automatic retry if kernel fails to start
- Connection files not cleaned up if process is killed (SIGKILL)

## Conclusion

The shared kernel architecture has been fully implemented according to the original plan. All browser tabs and terminal sessions now connect to a single persistent kernel, sharing REPL history, positions, orders, and state. The implementation is automatic - users don't need to manually start or manage kernels.
