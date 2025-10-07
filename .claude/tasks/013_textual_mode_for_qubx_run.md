# Task 013: Add --textual mode to qubx run command

**Status**: Completed
**Created**: 2025-10-06
**Completed**: 2025-10-06
**Priority**: Medium

## Objective

Add a `--textual` flag to the `qubx run` command that launches a Textual-based TUI for running strategies with live Jupyter kernel interaction and log viewing.

## Requirements

### Phase 1: Basic Implementation
- ✅ Create Textual app with two main panels:
  - Left panel: REPL output (kernel interaction)
  - Right panel: Application logs (qubx logger output)
  - Bottom: Input bar for sending Python code to kernel
- ✅ Use AsyncKernelManager for kernel management (compatible with Textual)
- ✅ Integrate with strategy context like Jupyter mode
- ✅ Add --textual flag to CLI

### Phase 2: Future Enhancements (Optional)

- ✅ Add positions DataTable
- [ ] Add order book DataTable
- [ ] Add bid/ask prices panel

## Implementation Details

### Files Created
- `src/qubx/utils/runner/textual_runner.py` - Main Textual app implementation

### Files Modified
- `src/qubx/cli/commands.py` - Added --textual flag to run command

### Key Components

1. **IPyKernel class**: Wrapper around AsyncKernelManager
   - Manages kernel lifecycle (start/stop)
   - Handles iopub message draining
   - Executes code asynchronously
   - Supports kernel interruption

2. **TextualStrategyApp**: Main Textual application
   - Layout: Horizontal split (REPL | Logs)
   - Input bar at bottom
   - Key bindings: Enter (run), Ctrl+L (clear REPL), Ctrl+C (interrupt), Q (quit)

3. **run_strategy_yaml_in_textual()**: Entry point function
   - Similar to run_strategy_yaml_in_jupyter
   - Initializes strategy context
   - Launches Textual app with kernel

## Testing

Test with:
```bash
poetry run qubx run config.yml --paper --textual
```

## Dependencies

All dependencies already satisfied:
- textual: ^0.88.0
- jupyter-client: 8.6.3
- nest_asyncio (for nested event loops)

## References

- Sample Textual Jupyter console provided by user
- Existing Jupyter mode implementation in runner.py
- Existing TUI implementation in cli/tui.py (backtest browser)

## Notes

- Started simple with logs + REPL interaction
- Can expand later to include positions, order book, bid/ask prices
- Uses async/await throughout for Textual compatibility
- Logger integration via Rich logging handler

## Completion Summary

Successfully implemented `--textual` mode for `qubx run` command:

✅ **Files Created**:
- `src/qubx/utils/runner/textual_runner.py` (~340 lines, simplified)
  - IPyKernel class for kernel management
  - TextualStrategyApp with single-panel layout
  - run_strategy_yaml_in_textual() entry point

✅ **Files Modified**:
- `src/qubx/cli/commands.py`
  - Added --textual/-t flag
  - Added mutual exclusion check with --jupyter
  - Integrated textual runner import and call

✅ **Testing**:
- Tested with `examples/macd_crossover/config.yml --paper --textual`
- Verified TUI renders correctly with clean layout
- Confirmed strategy initialization and output streaming works
- Verified kernel starts and accepts commands via Enter key

**Usage**:
```bash
poetry run qubx run config.yml --paper --textual
```

**Key Features**:
- Single output window: All REPL output, logs, and print statements in one scrollable area
- Bottom: Input bar for executing Python code (press Enter to execute)
- Key bindings: Ctrl+L (clear output), Ctrl+C (interrupt kernel), Q (quit)
- Helper functions available: ctx, S, portfolio(), orders(), trade(), exit()
- Clean rendering without artifacts

## Bug Fixes Applied

Fixed several issues after initial implementation:

### Round 1 - Core Functionality
1. **Async/Await Errors**:
   - Fixed `kc.start_channels()` - not async (removed await)
   - Fixed `kc.stop_channels()` - not async (removed await)
   - Fixed `kc.execute()` - not async (removed await)

2. **Enter Key Not Working**:
   - Removed manual `action_send_code` binding for Enter key
   - Implemented `on_input_submitted()` event handler instead
   - Textual's Input widget handles Enter key via Submitted event

3. **Debug Message Clutter**:
   - Added `comm_close` to ignored message types
   - Silenced debug messages in kernel event handler

### Round 2 - UI/UX Improvements
4. **Black Square Rendering Artifacts**:
   - Removed `Panel.fit()` which caused rendering conflicts
   - Simplified Title widget to use plain styled text
   - Added explicit borders and backgrounds to containers

5. **Simplified Layout** (user request):
   - Removed dual-panel layout (was confusing/mostly empty)
   - Single output window for all content (REPL + logs combined)
   - Removed complex logger handler setup
   - Cleaned up ~30 lines of unnecessary code
   - Removed unused imports and widgets

**Final Result**: Clean, simple TUI with single output window, responsive input, and reliable rendering - similar to traditional REPL experience.

### Round 3 - Focus Rendering Artifacts Fix

6. **Input Focus Rendering Artifacts** (2025-10-07):
   - Added explicit `:focus` pseudo-class styling for Input widget to prevent default focus effects
   - Added `:focus-within` pseudo-class for output container to prevent border redraws
   - Added `scrollbar-gutter: stable` to RichLog to prevent layout shifts
   - Added border styling to Input widget to match theme
   - Implemented `on_input_changed()` handler to limit refresh scope and prevent full screen redraws

   **Issue**: When clicking on the input widget, cyan/green colored artifacts appeared on the right side
   **Root cause**: Focus state changes triggered full screen redraws and RichLog auto-scroll conflicts
   **Solution**: Explicit CSS styling + limited refresh scope in event handlers

   **Note**: Running the textual app directly in Claude Code breaks the terminal output. Ask user to verify fixes manually instead.

### Round 4 - Positions Panel (2025-10-07):

7. **Positions Panel with "P" Key Toggle**:
   - Added DataTable widget for displaying live positions
   - Implemented horizontal split layout (output left, positions right when visible)
   - Added "p" key binding to toggle positions panel visibility
   - Positions update every 1 second when panel is visible
   - Uses custom MIME type (`application/x-qubx-positions+json`) for IPC between kernel and Textual
   - Kernel-side helper functions:
     - `_positions_as_records()`: Converts ctx.positions to JSON-serializable format
     - `emit_positions()`: Publishes positions via IPython display with custom MIME type
   - Textual-side handling:
     - `IPyKernel._drain_iopub()`: Detects custom MIME and emits `qubx_positions` event
     - `_request_positions()`: Interval timer calls `emit_positions()` in kernel
     - `_update_positions_table()`: Updates DataTable with received positions data
   - Positions sorted by absolute market value (descending)
   - Shows: Symbol, Side (LONG/SHORT/FLAT), Qty, Avg Px, Last Px, PnL, Mkt Value

   **Implementation approach**: Since the strategy context (`ctx`) runs in a separate Jupyter kernel process, direct access is not possible. Used custom MIME type over IOPub channel for structured data exchange between kernel and Textual app.

   **Key benefits**:
   - Clean separation of concerns (kernel manages data, UI displays it)
   - Leverages existing Jupyter messaging infrastructure
   - Extensible pattern for future widgets (orders, order book, etc.)
   - Minimal performance impact (updates only when panel is visible)

8. **UI Polish and Rendering Fixes**:
   - Added margin around input widget (`margin: 1 2`) for better visual separation from footer
   - Added `scrollbar-gutter: stable` to DataTable and RichLog to prevent layout shifts
   - Removed unnecessary `on_input_changed` handler to prevent rendering artifacts
   - Optimized DataTable updates to only refresh when data changes
   - Added NaN/Inf handling in position data to prevent JSON serialization errors
   - Added try-except blocks to gracefully handle positions requests before ctx is ready
   - Fixed rendering squares/artifacts by preventing unnecessary widget refreshes
   - Explicit focus styling on Input and DataTable to prevent default focus effect redraws
