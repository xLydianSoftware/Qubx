# Task 005: Textual UI Enhancements

## Status: IN PROGRESS

## Overview
Enhancing the Textual-based strategy runner TUI at `src/qubx/utils/runner/textual/` with improved layout, additional panels, and better usability features.

## Completed Features

### ✅ 1. Command History Navigation
- **File**: `src/qubx/utils/runner/textual/widgets/command_input.py`
- Implemented up/down arrow key navigation through command history
- Uses circular buffer (deque) with max 100 commands
- Preserves current input when navigating history

### ✅ 2. Unified Dashboard Protocol
- **Files**: `src/qubx/core/interfaces.py`, `src/qubx/utils/runner/textual/init_code.py`, `src/qubx/utils/runner/textual/kernel.py`, `src/qubx/utils/runner/textual/handlers.py`
- Single MIME type: `application/x-qubx-dashboard+json`
- Atomic updates for positions, orders, and quotes together
- Added `get_dashboard_data()` method to IStrategy interface for custom data injection
- Debug mode available: `emit_dashboard(debug=True)` shows errors

### ✅ 3. Exchange Column in Tables
- **Files**: `src/qubx/utils/runner/textual/widgets/positions_table.py`, `src/qubx/utils/runner/textual/widgets/orders_table.py`
- Added Exchange as first column in both Positions and Orders tables
- Supports multi-exchange trading strategies

### ✅ 4. Orders Table Widget
- **File**: `src/qubx/utils/runner/textual/widgets/orders_table.py`
- Displays live orders with columns: Exchange, Symbol, Side, Type, Qty, Price, Filled, Status, Time
- Automatically clears when no orders exist (fixes stale data issue)
- Sorted by time (most recent first)
- Hotkey: `o` to toggle visibility

### ✅ 5. Quotes/Market Table Widget
- **File**: `src/qubx/utils/runner/textual/widgets/quotes_table.py`
- Displays market quotes with bid/ask/spread visualization
- Color-coded spread quality:
  - Green (●●●): < 0.05%
  - Yellow (●●○): 0.05-0.20%
  - Red (●○○): > 0.20%
- Hotkey: `m` to toggle visibility

### ✅ 6. Copy-to-Clipboard Functionality
- **File**: `src/qubx/utils/runner/textual/widgets/repl_output.py`
- Hotkey: `Ctrl+Y` copies last 50 lines to clipboard
- Cross-platform support (xclip/xsel/pbcopy)
- Shows success/failure messages

### ✅ 7. Layout Reorganization
- **Files**: `src/qubx/utils/runner/textual/app.py`, `src/qubx/utils/runner/textual/styles.tcss`
- Final layout structure:
  ```
  ┌─────────────────────────────────────────────────┐
  │ Header                                          │
  ├──────────────────────┬──────────────────────────┤
  │                      │                          │
  │  Logs/Output         │  Positions (top)         │
  │  (left side)         │  Orders (middle)         │
  │                      │  Market (bottom)         │
  │                      │  (stacked vertically)    │
  └──────────────────────┴──────────────────────────┘
  │ Input                                           │
  ├─────────────────────────────────────────────────┤
  │ Footer (Hotkeys) ← NOT VISIBLE IN ACTUAL APP   │
  └─────────────────────────────────────────────────┘
  ```
- Logs on left, tables stacked vertically on right
- Tables container only visible when at least one panel is toggled

## 🚨 CURRENT ISSUE: Footer Not Visible

### Problem Description
The Footer widget with hotkey bindings is **not visible in the actual running application**, but **DOES work correctly in unit tests**.

### Investigation Results

#### ✅ What Works (in tests)
Created test file: `tests/qubx/utils/runner/textual/test_app.py`

Test results show:
- Footer is mounted and visible: `Footer() (visible=True, mounted=True)`
- Footer has correct region: `Region(x=0, y=23, width=78, height=1)` (last line)
- Footer has 8 FooterKey widgets (for 7 bindings + palette)
- FooterKeys render correctly: `" ^c Interrupt "`, `" ^l Clear REPL "`, etc.
- Footer has good colors: background `Color(36, 47, 56)`, text `Color(224, 224, 224)`
- All styling is correct

#### ❌ What Doesn't Work (in actual app)
When user runs the actual app (`poetry run qubx run config.yml --paper`):
- Footer is completely invisible at the bottom
- No hotkey text is shown
- Empty/dark space where Footer should be

### Technical Details

**Bindings** (src/qubx/utils/runner/textual/app.py:25-33):
```python
BINDINGS = [
    Binding("ctrl+l", "clear_repl", "Clear REPL", show=True),
    Binding("ctrl+c", "interrupt", "Interrupt", show=True),
    Binding("ctrl+y", "copy_output", "Copy Output", show=True),
    Binding("p", "toggle_positions", "Positions", show=True),
    Binding("o", "toggle_orders", "Orders", show=True),
    Binding("m", "toggle_market", "Market", show=True),
    Binding("q", "quit", "Quit", show=True),
]
```

**Layout Structure** (src/qubx/utils/runner/textual/app.py:108-140):
```python
def compose(self) -> ComposeResult:
    yield Header(show_clock=True)
    with Vertical(id="content-wrapper"):
        with Horizontal(id="main-container"):
            # Output on left, tables on right
            ...
        with Vertical(id="input-container"):
            self.input = CommandInput(...)
            yield self.input
    yield Footer()  # ← This is present but not rendering in actual app
```

**CSS** (src/qubx/utils/runner/textual/styles.tcss:12-20, 114-117):
```css
#content-wrapper {
    height: 1fr;
    overflow: hidden;
}

#main-container {
    height: 1fr;
    layout: horizontal;
}

Footer {
    dock: bottom;
    height: 1;
}
```

### Hypothesis
There's likely a difference between how `run_test()` and the actual app (`run()`) handle layout or rendering:
1. The content-wrapper or main-container might be overflowing in the real app
2. The Footer might be getting pushed off-screen despite `dock: bottom`
3. There could be a z-index or layer issue
4. The actual terminal size vs test size might cause layout differences

### Screenshots/Evidence
- SVG export from tests shows NO Footer content in lines 28-30 (should have hotkey text)
- User confirms Footer is not visible in actual app screenshot
- Test output proves Footer exists and has content, but it's not rendering to screen

## Next Steps for Resolution

### Option 1: Force Footer to Overlay
Try making Footer position absolutely with overlay:
```css
Footer {
    dock: bottom;
    height: 1;
    layer: overlay;
}
```

### Option 2: Constrain Content Wrapper Better
Ensure content-wrapper leaves room for Footer:
```css
Screen {
    layout: grid;
    grid-size: 1 3;
    grid-rows: auto 1fr auto;
}

#content-wrapper {
    height: auto;
}
```

### Option 3: Debug Actual App Layout
Add temporary debug code to print Footer info in actual app on mount:
```python
async def on_mount(self):
    await super().on_mount()
    footer = self.query_one(Footer)
    self.output.write(f"[yellow]Footer region: {footer.region}")
    self.output.write(f"[yellow]Footer visible: {footer.display}")
```

### Option 4: Check for Widget Conflicts
The CommandInput widget might be interfering. Try temporarily removing it to see if Footer appears.

## Files Modified

### Core Files
- `src/qubx/core/interfaces.py` - Added `get_dashboard_data()` to IStrategy
- `src/qubx/utils/runner/textual/app.py` - Main app layout and bindings
- `src/qubx/utils/runner/textual/handlers.py` - Unified dashboard handler
- `src/qubx/utils/runner/textual/kernel.py` - Dashboard MIME type recognition
- `src/qubx/utils/runner/textual/init_code.py` - Unified `emit_dashboard()` function
- `src/qubx/utils/runner/textual/styles.tcss` - Layout and styling

### Widget Files
- `src/qubx/utils/runner/textual/widgets/command_input.py` - Created (history support)
- `src/qubx/utils/runner/textual/widgets/orders_table.py` - Created
- `src/qubx/utils/runner/textual/widgets/quotes_table.py` - Created (spread visualization)
- `src/qubx/utils/runner/textual/widgets/positions_table.py` - Updated (exchange column)
- `src/qubx/utils/runner/textual/widgets/repl_output.py` - Updated (copy support)

### Test Files
- `tests/qubx/utils/runner/textual/test_app.py` - Created (comprehensive Footer tests)

## Pending Features
- [ ] **Fix Footer visibility in actual app** ← CRITICAL BLOCKER
- [ ] Autocompletion support for CommandInput (optional/future)

## Notes
- All hotkeys are defined and should work even if Footer is invisible
- User can test if hotkeys work (try pressing `q` to quit, `p` for positions, etc.)
- If hotkeys don't work either, the issue is more severe than just rendering
