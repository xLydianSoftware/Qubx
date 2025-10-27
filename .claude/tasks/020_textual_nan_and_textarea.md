# Task 020: Fix Textual UI - NaN Warnings & REPL Selection

## Overview
Fix two issues in the Textual UI:
1. NaN warnings when emit_dashboard() sends invalid numeric data
2. Replace RichLog with TextArea to enable text selection and copying

## Problems

### Problem 1: NaN Warnings in emit_dashboard()
The dashboard emit functions occasionally generate warnings when trying to send NaN or infinity values to the UI tables.

### Problem 2: RichLog Doesn't Support Text Selection
Users cannot select text with the cursor and copy with Cmd+C/Ctrl+C because RichLog doesn't support text selection. The current workaround (Ctrl+Y to copy last 50 lines) is not user-friendly.

## Solution

### 1. Fix NaN Warnings
- Create `_sanitize_number()` helper function in init_code.py
- Apply to all numeric fields in `_positions_as_records()`, `_orders_as_records()`, `_quotes_as_records()`
- Convert NaN/infinity to 0.0 before sending to dashboard

### 2. Replace RichLog with TextArea
- Change ReplOutput base class from RichLog to TextArea
- Set `read_only=True` and `show_line_numbers=True`
- Implement manual line limit (10,000 lines)
- Implement smart autoscroll:
  - Check `cursor_at_last_line` before appending
  - Use `insert(text, location=document.end)` to append
  - If user was at last line: move cursor to end and call `scroll_cursor_visible()`
  - If user scrolled up: leave position unchanged
- Remove copy methods and subprocess imports

### 3. Update Integration Points
- app.py: Remove Ctrl+Y binding and copy action
- handlers.py: Convert Rich objects to strings before writing

## Files Modified

1. `src/qubx/utils/runner/textual/init_code.py`
2. `src/qubx/utils/runner/textual/widgets/repl_output.py`
3. `src/qubx/utils/runner/textual/app.py`
4. `src/qubx/utils/runner/textual/handlers.py`

## Implementation Details

### TextArea API Usage
- `cursor_at_last_line`: Read-only boolean property
- `insert(text, location)`: Append text at specific location
- `document.end`: Location at end of document
- `cursor_location`: Property to get/set cursor position
- `scroll_cursor_visible(animate=False)`: Scroll to make cursor visible

## Testing
- Verify text selection works with mouse
- Verify Cmd+C/Ctrl+C copies selected text
- Verify smart autoscroll behavior
- Verify line limit works (doesn't exceed 10K lines)
- Verify no NaN warnings in dashboard updates
- Verify line numbers display correctly

## Status
Completed

## Date
2025-10-27

## Implementation Summary

### 1. Fixed NaN Warnings (init_code.py)

Added `_sanitize_number()` helper function that converts NaN and infinity values to 0.0:
- Applied to all numeric fields in `_positions_as_records()`: qty, avg_px, last_px, pnl, mkt_value
- Applied to all numeric fields in `_orders_as_records()`: qty, price, filled
- Applied to all numeric fields in `_quotes_as_records()`: bid, ask, spread, spread_pct, last, volume

### 2. Replaced RichLog with TextArea (repl_output.py)

Complete rewrite of ReplOutput widget:
- Changed base class from `RichLog` to `TextArea`
- Enabled `read_only=True` and `show_line_numbers=True`
- Implemented manual line limit (10,000 lines default)
- Implemented smart autoscroll using `cursor_at_last_line` property:
  - If user is at last line: auto-scroll to follow new content
  - If user scrolled up: stay at current position
- Added check for `is_attached` before calling `scroll_cursor_visible()` to handle unmounted widgets
- Removed copy methods and subprocess imports
- Text selection now works natively with mouse and Cmd+C/Ctrl+C

### 3. Updated Integration Points

**app.py:**
- Removed `Ctrl+Y` copy_output binding from BINDINGS
- Removed `action_copy_output()` method
- Cleaned up ReplOutput instantiation (removed unsupported `wrap` and `markup` parameters)

**handlers.py:**
- Converted Markdown objects to strings
- Converted error tracebacks to plain text using `.plain` property
- Removed Rich markup from error messages

## Testing

Basic functionality tests passed:
- ✓ Module imports successfully
- ✓ Write method works with strings and Rich Text objects
- ✓ Line limit prevents unlimited memory growth
- ✓ Smart autoscroll logic implemented correctly

### 4. Fixed Rich Markup Tags Displaying as Literal Text

**Issue:** After switching to TextArea, Rich markup tags like `[bold cyan]` were displaying as literal text instead of being rendered as colors.

**Solution:**
- Added `from rich.text import Text` import to app.py
- Converted all 13 instances of Rich markup strings to Rich Text objects using `Text(content, style="...")`
- Examples:
  - `"[bold cyan]>>> {code}"` → `Text(f">>> {code}", style="bold cyan")`
  - `"[green]✓ Test mode initialized!"` → `Text("✓ Test mode initialized!", style="green")`

**Result:**
- Colors now render correctly in the terminal output via ANSI codes
- No more visible markup tags like `[bold cyan]` in the output
- User experience is clean and properly formatted

### 5. Added Ctrl+E Binding to Scroll to End

**Feature Request:** Add a keyboard shortcut to quickly scroll to the end of the REPL output.

**Implementation:**
- Added `Binding` import from `textual.binding`
- Added `BINDINGS` list to ReplOutput class with `Ctrl+E` binding
- Implemented `action_scroll_to_end()` method that:
  - Moves cursor to `self.document.end`
  - Calls `scroll_cursor_visible(animate=False)` if widget is attached

**Usage:**
- Press `Ctrl+E` while focused on the REPL output to instantly jump to the most recent output
- Useful after scrolling up to review previous output

## Notes

- ANSI color codes from Text objects are preserved when converted to strings in the write() method
- The widget automatically handles unmounted state by checking `is_attached` before scrolling
- All user-facing messages now use Rich Text objects for consistent formatting
