# Task 017: Textual UI Enhancements

## Overview
Enhance the Textual-based strategy runner with improved input handling, unified dashboard protocol, new data panels (orders, quotes), and better user experience features.

## Goals
1. Implement unified dashboard protocol for atomic data updates
2. Add command history navigation (Up/Down arrows)
3. Create orders and quotes panels with hotkeys
4. Add exchange columns to all tables
5. Implement spread visualization
6. Add Python autocompletion support

## Implementation Plan

### Phase 1: Core Infrastructure
- [x] Add `get_dashboard_data()` to IStrategy interface
- [ ] Implement unified dashboard protocol with single MIME type
- [ ] Update init_code.py with unified `emit_dashboard()` function
- [ ] Update KernelEventHandler to handle unified dashboard data

### Phase 2: Widget Improvements
- [ ] Add Exchange column to PositionsTable
- [ ] Create OrdersTable widget with exchange column
- [ ] Create QuotesTable widget
- [ ] Create CommandInput widget with history support

### Phase 3: UI Assembly
- [ ] Update app.py layout for new panels
- [ ] Add hotkey bindings (o=orders, m=market data)
- [ ] Update styles.tcss for new panels

### Phase 4: Advanced Features
- [ ] Implement spread visualization in QuotesTable
- [ ] Add autocompletion support to CommandInput
- [ ] Polish and test all features

## Technical Notes

### Unified Protocol Design
Single MIME type: `application/x-qubx-dashboard+json`

Payload structure:
```python
{
    "positions": [
        {"exchange": "BINANCE.UM", "symbol": "BTCUSDT", "side": "LONG", "qty": 0.5, ...}
    ],
    "orders": [
        {"exchange": "BINANCE.UM", "symbol": "BTCUSDT", "side": "buy", "type": "limit", ...}
    ],
    "quotes": {
        "BINANCE.UM:BTCUSDT": {"bid": 43000, "ask": 43001, "last": 43000.5, ...}
    },
    "custom": {
        # Strategy-provided data via get_dashboard_data()
    }
}
```

### Hotkey Bindings
- `p` - Toggle positions panel
- `o` - Toggle orders panel
- `m` - Toggle market data/quotes panel
- `ctrl+l` - Clear REPL
- `ctrl+c` - Interrupt kernel
- `q` - Quit

### Command History
- Store last 100 commands in circular buffer
- Up arrow: previous command
- Down arrow: next command
- Navigate to end of list shows empty input

## Implementation Progress

### 2025-10-13: Started implementation
- Created task file
- Ready to begin Phase 1

### 2025-10-13: Mock Testing Infrastructure
- Added `generate_mock_init_code()` to init_code.py for test mode
- Added `test_mode` parameter to TextualStrategyApp
- Created test suite in tests/qubx/utils/runner/textual_app/test_app.py
- Renamed test folder from `textual` to `textual_app` to avoid Python import conflicts
- Tests confirm Footer is visible and properly styled (dock=bottom, height=1, display=block)

### Footer Investigation Results
- Footer widget IS present and rendering correctly
- All bindings (p, o, m, q, ctrl+l, ctrl+c, ctrl+y) are registered with show=True
- CSS properly applies: dock=bottom, height=1, display=block
- Footer and FooterKeys are mounted and visible in widget tree

### Footer Rendering Debug (2025-10-13)
- Created `styles_minimal.tcss` and `test_minimal_css.py` to isolate Footer rendering
- Minimal CSS test showed Footer rendering PERFECTLY with bindings text visible
- Root cause identified: RichLog (ScrollView) expands to fill all available vertical space
- In real app, RichLog consumed all 24 terminal lines, leaving no space for Footer
- **Solution**: Modified `styles.tcss` to explicitly constrain heights:
  - Added `height: 1fr` to `#content-wrapper` and `#main-container`
  - Added `height: 1` to `Footer`
  - Added `height: 1fr` to `RichLog`
- Created `test_progressive_real_app.py` for systematic CSS testing with automatic snapshot cleanup
- Added `just snap` recipe to justfile for easy snapshot test execution
- **Resolution**: Footer now renders correctly with proper space allocation
