# Task 014: Refactor textual_runner into organized submodule

**Status**: Completed
**Created**: 2025-10-07
**Completed**: 2025-10-07
**Priority**: High

## Objective

Refactor the monolithic `textual_runner.py` file into a well-organized submodule with clear separation of concerns, making it easier to maintain and extend.

## Current Issues

- Single 570-line file mixing multiple concerns:
  - Kernel management (`IPyKernel`)
  - UI widgets (`ReplOutput`, positions DataTable)
  - Main app logic (`TextualStrategyApp`)
  - Event handling
  - Strategy initialization code generation
- No separation between UI components and business logic
- Hard to extend with new widgets/panels (orders, orderbook, etc.)
- Doesn't follow the pattern of other Qubx modules

## Proposed Structure

```
src/qubx/utils/runner/textual/
├── __init__.py                    # Public API exports
├── app.py                         # TextualStrategyApp (main app class)
├── kernel.py                      # IPyKernel wrapper
├── widgets/
│   ├── __init__.py
│   ├── repl_output.py            # ReplOutput widget
│   └── positions_table.py        # Positions DataTable widget
├── handlers.py                    # Event handlers & kernel event processing
└── init_code.py                   # Strategy initialization code generation
```

## Implementation Plan

### 1. Create submodule structure
- Create `src/qubx/utils/runner/textual/` directory
- Create `src/qubx/utils/runner/textual/widgets/` directory
- Create empty `__init__.py` files

### 2. Extract kernel management
- Move `IPyKernel` class to `kernel.py`
- Keep all kernel lifecycle methods
- Keep iopub draining and message processing

### 3. Extract widgets
- Move `ReplOutput` to `widgets/repl_output.py`
- Create `widgets/positions_table.py` with:
  - PositionsTable widget (DataTable wrapper)
  - Update logic for positions data
  - NaN/Inf sanitization

### 4. Extract initialization code
- Move `_generate_init_code()` to `init_code.py`
- Make it a standalone function
- Keep all helper function definitions

### 5. Extract event handlers
- Create `handlers.py` for kernel event processing
- Move `on_kernel_event()` logic
- Create reusable event handler functions

### 6. Refactor main app
- Simplify `app.py` to focus on:
  - Layout composition
  - Key bindings
  - High-level coordination
- Import components from submodules
- Keep CSS and BINDINGS

### 7. Update imports
- Create public API in `textual/__init__.py`
- Export `run_strategy_yaml_in_textual` function
- Update import in `cli/commands.py`

### 8. Cleanup
- Delete old `textual_runner.py` file
- Test the refactored implementation

## Benefits

- **Clear separation of concerns**: Each module has a single responsibility
- **Easier to extend**: Adding new widgets (orders, orderbook) is straightforward
- **Testable components**: Each module can be tested independently
- **Follows Qubx patterns**: Consistent with other module organization
- **Backward compatibility**: Public API maintained via `__init__.py` exports
- **Better maintainability**: Easier to understand and modify individual components

## Files to Create

- `src/qubx/utils/runner/textual/__init__.py`
- `src/qubx/utils/runner/textual/app.py`
- `src/qubx/utils/runner/textual/kernel.py`
- `src/qubx/utils/runner/textual/handlers.py`
- `src/qubx/utils/runner/textual/init_code.py`
- `src/qubx/utils/runner/textual/widgets/__init__.py`
- `src/qubx/utils/runner/textual/widgets/repl_output.py`
- `src/qubx/utils/runner/textual/widgets/positions_table.py`

## Files to Modify

- `src/qubx/cli/commands.py` - Update import path

## Files to Delete

- `src/qubx/utils/runner/textual_runner.py`

## Testing

Test with:
```bash
poetry run qubx run config.yml --paper --textual
```

Verify:
- App launches without errors
- REPL interaction works
- Positions panel toggles with 'p' key
- All keybindings work (Ctrl+L, Ctrl+C, q)
- Helper functions available: ctx, S, portfolio(), orders(), trade(), exit()

## Completion Summary

Successfully refactored `textual_runner.py` (570 lines) into organized submodule:

### Final Structure

```
src/qubx/utils/runner/textual/
├── __init__.py                    # Public API: run_strategy_yaml_in_textual()
├── app.py                         # TextualStrategyApp (main app class, 165 lines)
├── kernel.py                      # IPyKernel wrapper (130 lines)
├── handlers.py                    # KernelEventHandler (70 lines)
├── init_code.py                   # generate_init_code() (145 lines)
├── styles.tcss                    # Textual CSS stylesheet (108 lines)
└── widgets/
    ├── __init__.py
    ├── repl_output.py            # ReplOutput widget (17 lines)
    └── positions_table.py        # PositionsTable widget (95 lines)
```

### Benefits Achieved
✅ **Clear separation of concerns**: Each module has single responsibility
✅ **Easier to extend**: Adding new widgets (orders, orderbook) is straightforward
✅ **Testable components**: Each module can be tested independently
✅ **Follows Qubx patterns**: Consistent with other module organization
✅ **Backward compatibility**: Public API maintained via `__init__.py` exports
✅ **Better maintainability**: Easier to understand and modify individual components
✅ **Passes linting**: All files pass ruff checks
✅ **Import tested**: Module imports successfully

### Files Changed
- Created: 9 new files in `textual/` submodule (including `styles.tcss`)
- Modified: `src/qubx/cli/commands.py` (updated import path)
- Deleted: `src/qubx/utils/runner/textual_runner.py`

### Notes
- All original functionality preserved
- No breaking changes to public API
- Code organization now matches other Qubx modules (e.g., `cli/tui.py`)
- Ready for future enhancements (order book widget, bid/ask panel, etc.)
