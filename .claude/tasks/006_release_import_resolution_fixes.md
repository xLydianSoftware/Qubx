# Task 006: Release Import Resolution Fixes

## Overview
Fix critical issues in the `qubx release` command's import resolution system to ensure all dependencies are properly included in release packages.

## Identified Issues
1. **Broken `ast.Import` handling**: Direct imports like `import module` are not detected
2. **Missing relative import support**: `from .module import name` imports are ignored  
3. **Silent error suppression**: Missing dependencies fail silently without warnings
4. **Incomplete package copying**: Only `__init__.py` copied from packages, missing other files
5. **No import validation**: No verification that discovered dependencies actually exist

## Current Test Coverage Analysis
- **Existing**: `/tests/qubx/cli/release_test.py` has basic release functionality tests
- **Missing**: NO tests for `get_imports()`, `_get_imports()`, or dependency resolution logic
- **Risk**: Import bugs not caught by current test suite

## Implementation Plan

### Phase 1: Create Test Infrastructure
- [ ] Create `/tests/qubx/cli/test_import_resolution.py` 
- [ ] Add comprehensive test cases for each import scenario
- [ ] Create mock file system for testing

### Phase 2: Fix Core Import Detection
- [ ] Fix `get_imports()` function for `ast.Import` handling
- [ ] Add relative import resolution support
- [ ] Replace silent error handling with proper logging and exceptions

### Phase 3: Enhance Dependency Resolution  
- [ ] Improve `_copy_dependencies()` to copy complete packages
- [ ] Add dependency validation before packaging
- [ ] Implement proper error reporting for missing dependencies

### Phase 4: Integration Testing
- [ ] Test fixes with real strategy examples
- [ ] Verify complete dependency inclusion in releases
- [ ] Document new import resolution behavior

## Progress Log

### [Date] - Started Task
- Created task documentation
- Analyzed current codebase and test coverage
- Identified 5 critical issues in import resolution

---

*Next: Begin Phase 1 - Create comprehensive test infrastructure*