# Task 012: Add validate command

## Overview
Added a new `qubx validate` command to validate strategy configuration files without running them.

## Changes Made

### 1. Configuration Validation Function (`src/qubx/utils/runner/configs.py`)
- Added `ValidationResult` Pydantic model to hold validation results
- Added `validate_strategy_config()` function with the following checks:
  - File exists
  - Valid YAML syntax
  - Valid Pydantic model structure (all required fields present)
  - Strategy class can be imported (optional, controlled by `check_imports` parameter)
  - Live configuration validation (exchanges, universe, connectors)
  - Simulation configuration validation (instruments, capital, dates)
  - At least one mode (live or simulation) is configured (warning only)

### 2. CLI Command (`src/qubx/cli/commands.py`)
- Added `validate` command with the following features:
  - Takes config file path as argument
  - Optional `--no-check-imports` flag to skip strategy import validation
  - Colored output (green for success, red for errors, yellow for warnings)
  - Exit code 0 for valid config, 1 for invalid

### 3. Tests (`tests/qubx/utils/configs_test.py`)
- Added `test_validate_valid_config()` - tests valid configuration
- Added `test_validate_nonexistent_config()` - tests missing file error
- Added `test_validate_no_exchanges_config()` - tests config without exchanges

### 4. Documentation (`CLAUDE.md`)
- Added validate command examples to Strategy Development section

## Usage

```bash
# Validate a configuration file
poetry run qubx validate config.yml

# Validate without checking if strategy class can be imported
poetry run qubx validate config.yml --no-check-imports

# View help
poetry run qubx validate --help
```

## Benefits

1. **Early Error Detection**: Catch configuration errors before running strategy
2. **CI/CD Integration**: Can be used in automated pipelines to validate configs
3. **Development Speed**: Quick validation without needing to run the full strategy
4. **Clear Feedback**: Color-coded output with specific error messages

## Example Output

Valid configuration:
```
✓ Configuration is valid
```

Invalid configuration:
```
✗ Configuration is invalid

Errors:
  - Failed to import strategy 'invalid.strategy': No module named 'invalid'
  - Exchange 'BINANCE.UM' has no symbols in universe

Warnings:
  - Configuration has neither 'live' nor 'simulation' section
```

## Notes

- The `--no-check-imports` flag is useful when validating configs in environments where the strategy code is not available (e.g., in a different project or during config development)
- The validation is strict for critical errors but provides warnings for non-critical issues
- All validation checks leverage Pydantic's built-in validation, ensuring consistency with the rest of the codebase
- **Important**: Added `extra="forbid"` to all Pydantic config models to catch typos in field names (e.g., `exporters1` instead of `exporters`)

## Bug Fix: Strict Field Validation

During implementation, discovered that Pydantic by default allows extra fields. Created a `StrictBaseModel` base class with `model_config = ConfigDict(extra="forbid")` and updated all config models to inherit from it.

```python
class StrictBaseModel(BaseModel):
    """Base model with strict validation that forbids extra fields."""
    model_config = ConfigDict(extra="forbid")
```

All config models now inherit from `StrictBaseModel`:

- `ConnectorConfig`, `ExchangeConfig`, `ReaderConfig`, `TypedReaderConfig`
- `RestorerConfig`, `PrefetchConfig`, `WarmupConfig`
- `LoggingConfig`, `ExporterConfig`, `EmitterConfig`, `EmissionConfig`
- `NotifierConfig`, `HealthConfig`, `LiveConfig`, `SimulationConfig`
- `StrategyConfig`, `ValidationResult`

This ensures validation catches typos like `exporters1` instead of silently ignoring them, following DRY principles.

## Test Config Updates

Updated test configs to use proper structure with `live` section:

- `tests/qubx/utils/configs/no_aux.yaml` - moved `exchanges` and `logging` under `live`
- `tests/qubx/utils/configs/no_exchanges.yaml` - removed top-level `logging`
- `tests/qubx/cli/release_test.py` - updated `mock_strategy_config` fixture to create `LiveConfig` with `exchanges` and `logging`, then pass it to `StrategyConfig`

## Example Config Updates

Updated example configs to use proper structure with `live` section:

- `examples/macd_crossover/config.yml` - moved `exchanges`, `logging`, and `warmup` under `live`
