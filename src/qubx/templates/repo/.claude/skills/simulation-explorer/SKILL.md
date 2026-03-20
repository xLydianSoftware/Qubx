---
name: simulation-explorer
description: Use when loading, comparing, or debugging backtest simulation results — tearsheets, execution logs, portfolio analysis, log inspection, and diagnosing data issues.
argument-hint: [backtest_path_or_paths] [question]
---

# Simulation Explorer

You help load, compare, analyze, and debug backtest simulation results produced by `qubx simulate`.

## Loading Results

### From Python (notebook or script):
```python
from qubx.backtester.management import BacktestStorage
from qubx.core.metrics import tearsheet

# Account URI — uses named S3 account from ~/.qubx/config.json
bs = BacktestStorage("r2:backtests")

# Also supports direct S3 or local paths
bs = BacktestStorage("s3://my-bucket/backtests")
bs = BacktestStorage("results")  # local directory

# Load a specific run
r = bs.load("STRATEGY_CLASS/run_name/YYYYMMDD_HHMMSS")

# Compare multiple runs
r1 = bs.load("STRATEGY_CLASS/run1/20260320_120000")
r2 = bs.load("STRATEGY_CLASS/run2/20260320_130000")
tearsheet([r1, r2])
```

### From CLI:
```bash
# Browse results interactively (account URI)
uv run qubx browse r2:backtests

# Also works with S3 or local paths
uv run qubx browse s3://my-bucket/backtests
uv run qubx browse results/

# Query results with filters
uv run qubx backtests r2:backtests -w "sharpe > 1.5" -n 10
```

## Result Object API

A loaded result `r` provides:

| Property | Description |
|----------|-------------|
| `r.tearsheet()` | Full performance report |
| `r.equity` | Equity curve (Series) |
| `r.portfolio_log` | Portfolio state over time (DataFrame) |
| `r.executions_log` | All executed trades (DataFrame) |
| `r.signals_log` | All signals generated (DataFrame) |
| `r.emitter_data` | Custom emitter data (if `enable_inmemory_emitter: true`) |

### Common analysis patterns:

```python
# Equity curve
r.equity.plot()

# Positions over time
r.portfolio_log.filter(like="_Pos")

# Funding PnL per asset
r.portfolio_log.filter(like="_Funding")

# Funding PnL breakdown
fp = r.get_funding_per_asset()
fp.iloc[-1].sort_values(ascending=False).head(15).plot(kind="bar")

# Trade list
r.executions_log

# Compare funding between runs
from qubx.pandaz.utils import scols
fp1 = r1.get_funding_per_asset()
fp2 = r2.get_funding_per_asset()
scols(
    fp1["BTC"].rename("run1"),
    fp2["BTC"].rename("run2"),
).plot()
```

## Comparing Two Runs

### Tearsheet comparison:
```python
tearsheet([r1, r2])
```

### Execution diff — find trades in one run but not the other:
```python
df1 = r1.executions_log.set_index(["symbol", "exchange"], append=True)
df2 = r2.executions_log.set_index(["symbol", "exchange"], append=True)
df_diff = df2[~df2.index.isin(df1.index)]
len(df_diff)  # number of different executions
df_diff.head()
```

## Debugging via Log Files

Simulation logs are stored alongside results when run with `-L` flag:
```
STRATEGY_CLASS/run_name/YYYYMMDD_HHMMSS/
├── run_name.log        # simulation log
├── run_name.yml        # config used
├── portfolio.parquet
├── executions.parquet
├── signals.parquet
└── ...
```

### Log analysis patterns:

```bash
# Find selector activity (pair entries/exits)
grep "selectors" /path/to/run.log

# Compare selector lines between two runs
diff <(grep "selectors" run1.log) <(grep "selectors" run2.log)

# Find first divergence point
paste <(grep "selectors" run1.log) <(grep "selectors" run2.log) | awk -F'\t' '$1 != $2 {print NR; exit}'

# Check funding payment data
grep "\[FP\]" /path/to/run.log | head -20

# Check errors
grep "❌" /path/to/run.log
```

## Running Simulations

### From CLI:
```bash
# Basic simulation
uv run qubx simulate configs/my_strategy.yaml -o s3://backtests -L

# Override dates
uv run qubx simulate configs/my_strategy.yaml -s 2026-01-01 -e 2026-03-01 -o s3://backtests -L

# Custom run name
uv run qubx simulate configs/my_strategy.yaml -o s3://backtests -n "baseline_v1" -L
```

### From notebook (quick iteration):
```python
from qubx.utils.runner.runner import simulate_config

results = simulate_config(
    "configs/my_strategy.yaml",
    start="2026-01-15",
    stop="2026-02-01",
    # Override any strategy parameter:
    max_pairs=3,
    min_entry_ev=0,
)
r = results[0]
r.tearsheet()
```

## Output Paths

Results are organized as:
```
<output>/<strategy_class>/<config_name>/<YYYYMMDD_HHMMSS>/
```

- Local output: `-o results` or `-o /tmp/my_test`
- Account URI: `-o r2:backtests` (uses named S3 account from `~/.qubx/config.json`)
- S3 output: `-o s3://backtests`

## Common Debugging Scenarios

### Results differ between runs with same config
1. Compare tearsheets: `tearsheet([r1, r2])`
2. Find first selector divergence in logs
3. Check if OHLC data or funding payment data differs at the divergence point
4. Look for cache-related issues if prefetch is enabled

### Strategy not trading
1. Check if `on_event` fires: look for selector log lines
2. Check `on_fit` timing: look for universe/pair discovery logs
3. Verify funding payment data is available
4. Check if `on_start` was called (requires market data for initial instruments)

### Unexpected PnL
1. Check `r.portfolio_log.filter(like="_Funding")` for funding PnL
2. Check `r.portfolio_log.filter(like="_Pos")` for position changes
3. Compare `r.executions_log` with expected trades
4. Check spread PnL vs funding PnL breakdown
