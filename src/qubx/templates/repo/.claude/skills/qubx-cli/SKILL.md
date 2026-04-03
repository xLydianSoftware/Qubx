# Qubx CLI Reference

Complete reference for the `qubx` command-line interface. All commands should be run with `uv run qubx` in a project that depends on qubx.

## Commands Overview

| Command | Purpose |
|---|---|
| `qubx simulate` | Run a backtest simulation |
| `qubx run` | Execute strategy (paper or live) |
| `qubx validate` | Check config file validity |
| `qubx browse` | Interactive backtest results browser (TUI) |
| `qubx backtests` | Query backtest results (CLI table) |
| `qubx init` | Create strategy from template |
| `qubx release` | Package strategy for distribution |
| `qubx deploy` | Install a released strategy package |
| `qubx ls` | List strategies in directory |
| `qubx kernel list` | Show active persistent kernels |
| `qubx kernel stop` | Stop a persistent kernel |
| `qubx s3 ls/rm/cp/accounts` | S3 storage operations |

## Global Options

```bash
uv run qubx [--debug] [--debug-port PORT] [--log-level LEVEL] COMMAND
```

| Option | Default | Description |
|---|---|---|
| `--debug, -d` | False | Enable debugpy for remote attach |
| `--debug-port, -p` | 5678 | Debugger port |
| `--log-level, -l` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## qubx simulate

Run a backtest simulation on historical data.

```bash
uv run qubx simulate <config.yaml> [OPTIONS]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `CONFIG_FILE` | Yes | — | Strategy YAML config file |
| `-s, --start TEXT` | No | Config value | Simulation start date (YYYY-MM-DD) |
| `-e, --end TEXT` | No | Config value | Simulation end date (YYYY-MM-DD) |
| `-o, --output TEXT` | No | `results` | Output directory (local path or S3 URI) |
| `-n, --name TEXT` | No | Auto | Run name for output subfolder |
| `-r, --report TEXT` | No | None | Generate report to this directory |
| `-L, --log` | No | False | Write simulation logs to output directory |
| `--log-file PATH` | No | None | Write logs to specific file path |

### Examples

```bash
# Basic simulation
uv run qubx simulate configs/my_strategy.yaml

# With date range and S3 output
uv run qubx simulate configs/my_strategy.yaml -s 2024-01-01 -e 2024-12-31 -o s3://backtests

# With named run and logging
uv run qubx simulate configs/my_strategy.yaml -o s3://backtests -n "baseline_v1" -L

# With account URI output (e.g., Cloudflare R2)
uv run qubx simulate configs/my_strategy.yaml -o r2:backtests -L
```

### Output Structure

Results are stored as:
```
<output>/<config_name>/<strategy_class>/<YYYYMMDD_HHMMSS>/
├── metadata.json
├── portfolio.parquet
├── trades.parquet
└── logs/ (if -L flag used)
```

---

## qubx run

Execute a strategy in paper or live trading mode.

```bash
uv run qubx run <config.yaml> [OPTIONS]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `CONFIG_FILE` | Yes | — | Strategy YAML config file |
| `-a, --account-file PATH` | No | Auto-discovered | Account configuration file |
| `-p, --paper` | No | False | Paper trading mode |
| `-j, --jupyter` | No | False | Run in Jupyter console |
| `-t, --textual` | No | False | Run in Textual TUI dashboard |
| `--textual-web` | No | False | Serve TUI in web browser |
| `--textual-port INT` | No | 8000 | Port for web/devtools TUI |
| `--textual-host TEXT` | No | 0.0.0.0 | Host for web TUI |
| `--kernel-only` | No | False | Start kernel without UI |
| `--connect PATH` | No | None | Connect TUI to existing kernel |
| `-r, --restore` | No | False | Restore state from previous run |
| `--override PATH` | No | None | Sparse YAML to deep-merge on config |
| `--no-emission` | No | False | Disable metric emission |
| `--no-notifiers` | No | False | Disable lifecycle notifiers |
| `--no-exporters` | No | False | Disable trade exporters |
| `--no-color` | No | False | Disable colored logging |
| `--dev` | No | False | Dev mode (adds ~/projects to path) |

### Examples

```bash
# Paper trading
uv run qubx run configs/strategy.yaml --paper

# Paper with TUI dashboard
uv run qubx run configs/strategy.yaml --paper --textual

# Paper without external integrations (local testing)
uv run qubx run configs/strategy.yaml --paper --no-notifiers --no-exporters --no-emission

# Jupyter interactive mode
uv run qubx run configs/strategy.yaml --paper -j

# Live trading with explicit account file
uv run qubx run configs/strategy.yaml -a accounts.toml

# With config overrides
uv run qubx run configs/strategy.yaml --override local_tweaks.yaml

# Persistent kernel mode (start in one terminal, connect from another)
uv run qubx run configs/strategy.yaml --paper --kernel-only
# Then in another terminal:
uv run qubx run configs/strategy.yaml --connect /tmp/qubx_kernel_*.json --textual
```

### Account File Discovery

For live trading, accounts are searched in order:
1. `-a <path>` if provided
2. `accounts.toml` in the config file's directory
3. `~/qubx/accounts.toml`

---

## qubx browse

Interactive TUI for browsing backtest results.

```bash
uv run qubx browse [RESULTS_PATH]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `RESULTS_PATH` | No | Config default or `results` | Local path, S3 URI, or account URI |

### Examples

```bash
# Browse local results
uv run qubx browse results/

# Browse S3 results
uv run qubx browse s3://backtests

# Browse with account URI
uv run qubx browse r2:backtests
```

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `q` | Quit |
| `j/k` | Navigate down/up |
| `h/l` | Focus tree/table |
| `Enter` | Open details |
| `S` | Sort by Sharpe |
| `C` | Sort by CAGR |
| `T` | Sort by creation time |
| `r` | Refresh |
| `d` | Delete (with confirmation) |
| `cc` | Copy backtest ID |
| `1/2` | Switch detail tabs |

---

## qubx backtests

Query backtest results from the command line (non-interactive).

```bash
uv run qubx backtests [STORAGE_PATH] [OPTIONS]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `STORAGE_PATH` | No | `results` | Path to results (local/S3/account URI) |
| `-w, --where TEXT` | No | None | SQL WHERE filter (e.g., `"sharpe > 1.5"`) |
| `-O, --order-by TEXT` | No | `creation_time DESC` | SQL ORDER BY clause |
| `-n, --limit INT` | No | None | Max results to show |
| `-p, --params` | No | False | Show strategy parameters |

### Examples

```bash
# List all backtests
uv run qubx backtests s3://backtests

# Filter by Sharpe ratio
uv run qubx backtests s3://backtests -w "sharpe > 1.5" -n 10

# Show with parameters, sorted by CAGR
uv run qubx backtests s3://backtests -O "cagr DESC" --params
```

---

## qubx validate

Check a strategy configuration file for errors.

```bash
uv run qubx validate <config.yaml> [--no-check-imports]
```

Validates YAML syntax, required fields, exchange configs, and (by default) that the strategy class can be imported. Exit code 0 = valid, 1 = invalid.

---

## qubx init

Create a new strategy project from a template.

```bash
uv run qubx init [OPTIONS]
```

### Templates

| Template | Description |
|---|---|
| `simple` | Flat directory, minimal boilerplate |
| `project` | Full project with pyproject.toml and MACD example |
| `repo` | Full repository with CI, tests, justfile, VS Code configs, CLI scaffold |

### Examples

```bash
# List available templates
uv run qubx init --list-templates

# Quick strategy scaffold
uv run qubx init -t simple -n my_strategy -s BTCUSDT,ETHUSDT

# Full repo with interactive wizard
uv run qubx init -t repo -n my_strategy -o ~/projects

# Non-interactive repo creation
uv run qubx init -t repo -n my_strategy -d "My strategy" \
  --author "Name" --email "name@example.com" \
  --github-org myorg --create-repo -o ~/projects
```

The `repo` template creates a complete project with git init (main branch), optional GitHub repo creation, and persists author/org preferences to `~/.qubx/init.json`.

---

## qubx s3

S3-compatible storage operations using named accounts from `~/.qubx/config.json`.

```bash
# List configured accounts
uv run qubx s3 accounts

# List files
uv run qubx s3 ls <account:bucket/path> [-r] [-l]

# Delete files
uv run qubx s3 rm <account:bucket/path> [-r] [-y]

# Copy files (at least one path must be S3)
uv run qubx s3 cp <src> <dst> [-r]
```

### Path Format

S3 paths use the `account:bucket/path` format where `account` is a named S3 account from `~/.qubx/config.json`.

```bash
uv run qubx s3 ls r2:backtests/ -r -l
uv run qubx s3 cp r2:backtests/run1/ ./local_copy/ -r
uv run qubx s3 rm r2:backtests/old_run/ -r -y
```

---

## VS Code Integration

Strategy repos include VS Code tasks and launch configs for common operations:

### Tasks (Ctrl+Shift+P → Tasks: Run Task)

| Task | Description |
|---|---|
| Backtest | Run simulation in tmux session |
| Backtest [Named] | Run simulation with custom run name |
| Paper [Textual] | Run paper trading with TUI |
| Live [Textual] | Run live trading with TUI |
| Validate Config | Validate the current YAML file |

All tasks operate on the **currently open file** in the editor and run in tmux sessions for persistence.

### Launch Configs (F5 → Select Configuration)

| Config | Description |
|---|---|
| (Backtest) Debug Current Yaml Config | Debug simulation with breakpoints |
| (Paper) Run Current Yaml Config | Debug paper trading |
| (Paper) (Jupyter) Run Current Yaml Config | Debug with Jupyter console |
| (Paper) (Textual) Run Current Yaml Config | Debug with TUI |
| Python Debugger: Remote Attach | Attach to running strategy (port 5678) |
