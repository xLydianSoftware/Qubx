# Strategy Release, Deploy & Run

Guide for building, deploying, and running strategy release packages.

## Commands Overview

| Command | Purpose |
|---|---|
| `qubx release` | Package strategy into a release zip |
| `qubx deploy` | Install a release zip into a runnable directory |
| `qubx run` | Start a strategy from a config file |

---

## qubx release

Packages a strategy config into a self-contained zip with compiled wheels and pinned dependencies.

```bash
qubx release -c <config.yaml> [-o <output_dir>] [<project_dir>] [--commit] [-t <tag>] [-m <message>]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-c, --config PATH` | Yes | — | Path to strategy YAML config file |
| `-o, --output-dir PATH` | No | `.releases` | Directory for the output zip |
| `DIRECTORY` (positional) | No | Current dir | Project root to scan for strategy classes |
| `--commit` | No | False | Commit changes and create git tag |
| `-t, --tag TEXT` | No | — | Additional tag suffix for the release |
| `-m, --message TEXT` | No | — | Release message added to info file |

### Examples

```bash
# Basic release (no git commit/tag)
uv run qubx release -c configs/my_strategy.yaml -o /tmp/releases

# Release from a different project directory
uv run qubx release -c /path/to/configs/strategy.yaml -o /tmp/releases /path/to/myproject

# Release with git commit and custom tag
uv run qubx release -c configs/strategy.yaml --commit -t v2.1 -m "Production rollout"
```

### What it does

1. **Loads config** — parses YAML without resolving `env:` variables (env vars are only needed at runtime)
2. **Resolves imports recursively** — BFS from strategy entry point files, following all internal imports transitively. Only the files actually needed are included.
3. **Scans dependencies** — maps external imports to packages, pins versions from `uv.lock`
4. **Builds strategy wheel** — selectively copies resolved files, compiles via Cython
5. **Bundles private wheels** — downloads from private registries or builds from local paths
6. **Generates release pyproject.toml + uv.lock** — minimal, self-contained
7. **Creates zip** — everything needed to deploy

### Three config types

**Local code only** (strategy classes from the source project):
- Wheel built from source package
- Transitive internal imports resolved and included
- External deps scanned from all resolved files

**External only** (strategy classes only from external packages):
- No wheel built
- External packages listed directly as dependencies
- Private deps bundled as wheels

**Mixed** (local + external strategy classes):
- Wheel built for local code
- External strategy packages added as additional dependencies
- Both bundled as wheels if private

### Plugin handling

Configs can declare plugins:
```yaml
plugins:
  modules:
    - qubx_myplugin
```

Plugin modules are resolved to package specs from `[project.optional-dependencies]` or `uv.lock`, added as dependencies, and bundled as wheels if they're from a private registry.

### Version pinning

`uv.lock` is the single source of truth for all versions. If `uv.lock` doesn't exist in the source project, it's generated automatically. The release never relies on the current environment's installed packages.

---

## qubx deploy

Installs a release zip into a runnable directory.

```bash
qubx deploy <zip_file> [-o <output_dir>] [--force] [--system]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `ZIP_FILE` (positional) | Yes | — | Path to the release zip |
| `-o, --output-dir PATH` | No | Same dir as zip | Output directory |
| `-f, --force` | No | False | Overwrite existing output directory |
| `--system` | No | False | Docker mode: pip install into system site-packages |

### Examples

```bash
# Deploy to a directory (creates venv)
uv run qubx deploy /tmp/releases/R_Strategy_20260317.zip -o ~/deployments/strategy1

# Force overwrite existing deployment
uv run qubx deploy release.zip -o ~/deployments/strategy1 --force

# Docker mode (no venv, installs to system Python)
qubx deploy release.zip -o /app/strategy --system
```

### Default mode (dev/local)

1. Extracts zip to output directory
2. Detects package manager from lock file
3. Runs `uv sync` to create `.venv/` and install all dependencies
4. Strategy is ready to run with `uv run qubx run`

### System mode (Docker)

1. Extracts zip to output directory
2. Runs `pip install wheels/*.whl` into system site-packages
3. No venv, no uv needed
4. Strategy is ready to run with `qubx run` directly

---

## qubx run

Starts a strategy from a config file.

```bash
qubx run <config_file> [-a <account_file>] [-p] [--override <override.yaml>] [flags...]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `CONFIG_FILE` (positional) | Yes | — | Path to strategy YAML config |
| `-a, --account-file PATH` | No | Auto-discovered | Account configuration file |
| `-p, --paper` | No | False | Paper trading mode (no real account needed) |
| `-j, --jupyter` | No | False | Run in Jupyter console |
| `-t, --textual` | No | False | Run in Textual TUI |
| `-r, --restore` | No | False | Restore state from previous run |
| `--override PATH` | No | — | Sparse YAML to deep-merge on top of config |
| `--no-emission` | No | False | Disable metric emission |
| `--no-notifiers` | No | False | Disable lifecycle notifiers |
| `--no-exporters` | No | False | Disable trade exporters |
| `--no-color` | No | False | Disable colored logging |
| `--dev` | No | False | Dev mode (adds ~/projects to path) |

### Examples

```bash
# Paper trading (from deployed directory)
cd ~/deployments/strategy1
uv run qubx run config.yml --paper

# Paper trading without notifiers/exporters (local testing)
uv run qubx run config.yml --paper --no-notifiers --no-exporters --no-emission

# Live trading with account file
uv run qubx run config.yml -a accounts.toml

# With config overrides
uv run qubx run config.yml --override local_overrides.yaml

# In Textual TUI
uv run qubx run config.yml --paper --textual
```

### Environment variables in configs

Config YAML files can reference environment variables using `env:` syntax. These are resolved **only at runtime** (`qubx run`), not during release or deploy.

**Supported formats:**
```yaml
# Legacy format — fails if variable missing
bot_token: env:SLACK_BOT_TOKEN

# New format — fails if variable missing
bot_token: env:{SLACK_BOT_TOKEN}

# New format with default — uses default if missing
redis_url: env:{REDIS_URL:redis://localhost:6379}
```

**Important:** `qubx release` does NOT resolve env vars. This means:
- You can release configs that reference env vars without having them set
- Env vars only need to exist on the machine where `qubx run` executes
- Use `--no-notifiers --no-exporters --no-emission` flags to skip sections that require env vars during local testing

### Account file discovery

When running live (not `--paper`), qubx searches for account configs in order:
1. `-a <path>` if provided
2. `accounts.toml` in the same directory as the config
3. `~/qubx/accounts.toml`

---

## Common Issues

### ModuleNotFoundError at runtime

The recursive import resolver may miss files if they're imported dynamically (e.g. `importlib.import_module()`). Check the release log for "Resolved N internal files" — if the count seems low, the missing module may not be statically imported.

### Private dependency not resolving

If `uv lock` fails during release with "no version found", the private dependency likely needs to be bundled. Check that:
- The package is listed in `[tool.uv.sources]` with an `index` or `path` source
- The version exists in `uv.lock`
- The private registry is accessible

### Environment variable errors during release

This should NOT happen — `qubx release` skips env var resolution. If you see `Environment variable 'X' not found` during release, the installed qubx version may be outdated.

### Platform-specific wheels

The strategy wheel is compiled for the build machine's platform. If deploying to a different architecture (e.g. building on Mac, deploying to Linux), the wheel won't work. Build on the same platform as the target, or use Docker for cross-platform builds.

### Missing `__init__.py` re-exports

If a package's `__init__.py` re-exports from sibling modules (`from .utils import helper`), those siblings are automatically included. The resolver iteratively scans all `__init__.py` files added to the package until no new files are discovered.
