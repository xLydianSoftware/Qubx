# Release & Deployment

This document describes how strategies are packaged for release and deployed to production environments.

## Overview

The release flow builds a strategy into a **pre-compiled wheel** with only the dependencies it actually uses, bundles private dependency wheels, and packages everything into a zip. The deploy flow installs from that zip — either into a venv (dev machines) or directly into system site-packages (Docker).

### Release ZIP Structure

```
release.zip
├── config.yml              # Strategy configuration
├── pyproject.toml          # Generated, minimal — just the strategy wheel as a dependency
├── uv.lock                 # For venv-based deploy (dev machines)
├── {StrategyName}.info     # Release metadata (tag, commit, author, date)
├── README.md
└── wheels/
    ├── xincubator-0.3.0-cp312-linux_x86_64.whl      # Compiled strategy (no source code)
    ├── quantkit-1.3.0.dev7-cp312-linux_x86_64.whl    # Private dep (not on PyPI)
    └── qubx-lighter-0.1.0-py3-none-any.whl           # Plugin (if needed)
```

No source code. No build.py. No unnecessary dependencies.

## Release Flow

```
qubx release -c <config.yaml> -o <output_dir> [<project_dir>]
```

### Step 1: Analyze strategy

- Load the YAML config and find the strategy classes + source files
- Scan all strategy `.py` files for external imports (top-level module names)
- Map import names to package names using `importlib.metadata`
- Cross-reference with the project's `uv.lock` to get exact pinned versions
- Result: a minimal list like `["cachetools==6.2.5", "pyyaml==6.0.3", "QuantKit==1.3.0.dev7"]`

### Step 2: Build strategy wheel

- Create a temporary directory
- Copy the entire source package (e.g. `src/xincubator/`)
- Generate a `pyproject.toml` with only the scanned deps + the source project's build system
- Copy `build.py` for Cython compilation
- Run `uv build --wheel .` — produces a compiled `.whl` with `.so` files, no `.py` source
- Move the wheel to `release_dir/wheels/`

### Step 3: Bundle private dependency wheels

For each dependency that is required by the strategy:

- **Path source** (local package like QuantKit): build wheel with `uv build --wheel`, bundle if not on public PyPI
- **Index source** (private registry like gtradex): download wheel with `pip download`, bundle if not on public PyPI
- **Public PyPI packages**: skip — they'll be resolved from PyPI at deploy time

### Step 4: Generate release pyproject.toml

A fresh, minimal `pyproject.toml` is generated (not copied from the source project):

```toml
[project]
name = "strategy-release"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["xincubator==0.3.0"]   # just the strategy wheel

[tool.uv]
package = false
find-links = ["./wheels"]
prerelease = "allow"
```

For configs that only reference external packages (no custom code), the strategy wheel is omitted and the external packages are listed directly as dependencies.

### Step 5: Generate lock + zip

- `uv lock` in the release directory to produce `uv.lock`
- Zip everything and clean up the temp directory

### External-deps-only configs

For configs like aggregators that only reference quantkit strategies (no xincubator code):

- No strategy wheel is built
- Release pyproject lists external packages directly as dependencies
- Private dependency wheels are still bundled

## Deploy Flow

```
qubx deploy <zip_file> [-o <output_dir>] [--force] [--system]
```

### Default mode (dev machines)

```
qubx deploy release.zip -o ~/deployed/strategy
```

1. Extracts the zip
2. Detects the package manager from lock file (uv or legacy poetry)
3. Ensures lock file exists (generates if missing)
4. Runs `uv sync` to create a venv and install all dependencies
5. Creates a `run_paper.sh` runner script

### System mode (Docker)

```
qubx deploy release.zip -o /app/strategy --system
```

1. Extracts the zip
2. Runs `pip install wheels/*.whl` directly into system site-packages
3. pip resolves public dependencies from PyPI, private deps from bundled wheels
4. No venv created, no uv needed

## Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim

ARG QUBX_VERSION

# No uv needed — wheels installed via pip
RUN pip install --no-cache-dir "qubx[k8]==${QUBX_VERSION}" boto3

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
```

### entrypoint.sh

The entrypoint:

1. Downloads the strategy artifact (from S3, HTTP, or local mount)
2. Deploys with `qubx deploy --system` (pip install, no venv)
3. Runs `qubx run` directly (no `uv run` needed since packages are in system site-packages)

### Environment variables

| Variable | Description |
|---|---|
| `STRATEGY_ARTIFACT_URL` | S3 URI, HTTP URL, or local path to the release zip |
| `STRATEGY_CONFIG_PATH` | Override config path (default: `/app/strategy/config.yml`) |
| `QUBX_PAPER` | Set to `true` for paper trading mode |

### Example: local Docker test

```bash
# Build the image
docker build --build-arg QUBX_VERSION=1.0.1.dev2 -t qubx-test .

# Run with a local release zip
docker run --rm \
  -e STRATEGY_ARTIFACT_URL=/releases/my_release.zip \
  -v /path/to/releases:/releases \
  qubx-test
```

## Import Scanning Details

The dependency scanner works as follows:

1. Parse each strategy source file's AST for all `import` / `from ... import` statements
2. Collect the set of top-level module names (e.g. `numpy`, `cachetools`, `qubx`)
3. For each dependency declared in `pyproject.toml`:
   - Look up its top-level import names via `importlib.metadata.distribution().read_text('top_level.txt')`
   - If any of those import names appear in the strategy's imports, include the dependency
4. Pin each matched dependency to the exact version from `uv.lock`

This ensures the strategy wheel declares only the dependencies it actually uses, not all 30+ from the source project.
