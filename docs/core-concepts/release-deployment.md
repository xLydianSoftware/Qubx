# Release & Deployment

This document describes how strategies are packaged for release and deployed to production environments.

## Overview

The release flow builds a strategy into a **pre-compiled wheel** with only the files and dependencies it actually uses, bundles private dependency wheels, and packages everything into a zip. The deploy flow installs from that zip — either into a venv (dev machines) or directly into system site-packages (Docker).

### Release ZIP Structure

```
release.zip
├── config.yml              # Strategy configuration
├── pyproject.toml          # Generated, minimal — just the strategy wheel as a dependency
├── uv.lock                 # For venv-based deploy (dev machines)
├── {StrategyName}.info     # Release metadata (tag, commit, author, date)
├── README.md
└── wheels/
    ├── myproject-0.3.0-cp312-linux_x86_64.whl    # Compiled strategy (no source code)
    ├── myprivatedep-1.3.0-cp312-linux_x86_64.whl # Private dep (not on PyPI)
    └── qubx-plugin-0.1.0-py3-none-any.whl        # Plugin (if needed)
```

No source code. No build.py. No unnecessary dependencies.

## Release Flow

```
qubx release -c <config.yaml> -o <output_dir> [<project_dir>]
```

### Step 1: Recursive import resolution

The `ModuleResolver` performs BFS from strategy entry point files, following all internal imports transitively:

1. Load the YAML config (without resolving `env:` variables) and find strategy classes + source files
2. For each entry point file, parse its AST for imports
3. For each internal import (e.g. `from myproject.utils.dataview import ...`), resolve it to a file on disk and enqueue
4. For `from pkg.utils import dataview` — also check if `dataview` is a submodule file (not just a symbol)
5. Continue BFS until no new files are discovered
6. Add all parent `__init__.py` files (needed for valid package structure)
7. **Re-scan newly added `__init__.py` files** — they may re-export siblings (e.g. `from .helper import ...`)
8. Repeat until stable — no new files found

Result: a precise set of internal files and external top-level import names.

**Edge cases handled:**
- **Circular imports**: visited set prevents infinite loops
- **Relative imports**: `from .sibling import X` resolved via `resolve_relative_import()`
- **`__init__.py` re-exports**: parent init files are scanned and their imports followed
- **`.pyx` files**: resolved by file extension, regex fallback for import extraction (AST fails on Cython)
- **Missing modules**: logged warning, no crash
- **Flat layout** (no `src/`): works with explicit `package_root`

### Step 2: Scan external dependencies

- Map external import names to package names using `importlib.metadata`
- Cross-reference with `pyproject.toml` declared dependencies
- Pin each matched dependency to the exact version from `uv.lock` (single source of truth)
- Result: a minimal list like `["cachetools==6.2.5", "pyyaml==6.0.3", "qubx[connectors]==1.0.6.dev1"]`

### Step 3: Build strategy wheel

- Create a temporary directory
- **Selectively copy** only the resolved internal files (not the entire source package)
- Generate a `pyproject.toml` with only the scanned deps + the source project's build system
- Copy `build.py` for Cython compilation
- Run `uv build --wheel .` — produces a compiled `.whl` with `.so` files, no `.py` source
- Move the wheel to `release_dir/wheels/`

### Step 4: Detect external strategy packages and plugins

**External strategy packages**: configs can reference strategy classes from external packages
(e.g. `extpkg.universe.basics.TopNUniverse`). These are detected regardless of whether there's
also local strategy code, and added as pinned dependencies.

**Plugins**: `plugins.modules` entries are resolved to package specs from optional-deps or
uv.lock and added as dependencies.

This handles three config types:
- **Local code only**: wheel built from source package, external deps scanned from all resolved files
- **External only**: no wheel, external packages listed as deps directly
- **Mixed** (local + external strategy classes): wheel built + external packages added alongside

### Step 5: Bundle private dependency wheels

For each dependency that is required by the strategy:

- **Path source** (local package): build wheel with `uv build --wheel`, bundle if not on public PyPI
- **Index source** (private registry): download wheel with `pip download`, bundle if not on public PyPI
- **Public PyPI packages**: skip — they'll be resolved from PyPI at deploy time

All versions come from `uv.lock` — no reliance on the current environment's installed packages.

### Step 6: Generate release pyproject.toml

A fresh, minimal `pyproject.toml` is generated (not copied from the source project):

```toml
[project]
name = "strategy-release"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["myproject==0.3.0", "extpkg==2.0.4"]  # wheel + external packages

[tool.uv]
package = false
find-links = ["./wheels"]
prerelease = "allow"
```

### Step 7: Generate lock + zip

- `uv lock` in the release directory to produce `uv.lock`
- Zip everything and clean up the temp directory

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

## Version Pinning

`uv.lock` is the single source of truth for all package versions during release. This ensures:

- Deterministic builds regardless of which environment runs the release
- No dependency on `importlib.metadata` (which reflects the current venv, not the project)
- If `uv.lock` is missing, it's generated automatically via `uv lock` in the source project

## Recursive Import Resolution Details

The `ModuleResolver` class (`qubx.cli.resolver`) resolves the minimal set of internal files needed:

```python
resolver = ModuleResolver(
    package_root="/path/to/src/mypackage",
    project_root="/path/to/project",
    package_name="mypackage",
)
internal_files, external_imports = resolver.resolve(entry_files)
```

### Resolution order for module paths

For `["mypackage", "utils", "dataview"]`:
1. Check `mypackage/utils/dataview.py`
2. Check `mypackage/utils/dataview.pyx`
3. Check `mypackage/utils/dataview/__init__.py`
4. Return None (external package)

### What gets included

- All files reachable via imports from entry points
- All parent `__init__.py` files (for valid package structure)
- All files imported by those `__init__.py` files (recursive)

### What doesn't get included

- Files in the source package not reachable from entry points
- Test files, research notebooks, unrelated models
- External packages (tracked as dependencies, not copied)
