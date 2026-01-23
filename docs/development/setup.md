# Development Setup

This guide covers setting up a development environment for Qubx, including dependency management, building, and the project's build system architecture.

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- C compiler (gcc, clang, or MSVC) - Required for Cython compilation
- [just](https://just.systems/) - Command runner (optional but recommended)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/xLydianSoftware/Qubx.git
cd Qubx

# Install dependencies + dev tools
uv sync

# Compile Cython extensions
just compile

# Run tests
just test
```

## Dependency Management

Qubx uses **uv** for dependency management with a hybrid build system.

### Installing Dependencies

```bash
# Install main dependencies + dev tools
uv sync

# Install with optional runtime features
uv sync --extra k8        # Kubernetes/Prometheus support
uv sync --extra hft       # HFT backtesting support
uv sync --all-extras      # All optional features

# Update lock file after changing pyproject.toml
uv lock
```

### Dependency Structure

The project uses two types of optional dependencies:

| Section | Purpose | Shipped to PyPI | Install with |
|---------|---------|-----------------|--------------|
| `[project.optional-dependencies]` | Optional **runtime** features | Yes | `pip install qubx[k8]` |
| `[dependency-groups]` | **Dev/test** tools | No | `uv sync` |

**Runtime optional features** (`[project.optional-dependencies]`):

- `k8` - Prometheus client for Kubernetes deployments
- `hft` - HFT backtesting support

**Dev tools** (`[dependency-groups].dev`):

- pytest, ruff, pre-commit, mkdocs, etc.
- Not shipped with the package, only for development

### End User Installation

```bash
# Basic installation
pip install qubx

# With optional features
pip install qubx[k8]
pip install qubx[hft]
pip install qubx[k8,hft]  # Multiple features
```

## Build System

### Architecture

Qubx uses a **hybrid build system**:

```
┌─────────────────────────────────────────────────────────────┐
│                      pyproject.toml                          │
├─────────────────────────────────────────────────────────────┤
│  build-backend = "poetry.core.masonry.api"                  │
│                                                              │
│  [tool.poetry.build]                                        │
│  script = "build.py"  ←── Runs during build                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        build.py                              │
├─────────────────────────────────────────────────────────────┤
│  • Compiles Cython extensions (.pyx → .so)                  │
│  • Copies compiled files to source tree                     │
│  • Strips debug symbols (release builds)                    │
│  • (Future: Rust/PyO3 compilation)                          │
└─────────────────────────────────────────────────────────────┘
```

**Why poetry-core?**

We use `poetry.core.masonry.api` as the build backend because it supports `script = "build.py"` - running custom build scripts during package building. Plain setuptools doesn't have this feature, and uv doesn't yet support custom build scripts natively.

This is the same approach used by [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader).

### Build Commands

```bash
# Compile Cython extensions (for development, no wheel)
just compile
# or
uv run python build.py

# Build package (compiles Cython + creates wheel)
uv build
# or
just build
```

### What Gets Compiled

The project has several Cython modules:

| File | Purpose |
|------|---------|
| `src/qubx/core/series.pyx` | High-performance time series |
| `src/qubx/core/utils.pyx` | Core utilities |
| `src/qubx/ta/indicators.pyx` | Technical analysis indicators |
| `src/qubx/utils/ringbuffer.pyx` | Ring buffer implementation |
| `src/qubx/utils/hft/orderbook.pyx` | Order book for HFT |

### Build Modes

Control build behavior with environment variables:

```bash
# Release build (default) - optimized
BUILD_MODE=release uv run python build.py

# Debug build - with symbols
BUILD_MODE=debug uv run python build.py

# Profile mode - for profiling
PROFILE_MODE=true uv run python build.py

# Annotation mode - generate HTML annotations
ANNOTATION_MODE=true uv run python build.py
```

## Package Distribution

### Building for Distribution

```bash
# Build source distribution + wheel
uv build

# Output:
# dist/qubx-x.y.z.tar.gz           (source)
# dist/qubx-x.y.z-cp312-...-whl    (wheel with compiled extensions)
```

### Platform-Specific Wheels

The wheel contains **pre-compiled** Cython extensions. Users on matching platforms get the wheel directly (no compilation needed).

Current limitation: We only build wheels for the CI platform. Users on other platforms must compile from source (requires C compiler).

See [GitHub Issue #123](https://github.com/xLydianSoftware/Qubx/issues/123) for adding cibuildwheel to build wheels for all platforms.

### Source Installation Flow

When no matching wheel exists, pip/uv builds from source:

```
pip install qubx
       │
       ▼
Download source distribution (.tar.gz)
       │
       ▼
Read pyproject.toml → build-backend = "poetry.core.masonry.api"
       │
       ▼
poetry-core runs build.py
       │
       ▼
Cython compiles .pyx → .so
       │
       ▼
Package installed with compiled extensions
```

**Requirements for source install:**

- C compiler (gcc, clang, or MSVC)
- Python development headers

## Future: Rust/PyO3 Support

The build system is designed to support Rust extensions via PyO3. When adding Rust:

1. Add `cargo build` to `build.py`
2. Link Cython extensions to Rust static libraries
3. Copy PyO3 dynamic libraries to the package

## Common Tasks

### Running Tests

```bash
# Unit tests (parallel)
just test

# Verbose output
just test-verbose

# Specific test file
uv run pytest tests/path/to/test.py -v

# With coverage
uv run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Lint and format check
just style-check

# Auto-fix with ruff
uv run ruff check --fix .
uv run ruff format .
```

### Building Documentation

```bash
# Update docs
just update-docs

# Serve locally
uv run mkdocs serve
```

## Troubleshooting

### "No module named 'qubx.core.series'"

Cython extensions aren't compiled. Run:

```bash
just compile
```

### Build fails with "gcc not found"

Install a C compiler:

```bash
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

### Lock file conflicts

Regenerate the lock file:

```bash
uv lock
```
