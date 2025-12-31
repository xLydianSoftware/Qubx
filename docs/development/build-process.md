# Build Process

Qubx uses a hybrid build system that compiles both **Cython** and **Rust** extensions alongside pure Python code. This document explains how the build process works and how to build, install, and develop with native extensions.

## Overview

The build system consists of:

- **Poetry** - Python package manager and build orchestrator
- **Cython** - Compiles `.pyx` files to C extensions for performance-critical Python code
- **Rust/PyO3** - Compiles Rust code to Python extensions via maturin
- **Custom `build.py`** - Orchestrates both Cython and Rust compilation

## Directory Structure

```
Qubx/
├── Cargo.toml                  # Rust workspace manifest (for rust-analyzer)
├── target/                     # Rust build output (shared workspace)
├── src/qubx/
│   ├── core/
│   │   ├── series.pyx          # Cython source
│   │   ├── series.pxd          # Cython header
│   │   ├── series.pyi          # Type stubs
│   │   └── *.so                # Compiled extension
│   ├── ta/
│   │   └── indicators.pyx      # Technical analysis indicators
│   ├── utils/
│   │   ├── ringbuffer.pyx
│   │   └── hft/orderbook.pyx
│   └── _rust/
│       ├── __init__.py         # Python wrapper
│       ├── __init__.pyi        # Type stubs
│       └── qubx_rust.*.so      # Compiled Rust extension
├── rust/
│   └── qubx_rust/
│       ├── Cargo.toml          # Rust crate manifest
│       └── src/lib.rs          # Rust source code
├── build.py                    # Build orchestration script
└── pyproject.toml              # Poetry/build configuration
```

The root `Cargo.toml` defines a Rust workspace, allowing rust-analyzer to properly index the code and providing a shared `target/` directory for builds.

## Build Commands

### Full Build (Recommended)

```bash
# Clean build - removes old artifacts and rebuilds everything
just build
```

This command:
1. Removes the `build/` directory
2. Removes old `.pyd` files
3. Runs `poetry build` which triggers `build.py`

### Development Build

```bash
# Install in development mode with all extensions compiled
poetry install
```

### Fast Build (Skip Cython)

```bash
# Use pre-compiled Cython binaries if they exist
just build-fast
```

### Compile Only (No Wheel)

```bash
# Just compile Cython and Rust without building a wheel
just compile
```

### Rust Only

```bash
# Compile Rust code only
just compile-rust

# Run Rust tests
just test-rust
```

## How the Build Works

### 1. Build Trigger

When you run `poetry build` or `poetry install`, Poetry executes `build.py` as specified in `pyproject.toml`:

```toml
[tool.poetry.build]
script = "build.py"
```

### 2. Rust Compilation

The `_build_rust_libs()` function in `build.py`:

1. Locates the Rust crate at `rust/qubx_rust/`
2. Invokes `maturin build` to compile the Rust code
3. Extracts the compiled `.so` file from the wheel
4. Copies it to `src/qubx/_rust/`

```python
# Simplified flow
maturin build --manifest-path rust/qubx_rust/Cargo.toml --release
# Output: rust/qubx_rust/target/wheels/qubx_rust-*.whl
# Extract .so and copy to src/qubx/_rust/
```

### 3. Cython Compilation

The Cython build process:

1. Discovers all `.pyx` files in `src/qubx/`
2. Transpiles them to C using Cython
3. Compiles C to shared objects using the system compiler
4. Copies `.so` files back to source tree

Key compiler directives:
```python
CYTHON_COMPILER_DIRECTIVES = {
    "language_level": "3",
    "cdivision": True,          # 35% speed improvement
    "nonecheck": True,
    "embedsignature": True,
}
```

### 4. Symbol Stripping

For release builds on Linux/macOS, unneeded symbols are stripped from binaries to reduce size.

## Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_MODE` | `release` | Build profile (`release` or `debug`) |
| `PROFILE_MODE` | `""` | Enable profiling/tracing |
| `ANNOTATION_MODE` | `""` | Generate Cython HTML annotations |
| `PYO3_ONLY` | `false` | Skip Cython, use existing binaries |
| `COPY_TO_SOURCE` | `true` | Copy compiled files to source tree |

Example:
```bash
# Debug build with profiling
BUILD_MODE=debug PROFILE_MODE=1 poetry build
```

## Adding New Cython Modules

1. Create a `.pyx` file in the appropriate directory:
   ```python
   # src/qubx/mymodule/fast_ops.pyx
   def fast_function(double[:] data):
       cdef int i
       cdef double total = 0.0
       for i in range(data.shape[0]):
           total += data[i]
       return total
   ```

2. (Optional) Create a `.pxd` header for cimports from other modules

3. Create a `.pyi` stub for type hints:
   ```python
   # src/qubx/mymodule/fast_ops.pyi
   import numpy.typing as npt
   def fast_function(data: npt.NDArray) -> float: ...
   ```

4. The build system auto-discovers `.pyx` files - just run `just compile`

## Adding New Rust Functions

1. Add functions to `rust/qubx_rust/src/lib.rs`:
   ```rust
   #[pyfunction]
   fn my_function(x: f64) -> f64 {
       x * 2.0
   }

   #[pymodule]
   fn qubx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add_function(wrap_pyfunction!(my_function, m)?)?;
       Ok(())
   }
   ```

2. Export from Python wrapper `src/qubx/_rust/__init__.py`:
   ```python
   from qubx._rust.qubx_rust import fibonacci, my_function
   __all__ = ["fibonacci", "my_function"]
   ```

3. Add type stub to `src/qubx/_rust/__init__.pyi`:
   ```python
   def my_function(x: float) -> float: ...
   ```

4. Run `just compile` or `poetry build`

## Testing Native Extensions

### Python Tests
```bash
# Run all Python tests
just test

# Test specific module
poetry run pytest tests/qubx/core/test_series.py -v
```

### Rust Tests
```bash
# Run Rust unit tests
just test-rust
```

## Troubleshooting

### "maturin not found"

Install maturin in your dev environment:
```bash
poetry install --with dev
```

### Cython compilation errors

Check that you have a C compiler installed:
```bash
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install
```

### Rust compilation errors

Ensure Rust is installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Import errors after rebuild

Clear Python's import cache:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
poetry install
```

## Build Artifacts

After a successful build, you'll find:

- `dist/qubx-*.whl` - Wheel package with all compiled extensions
- `dist/qubx-*.tar.gz` - Source distribution
- `src/qubx/**/*.so` - Compiled extensions in source tree (for development)
- `build/optimized/` - Intermediate Cython build files
- `target/` - Rust workspace build cache (shared by all crates)

## Performance Tips

1. **Use `just build-fast`** during development if Cython modules haven't changed
2. **Run `just compile`** for quick iteration on native code
3. **Keep Rust's `target/` directory** - incremental compilation is fast
4. The build uses all available CPU cores for parallel compilation
