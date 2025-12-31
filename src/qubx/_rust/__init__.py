# Re-export Rust functions for clean Python API
from qubx._rust.qubx_rust import fibonacci

__all__ = ["fibonacci"]
