"""
State persistence module for storing and retrieving strategy state.

This module provides interfaces and implementations for persisting strategy
state to external storage backends.
"""

from qubx.state.dummy import DummyStatePersistence
from qubx.state.redis import RedisStatePersistence

__all__ = [
    "DummyStatePersistence",
    "RedisStatePersistence",
]
