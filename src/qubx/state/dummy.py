"""
Dummy (no-op) implementation of state persistence.

Used as default when no persistence backend is configured.
"""

from typing import Any

from qubx.core.interfaces import IStatePersistence


class DummyStatePersistence(IStatePersistence):
    """
    No-op implementation of state persistence.

    This implementation stores nothing and always returns defaults.
    Used as the default when no persistence backend is configured,
    similar to DummyHealthMonitor.
    """

    def save(self, key: str, value: Any) -> None:
        """No-op save - does nothing."""
        pass

    def load(self, key: str, default: Any = None) -> Any:
        """Always returns the default value."""
        return default

    def delete(self, key: str) -> bool:
        """Always returns False (nothing to delete)."""
        return False

    def exists(self, key: str) -> bool:
        """Always returns False (nothing exists)."""
        return False
