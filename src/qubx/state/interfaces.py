"""
State persistence interfaces for storing and retrieving strategy state.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IStatePersistence(Protocol):
    """
    Interface for persisting strategy state to external storage.

    This interface provides a simple key-value store abstraction that can be
    implemented by different backends (Redis, file system, database, etc.).

    All keys are automatically namespaced by strategy name to prevent collisions.
    """

    def save(self, key: str, value: Any) -> None:
        """
        Save a value to persistent storage.

        Args:
            key: The key to store the value under (will be namespaced by strategy)
            value: The value to store (must be JSON-serializable)

        Raises:
            TypeError: If the value cannot be serialized
            ConnectionError: If the backend is unavailable
        """
        ...

    def load(self, key: str, default: Any = None) -> Any:
        """
        Load a value from persistent storage.

        Args:
            key: The key to load (will be namespaced by strategy)
            default: Value to return if key doesn't exist (default: None)

        Returns:
            The stored value, or the default if the key doesn't exist

        Raises:
            ConnectionError: If the backend is unavailable
        """
        ...

    def delete(self, key: str) -> bool:
        """
        Delete a key from persistent storage.

        Args:
            key: The key to delete (will be namespaced by strategy)

        Returns:
            True if the key existed and was deleted, False if it didn't exist

        Raises:
            ConnectionError: If the backend is unavailable
        """
        ...

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in persistent storage.

        Args:
            key: The key to check (will be namespaced by strategy)

        Returns:
            True if the key exists, False otherwise

        Raises:
            ConnectionError: If the backend is unavailable
        """
        ...
