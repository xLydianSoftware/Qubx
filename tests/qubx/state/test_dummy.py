"""
Unit tests for DummyStatePersistence.
"""

import pytest

from qubx.state import DummyStatePersistence
from qubx.state.interfaces import IStatePersistence


class TestDummyStatePersistence:
    """Tests for DummyStatePersistence."""

    def test_implements_interface(self):
        """Test that DummyStatePersistence implements IStatePersistence."""
        persistence = DummyStatePersistence()
        assert isinstance(persistence, IStatePersistence)

    def test_save_does_nothing(self):
        """Test that save is a no-op."""
        persistence = DummyStatePersistence()
        # Should not raise
        persistence.save("key", {"data": "value"})
        persistence.save("another_key", [1, 2, 3])
        persistence.save("number", 42)

    def test_load_returns_none_by_default(self):
        """Test that load returns None when no default is provided."""
        persistence = DummyStatePersistence()
        assert persistence.load("nonexistent") is None

    def test_load_returns_default(self):
        """Test that load returns the provided default value."""
        persistence = DummyStatePersistence()

        assert persistence.load("nonexistent", default="fallback") == "fallback"
        assert persistence.load("nonexistent", default=42) == 42
        assert persistence.load("nonexistent", default=[1, 2, 3]) == [1, 2, 3]
        assert persistence.load("nonexistent", default={"key": "value"}) == {"key": "value"}

    def test_delete_returns_false(self):
        """Test that delete always returns False."""
        persistence = DummyStatePersistence()

        assert persistence.delete("any_key") is False
        assert persistence.delete("another_key") is False

    def test_exists_returns_false(self):
        """Test that exists always returns False."""
        persistence = DummyStatePersistence()

        assert persistence.exists("any_key") is False
        assert persistence.exists("another_key") is False

    def test_save_then_load_returns_default(self):
        """Test that save doesn't actually persist anything."""
        persistence = DummyStatePersistence()

        persistence.save("key", {"important": "data"})
        assert persistence.load("key") is None
        assert persistence.load("key", default="fallback") == "fallback"

    def test_save_then_exists_returns_false(self):
        """Test that save doesn't make key exist."""
        persistence = DummyStatePersistence()

        persistence.save("key", "value")
        assert persistence.exists("key") is False

    def test_multiple_operations(self):
        """Test multiple operations in sequence."""
        persistence = DummyStatePersistence()

        # Save multiple values
        persistence.save("key1", "value1")
        persistence.save("key2", {"nested": "data"})
        persistence.save("key3", [1, 2, 3])

        # All loads should return defaults
        assert persistence.load("key1") is None
        assert persistence.load("key2", default={}) == {}
        assert persistence.load("key3", default=[]) == []

        # All exists should return False
        assert persistence.exists("key1") is False
        assert persistence.exists("key2") is False
        assert persistence.exists("key3") is False

        # All deletes should return False
        assert persistence.delete("key1") is False
        assert persistence.delete("key2") is False
        assert persistence.delete("key3") is False
