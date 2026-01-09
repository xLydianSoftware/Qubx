"""
Integration tests for RedisStatePersistence.

These tests require Redis to be running.
"""

import time

import pytest
import redis

from qubx.state import RedisStatePersistence
from qubx.state.interfaces import IStatePersistence


@pytest.fixture
def state_persistence(redis_service):
    """Create a RedisStatePersistence instance for testing."""
    return RedisStatePersistence(
        redis_url=redis_service,
        strategy_name="test_strategy",
    )


@pytest.mark.integration
class TestRedisStatePersistence:
    """Integration tests for RedisStatePersistence."""

    def test_implements_interface(self, state_persistence):
        """Test that RedisStatePersistence implements IStatePersistence."""
        assert isinstance(state_persistence, IStatePersistence)

    def test_save_and_load_string(self, state_persistence, clear_state_keys):
        """Test saving and loading a string value."""
        state_persistence.save("test_key", "test_value")

        result = state_persistence.load("test_key")
        assert result == "test_value"

    def test_save_and_load_dict(self, state_persistence, clear_state_keys):
        """Test saving and loading a dictionary."""
        data = {"signal": 1.0, "timestamp": "2024-01-01T00:00:00", "instruments": ["BTC", "ETH"]}
        state_persistence.save("complex_key", data)

        result = state_persistence.load("complex_key")
        assert result == data

    def test_save_and_load_list(self, state_persistence, clear_state_keys):
        """Test saving and loading a list."""
        data = [1, 2, 3, "four", {"five": 5}]
        state_persistence.save("list_key", data)

        result = state_persistence.load("list_key")
        assert result == data

    def test_save_and_load_number(self, state_persistence, clear_state_keys):
        """Test saving and loading numeric values."""
        state_persistence.save("int_key", 42)
        state_persistence.save("float_key", 3.14159)

        assert state_persistence.load("int_key") == 42
        assert state_persistence.load("float_key") == 3.14159

    def test_save_and_load_bool(self, state_persistence, clear_state_keys):
        """Test saving and loading boolean values."""
        state_persistence.save("bool_true", True)
        state_persistence.save("bool_false", False)

        assert state_persistence.load("bool_true") is True
        assert state_persistence.load("bool_false") is False

    def test_save_and_load_null(self, state_persistence, clear_state_keys):
        """Test saving and loading null values."""
        state_persistence.save("null_key", None)

        result = state_persistence.load("null_key")
        assert result is None

    def test_load_nonexistent_returns_none(self, state_persistence, clear_state_keys):
        """Test that loading a nonexistent key returns None."""
        result = state_persistence.load("nonexistent")
        assert result is None

    def test_load_nonexistent_returns_default(self, state_persistence, clear_state_keys):
        """Test that loading a nonexistent key returns the provided default."""
        assert state_persistence.load("nonexistent", default="fallback") == "fallback"
        assert state_persistence.load("nonexistent", default=42) == 42
        assert state_persistence.load("nonexistent", default=[1, 2, 3]) == [1, 2, 3]

    def test_exists_false_for_nonexistent(self, state_persistence, clear_state_keys):
        """Test that exists returns False for nonexistent keys."""
        assert state_persistence.exists("nonexistent") is False

    def test_exists_true_after_save(self, state_persistence, clear_state_keys):
        """Test that exists returns True after saving."""
        assert state_persistence.exists("test_key") is False

        state_persistence.save("test_key", "value")

        assert state_persistence.exists("test_key") is True

    def test_delete_returns_true_for_existing(self, state_persistence, clear_state_keys):
        """Test that delete returns True for existing keys."""
        state_persistence.save("to_delete", "value")

        assert state_persistence.delete("to_delete") is True
        assert state_persistence.exists("to_delete") is False

    def test_delete_returns_false_for_nonexistent(self, state_persistence, clear_state_keys):
        """Test that delete returns False for nonexistent keys."""
        assert state_persistence.delete("nonexistent") is False

    def test_overwrite_value(self, state_persistence, clear_state_keys):
        """Test that saving to an existing key overwrites the value."""
        state_persistence.save("key", "original")
        assert state_persistence.load("key") == "original"

        state_persistence.save("key", "updated")
        assert state_persistence.load("key") == "updated"

    def test_key_namespacing(self, redis_service, clear_state_keys):
        """Test that keys are properly namespaced by strategy name."""
        persistence1 = RedisStatePersistence(
            redis_url=redis_service,
            strategy_name="strategy_a",
        )
        persistence2 = RedisStatePersistence(
            redis_url=redis_service,
            strategy_name="strategy_b",
        )

        persistence1.save("shared_key", "value_a")
        persistence2.save("shared_key", "value_b")

        assert persistence1.load("shared_key") == "value_a"
        assert persistence2.load("shared_key") == "value_b"

        # Verify actual Redis keys
        r = redis.from_url(redis_service)
        assert r.exists("state:strategy_a:shared_key")
        assert r.exists("state:strategy_b:shared_key")

        # Cleanup
        r.delete("state:strategy_a:shared_key")
        r.delete("state:strategy_b:shared_key")

    def test_custom_key_prefix(self, redis_service, clear_state_keys):
        """Test that custom key prefix is respected."""
        persistence = RedisStatePersistence(
            redis_url=redis_service,
            strategy_name="test_strategy",
            key_prefix="custom_prefix",
        )

        persistence.save("my_key", "my_value")

        # Verify the key was created with custom prefix
        r = redis.from_url(redis_service)
        assert r.exists("custom_prefix:test_strategy:my_key")
        assert not r.exists("state:test_strategy:my_key")

        # Cleanup
        r.delete("custom_prefix:test_strategy:my_key")

    def test_ttl_support(self, redis_service, clear_state_keys):
        """Test that TTL is properly set when configured."""
        persistence = RedisStatePersistence(
            redis_url=redis_service,
            strategy_name="test_strategy",
            ttl_seconds=2,  # 2 seconds TTL
        )

        persistence.save("ttl_key", "will_expire")

        # Verify key exists and has TTL
        r = redis.from_url(redis_service)
        ttl = r.ttl("state:test_strategy:ttl_key")
        assert 0 < ttl <= 2

        # Wait for expiry
        time.sleep(3)

        assert persistence.exists("ttl_key") is False
        assert persistence.load("ttl_key") is None

    def test_no_ttl_by_default(self, redis_service, clear_state_keys):
        """Test that keys have no TTL when not configured."""
        persistence = RedisStatePersistence(
            redis_url=redis_service,
            strategy_name="test_strategy",
        )

        persistence.save("no_ttl_key", "value")

        r = redis.from_url(redis_service)
        ttl = r.ttl("state:test_strategy:no_ttl_key")
        assert ttl == -1  # -1 means no expiry

    def test_complex_nested_data(self, state_persistence, clear_state_keys):
        """Test saving and loading complex nested data structures."""
        data = {
            "signals": [
                {"instrument": "BTC-USDT", "signal": 1.0, "price": 50000.0},
                {"instrument": "ETH-USDT", "signal": -0.5, "price": 3000.0},
            ],
            "metadata": {
                "timestamp": "2024-01-01T00:00:00",
                "version": 1,
                "config": {"leverage": 2.0, "enabled": True},
            },
            "counters": {"trades": 10, "errors": 0},
        }
        state_persistence.save("complex_data", data)

        result = state_persistence.load("complex_data")
        assert result == data

    def test_multiple_keys(self, state_persistence, clear_state_keys):
        """Test saving and loading multiple keys."""
        state_persistence.save("key1", "value1")
        state_persistence.save("key2", "value2")
        state_persistence.save("key3", "value3")

        assert state_persistence.load("key1") == "value1"
        assert state_persistence.load("key2") == "value2"
        assert state_persistence.load("key3") == "value3"

        # Delete one and verify others are unaffected
        state_persistence.delete("key2")
        assert state_persistence.load("key1") == "value1"
        assert state_persistence.load("key2") is None
        assert state_persistence.load("key3") == "value3"
