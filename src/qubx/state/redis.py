"""
Redis implementation of state persistence.
"""

import json
from typing import Any, cast

import redis

from qubx import logger
from qubx.state.interfaces import IStatePersistence


class RedisStatePersistence(IStatePersistence):
    """
    Redis-backed state persistence implementation.

    Stores strategy state as JSON-serialized values in Redis with optional TTL.
    Keys are namespaced using the pattern: {prefix}:{strategy_name}:{user_key}

    Example usage:
        persistence = RedisStatePersistence(
            redis_url="redis://localhost:6379/0",
            strategy_name="my_strategy",
            ttl_seconds=86400  # Optional: expire keys after 24 hours
        )

        # In strategy
        ctx.persistence.save("last_signal_time", str(ctx.time()))
        last_time = ctx.persistence.load("last_signal_time")
    """

    def __init__(
        self,
        redis_url: str,
        strategy_name: str,
        key_prefix: str = "state",
        ttl_seconds: int | None = None,
    ):
        """
        Initialize Redis state persistence.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            strategy_name: Name of the strategy (used in key namespace)
            key_prefix: Prefix for all keys (default: "state")
            ttl_seconds: Optional TTL in seconds for all keys (None = no expiry)
        """
        self._redis = redis.from_url(redis_url)
        self._strategy_name = strategy_name
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

        logger.info(
            f"[RedisStatePersistence] Initialized for strategy '{strategy_name}' "
            f"with prefix '{key_prefix}' and TTL={ttl_seconds}s"
        )

    def _make_key(self, key: str) -> str:
        """
        Create a namespaced Redis key.

        Args:
            key: User-provided key

        Returns:
            Fully qualified Redis key: {prefix}:{strategy_name}:{key}
        """
        return f"{self._key_prefix}:{self._strategy_name}:{key}"

    def save(self, key: str, value: Any) -> None:
        """
        Save a JSON-serialized value to Redis.

        Args:
            key: The key to store the value under
            value: The value to store (must be JSON-serializable)

        Raises:
            TypeError: If value is not JSON-serializable
            redis.RedisError: If Redis operation fails
        """
        full_key = self._make_key(key)
        try:
            serialized = json.dumps(value)
            if self._ttl_seconds is not None:
                self._redis.setex(full_key, self._ttl_seconds, serialized)
            else:
                self._redis.set(full_key, serialized)
            logger.debug(f"[RedisStatePersistence] Saved key '{full_key}'")
        except (TypeError, ValueError) as e:
            logger.error(f"[RedisStatePersistence] Failed to serialize value for key '{key}': {e}")
            raise
        except redis.RedisError as e:
            logger.error(f"[RedisStatePersistence] Redis error saving key '{key}': {e}")
            raise

    def load(self, key: str, default: Any = None) -> Any:
        """
        Load a value from Redis and deserialize from JSON.

        Args:
            key: The key to load
            default: Value to return if key doesn't exist

        Returns:
            The deserialized value, or default if key doesn't exist

        Raises:
            json.JSONDecodeError: If stored value is not valid JSON
            redis.RedisError: If Redis operation fails
        """
        full_key = self._make_key(key)
        try:
            value = cast(bytes | None, self._redis.get(full_key))
            if value is None:
                return default
            return json.loads(value.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"[RedisStatePersistence] Failed to deserialize value for key '{key}': {e}")
            raise
        except redis.RedisError as e:
            logger.error(f"[RedisStatePersistence] Redis error loading key '{key}': {e}")
            raise

    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: The key to delete

        Returns:
            True if the key existed and was deleted, False otherwise

        Raises:
            redis.RedisError: If Redis operation fails
        """
        full_key = self._make_key(key)
        try:
            result = cast(int, self._redis.delete(full_key))
            deleted = bool(result)
            if deleted:
                logger.debug(f"[RedisStatePersistence] Deleted key '{full_key}'")
            return deleted
        except redis.RedisError as e:
            logger.error(f"[RedisStatePersistence] Redis error deleting key '{key}': {e}")
            raise

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise

        Raises:
            redis.RedisError: If Redis operation fails
        """
        full_key = self._make_key(key)
        try:
            return bool(self._redis.exists(full_key))
        except redis.RedisError as e:
            logger.error(f"[RedisStatePersistence] Redis error checking key '{key}': {e}")
            raise
