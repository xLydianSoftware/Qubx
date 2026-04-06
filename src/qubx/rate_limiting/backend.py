"""
Rate limit storage backends.

Provides the interface and in-memory implementation for token bucket state.
Redis backend can be added later for cross-bot coordination.
"""

import time
from abc import ABC, abstractmethod

from qubx.utils.rate_limiter import TokenBucketRateLimiter


class IRateLimitBackend(ABC):
    """Interface for rate limit token storage.

    Backends manage token bucket state — either in-process (InMemoryBackend)
    or distributed (RedisBackend for cross-bot coordination).
    """

    @abstractmethod
    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        """Acquire tokens from a bucket. Blocks until available.

        Args:
            key: Unique bucket identifier (e.g., "ratelimit:binance:rest:ip_1.2.3.4")
            weight: Tokens to consume
            capacity: Bucket capacity (used to create bucket on first access)
            refill_rate: Tokens per second (used to create bucket on first access)

        Returns:
            Time spent waiting in seconds (0.0 if immediate)
        """

    @abstractmethod
    async def get_remaining(self, key: str) -> float | None:
        """Get remaining tokens for a key (non-blocking).

        Returns:
            Token count, or None if key doesn't exist
        """

    @abstractmethod
    async def set_remaining(self, key: str, remaining: float) -> None:
        """Force-set remaining tokens (for syncing with exchange-reported state).

        Used when exchange headers/responses tell us the actual remaining budget,
        correcting any drift in our model.
        """


class InMemoryBackend(IRateLimitBackend):
    """In-process token bucket backend.

    Each key gets its own TokenBucketRateLimiter instance.
    Suitable for single-bot usage without cross-bot coordination.
    """

    def __init__(self):
        self._limiters: dict[str, TokenBucketRateLimiter] = {}

    def _get_or_create(self, key: str, capacity: float, refill_rate: float) -> TokenBucketRateLimiter:
        if key not in self._limiters:
            self._limiters[key] = TokenBucketRateLimiter(capacity, refill_rate, name=key)
        return self._limiters[key]

    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        limiter = self._get_or_create(key, capacity, refill_rate)
        start = time.monotonic()
        await limiter.acquire(weight)
        return time.monotonic() - start

    async def get_remaining(self, key: str) -> float | None:
        limiter = self._limiters.get(key)
        if limiter is None:
            return None
        return limiter.get_available_tokens()

    async def set_remaining(self, key: str, remaining: float) -> None:
        limiter = self._limiters.get(key)
        if limiter is not None:
            limiter._tokens = remaining
            limiter._last_refill = time.monotonic()
