"""
Generic rate limiting utilities using token bucket algorithm.

This module provides reusable rate limiting functionality that can be used
across different exchange connectors to comply with API rate limits.
"""

import asyncio
import time
from functools import wraps
from typing import Callable, Optional

from qubx import logger


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for async operations.

    The token bucket algorithm allows bursts of requests up to the capacity,
    then enforces the average rate defined by refill_rate.

    Usage:
        >>> limiter = TokenBucketRateLimiter(capacity=100, refill_rate=100)
        >>> await limiter.acquire(weight=10)  # Consumes 10 tokens
    """

    def __init__(self, capacity: float, refill_rate: float, name: Optional[str] = None):
        """
        Initialize token bucket rate limiter.

        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Tokens added per second
            name: Optional name for logging
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.name = name or "rate_limiter"

        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Calculate tokens to add based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    async def acquire(self, weight: float = 1.0) -> None:
        """
        Acquire tokens from the bucket.

        If not enough tokens are available, waits until sufficient tokens
        have been refilled.

        Args:
            weight: Number of tokens to acquire (default: 1.0)

        Raises:
            ValueError: If weight exceeds bucket capacity
        """
        if weight > self.capacity:
            raise ValueError(f"Requested weight {weight} exceeds bucket capacity {self.capacity} for {self.name}")

        async with self._lock:
            while True:
                # Refill tokens based on elapsed time
                self._refill_tokens()

                # Check if we have enough tokens
                if self._tokens >= weight:
                    self._tokens -= weight
                    return

                # Calculate wait time for sufficient tokens
                tokens_needed = weight - self._tokens
                wait_time = tokens_needed / self.refill_rate

                # Log warning for significant delays
                if wait_time > 1.0:
                    logger.debug(
                        f"Rate limiter {self.name}: waiting {wait_time:.2f}s "
                        f"(need {tokens_needed:.1f} tokens, refill rate: {self.refill_rate}/s)"
                    )

                # Release lock while waiting to allow other tasks to check
                await asyncio.sleep(wait_time)

    def get_available_tokens(self) -> float:
        """
        Get current number of available tokens (non-blocking).

        Returns:
            Current token count
        """
        self._refill_tokens()
        return self._tokens


class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters.

    Allows organizing different rate limiters by key (e.g., "rest", "ws_sub")
    within a single instance.

    Usage:
        >>> registry = RateLimiterRegistry()
        >>> registry.register_limiter("rest", TokenBucketRateLimiter(100, 100))
        >>> limiter = registry.get_limiter("rest")
        >>> await limiter.acquire(10)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._limiters: dict[str, TokenBucketRateLimiter] = {}

    def register_limiter(self, key: str, limiter: TokenBucketRateLimiter) -> None:
        """
        Register a rate limiter with a key.

        Args:
            key: Identifier for the limiter (e.g., "rest", "ws_sub")
            limiter: TokenBucketRateLimiter instance
        """
        self._limiters[key] = limiter
        logger.debug(f"Registered rate limiter '{key}': {limiter.name}")

    def get_limiter(self, key: str) -> TokenBucketRateLimiter:
        """
        Get a rate limiter by key.

        Args:
            key: Identifier for the limiter

        Returns:
            TokenBucketRateLimiter instance

        Raises:
            KeyError: If limiter not found
        """
        if key not in self._limiters:
            raise KeyError(f"Rate limiter '{key}' not found in registry")
        return self._limiters[key]

    def has_limiter(self, key: str) -> bool:
        """
        Check if a limiter is registered.

        Args:
            key: Identifier for the limiter

        Returns:
            True if limiter exists
        """
        return key in self._limiters

    def list_limiters(self) -> list[str]:
        """
        List all registered limiter keys.

        Returns:
            List of limiter keys
        """
        return list(self._limiters.keys())


def rate_limited(limiter_key: str, weight: float = 1.0) -> Callable:
    """
    Decorator for rate limiting async methods.

    The decorated method's instance must have a `_rate_limiters` attribute
    of type RateLimiterRegistry.

    Args:
        limiter_key: Key to look up rate limiter in the registry
        weight: Number of tokens to acquire (default: 1.0)

    Usage:
        >>> class MyClient:
        ...     def __init__(self):
        ...         self._rate_limiters = RateLimiterRegistry()
        ...         self._rate_limiters.register_limiter("rest", TokenBucketRateLimiter(100, 100))
        ...
        ...     @rate_limited("rest", weight=10)
        ...     async def get_data(self):
        ...         return "data"

    Raises:
        AttributeError: If instance doesn't have _rate_limiters attribute
        KeyError: If limiter_key not found in registry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get rate limiter registry from instance
            if not hasattr(self, "_rate_limiters"):
                raise AttributeError(
                    f"{self.__class__.__name__} must have '_rate_limiters' attribute to use @rate_limited decorator"
                )

            registry: RateLimiterRegistry = self._rate_limiters

            # Get the specific limiter
            limiter = registry.get_limiter(limiter_key)

            # Acquire tokens before executing method
            await limiter.acquire(weight)

            # Execute the original method
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator
