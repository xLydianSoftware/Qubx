"""
Exchange-aware rate limiting engine.

Provides a shared, multi-pool rate limiting system that connectors use
to comply with exchange API limits. Supports per-IP and per-account scoping,
gate mechanism for cooldowns, and optional Redis backend for cross-bot coordination.

Usage:
    >>> from qubx.rate_limiting import ExchangeRateLimiter, ExchangeRateLimitConfig, PoolConfig, EndpointCosts
    >>>
    >>> config = ExchangeRateLimitConfig(
    ...     pools={"rest": PoolConfig("rest", "ip", 1200, 20.0)},
    ...     endpoint_map={"fetch_ohlcv": EndpointCosts([("rest", 5)])},
    ...     default_costs=EndpointCosts([("rest", 1)]),
    ... )
    >>> limiter = ExchangeRateLimiter("binance", config)
    >>> await limiter.acquire("fetch_ohlcv")
"""

from qubx.rate_limiting.backend import IRateLimitBackend, InMemoryBackend
from qubx.rate_limiting.config import EndpointCosts, ExchangeRateLimitConfig, PoolConfig
from qubx.rate_limiting.engine import ExchangeRateLimiter, RateLimitGateTimeout

__all__ = [
    "EndpointCosts",
    "ExchangeRateLimitConfig",
    "ExchangeRateLimiter",
    "IRateLimitBackend",
    "InMemoryBackend",
    "PoolConfig",
    "RateLimitGateTimeout",
]
