"""Exchange rate limiting engine.

The central component that connectors interact with. Delegates pool-specific
behavior (gate management, acquisition, sync) to polymorphic pool objects.
"""

import asyncio
import time
from typing import Any

from qubx import logger

from .backend import InMemoryBackend, IRateLimitBackend
from .config import ExchangeRateLimitConfig
from .pools import BasePool, QuotaPool, RatePool
from .pools import RateLimitGateTimeout as RateLimitGateTimeout  # re-export


class ExchangeRateLimiter:
    """Multi-pool rate limiting engine for exchange API calls.

    Connectors create one instance per exchange, configured with pools and
    endpoint mappings. The engine handles:
    - Multi-key acquisition (wait until ALL pools have budget)
    - Gate mechanism (pause requests on rate limit hits)
    - Exchange state sync (calibrate model from response headers/WS events)
    - Quota pools (externally-managed, no time-based refill)
    - Metric collection for dashboarding

    Usage:
        >>> config = ExchangeRateLimitConfig(
        ...     pools={"rest": PoolConfig("rest", "ip", 24000, 400.0)},
        ...     endpoint_map={"fetch_candles": EndpointCosts([("rest", 50)])},
        ...     default_costs=EndpointCosts([("rest", 300)]),
        ... )
        >>> limiter = ExchangeRateLimiter("lighter", config)
        >>> await limiter.acquire("fetch_candles")
    """

    def __init__(
        self,
        exchange: str,
        config: ExchangeRateLimitConfig,
        backend: IRateLimitBackend | None = None,
        scope_ids: dict[str, str] | None = None,
        event_loop: "asyncio.AbstractEventLoop | None" = None,
    ):
        """
        Args:
            exchange: Exchange name (e.g., "binance", "lighter")
            config: Rate limit configuration with pools and endpoint mappings
            backend: Storage backend (default: InMemoryBackend)
            scope_ids: Maps scope type to identifier, e.g. {"ip": "1.2.3.4", "account": "abc123"}
            event_loop: Event loop this rate limiter operates on (for metrics collection)
        """
        self._exchange = exchange
        self._config = config
        self._scope_ids = scope_ids or {}
        self._event_loop = event_loop
        self._last_metrics_time = time.monotonic()

        backend = backend or InMemoryBackend()
        self._pools: dict[str, BasePool] = {}
        for pool_name, pool_config in config.pools.items():
            scope_id = self._scope_ids.get(pool_config.scope, "local")
            if pool_config.pool_type == "quota":
                self._pools[pool_name] = QuotaPool(pool_config, exchange, scope_id)
            else:
                self._pools[pool_name] = RatePool(pool_config, exchange, scope_id, backend)

    @property
    def exchange(self) -> str:
        return self._exchange

    @property
    def event_loop(self):
        """Event loop this rate limiter operates on (set by connector)."""
        return getattr(self, "_event_loop", None)

    @event_loop.setter
    def event_loop(self, loop):
        self._event_loop = loop

    @property
    def config(self) -> ExchangeRateLimitConfig:
        return self._config

    def update_scope_id(self, scope: str, new_id: str) -> None:
        """Update scope identifier (e.g., when egress IP changes).

        The old key's tokens will naturally expire. New requests use the new key.
        """
        old_id = self._scope_ids.get(scope)
        if old_id != new_id:
            logger.info(f"Rate limiter {self._exchange}: {scope} scope changed {old_id} → {new_id}")
            self._scope_ids[scope] = new_id
            for pool in self._pools.values():
                if pool.config.scope == scope:
                    pool.update_scope_id(new_id)

    async def acquire(self, endpoint: str, weight_override: float | None = None) -> None:
        """Wait until all pools for this endpoint have sufficient budget.

        This is the main entry point for connectors. Call before making any
        exchange API request.

        Args:
            endpoint: Endpoint name matching endpoint_map keys
            weight_override: Override the weight for the first (primary) pool.
                Useful when CCXT computes dynamic costs (e.g., Binance depth limit tiers).

        Raises:
            RateLimitGateTimeout: If a gate doesn't reopen within gate_max_wait
        """
        endpoint_costs = self._config.endpoint_map.get(endpoint, self._config.default_costs)
        if not endpoint_costs.costs:
            return

        for i, (pool_name, weight) in enumerate(endpoint_costs.costs):
            pool = self._pools.get(pool_name)
            if pool is None:
                continue
            actual_weight = weight_override if (weight_override is not None and i == 0) else weight
            await pool.acquire(actual_weight, self._config.gate_max_wait)

    def report_limit_hit(
        self,
        pool_name: str | None = None,
        endpoint: str | None = None,
        retry_after: float | None = None,
        reason: str = "",
    ) -> None:
        """Report a rate limit hit (429, WS error, etc.).

        Closes the gate for affected pools and applies cooldown.
        Call this when the exchange tells you you've been rate limited.

        Args:
            pool_name: Specific pool that was hit (if known)
            endpoint: Endpoint that was called (to look up affected pools)
            retry_after: Seconds to wait (from Retry-After header)
            reason: Human-readable reason for logging
        """
        pools_to_close: list[str] = []

        if pool_name:
            pools_to_close.append(pool_name)
        elif endpoint:
            costs = self._config.endpoint_map.get(endpoint, self._config.default_costs)
            pools_to_close = [p for p, _ in costs.costs]
        else:
            # Close all rate pools
            pools_to_close = [
                name for name, pool in self._pools.items() if pool.config.pool_type == "rate"
            ]

        for pname in pools_to_close:
            pool = self._pools.get(pname)
            if pool is None:
                continue
            cooldown = retry_after if retry_after is not None else pool.config.cooldown
            pool.close_gate(cooldown, reason or f"rate limit hit on {pname}")
            pool.hits += 1

    def sync_from_exchange(
        self,
        pool_name: str,
        remaining: float | None = None,
        used: float | None = None,
        capacity: float | None = None,
    ) -> None:
        """Sync modeled state with exchange-reported state.

        Called when response headers or WS messages report actual usage.
        Corrects drift between our model and reality (e.g., other bots
        on the same account consuming budget).

        Args:
            pool_name: Pool to sync
            remaining: Remaining tokens reported by exchange
            used: Used tokens reported by exchange (remaining = capacity - used)
            capacity: Total capacity reported by exchange (overrides config if provided)
        """
        pool = self._pools.get(pool_name)
        if pool is None:
            return

        actual_capacity = capacity or pool.config.capacity

        if remaining is not None:
            actual_remaining = remaining
        elif used is not None:
            actual_remaining = actual_capacity - used
        else:
            return

        pool.sync(actual_remaining, capacity)

    def get_quota_remaining(self, pool_name: str) -> float:
        """Get remaining budget for a quota pool.

        Returns 0 if pool doesn't exist or isn't a quota pool.
        """
        pool = self._pools.get(pool_name)
        if isinstance(pool, QuotaPool):
            return pool.remaining
        return 0

    def reset_gates(self) -> None:
        """Reopen all gates and cancel pending reopen tasks.

        Call on connection reset / reconnection.
        """
        for pool in self._pools.values():
            pool.reset_gate()

    def is_gate_closed(self, pool_name: str | None = None) -> bool:
        """Check if a gate is closed.

        Args:
            pool_name: Specific pool, or None to check if ANY gate is closed
        """
        if pool_name:
            pool = self._pools.get(pool_name)
            return pool is not None and pool.is_gate_closed
        return any(pool.is_gate_closed for pool in self._pools.values())

    async def get_pool_state(self, pool_name: str) -> dict[str, Any] | None:
        """Get current state of a pool for monitoring.

        Returns:
            Dict with remaining, capacity, gate_closed, hits, etc. or None if pool doesn't exist
        """
        pool = self._pools.get(pool_name)
        if pool is None:
            return None
        return await pool.get_state()

    async def collect_metrics(self) -> list[dict[str, Any]]:
        """Collect current metrics for all pools.

        Returns a list of metric dicts suitable for emitting via ctx.emitter.emit().

        Call this periodically (e.g., once per minute) and emit via:
            for m in await rate_limiter.collect_metrics():
                ctx.emitter.emit(m["name"], m["value"], m["tags"])
        """
        metrics = []
        for pool_name in self._pools:
            state = await self.get_pool_state(pool_name)
            if state is None:
                continue

            tags = {
                "exchange": self._exchange,
                "pool": pool_name,
                "scope": state["scope"],
                "type": "rate_limit",
            }

            metrics.append({"name": "rate_limit.remaining", "value": state["remaining"], "tags": {**tags}})
            metrics.append({"name": "rate_limit.capacity", "value": state["capacity"], "tags": {**tags}})
            metrics.append({"name": "rate_limit.utilization", "value": state["utilization"], "tags": {**tags}})
            metrics.append(
                {"name": "rate_limit.gate_closed", "value": 1.0 if state["gate_closed"] else 0.0, "tags": {**tags}}
            )
            metrics.append({"name": "rate_limit.hits", "value": float(state["hits"]), "tags": {**tags}})
            metrics.append({"name": "rate_limit.wait_seconds", "value": state["total_wait_s"], "tags": {**tags}})
            metrics.append({"name": "rate_limit.consumed", "value": state["consumed"], "tags": {**tags}})

        return metrics

    def __repr__(self) -> str:
        pools = ", ".join(f"{name}({pool.config.scope})" for name, pool in self._pools.items())
        return f"ExchangeRateLimiter({self._exchange}, pools=[{pools}])"
