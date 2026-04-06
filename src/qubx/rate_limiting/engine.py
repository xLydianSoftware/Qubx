"""
Exchange rate limiting engine.

The central component that connectors interact with. Handles multi-pool
token acquisition, gate mechanism for cooldowns, exchange state sync,
and metric collection.
"""

import asyncio
import time
from typing import Any

from qubx import logger

from .backend import InMemoryBackend, IRateLimitBackend
from .config import ExchangeRateLimitConfig, PoolConfig


class RateLimitGateTimeout(Exception):
    """Raised when acquire() times out waiting for a gate to reopen."""


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
        self._backend = backend or InMemoryBackend()
        self._scope_ids = scope_ids or {}
        self._event_loop = event_loop

        # Per-pool gates (asyncio.Event: set = open, clear = closed)
        self._gates: dict[str, asyncio.Event] = {}
        self._gate_tasks: dict[str, asyncio.Task] = {}
        for pool_name in config.pools:
            gate = asyncio.Event()
            gate.set()
            self._gates[pool_name] = gate

        # Quota pools: externally-managed remaining count
        self._quota_remaining: dict[str, float] = {}
        for pool_name, pool in config.pools.items():
            if pool.pool_type == "quota":
                self._quota_remaining[pool_name] = pool.capacity

        # Stats tracking
        self._hits: dict[str, int] = {}
        self._total_wait: dict[str, float] = {}
        self._consumed: dict[str, float] = {}
        self._last_metrics_time = time.monotonic()

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

    def _key_for(self, pool: PoolConfig) -> str:
        """Construct Redis/backend key for a pool based on its scope."""
        scope_id = self._scope_ids.get(pool.scope, "local")
        return f"ratelimit:{self._exchange}:{pool.name}:{scope_id}"

    def update_scope_id(self, scope: str, new_id: str) -> None:
        """Update scope identifier (e.g., when egress IP changes).

        The old key's tokens will naturally expire. New requests use the new key.
        """
        old_id = self._scope_ids.get(scope)
        if old_id != new_id:
            logger.info(f"Rate limiter {self._exchange}: {scope} scope changed {old_id} → {new_id}")
            self._scope_ids[scope] = new_id

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
            pool = self._config.pools.get(pool_name)
            if pool is None:
                continue

            actual_weight = weight_override if (weight_override is not None and i == 0) else weight

            # Wait for gate to be open
            gate = self._gates.get(pool_name)
            if gate is not None and not gate.is_set():
                try:
                    await asyncio.wait_for(gate.wait(), timeout=self._config.gate_max_wait)
                except asyncio.TimeoutError:
                    raise RateLimitGateTimeout(
                        f"{self._exchange}: gate for pool '{pool_name}' did not reopen "
                        f"within {self._config.gate_max_wait:.0f}s"
                    ) from None

            # For quota pools, check remaining (no time-based refill)
            if pool.pool_type == "quota":
                remaining = self._quota_remaining.get(pool_name, 0)
                if remaining <= 0:
                    # Gate should already be closed, but double-check
                    self._close_gate(pool_name, pool.cooldown, "quota depleted")
                    gate = self._gates.get(pool_name)
                    if gate is not None:
                        try:
                            await asyncio.wait_for(gate.wait(), timeout=self._config.gate_max_wait)
                        except asyncio.TimeoutError:
                            raise RateLimitGateTimeout(
                                f"{self._exchange}: quota pool '{pool_name}' depleted, "
                                f"gate did not reopen within {self._config.gate_max_wait:.0f}s"
                            ) from None
                # Decrement quota locally
                if pool_name in self._quota_remaining:
                    self._quota_remaining[pool_name] = max(0, self._quota_remaining[pool_name] - actual_weight)
                continue

            # For rate pools, acquire from backend
            key = self._key_for(pool)
            wait_time = await self._backend.acquire(key, actual_weight, pool.capacity, pool.refill_rate)

            # Track stats
            self._total_wait[pool_name] = self._total_wait.get(pool_name, 0) + wait_time
            self._consumed[pool_name] = self._consumed.get(pool_name, 0) + actual_weight

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
        pools_to_close = []

        if pool_name:
            pools_to_close.append(pool_name)
        elif endpoint:
            costs = self._config.endpoint_map.get(endpoint, self._config.default_costs)
            pools_to_close = [p for p, _ in costs.costs]
        else:
            # Close all rate pools
            pools_to_close = [name for name, pool in self._config.pools.items() if pool.pool_type == "rate"]

        for pname in pools_to_close:
            pool = self._config.pools.get(pname)
            if pool is None:
                continue
            cooldown = retry_after if retry_after is not None else pool.cooldown
            self._close_gate(pname, cooldown, reason or f"rate limit hit on {pname}")
            self._hits[pname] = self._hits.get(pname, 0) + 1

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
        pool = self._config.pools.get(pool_name)
        if pool is None:
            return

        actual_capacity = capacity or pool.capacity

        if remaining is not None:
            actual_remaining = remaining
        elif used is not None:
            actual_remaining = actual_capacity - used
        else:
            return

        if pool.pool_type == "quota":
            self._quota_remaining[pool_name] = actual_remaining
            if actual_remaining <= 0:
                self._close_gate(pool_name, pool.cooldown, f"exchange reports {pool_name} depleted")
            return

        # For rate pools, update backend
        key = self._key_for(pool)
        # Fire-and-forget since set_remaining may be async but we're in sync context
        # Use a helper to run it
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._backend.set_remaining(key, actual_remaining))
        except RuntimeError:
            pass  # No running loop — skip sync (e.g., during shutdown)

    def sync_quota(self, pool_name: str, remaining: float) -> None:
        """Update externally-managed quota pool from exchange response.

        Convenience wrapper for quota pools (e.g., Lighter volume quota
        reported in sendTx responses).

        Args:
            pool_name: Quota pool name
            remaining: Remaining quota reported by exchange
        """
        self._quota_remaining[pool_name] = remaining
        if remaining <= 0:
            pool = self._config.pools.get(pool_name)
            cooldown = pool.cooldown if pool else 15.0
            self._close_gate(pool_name, cooldown, f"quota {pool_name} depleted (remaining={remaining})")

    def _close_gate(self, pool_name: str, cooldown: float, reason: str) -> None:
        """Close a pool's gate for the given cooldown period."""
        gate = self._gates.get(pool_name)
        if gate is None:
            return

        verb = "extended" if not gate.is_set() else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{pool_name} ({cooldown:.1f}s): {reason}")
        gate.clear()

        # Cancel existing reopen task
        old_task = self._gate_tasks.get(pool_name)
        if old_task is not None and not old_task.done():
            old_task.cancel()

        self._gate_tasks[pool_name] = asyncio.ensure_future(self._reopen_gate_after(pool_name, cooldown))

    async def _reopen_gate_after(self, pool_name: str, delay: float) -> None:
        """Reopen a pool's gate after a cooldown delay."""
        try:
            await asyncio.sleep(delay)
            gate = self._gates.get(pool_name)
            if gate is not None:
                gate.set()
                logger.info(f"Rate limit gate reopened for {self._exchange}:{pool_name} after {delay:.1f}s")
        except asyncio.CancelledError:
            pass

    def reset_gates(self) -> None:
        """Reopen all gates and cancel pending reopen tasks.

        Call on connection reset / reconnection.
        """
        for pool_name, gate in self._gates.items():
            gate.set()
        for pool_name, task in self._gate_tasks.items():
            if not task.done():
                task.cancel()
        self._gate_tasks.clear()

    def is_gate_closed(self, pool_name: str | None = None) -> bool:
        """Check if a gate is closed.

        Args:
            pool_name: Specific pool, or None to check if ANY gate is closed
        """
        if pool_name:
            gate = self._gates.get(pool_name)
            return gate is not None and not gate.is_set()
        return any(not gate.is_set() for gate in self._gates.values())

    async def get_pool_state(self, pool_name: str) -> dict[str, Any] | None:
        """Get current state of a pool for monitoring.

        Returns:
            Dict with remaining, capacity, gate_closed, hits, etc. or None if pool doesn't exist
        """
        pool = self._config.pools.get(pool_name)
        if pool is None:
            return None

        if pool.pool_type == "quota":
            remaining = self._quota_remaining.get(pool_name, 0)
        else:
            key = self._key_for(pool)
            remaining = await self._backend.get_remaining(key, pool.capacity, pool.refill_rate)
            if remaining is None:
                remaining = pool.capacity

        return {
            "pool": pool_name,
            "exchange": self._exchange,
            "scope": pool.scope,
            "scope_id": self._scope_ids.get(pool.scope, "local"),
            "pool_type": pool.pool_type,
            "remaining": remaining,
            "capacity": pool.capacity,
            "utilization": 1.0 - (remaining / pool.capacity) if pool.capacity > 0 else 0,
            "gate_closed": self.is_gate_closed(pool_name),
            "hits": self._hits.get(pool_name, 0),
            "total_wait_s": self._total_wait.get(pool_name, 0),
            "consumed": self._consumed.get(pool_name, 0),
        }

    async def collect_metrics(self) -> list[dict[str, Any]]:
        """Collect current metrics for all pools.

        Returns a list of metric dicts suitable for emitting via ctx.emitter.emit().

        Call this periodically (e.g., once per minute) and emit via:
            for m in await rate_limiter.collect_metrics():
                ctx.emitter.emit(m["name"], m["value"], m["tags"])
        """
        metrics = []
        for pool_name in self._config.pools:
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
        pools = ", ".join(f"{name}({pool.scope})" for name, pool in self._config.pools.items())
        return f"ExchangeRateLimiter({self._exchange}, pools=[{pools}])"
