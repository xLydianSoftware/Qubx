"""Pool implementations for the rate limiting engine.

Each pool type encapsulates its own gate behavior, acquisition logic,
sync mechanism, and metrics state. The engine delegates to pools
polymorphically — no pool_type branching in the engine.
"""

import asyncio
import math
import time
from typing import Any

from qubx import logger

from .backend import IRateLimitBackend
from .config import PoolConfig

_GATE_POLL_S = 0.25  # bounds only how late an early reopen (reset_gate / quota sync) is noticed


class RateLimitGateTimeout(Exception):
    """Raised when acquire() times out waiting for a gate to reopen."""

    def __init__(self, message: str, pool_name: str | None = None):
        super().__init__(message)
        self.pool_name = pool_name


class BasePool:
    """Base pool with gate mechanism and metrics tracking.

    The gate is a monotonic deadline rather than an event, so it is observable and settable from
    any thread or event loop without binding to one.
    """

    def __init__(self, config: PoolConfig, exchange: str, scope_id: str):
        self._config = config
        self._exchange = exchange
        self._scope_id = scope_id
        self._gate_until: float = 0.0
        self.hits: int = 0
        self.total_wait: float = 0
        self.consumed: float = 0
        self.timeouts: int = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> PoolConfig:
        return self._config

    @property
    def scope_id(self) -> str:
        return self._scope_id

    @property
    def is_gate_closed(self) -> bool:
        return time.monotonic() < self._gate_until

    def update_scope_id(self, scope_id: str) -> None:
        self._scope_id = scope_id

    async def acquire(self, weight: float, gate_max_wait: float) -> None:
        raise NotImplementedError

    def close_gate(self, cooldown: float, reason: str) -> None:
        verb = "extended" if self.is_gate_closed else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{self.name} ({cooldown:.1f}s): {reason}")
        self._gate_until = max(self._gate_until, time.monotonic() + cooldown)

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        raise NotImplementedError

    def reset_gate(self) -> None:
        self._gate_until = 0.0

    async def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    async def _wait_for_gate(self, gate_max_wait: float) -> None:
        give_up_at = time.monotonic() + gate_max_wait
        waited = False
        while (now := time.monotonic()) < self._gate_until:
            if now >= give_up_at:
                self.timeouts += 1
                raise RateLimitGateTimeout(
                    f"{self._exchange}: gate for pool '{self.name}' did not reopen within {gate_max_wait:.0f}s",
                    pool_name=self.name,
                )
            waited = True
            await asyncio.sleep(min(self._gate_until - now, give_up_at - now, _GATE_POLL_S))
        if waited:
            logger.info(f"Rate limit gate reopened for {self._exchange}:{self.name}")

    def _base_state(self, remaining: float, capacity: float) -> dict[str, Any]:
        return {
            "pool": self.name,
            "exchange": self._exchange,
            "scope": self._config.scope,
            "scope_id": self._scope_id,
            "pool_type": self._config.pool_type,
            "remaining": remaining,
            "capacity": capacity,
            "utilization": 1.0 - (remaining / capacity) if capacity > 0 else 0,
            "gate_closed": self.is_gate_closed,
            "hits": self.hits,
            "total_wait_s": self.total_wait,
            "consumed": self.consumed,
            "timeouts": self.timeouts,
        }


class RatePool(BasePool):
    """Time-based token bucket pool. Gate reopens once the cooldown deadline passes."""

    def __init__(self, config: PoolConfig, exchange: str, scope_id: str, backend: IRateLimitBackend):
        super().__init__(config, exchange, scope_id)
        self._backend = backend
        self._key = self._make_key()
        self._sync_tasks: set[asyncio.Task] = set()

    def _make_key(self) -> str:
        return f"ratelimit:{self._exchange}:{self._config.name}:{self._scope_id}"

    def update_scope_id(self, scope_id: str) -> None:
        super().update_scope_id(scope_id)
        self._key = self._make_key()

    async def acquire(self, weight: float, gate_max_wait: float) -> None:
        if weight > self._config.capacity:
            # a weight no bucket can ever hold never completes: the Redis backend retries forever
            logger.warning(
                f"Rate limit {self._exchange}:{self.name}: weight {weight} exceeds capacity "
                f"{self._config.capacity}, clamping"
            )
            weight = self._config.capacity

        await self._wait_for_gate(gate_max_wait)

        wait_time = await self._backend.acquire(self._key, weight, self._config.capacity, self._config.refill_rate)
        self.total_wait += wait_time
        self.consumed += weight

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop — skip sync (e.g. during shutdown)
        task = loop.create_task(
            self._backend.set_remaining(self._key, remaining, self._config.capacity, self._config.refill_rate)
        )
        # keep a reference so the task cannot be collected mid-flight, and retrieve its result:
        # a header sync fires per response, so an unretrieved Redis error would log on every one
        self._sync_tasks.add(task)
        task.add_done_callback(self._on_sync_done)

    def _on_sync_done(self, task: asyncio.Task) -> None:
        self._sync_tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.debug(f"Rate limit sync failed for {self._exchange}:{self.name}: {exc}")

    async def get_state(self) -> dict[str, Any]:
        remaining = await self._backend.get_remaining(self._key, self._config.capacity, self._config.refill_rate)
        if remaining is None:
            remaining = self._config.capacity
        return self._base_state(remaining, self._config.capacity)


class QuotaPool(BasePool):
    """Externally-managed quota pool. Gate reopens only via sync() or reset_gate()."""

    def __init__(self, config: PoolConfig, exchange: str, scope_id: str):
        super().__init__(config, exchange, scope_id)
        self._remaining: float = config.capacity

    @property
    def remaining(self) -> float:
        return self._remaining

    async def acquire(self, weight: float, gate_max_wait: float) -> None:
        if self.is_gate_closed or self._remaining <= 0:
            if self._remaining <= 0:
                self.close_gate(self._config.cooldown, "quota depleted")
            self.timeouts += 1
            raise RateLimitGateTimeout(
                f"{self._exchange}: quota pool '{self.name}' depleted",
                pool_name=self.name,
            )
        self._remaining = max(0, self._remaining - weight)
        self.consumed += weight

    def close_gate(self, cooldown: float, reason: str) -> None:
        verb = "extended" if self.is_gate_closed else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{self.name}: {reason}")
        self._gate_until = math.inf  # no timer can reopen a quota gate, only the exchange can

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        self._remaining = remaining
        if remaining <= 0:
            self.close_gate(self._config.cooldown, f"quota {self.name} depleted (remaining={remaining})")
        else:
            # When no explicit capacity provided, grow capacity to track the
            # real account quota (best approximation when exchange only reports remaining)
            if capacity is None:
                self._config.capacity = max(self._config.capacity, remaining)
            if self.is_gate_closed:
                self._gate_until = 0.0
                logger.info(f"Rate limit gate reopened for {self._exchange}:{self.name} (remaining={remaining})")

    async def get_state(self) -> dict[str, Any]:
        return self._base_state(self._remaining, self._config.capacity)
