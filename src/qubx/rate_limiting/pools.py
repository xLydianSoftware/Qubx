"""Pool implementations for the rate limiting engine.

Each pool type encapsulates its own gate behavior, acquisition logic,
sync mechanism, and metrics state. The engine delegates to pools
polymorphically — no pool_type branching in the engine.
"""

import asyncio
from typing import Any

from qubx import logger

from .backend import IRateLimitBackend
from .config import PoolConfig


class RateLimitGateTimeout(Exception):
    """Raised when acquire() times out waiting for a gate to reopen."""

    def __init__(self, message: str, pool_name: str | None = None):
        super().__init__(message)
        self.pool_name = pool_name


class BasePool:
    """Base pool with gate mechanism and metrics tracking."""

    def __init__(self, config: PoolConfig, exchange: str, scope_id: str):
        self._config = config
        self._exchange = exchange
        self._scope_id = scope_id
        self._gate = asyncio.Event()
        self._gate.set()
        self._gate_task: asyncio.Task | None = None
        self.hits: int = 0
        self.total_wait: float = 0
        self.consumed: float = 0

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
        return not self._gate.is_set()

    def update_scope_id(self, scope_id: str) -> None:
        self._scope_id = scope_id

    async def acquire(self, weight: float, gate_max_wait: float) -> None:
        raise NotImplementedError

    def close_gate(self, cooldown: float, reason: str) -> None:
        raise NotImplementedError

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        raise NotImplementedError

    def reset_gate(self) -> None:
        """Reopen gate and cancel any pending reopen task."""
        self._gate.set()
        self._cancel_gate_task()

    async def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _cancel_gate_task(self) -> None:
        if self._gate_task is not None and not self._gate_task.done():
            self._gate_task.cancel()
        self._gate_task = None

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
        }


class RatePool(BasePool):
    """Time-based token bucket pool. Gate reopens on timer after cooldown."""

    def __init__(self, config: PoolConfig, exchange: str, scope_id: str, backend: IRateLimitBackend):
        super().__init__(config, exchange, scope_id)
        self._backend = backend
        self._key = self._make_key()

    def _make_key(self) -> str:
        return f"ratelimit:{self._exchange}:{self._config.name}:{self._scope_id}"

    def update_scope_id(self, scope_id: str) -> None:
        super().update_scope_id(scope_id)
        self._key = self._make_key()

    async def acquire(self, weight: float, gate_max_wait: float) -> None:
        if not self._gate.is_set():
            try:
                await asyncio.wait_for(self._gate.wait(), timeout=gate_max_wait)
            except asyncio.TimeoutError:
                raise RateLimitGateTimeout(
                    f"{self._exchange}: gate for pool '{self.name}' did not reopen "
                    f"within {gate_max_wait:.0f}s",
                    pool_name=self.name,
                ) from None

        wait_time = await self._backend.acquire(self._key, weight, self._config.capacity, self._config.refill_rate)
        self.total_wait += wait_time
        self.consumed += weight

    def close_gate(self, cooldown: float, reason: str) -> None:
        verb = "extended" if not self._gate.is_set() else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{self.name} ({cooldown:.1f}s): {reason}")
        self._gate.clear()
        self._cancel_gate_task()
        self._gate_task = asyncio.ensure_future(self._reopen_after(cooldown))

    async def _reopen_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            self._gate.set()
            logger.info(f"Rate limit gate reopened for {self._exchange}:{self.name} after {delay:.1f}s")
        except asyncio.CancelledError:
            pass

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._backend.set_remaining(self._key, remaining))
        except RuntimeError:
            pass  # No running loop — skip sync (e.g., during shutdown)

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
        if not self._gate.is_set() or self._remaining <= 0:
            if self._remaining <= 0:
                self.close_gate(self._config.cooldown, "quota depleted")
            raise RateLimitGateTimeout(
                f"{self._exchange}: quota pool '{self.name}' depleted",
                pool_name=self.name,
            )
        self._remaining = max(0, self._remaining - weight)
        self.consumed += weight

    def close_gate(self, cooldown: float, reason: str) -> None:
        verb = "extended" if not self._gate.is_set() else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{self.name} ({cooldown:.1f}s): {reason}")
        self._gate.clear()
        self._cancel_gate_task()

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        self._remaining = remaining
        if remaining <= 0:
            self.close_gate(self._config.cooldown, f"quota {self.name} depleted (remaining={remaining})")
        else:
            # When no explicit capacity provided, grow capacity to track the
            # real account quota (best approximation when exchange only reports remaining)
            if capacity is None:
                self._config.capacity = max(self._config.capacity, remaining)
            if not self._gate.is_set():
                self._cancel_gate_task()
                self._gate.set()
                logger.info(
                    f"Rate limit gate reopened for {self._exchange}:{self.name} (remaining={remaining})"
                )

    async def get_state(self) -> dict[str, Any]:
        return self._base_state(self._remaining, self._config.capacity)
