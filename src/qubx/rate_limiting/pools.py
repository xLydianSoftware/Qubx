"""Pool implementations for the rate limiting engine.

Each pool type encapsulates its own gate behavior, acquisition logic,
sync mechanism, and metrics state. The engine delegates to pools
polymorphically — no pool_type branching in the engine.

Loop-agnostic (quantkit#106): one ExchangeRateLimiter spans a venue's realtime
websocket loop AND the process-wide BulkRestLoop, so pool state must be usable from
coroutines on multiple event loops. Gate truth lives in ``_MultiLoopGate`` (plain
bool + per-loop mirror events), gate reopen runs on a ``threading.Timer`` (no
loop-bound task), and mutable pool state is guarded by a ``threading.Lock``.
"""

import asyncio
import threading
import time
from typing import Any
from weakref import WeakKeyDictionary

from qubx import logger

from .backend import IRateLimitBackend
from .config import PoolConfig


class RateLimitGateTimeout(Exception):
    """Raised when acquire() times out waiting for a gate to reopen."""

    def __init__(self, message: str, pool_name: str | None = None):
        super().__init__(message)
        self.pool_name = pool_name


class _MultiLoopGate:
    """Open/closed gate awaitable from coroutines on multiple event loops.

    ``asyncio.Event`` is loop-affine: ``wait()`` parks a future on the waiting loop and
    ``set()`` completes it without cross-thread scheduling, so a single Event cannot
    serve both the realtime and the bulk REST loop. Here the authoritative state is a
    plain bool behind a ``threading.Lock``; every loop that waits gets its own mirror
    ``asyncio.Event``, flipped via ``loop.call_soon_threadsafe`` so wakeups always
    happen on the waiter's own loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open = True
        self._events: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Event] = WeakKeyDictionary()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def open(self) -> None:
        self._set_state(True)

    def close(self) -> None:
        self._set_state(False)

    def _set_state(self, is_open: bool) -> None:
        with self._lock:
            self._open = is_open
            for loop, event in list(self._events.items()):
                try:
                    loop.call_soon_threadsafe(event.set if is_open else event.clear)
                except RuntimeError:
                    continue  # loop already closed — its waiters are gone anyway

    def _event_for_running_loop(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        with self._lock:
            event = self._events.get(loop)
            if event is None:
                event = asyncio.Event()
                if self._open:
                    event.set()
                self._events[loop] = event
            return event

    async def wait_open(self, timeout: float) -> None:
        """Wait until the gate is open; raises ``TimeoutError`` on expiry.

        The bool is the authority — the mirror event only provides the sleep. The loop
        re-checks after every wakeup so a gate extension (close while waiting) keeps the
        waiter parked, and a mirror that briefly lags its scheduled clear/set callback
        (queued on this very loop) only costs an extra yield, never a lost wakeup.
        """
        deadline = time.monotonic() + timeout
        while True:
            event = self._event_for_running_loop()
            if self.is_open:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(event.wait(), timeout=remaining)


class BasePool:
    """Base pool with gate mechanism and metrics tracking.

    Gate reopen scheduling is epoch-guarded: every transition (close, extension,
    reset, sync-reopen) bumps ``_gate_epoch`` under ``_state_lock``, and a pending
    reopen timer only fires if its captured epoch is still current — a stale timer
    from a superseded close can never reopen a freshly-closed gate.

    Metrics counters (``hits``/``total_wait``/``consumed``) are updated without the
    lock; cross-thread drift is cosmetic and not worth serializing every acquire.
    """

    def __init__(self, config: PoolConfig, exchange: str, scope_id: str):
        self._config = config
        self._exchange = exchange
        self._scope_id = scope_id
        self._gate = _MultiLoopGate()
        self._gate_timer: threading.Timer | None = None
        self._gate_epoch = 0
        self._state_lock = threading.Lock()
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
        return not self._gate.is_open

    def update_scope_id(self, scope_id: str) -> None:
        self._scope_id = scope_id

    async def acquire(self, weight: float, gate_max_wait: float) -> None:
        raise NotImplementedError

    def close_gate(self, cooldown: float, reason: str) -> None:
        raise NotImplementedError

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        raise NotImplementedError

    def reset_gate(self) -> None:
        """Reopen gate and invalidate any pending reopen timer."""
        self._open_gate_now()

    async def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _open_gate_now(self) -> None:
        with self._state_lock:
            self._gate_epoch += 1
            self._cancel_gate_timer()
            self._gate.open()

    def _close_gate_now(self) -> None:
        with self._state_lock:
            self._gate_epoch += 1
            self._cancel_gate_timer()
            self._gate.close()

    def _cancel_gate_timer(self) -> None:
        """Cancel a pending reopen timer. Caller holds ``_state_lock``."""
        if self._gate_timer is not None:
            self._gate_timer.cancel()
        self._gate_timer = None

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
        if not self._gate.is_open:
            try:
                await self._gate.wait_open(gate_max_wait)
            except TimeoutError:
                raise RateLimitGateTimeout(
                    f"{self._exchange}: gate for pool '{self.name}' did not reopen "
                    f"within {gate_max_wait:.0f}s",
                    pool_name=self.name,
                ) from None

        wait_time = await self._backend.acquire(self._key, weight, self._config.capacity, self._config.refill_rate)
        self.total_wait += wait_time
        self.consumed += weight

    def close_gate(self, cooldown: float, reason: str) -> None:
        verb = "extended" if not self._gate.is_open else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{self.name} ({cooldown:.1f}s): {reason}")
        with self._state_lock:
            self._gate_epoch += 1
            epoch = self._gate_epoch
            self._cancel_gate_timer()
            self._gate.close()
            # threading.Timer, not an asyncio task: reopen must not be bound to any of
            # the loops acquiring from this pool (the closing loop may not be the only
            # waiter, and a timer is cancellable from any thread).
            timer = threading.Timer(cooldown, self._reopen_after, args=(epoch, cooldown))
            timer.daemon = True
            self._gate_timer = timer
        timer.start()

    def _reopen_after(self, epoch: int, delay: float) -> None:
        with self._state_lock:
            if epoch != self._gate_epoch:
                return  # superseded by a later close/extension/reset
            self._gate_timer = None
            self._gate.open()
        logger.info(f"Rate limit gate reopened for {self._exchange}:{self.name} after {delay:.1f}s")

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
        with self._state_lock:
            depleted = self._remaining <= 0
            if not depleted and self._gate.is_open:
                # atomic check-and-decrement — no over-issue across loops
                self._remaining = max(0, self._remaining - weight)
                self.consumed += weight
                return
        if depleted:
            self.close_gate(self._config.cooldown, "quota depleted")
        raise RateLimitGateTimeout(
            f"{self._exchange}: quota pool '{self.name}' depleted",
            pool_name=self.name,
        )

    def close_gate(self, cooldown: float, reason: str) -> None:
        verb = "extended" if not self._gate.is_open else "closed"
        logger.warning(f"Rate limit gate {verb} for {self._exchange}:{self.name} ({cooldown:.1f}s): {reason}")
        self._close_gate_now()

    def sync(self, remaining: float, capacity: float | None = None) -> None:
        with self._state_lock:
            self._remaining = remaining
        if remaining <= 0:
            self.close_gate(self._config.cooldown, f"quota {self.name} depleted (remaining={remaining})")
        else:
            # When no explicit capacity provided, grow capacity to track the
            # real account quota (best approximation when exchange only reports remaining)
            if capacity is None:
                self._config.capacity = max(self._config.capacity, remaining)
            if not self._gate.is_open:
                self._open_gate_now()
                logger.info(
                    f"Rate limit gate reopened for {self._exchange}:{self.name} (remaining={remaining})"
                )

    async def get_state(self) -> dict[str, Any]:
        return self._base_state(self._remaining, self._config.capacity)
