import time

from redis.exceptions import RedisError

from qubx import logger

from .backend import InMemoryBackend, IRateLimitBackend

_BACKEND_ERRORS = (RedisError, OSError)


class ResilientRateLimitBackend(IRateLimitBackend):
    """Composite: primary (redis) with local fallback and a circuit breaker.

    Primary healthy → pass-through. Primary error → breaker opens for
    breaker_cooldown_s and calls are served by the local fallback (per-bot
    pacing, no cross-bot coordination). Cooldown expiry → exactly one caller
    probes the primary; concurrent callers stay on the fallback until the
    probe resolves. State is plain monotonic floats shared across event
    loops — races are benign (worst case: two probes).
    """

    def __init__(
        self,
        primary: IRateLimitBackend,
        fallback: IRateLimitBackend | None = None,
        breaker_cooldown_s: float = 30.0,
    ):
        self._primary = primary
        self._fallback = fallback or InMemoryBackend()
        self._breaker_cooldown_s = breaker_cooldown_s
        self._broken = False
        self._broken_until = 0.0
        self._probe_inflight = False

    def _use_primary(self) -> bool:
        """Route decision for this call; claims the probe slot when one is due."""
        if not self._broken:
            return True
        if self._probe_inflight or time.monotonic() < self._broken_until:
            return False
        self._probe_inflight = True
        return True

    def _on_success(self) -> None:
        self._probe_inflight = False
        if self._broken:
            self._broken = False
            logger.info("Rate-limit redis backend recovered — resuming cross-bot coordination")

    def _on_error(self, exc: Exception) -> None:
        self._probe_inflight = False
        if not self._broken:
            logger.warning(
                f"Rate-limit redis backend unavailable ({exc!r}) — "
                f"falling back to local buckets for {self._breaker_cooldown_s:.0f}s"
            )
        self._broken = True
        self._broken_until = time.monotonic() + self._breaker_cooldown_s

    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        if self._use_primary():
            try:
                waited = await self._primary.acquire(key, weight, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e)
            else:
                self._on_success()
                return waited
        return await self._fallback.acquire(key, weight, capacity, refill_rate)

    async def get_remaining(self, key: str, capacity: float = 0, refill_rate: float = 0) -> float | None:
        if self._use_primary():
            try:
                remaining = await self._primary.get_remaining(key, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e)
            else:
                self._on_success()
                return remaining
        return await self._fallback.get_remaining(key, capacity, refill_rate)

    async def set_remaining(self, key: str, remaining: float, capacity: float = 0, refill_rate: float = 0) -> None:
        if self._use_primary():
            try:
                await self._primary.set_remaining(key, remaining, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e)
            else:
                self._on_success()
                return
        await self._fallback.set_remaining(key, remaining, capacity, refill_rate)

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()
