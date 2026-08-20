"""Resilient rate-limit backend: redis primary + local fallback + circuit breaker.

``ResilientRateLimitBackend`` composes a primary backend (normally redis, for
cross-bot coordination) with a local fallback (``InMemoryBackend``, per-bot
pacing only) behind a circuit breaker, so trading never blocks on rate-limit
redis. See design doc "Rate-limit backend resilience (increment 2)" (RL-D1
through RL-D4) for the accepted degradation and recovery contract.

Breaker protocol:
    - Healthy (``_broken`` is False): every call goes to the primary.
    - Primary error opens the breaker: ``_broken = True``,
      ``_broken_until = now + breaker_cooldown_s``; every call is served by
      the fallback until the cooldown expires — no per-call timeout tax.
    - Once the cooldown expires, exactly one caller claims the probe slot
      (``_route()``, ``_probe_inflight``) and is routed to the primary; other
      concurrent callers stay on the fallback while the probe is in flight.
    - Open and close are asymmetric. Closing requires the probe: only the
      probe's success closes the breaker, so a *stale* success — from a call
      dispatched before the breaker opened, landing after — is ignored (see
      F2 / review Finding 2). Opening does not: ANY primary error opens or
      re-arms the cooldown, stale or not, because an error is evidence of an
      unhealthy primary regardless of when the call was dispatched. A stale
      error landing right after a successful probe therefore re-opens the
      breaker for a full cooldown — accepted: it costs one 30 s local-only
      window at the tail of an outage, never a hang.
    - The probe slot is released on every exit path, including
      ``asyncio.CancelledError`` and any exception outside
      ``_BACKEND_ERRORS`` (see F1 / review Finding 1): those propagate to the
      caller with the breaker state left as-is, but the slot is freed so a
      later call can still probe and recover.

Accepted limitations (parked by design, not bugs):
    - A probe against a *healthy but saturated* redis can hold the slot for
      the bucket's full deficit, not just the socket-timeout bound, because
      ``RedisBackend.acquire`` loops internally until it has budget. During
      that window every other call in the bot runs uncoordinated on a fresh
      local bucket (review Finding 3).
    - If the primary fails after already sleeping through part of a
      multi-iteration acquire, the accumulated wait is discarded and the
      fallback call starts its wait from zero — the caller's reported
      ``waited`` undercounts the real wall-clock wait during that one call.
      This is conservative (it can only under-report pressure, never make
      the caller wait less than it actually did) and never affects
      correctness (review Finding 4).
    - The ``redis`` import below is module-level, not lazy like
      ``redis_backend.py``'s in-function imports — acceptable because
      ``redis`` is already a hard ``qubx`` dependency (review Finding 7).

Races: two callers can both observe the cooldown as expired and both start a
probe before either sets ``_probe_inflight`` — this is a plain read-modify
write on process-local bytecode-level attributes, not a lock. That race hits
the primary twice instead of once; it is benign and accepted.
"""

import time

from redis.exceptions import RedisError

from qubx import logger

from .backend import InMemoryBackend, IRateLimitBackend

_BACKEND_ERRORS = (RedisError, OSError)


class ResilientRateLimitBackend(IRateLimitBackend):
    """Composite: primary (redis) with local fallback and a circuit breaker.

    See the module docstring for the full breaker protocol and the accepted
    limitations. In short: primary healthy → pass-through; primary error →
    breaker opens for ``breaker_cooldown_s`` and calls are served by the
    local fallback; cooldown expiry → exactly one caller probes the primary
    while concurrent callers stay on the fallback; only the probe's success
    closes the breaker, while any primary error (probe or stale) opens or
    re-arms it. Breaker state (``_broken``, ``_broken_until``,
    ``_probe_inflight``) is plain attributes shared across event loops — the
    one accepted race is two concurrent probes.
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

    def _route(self) -> tuple[bool, bool]:
        """(use_primary, is_probe) for this call; claims the probe slot when one is due."""
        if not self._broken:
            return True, False
        if self._probe_inflight or time.monotonic() < self._broken_until:
            return False, False
        self._probe_inflight = True
        return True, True

    def _on_success(self, is_probe: bool) -> None:
        if is_probe:
            self._probe_inflight = False
            self._broken = False
            logger.info("Rate-limit redis backend recovered — resuming cross-bot coordination")

    def _on_error(self, exc: Exception, is_probe: bool) -> None:
        # re-arm the breaker BEFORE releasing the probe slot: once the slot frees,
        # a racing _route() must already see the fresh cooldown deadline
        was_broken = self._broken
        self._broken = True
        self._broken_until = time.monotonic() + self._breaker_cooldown_s
        if is_probe:
            self._probe_inflight = False
        if not was_broken:
            logger.warning(
                f"Rate-limit redis backend unavailable ({exc!r}) — "
                f"falling back to local buckets for {self._breaker_cooldown_s:.0f}s"
            )

    async def acquire(self, key: str, weight: float, capacity: float, refill_rate: float) -> float:
        use_primary, is_probe = self._route()
        if use_primary:
            try:
                waited = await self._primary.acquire(key, weight, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e, is_probe)
            except BaseException:
                # CancelledError and non-enumerated errors: release the probe slot and
                # propagate — the breaker state itself is left as-is
                if is_probe:
                    self._probe_inflight = False
                raise
            else:
                self._on_success(is_probe)
                return waited
        return await self._fallback.acquire(key, weight, capacity, refill_rate)

    async def get_remaining(self, key: str, capacity: float = 0, refill_rate: float = 0) -> float | None:
        use_primary, is_probe = self._route()
        if use_primary:
            try:
                remaining = await self._primary.get_remaining(key, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e, is_probe)
            except BaseException:
                if is_probe:
                    self._probe_inflight = False
                raise
            else:
                self._on_success(is_probe)
                return remaining
        return await self._fallback.get_remaining(key, capacity, refill_rate)

    async def set_remaining(self, key: str, remaining: float, capacity: float = 0, refill_rate: float = 0) -> None:
        use_primary, is_probe = self._route()
        if use_primary:
            try:
                await self._primary.set_remaining(key, remaining, capacity, refill_rate)
            except _BACKEND_ERRORS as e:
                self._on_error(e, is_probe)
            except BaseException:
                if is_probe:
                    self._probe_inflight = False
                raise
            else:
                self._on_success(is_probe)
                return
        await self._fallback.set_remaining(key, remaining, capacity, refill_rate)

    async def close(self) -> None:
        try:
            await self._primary.close()
        finally:
            await self._fallback.close()
