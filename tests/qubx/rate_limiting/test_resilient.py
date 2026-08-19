"""Tests for ResilientRateLimitBackend — redis primary + local fallback + circuit breaker.

No sleeps: breaker timing is forced by assigning ``backend._broken_until`` directly, and the
single-probe race is forced by blocking the primary on an ``asyncio.Event``.
"""

import asyncio
import time

import pytest

from qubx.rate_limiting.backend import IRateLimitBackend
from qubx.rate_limiting.resilient import ResilientRateLimitBackend


class FakeBackend(IRateLimitBackend):
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.get_calls = 0
        self.set_calls = 0
        self.closed = False
        self.raise_exc: Exception | None = None
        self.block_on: asyncio.Event | None = None

    async def acquire(self, key, weight, capacity, refill_rate) -> float:
        self.acquire_calls += 1
        if self.block_on is not None:
            await self.block_on.wait()
        if self.raise_exc is not None:
            raise self.raise_exc
        return 0.0

    async def get_remaining(self, key, capacity=0, refill_rate=0):
        self.get_calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return 1.0

    async def set_remaining(self, key, remaining, capacity=0, refill_rate=0) -> None:
        self.set_calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_passthrough_when_primary_healthy():
    primary, fallback = FakeBackend(), FakeBackend()
    resilient = ResilientRateLimitBackend(primary, fallback)

    waited = await resilient.acquire("k", 1.0, 10.0, 5.0)
    remaining = await resilient.get_remaining("k", 10.0, 5.0)
    await resilient.set_remaining("k", 5.0, 10.0, 5.0)

    assert waited == 0.0
    assert remaining == 1.0
    assert primary.acquire_calls == 1
    assert primary.get_calls == 1
    assert primary.set_calls == 1
    assert fallback.acquire_calls == 0
    assert fallback.get_calls == 0
    assert fallback.set_calls == 0


@pytest.mark.asyncio
async def test_error_opens_breaker_and_serves_fallback():
    primary, fallback = FakeBackend(), FakeBackend()
    primary.raise_exc = ConnectionError("redis down")
    resilient = ResilientRateLimitBackend(primary, fallback)

    waited = await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert waited == 0.0
    # load-bearing assertions: FakeBackend returns 0.0 on both sides, so `waited` alone
    # doesn't prove routing — the call counters do.
    assert primary.acquire_calls == 1
    assert fallback.acquire_calls == 1
    assert resilient._broken is True

    # second call must not touch primary — breaker is open
    await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert primary.acquire_calls == 1
    assert fallback.acquire_calls == 2


@pytest.mark.asyncio
async def test_probe_after_cooldown_single_caller():
    primary, fallback = FakeBackend(), FakeBackend()
    primary.raise_exc = ConnectionError("redis down")
    resilient = ResilientRateLimitBackend(primary, fallback)

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # opens breaker
    assert primary.acquire_calls == 1

    resilient._broken_until = 0.0  # force cooldown expiry
    primary.raise_exc = None  # primary healed

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # probes primary, closes breaker
    assert primary.acquire_calls == 2
    assert resilient._broken is False

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # breaker closed, hits primary directly
    assert primary.acquire_calls == 3
    assert fallback.acquire_calls == 1  # only the original failure


@pytest.mark.asyncio
async def test_concurrent_callers_stay_local_while_probe_inflight():
    primary, fallback = FakeBackend(), FakeBackend()
    primary.raise_exc = ConnectionError("redis down")
    resilient = ResilientRateLimitBackend(primary, fallback)

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # opens breaker
    assert primary.acquire_calls == 1

    resilient._broken_until = 0.0  # force cooldown expiry
    primary.raise_exc = None  # primary healed, but ...
    primary.block_on = asyncio.Event()  # ... the probe blocks in-flight

    task_a = asyncio.create_task(resilient.acquire("k", 1.0, 10.0, 5.0))
    await asyncio.sleep(0)  # let A claim the probe slot and start blocking on primary
    await asyncio.sleep(0)
    assert primary.acquire_calls == 2  # 1 (initial failure) + 1 (A's in-flight probe)
    assert resilient._probe_inflight is True

    # B arrives while the probe is in flight — must be served by the fallback
    waited_b = await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert waited_b == 0.0
    assert primary.acquire_calls == 2  # unchanged — only A touched primary
    assert fallback.acquire_calls == 2  # original failure + B

    primary.block_on.set()  # release A
    await task_a
    assert resilient._broken is False

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # breaker closed now, hits primary
    assert primary.acquire_calls == 3


@pytest.mark.asyncio
async def test_probe_failure_reopens():
    primary, fallback = FakeBackend(), FakeBackend()
    primary.raise_exc = ConnectionError("redis down")
    resilient = ResilientRateLimitBackend(primary, fallback)

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # opens breaker
    assert primary.acquire_calls == 1

    resilient._broken_until = 0.0  # force cooldown expiry, primary still broken
    await resilient.acquire("k", 1.0, 10.0, 5.0)  # probes primary again — fails
    assert primary.acquire_calls == 2
    assert resilient._broken is True

    # immediate next call must go straight to fallback, no further primary touch
    await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert primary.acquire_calls == 2
    assert fallback.acquire_calls == 3


@pytest.mark.asyncio
async def test_set_and_get_follow_policy():
    primary, fallback = FakeBackend(), FakeBackend()
    resilient = ResilientRateLimitBackend(primary, fallback)
    resilient._broken = True
    resilient._broken_until = time.monotonic() + 30.0

    remaining = await resilient.get_remaining("k", 10.0, 5.0)
    await resilient.set_remaining("k", 5.0, 10.0, 5.0)

    assert remaining == 1.0  # fallback's canned value
    assert primary.get_calls == 0
    assert primary.set_calls == 0
    assert fallback.get_calls == 1
    assert fallback.set_calls == 1


@pytest.mark.asyncio
async def test_cancelled_error_propagates():
    primary, fallback = FakeBackend(), FakeBackend()
    primary.raise_exc = asyncio.CancelledError()
    resilient = ResilientRateLimitBackend(primary, fallback)

    with pytest.raises(asyncio.CancelledError):
        await resilient.acquire("k", 1.0, 10.0, 5.0)

    assert resilient._broken is False
    assert fallback.acquire_calls == 0


@pytest.mark.asyncio
async def test_close_closes_both():
    primary, fallback = FakeBackend(), FakeBackend()
    resilient = ResilientRateLimitBackend(primary, fallback)

    await resilient.close()

    assert primary.closed is True
    assert fallback.closed is True


@pytest.mark.asyncio
async def test_cancelled_probe_releases_slot():
    """Fix round 1 / F1 (review Finding 1): a cancelled probe must not wedge the breaker open forever."""
    primary, fallback = FakeBackend(), FakeBackend()
    primary.raise_exc = ConnectionError("redis down")
    resilient = ResilientRateLimitBackend(primary, fallback)

    await resilient.acquire("k", 1.0, 10.0, 5.0)  # opens breaker
    assert primary.acquire_calls == 1

    resilient._broken_until = 0.0  # force cooldown expiry
    primary.raise_exc = None  # primary healed, but ...
    primary.block_on = asyncio.Event()  # ... the probe blocks in-flight

    probe = asyncio.create_task(resilient.acquire("k", 1.0, 10.0, 5.0))
    await asyncio.sleep(0)  # let the probe claim the slot and start blocking on primary
    await asyncio.sleep(0)
    assert primary.acquire_calls == 2
    assert resilient._probe_inflight is True

    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe

    assert resilient._probe_inflight is False, "cancelled probe leaked the slot — breaker is wedged open"
    assert resilient._broken is True  # cancellation doesn't heal the breaker by itself

    # the breaker must still be able to recover: force cooldown expiry again and probe once more
    primary.block_on = None
    resilient._broken_until = 0.0
    await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert primary.acquire_calls == 3  # initial failure + cancelled probe + this recovering probe
    assert resilient._broken is False


@pytest.mark.asyncio
async def test_stale_success_does_not_close_breaker():
    """Fix round 1 / F2 (review Finding 2): only the probe's own success may close the breaker."""
    primary, fallback = FakeBackend(), FakeBackend()
    resilient = ResilientRateLimitBackend(primary, fallback)

    event = asyncio.Event()
    primary.block_on = event  # call A blocks here while the breaker is still closed

    task_a = asyncio.create_task(resilient.acquire("k", 1.0, 10.0, 5.0))
    await asyncio.sleep(0)  # let A dispatch to primary and start blocking, while healthy
    await asyncio.sleep(0)
    assert primary.acquire_calls == 1

    # call B fails and opens the breaker while A is still in flight
    primary.block_on = None  # only A's already-captured Event reference blocks
    primary.raise_exc = ConnectionError("redis down")
    await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert resilient._broken is True
    assert primary.acquire_calls == 2

    # release A: it now succeeds, but it was dispatched before the breaker opened —
    # a stale success carries no evidence redis is healthy and must not close the breaker
    primary.raise_exc = None
    event.set()
    await task_a
    assert resilient._broken is True

    # the next call must go straight to the fallback — no further primary touch
    await resilient.acquire("k", 1.0, 10.0, 5.0)
    assert primary.acquire_calls == 2
