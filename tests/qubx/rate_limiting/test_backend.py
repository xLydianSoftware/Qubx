"""Tests for InMemoryBackend — bucket creation from a header sync and reservation-safe pinning."""

import asyncio

import pytest

from qubx.rate_limiting import PoolConfig
from qubx.rate_limiting.backend import InMemoryBackend
from qubx.rate_limiting.pools import RatePool


class TestSetRemaining:
    @pytest.mark.asyncio
    async def test_creates_missing_bucket_when_capacity_is_given(self):
        """A header can arrive before the first acquire; the bucket has to be created for it."""
        backend = InMemoryBackend()
        await backend.set_remaining("k", 5.0, capacity=100, refill_rate=10)
        assert await backend.get_remaining("k") == pytest.approx(5.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_without_capacity_is_a_noop(self):
        """Negative control: a zero-capacity bucket would clamp every later acquire to 0."""
        backend = InMemoryBackend()
        await backend.set_remaining("k", 5.0)
        assert await backend.get_remaining("k") is None

    @pytest.mark.asyncio
    async def test_clamps_to_capacity(self):
        backend = InMemoryBackend()
        await backend.set_remaining("k", 500.0, capacity=100, refill_rate=10)
        assert await backend.get_remaining("k") == pytest.approx(100.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_preserves_outstanding_reservations(self):
        """R1: pinning to an exchange-reported level must not erase a sleeping waiter's debt.

        A naive `self._tokens = min(capacity, tokens)` reports the full 10 here — the waiter would
        still fire *and* new arrivals would spend against a balance that no longer carries it.
        """
        backend = InMemoryBackend()
        key, capacity, refill = "k", 10.0, 0.1

        await backend.acquire(key, 10.0, capacity, refill)  # drains the bucket
        waiter = asyncio.create_task(backend.acquire(key, 6.0, capacity, refill))
        await asyncio.sleep(0.05)

        await backend.set_remaining(key, 10.0, capacity, refill)

        remaining = await backend.get_remaining(key, capacity, refill)
        assert remaining == pytest.approx(4.0, abs=0.5), "outstanding reservation was not kept owed"

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    @pytest.mark.asyncio
    async def test_preserves_reservations_across_repeated_syncs(self):
        """Every response header re-pins, so one sync must not be the whole life of the debt."""
        backend = InMemoryBackend()
        key, capacity, refill = "k", 10.0, 0.1

        await backend.acquire(key, 10.0, capacity, refill)  # drains the bucket
        waiter = asyncio.create_task(backend.acquire(key, 6.0, capacity, refill))
        await asyncio.sleep(0.05)

        for sync in (1, 2, 3):
            await backend.set_remaining(key, 10.0, capacity, refill)
            remaining = await backend.get_remaining(key, capacity, refill)
            assert remaining == pytest.approx(4.0, abs=0.5), f"sync #{sync} dropped the reservation"

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter


class TestGetRemaining:
    @pytest.mark.asyncio
    async def test_is_safe_to_call_during_an_acquire(self):
        backend = InMemoryBackend()
        key, capacity, refill = "k", 10.0, 0.1

        await backend.acquire(key, 10.0, capacity, refill)
        waiter = asyncio.create_task(backend.acquire(key, 5.0, capacity, refill))
        await asyncio.sleep(0.05)

        remaining = await asyncio.wait_for(backend.get_remaining(key, capacity, refill), timeout=1.0)
        assert remaining == 0.0, "outstanding debt must read as 0, never negative"

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    @pytest.mark.asyncio
    async def test_reports_a_positive_balance_when_nothing_is_owed(self):
        """Negative control: the 0 above is the debt, not a backend that always reads 0."""
        backend = InMemoryBackend()
        await backend.acquire("k", 4.0, 10.0, 0.1)
        assert await backend.get_remaining("k", 10.0, 0.1) == pytest.approx(6.0, abs=0.5)


class TestSyncTaskLifecycle:
    """``RatePool.sync`` fires a background task per response header — it must not leak or go quiet."""

    async def test_sync_failure_is_retrieved_not_left_unhandled(self):
        pool = RatePool(PoolConfig("p", "ip", 100, 10.0), "ex", "local", _ExplodingBackend())
        unhandled: list = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: unhandled.append(ctx))

        pool.sync(50)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert unhandled == [], f"unretrieved task exception: {unhandled}"
        assert pool._sync_tasks == set(), "task reference leaked after completion"

    async def test_sync_task_is_referenced_while_in_flight(self):
        # negative control: an unreferenced task can be collected mid-flight
        pool = RatePool(PoolConfig("p", "ip", 100, 10.0), "ex", "local", _SlowBackend())
        pool.sync(50)
        assert len(pool._sync_tasks) == 1
        await asyncio.sleep(0.05)
        assert pool._sync_tasks == set()


class _ExplodingBackend(InMemoryBackend):
    async def set_remaining(self, key, remaining, capacity=0, refill_rate=0):
        raise RuntimeError("redis blip")


class _SlowBackend(InMemoryBackend):
    async def set_remaining(self, key, remaining, capacity=0, refill_rate=0):
        await asyncio.sleep(0.01)
