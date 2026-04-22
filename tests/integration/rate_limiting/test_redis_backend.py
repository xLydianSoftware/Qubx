"""
Integration tests for RedisBackend rate limit backend.

These tests verify behavior that cannot be exercised with an in-memory backend,
primarily around Redis client / event-loop interaction: the same backend instance
is shared across the simulation event loop and the live ccxt.pro AsyncThreadLoop,
so it must not bind loop-specific state (like asyncio.Lock inside the Redis client)
to the first loop that uses it.
"""

import asyncio
import threading

import pytest

from qubx.rate_limiting.redis_backend import RedisBackend


@pytest.mark.integration
class TestRedisBackendCrossLoop:
    def test_reused_after_first_loop_closed(self, redis_service, clear_rate_limit_keys):
        """
        A backend created before any event loop must still work after the first
        loop that used it has been torn down. This reproduces the warmup → live
        transition: simulation loop uses the rate limiter, then is closed, then
        the live loop uses the same backend instance.
        """
        backend = RedisBackend(redis_service)
        key = "qubx:test:rl:after_close"

        async def use():
            assert await backend.acquire(key, weight=1.0, capacity=100.0, refill_rate=10.0) == 0.0

        first = asyncio.new_event_loop()
        try:
            first.run_until_complete(use())
        finally:
            first.close()

        second = asyncio.new_event_loop()
        try:
            second.run_until_complete(use())
        finally:
            second.close()

    def test_used_concurrently_from_two_loops_on_different_threads(
        self, redis_service, clear_rate_limit_keys
    ):
        """
        Qubx runs the simulator on one loop and the live ccxt AsyncThreadLoop
        on a different thread. Both may hold references to the same backend.
        The backend must not raise 'Future attached to a different loop' when
        used from a second loop on a second thread.
        """
        backend = RedisBackend(redis_service)
        key = "qubx:test:rl:two_threads"

        errors: list[BaseException] = []

        def run_on_new_loop():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    backend.acquire(key, weight=1.0, capacity=100.0, refill_rate=10.0)
                )
            except BaseException as e:
                errors.append(e)
            finally:
                loop.close()

        main = asyncio.new_event_loop()
        try:
            main.run_until_complete(
                backend.acquire(key, weight=1.0, capacity=100.0, refill_rate=10.0)
            )

            t = threading.Thread(target=run_on_new_loop)
            t.start()
            t.join(timeout=5)
            assert not t.is_alive(), "secondary thread did not finish in time"
        finally:
            main.close()

        assert not errors, f"secondary loop failed: {errors[0]!r}"

    def test_get_and_set_remaining_across_loops(self, redis_service, clear_rate_limit_keys):
        """get_remaining / set_remaining should also work after the creating loop is gone."""
        backend = RedisBackend(redis_service)
        key = "qubx:test:rl:get_set"

        async def write():
            await backend.set_remaining(key, remaining=42.0)

        async def read() -> float | None:
            return await backend.get_remaining(key, capacity=100.0, refill_rate=0.0)

        first = asyncio.new_event_loop()
        try:
            first.run_until_complete(write())
        finally:
            first.close()

        second = asyncio.new_event_loop()
        try:
            remaining = second.run_until_complete(read())
        finally:
            second.close()

        assert remaining == pytest.approx(42.0, abs=0.5)
