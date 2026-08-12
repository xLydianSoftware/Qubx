"""Cross-loop / cross-thread behaviour of the gate and the token bucket.

The gate used to be an `asyncio.Event` and the reopen a task, both of which bind to the first loop
that touches them; the bucket used to hold an `asyncio.Lock`. Qubx drives one limiter from the
runner loop, the ccxt `AsyncThreadLoop` and the storage/warmup loops, so any such binding surfaces
as `RuntimeError: ... attached to a different loop`. These are plain unit tests — no Redis.
"""

import asyncio
import threading
from typing import Any, Callable, Coroutine

import pytest

from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, ExchangeRateLimiter, PoolConfig
from qubx.rate_limiting.engine import RateLimitGateTimeout
from qubx.utils.rate_limiter import TokenBucketRateLimiter


def _make_config(cooldown: float = 30.0, gate_max_wait: float = 0.2) -> ExchangeRateLimitConfig:
    return ExchangeRateLimitConfig(
        pools={
            "rest": PoolConfig("rest", "ip", capacity=100, refill_rate=50.0, cooldown=cooldown),
            "orders": PoolConfig("orders", "account", capacity=10, refill_rate=5.0, cooldown=cooldown),
        },
        endpoint_map={"r": EndpointCosts([("rest", 1)])},
        default_costs=EndpointCosts([("rest", 1)]),
        gate_max_wait=gate_max_wait,
    )


def _run_on_new_loop(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(factory())
    finally:
        loop.close()


class TestGateIsNotBoundToALoop:
    def test_gate_awaited_on_one_loop_is_still_a_gate_on_another(self):
        """The gate must survive the loop that first waited on it (bound `asyncio.Event` today)."""
        limiter = ExchangeRateLimiter("x", _make_config(cooldown=30.0, gate_max_wait=0.2))
        try:

            async def on_loop_a():
                limiter.report_limit_hit(pool_name="rest", retry_after=30.0, reason="test")
                await limiter.acquire("r")

            with pytest.raises(RateLimitGateTimeout):
                _run_on_new_loop(on_loop_a)

            try:
                _run_on_new_loop(lambda: limiter.acquire("r"))
            except RateLimitGateTimeout as e:
                assert e.pool_name == "rest"
            except RuntimeError as e:
                pytest.fail(f"cross-loop RuntimeError leaked out of the gate: {e}")
            else:
                pytest.fail("gate should still be closed on the second loop")
        finally:
            limiter.reset_gates()

    def test_gate_max_wait_is_still_enforced(self):
        """Negative control: a deadline-based gate must still time out, not silently pass through."""
        limiter = ExchangeRateLimiter("x", _make_config(cooldown=30.0, gate_max_wait=0.1))
        limiter.report_limit_hit(pool_name="rest", retry_after=30.0, reason="test")
        try:
            with pytest.raises(RateLimitGateTimeout) as exc_info:
                _run_on_new_loop(lambda: limiter.acquire("r"))
            assert exc_info.value.pool_name == "rest"
        finally:
            limiter.reset_gates()


class TestGateFromThreadWithoutLoop:
    def test_report_limit_hit_does_not_raise(self):
        """`report_limit_hit` is called from ccxt callbacks that may run on a bare thread."""
        limiter = ExchangeRateLimiter("x", _make_config(cooldown=30.0))
        errors: list[BaseException] = []

        def worker():
            try:
                limiter.report_limit_hit(pool_name="rest", retry_after=30.0, reason="bare thread")
            except BaseException as e:
                errors.append(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        try:
            assert not thread.is_alive(), "worker thread did not finish"
            assert not errors, f"report_limit_hit raised without a running loop: {errors[0]!r}"
            assert limiter.is_gate_closed("rest")
        finally:
            limiter.reset_gates()

    def test_gate_closed_from_thread_reopens_for_a_waiter_on_another_loop(self):
        limiter = ExchangeRateLimiter("x", _make_config(cooldown=1.0, gate_max_wait=10.0))
        thread = threading.Thread(
            target=lambda: limiter.report_limit_hit(pool_name="rest", retry_after=1.0, reason="bare thread")
        )
        thread.start()
        thread.join(timeout=5)
        try:
            assert limiter.is_gate_closed("rest")

            async def waiter():
                await asyncio.wait_for(limiter.acquire("r"), timeout=10.0)

            _run_on_new_loop(waiter)
            assert not limiter.is_gate_closed("rest")
        finally:
            limiter.reset_gates()


def _one_pool_config(capacity: float, refill_rate: float) -> ExchangeRateLimitConfig:
    return ExchangeRateLimitConfig(
        pools={"rest": PoolConfig("rest", "ip", capacity=capacity, refill_rate=refill_rate)},
        endpoint_map={"r": EndpointCosts([("rest", 1)])},
        default_costs=EndpointCosts([("rest", 1)]),
        gate_max_wait=5.0,
    )


class TestTokenBucketAcrossLoops:
    def test_a_second_loop_pays_for_what_the_first_loop_spent(self):
        # refill is slow enough that loop A's drain is still owed when loop B asks
        limiter = ExchangeRateLimiter("x", _one_pool_config(capacity=10, refill_rate=0.5))

        _run_on_new_loop(lambda: limiter.acquire("r", weight_override=10))

        async def contend_on_loop_b():
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(limiter.acquire("r"), 0.1)

        try:
            _run_on_new_loop(contend_on_loop_b)
        except RuntimeError as e:
            pytest.fail(f"cross-loop RuntimeError leaked out of the token bucket: {e}")

    def test_an_undrained_bucket_serves_a_second_loop_at_once(self):
        """Negative control: the timeout above is loop A's spend, not a per-loop stall."""
        limiter = ExchangeRateLimiter("x", _one_pool_config(capacity=10, refill_rate=0.5))

        _run_on_new_loop(lambda: limiter.acquire("r"))

        async def contend_on_loop_b():
            await asyncio.wait_for(asyncio.gather(limiter.acquire("r"), limiter.acquire("r")), 0.1)

        try:
            _run_on_new_loop(contend_on_loop_b)
        except RuntimeError as e:
            pytest.fail(f"cross-loop RuntimeError leaked out of the token bucket: {e}")

    def test_cancelled_acquire_leaks_its_reservation(self):
        """R7: the debit happens before the sleep and is never rolled back — refill absorbs it."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0, name="test")

        async def scenario():
            await limiter.acquire(10)
            leaked = asyncio.create_task(limiter.acquire(5))
            await asyncio.sleep(0.05)
            leaked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await leaked

            # the cancelled task's 5 tokens stay owed, so a fresh caller still has to wait them off
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(limiter.acquire(1), timeout=0.2)
            assert limiter.get_available_tokens() == 0.0

        _run_on_new_loop(scenario)
