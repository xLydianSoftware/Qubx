"""Rate limiter correctness across two real event loops (quantkit#106).

One ExchangeRateLimiter (LOCAL in-memory backend) is shared by a venue's realtime
websocket loop and the process-wide BulkRestLoop. These tests drive acquires from two
real ``BackgroundEventLoop`` threads and assert:

- token accounting is shared (no double budget, no over-issue),
- gates close/reopen for waiters on BOTH loops (per-loop mirror events, no cross-loop
  ``asyncio.Event`` usage),
- gate timeouts and resets behave identically on either loop.
"""

import time
from concurrent.futures import Future

import pytest

from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, ExchangeRateLimiter, PoolConfig
from qubx.rate_limiting.engine import RateLimitGateTimeout
from qubx.utils.misc import BackgroundEventLoop
from qubx.utils.rate_limiter import TokenBucketRateLimiter


@pytest.fixture
def two_loops():
    a = BackgroundEventLoop(name="rl-test-realtime")
    b = BackgroundEventLoop(name="rl-test-bulk")
    yield a, b
    a.stop()
    b.stop()


def _config(
    *,
    capacity: float = 100.0,
    refill_rate: float = 0.001,
    cooldown: float = 0.5,
    gate_max_wait: float = 1.0,
    quota_capacity: float = 100.0,
) -> ExchangeRateLimitConfig:
    return ExchangeRateLimitConfig(
        pools={
            "rest": PoolConfig(
                name="rest",
                scope="ip",
                capacity=capacity,
                refill_rate=refill_rate,
                pool_type="rate",
                cooldown=cooldown,
            ),
            "quota": PoolConfig(
                name="quota",
                scope="address",
                capacity=quota_capacity,
                refill_rate=0,
                pool_type="quota",
                cooldown=cooldown,
            ),
        },
        endpoint_map={
            "rest_call": EndpointCosts([("rest", 1)]),
            "quota_call": EndpointCosts([("quota", 1)]),
        },
        default_costs=EndpointCosts([]),
        gate_max_wait=gate_max_wait,
    )


def _submit_acquires(loop: BackgroundEventLoop, limiter: ExchangeRateLimiter, endpoint: str, n: int) -> Future:
    async def do_acquires():
        for _ in range(n):
            await limiter.acquire(endpoint)

    return loop.submit(do_acquires())


class TestSharedTokens:
    def test_tokens_shared_across_two_loops(self, two_loops):
        """20 acquires split across two loops consume ONE shared budget (~80 of 100 left)."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(refill_rate=0.001))

        fa = _submit_acquires(a, limiter, "rest_call", 10)
        fb = _submit_acquires(b, limiter, "rest_call", 10)
        fa.result(5)
        fb.result(5)

        state = a.run_sync(limiter.get_pool_state("rest"), timeout=5)
        assert state is not None
        assert state["remaining"] == pytest.approx(80, abs=1)
        assert state["consumed"] == pytest.approx(20)

    def test_over_capacity_waits_for_shared_refill(self, two_loops):
        """Draining beyond capacity from two loops blocks on the SAME refill clock."""
        a, b = two_loops
        # capacity 6, refill 10/s: 10 total acquires need 4 extra tokens -> >= 0.4s
        limiter = ExchangeRateLimiter("test", _config(capacity=6, refill_rate=10.0))

        t0 = time.monotonic()
        fa = _submit_acquires(a, limiter, "rest_call", 5)
        fb = _submit_acquires(b, limiter, "rest_call", 5)
        fa.result(10)
        fb.result(10)
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.3  # would be ~0 if each loop had its own bucket

    def test_quota_pool_no_over_issue_across_loops(self, two_loops):
        """Exactly quota_capacity acquires across two loops succeed; the pool hits 0, not negative."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(quota_capacity=100))

        fa = _submit_acquires(a, limiter, "quota_call", 50)
        fb = _submit_acquires(b, limiter, "quota_call", 50)
        fa.result(5)
        fb.result(5)

        assert limiter.get_quota_remaining("quota") == 0
        with pytest.raises(RateLimitGateTimeout):
            a.run_sync(limiter.acquire("quota_call"), timeout=5)


class TestGatesAcrossLoops:
    def test_gate_closed_blocks_waiters_on_both_loops_until_timer_reopen(self, two_loops):
        """A 429 reported from anywhere gates BOTH loops; the timer reopen wakes both."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(gate_max_wait=5.0))

        limiter.report_limit_hit(pool_name="rest", retry_after=0.4, reason="test 429")
        assert limiter.is_gate_closed("rest")

        t0 = time.monotonic()
        fa = _submit_acquires(a, limiter, "rest_call", 1)
        fb = _submit_acquires(b, limiter, "rest_call", 1)
        fa.result(5)
        fb.result(5)
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.3  # both waited for the reopen, not just the closing loop
        assert not limiter.is_gate_closed("rest")

    def test_gate_timeout_raises_on_both_loops(self, two_loops):
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(gate_max_wait=0.15))

        limiter.report_limit_hit(pool_name="rest", retry_after=10.0, reason="long cooldown")

        for loop in (a, b):
            with pytest.raises(RateLimitGateTimeout) as exc_info:
                loop.run_sync(limiter.acquire("rest_call"), timeout=5)
            assert exc_info.value.pool_name == "rest"

    def test_reset_gates_wakes_waiters_on_both_loops(self, two_loops):
        """reset_gates() from a plain thread (no loop) reopens for waiters on both loops."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(gate_max_wait=10.0))

        limiter.report_limit_hit(pool_name="rest", retry_after=60.0, reason="stuck gate")
        fa = _submit_acquires(a, limiter, "rest_call", 1)
        fb = _submit_acquires(b, limiter, "rest_call", 1)
        time.sleep(0.1)  # let both park on their per-loop mirror events
        assert not fa.done() and not fb.done()

        limiter.reset_gates()
        fa.result(2)
        fb.result(2)
        assert not limiter.is_gate_closed("rest")

    def test_gate_extension_keeps_both_loops_parked(self, two_loops):
        """A second hit while closed extends the cooldown; a stale first timer must not reopen early."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(gate_max_wait=10.0))

        limiter.report_limit_hit(pool_name="rest", retry_after=0.2, reason="first hit")
        time.sleep(0.05)
        limiter.report_limit_hit(pool_name="rest", retry_after=0.6, reason="extension")

        t0 = time.monotonic()
        fa = _submit_acquires(a, limiter, "rest_call", 1)
        fb = _submit_acquires(b, limiter, "rest_call", 1)
        fa.result(5)
        fb.result(5)
        elapsed = time.monotonic() - t0

        # The first (0.2s) timer was superseded — waiters see the extended cooldown.
        assert elapsed >= 0.4

    def test_quota_sync_reopens_for_other_loop(self, two_loops):
        """Quota depleted (gate closed) then synced positive — the OTHER loop can acquire again."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config())

        limiter.sync_from_exchange("quota", remaining=0)
        assert limiter.is_gate_closed("quota")
        with pytest.raises(RateLimitGateTimeout):
            a.run_sync(limiter.acquire("quota_call"), timeout=5)

        limiter.sync_from_exchange("quota", remaining=10)
        b.run_sync(limiter.acquire("quota_call"), timeout=5)
        assert limiter.get_quota_remaining("quota") == 9


class TestTokenBucketTwoLoops:
    def test_token_bucket_shared_by_two_real_loops(self, two_loops):
        """The LOCAL bucket primitive itself is loop-agnostic: interleaved acquires from
        two loops never trip a loop-affinity error and never over-issue."""
        a, b = two_loops
        bucket = TokenBucketRateLimiter(capacity=50, refill_rate=0.001, name="two-loop")

        async def take(n: int) -> None:
            for _ in range(n):
                await bucket.acquire(1)

        fa = a.submit(take(25))
        fb = b.submit(take(25))
        fa.result(5)
        fb.result(5)

        assert bucket.get_available_tokens() == pytest.approx(0, abs=1)

    def test_token_bucket_refill_wait_spans_loops(self, two_loops):
        a, b = two_loops
        bucket = TokenBucketRateLimiter(capacity=4, refill_rate=10.0, name="two-loop-wait")

        async def take(n: int) -> None:
            for _ in range(n):
                await bucket.acquire(1)

        t0 = time.monotonic()
        fa = a.submit(take(4))
        fb = b.submit(take(4))
        fa.result(10)
        fb.result(10)
        elapsed = time.monotonic() - t0

        # 8 tokens from a 4-token bucket at 10/s -> at least ~0.4s of shared refill
        assert elapsed >= 0.3

    def test_set_tokens_visible_across_loops(self, two_loops):
        a, _ = two_loops
        bucket = TokenBucketRateLimiter(capacity=100, refill_rate=0.001, name="sync")

        a.run_sync(bucket.acquire(60), timeout=5)
        bucket.set_tokens(5.0)
        assert bucket.get_available_tokens() == pytest.approx(5, abs=1)


class TestMirrorEventHygiene:
    def test_no_cross_loop_event_usage(self, two_loops):
        """Each loop gets its OWN mirror event — the gate never awaits one loop's Event
        from another loop (which asyncio forbids)."""
        a, b = two_loops
        limiter = ExchangeRateLimiter("test", _config(gate_max_wait=5.0))
        pool = limiter._pools["rest"]

        # Park a waiter on each loop while the gate is closed
        limiter.report_limit_hit(pool_name="rest", retry_after=0.3, reason="hygiene")
        fa = _submit_acquires(a, limiter, "rest_call", 1)
        fb = _submit_acquires(b, limiter, "rest_call", 1)
        time.sleep(0.1)

        events = dict(pool._gate._events)
        assert set(events.keys()) == {a.loop, b.loop}
        assert events[a.loop] is not events[b.loop]

        fa.result(5)
        fb.result(5)
