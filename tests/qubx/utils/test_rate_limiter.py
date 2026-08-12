"""Unit tests for rate limiter module."""

import asyncio
import time

import pytest

from qubx import logger
from qubx.utils.misc import BackgroundEventLoop
from qubx.utils.rate_limiter import RateLimiterRegistry, TokenBucketRateLimiter, rate_limited


@pytest.fixture
def captured_logs():
    lines: list[tuple[str, str]] = []
    sink_id = logger.add(lambda m: lines.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    yield lines
    logger.remove(sink_id)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter class."""

    @pytest.mark.asyncio
    async def test_basic_acquire(self):
        """Test basic token acquisition."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        # Should not block
        await limiter.acquire(5)
        assert limiter.get_available_tokens() == pytest.approx(5, rel=0.1)

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_insufficient_tokens(self):
        """Test that acquire blocks when tokens are insufficient."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        # Consume all tokens
        await limiter.acquire(10)
        # Tokens should be very close to 0 (allow for tiny refill during execution)
        assert limiter.get_available_tokens() < 0.1

        # Next acquire should block until tokens refill
        start = time.monotonic()
        await limiter.acquire(5)  # Need 5 tokens at 10/sec = 0.5s wait
        elapsed = time.monotonic() - start

        assert elapsed >= 0.4  # Allow some slack for timing
        assert elapsed < 2.0  # But bounded — generous, since `just test` runs under xdist load

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=100, name="test")

        # Consume tokens
        await limiter.acquire(50)
        assert limiter.get_available_tokens() == pytest.approx(50, rel=0.1)

        # Wait for refill (0.3 seconds = 30 tokens at 100/sec)
        await asyncio.sleep(0.3)

        # Should have ~80 tokens now (50 + 30), never above capacity
        available = limiter.get_available_tokens()
        assert 75 <= available <= 100

    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        """Test that tokens don't exceed capacity."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=100, name="test")

        # Wait for refill
        await asyncio.sleep(0.5)

        # Should be capped at capacity
        assert limiter.get_available_tokens() == pytest.approx(10, rel=0.1)

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity_clamps_and_warns(self, captured_logs):
        """A weight no bucket can ever hold must not raise (nor stall) — it is clamped."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        await asyncio.wait_for(limiter.acquire(20), timeout=2.0)

        assert limiter.get_available_tokens() == pytest.approx(0.0, abs=0.1)
        assert [m for lvl, m in captured_logs if lvl == "WARNING" and "exceeds capacity" in m]

    @pytest.mark.asyncio
    async def test_acquire_at_capacity_does_not_warn(self, captured_logs):
        """Negative control: the clamp warning fires only above capacity."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        await limiter.acquire(10)

        assert not [m for _, m in captured_logs if "exceeds capacity" in m]

    @pytest.mark.asyncio
    async def test_waiter_does_not_block_a_caller_with_budget(self):
        """The reservation is taken under the lock; the sleep happens outside it."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1, name="test")

        await limiter.acquire(10)  # drains
        waiter = asyncio.create_task(limiter.acquire(5))  # ~5s deficit
        await asyncio.sleep(0.05)

        limiter.set_tokens(10)  # exchange re-pins to full; the 5 stays owed → 5 free

        start = time.monotonic()
        await asyncio.wait_for(limiter.acquire(1), timeout=0.5)
        assert time.monotonic() - start < 0.5

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pinned,expected", [(100, 70), (10, 0)])
    async def test_set_tokens_keeps_outstanding_debt_owed(self, pinned: float, expected: float):
        """A pin is reduced by what sleepers still owe; a pin below the debt leaves nothing."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1, name="test")

        await limiter.acquire(100)  # drains
        waiter = asyncio.create_task(limiter.acquire(30))  # 30 owed
        await asyncio.sleep(0.05)

        limiter.set_tokens(pinned)
        assert limiter.get_available_tokens() == pytest.approx(expected, abs=1.0)

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    @pytest.mark.asyncio
    async def test_set_tokens_keeps_debt_owed_across_repeated_syncs(self):
        """Headers re-pin on every response, so the debt has to outlive every sync, not just the first."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1, name="test")

        await limiter.acquire(100)  # drains
        waiter = asyncio.create_task(limiter.acquire(30))  # 30 owed
        await asyncio.sleep(0.05)

        for sync in (1, 2, 3):
            limiter.set_tokens(100)
            assert limiter.get_available_tokens() == pytest.approx(70, abs=2.0), f"sync #{sync} dropped the debt"

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    @pytest.mark.asyncio
    async def test_set_tokens_pins_verbatim_once_the_waiter_woke(self):
        """The debt is released when the reservation becomes visible to the venue."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=20, name="test")

        await limiter.acquire(100)  # drains
        await asyncio.wait_for(limiter.acquire(30), timeout=10)  # ~1.5s of deficit, then wakes

        limiter.set_tokens(70)
        assert limiter.get_available_tokens() == pytest.approx(70, abs=5.0)

    @pytest.mark.asyncio
    async def test_set_tokens_without_debt_pins_verbatim(self):
        """Negative control: with nothing owed the pin is applied as-is (and clamped)."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1, name="test")

        limiter.set_tokens(40)
        assert limiter.get_available_tokens() == pytest.approx(40, abs=1.0)

        limiter.set_tokens(500)
        assert limiter.get_available_tokens() == pytest.approx(100, abs=1.0)

    @pytest.mark.asyncio
    async def test_repeated_set_tokens_without_debt_pins_verbatim(self):
        """Negative control for the repeated-sync case: nothing owed, nothing subtracted."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1, name="test")

        for sync in (1, 2, 3):
            limiter.set_tokens(100)
            assert limiter.get_available_tokens() == pytest.approx(100, abs=1.0), f"sync #{sync} lost tokens"

    @pytest.mark.asyncio
    async def test_zero_weight_acquire_skips_an_existing_deficit(self):
        """ccxt prices Kraken AddOrder/CancelOrder at 0; such a call must not queue behind data-fetch
        debt (this bucket is Kraken spot: a -10 deficit at 0.33/s is a 30s sleep)."""
        limiter = TokenBucketRateLimiter(capacity=20, refill_rate=0.33, name="test")

        await limiter.acquire(20)  # drains
        debtor = asyncio.create_task(limiter.acquire(10))
        await asyncio.sleep(0.05)
        owed = limiter._tokens  # the deficit is not observable through the public read (it clamps at 0)

        await asyncio.wait_for(limiter.acquire(0), timeout=0.5)

        assert limiter._tokens == pytest.approx(owed, abs=0.1), "a free call must not touch the bucket"

        debtor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await debtor

    @pytest.mark.asyncio
    async def test_positive_weight_still_waits_out_the_same_deficit(self):
        """Negative control: only weight 0 is free."""
        limiter = TokenBucketRateLimiter(capacity=20, refill_rate=0.33, name="test")

        await limiter.acquire(20)
        debtor = asyncio.create_task(limiter.acquire(10))
        await asyncio.sleep(0.05)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire(1), timeout=0.2)

        debtor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await debtor

    @pytest.mark.asyncio
    async def test_fifo_order_preserved_under_deficit(self):
        """Each acquirer gets a distinct increasing deadline, so wakeups keep launch order."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")
        order: list[int] = []

        async def worker(task_id: int):
            await limiter.acquire(5)
            order.append(task_id)

        await asyncio.gather(*(worker(i) for i in range(5)))

        assert order == [0, 1, 2, 3, 4]

    def test_shared_across_threads(self):
        """One limiter driven from two loops on two threads — one budget, no loop binding."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1.0, name="shared")
        loop_a = BackgroundEventLoop("rl-test-a")
        loop_b = BackgroundEventLoop("rl-test-b")

        try:
            future_a = loop_a.submit(limiter.acquire(30))
            future_b = loop_b.submit(limiter.acquire(20))
            future_a.result(timeout=10)
            future_b.result(timeout=10)
        finally:
            loop_a.stop()
            loop_b.stop()

        assert limiter.get_available_tokens() == pytest.approx(50, abs=5)

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Test that concurrent acquires are properly serialized."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        results = []

        async def acquire_and_record(weight: float, task_id: int):
            await limiter.acquire(weight)
            results.append((task_id, time.monotonic()))

        # Start 5 tasks that need 5 tokens each (total 25 tokens)
        # Should take ~1.5 seconds for all to complete
        start = time.monotonic()
        tasks = [acquire_and_record(5, i) for i in range(5)]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        # All tasks should complete
        assert len(results) == 5

        # Should take at least 1 second (need 15 extra tokens at 10/sec)
        assert elapsed >= 1.0

    @pytest.mark.asyncio
    async def test_fractional_weights(self):
        """Test that fractional weights work correctly."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        # Acquire fractional tokens
        await limiter.acquire(2.5)
        assert limiter.get_available_tokens() == pytest.approx(7.5, rel=0.1)

        await limiter.acquire(3.5)
        assert limiter.get_available_tokens() == pytest.approx(4.0, rel=0.1)


class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry class."""

    def test_register_and_get_limiter(self):
        """Test registering and retrieving limiters."""
        registry = RateLimiterRegistry()
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        registry.register_limiter("rest", limiter)
        retrieved = registry.get_limiter("rest")

        assert retrieved is limiter

    def test_get_nonexistent_limiter_raises_error(self):
        """Test that getting non-existent limiter raises KeyError."""
        registry = RateLimiterRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_limiter("nonexistent")

    def test_has_limiter(self):
        """Test checking if limiter exists."""
        registry = RateLimiterRegistry()
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        assert not registry.has_limiter("rest")

        registry.register_limiter("rest", limiter)
        assert registry.has_limiter("rest")

    def test_list_limiters(self):
        """Test listing all registered limiters."""
        registry = RateLimiterRegistry()
        limiter1 = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test1")
        limiter2 = TokenBucketRateLimiter(capacity=20, refill_rate=20, name="test2")

        registry.register_limiter("rest", limiter1)
        registry.register_limiter("ws", limiter2)

        limiters = registry.list_limiters()
        assert sorted(limiters) == ["rest", "ws"]

    def test_multiple_limiters(self):
        """Test managing multiple limiters."""
        registry = RateLimiterRegistry()
        limiter1 = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="rest")
        limiter2 = TokenBucketRateLimiter(capacity=20, refill_rate=20, name="ws")

        registry.register_limiter("rest", limiter1)
        registry.register_limiter("ws", limiter2)

        assert registry.get_limiter("rest") is limiter1
        assert registry.get_limiter("ws") is limiter2


class TestRateLimitedDecorator:
    """Tests for @rate_limited decorator."""

    @pytest.mark.asyncio
    async def test_decorator_acquires_tokens(self):
        """Test that decorator acquires tokens before method execution."""

        class TestClient:
            def __init__(self):
                self._rate_limiters = RateLimiterRegistry()
                limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")
                self._rate_limiters.register_limiter("rest", limiter)

            @rate_limited("rest", weight=5)
            async def fetch_data(self):
                return "data"

        client = TestClient()
        limiter = client._rate_limiters.get_limiter("rest")

        # Initial tokens
        assert limiter.get_available_tokens() == pytest.approx(10, rel=0.1)

        # Call decorated method
        result = await client.fetch_data()
        assert result == "data"

        # Tokens should be consumed
        assert limiter.get_available_tokens() == pytest.approx(5, rel=0.1)

    @pytest.mark.asyncio
    async def test_decorator_blocks_when_rate_limited(self):
        """Test that decorator blocks when rate limit is reached."""

        class TestClient:
            def __init__(self):
                self._rate_limiters = RateLimiterRegistry()
                limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")
                self._rate_limiters.register_limiter("rest", limiter)

            @rate_limited("rest", weight=10)
            async def fetch_data(self):
                return "data"

        client = TestClient()

        # First call should succeed immediately
        await client.fetch_data()

        # Second call should block until tokens refill
        start = time.monotonic()
        await client.fetch_data()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.9  # Should wait ~1 second

    @pytest.mark.asyncio
    async def test_decorator_without_rate_limiters_raises_error(self):
        """Test that decorator raises error if _rate_limiters missing."""

        class BadClient:
            @rate_limited("rest", weight=5)
            async def fetch_data(self):
                return "data"

        client = BadClient()

        with pytest.raises(AttributeError, match="_rate_limiters"):
            await client.fetch_data()

    @pytest.mark.asyncio
    async def test_decorator_with_unknown_limiter_raises_error(self):
        """Test that decorator raises error if limiter key not found."""

        class TestClient:
            def __init__(self):
                self._rate_limiters = RateLimiterRegistry()

            @rate_limited("unknown", weight=5)
            async def fetch_data(self):
                return "data"

        client = TestClient()

        with pytest.raises(KeyError, match="not found"):
            await client.fetch_data()

    @pytest.mark.asyncio
    async def test_decorator_with_multiple_limiters(self):
        """Test that decorator works with multiple limiters."""

        class TestClient:
            def __init__(self):
                self._rate_limiters = RateLimiterRegistry()
                # low refill so the post-acquire reads don't drift under parallel test load
                rest_limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1, name="rest")
                ws_limiter = TokenBucketRateLimiter(capacity=20, refill_rate=1, name="ws")
                self._rate_limiters.register_limiter("rest", rest_limiter)
                self._rate_limiters.register_limiter("ws", ws_limiter)

            @rate_limited("rest", weight=5)
            async def fetch_rest(self):
                return "rest_data"

            @rate_limited("ws", weight=10)
            async def fetch_ws(self):
                return "ws_data"

        client = TestClient()

        # Call both methods
        await client.fetch_rest()
        await client.fetch_ws()

        # Check tokens were consumed from correct limiters
        rest_limiter = client._rate_limiters.get_limiter("rest")
        ws_limiter = client._rate_limiters.get_limiter("ws")

        assert rest_limiter.get_available_tokens() == pytest.approx(5, rel=0.1)
        assert ws_limiter.get_available_tokens() == pytest.approx(10, rel=0.1)
