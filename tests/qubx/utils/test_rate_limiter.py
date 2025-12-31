"""Unit tests for rate limiter module."""

import asyncio
import time

import pytest

from qubx.utils.rate_limiter import RateLimiterRegistry, TokenBucketRateLimiter, rate_limited


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
        assert elapsed < 0.7  # But not too much

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketRateLimiter(capacity=100, refill_rate=100, name="test")

        # Consume tokens
        await limiter.acquire(50)
        assert limiter.get_available_tokens() == pytest.approx(50, rel=0.1)

        # Wait for refill (0.3 seconds = 30 tokens at 100/sec)
        await asyncio.sleep(0.3)

        # Should have ~80 tokens now (50 + 30)
        assert limiter.get_available_tokens() >= 75  # Allow slack
        assert limiter.get_available_tokens() <= 85

    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        """Test that tokens don't exceed capacity."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=100, name="test")

        # Wait for refill
        await asyncio.sleep(0.5)

        # Should be capped at capacity
        assert limiter.get_available_tokens() == pytest.approx(10, rel=0.1)

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity(self):
        """Test that acquiring more than capacity raises error."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="test")

        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await limiter.acquire(20)

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
                rest_limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10, name="rest")
                ws_limiter = TokenBucketRateLimiter(capacity=20, refill_rate=20, name="ws")
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
