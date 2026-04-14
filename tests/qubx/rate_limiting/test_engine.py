"""Tests for the rate limiting engine — quota pool behavior."""

import asyncio

import pytest

from qubx.rate_limiting import ExchangeRateLimiter, ExchangeRateLimitConfig, PoolConfig, EndpointCosts
from qubx.rate_limiting.engine import RateLimitGateTimeout


def _make_config(
    quota_capacity: float = 1000,
    cooldown: float = 0.5,
    gate_max_wait: float = 1.0,
) -> ExchangeRateLimitConfig:
    """Create a minimal config with one rate pool and one quota pool."""
    return ExchangeRateLimitConfig(
        pools={
            "rate_pool": PoolConfig(
                name="rate_pool",
                scope="ip",
                capacity=100,
                refill_rate=10.0,
                pool_type="rate",
                cooldown=cooldown,
            ),
            "quota_pool": PoolConfig(
                name="quota_pool",
                scope="address",
                capacity=quota_capacity,
                refill_rate=0,
                pool_type="quota",
                cooldown=cooldown,
            ),
        },
        endpoint_map={
            "use_quota": EndpointCosts([("rate_pool", 1), ("quota_pool", 1)]),
            "use_rate_only": EndpointCosts([("rate_pool", 1)]),
        },
        default_costs=EndpointCosts([]),
        gate_max_wait=gate_max_wait,
    )


class TestSyncQuotaReopensGate:
    @pytest.mark.asyncio
    async def test_sync_quota_reopens_gate_on_positive_remaining(self):
        config = _make_config()
        limiter = ExchangeRateLimiter("test", config)

        # Deplete quota → gate closes
        limiter.sync_from_exchange("quota_pool", remaining=0)
        assert limiter.is_gate_closed("quota_pool")

        # Sync with positive remaining → gate reopens
        limiter.sync_from_exchange("quota_pool", remaining=50)
        assert not limiter.is_gate_closed("quota_pool")

    @pytest.mark.asyncio
    async def test_sync_quota_closes_gate_on_zero(self):
        config = _make_config()
        limiter = ExchangeRateLimiter("test", config)

        limiter.sync_from_exchange("quota_pool", remaining=0)
        assert limiter.is_gate_closed("quota_pool")

    @pytest.mark.asyncio
    async def test_sync_quota_updates_capacity_when_remaining_exceeds_it(self):
        config = _make_config(quota_capacity=1000)
        limiter = ExchangeRateLimiter("test", config)

        # Remaining exceeds initial capacity (account earned more quota)
        limiter.sync_from_exchange("quota_pool", remaining=5000)
        assert config.pools["quota_pool"].capacity == 5000

        # Verify utilization is sane (not negative)
        state = await limiter.get_pool_state("quota_pool")
        assert state is not None
        assert 0 <= state["utilization"] <= 1.0


class TestQuotaPoolNoTimedGateReopen:
    @pytest.mark.asyncio
    async def test_quota_pool_gate_stays_closed(self):
        """Quota pool gate should NOT reopen on a timer — only via sync_from_exchange."""
        config = _make_config(cooldown=0.1)
        limiter = ExchangeRateLimiter("test", config)

        limiter.sync_from_exchange("quota_pool", remaining=0)
        assert limiter.is_gate_closed("quota_pool")

        # Wait longer than cooldown — gate should still be closed
        await asyncio.sleep(0.3)
        assert limiter.is_gate_closed("quota_pool")

    @pytest.mark.asyncio
    async def test_rate_pool_timed_gate_reopen(self):
        """Rate pools should still get timer-based reopen (no regression)."""
        config = _make_config(cooldown=0.2)
        limiter = ExchangeRateLimiter("test", config)

        limiter.report_limit_hit(pool_name="rate_pool", retry_after=0.2, reason="test")
        assert limiter.is_gate_closed("rate_pool")

        await asyncio.sleep(0.3)
        assert not limiter.is_gate_closed("rate_pool")


class TestQuotaAcquireFailsFast:
    @pytest.mark.asyncio
    async def test_quota_acquire_raises_immediately_when_depleted(self):
        """Depleted quota pool should raise immediately, not wait gate_max_wait."""
        config = _make_config(gate_max_wait=5.0)
        limiter = ExchangeRateLimiter("test", config)

        limiter.sync_from_exchange("quota_pool", remaining=0)

        t0 = asyncio.get_event_loop().time()
        with pytest.raises(RateLimitGateTimeout) as exc_info:
            await limiter.acquire("use_quota")
        elapsed = asyncio.get_event_loop().time() - t0

        # Should be near-instant, not 5s
        assert elapsed < 1.0
        assert exc_info.value.pool_name == "quota_pool"


class TestGateTimeoutHasPoolName:
    @pytest.mark.asyncio
    async def test_gate_timeout_has_pool_name(self):
        config = _make_config(gate_max_wait=0.1, cooldown=5.0)
        limiter = ExchangeRateLimiter("test", config)

        # Close the rate pool gate with long cooldown
        limiter.report_limit_hit(pool_name="rate_pool", retry_after=5.0, reason="test")

        with pytest.raises(RateLimitGateTimeout) as exc_info:
            await limiter.acquire("use_rate_only")

        assert exc_info.value.pool_name == "rate_pool"

    @pytest.mark.asyncio
    async def test_quota_timeout_has_pool_name(self):
        config = _make_config()
        limiter = ExchangeRateLimiter("test", config)
        limiter.sync_from_exchange("quota_pool", remaining=0)

        with pytest.raises(RateLimitGateTimeout) as exc_info:
            await limiter.acquire("use_quota")

        assert exc_info.value.pool_name == "quota_pool"


class TestSyncFromExchangeReopensQuotaGate:
    @pytest.mark.asyncio
    async def test_sync_from_exchange_reopens_quota_gate(self):
        config = _make_config()
        limiter = ExchangeRateLimiter("test", config)

        limiter.sync_from_exchange("quota_pool", remaining=0)
        assert limiter.is_gate_closed("quota_pool")

        # sync_from_exchange with positive remaining reopens
        limiter.sync_from_exchange("quota_pool", remaining=100)
        assert not limiter.is_gate_closed("quota_pool")
