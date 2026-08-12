"""Tests for the rate limiting engine — quota pool behavior."""

import asyncio
import math
import time
from typing import Any

import pytest

from qubx import logger
from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, ExchangeRateLimiter, PoolConfig
from qubx.rate_limiting.engine import RateLimitGateTimeout
from qubx.rate_limiting.pools import QuotaPool


@pytest.fixture
def captured_logs():
    lines: list[tuple[str, str]] = []
    sink_id = logger.add(lambda m: lines.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    yield lines
    logger.remove(sink_id)


def _metric(metrics: list[dict[str, Any]], name: str, pool: str) -> float:
    values = [m["value"] for m in metrics if m["name"] == name and m["tags"]["pool"] == pool]
    assert len(values) == 1, f"expected exactly one {name} for pool {pool}, got {values}"
    return values[0]


def _make_config(
    quota_capacity: float = 1000,
    # cooldown must stay well above gate_max_wait, else a closed gate reopens inside the
    # wait and a test meant to observe the timeout passes for the wrong reason
    cooldown: float = 30.0,
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

        # the wait_for bound *is* the assertion: waiting out gate_max_wait surfaces as
        # TimeoutError instead of RateLimitGateTimeout
        with pytest.raises(RateLimitGateTimeout) as exc_info:
            await asyncio.wait_for(limiter.acquire("use_quota"), timeout=2.0)

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


class TestGateTimeoutMetric:
    @pytest.mark.asyncio
    async def test_gate_timeouts_metric_counts_a_forced_timeout(self):
        config = _make_config(gate_max_wait=0.05, cooldown=30.0)
        limiter = ExchangeRateLimiter("test", config)
        try:
            assert _metric(await limiter.collect_metrics(), "rate_limit.gate_timeouts", "rate_pool") == 0.0

            limiter.report_limit_hit(pool_name="rate_pool", retry_after=30.0, reason="test")
            with pytest.raises(RateLimitGateTimeout):
                await limiter.acquire("use_rate_only")

            metrics = await limiter.collect_metrics()
            assert _metric(metrics, "rate_limit.gate_timeouts", "rate_pool") == 1.0
            # negative control: the untouched pool's counter stays put
            assert _metric(metrics, "rate_limit.gate_timeouts", "quota_pool") == 0.0
        finally:
            limiter.reset_gates()


class TestRefillRatePrecondition:
    def test_rate_pool_without_refill_rate_is_rejected(self):
        """The bucket divides a deficit by refill_rate; 0 there is a hang, not a limit."""
        config = ExchangeRateLimitConfig(pools={"bad": PoolConfig("bad", "ip", capacity=100, refill_rate=0.0)})

        with pytest.raises(ValueError) as exc_info:
            ExchangeRateLimiter("test", config)

        assert "refill_rate" in str(exc_info.value)
        assert "'bad'" in str(exc_info.value)

    def test_negative_refill_rate_is_rejected(self):
        config = ExchangeRateLimitConfig(pools={"bad": PoolConfig("bad", "ip", capacity=100, refill_rate=-1.0)})

        with pytest.raises(ValueError, match="refill_rate"):
            ExchangeRateLimiter("test", config)

    def test_quota_pool_without_refill_rate_is_fine(self):
        """Negative control: quota pools are externally managed and legitimately have no refill."""
        config = ExchangeRateLimitConfig(
            pools={"q": PoolConfig("q", "address", capacity=100, refill_rate=0.0, pool_type="quota")}
        )

        limiter = ExchangeRateLimiter("test", config)
        assert limiter.get_quota_remaining("q") == 100


class TestClosedQuotaGateFailsFast:
    @pytest.mark.asyncio
    async def test_closed_quota_gate_does_not_wait_gate_max_wait(self):
        config = _make_config(gate_max_wait=5.0)
        limiter = ExchangeRateLimiter("test", config)

        limiter.report_limit_hit(pool_name="quota_pool", retry_after=1.0, reason="test")
        assert limiter._pools["quota_pool"]._gate_until == math.inf

        # as above: the wait_for bound is what pins "does not sit out gate_max_wait"
        with pytest.raises(RateLimitGateTimeout) as exc_info:
            await asyncio.wait_for(limiter.acquire("use_quota"), timeout=2.0)

        assert exc_info.value.pool_name == "quota_pool"

    @pytest.mark.asyncio
    async def test_infinite_gate_deadline_still_honours_gate_max_wait(self):
        """An `inf` deadline must not produce a nan/infinite sleep inside `_wait_for_gate`."""
        pool = QuotaPool(PoolConfig("q", "address", capacity=100, refill_rate=0.0, pool_type="quota"), "test", "s")
        pool.close_gate(1.0, "test")
        assert pool._gate_until == math.inf

        start = time.monotonic()
        with pytest.raises(RateLimitGateTimeout) as exc_info:
            await asyncio.wait_for(pool._wait_for_gate(0.3), timeout=5.0)

        assert time.monotonic() - start >= 0.25
        assert exc_info.value.pool_name == "q"
        assert pool.timeouts == 1


class TestZeroWeightStillRespectsTheGate:
    @pytest.mark.asyncio
    async def test_closed_gate_rejects_a_zero_weight_acquire(self):
        """A free call (ccxt costs Kraken orders 0) skips the bucket, never the gate — the gate is
        the venue telling us to stop, and cost 0 is a ccxt pricing quirk, not a free pass."""
        limiter = ExchangeRateLimiter("test", _make_config(gate_max_wait=0.05, cooldown=30.0))
        limiter.report_limit_hit(pool_name="rate_pool", retry_after=30.0, reason="test")
        try:
            with pytest.raises(RateLimitGateTimeout) as exc_info:
                await asyncio.wait_for(limiter.acquire("use_rate_only", weight_override=0), timeout=2.0)

            assert exc_info.value.pool_name == "rate_pool"
        finally:
            limiter.reset_gates()

    @pytest.mark.asyncio
    async def test_open_gate_admits_a_zero_weight_acquire_free(self):
        """Negative control: with the gate open the free call costs nothing and does not wait."""
        limiter = ExchangeRateLimiter("test", _make_config())

        await asyncio.wait_for(limiter.acquire("use_rate_only", weight_override=0), timeout=2.0)

        state = await limiter.get_pool_state("rate_pool")
        assert state is not None
        assert state["consumed"] == 0.0
        assert state["remaining"] == pytest.approx(100, abs=1.0)


class TestOversizedWeightClamp:
    @pytest.mark.asyncio
    async def test_weight_above_capacity_is_clamped_and_warns(self, captured_logs):
        """R3: an unclampable weight makes RedisBackend.acquire retry forever with no log."""
        limiter = ExchangeRateLimiter("test", _make_config())

        await asyncio.wait_for(limiter.acquire("use_rate_only", weight_override=1e6), timeout=2.0)

        state = await limiter.get_pool_state("rate_pool")
        assert state is not None
        assert state["consumed"] == 100.0
        assert [m for lvl, m in captured_logs if lvl == "WARNING" and "exceeds capacity" in m]

    @pytest.mark.asyncio
    async def test_weight_within_capacity_is_charged_verbatim(self, captured_logs):
        """Negative control: the clamp is conditional, not a blanket capacity charge."""
        limiter = ExchangeRateLimiter("test", _make_config())

        await asyncio.wait_for(limiter.acquire("use_rate_only", weight_override=40), timeout=2.0)

        state = await limiter.get_pool_state("rate_pool")
        assert state is not None
        assert state["consumed"] == 40.0
        assert not [m for _, m in captured_logs if "exceeds capacity" in m]
