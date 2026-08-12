"""Tests for RateLimitManager and EgressIPResolver."""

import asyncio

import pytest

from qubx import logger
from qubx.connectors.registry import ConnectorRegistry
from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, PoolConfig
from qubx.rate_limiting.engine import _UNRESOLVED_SCOPE_ID
from qubx.rate_limiting.ip_resolver import EgressIPResolver
from qubx.rate_limiting.manager import RateLimitManager
from qubx.utils.runner.configs import RateLimitingConfig

_EGRESS_IP = "203.0.113.7"  # literal, so no IP discovery (and no network I/O) happens


def _rl_config() -> ExchangeRateLimitConfig:
    return ExchangeRateLimitConfig(
        pools={
            "rest": PoolConfig("rest", "ip", capacity=100, refill_rate=50.0),
            "orders": PoolConfig("orders", "account", capacity=10, refill_rate=5.0),
        },
        endpoint_map={"r": EndpointCosts([("rest", 1)])},
        default_costs=EndpointCosts([("rest", 1)]),
    )


@pytest.fixture
def captured_logs():
    lines: list[tuple[str, str]] = []
    sink_id = logger.add(lambda m: lines.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    yield lines
    logger.remove(sink_id)


@pytest.fixture
def registered_config(monkeypatch):
    monkeypatch.setattr(ConnectorRegistry, "get_rate_limit_config", lambda name, exchange_name: _rl_config())


def _limiter_for(api_key: str | None, exchange: str = "venue"):
    loop = asyncio.new_event_loop()
    try:
        manager = RateLimitManager(RateLimitingConfig(egress_ip=_EGRESS_IP), loop)
        return manager.get_or_create(exchange, "ccxt", api_key=api_key)
    finally:
        loop.close()


class TestDisabledWarning:
    def test_disabled_manager_warns_once(self, captured_logs):
        loop = asyncio.new_event_loop()
        try:
            manager = RateLimitManager(None, loop)
            assert not manager.is_enabled
        finally:
            loop.close()

        warnings = [m for lvl, m in captured_logs if lvl == "WARNING" and "disabled" in m.lower()]
        assert len(warnings) == 1, warnings

    def test_enabled_manager_does_not_warn(self, captured_logs):
        """Negative control: the warning is about the missing section, not about construction."""
        loop = asyncio.new_event_loop()
        try:
            manager = RateLimitManager(RateLimitingConfig(egress_ip=_EGRESS_IP), loop)
            assert manager.is_enabled
        finally:
            loop.close()

        assert not [m for lvl, m in captured_logs if lvl == "WARNING" and "disabled" in m.lower()]


class TestAccountScoping:
    def test_different_api_keys_get_different_account_pool_keys(self, registered_config):
        """R2: without this, every bot on a shared backend shares one `orders` bucket."""
        a = _limiter_for("KEY-A")
        b = _limiter_for("KEY-B")
        a_again = _limiter_for("KEY-A")
        assert a is not None and b is not None and a_again is not None

        key_a = a._pools["orders"]._key
        key_b = b._pools["orders"]._key
        assert key_a != key_b
        assert key_a == a_again._pools["orders"]._key, "same api_key must resolve to the same bucket"
        assert "KEY-A" not in key_a, "raw api key must never reach a bucket key"

    def test_ip_scoped_pool_is_unaffected_by_the_api_key(self, registered_config):
        """Negative control: only the account scope moves — the IP scope stays shared."""
        a = _limiter_for("KEY-A")
        b = _limiter_for("KEY-B")
        assert a is not None and b is not None
        assert a._pools["rest"]._key == b._pools["rest"]._key

    def test_missing_api_key_falls_back_to_a_per_process_id(self, registered_config):
        """R2 fallback: unresolvable is not evidence of shared — never the literal `local`."""
        limiter = _limiter_for(None)
        assert limiter is not None

        key = limiter._pools["orders"]._key
        assert not key.endswith(":local")
        assert key.endswith(_UNRESOLVED_SCOPE_ID)
        assert _UNRESOLVED_SCOPE_ID.startswith("local_")


class TestEgressIPResolver:
    @pytest.mark.asyncio
    async def test_start_skips_discovery_when_already_primed(self, monkeypatch):
        resolver = EgressIPResolver(check_interval=3600, initial_ip="1.2.3.4")
        calls: list[int] = []

        async def fake_discover():
            calls.append(1)
            return "5.6.7.8"

        monkeypatch.setattr(resolver, "discover", fake_discover)
        try:
            await resolver.start()
            assert calls == []
            assert resolver.current_ip == "1.2.3.4"
        finally:
            resolver.stop()
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_start_discovers_when_not_primed(self, monkeypatch):
        """Negative control: the skip is conditional, not a dead discovery path."""
        resolver = EgressIPResolver(check_interval=3600)
        calls: list[int] = []

        async def fake_discover():
            calls.append(1)
            return "5.6.7.8"

        monkeypatch.setattr(resolver, "discover", fake_discover)
        try:
            await resolver.start()
            assert len(calls) == 1
            assert resolver.current_ip == "5.6.7.8"
        finally:
            resolver.stop()
            await asyncio.sleep(0)
