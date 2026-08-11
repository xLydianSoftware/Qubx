import pytest
from pydantic import ValidationError

from qubx.utils.runner.configs import LiveConfig, MarketCacheConfig


def _live(**over):
    base = dict(exchanges={}, logging={"logger": "InMemoryLogsWriter"})
    base.update(over)
    return LiveConfig(**base)


def test_defaults_reproduce_current_behavior():
    cfg = _live()
    assert cfg.market_cache.default_length == 10_000
    assert cfg.market_cache.per_type == {}


def test_per_type_parses():
    cfg = _live(market_cache={"per_type": {"orderbook": 4, "funding_rate": 64}})
    assert cfg.market_cache.per_type == {"orderbook": 4, "funding_rate": 64}
    assert cfg.market_cache.default_length == 10_000


def test_zero_cap_rejected():
    with pytest.raises(ValidationError):
        MarketCacheConfig(per_type={"orderbook": 0})
    with pytest.raises(ValidationError):
        MarketCacheConfig(default_length=0)
