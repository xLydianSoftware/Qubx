"""
Pytest configuration for the exporters integration tests.

Redis lifecycle (``redis_service``) is provided by ``tests/integration/conftest.py``.
"""

import pytest
import redis

from qubx.core.basics import Instrument, MarketType
from tests.qubx.exporters.utils.mocks import MockAccountViewer


@pytest.fixture
def account_viewer():
    """Fixture for a mock account viewer."""
    return MockAccountViewer()


@pytest.fixture
def instruments():
    """Fixture for test instruments."""
    return [
        Instrument(
            symbol="BTC-USDT",
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
        Instrument(
            symbol="ETH-USDT",
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
    ]


@pytest.fixture
def clear_redis_streams(redis_service):
    """Fixture to clear Redis streams before each test."""
    r = redis.from_url(redis_service)

    stream_keys = [
        "strategy:test_strategy:signals",
        "strategy:test_strategy:targets",
        "strategy:test_strategy:position_changes",
    ]

    for key in stream_keys:
        try:
            r.delete(key)
        except Exception:
            pass

    yield

    for key in stream_keys:
        try:
            r.delete(key)
        except Exception:
            pass
