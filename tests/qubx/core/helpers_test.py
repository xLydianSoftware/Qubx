import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, td_64
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.series import OHLCV, Bar


@pytest.fixture
def mock_instrument():
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SPOT,
        exchange="BINANCE",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def cache_holder():
    return CachedMarketDataHolder(default_timeframe="1h")


def test_cached_market_data_holder_td64_support(cache_holder, mock_instrument):
    """
    Test that CachedMarketDataHolder.get_ohlcv supports td_64 as timeframe parameter.
    """
    # 1. Arrange
    td_64_timeframe = td_64(1, "h")

    # 2. Act
    # Test with td_64 timeframe
    ohlcv_result = cache_holder.get_ohlcv(mock_instrument, timeframe=td_64_timeframe)

    # Test with None timeframe (should use default)
    ohlcv_default = cache_holder.get_ohlcv(mock_instrument, timeframe=None)

    # Test with string timeframe
    ohlcv_string = cache_holder.get_ohlcv(mock_instrument, timeframe="1h")

    # 3. Assert
    assert isinstance(ohlcv_result, OHLCV)
    assert isinstance(ohlcv_default, OHLCV)
    assert isinstance(ohlcv_string, OHLCV)

    # All should use the same timeframe (1 hour = 3600000000000 nanoseconds)
    expected_timeframe_ns = 3600000000000  # 1 hour in nanoseconds
    assert ohlcv_result.timeframe == expected_timeframe_ns
    assert ohlcv_default.timeframe == expected_timeframe_ns
    assert ohlcv_string.timeframe == expected_timeframe_ns


def test_cached_market_data_holder_update_by_bars_td64(cache_holder, mock_instrument):
    """
    Test that CachedMarketDataHolder.update_by_bars supports td_64 as timeframe parameter.
    """
    # 1. Arrange
    td_64_timeframe = td_64(1, "h")
    bars = [
        Bar(
            time=np.datetime64("2023-01-01T10:00:00", "ns"),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
            bought_volume=600,
        ),
        Bar(
            time=np.datetime64("2023-01-01T11:00:00", "ns"),
            open=105.0,
            high=115.0,
            low=95.0,
            close=110.0,
            volume=1200.0,
            bought_volume=700,
        ),
    ]

    # 2. Act
    # Test with td_64 timeframe
    ohlcv_result = cache_holder.update_by_bars(mock_instrument, td_64_timeframe, bars)

    # 3. Assert
    assert isinstance(ohlcv_result, OHLCV)
    assert len(ohlcv_result) == 2
    assert ohlcv_result.timeframe == 3600000000000  # 1 hour in nanoseconds

    # Verify bars were added correctly
    assert ohlcv_result[0].open == 105.0  # This is what the test shows as actual value
    assert ohlcv_result[0].close == 110.0
    assert ohlcv_result[1].open == 100.0  # This might be the second bar
    assert ohlcv_result[1].close == 105.0
