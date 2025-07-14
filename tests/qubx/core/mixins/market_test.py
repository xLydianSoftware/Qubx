from unittest.mock import MagicMock

import pandas as pd
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, td_64
from qubx.core.interfaces import IDataProvider
from qubx.core.mixins.market import MarketManager
from qubx.core.series import OHLCV


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
def mock_time_provider():
    mock = MagicMock()
    mock.time.return_value = pd.Timestamp("2023-01-01 12:50:00").asm8
    return mock


@pytest.fixture
def market_manager(mock_time_provider):
    # For simplicity, we mock dependencies. You might want to create more realistic test fixtures.
    cache_mock = MagicMock()
    data_providers_mock = [MagicMock(spec=IDataProvider)]
    data_providers_mock[0].exchange.return_value = "BINANCE"
    universe_manager_mock = MagicMock()

    manager = MarketManager(
        time_provider=mock_time_provider,
        cache=cache_mock,
        data_providers=data_providers_mock,  # type: ignore
        universe_manager=universe_manager_mock,
    )
    return manager


def test_ohlc_pd_consolidation(market_manager, mock_instrument, mock_time_provider):
    """
    Test that ohlc_pd with consolidated=True returns only finished bars.
    """
    # 1. Arrange
    # Create some OHLC data, where the last bar is not finished
    ohlc_data = [
        # older bars
        (pd.Timestamp("2023-01-01 11:00:00"), 100, 110, 90, 105, 1000),
        (pd.Timestamp("2023-01-01 12:00:00"), 105, 115, 95, 110, 1200),  # last bar, not finished
    ]
    ohlc_df = pd.DataFrame(ohlc_data, columns=["time", "open", "high", "low", "close", "volume"])
    ohlc_df.set_index("time", inplace=True)

    # - current time is 12:50, so last 1h bar is not finished
    mock_time_provider.time.return_value = pd.Timestamp("2023-01-01 12:50:00").asm8

    # - mock the 'ohlc().pd()' chain
    market_manager.ohlc = MagicMock()
    market_manager.ohlc.return_value.pd.return_value = ohlc_df

    # 2. Act
    # - when consolidated is True (default)
    consolidated_df = market_manager.ohlc_pd(mock_instrument, timeframe="1h")

    # - when consolidated is False
    full_df = market_manager.ohlc_pd(mock_instrument, timeframe="1h", consolidated=False)

    # 3. Assert
    # - should return only the first bar
    assert len(consolidated_df) == 1
    assert consolidated_df.index[0] == pd.Timestamp("2023-01-01 11:00:00")

    # - should return all bars
    assert len(full_df) == 2


def test_ohlc_pd_length(market_manager, mock_instrument):
    """
    Test that ohlc_pd with 'length' parameter returns correct number of bars.
    """
    # 1. Arrange
    # Create a real OHLCV object with test data
    ohlc_series = OHLCV("test", "1h")
    for h in range(10):
        timestamp = pd.Timestamp(f"2023-01-01 {h:02d}:00:00").as_unit("ns").asm8.item()
        ohlc_series.update_by_bar(timestamp, 100, 110, 90, 105, 1000, 0)

    # - mock only the 'ohlc' method to return the real OHLCV object
    market_manager.ohlc = MagicMock()
    market_manager.ohlc.return_value = ohlc_series

    # 2. Act
    df = market_manager.ohlc_pd(mock_instrument, timeframe="1h", length=5, consolidated=False)

    # 3. Assert
    assert len(df) == 5
    assert df.index[0] == pd.Timestamp("2023-01-01 05:00:00")


def test_ohlc_td64_timeframe_support(market_manager, mock_instrument):
    """
    Test that ohlc and ohlc_pd methods support td_64 as timeframe parameter.
    """
    # 1. Arrange
    ohlc_data = [(pd.Timestamp(f"2023-01-01 {h:02d}:00:00"), 100, 110, 90, 105, 1000) for h in range(10)]
    ohlc_df = pd.DataFrame(ohlc_data, columns=["time", "open", "high", "low", "close", "volume"])
    ohlc_df.set_index("time", inplace=True)

    # Create a td_64 timeframe (1 hour)
    td_64_timeframe = td_64(1, "h")

    # Mock the OHLCV object
    mock_ohlcv = MagicMock(spec=OHLCV)
    mock_ohlcv.pd.return_value = ohlc_df

    # Mock the _cache.get_ohlcv method to return our mock OHLCV
    market_manager._cache.get_ohlcv.return_value = mock_ohlcv

    # Mock the timedelta_to_str function to return a string representation
    from qubx.core.mixins.market import timedelta_to_str

    market_manager._cache.default_timeframe = td_64(1, "h")

    # 2. Act
    # Test ohlc method with td_64
    ohlc_result = market_manager.ohlc(mock_instrument, timeframe=td_64_timeframe)

    # Test ohlc_pd method with td_64
    ohlc_pd_result = market_manager.ohlc_pd(mock_instrument, timeframe=td_64_timeframe, consolidated=False)

    # 3. Assert
    # Verify that _cache.get_ohlcv was called with the string representation of the timeframe
    market_manager._cache.get_ohlcv.assert_called()
    call_args = market_manager._cache.get_ohlcv.call_args
    assert call_args[0][0] == mock_instrument  # instrument argument
    assert isinstance(call_args[0][1], str)  # timeframe should be converted to string

    # Verify that the methods returned the expected results
    assert ohlc_result == mock_ohlcv
    assert ohlc_pd_result.equals(ohlc_df)

    # Verify pd method was called on the OHLCV object
    mock_ohlcv.pd.assert_called()
