from unittest.mock import MagicMock

import pandas as pd
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType
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
    ohlc_data = [(pd.Timestamp(f"2023-01-01 {h:02d}:00:00"), 100, 110, 90, 105, 1000) for h in range(10)]
    ohlc_df = pd.DataFrame(ohlc_data, columns=["time", "open", "high", "low", "close", "volume"])
    ohlc_df.set_index("time", inplace=True)

    # - mock the 'ohlc().pd()' chain
    market_manager.ohlc = MagicMock()
    market_manager.ohlc.return_value.pd.return_value = ohlc_df

    # 2. Act
    df = market_manager.ohlc_pd(mock_instrument, timeframe="1h", length=5, consolidated=False)

    # 3. Assert
    assert len(df) == 5
    assert df.index[0] == pd.Timestamp("2023-01-01 05:00:00")
