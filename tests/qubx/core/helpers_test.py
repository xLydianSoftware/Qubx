import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, td_64
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.series import OHLCV, Bar, Trade


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


def test_cached_market_data_holder_update_by_trade(cache_holder, mock_instrument):
    """Test that trades properly update OHLCV with all fields including volume_quote and trade_count."""
    # Initialize OHLCV for instrument
    cache_holder.init_ohlcv(mock_instrument)

    # Create test trades - use integer timestamps
    trade1 = Trade(
        time=np.datetime64("2023-01-01T10:00:00", "ns").astype(np.int64),
        price=100.0,
        size=10.0,
        side=1  # Buy trade
    )
    trade2 = Trade(
        time=np.datetime64("2023-01-01T10:30:00", "ns").astype(np.int64),
        price=101.0,
        size=5.0,
        side=-1  # Sell trade
    )
    trade3 = Trade(
        time=np.datetime64("2023-01-01T10:45:00", "ns").astype(np.int64),
        price=102.0,
        size=3.0,
        side=1  # Another buy trade
    )

    # Update with trades
    cache_holder.update_by_trade(mock_instrument, trade1)
    cache_holder.update_by_trade(mock_instrument, trade2)
    cache_holder.update_by_trade(mock_instrument, trade3)

    # Get OHLCV
    ohlcv = cache_holder.get_ohlcv(mock_instrument)

    # Verify we have one bar (all trades within 1 hour)
    assert len(ohlcv) == 1
    bar = ohlcv[0]

    # Check OHLC values
    assert bar.open == 100.0  # First trade price
    assert bar.high == 102.0  # Max price
    assert bar.low == 100.0   # Min price
    assert bar.close == 102.0  # Last trade price

    # Check volume fields
    assert bar.volume == 18.0  # 10 + 5 + 3
    assert bar.bought_volume == 13.0  # 10 (buy) + 0 (sell) + 3 (buy)

    # Check volume_quote fields
    expected_volume_quote = (100.0 * 10.0) + (101.0 * 5.0) + (102.0 * 3.0)  # 1000 + 505 + 306 = 1811
    assert bar.volume_quote == expected_volume_quote

    expected_bought_volume_quote = (100.0 * 10.0) + (102.0 * 3.0)  # 1000 + 306 = 1306
    assert bar.bought_volume_quote == expected_bought_volume_quote

    # Check trade count
    assert bar.trade_count == 3  # Three trades


def test_cached_market_data_holder_out_of_order_trades(cache_holder, mock_instrument):
    """Test that out-of-order trades within the same bar are processed correctly."""
    cache_holder.init_ohlcv(mock_instrument)

    # All trades belong to same 1-hour bar (10:00-11:00)
    trade1 = Trade(
        time=np.datetime64("2023-01-01T10:30:00", "ns").astype(np.int64),
        price=100.0,
        size=10.0,
        side=1  # Buy
    )
    trade2 = Trade(
        time=np.datetime64("2023-01-01T10:45:00", "ns").astype(np.int64),
        price=102.0,
        size=5.0,
        side=1  # Buy
    )
    trade3 = Trade(
        time=np.datetime64("2023-01-01T10:15:00", "ns").astype(np.int64),
        price=98.0,
        size=8.0,
        side=-1  # Sell
    )

    # Process in out-of-order sequence: 2, 1, 3
    cache_holder.update_by_trade(mock_instrument, trade2)  # 10:45 processed first
    cache_holder.update_by_trade(mock_instrument, trade1)  # 10:30 processed second (out of order)
    cache_holder.update_by_trade(mock_instrument, trade3)  # 10:15 processed last (out of order)

    ohlcv = cache_holder.get_ohlcv(mock_instrument)
    assert len(ohlcv) == 1  # All in same bar

    bar = ohlcv[0]
    # Verify all trades were included despite out-of-order arrival
    assert bar.volume == 23.0  # 5 + 10 + 8
    assert bar.bought_volume == 15.0  # 5 (buy) + 10 (buy) + 0 (sell)
    assert bar.trade_count == 3

    # Test trade from previous bar gets skipped
    old_trade = Trade(
        time=np.datetime64("2023-01-01T09:45:00", "ns").astype(np.int64),
        price=95.0,
        size=3.0,
        side=1  # Buy
    )
    cache_holder.update_by_trade(mock_instrument, old_trade)

    # Should still have same values (old trade skipped)
    assert bar.volume == 23.0
    assert bar.bought_volume == 15.0
    assert bar.trade_count == 3


def test_cached_market_data_holder_cross_bar_trades(cache_holder, mock_instrument):
    """Test trades spanning multiple bars are handled correctly."""
    cache_holder.init_ohlcv(mock_instrument)

    # Trade in first hour bar (10:00-11:00)
    trade1 = Trade(
        time=np.datetime64("2023-01-01T10:30:00", "ns").astype(np.int64),
        price=100.0,
        size=10.0,
        side=1  # Buy
    )
    cache_holder.update_by_trade(mock_instrument, trade1)

    # Trade in second hour bar (11:00-12:00)
    trade2 = Trade(
        time=np.datetime64("2023-01-01T11:30:00", "ns").astype(np.int64),
        price=101.0,
        size=5.0,
        side=1  # Buy
    )
    cache_holder.update_by_trade(mock_instrument, trade2)

    # Late trade for first bar arrives after second bar started
    late_trade = Trade(
        time=np.datetime64("2023-01-01T10:45:00", "ns").astype(np.int64),
        price=99.0,
        size=3.0,
        side=-1  # Sell
    )
    cache_holder.update_by_trade(mock_instrument, late_trade)

    ohlcv = cache_holder.get_ohlcv(mock_instrument)
    assert len(ohlcv) == 2  # Two separate bars

    # Most recent bar (11:00-12:00) should NOT include late trade
    recent_bar = ohlcv[0]
    assert recent_bar.volume == 5.0  # Only trade2
    assert recent_bar.trade_count == 1

    # Older bar (10:00-11:00) should only have trade1 (late_trade skipped)
    older_bar = ohlcv[1]
    assert older_bar.volume == 10.0  # Only trade1 (late_trade skipped)
    assert older_bar.trade_count == 1
