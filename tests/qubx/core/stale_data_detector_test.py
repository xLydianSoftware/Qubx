"""
Unit tests for the StaleDataDetector class.
"""

import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, dt_64, td_64
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import ITimeProvider
from qubx.core.series import Bar
from qubx.core.stale_data_detector import StaleDataDetector


class MockTimeProvider(ITimeProvider):
    """Mock time provider for testing."""

    def __init__(self, current_time: dt_64):
        self._current_time = current_time

    def time(self) -> dt_64:
        return self._current_time

    def set_time(self, time: dt_64):
        self._current_time = time


@pytest.fixture
def mock_time_provider():
    """Create a mock time provider."""
    return MockTimeProvider(dt_64(0, "ns"))


@pytest.fixture
def cache():
    """Create a cached market data holder."""
    return CachedMarketDataHolder(default_timeframe="1m")


@pytest.fixture
def instrument():
    """Create a test instrument."""
    return Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SWAP,
        exchange="binance",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


@pytest.fixture
def detector(cache, mock_time_provider):
    """Create a stale data detector."""
    return StaleDataDetector(
        cache=cache,
        time_provider=mock_time_provider,
        detection_period=td_64(2, "h"),
        check_interval=td_64(10, "m"),
        min_bars_required=3,
    )


def create_flat_bars(start_time: int, count: int, price: float, timeframe_ns: int = 60_000_000_000) -> list[Bar]:
    """Create a list of flat bars (open==high==low==close) with the same price."""
    bars = []
    for i in range(count):
        time = start_time + i * timeframe_ns
        bars.append(Bar(time, price, price, price, price, volume=100.0, bought_volume=50.0))
    return bars


def create_moving_bars(start_time: int, count: int, base_price: float, timeframe_ns: int = 60_000_000_000) -> list[Bar]:
    """Create a list of bars with price movement."""
    bars = []
    for i in range(count):
        time = start_time + i * timeframe_ns
        price = base_price + i * 0.01  # Price increases by 0.01 each bar
        bars.append(Bar(time, price, price + 0.005, price - 0.005, price, volume=100.0, bought_volume=50.0))
    return bars


class TestStaleDataDetector:
    """Test cases for StaleDataDetector."""

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector._detection_period == td_64(2, "h")
        assert detector._check_interval == td_64(10, "m")
        assert detector._min_bars_required == 3
        assert len(detector._last_check_time) == 0

    def test_should_check_instrument_never_checked(self, detector, instrument):
        """Test should_check_instrument for instrument never checked before."""
        assert detector.should_check_instrument(instrument) is True

    def test_should_check_instrument_too_soon(self, detector, instrument, mock_time_provider):
        """Test should_check_instrument when last check was too recent."""
        # Set initial time and check
        mock_time_provider.set_time(dt_64(0, "ns"))
        detector._last_check_time[instrument] = dt_64(0, "ns")

        # Move time forward by only 5 minutes (less than check interval of 10 minutes)
        mock_time_provider.set_time(dt_64(5 * 60 * 1_000_000_000, "ns"))
        result = detector.should_check_instrument(instrument)
        assert not result

    def test_should_check_instrument_enough_time_passed(self, detector, instrument, mock_time_provider):
        """Test should_check_instrument when enough time has passed."""
        # Set initial time and check
        mock_time_provider.set_time(dt_64(0, "ns"))
        detector._last_check_time[instrument] = dt_64(0, "ns")

        # Move time forward by 15 minutes (more than check interval)
        mock_time_provider.set_time(dt_64(15 * 60 * 1_000_000_000, "ns"))
        assert detector.should_check_instrument(instrument)

    def test_is_instrument_stale_insufficient_data(self, detector, instrument, cache):
        """Test is_instrument_stale with insufficient data."""
        # Add only 2 bars (less than min_bars_required=3)
        cache.init_ohlcv(instrument)
        bars = create_flat_bars(0, 2, 100.0)
        cache.update_by_bars(instrument, "1m", bars)

        assert detector.is_instrument_stale(instrument) is False

    def test_is_instrument_stale_with_price_movement_within_bar(self, detector, instrument, cache):
        """Test is_instrument_stale with price movement within bars."""
        cache.init_ohlcv(instrument)

        # Create bars with intra-bar price movement
        bars = []
        for i in range(5):
            time = i * 60_000_000_000  # 1 minute intervals
            bars.append(Bar(time, 100.0, 100.1, 99.9, 100.0, volume=100.0, bought_volume=50.0))  # high != low

        cache.update_by_bars(instrument, "1m", bars)
        assert detector.is_instrument_stale(instrument) is False

    def test_is_instrument_stale_with_price_movement_between_bars(self, detector, instrument, cache):
        """Test is_instrument_stale with price movement between bars."""
        cache.init_ohlcv(instrument)

        # Create flat bars but with different prices between bars
        bars = []
        for i in range(5):
            time = i * 60_000_000_000  # 1 minute intervals
            price = 100.0 + i * 0.01  # Different price for each bar
            bars.append(Bar(time, price, price, price, price, volume=100.0, bought_volume=50.0))

        cache.update_by_bars(instrument, "1m", bars)
        assert detector.is_instrument_stale(instrument) is False

    def test_is_instrument_stale_truly_stale(self, detector, instrument, cache):
        """Test is_instrument_stale with truly stale data."""
        cache.init_ohlcv(instrument)

        # Create completely flat bars with same price
        bars = create_flat_bars(0, 150, 100.0)  # 150 bars of 1 minute = 2.5 hours
        cache.update_by_bars(instrument, "1m", bars)

        assert detector.is_instrument_stale(instrument) is True

    def test_is_instrument_stale_mixed_data(self, detector, instrument, cache):
        """Test is_instrument_stale with mixed stale and moving data."""
        cache.init_ohlcv(instrument)

        # Create bars: first 100 are flat, then 50 have movement
        flat_bars = create_flat_bars(0, 100, 100.0)
        moving_bars = create_moving_bars(100 * 60_000_000_000, 50, 100.0)

        all_bars = flat_bars + moving_bars
        cache.update_by_bars(instrument, "1m", all_bars)

        # Since recent bars have movement, it should not be stale
        assert detector.is_instrument_stale(instrument) is False

    def test_detect_stale_instruments_empty_list(self, detector):
        """Test detect_stale_instruments with empty instrument list."""
        result = detector.detect_stale_instruments([])
        assert result == []

    def test_detect_stale_instruments_no_stale_data(self, detector, instrument, cache):
        """Test detect_stale_instruments with no stale instruments."""
        cache.init_ohlcv(instrument)

        # Create bars with price movement
        bars = create_moving_bars(0, 150, 100.0)
        cache.update_by_bars(instrument, "1m", bars)

        result = detector.detect_stale_instruments([instrument])
        assert result == []

    def test_detect_stale_instruments_with_stale_data(self, detector, instrument, cache):
        """Test detect_stale_instruments with stale instruments."""
        cache.init_ohlcv(instrument)

        # Create stale bars
        bars = create_flat_bars(0, 150, 100.0)
        cache.update_by_bars(instrument, "1m", bars)

        result = detector.detect_stale_instruments([instrument])
        assert result == [instrument]

    def test_detect_stale_instruments_respects_check_interval(self, detector, instrument, cache, mock_time_provider):
        """Test that detect_stale_instruments respects check interval."""
        cache.init_ohlcv(instrument)
        bars = create_flat_bars(0, 150, 100.0)
        cache.update_by_bars(instrument, "1m", bars)

        # First check
        mock_time_provider.set_time(dt_64(0, "ns"))
        result1 = detector.detect_stale_instruments([instrument])
        assert result1 == [instrument]

        # Second check too soon (only 5 minutes later)
        mock_time_provider.set_time(dt_64(5 * 60 * 1_000_000_000, "ns"))
        result2 = detector.detect_stale_instruments([instrument])
        assert result2 == []  # Should not check again

        # Third check after enough time (15 minutes later)
        mock_time_provider.set_time(dt_64(15 * 60 * 1_000_000_000, "ns"))
        result3 = detector.detect_stale_instruments([instrument])
        assert result3 == [instrument]  # Should check again

    def test_reset_check_time(self, detector, instrument, mock_time_provider):
        """Test reset_check_time method."""
        # Set a check time
        mock_time_provider.set_time(dt_64(0, "ns"))
        detector._last_check_time[instrument] = dt_64(0, "ns")

        # Reset it
        detector.reset_check_time(instrument)
        assert instrument not in detector._last_check_time

    def test_get_last_check_time(self, detector, instrument, mock_time_provider):
        """Test get_last_check_time method."""
        # Initially no check time
        assert detector.get_last_check_time(instrument) is None

        # Set a check time
        check_time = dt_64(12345, "ns")
        detector._last_check_time[instrument] = check_time

        # Get it back
        assert detector.get_last_check_time(instrument) == check_time

    def test_is_instrument_stale_exception_handling(self, detector, instrument, cache):
        """Test that is_instrument_stale handles exceptions gracefully."""
        # Don't initialize OHLCV to cause an exception
        # This should not crash but return False
        assert detector.is_instrument_stale(instrument) is False

    def test_different_detection_periods(self, cache, mock_time_provider, instrument):
        """Test detector with different detection periods."""
        # Create detector with 1-hour detection period
        detector = StaleDataDetector(
            cache=cache,
            time_provider=mock_time_provider,
            detection_period=td_64(1, "h"),
            check_interval=td_64(5, "m"),
            min_bars_required=3,
        )

        cache.init_ohlcv(instrument)

        # Create 30 minutes of stale data (should not be stale for 1-hour period)
        bars = create_flat_bars(0, 30, 100.0)
        cache.update_by_bars(instrument, "1m", bars)

        assert detector.is_instrument_stale(instrument) is False

        # Create 90 minutes of stale data (should be stale for 1-hour period)
        bars = create_flat_bars(0, 90, 100.0)
        cache.update_by_bars(instrument, "1m", bars)

        assert detector.is_instrument_stale(instrument) is True

    def test_different_timeframes(self, cache, mock_time_provider, instrument):
        """Test detector with different timeframes."""
        # Use 5-minute timeframe
        cache.update_default_timeframe("5m")

        detector = StaleDataDetector(
            cache=cache,
            time_provider=mock_time_provider,
            detection_period=td_64(2, "h"),
            check_interval=td_64(10, "m"),
            min_bars_required=3,
        )

        cache.init_ohlcv(instrument)

        # Create 2.5 hours of stale data in 5-minute bars (30 bars)
        bars = create_flat_bars(0, 30, 100.0, timeframe_ns=5 * 60_000_000_000)
        cache.update_by_bars(instrument, "5m", bars)

        assert detector.is_instrument_stale(instrument) is True

    def test_incremental_analysis_caching(self, detector, instrument, cache):
        """Test that incremental analysis maintains state correctly between calls."""
        cache.init_ohlcv(instrument)
        
        # Create initial stale bars
        bars = create_flat_bars(0, 50, 100.0)
        cache.update_by_bars(instrument, "1m", bars)
        
        # First call should analyze all bars and create state
        result1 = detector.is_instrument_stale(instrument)
        state = detector._instrument_states[instrument]
        assert state.last_checked_bar_time is not None
        assert state.last_analysis_time is not None
        initial_analysis_time = state.last_analysis_time
        
        # Second call with same data should reuse cached analysis
        result2 = detector.is_instrument_stale(instrument)
        assert result1 == result2  # Same result
        
        # Add more bars and verify analysis is updated
        more_bars = create_flat_bars(50 * 60_000_000_000, 10, 100.0)
        cache.update_by_bars(instrument, "1m", more_bars)
        
        detector.is_instrument_stale(instrument)
        # Analysis time should be updated when new data is processed
        assert state.last_analysis_time >= initial_analysis_time

    def test_cache_invalidation_on_data_change(self, detector, instrument, cache):
        """Test that cache is properly invalidated when data changes significantly."""
        cache.init_ohlcv(instrument)
        
        # Create initial stale data (enough bars to meet detection period)
        bars = create_flat_bars(0, 150, 100.0)  # 150 minutes > 2 hour detection period
        cache.update_by_bars(instrument, "1m", bars)
        
        # First analysis should detect stale data
        result1 = detector.is_instrument_stale(instrument)
        assert result1 is True  # Should be stale
        
        # Simulate data replacement (cache gets reset)
        cache.init_ohlcv(instrument)  # This effectively resets the data
        new_bars = create_moving_bars(0, 150, 200.0)  # Enough bars with movement
        cache.update_by_bars(instrument, "1m", new_bars)
        
        # Next analysis should detect the change and return non-stale
        result2 = detector.is_instrument_stale(instrument)
        assert result2 is False  # Should not be stale with moving data

    def test_state_reset_functionality(self, detector, instrument, cache):
        """Test that reset_check_time properly clears cached state."""
        cache.init_ohlcv(instrument)
        bars = create_flat_bars(0, 20, 100.0)
        cache.update_by_bars(instrument, "1m", bars)
        
        # Analyze to create state
        detector.is_instrument_stale(instrument)
        assert instrument in detector._instrument_states
        state = detector._instrument_states[instrument]
        assert state.last_analysis_time is not None
        assert state.last_checked_bar_time is not None
        
        # Reset should clear state
        detector.reset_check_time(instrument)
        assert state.last_checked_bar_time is None
        assert state.last_analysis_time is None
        assert state.consecutive_stale_duration == 0

    def test_incremental_analysis_with_mixed_new_data(self, detector, instrument, cache):
        """Test incremental analysis when adding both stale and moving bars."""
        cache.init_ohlcv(instrument)
        
        # Start with some stale bars
        stale_bars = create_flat_bars(0, 30, 100.0)
        cache.update_by_bars(instrument, "1m", stale_bars)
        
        detector.is_instrument_stale(instrument)
        state = detector._instrument_states[instrument]
        initial_check_time = state.last_checked_bar_time
        
        # Add moving bars (should make it non-stale)
        moving_bars = create_moving_bars(30 * 60_000_000_000, 10, 100.0)
        cache.update_by_bars(instrument, "1m", moving_bars)
        
        result2 = detector.is_instrument_stale(instrument)
        # Should now be non-stale due to the moving bars
        assert result2 is False
        # State should be updated with new bars processed
        assert state.last_checked_bar_time > initial_check_time
