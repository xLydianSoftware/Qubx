"""
Tests for the time_finders module.

This file contains tests for the TimeFinder class and its methods.
"""

import numpy as np
import pytest

from qubx.core.basics import (
    AssetBalance,
    Instrument,
    MarketType,
    Position,
    RestoredState,
    Signal,
    TargetPosition,
)
from qubx.restarts.time_finders import TimeFinder


# Test fixtures
@pytest.fixture
def sample_time():
    """Return a sample time for testing."""
    return np.datetime64("2023-01-01T12:00:00")


@pytest.fixture
def sample_instrument():
    """Return a sample instrument for testing."""
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.FUTURE,
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
def sample_signal(sample_instrument):
    """Return a sample signal for testing."""
    return Signal(
        time="2023-01-01T11:00:00",
        instrument=sample_instrument,
        signal=1.0,
        price=50000.0,
    )


@pytest.fixture
def sample_target_position(sample_signal):
    """Return a sample target position for testing."""
    return TargetPosition(
        time=np.datetime64("2023-01-01T11:00:00"),
        instrument=sample_signal.instrument,
        target_position_size=1.0,
    )


@pytest.fixture
def sample_position(sample_instrument):
    """Return a sample position for testing."""
    return Position(
        instrument=sample_instrument,
        quantity=1.0,
        pos_average_price=50000.0,
    )


@pytest.fixture
def empty_state(sample_time):
    """Return an empty restored state for testing."""
    return RestoredState(
        time=sample_time,
        balances=[],
        instrument_to_target_positions={},
        instrument_to_signal_positions={},
        positions={},
    )


@pytest.fixture
def state_with_nonzero_positions(sample_time, sample_instrument, sample_signal):
    """Return a restored state with nonzero positions for testing."""
    # Create target positions with different times
    target_positions = [
        TargetPosition(
            time=np.datetime64("2023-01-01T11:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=1.0,
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T10:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=2.0,
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T09:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=3.0,
        ),
    ]

    return RestoredState(
        time=sample_time,
        balances=[AssetBalance(exchange="TEST", currency="USDT", free=10000.0, locked=0.0, total=10000.0)],
        instrument_to_target_positions={sample_instrument: target_positions},
        instrument_to_signal_positions={},
        positions={sample_instrument: Position(sample_instrument, 1.0, 50000.0)},
    )


@pytest.fixture
def state_with_zero_positions(sample_time, sample_instrument, sample_signal):
    """Return a restored state with a mix of zero and nonzero positions for testing."""
    # Create target positions with different times and some zero positions
    target_positions = [
        TargetPosition(
            time=np.datetime64("2023-01-01T11:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=1.0,
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T10:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=0.0,  # Zero position
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T09:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=2.0,
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T08:00:00"),
            instrument=sample_signal.instrument,
            target_position_size=0.0,  # Zero position
        ),
    ]

    return RestoredState(
        time=sample_time,
        balances=[AssetBalance(exchange="TEST", currency="USDT", free=10000.0, locked=0.0, total=10000.0)],
        instrument_to_signal_positions={},
        instrument_to_target_positions={sample_instrument: target_positions},
        positions={sample_instrument: Position(sample_instrument, 1.0, 50000.0)},
    )


@pytest.fixture
def state_with_multiple_instruments(sample_time, sample_instrument, sample_signal):
    """Return a restored state with multiple instruments for testing."""
    # Create a second instrument
    eth_instrument = Instrument(
        symbol="ETHUSDT",
        market_type=MarketType.FUTURE,
        exchange="BINANCE",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.01,
        min_size=0.01,
    )

    # Create target positions for BTC
    btc_target_positions = [
        TargetPosition(
            time=np.datetime64("2023-01-01T11:00:00"),
            instrument=sample_instrument,
            target_position_size=1.0,
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T10:00:00"),
            instrument=sample_instrument,
            target_position_size=0.0,  # Zero position
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T09:00:00"),
            instrument=sample_instrument,
            target_position_size=2.0,
        ),
    ]

    # Create target positions for ETH with different times
    eth_target_positions = [
        TargetPosition(
            time=np.datetime64("2023-01-01T10:30:00"),
            instrument=eth_instrument,
            target_position_size=-1.0,
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T09:30:00"),
            instrument=eth_instrument,
            target_position_size=0.0,  # Zero position
        ),
        TargetPosition(
            time=np.datetime64("2023-01-01T08:30:00"),
            instrument=eth_instrument,
            target_position_size=-2.0,
        ),
    ]

    return RestoredState(
        time=sample_time,
        balances=[AssetBalance(exchange="TEST", currency="USDT", free=10000.0, locked=0.0, total=10000.0)],
        instrument_to_signal_positions={},
        instrument_to_target_positions={
            sample_instrument: btc_target_positions,
            eth_instrument: eth_target_positions,
        },
        positions={
            sample_instrument: Position(sample_instrument, 1.0, 50000.0),
            eth_instrument: Position(eth_instrument, -1.0, 3000.0),
        },
    )


# Tests for TimeFinder.NOW
class TestTimeFinderNOW:
    """Tests for the TimeFinder.NOW method."""

    def test_now_returns_current_time(self, sample_time, empty_state):
        """Test that NOW returns the current time."""
        result = TimeFinder.NOW(sample_time, empty_state)
        assert result == sample_time

    def test_now_ignores_state(self, sample_time, state_with_nonzero_positions):
        """Test that NOW ignores the state and returns the current time."""
        result = TimeFinder.NOW(sample_time, state_with_nonzero_positions)
        assert result == sample_time


# Tests for TimeFinder.LAST_SIGNAL
class TestTimeFinderLASTSIGNAL:
    """Tests for the TimeFinder.LAST_SIGNAL method."""

    def test_empty_state_returns_current_time(self, sample_time, empty_state):
        """Test that LAST_SIGNAL returns the current time for an empty state."""
        result = TimeFinder.LAST_TARGET(sample_time, empty_state)
        assert result == sample_time

    def test_all_nonzero_positions(self, sample_time, state_with_nonzero_positions):
        """Test with all nonzero positions."""
        result = TimeFinder.LAST_TARGET(sample_time, state_with_nonzero_positions)
        # Should return the oldest time since all positions are nonzero
        assert result == np.datetime64("2023-01-01T09:00:00")

    def test_with_zero_positions(self, sample_time, state_with_zero_positions):
        """Test with a mix of zero and nonzero positions."""
        result = TimeFinder.LAST_TARGET(sample_time, state_with_zero_positions)
        # Should return the time of the position right before the first zero position
        assert result == np.datetime64("2023-01-01T11:00:00")

    def test_multiple_instruments(self, sample_time, state_with_multiple_instruments):
        """Test with multiple instruments."""
        result = TimeFinder.LAST_TARGET(sample_time, state_with_multiple_instruments)
        # Should return the minimum time among all instruments
        # For BTC: 2023-01-01T11:00:00
        # For ETH: 2023-01-01T10:30:00
        # Minimum: 2023-01-01T10:30:00
        assert result == np.datetime64("2023-01-01T10:30:00")

    def test_ignores_instruments_without_open_positions(self, sample_time, sample_instrument):
        """Stale instruments (no current open position) must not pull warmup start back."""
        stale_instrument = Instrument(
            symbol="ETHUSDT",
            market_type=MarketType.FUTURE,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.01,
            min_size=0.01,
        )
        # Open position for BTC with recent signals
        btc_signals = [
            Signal(time=np.datetime64("2023-01-01T11:00:00"), instrument=sample_instrument, signal=1.0, price=50000.0),
            Signal(time=np.datetime64("2023-01-01T10:00:00"), instrument=sample_instrument, signal=0.0, price=50000.0),
        ]
        # Stale instrument with very old signals — should be ignored
        stale_signals = [
            Signal(time=np.datetime64("2022-06-01T00:00:00"), instrument=stale_instrument, signal=1.0, price=3000.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={},
            instrument_to_signal_positions={
                sample_instrument: btc_signals,
                stale_instrument: stale_signals,
            },
            positions={
                sample_instrument: Position(sample_instrument, 1.0, 50000.0),
                # stale_instrument has no open position
                stale_instrument: Position(stale_instrument, 0.0, 3000.0),
            },
        )
        result = TimeFinder.LAST_SIGNAL(sample_time, state)
        # Should return BTC's entry time (11:00), ignoring the stale ETH signal from 2022
        assert result == np.datetime64("2023-01-01T11:00:00")

    def test_returns_current_time_when_no_open_positions(self, sample_time, sample_instrument):
        """When no positions are open, should return current time (no warmup needed)."""
        signals = [
            Signal(time=np.datetime64("2022-06-01T00:00:00"), instrument=sample_instrument, signal=1.0, price=50000.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={},
            instrument_to_signal_positions={sample_instrument: signals},
            positions={sample_instrument: Position(sample_instrument, 0.0, 50000.0)},
        )
        result = TimeFinder.LAST_SIGNAL(sample_time, state)
        assert result == sample_time

    def test_last_target_ignores_instruments_without_open_positions(self, sample_time, sample_instrument):
        """LAST_TARGET should also filter by open positions."""
        stale_instrument = Instrument(
            symbol="ETHUSDT",
            market_type=MarketType.FUTURE,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.01,
            min_size=0.01,
        )
        btc_targets = [
            TargetPosition(time=np.datetime64("2023-01-01T11:00:00"), instrument=sample_instrument, target_position_size=1.0),
            TargetPosition(time=np.datetime64("2023-01-01T10:00:00"), instrument=sample_instrument, target_position_size=0.0),
        ]
        stale_targets = [
            TargetPosition(time=np.datetime64("2022-06-01T00:00:00"), instrument=stale_instrument, target_position_size=1.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={
                sample_instrument: btc_targets,
                stale_instrument: stale_targets,
            },
            instrument_to_signal_positions={},
            positions={
                sample_instrument: Position(sample_instrument, 1.0, 50000.0),
                stale_instrument: Position(stale_instrument, 0.0, 3000.0),
            },
        )
        result = TimeFinder.LAST_TARGET(sample_time, state)
        assert result == np.datetime64("2023-01-01T11:00:00")


# Tests for TimeFinder.with_max_lookback
class TestTimeFinderWithMaxLookback:
    """Tests for the TimeFinder.with_max_lookback wrapper."""

    def test_does_not_clamp_when_within_limit(self, sample_time, sample_instrument):
        """When the finder returns a time within max_lookback, it should pass through unchanged."""
        # Signal 1 hour ago — well within 30d cap
        signals = [
            Signal(time=np.datetime64("2023-01-01T11:00:00"), instrument=sample_instrument, signal=1.0, price=50000.0),
            Signal(time=np.datetime64("2023-01-01T10:00:00"), instrument=sample_instrument, signal=0.0, price=50000.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={},
            instrument_to_signal_positions={sample_instrument: signals},
            positions={sample_instrument: Position(sample_instrument, 1.0, 50000.0)},
        )
        capped_finder = TimeFinder.with_max_lookback(TimeFinder.LAST_SIGNAL, "30d")
        result = capped_finder(sample_time, state)
        # LAST_SIGNAL returns 11:00 (the nonzero signal before zero), which is 1h ago — not clamped
        assert result == np.datetime64("2023-01-01T11:00:00")

    def test_clamps_when_exceeding_limit(self, sample_time, sample_instrument):
        """When the finder returns a time beyond max_lookback, it should be clamped."""
        # Signal 60 days ago — exceeds 30d cap
        old_time = np.datetime64("2022-12-01T12:00:00")
        signals = [
            Signal(time=old_time, instrument=sample_instrument, signal=1.0, price=50000.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={},
            instrument_to_signal_positions={sample_instrument: signals},
            positions={sample_instrument: Position(sample_instrument, 1.0, 50000.0)},
        )
        capped_finder = TimeFinder.with_max_lookback(TimeFinder.LAST_SIGNAL, "30d")
        result = capped_finder(sample_time, state)
        expected = sample_time - np.timedelta64(30, "D")
        assert result == expected

    def test_clamps_exactly_at_boundary(self, sample_time, sample_instrument):
        """When the finder returns a time exactly at the max_lookback boundary, it should pass through."""
        exactly_2h_ago = sample_time - np.timedelta64(2, "h")
        signals = [
            Signal(time=exactly_2h_ago, instrument=sample_instrument, signal=1.0, price=50000.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={},
            instrument_to_signal_positions={sample_instrument: signals},
            positions={sample_instrument: Position(sample_instrument, 1.0, 50000.0)},
        )
        capped_finder = TimeFinder.with_max_lookback(TimeFinder.LAST_SIGNAL, "2h")
        result = capped_finder(sample_time, state)
        assert result == exactly_2h_ago

    def test_works_with_now_finder(self, sample_time, empty_state):
        """Should compose with any finder, including NOW."""
        capped_finder = TimeFinder.with_max_lookback(TimeFinder.NOW, "30d")
        result = capped_finder(sample_time, empty_state)
        # NOW returns current_time, which is always within the cap
        assert result == sample_time

    def test_works_with_last_target(self, sample_time, sample_instrument):
        """Should compose with LAST_TARGET the same way."""
        old_time = np.datetime64("2022-11-01T12:00:00")
        targets = [
            TargetPosition(time=old_time, instrument=sample_instrument, target_position_size=1.0),
        ]
        state = RestoredState(
            time=sample_time,
            balances=[],
            instrument_to_target_positions={sample_instrument: targets},
            instrument_to_signal_positions={},
            positions={sample_instrument: Position(sample_instrument, 1.0, 50000.0)},
        )
        capped_finder = TimeFinder.with_max_lookback(TimeFinder.LAST_TARGET, "7d")
        result = capped_finder(sample_time, state)
        expected = sample_time - np.timedelta64(7, "D")
        assert result == expected

    def test_empty_state_returns_current_time(self, sample_time, empty_state):
        """When LAST_SIGNAL returns current_time (no signals), capped result should be the same."""
        capped_finder = TimeFinder.with_max_lookback(TimeFinder.LAST_SIGNAL, "30d")
        result = capped_finder(sample_time, empty_state)
        assert result == sample_time
