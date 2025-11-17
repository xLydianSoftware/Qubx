"""
Tests for the time_finders module.

This file contains tests for the TimeFinder class and its methods.
"""

import numpy as np
import pytest

from qubx.core.basics import (
    AssetBalance,
    AssetType,
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
        asset_type=AssetType.CRYPTO,
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
        asset_type=AssetType.CRYPTO,
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
