"""
Tests for the restorers module.

This file contains all tests for the restorers module, including:
- Protocol implementation tests
- Position restorer tests
- Signal restorer tests
- Balance restorer tests
- State restorer tests
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import AssetBalance
from qubx.restorers.balance import CsvBalanceRestorer
from qubx.restorers.factory import (
    create_balance_restorer,
    create_position_restorer,
    create_signal_restorer,
    create_state_restorer,
)
from qubx.restorers.interfaces import (
    IBalanceRestorer,
    IPositionRestorer,
    ISignalRestorer,
    IStateRestorer,
)
from qubx.restorers.position import CsvPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer
from qubx.restorers.state import CsvStateRestorer


# Test fixtures for sample data
@pytest.fixture
def sample_data_dir():
    """Create a temporary directory with sample data for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a run folder
        run_folder = Path(temp_dir) / "run_20250306093316"
        run_folder.mkdir()

        # Create position data
        position_data = {
            "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:30:00", "2023-01-01 13:00:00"],
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "exchange": ["BINANCE", "BINANCE", "BINANCE"],
            "market_type": ["FUTURE", "FUTURE", "FUTURE"],
            "quantity": [1.0, 2.0, 3.0],
            "avg_position_price": [50000.0, 51000.0, 3000.0],
            "unrealized_pnl": [100.0, 200.0, 300.0],
            "realized_pnl_quoted": [50.0, 60.0, 70.0],
            "liquidation_price": [40000.0, 41000.0, 2000.0],
            "current_price": [50100.0, 51200.0, 3050.0],
        }
        position_df = pd.DataFrame(position_data)
        position_df.to_csv(run_folder / "test_strategy_positions.csv", index=False)

        # Create signal data
        now = pd.Timestamp.now()
        yesterday = now - pd.Timedelta(days=1)
        two_days_ago = now - pd.Timedelta(days=2)
        signal_data = {
            "timestamp": [
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                two_days_ago.strftime("%Y-%m-%d %H:%M:%S"),
            ],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
            "exchange": ["BINANCE", "BINANCE", "BINANCE"],
            "market_type": ["FUTURE", "FUTURE", "FUTURE"],
            "signal": [1.0, -1.0, 1.0],  # Buy, Sell, Buy
            "price": [50000.0, 3000.0, 49000.0],
            "size": [1.0, 2.0, 0.5],
            "meta": ["{}", "{}", "{}"],
        }
        signal_df = pd.DataFrame(signal_data)
        signal_df.to_csv(run_folder / "test_strategy_signals.csv", index=False)

        # Create balance data
        balance_data = {
            "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:30:00", "2023-01-01 13:00:00"],
            "currency": ["USDT", "BTC", "USDT"],
            "total": [100000.0, 1.5, 99000.0],
            "locked": [0.0, 0.0, 1000.0],
            "run_id": ["test-123", "test-123", "test-123"],
        }
        balance_df = pd.DataFrame(balance_data)
        balance_df.to_csv(run_folder / "test_strategy_balance.csv", index=False)

        yield temp_dir


@pytest.fixture
def real_data_dir():
    """Path to real log data for testing."""
    return Path("tests/data/logs")


# Protocol implementation tests
class TestProtocolImplementations:
    """Tests for protocol implementations."""

    def test_position_restorer_protocol(self):
        """Test that CsvPositionRestorer implements IPositionRestorer."""
        assert isinstance(CsvPositionRestorer(), IPositionRestorer)
        position_restorer = create_position_restorer("CsvPositionRestorer")
        assert isinstance(position_restorer, IPositionRestorer)

    def test_signal_restorer_protocol(self):
        """Test that CsvSignalRestorer implements ISignalRestorer."""
        assert isinstance(CsvSignalRestorer(), ISignalRestorer)
        signal_restorer = create_signal_restorer("CsvSignalRestorer")
        assert isinstance(signal_restorer, ISignalRestorer)

    def test_balance_restorer_protocol(self):
        """Test that CsvBalanceRestorer implements IBalanceRestorer."""
        assert isinstance(CsvBalanceRestorer(), IBalanceRestorer)
        balance_restorer = create_balance_restorer("CsvBalanceRestorer")
        assert isinstance(balance_restorer, IBalanceRestorer)

    def test_state_restorer_protocol(self):
        """Test that CsvStateRestorer implements IStateRestorer."""
        assert isinstance(CsvStateRestorer(), IStateRestorer)
        state_restorer = create_state_restorer("CsvStateRestorer")
        assert isinstance(state_restorer, IStateRestorer)


# Position restorer tests
class TestPositionRestorer:
    """Tests for position restorers."""

    def test_with_sample_data(self, sample_data_dir):
        """Test the CsvPositionRestorer with sample data."""
        # Create the restorer
        restorer = CsvPositionRestorer(base_dir=sample_data_dir, file_pattern="*_positions.csv")

        # Restore positions
        positions = restorer.restore_positions()

        # Check the results
        assert len(positions) == 2

        # Find the instruments in the positions dictionary
        btc_position = None
        eth_position = None
        for instrument, position in positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_position = position
            elif instrument.symbol == "ETHUSDT":
                eth_position = position

        # Check the positions
        assert btc_position is not None
        assert eth_position is not None

        assert btc_position.quantity == 2.0
        assert btc_position.position_avg_price == 51000.0

        assert eth_position.quantity == 3.0
        assert eth_position.position_avg_price == 3000.0

    def test_with_real_data(self, real_data_dir):
        """Test the CsvPositionRestorer with real log data."""
        # Create the restorer
        restorer = CsvPositionRestorer(base_dir=str(real_data_dir))

        # Restore positions
        positions = restorer.restore_positions()

        # Check the results
        assert len(positions) > 0

        # Find the BTC position
        btc_position = None
        for instrument, position in positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_position = position
                break

        # Check the BTC position
        assert btc_position is not None

        # Note: We're not checking the exact quantity or price since these may change
        # in the real data. Instead, we just verify that the position exists and has
        # reasonable values.
        assert isinstance(btc_position.quantity, float)
        assert isinstance(btc_position.position_avg_price, float)
        assert btc_position.position_avg_price > 0


# Signal restorer tests
class TestSignalRestorer:
    """Tests for signal restorers."""

    def test_with_sample_data(self, sample_data_dir):
        """Test the CsvSignalRestorer with sample data."""
        # Create the restorer
        restorer = CsvSignalRestorer(base_dir=sample_data_dir, file_pattern="*_signals.csv", lookback_days=7)

        # Restore signals
        signals = restorer.restore_signals()

        # Check the results
        assert len(signals) == 2

        # Find the signals for each instrument
        btc_signals = []
        eth_signals = []
        for instrument, signal_list in signals.items():
            if instrument.symbol == "BTCUSDT":
                btc_signals = signal_list
            elif instrument.symbol == "ETHUSDT":
                eth_signals = signal_list

        # Check the signals
        assert len(btc_signals) == 2
        assert len(eth_signals) == 1

        # Check signal values
        assert all(signal.signal == 1.0 for signal in btc_signals)  # All buy signals
        assert eth_signals[0].signal == -1.0  # Sell signal

        # Check prices
        assert any(signal.price == 49000.0 for signal in btc_signals)
        assert any(signal.price == 50000.0 for signal in btc_signals)
        assert eth_signals[0].price == 3000.0

    def test_with_real_data(self, real_data_dir):
        """Test the CsvSignalRestorer with real log data."""
        # Create the restorer
        restorer = CsvSignalRestorer(
            base_dir=str(real_data_dir),
            lookback_days=30,  # Use a large value to ensure we get all signals
        )

        # Restore signals
        signals = restorer.restore_signals()

        # Check the results
        assert len(signals) > 0

        # Find the BTC signals
        btc_signals = []
        for instrument, signal_list in signals.items():
            if instrument.symbol == "BTCUSDT":
                btc_signals = signal_list
                break

        # Check the BTC signals
        assert len(btc_signals) > 0

        # Check that we have both buy and sell signals
        buy_signals = [s for s in btc_signals if s.signal > 0]
        sell_signals = [s for s in btc_signals if s.signal < 0]

        assert len(buy_signals) > 0
        assert len(sell_signals) > 0


# Balance restorer tests
class TestBalanceRestorer:
    """Tests for balance restorers."""

    def test_with_sample_data(self, sample_data_dir):
        """Test the CsvBalanceRestorer with sample data."""
        # Create the restorer
        restorer = CsvBalanceRestorer(base_dir=sample_data_dir, file_pattern="*_balance.csv")

        # Restore balances
        balances = restorer.restore_balances()

        # Check the results
        assert len(balances) == 2

        # Check USDT balance (should be the latest entry)
        assert "USDT" in balances
        assert balances["USDT"].total == 99000.0
        assert balances["USDT"].locked == 1000.0
        expected_free = balances["USDT"].total - balances["USDT"].locked
        assert balances["USDT"].free == expected_free

        # Check BTC balance
        assert "BTC" in balances
        assert balances["BTC"].total == 1.5
        assert balances["BTC"].locked == 0.0
        expected_free = balances["BTC"].total - balances["BTC"].locked
        assert balances["BTC"].free == expected_free

    def test_with_real_data(self, real_data_dir):
        """Test the CsvBalanceRestorer with real log data."""
        # Create the restorer
        restorer = CsvBalanceRestorer(base_dir=str(real_data_dir), file_pattern="*_balance.csv")

        # Restore balances
        balances = restorer.restore_balances()

        # Check the results
        assert len(balances) > 0

        # Check that we have USDT balance
        assert "USDT" in balances
        assert isinstance(balances["USDT"].total, float)
        assert isinstance(balances["USDT"].locked, float)
        assert balances["USDT"].total > 0


# State restorer tests
class TestStateRestorer:
    """Tests for state restorers."""

    def test_with_sample_data(self, sample_data_dir):
        """Test the CsvStateRestorer with sample data."""
        # Create the restorer
        restorer = CsvStateRestorer(
            base_dir=sample_data_dir,
            strategy_name="test_strategy",
        )

        # Restore state
        state = restorer.restore_state()

        # Check the state
        assert state is not None
        assert isinstance(state.time, np.datetime64)

        # Check positions
        assert len(state.positions) == 2

        # Find the instruments in the positions dictionary
        btc_position = None
        eth_position = None
        for instrument, position in state.positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_position = position
            elif instrument.symbol == "ETHUSDT":
                eth_position = position

        # Check the positions
        assert btc_position is not None
        assert eth_position is not None

        assert btc_position.quantity == 2.0
        assert btc_position.position_avg_price == 51000.0

        assert eth_position.quantity == 3.0
        assert eth_position.position_avg_price == 3000.0

        # Check signals
        assert len(state.instrument_to_signals) == 2

        # Find the signals for each instrument
        btc_signals = []
        eth_signals = []
        for instrument, signal_list in state.instrument_to_signals.items():
            if instrument.symbol == "BTCUSDT":
                btc_signals = signal_list
            elif instrument.symbol == "ETHUSDT":
                eth_signals = signal_list

        # Check the signals
        assert len(btc_signals) > 0
        assert len(eth_signals) > 0

        # Check balances
        assert len(state.balances) == 2
        assert "USDT" in state.balances
        assert "BTC" in state.balances

        assert state.balances["USDT"].total > 0
        assert state.balances["BTC"].total > 0

    def test_with_real_data(self, real_data_dir):
        """Test the CsvStateRestorer with real log data."""
        # Create the restorer
        restorer = CsvStateRestorer(
            base_dir=str(real_data_dir),
        )

        # Restore state
        state = restorer.restore_state()

        # Check the state
        assert state is not None
        assert isinstance(state.time, np.datetime64)

        # Check that we have positions
        assert len(state.positions) > 0

        # Check that we have signals
        assert len(state.instrument_to_signals) > 0

        # Check that we have balances
        assert len(state.balances) > 0
        assert "USDT" in state.balances
        assert isinstance(state.balances["USDT"], AssetBalance)
        assert state.balances["USDT"].total > 0
