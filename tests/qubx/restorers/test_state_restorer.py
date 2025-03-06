"""
Tests for the state restorer.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from qubx.core.basics import AssetBalance
from qubx.restorers.state import CsvStateRestorer


def test_csv_state_restorer_with_sample_data():
    """Test the CsvStateRestorer with sample data."""
    # Create a temporary directory structure for test files
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
        }
        position_df = pd.DataFrame(position_data)
        position_df.to_csv(run_folder / "test_strategy_positions.csv", index=False)

        # Create signal data
        now = pd.Timestamp.now()
        yesterday = now - pd.Timedelta(days=1)
        signal_data = {
            "timestamp": [
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
            ],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "exchange": ["BINANCE", "BINANCE"],
            "market_type": ["FUTURE", "FUTURE"],
            "signal": [1.0, -1.0],  # Buy, Sell
            "price": [50000.0, 3000.0],
            "size": [1.0, 2.0],
            "meta": ["{}", "{}"],
        }
        signal_df = pd.DataFrame(signal_data)
        signal_df.to_csv(run_folder / "test_strategy_signals.csv", index=False)

        # Create balance data
        balance_data = {
            "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:30:00"],
            "currency": ["USDT", "BTC"],
            "total": [100000.0, 1.5],
            "locked": [0.0, 0.0],
            "run_id": ["test-123", "test-123"],
        }
        balance_df = pd.DataFrame(balance_data)
        balance_df.to_csv(run_folder / "test_strategy_balance.csv", index=False)

        # Create the restorer
        restorer = CsvStateRestorer(
            base_dir=temp_dir,
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
        assert len(btc_signals) == 1
        assert len(eth_signals) == 1

        # Check signal values
        assert btc_signals[0].signal == 1.0  # Buy signal
        assert eth_signals[0].signal == -1.0  # Sell signal

        # Check balances
        assert len(state.balances) == 2
        assert "USDT" in state.balances
        assert "BTC" in state.balances

        assert state.balances["USDT"].total == 100000.0
        assert state.balances["USDT"].locked == 0.0

        assert state.balances["BTC"].total == 1.5
        assert state.balances["BTC"].locked == 0.0


def test_csv_state_restorer_with_real_data():
    """Test the CsvStateRestorer with real log data."""
    # Path to the real log data
    log_dir = Path("tests/data/logs")

    # Create the restorer
    restorer = CsvStateRestorer(
        base_dir=str(log_dir),
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
