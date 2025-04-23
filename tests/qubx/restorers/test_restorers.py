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
import mongomock

from unittest.mock import patch

from qubx.core.basics import AssetBalance, RestoredState
from qubx.restorers.balance import CsvBalanceRestorer, MongoDBBalanceRestorer
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
from qubx.restorers.position import CsvPositionRestorer, MongoDBPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer, MongoDBSignalRestorer
from qubx.restorers.state import CsvStateRestorer, MongoDBStateRestorer


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
            "target_position": [1.0, -2.0, 0.5],  # Added target_position column
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
        params = {
            "bot_id": "test_bot",
            "strategy_id": "test_strategy",
        }
        assert isinstance(CsvPositionRestorer(), IPositionRestorer)
        assert isinstance(MongoDBPositionRestorer(**params), IPositionRestorer)
        csv_position_restorer = create_position_restorer("CsvPositionRestorer")
        mongo_position_restorer = create_position_restorer("MongoDBPositionRestorer", params)
        assert isinstance(csv_position_restorer, IPositionRestorer)
        assert isinstance(mongo_position_restorer, IPositionRestorer)

    def test_signal_restorer_protocol(self):
        """Test that CsvSignalRestorer implements ISignalRestorer."""
        params = {
            "bot_id": "test_bot",
            "strategy_id": "test_strategy",
        }
        assert isinstance(CsvSignalRestorer(), ISignalRestorer)
        assert isinstance(MongoDBSignalRestorer(**params), ISignalRestorer)
        csv_signal_restorer = create_signal_restorer("CsvSignalRestorer")
        mongo_signal_restorer = create_signal_restorer("MongoDBSignalRestorer", params)
        assert isinstance(csv_signal_restorer, ISignalRestorer)
        assert isinstance(mongo_signal_restorer, ISignalRestorer)

    def test_balance_restorer_protocol(self):
        """Test that CsvBalanceRestorer implements IBalanceRestorer."""
        params = {
            "bot_id": "test_bot",
            "strategy_id": "test_strategy",
        }
        assert isinstance(CsvBalanceRestorer(), IBalanceRestorer)
        assert isinstance(MongoDBBalanceRestorer(**params), IBalanceRestorer)
        csv_balance_restorer = create_balance_restorer("CsvBalanceRestorer")
        mongo_balance_restorer = create_balance_restorer("MongoDBBalanceRestorer", params)
        assert isinstance(csv_balance_restorer, IBalanceRestorer)
        assert isinstance(mongo_balance_restorer, IBalanceRestorer)

    def test_state_restorer_protocol(self):
        """Test that CsvStateRestorer implements IStateRestorer."""
        params = {
            "bot_id": "test_bot",
            "strategy_id": "test_strategy",
        }
        assert isinstance(CsvStateRestorer(), IStateRestorer)
        assert isinstance(MongoDBStateRestorer(**params), IStateRestorer)
        csv_state_restorer = create_state_restorer("CsvStateRestorer")
        mongo_state_restorer = create_state_restorer("MongoDBStateRestorer", params)
        assert isinstance(csv_state_restorer, IStateRestorer)
        assert isinstance(mongo_state_restorer, IStateRestorer)


# Position restorer tests
class TestCsvPositionRestorer:
    """Tests for CSV position restorer."""

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


class TestMongoDBPositionRestorer:
    """Tests for MongoDB position restorer."""
    _mongo_uri = "mongodb://localhost:27017/"
    _bot_id = "1"
    _strategy_id = "tests"

    @patch("qubx.restorers.position.MongoClient", new=mongomock.MongoClient)
    def test_with_no_data(self): 
        restorer = MongoDBPositionRestorer(bot_id = self._bot_id, 
                                           strategy_id = self._strategy_id,
                                           mongo_uri= self._mongo_uri)

        result = restorer.restore_positions()

        assert isinstance(result, dict)
        assert len(result) == 0
    
    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]

        collection.insert_one({
            "timestamp": "2025-01-01T00:00:00.000Z",
            "symbol": "BTCUSDT",
            "exchange": "BINANCE.UM",
            "market_type": "SWAP",
            "pnl_quoted": 1,
            "quantity": 1,
            "realized_pnl_quoted": 1,
            "avg_position_price": 90000,
            "market_value_quoted": 0,
            "run_id": "testing-1745335068910429952",
            "strategy_id": self._strategy_id,
            "bot_id": self._bot_id,
            "log_type": "positions"
        })

    def test_with_sample_data(self): 
        mock_client = mongomock.MongoClient(self._mongo_uri)
        
        self._insert_test_data(mock_client)

        with patch("qubx.restorers.position.MongoClient", return_value=mock_client):
            restorer = MongoDBPositionRestorer(bot_id = self._bot_id, 
                                               strategy_id = self._strategy_id,
                                               mongo_uri= self._mongo_uri)

            result = restorer.restore_positions()

            assert isinstance(result, dict)
            assert len(result) > 0

            btc_position = None
            for instrument, position in result.items():
                if instrument.symbol == "BTCUSDT":
                    btc_position = position
                    break

            assert btc_position is not None
            assert btc_position.position_avg_price > 0
            assert btc_position.quantity > 0


# Signal restorer tests
class TestCsvSignalRestorer:
    """Tests for CSV signal restorer."""

    def test_with_sample_data(self, sample_data_dir):
        """Test the CsvSignalRestorer with sample data."""
        # Create the restorer
        restorer = CsvSignalRestorer(base_dir=sample_data_dir, file_pattern="*_signals.csv", lookback_days=7)

        # Restore signals
        target_positions = restorer.restore_signals()

        # Check the results
        assert len(target_positions) == 2

        # Find the target positions for each instrument
        btc_targets = []
        eth_targets = []
        for instrument, target_list in target_positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_targets = target_list
            elif instrument.symbol == "ETHUSDT":
                eth_targets = target_list

        # Check the target positions
        assert len(btc_targets) == 2
        assert len(eth_targets) == 1

        # Check signal values
        assert all(target.signal.signal == 1.0 for target in btc_targets)  # All buy signals
        assert eth_targets[0].signal.signal == -1.0  # Sell signal

        # Check prices
        assert any(target.signal.price == 49000.0 for target in btc_targets)
        assert any(target.signal.price == 50000.0 for target in btc_targets)
        assert eth_targets[0].signal.price == 3000.0

    def test_with_real_data(self, real_data_dir):
        """Test the CsvSignalRestorer with real log data."""
        # Create the restorer
        restorer = CsvSignalRestorer(
            base_dir=str(real_data_dir),
            lookback_days=100_000,  # Use a large value to ensure we get all signals
        )

        # Restore signals
        target_positions = restorer.restore_signals()

        # Check the results
        assert len(target_positions) > 0

        # Find the BTC target positions
        btc_targets = []
        for instrument, target_list in target_positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_targets = target_list
                break

        # Check the BTC target positions
        assert len(btc_targets) > 0

        # Check that we have both buy and sell signals
        buy_targets = [t for t in btc_targets if t.signal.signal > 0.0]
        sell_targets = [t for t in btc_targets if t.signal.signal < 0.0]

        assert len(buy_targets) > 0
        assert len(sell_targets) > 0


class TestMongoDbSignalRestorer:
    """Tests for MongoDB signal restorer."""
    _mongo_uri = "mongodb://localhost:27017/"
    _bot_id = "1"
    _strategy_id = "tests"

    @patch("qubx.restorers.signal.MongoClient", new=mongomock.MongoClient)
    def test_with_no_data(self): 
        restorer = MongoDBSignalRestorer(bot_id = self._bot_id, 
                                           strategy_id = self._strategy_id,
                                           mongo_uri= self._mongo_uri)

        result = restorer.restore_signals()

        assert isinstance(result, dict)
        assert len(result) == 0
    
    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]

        collection.insert_one({
		  "timestamp": "2025-01-01T00:00:00.000Z",
		  "symbol": "BTCUSDT",
		  "exchange": "BINANCE.UM",
		  "market_type": "SWAP",
		  "signal": 1,
		  "target_position": 1,
		  "reference_price": 90000,
		  "run_id": "testing-1745335068910429952",
		  "strategy_id": self._strategy_id,
          "bot_id": self._bot_id,
		  "log_type": "signals"
		})

    def test_with_sample_data(self): 
        mock_client = mongomock.MongoClient(self._mongo_uri)
        
        self._insert_test_data(mock_client)

        with patch("qubx.restorers.signal.MongoClient", return_value=mock_client):
            restorer = MongoDBSignalRestorer(bot_id = self._bot_id, 
                                             strategy_id = self._strategy_id,
                                             mongo_uri= self._mongo_uri)

            result = restorer.restore_signals()

            assert isinstance(result, dict)
            assert len(result) > 0

            btc_targets = []
            for instrument, target_list in result.items():
                if instrument.symbol == "BTCUSDT":
                    btc_targets = target_list
                    break

            assert len(btc_targets) > 0


# Balance restorer tests
class TestCsvBalanceRestorer:
    """Tests for CSV balance restorer."""

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


class TestMongoDBBalanceRestorer:
    """Tests for MongoDB balance restorer."""
    _mongo_uri = "mongodb://localhost:27017/"
    _bot_id = "1"
    _strategy_id = "tests"

    @patch("qubx.restorers.balance.MongoClient", new=mongomock.MongoClient)
    def test_with_no_data(self): 
        restorer = MongoDBBalanceRestorer(bot_id = self._bot_id,
                                          strategy_id = self._strategy_id,
                                          mongo_uri= self._mongo_uri)

        result = restorer.restore_balances()

        assert isinstance(result, dict)
        assert len(result) == 0
    
    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]

        collection.insert_one({
		  "timestamp": "2025-01-01T00:00:00.000Z",
		  "currency": "USDT",
		  "total": 10000,
		  "locked": 1000,
		  "run_id": "testing-1745335068910429952",
		  "strategy_id": self._strategy_id,
          "bot_id": self._bot_id,
		  "log_type": "balance"
		})

    def test_with_sample_data(self): 
        mock_client = mongomock.MongoClient(self._mongo_uri)
        
        self._insert_test_data(mock_client)

        with patch("qubx.restorers.balance.MongoClient", return_value=mock_client):
            restorer = MongoDBBalanceRestorer(bot_id = self._bot_id,
                                              strategy_id = self._strategy_id,
                                              mongo_uri= self._mongo_uri)

            result = restorer.restore_balances()

            assert isinstance(result, dict)
            assert len(result) > 0

            assert "USDT" in result
            assert result["USDT"].total == 10000.0
            assert result["USDT"].locked == 1000.0
            expected_free = result["USDT"].total - result["USDT"].locked
            assert result["USDT"].free == expected_free


# State restorer tests
class TestCsvStateRestorer:
    """Tests for CSV state restorers."""

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

        # Check target positions
        assert len(state.instrument_to_target_positions) == 2

        # Find the target positions for each instrument
        btc_targets = []
        eth_targets = []
        for instrument, target_list in state.instrument_to_target_positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_targets = target_list
            elif instrument.symbol == "ETHUSDT":
                eth_targets = target_list

        # Check the target positions
        assert len(btc_targets) > 0
        assert len(eth_targets) > 0

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
            lookback_days=100_000,
        )

        # Restore state
        state = restorer.restore_state()

        # Check the state
        assert state is not None
        assert isinstance(state.time, np.datetime64)

        # Check that we have positions
        assert len(state.positions) > 0

        # Check that we have target positions
        assert len(state.instrument_to_target_positions) > 0

        # Check that we have balances
        assert len(state.balances) > 0
        assert "USDT" in state.balances
        assert isinstance(state.balances["USDT"], AssetBalance)
        assert state.balances["USDT"].total > 0


class TestMongoDBStateRestorer:
    """Tests for MongoDB state restorer."""
    _mongo_uri = "mongodb://localhost:27017/"
    _bot_id = "1"
    _strategy_id = "tests"

    @patch("qubx.restorers.state.MongoClient", new=mongomock.MongoClient)
    def test_with_no_data(self): 
        restorer = MongoDBStateRestorer(bot_id = self._bot_id, 
                                        strategy_id = self._strategy_id,
                                        mongo_uri= self._mongo_uri)

        result = restorer.restore_state()

        assert isinstance(result, RestoredState)
        assert len(result.positions) == 0
        assert len(result.balances) == 0
        assert len(result.instrument_to_target_positions) == 0

    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]

        collection.insert_many([
            {
                "timestamp": "2025-01-01T00:00:00.000Z",
                "currency": "USDT",
                "total": 10000,
                "locked": 1000,
                "run_id": "testing-1745335068910429952",
                "strategy_id": self._strategy_id,
                "bot_id": self._bot_id,
                "log_type": "balance"
            },
            {
                "timestamp": "2025-01-01T00:00:00.000Z",
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "signal": 1,
                "target_position": 1,
                "reference_price": 90000,
                "run_id": "testing-1745335068910429952",
                "strategy_id": self._strategy_id,
                "bot_id": self._bot_id,
                "log_type": "signals"
            },
            {
                "timestamp": "2025-01-01T00:00:00.000Z",
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 1,
                "quantity": 1,
                "realized_pnl_quoted": 1,
                "avg_position_price": 90000,
                "market_value_quoted": 0,
                "run_id": "testing-1745335068910429952",
                "strategy_id": self._strategy_id,
                "bot_id": self._bot_id,
                "log_type": "positions"
            }
        ])

    def test_with_sample_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)

        self._insert_test_data(mock_client)

        with patch("qubx.restorers.state.MongoClient", return_value=mock_client), \
                patch("qubx.restorers.position.MongoClient", return_value=mock_client), \
                patch("qubx.restorers.signal.MongoClient", return_value=mock_client), \
                patch("qubx.restorers.balance.MongoClient", return_value=mock_client):
            restorer = MongoDBStateRestorer(bot_id = self._bot_id, 
                                            strategy_id = self._strategy_id,
                                            mongo_uri= self._mongo_uri)

            result = restorer.restore_state()

            assert isinstance(result, RestoredState)
            assert len(result.positions) > 0
            assert len(result.balances) > 0
            assert len(result.instrument_to_target_positions) > 0

            assert "USDT" in result.balances
            assert result.balances["USDT"].total == 10000.0
            assert result.balances["USDT"].locked == 1000.0
            expected_free = result.balances["USDT"].total - result.balances["USDT"].locked
            assert result.balances["USDT"].free == expected_free
            
            btc_targets = []
            for instrument, target_list in result.instrument_to_target_positions.items():
                if instrument.symbol == "BTCUSDT":
                    btc_targets = target_list
                    break

            assert len(btc_targets) > 0

            btc_position = None
            for instrument, position in result.positions.items():
                if instrument.symbol == "BTCUSDT":
                    btc_position = position
                    break

            assert btc_position is not None
            assert btc_position.position_avg_price > 0
            assert btc_position.quantity > 0
