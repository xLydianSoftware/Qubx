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
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import mongomock
import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import (
    Balance,
    Instrument,
    MarketType,
    RestoredState,
)
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
from qubx.restorers.utils import mongo_canonical_run_id, mongo_latest_run_id


# Mock instruments for testing
def create_mock_instruments():
    """Create test instruments for various markets."""
    return {
        "BINANCE:FUTURE:BTCUSDT": Instrument(
            symbol="BTCUSDT",

            market_type=MarketType.FUTURE,
            exchange="BINANCE",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTC/USDT:USDT",
            tick_size=0.1,
            lot_size=0.001,
            min_size=0.001,
        ),
        "BINANCE:FUTURE:ETHUSDT": Instrument(
            symbol="ETHUSDT",

            market_type=MarketType.FUTURE,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETH/USDT:USDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        ),
        "BINANCE:FUTURE:SOLUSDT": Instrument(
            symbol="SOLUSDT",

            market_type=MarketType.FUTURE,
            exchange="BINANCE",
            base="SOL",
            quote="USDT",
            settle="USDT",
            exchange_symbol="SOL/USDT:USDT",
            tick_size=0.001,
            lot_size=0.01,
            min_size=0.01,
        ),
        "BINANCE:SWAP:BTCUSDT": Instrument(
            symbol="BTCUSDT",

            market_type=MarketType.SWAP,
            exchange="BINANCE",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTC/USDT:USDT",
            tick_size=0.1,
            lot_size=0.001,
            min_size=0.001,
        ),
        "BINANCE:SWAP:ETHUSDT": Instrument(
            symbol="ETHUSDT",

            market_type=MarketType.SWAP,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETH/USDT:USDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        ),
        "BINANCE:SWAP:SOLUSDT": Instrument(
            symbol="SOLUSDT",

            market_type=MarketType.SWAP,
            exchange="BINANCE",
            base="SOL",
            quote="USDT",
            settle="USDT",
            exchange_symbol="SOL/USDT:USDT",
            tick_size=0.001,
            lot_size=0.01,
            min_size=0.01,
        ),
    }


@pytest.fixture
def mock_lookup():
    """Mock the lookup service to return test instruments."""
    mock_instruments = create_mock_instruments()
    
    def mock_find_symbol(exchange, symbol, market_type=None):
        # Try to find exact match first
        for key, instrument in mock_instruments.items():
            if instrument.exchange == exchange and instrument.symbol == symbol:
                if market_type is None or instrument.market_type == market_type:
                    return instrument
        return None
    
    with patch('qubx.restorers.position.lookup') as mock_pos, \
         patch('qubx.restorers.signal.lookup') as mock_sig:
        
        mock_pos.find_symbol = mock_find_symbol
        mock_sig.find_symbol = mock_find_symbol
        
        yield mock_pos


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
        three_days_ago = now - pd.Timedelta(days=3)
        signal_data = {
            "timestamp": [
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                two_days_ago.strftime("%Y-%m-%d %H:%M:%S"),
                three_days_ago.strftime("%Y-%m-%d %H:%M:%S"),
            ],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT"],
            "exchange": ["BINANCE", "BINANCE", "BINANCE", "BINANCE"],
            "market_type": ["FUTURE", "FUTURE", "FUTURE", "SWAP"],
            "signal": [1.0, -1.0, 1.0, 0.0],  # Buy, Sell, Buy, Buy
            "price": [50000.0, 3000.0, 49000.0, np.nan],
            "service": [False, False, False, True],
            # - - - - - - - - - - - - - - - - - - - - - -
            # - 2025-06-26: we don't have size and target_position in the signals log anymore
            # - it's moved to the targets log
            # "size": [1.0, 2.0, 0.5],
            # "target_position": [1.0, -2.0, 0.5],  # Added target_position column
            # - - - - - - - - - - - - - - - - - - - - - -
            "meta": ["{}", "{}", "{}", "{}"],
        }
        signal_df = pd.DataFrame(signal_data)
        signal_df.to_csv(run_folder / "test_strategy_signals.csv", index=False)

        # Create targets data
        targets_data = {
            "timestamp": [
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                two_days_ago.strftime("%Y-%m-%d %H:%M:%S"),
                three_days_ago.strftime("%Y-%m-%d %H:%M:%S"),
            ],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT"],
            "exchange": ["BINANCE", "BINANCE", "BINANCE", "BINANCE"],
            "market_type": ["FUTURE", "FUTURE", "FUTURE", "SWAP"],
            "target_position": [0.5, -1.2, 0.3, 0.1],
        }
        targets_df = pd.DataFrame(targets_data)
        targets_df.to_csv(run_folder / "test_strategy_targets.csv", index=False)

        # Create balance data
        balance_data = {
            "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:30:00", "2023-01-01 13:00:00"],
            "exchange": ["BINANCE", "BINANCE", "BINANCE"],
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
        params = {"strategy_name": "test_strategy", "mongo_client": mongomock.MongoClient()}
        assert isinstance(CsvPositionRestorer(), IPositionRestorer)
        assert isinstance(MongoDBPositionRestorer(**params), IPositionRestorer)
        csv_position_restorer = create_position_restorer("CsvPositionRestorer")
        mongo_position_restorer = create_position_restorer("MongoDBPositionRestorer", params)
        assert isinstance(csv_position_restorer, IPositionRestorer)
        assert isinstance(mongo_position_restorer, IPositionRestorer)

    def test_signal_restorer_protocol(self):
        """Test that CsvSignalRestorer implements ISignalRestorer."""
        params = {"strategy_name": "test_strategy", "mongo_client": mongomock.MongoClient()}
        assert isinstance(CsvSignalRestorer(), ISignalRestorer)
        assert isinstance(MongoDBSignalRestorer(**params), ISignalRestorer)
        csv_signal_restorer = create_signal_restorer("CsvSignalRestorer")
        mongo_signal_restorer = create_signal_restorer("MongoDBSignalRestorer", params)
        assert isinstance(csv_signal_restorer, ISignalRestorer)
        assert isinstance(mongo_signal_restorer, ISignalRestorer)

    def test_balance_restorer_protocol(self):
        """Test that CsvBalanceRestorer implements IBalanceRestorer."""
        params = {"strategy_name": "test_strategy", "mongo_client": mongomock.MongoClient()}
        assert isinstance(CsvBalanceRestorer(), IBalanceRestorer)
        assert isinstance(MongoDBBalanceRestorer(**params), IBalanceRestorer)
        csv_balance_restorer = create_balance_restorer("CsvBalanceRestorer")
        mongo_balance_restorer = create_balance_restorer("MongoDBBalanceRestorer", params)
        assert isinstance(csv_balance_restorer, IBalanceRestorer)
        assert isinstance(mongo_balance_restorer, IBalanceRestorer)

    def test_state_restorer_protocol(self):
        """Test that CsvStateRestorer implements IStateRestorer."""
        params = {
            "strategy_name": "test_strategy",
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

    def test_with_sample_data(self, sample_data_dir, mock_lookup):
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

    def test_episode_fields_roundtrip_and_legacy(self, mock_lookup):
        """Episode baselines round-trip through the CSV log; a legacy row (columns absent/NaN) gets
        episode-at-restore: baselines = restored accumulators, episode_start_time = the row timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_folder = Path(temp_dir) / "run_20250308120000"
            run_folder.mkdir()
            ts = "2025-01-08 12:00:00"

            # BTC: full episode fields present -> round-trip verbatim
            # ETH: episode columns NaN (legacy) -> episode-at-restore
            df = pd.DataFrame(
                {
                    "timestamp": [ts, ts],
                    "symbol": ["BTCUSDT", "ETHUSDT"],
                    "exchange": ["BINANCE", "BINANCE"],
                    "market_type": ["FUTURE", "FUTURE"],
                    "quantity": [1.0, 2.0],
                    "avg_position_price": [50000.0, 3000.0],
                    "realized_pnl_quoted": [100.0, 70.0],
                    "funding_pnl_quoted": [-5.0, -1.0],
                    "commissions_quoted": [3.0, 4.0],
                    "current_price": [50100.0, 3050.0],
                    "episode_start_time": ["2025-01-01 00:00:00", np.nan],
                    "realized_pnl_at_open_quoted": [90.0, np.nan],
                    "commissions_at_open_quoted": [2.0, np.nan],
                    "funding_at_open_quoted": [-4.0, np.nan],
                }
            )
            df.to_csv(run_folder / "test_strategy_positions.csv", index=False)

            restorer = CsvPositionRestorer(base_dir=temp_dir, file_pattern="*_positions.csv")
            positions = restorer.restore_positions()

            btc = next(p for i, p in positions.items() if i.symbol == "BTCUSDT")
            eth = next(p for i, p in positions.items() if i.symbol == "ETHUSDT")

            # BTC: episode fields round-trip verbatim
            assert btc.episode_start_time == pd.Timestamp("2025-01-01 00:00:00").asm8
            assert btc.r_pnl_at_open == 90.0
            assert btc.commissions_at_open == 2.0
            assert btc.cumulative_funding_at_open == -4.0

            # ETH: legacy -> episode-at-restore (baselines = accumulators, start = row timestamp)
            assert eth.episode_start_time == pd.Timestamp(ts).asm8
            assert eth.r_pnl_at_open == 70.0
            assert eth.commissions_at_open == 4.0
            assert eth.cumulative_funding_at_open == -1.0


class TestMongoDBPositionRestorer:
    """Tests for MongoDB position restorer."""

    _mongo_uri = "mongodb://localhost:27017/"
    _strategy_name = "test_strategy"

    def test_with_no_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)
        restorer = MongoDBPositionRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)

        result = restorer.restore_positions()

        assert isinstance(result, dict)
        assert len(result) == 0

    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]
        now = datetime.now()
        log_timestamp = now - timedelta(days=1)

        collection.insert_one(
            {
                "timestamp": log_timestamp,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 1,
                "quantity": 1,
                "realized_pnl_quoted": 1,
                "avg_position_price": 90000,
                "market_value_quoted": 0,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "positions",
            }
        )

    def test_with_sample_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)

        self._insert_test_data(mock_client)

        restorer = MongoDBPositionRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)

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

    def test_episode_fields_roundtrip_and_legacy(self):
        """Episode baselines round-trip through the Mongo log; a legacy doc (fields absent) gets
        episode-at-restore: baselines = restored accumulators, episode_start_time = the doc timestamp."""
        mock_client = mongomock.MongoClient(self._mongo_uri)
        col = mock_client["default_logs_db"]["qubx_logs"]
        ts = (datetime.now() - timedelta(days=1)).replace(microsecond=0)

        # BTC: full episode fields present -> round-trip verbatim
        col.insert_one(
            {
                "timestamp": ts, "symbol": "BTCUSDT", "exchange": "BINANCE.UM", "market_type": "SWAP",
                "quantity": 1, "realized_pnl_quoted": 100.0, "avg_position_price": 90000,
                "funding_pnl_quoted": -5.0, "commissions_quoted": 3.0,
                "episode_start_time": "2025-01-01T00:00:00", "realized_pnl_at_open_quoted": 90.0,
                "commissions_at_open_quoted": 2.0, "funding_at_open_quoted": -4.0,
                "run_id": "run-1", "strategy_name": self._strategy_name, "log_type": "positions",
            }
        )
        # ETH: legacy doc (no episode fields) -> episode-at-restore
        col.insert_one(
            {
                "timestamp": ts, "symbol": "ETHUSDT", "exchange": "BINANCE.UM", "market_type": "SWAP",
                "quantity": 2, "realized_pnl_quoted": 70.0, "avg_position_price": 3000,
                "funding_pnl_quoted": -1.0, "commissions_quoted": 4.0,
                "run_id": "run-1", "strategy_name": self._strategy_name, "log_type": "positions",
            }
        )

        restorer = MongoDBPositionRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)
        result = restorer.restore_positions()
        btc = next(p for i, p in result.items() if i.symbol == "BTCUSDT")
        eth = next(p for i, p in result.items() if i.symbol == "ETHUSDT")

        assert btc.episode_start_time == pd.Timestamp("2025-01-01T00:00:00").asm8
        assert btc.r_pnl_at_open == 90.0
        assert btc.commissions_at_open == 2.0
        assert btc.cumulative_funding_at_open == -4.0

        assert eth.episode_start_time == pd.Timestamp(ts).asm8
        assert eth.r_pnl_at_open == 70.0
        assert eth.commissions_at_open == 4.0
        assert eth.cumulative_funding_at_open == -1.0


# Signal restorer tests
class TestCsvSignalRestorer:
    """Tests for CSV signal restorer."""

    def test_with_sample_data(self, sample_data_dir, mock_lookup):
        """Test the CsvSignalRestorer with sample data."""
        # Create the restorer
        restorer = CsvSignalRestorer(
            base_dir=sample_data_dir,
            signals_file_pattern="*_signals.csv",
            targets_file_pattern="*_targets.csv",
            lookback_days=7,
        )

        # Restore signals
        signals = restorer.restore_signals()

        # Check the results
        assert len(signals) == 3

        # Find the target positions for each instrument
        btc_signals = []
        eth_signals = []
        for instrument, signals_list in signals.items():
            if instrument.symbol == "BTCUSDT":
                btc_signals = signals_list
            elif instrument.symbol == "ETHUSDT":
                eth_signals = signals_list

        # Check the target positions
        assert len(btc_signals) == 2
        assert len(eth_signals) == 1

        # Check signal values
        assert all(signal.signal == 1.0 for signal in btc_signals)  # All buy signals
        assert eth_signals[0].signal == -1.0  # Sell signal

        # Check prices
        assert any(signal.price == 49000.0 for signal in btc_signals)
        assert any(signal.price == 50000.0 for signal in btc_signals)
        assert eth_signals[0].price == 3000.0

    def test_restore_targets_roundtrips_options_group(self, mock_lookup):
        """Regression: the persisted ``options`` column (e.g. ``{'group': 'STATARB'}``) must
        round-trip into ``TargetPosition.options`` on restore. Grouped strategies (maker/taker
        pairs) key on ``options['group']``; dropping it made them un-restorable (the gatherer
        raised on an ungrouped non-zero target)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_folder = Path(temp_dir) / "run_20260701160000"
            run_folder.mkdir()
            ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            targets_df = pd.DataFrame(
                {
                    "timestamp": [ts, ts],
                    "symbol": ["BTCUSDT", "ETHUSDT"],
                    "exchange": ["BINANCE", "BINANCE"],
                    "market_type": ["FUTURE", "SWAP"],
                    "target_position": [0.5, -1.2],
                    "entry_price": [np.nan, np.nan],
                    "take_price": [np.nan, np.nan],
                    "stop_price": [np.nan, np.nan],
                    # As persisted by CsvFileLogsWriter: str() of the options dict.
                    "options": ["{'group': 'STATARB'}", "{'group': 'STATARB'}"],
                }
            )
            targets_df.to_csv(run_folder / "test_strategy_targets.csv", index=False)

            restorer = CsvSignalRestorer(base_dir=temp_dir, targets_file_pattern="*_targets.csv")
            targets = restorer.restore_targets()

            assert len(targets) == 2
            for inst, tlist in targets.items():
                assert tlist, f"no targets restored for {inst.symbol}"
                assert tlist[-1].options.get("group") == "STATARB", (
                    f"group dropped on restore for {inst.symbol}: options={tlist[-1].options}"
                )

    def test_with_real_data(self, real_data_dir):
        """Test the CsvSignalRestorer with real log data."""
        # Create the restorer
        restorer = CsvSignalRestorer(
            base_dir=str(real_data_dir),
            lookback_days=100_000,  # Use a large value to ensure we get all signals
        )

        # Restore signals
        signals = restorer.restore_signals()

        # Check the results
        assert len(signals) > 0

        # Find the BTC target positions
        btc_signals = []
        for instrument, target_list in signals.items():
            if instrument.symbol == "BTCUSDT":
                btc_signals = target_list
                break

        # Check the BTC target positions
        assert len(btc_signals) > 0

        # Check that we have both buy and sell signals
        buy_targets = [t for t in btc_signals if t.signal > 0.0]
        sell_targets = [t for t in btc_signals if t.signal < 0.0]

        assert len(buy_targets) > 0
        assert len(sell_targets) > 0


class TestMongoDbSignalRestorer:
    """Tests for MongoDB signal restorer."""

    _mongo_uri = "mongodb://localhost:27017/"
    _strategy_name = "test_strategy"

    def test_with_no_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)
        restorer = MongoDBSignalRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)

        result = restorer.restore_signals()

        assert isinstance(result, dict)
        assert len(result) == 0

    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]
        now = datetime.now()
        log_timestamp = now - timedelta(days=1)

        collection.insert_one(
            {
                "timestamp": log_timestamp,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "signal": 1,
                # "target_position": 1,
                "reference_price": 90000,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "signals",
            }
        )

    def test_with_sample_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)

        self._insert_test_data(mock_client)

        restorer = MongoDBSignalRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)

        result = restorer.restore_signals()

        assert isinstance(result, dict)
        assert len(result) > 0

        btc_targets = []
        for instrument, signals_list in result.items():
            if instrument.symbol == "BTCUSDT":
                btc_targets = signals_list
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
        balances_list = restorer.restore_balances()

        # Check the results
        assert len(balances_list) == 2

        # Convert to dict for easier testing
        balances = {b.currency: b for b in balances_list}

        # Check USDT balance (should be the latest entry)
        assert "USDT" in balances
        assert balances["USDT"].total == 99000.0
        assert balances["USDT"].locked == 1000.0
        expected_free = balances["USDT"].total - balances["USDT"].locked
        assert balances["USDT"].free == expected_free
        assert balances["USDT"].exchange == "BINANCE"

        # Check BTC balance
        assert "BTC" in balances
        assert balances["BTC"].total == 1.5
        assert balances["BTC"].locked == 0.0
        expected_free = balances["BTC"].total - balances["BTC"].locked
        assert balances["BTC"].free == expected_free
        assert balances["BTC"].exchange == "BINANCE"

    def test_with_real_data(self, real_data_dir):
        """Test the CsvBalanceRestorer with real log data."""
        # Create the restorer
        restorer = CsvBalanceRestorer(base_dir=str(real_data_dir), file_pattern="*_balance.csv")

        # Restore balances
        balances_list = restorer.restore_balances()

        # Check the results
        assert len(balances_list) > 0

        # Convert to dict for easier testing
        balances = {b.currency: b for b in balances_list}

        # Check that we have USDT balance
        assert "USDT" in balances
        assert isinstance(balances["USDT"].total, float)
        assert isinstance(balances["USDT"].locked, float)
        assert balances["USDT"].total > 0


class TestMongoDBBalanceRestorer:
    """Tests for MongoDB balance restorer."""

    _mongo_uri = "mongodb://localhost:27017/"
    _strategy_name = "test_strategy"

    def test_with_no_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)
        restorer = MongoDBBalanceRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)

        result = restorer.restore_balances()

        assert isinstance(result, list)
        assert len(result) == 0

    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]
        collection = db["qubx_logs"]
        now = datetime.now()
        log_timestamp = now - timedelta(days=1)

        collection.insert_one(
            {
                "timestamp": log_timestamp,
                "exchange": "BINANCE",
                "currency": "USDT",
                "total": 10000,
                "locked": 1000,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "balance",
            }
        )

    def test_with_sample_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)

        self._insert_test_data(mock_client)

        restorer = MongoDBBalanceRestorer(strategy_name=self._strategy_name, mongo_client=mock_client)

        result_list = restorer.restore_balances()

        assert isinstance(result_list, list)
        assert len(result_list) > 0

        # Convert to dict for easier testing
        result = {b.currency: b for b in result_list}

        assert "USDT" in result
        assert result["USDT"].total == 10000.0
        assert result["USDT"].locked == 1000.0
        expected_free = result["USDT"].total - result["USDT"].locked
        assert result["USDT"].free == expected_free
        assert result["USDT"].exchange == "BINANCE"


# State restorer tests
class TestCsvStateRestorer:
    """Tests for CSV state restorers."""

    def test_with_sample_data(self, sample_data_dir, mock_lookup):
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

        # Check signal positions
        assert len(state.instrument_to_signal_positions) == 3

        # Find the signals for each instrument
        btc_signals = []
        eth_signals = []
        for instrument, signals_list in state.instrument_to_signal_positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_signals = signals_list
            elif instrument.symbol == "ETHUSDT":
                eth_signals = signals_list

        # Check the signals
        assert len(btc_signals) > 0
        assert len(eth_signals) > 0

        # Check target positions
        assert len(state.instrument_to_target_positions) == 3

        btc_targets = []
        eth_targets = []
        for instrument, targets_list in state.instrument_to_target_positions.items():
            if instrument.symbol == "BTCUSDT":
                btc_targets = targets_list
            elif instrument.symbol == "ETHUSDT":
                eth_targets = targets_list

        # Check the signals
        assert len(btc_targets) > 0
        assert len(eth_targets) > 0

        # Check balances
        assert len(state.balances) == 2

        # Convert to dict for easier testing
        balances = {b.currency: b for b in state.balances}
        assert "USDT" in balances
        assert "BTC" in balances

        assert balances["USDT"].total > 0
        assert balances["BTC"].total > 0

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

        # Check that we have signal positions
        assert len(state.instrument_to_signal_positions) > 0

        # Check that we have target positions
        # assert len(state.instrument_to_target_positions) > 0

        # Check that we have balances
        assert len(state.balances) > 0

        # Convert to dict for easier testing
        balances = {b.currency: b for b in state.balances}
        assert "USDT" in balances
        assert isinstance(balances["USDT"], Balance)
        assert balances["USDT"].total > 0


class TestMongoDBStateRestorer:
    """Tests for MongoDB state restorer."""

    _mongo_uri = "mongodb://localhost:27017/"
    _strategy_name = "test_strategy"

    @patch("qubx.restorers.state.MongoClient", new=mongomock.MongoClient)
    def test_with_no_data(self):
        restorer = MongoDBStateRestorer(strategy_name=self._strategy_name, mongo_uri=self._mongo_uri)

        result = restorer.restore_state()

        assert isinstance(result, RestoredState)
        assert len(result.positions) == 0
        assert len(result.balances) == 0
        assert len(result.instrument_to_target_positions) == 0

    def _insert_test_data(self, mongo_client):
        db = mongo_client["default_logs_db"]

        now = datetime.now()
        log_timestamp = now - timedelta(days=1)
        db["qubx_logs_positions"].insert_one(
            {
                "timestamp": log_timestamp,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "pnl_quoted": 1,
                "quantity": 1,
                "realized_pnl_quoted": 1,
                "avg_position_price": 90000,
                "market_value_quoted": 0,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "positions",
            }
        )

        db["qubx_logs_signals"].insert_one(
            {
                "timestamp": log_timestamp,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "signal": 1,
                "reference_price": 90000,
                "service": False,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "signals",
            }
        )

        db["qubx_logs_balance"].insert_one(
            {
                "timestamp": log_timestamp,
                "exchange": "BINANCE.UM",
                "currency": "USDT",
                "total": 10000,
                "locked": 1000,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "balance",
            }
        )

        db["qubx_logs_targets"].insert_one(
            {
                "timestamp": log_timestamp,
                "symbol": "BTCUSDT",
                "exchange": "BINANCE.UM",
                "market_type": "SWAP",
                "target_position": 1,
                "entry_price": 90000,
                "run_id": "testing-1745335068910429952",
                "strategy_name": self._strategy_name,
                "log_type": "targets",
            }
        )

    def test_with_sample_data(self):
        mock_client = mongomock.MongoClient(self._mongo_uri)

        self._insert_test_data(mock_client)

        with patch("qubx.restorers.state.MongoClient", return_value=mock_client):
            restorer = MongoDBStateRestorer(strategy_name=self._strategy_name, mongo_uri=self._mongo_uri)

            result = restorer.restore_state()

            assert isinstance(result, RestoredState)
            assert len(result.positions) > 0
            assert len(result.balances) > 0
            assert len(result.instrument_to_target_positions) > 0

            # Convert to dict for easier testing
            balances = {b.currency: b for b in result.balances}
            assert "USDT" in balances
            assert balances["USDT"].total == 10000.0
            assert balances["USDT"].locked == 1000.0
            expected_free = balances["USDT"].total - balances["USDT"].locked
            assert balances["USDT"].free == expected_free

            btc_signals = []
            for instrument, signals_list in result.instrument_to_signal_positions.items():
                if instrument.symbol == "BTCUSDT":
                    btc_signals = signals_list
                    break

            assert len(btc_signals) > 0

            btc_targets = []
            for instrument, targets_list in result.instrument_to_target_positions.items():
                if instrument.symbol == "BTCUSDT":
                    btc_targets = targets_list
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


class TestMongoRunScoping:
    """Run-scoping for the MongoDB restorers: restore only the previous run's state,
    so a de-universed instrument's stale doc from an older run is never resurrected."""

    _strategy_name = "test_strategy"

    @staticmethod
    def _pos_doc(run_id, symbol, ts, strategy="test_strategy"):
        return {
            "timestamp": ts, "symbol": symbol, "exchange": "BINANCE.UM", "market_type": "SWAP",
            "quantity": 1, "realized_pnl_quoted": 0, "avg_position_price": 100.0,
            "run_id": run_id, "strategy_name": strategy, "log_type": "positions",
        }

    @staticmethod
    def _sig_doc(run_id, symbol, ts, strategy="test_strategy"):
        return {
            "timestamp": ts, "symbol": symbol, "exchange": "BINANCE.UM", "market_type": "SWAP",
            "signal": 1, "reference_price": 90000, "run_id": run_id,
            "strategy_name": strategy, "log_type": "signals",
        }

    @staticmethod
    def _bal_doc(run_id, currency, ts, strategy="test_strategy"):
        return {
            "timestamp": ts, "exchange": "BINANCE.UM", "currency": currency,
            "total": 1000.0, "locked": 0.0, "run_id": run_id,
            "strategy_name": strategy, "log_type": "balance",
        }

    # ---- utils helpers ----

    def test_mongo_latest_run_id_picks_most_recent(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._pos_doc("run-old", "ETHUSDT", now - timedelta(hours=3)))
        col.insert_one(self._pos_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        match = {"log_type": "positions", "strategy_name": self._strategy_name}
        assert mongo_latest_run_id(col, match) == "run-new"

    def test_mongo_latest_run_id_none_when_empty(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        match = {"log_type": "positions", "strategy_name": self._strategy_name}
        assert mongo_latest_run_id(col, match) is None

    def test_mongo_canonical_run_id_across_collections(self):
        client = mongomock.MongoClient()
        db = client["default_logs_db"]
        now = datetime.now()
        db["qubx_logs_positions"].insert_one(self._pos_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        db["qubx_logs_targets"].insert_one(
            {**self._pos_doc("run-old", "ETHUSDT", now - timedelta(hours=5)), "log_type": "targets"}
        )
        sources = [
            (db["qubx_logs_positions"], {"log_type": "positions", "strategy_name": self._strategy_name}),
            (db["qubx_logs_targets"], {"log_type": "targets", "strategy_name": self._strategy_name}),
        ]
        assert mongo_canonical_run_id(sources) == "run-new"

    def test_mongo_canonical_run_id_none_when_no_sources(self):
        assert mongo_canonical_run_id([]) is None

    # ---- position restorer ----

    def test_position_scopes_to_latest_run(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._pos_doc("run-old", "ETHUSDT", now - timedelta(hours=3)))
        col.insert_one(self._pos_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        restorer = MongoDBPositionRestorer(strategy_name=self._strategy_name, mongo_client=client)
        result = restorer.restore_positions()
        assert {i.symbol for i in result} == {"BTCUSDT"}

    def test_position_uses_injected_run_id(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._pos_doc("run-old", "ETHUSDT", now - timedelta(hours=3)))
        col.insert_one(self._pos_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        restorer = MongoDBPositionRestorer(strategy_name=self._strategy_name, mongo_client=client, run_id="run-old")
        result = restorer.restore_positions()
        assert {i.symbol for i in result} == {"ETHUSDT"}

    # ---- signal restorer (the core bug) ----

    def test_signal_scopes_to_latest_run(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._sig_doc("run-old", "ETHUSDT", now - timedelta(hours=3)))
        col.insert_one(self._sig_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        restorer = MongoDBSignalRestorer(strategy_name=self._strategy_name, mongo_client=client)
        result = restorer.restore_signals()
        assert {i.symbol for i in result} == {"BTCUSDT"}

    def test_signal_uses_injected_run_id(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._sig_doc("run-old", "ETHUSDT", now - timedelta(hours=3)))
        col.insert_one(self._sig_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        restorer = MongoDBSignalRestorer(strategy_name=self._strategy_name, mongo_client=client, run_id="run-old")
        result = restorer.restore_signals()
        assert {i.symbol for i in result} == {"ETHUSDT"}

    def test_signal_empty_when_no_run(self):
        client = mongomock.MongoClient()
        restorer = MongoDBSignalRestorer(strategy_name=self._strategy_name, mongo_client=client)
        assert restorer.restore_signals() == {}

    # ---- balance restorer ----

    def test_balance_scopes_to_latest_run(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._bal_doc("run-old", "BTC", now - timedelta(hours=3)))
        col.insert_one(self._bal_doc("run-new", "USDT", now - timedelta(hours=1)))
        restorer = MongoDBBalanceRestorer(strategy_name=self._strategy_name, mongo_client=client)
        result = restorer.restore_balances()
        assert {b.currency for b in result} == {"USDT"}

    def test_balance_uses_injected_run_id(self):
        client = mongomock.MongoClient()
        col = client["default_logs_db"]["qubx_logs"]
        now = datetime.now()
        col.insert_one(self._bal_doc("run-old", "BTC", now - timedelta(hours=3)))
        col.insert_one(self._bal_doc("run-new", "USDT", now - timedelta(hours=1)))
        restorer = MongoDBBalanceRestorer(strategy_name=self._strategy_name, mongo_client=client, run_id="run-old")
        result = restorer.restore_balances()
        assert {b.currency for b in result} == {"BTC"}

    # ---- state restorer: one shared canonical run (Option B) ----

    def test_state_flat_previous_run_restores_no_targets(self):
        """Canonical run (from positions) that logged no targets restores no targets,
        never an older run's — the flat-previous-run guarantee."""
        client = mongomock.MongoClient()
        db = client["default_logs_db"]
        now = datetime.now()
        db["qubx_logs_positions"].insert_one(self._pos_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        db["qubx_logs_balance"].insert_one(self._bal_doc("run-new", "USDT", now - timedelta(hours=1)))
        db["qubx_logs_targets"].insert_one(
            {
                "timestamp": now - timedelta(hours=5), "symbol": "ETHUSDT", "exchange": "BINANCE.UM",
                "market_type": "SWAP", "target_position": 1, "entry_price": 90000,
                "run_id": "run-old", "strategy_name": self._strategy_name, "log_type": "targets",
            }
        )
        with patch("qubx.restorers.state.MongoClient", return_value=client):
            restorer = MongoDBStateRestorer(strategy_name=self._strategy_name)
            state = restorer.restore_state()
        assert {i.symbol for i in state.positions} == {"BTCUSDT"}
        assert state.instrument_to_target_positions == {}

    def test_state_injects_canonical_run_into_sub_restorers(self):
        client = mongomock.MongoClient()
        db = client["default_logs_db"]
        now = datetime.now()
        db["qubx_logs_positions"].insert_one(self._pos_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        db["qubx_logs_signals"].insert_one(self._sig_doc("run-new", "BTCUSDT", now - timedelta(hours=1)))
        db["qubx_logs_balance"].insert_one(self._bal_doc("run-new", "USDT", now - timedelta(hours=1)))
        with patch("qubx.restorers.state.MongoClient", return_value=client):
            restorer = MongoDBStateRestorer(strategy_name=self._strategy_name)
            restorer.restore_state()
            assert restorer.position_restorer.run_id == "run-new"
            assert restorer.signal_restorer.run_id == "run-new"
            assert restorer.targets_restorer.run_id == "run-new"
            assert restorer.balance_restorer.run_id == "run-new"
