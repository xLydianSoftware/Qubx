"""
Tests for the restorers module.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, Position, Signal
from qubx.restorers.factory import create_position_restorer, create_signal_restorer
from qubx.restorers.interfaces import IPositionRestorer, ISignalRestorer, RestartState
from qubx.restorers.position import CsvPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer


def test_protocol_implementations():
    """Test that the implementations satisfy the protocols."""
    # Check that CsvPositionRestorer implements IPositionRestorer
    assert isinstance(CsvPositionRestorer(), IPositionRestorer)

    # Check that CsvSignalRestorer implements ISignalRestorer
    assert isinstance(CsvSignalRestorer(), ISignalRestorer)

    # Check that the factory functions return objects that implement the protocols
    position_restorer = create_position_restorer("CsvPositionRestorer")
    assert isinstance(position_restorer, IPositionRestorer)

    signal_restorer = create_signal_restorer("CsvSignalRestorer")
    assert isinstance(signal_restorer, ISignalRestorer)


def test_csv_position_restorer():
    """Test the CsvPositionRestorer."""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test CSV file
        strategy_id = "test_strategy"
        file_path = Path(temp_dir) / f"{strategy_id}_positions.csv"

        # Create test data
        data = {
            "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:30:00", "2023-01-01 13:00:00"],
            "instrument": ["BINANCE:SPOT:BTCUSDT", "BINANCE:SPOT:BTCUSDT", "BINANCE:SPOT:ETHUSDT"],
            "size": [1.0, 2.0, 3.0],
            "avg_price": [50000.0, 51000.0, 3000.0],
            "unrealized_pnl": [100.0, 200.0, 300.0],
            "realized_pnl": [50.0, 60.0, 70.0],
            "liquidation_price": [40000.0, 41000.0, 2000.0],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        # Create the restorer
        restorer = create_position_restorer(
            "CsvPositionRestorer", {"base_dir": temp_dir, "file_pattern": "{strategy_id}_positions.csv"}
        )

        # Restore positions
        positions = restorer.restore_positions(strategy_id)

        # Check the results
        assert len(positions) == 2

        # Create expected instruments for comparison
        btc_instrument = Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="BTC",
            quote="USD",
            settle="USD",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        eth_instrument = Instrument(
            symbol="ETHUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ETH",
            quote="USD",
            settle="USD",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

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


def test_csv_signal_restorer():
    """Test the CsvSignalRestorer."""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test CSV file
        strategy_id = "test_strategy"
        file_path = Path(temp_dir) / f"{strategy_id}_signals.csv"

        # Create test data with recent timestamps
        now = pd.Timestamp.now()
        yesterday = now - pd.Timedelta(days=1)
        two_days_ago = now - pd.Timedelta(days=2)

        data = {
            "timestamp": [
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                yesterday.strftime("%Y-%m-%d %H:%M:%S"),
                two_days_ago.strftime("%Y-%m-%d %H:%M:%S"),
            ],
            "instrument": ["BINANCE:SPOT:BTCUSDT", "BINANCE:SPOT:ETHUSDT", "BINANCE:SPOT:BTCUSDT"],
            "side": ["buy", "sell", "buy"],
            "price": [50000.0, 3000.0, 49000.0],
            "size": [1.0, 2.0, 0.5],
            "meta": ["{}", "{}", "{}"],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        # Create the restorer
        restorer = create_signal_restorer(
            "CsvSignalRestorer",
            {
                "base_dir": temp_dir,
                "file_pattern": "{strategy_id}_signals.csv",
                "lookback_days": 7,
            },
        )

        # Restore signals
        signals = restorer.restore_signals(strategy_id)

        # Check the results
        assert len(signals) == 2

        # Create expected instruments for comparison
        btc_instrument = Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="BTC",
            quote="USD",
            settle="USD",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        eth_instrument = Instrument(
            symbol="ETHUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ETH",
            quote="USD",
            settle="USD",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

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


def test_restart_state():
    """Test the RestartState dataclass."""
    # Create test data
    time = np.datetime64("2023-01-01T12:00:00")

    # Create instruments
    btc_instrument = Instrument(
        symbol="BTCUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SPOT,
        exchange="BINANCE",
        base="BTC",
        quote="USD",
        settle="USD",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )

    eth_instrument = Instrument(
        symbol="ETHUSDT",
        asset_type=AssetType.CRYPTO,
        market_type=MarketType.SPOT,
        exchange="BINANCE",
        base="ETH",
        quote="USD",
        settle="USD",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )

    # Create positions
    btc_position = Position(
        instrument=btc_instrument,
        quantity=1.0,
        pos_average_price=50000.0,
    )

    eth_position = Position(
        instrument=eth_instrument,
        quantity=-2.0,
        pos_average_price=3000.0,
    )

    # Create signals
    btc_signal = Signal(
        instrument=btc_instrument,
        signal=1.0,  # Buy
        price=49000.0,
    )

    eth_signal = Signal(
        instrument=eth_instrument,
        signal=-1.0,  # Sell
        price=3100.0,
    )

    # Create the restart state
    restart_state = RestartState(
        time=time,
        instrument_to_signals={
            btc_instrument: [btc_signal],
            eth_instrument: [eth_signal],
        },
        positions={
            btc_instrument: btc_position,
            eth_instrument: eth_position,
        },
    )

    # Check the restart state
    assert restart_state.time == time
    assert len(restart_state.instrument_to_signals) == 2
    assert len(restart_state.positions) == 2
    assert restart_state.instrument_to_signals[btc_instrument][0] == btc_signal
    assert restart_state.instrument_to_signals[eth_instrument][0] == eth_signal
    assert restart_state.positions[btc_instrument] == btc_position
    assert restart_state.positions[eth_instrument] == eth_position
