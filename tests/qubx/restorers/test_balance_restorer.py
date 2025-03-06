"""
Tests for the balance restorer.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from qubx.restorers.balance import CsvBalanceRestorer


def test_csv_balance_restorer_with_sample_data():
    """Test the CsvBalanceRestorer with sample data."""
    # Create a temporary directory structure for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a run folder
        run_folder = Path(temp_dir) / "run_20250306093316"
        run_folder.mkdir()

        # Create a test CSV file
        file_path = run_folder / "test_strategy_balance.csv"

        # Create test data with the new format
        data = {
            "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:30:00", "2023-01-01 13:00:00"],
            "currency": ["USDT", "BTC", "USDT"],
            "total": [100000.0, 1.5, 99000.0],
            "locked": [0.0, 0.0, 1000.0],
            "run_id": ["test-123", "test-123", "test-123"],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        # Create the restorer
        restorer = CsvBalanceRestorer(base_dir=temp_dir, file_pattern="*_balance.csv")

        # Restore balances
        balances = restorer.restore_balances()

        # Check the results
        assert len(balances) == 2

        # Check USDT balance (should be the latest entry)
        assert "USDT" in balances
        assert balances["USDT"]["total"] == 99000.0
        assert balances["USDT"]["locked"] == 1000.0

        # Check BTC balance
        assert "BTC" in balances
        assert balances["BTC"]["total"] == 1.5
        assert balances["BTC"]["locked"] == 0.0


def test_csv_balance_restorer_with_real_data():
    """Test the CsvBalanceRestorer with real log data."""
    # Path to the real log data
    log_dir = Path("tests/data/logs")

    # Create the restorer
    restorer = CsvBalanceRestorer(base_dir=str(log_dir), file_pattern="*_balance.csv")

    # Restore balances
    balances = restorer.restore_balances()

    # Check the results
    assert len(balances) > 0

    # Check that we have USDT balance
    assert "USDT" in balances
    assert isinstance(balances["USDT"]["total"], float)
    assert isinstance(balances["USDT"]["locked"], float)
    assert balances["USDT"]["total"] > 0
