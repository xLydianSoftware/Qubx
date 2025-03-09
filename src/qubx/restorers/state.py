"""
State restorer implementations.

This module provides implementations for restoring the complete strategy state
from various sources.
"""

import os
from pathlib import Path

import numpy as np

from qubx import logger
from qubx.core.basics import RestoredState
from qubx.core.utils import recognize_time
from qubx.restorers.balance import CsvBalanceRestorer
from qubx.restorers.interfaces import IStateRestorer
from qubx.restorers.position import CsvPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer
from qubx.restorers.utils import find_latest_run_folder


class CsvStateRestorer(IStateRestorer):
    """
    State restorer that reads strategy state from CSV files.

    This restorer combines the functionality of CsvPositionRestorer, CsvSignalRestorer,
    and CsvBalanceRestorer to create a complete RestartState.
    """

    def __init__(
        self,
        base_dir: str | None = None,
        strategy_name: str | None = None,
        position_file_pattern: str = "*_positions.csv",
        signal_file_pattern: str = "*_signals.csv",
        balance_file_pattern: str = "*_balance.csv",
        lookback_days: int = 30,
    ):
        """
        Initialize the CSV state restorer.

        Args:
            base_dir: The base directory where log folders are stored.
                If None, defaults to the current working directory.
            strategy_name: Optional strategy name to filter files.
                If provided, only files matching the strategy name will be considered.
            position_file_pattern: The pattern for position CSV filenames.
            signal_file_pattern: The pattern for signal CSV filenames.
            balance_file_pattern: The pattern for balance CSV filenames.
            lookback_days: The number of days to look back for signals.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.strategy_name = strategy_name

        # Create individual restorers
        self.position_restorer = CsvPositionRestorer(
            base_dir=base_dir,
            file_pattern=position_file_pattern,
            strategy_name=strategy_name,
        )

        self.signal_restorer = CsvSignalRestorer(
            base_dir=base_dir,
            file_pattern=signal_file_pattern,
            strategy_name=strategy_name,
            lookback_days=lookback_days,
        )

        self.balance_restorer = CsvBalanceRestorer(
            base_dir=base_dir,
            file_pattern=balance_file_pattern,
            strategy_name=strategy_name,
        )

    def restore_state(self) -> RestoredState:
        """
        Restore the complete strategy state from CSV files.

        Returns:
            A RestoredState object containing positions, target positions, and balances.
        """
        # Find the latest run folder
        latest_run = find_latest_run_folder(self.base_dir)
        if not latest_run:
            logger.warning(f"No run folders found in {self.base_dir}")
            return RestoredState(
                time=np.datetime64("now"),
                positions={},
                instrument_to_target_positions={},
                balances={},
            )

        # Restore positions, target positions, and balances
        positions = self.position_restorer.restore_positions()
        target_positions = self.signal_restorer.restore_signals()
        balances = self.balance_restorer.restore_balances()

        # Get latest position timestamp
        latest_position_timestamp = (
            max(position.last_update_time for position in positions.values()) if positions else np.datetime64("now")
        )
        if np.isnan(latest_position_timestamp):
            latest_position_timestamp = np.datetime64("now")

        # Create and return the restored state
        return RestoredState(
            time=recognize_time(latest_position_timestamp),
            positions=positions,
            instrument_to_target_positions=target_positions,
            balances=balances,
        )
