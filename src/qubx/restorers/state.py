"""
State restorer implementations.

This module provides implementations for restoring the complete strategy state
from various sources.
"""

import os
from pathlib import Path

import numpy as np
from pymongo import MongoClient

from qubx import logger
from qubx.core.basics import RestoredState
from qubx.core.utils import recognize_time
from qubx.restorers.balance import CsvBalanceRestorer, MongoDBBalanceRestorer
from qubx.restorers.interfaces import IStateRestorer
from qubx.restorers.position import CsvPositionRestorer, MongoDBPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer, MongoDBSignalRestorer
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
        targets_file_pattern: str = "*_targets.csv",
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
            targets_file_pattern: The pattern for target CSV filenames.
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

        self.signal_targets_restorer = CsvSignalRestorer(
            base_dir=base_dir,
            signals_file_pattern=signal_file_pattern,
            targets_file_pattern=targets_file_pattern,
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
                instrument_to_signal_positions={},
                instrument_to_target_positions={},
                balances={},
            )

        logger.info(f"Restoring state from {latest_run}")

        # Restore positions, target positions, and balances
        positions = self.position_restorer.restore_positions()
        signals = self.signal_targets_restorer.restore_signals()
        targets = self.signal_targets_restorer.restore_targets()
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
            instrument_to_signal_positions=signals,
            instrument_to_target_positions=targets,
            balances=balances,
        )


class MongoDBStateRestorer(IStateRestorer):
    """
    State restorer that reads strategy state from MongoDB.

    This restorer combines the functionality of MongoDBPositionRestorer,
    MongoDBSignalRestorer, and MongoDBBalanceRestorer to create a complete RestartState.
    """

    def __init__(
        self,
        strategy_name: str,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "default_logs_db",
        collection_name_prefix: str = "qubx_logs",
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name_prefix = collection_name_prefix
        self.strategy_name = strategy_name

        self.client = MongoClient(mongo_uri)

        # Create individual restorers
        self.position_restorer = MongoDBPositionRestorer(
            strategy_name=strategy_name,
            mongo_client=self.client,
            db_name=db_name,
            collection_name=f"{collection_name_prefix}_positions",
        )

        self.signal_restorer = MongoDBSignalRestorer(
            strategy_name=strategy_name,
            mongo_client=self.client,
            db_name=db_name,
            collection_name=f"{collection_name_prefix}_signals",
        )

        self.targets_restorer = MongoDBSignalRestorer(
            strategy_name=strategy_name,
            mongo_client=self.client,
            db_name=db_name,
            collection_name=f"{collection_name_prefix}_targets",
        )

        self.balance_restorer = MongoDBBalanceRestorer(
            strategy_name=strategy_name,
            mongo_client=self.client,
            db_name=db_name,
            collection_name=f"{collection_name_prefix}_balance",
        )

    def restore_state(self) -> RestoredState:
        """
        Restore the complete strategy state from MongoDB.

        Returns:
            A RestoredState object containing positions, target positions, and balances.
        """
        mongo_collections = self.client[self.db_name].list_collection_names()
        required_suffixes = ["positions", "signals", "balance"]

        if not any(f"{self.collection_name_prefix}_{suffix}" in mongo_collections for suffix in required_suffixes):
            logger.warning(f"No logs collections found in MongodDB {self.db_name}.")
            self.client.close()
            return RestoredState(
                time=np.datetime64("now"),
                positions={},
                instrument_to_signal_positions={},
                instrument_to_target_positions={},
                balances={},
            )

        logger.info(f"Restoring state from MongoDB {self.db_name}")

        positions = self.position_restorer.restore_positions()
        signals = self.signal_restorer.restore_signals()
        targets = self.targets_restorer.restore_targets()
        balances = self.balance_restorer.restore_balances()

        latest_position_timestamp = (
            max(position.last_update_time for position in positions.values()) if positions else np.datetime64("now")
        )
        if np.isnan(latest_position_timestamp):
            latest_position_timestamp = np.datetime64("now")

        self.client.close()
        return RestoredState(
            time=recognize_time(latest_position_timestamp),
            positions=positions,
            instrument_to_signal_positions=signals,
            instrument_to_target_positions=targets,
            balances=balances,
        )
