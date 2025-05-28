"""
Balance restorer implementations.

This module provides implementations for restoring account balances
from various sources.
"""

import os
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime, timedelta

import pandas as pd

from qubx import logger
from qubx.core.basics import AssetBalance
from qubx.restorers.interfaces import IBalanceRestorer
from qubx.restorers.utils import find_latest_run_folder


class CsvBalanceRestorer(IBalanceRestorer):
    """
    Balance restorer that reads account balances from CSV files.

    This restorer reads the most recent account balances from CSV files generated
    by the CsvFileLogsWriter.
    """

    def __init__(
        self,
        base_dir: str | None = None,
        file_pattern: str = "*_balance.csv",
        strategy_name: str | None = None,
    ):
        """
        Initialize the CSV balance restorer.

        Args:
            base_dir: The base directory where log folders are stored.
                If None, defaults to the current working directory.
            file_pattern: The pattern for balance CSV filenames.
                Default is "*_balance.csv" which will match any strategy's balance file.
            strategy_name: Optional strategy name to filter files.
                If provided, only files matching the strategy name will be considered.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.file_pattern = file_pattern
        self.strategy_name = strategy_name

        # If strategy name is provided, update the file pattern
        if strategy_name:
            self.file_pattern = f"{strategy_name}*_balance.csv"

    def restore_balances(self) -> dict[str, AssetBalance]:
        """
        Restore account balances from the most recent run folder.

        Returns:
            A dictionary mapping currency codes to AssetBalance objects.
            Example: {'USDT': AssetBalance(total=100000.0, locked=0.0)}
        """
        # Find the latest run folder
        latest_run = find_latest_run_folder(self.base_dir)
        if not latest_run:
            print(f"No run folders found in {self.base_dir}")
            return {}

        # Find balance files in the latest run folder
        balance_files = list(latest_run.glob(self.file_pattern))
        if not balance_files:
            print(f"No balance files matching '{self.file_pattern}' found in {latest_run}")
            return {}

        # Use the first matching file (or the only one if there's just one)
        file_path = balance_files[0]

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Get the most recent balances
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")

            # Process the balances
            return self._restore_balances_from_df(df)

        except Exception as e:
            # Log the error and return an empty dictionary
            logger.error(f"Error restoring balances from {file_path}: {e}")
            return {}

    def _restore_balances_from_df(self, df: pd.DataFrame) -> dict[str, AssetBalance]:
        """
        Process balances from a DataFrame.

        Args:
            df: The DataFrame containing balance data.

        Returns:
            A dictionary mapping currency codes to dictionaries with 'total' and 'locked' amounts.
        """
        balances = {}

        # Group by currency to get the latest entry for each
        latest_balances = df.groupby("currency").last().reset_index()

        for _, row in latest_balances.iterrows():
            currency = row["currency"]
            total = row["total"]
            locked = row["locked"]

            # Create a balance entry
            balance = AssetBalance(
                total=total,
                locked=locked,
            )
            # Calculate the free balance
            balance.free = total - locked
            balances[currency] = balance

        return balances


class MongoDBBalanceRestorer(IBalanceRestorer):
    """
    Balance restorer that reads account balances from a MongoDB collection.

    This restorer queries the most recent balance entries stored using MongoDBLogsWriter.
    It restores data only from the most recent run_id for the given bot_id.
    """

    def __init__(
        self,
        strategy_name: str,
        mongo_client: MongoClient,
        db_name: str = "default_logs_db",
        collection_name: str = "qubx_logs",
    ):
        self.mongo_client = mongo_client
        self.db_name = db_name
        self.collection_name = collection_name
        self.strategy_name = strategy_name

        self.collection = self.mongo_client[db_name][collection_name]

    def restore_balances(self) -> dict[str, AssetBalance]:
        """
        Restore account balances from the most recent run.

        Returns:
            A dictionary mapping currency codes to AssetBalance objects.
            Example: {'USDT': AssetBalance(total=100000.0, locked=0.0)}
        """
        try:
            now = datetime.utcnow()
            lookup_range = now - timedelta(days=7)
            base_match = {
                "log_type": "balance",
                "strategy_name": self.strategy_name,
                "timestamp": {"$gte": lookup_range}
            }

            latest_run_doc = (
                self.collection.find(base_match, {"run_id": 1, "timestamp": 1})
                .sort("timestamp", -1)
                .limit(1)
            )

            latest_run = next(latest_run_doc, None)
            if not latest_run:
                logger.warning("No balance logs found for given filters.")
                return {}

            latest_run_id = latest_run["run_id"]

            logger.info(f"Restoring balances from MongoDB for run_id: {latest_run_id}")

            pipeline = [
                {"$match": {**base_match, "run_id": latest_run_id}},
                {"$sort": {"timestamp": -1}},
                {
                    "$group": {
                        "_id": "$currency",
                        "doc": {"$first": "$$ROOT"}
                    }
                }
            ]

            cursor = self.collection.aggregate(pipeline)
            balances: dict[str, AssetBalance] = {}

            for entry in cursor:
                log = entry["doc"]
                currency = log.get("currency")
                if not currency:
                    continue
                total = log.get("total", 0.0)
                locked = log.get("locked", 0.0)

                balance = AssetBalance(
                    total=total,
                    locked=locked,
                )
                balance.free = total - locked
                balances[currency] = balance

            return balances
        except Exception as e:
            logger.error(f"Error restoring balances from MongoDB: {e}")
            return {}