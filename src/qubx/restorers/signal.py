"""
Signal restorer implementations.

This module provides implementations of the ISignalRestorer interface
for restoring signals from various sources.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.command_cursor import CommandCursor

from qubx import logger
from qubx.core.basics import Instrument, Signal, TargetPosition
from qubx.core.lookups import lookup
from qubx.core.utils import recognize_time
from qubx.restorers.interfaces import ISignalRestorer
from qubx.restorers.utils import find_latest_run_folder


class CsvSignalRestorer(ISignalRestorer):
    """
    Signal restorer that reads signals from CSV files.

    This restorer reads historical signals from CSV files generated
    by the CsvFileLogsWriter.
    """

    def __init__(
        self,
        base_dir: str | None = None,
        signals_file_pattern: str = "*_signals.csv",
        targets_file_pattern: str = "*_targets.csv",
        lookback_days: int = 30,
        strategy_name: str | None = None,
    ):
        """
        Initialize the CSV signal restorer.

        Args:
            base_dir: The base directory where log folders are stored.
                If None, defaults to the current working directory.
            file_pattern: The pattern for signal CSV filenames.
                Default is "*_signals.csv" which will match any strategy's signals file.
            lookback_days: The number of days to look back for signals.
            strategy_name: Optional strategy name to filter files.
                If provided, only files matching the strategy name will be considered.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.signals_file_pattern = signals_file_pattern
        self.targets_file_pattern = targets_file_pattern
        self.lookback_days = lookback_days
        self.strategy_name = strategy_name

        # If strategy name is provided, update the file pattern
        if strategy_name:
            self.signals_file_pattern = f"{strategy_name}*_signals.csv"
            self.targets_file_pattern = f"{strategy_name}*_targets.csv"

    def _load_records_from_file_as_dataframe(self, f_type: str, file_pattern: str) -> pd.DataFrame | None:
        latest_run = find_latest_run_folder(self.base_dir)
        if not latest_run:
            return None

        # Find signal files in the latest run folder
        csv_files = list(latest_run.glob(file_pattern))
        if not csv_files:
            logger.warning(f"No {f_type} files matching '{file_pattern}' found in {latest_run}")
            return None

        # Use the first matching file (or the only one if there's just one)
        file_path = csv_files[0]

        try:
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.info(f"Could not read {f_type} file {file_path}: {e}")
                return None

            if df.empty:
                logger.info(f"No {f_type} found in {file_path}")
                return None

            # Filter signals from the lookback period
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent_records = df[df["timestamp"] >= cutoff_date]

            # return recent records as dataframe
            return recent_records

        except Exception as e:
            logger.error(f"Error restoring {f_type} from {file_path}: {e}")
            return None

    def restore_signals(self) -> dict[Instrument, list[Signal]]:
        """
        Restore signals from the most recent run folder.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        if (signals_df := self._load_records_from_file_as_dataframe("signals", self.signals_file_pattern)) is None:
            return {}
        return self._restore_signals_from_df(signals_df)

    def restore_targets(self) -> dict[Instrument, list[TargetPosition]]:
        """
        Restore targets from the most recent run folder.

        Returns:
            A dictionary mapping instruments to lists of targets.
        """
        if (signals_df := self._load_records_from_file_as_dataframe("targets", self.targets_file_pattern)) is None:
            return {}
        return self._restore_targets_from_df(signals_df)

    def _restore_targets_from_df(self, df: pd.DataFrame) -> dict[Instrument, list[TargetPosition]]:
        _targets_by_instrument = {}

        # Group by symbol, exchange, and market_type
        for (symbol, exchange, market_type_str), group in df.groupby(["symbol", "exchange", "market_type"]):
            # Create or find the instrument
            if (instrument := lookup.find_symbol(exchange, symbol)) is None:
                logger.warning(f"Instrument not found for {symbol} on {exchange}")
                continue

            # Create a list of target positions
            targets = []

            for _, row in group.iterrows():
                # Determine target position size
                target_size = float(row["target_position"])

                # Create options dictionary with additional data
                options = {}

                for key in ["comment", "meta"]:
                    if key in row and not pd.isna(row[key]):
                        options[key] = row[key]

                # Determine price
                price = None
                for price_col in ["entry_price"]:
                    if price_col in row and pd.notna(row[price_col]):
                        price = row[price_col]
                        break

                targets.append(
                    TargetPosition(
                        time=recognize_time(row["timestamp"]),
                        target_position_size=target_size,
                        instrument=instrument,
                        entry_price=price,
                        stop_price=row.get("stop_price", None),
                        take_price=row.get("take_price", None),
                        options=options,
                    )
                )

            _targets_by_instrument[instrument] = targets
        return _targets_by_instrument

    def _restore_signals_from_df(self, df: pd.DataFrame) -> dict[Instrument, list[Signal]]:
        """
        Process signals from a DataFrame.

        Args:
            df: The DataFrame containing signal data.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        _signals_by_instrument = {}

        # Group by symbol, exchange, and market_type
        for (symbol, exchange, market_type_str), group in df.groupby(["symbol", "exchange", "market_type"]):
            # Create or find the instrument
            if (instrument := lookup.find_symbol(exchange, symbol)) is None:
                logger.warning(f"Instrument not found for {symbol} on {exchange}")
                continue

            signals = []

            for _, row in group.iterrows():
                # Determine signal value
                if "signal" in row:
                    signal_value = float(row["signal"])
                elif "side" in row:
                    # Convert 'buy'/'sell' to +1/-1
                    side_str = str(row["side"]).lower()
                    signal_value = 1.0 if side_str == "buy" else -1.0
                else:
                    logger.warning(f"Warning: No signal or side column found for {symbol}")
                    continue

                # Create options dictionary with additional data
                options = {}

                # for key in ["target_position", "comment", "size", "meta"]:
                for key in ["comment", "size", "meta"]:
                    if key in row and not pd.isna(row[key]):
                        options[key] = row[key]

                # Determine price
                price = None
                for price_col in ["price", "reference_price"]:
                    if price_col in row and pd.notna(row[price_col]):
                        price = row[price_col]
                        break

                signals.append(
                    Signal(
                        time=recognize_time(row["timestamp"]),
                        instrument=instrument,
                        signal=signal_value,
                        price=price,
                        stop=None,
                        take=None,
                        reference_price=row.get("reference_price", None)
                        if pd.notna(row.get("reference_price", None))
                        else None,
                        group=row.get("group", "") if pd.notna(row.get("group", "")) else "",
                        comment=row.get("comment", "") if pd.notna(row.get("comment", "")) else "",
                        options=options,
                    )
                )

            _signals_by_instrument[instrument] = signals

        return _signals_by_instrument


class MongoDBSignalRestorer(ISignalRestorer):
    """
    Signal restorer that reads historical signals from MongoDB.

    This restorer reads signals written by the MongoDBLogsWriter
    for the most recent run_id associated with a given bot.
    """

    def __init__(
        self,
        strategy_name: str,
        mongo_client: MongoClient,
        db_name: str = "default_logs_db",
        collection_name: str = "qubx_logs",
        max_restored_records: int = 20,
    ):
        self.mongo_client = mongo_client
        self.db_name = db_name
        self.collection_name = collection_name
        self.strategy_name = strategy_name
        self.max_restored_records = max_restored_records
        self.collection = self.mongo_client[db_name][collection_name]

    def restore_signals(self) -> dict[Instrument, list[Signal]]:
        """
        Restore signals from MongoDB for the latest run_id.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """

        logger.info(f"Restoring latest {self.max_restored_records} signals per symbol from MongoDB")
        result: dict[Instrument, list[Signal]] = {}

        if (cursor := self._load_data_from_mongo("signals")) is None:
            return result

        for entry in cursor:
            log = entry["data"]
            try:
                if (instrument := lookup.find_symbol(log["exchange"], log["symbol"])) is None:
                    logger.warning(f"Instrument not found for {log['symbol']} on {log['exchange']}")
                    continue

                signal_value = log.get("signal")
                if signal_value is None and "side" in log:
                    signal_value = 1.0 if str(log["side"]).lower() == "buy" else -1.0

                if signal_value is None:
                    logger.warning(f"Missing signal or side for log: {log}")
                    continue

                price = log.get("price") or log.get("reference_price")
                options = {key: log[key] for key in ["comment", "size", "meta"] if key in log}

                signal = Signal(
                    time=recognize_time(log["timestamp"]),
                    instrument=instrument,
                    signal=signal_value,
                    price=price,
                    stop=None,
                    take=None,
                    reference_price=log.get("reference_price"),
                    group=log.get("group", ""),
                    comment=log.get("comment", ""),
                    options=options,
                    is_service=log.get("service", log.get("is_service", False)),
                )

                result.setdefault(instrument, []).append(signal)
            except Exception as e:
                logger.exception(f"Failed to process signal document: {e}")

        cursor.close()
        return result

    def restore_targets(self) -> dict[Instrument, list[TargetPosition]]:
        logger.info(f"Restoring latest {self.max_restored_records} targets per symbol from MongoDB")
        result: dict[Instrument, list[TargetPosition]] = {}

        if (cursor := self._load_data_from_mongo("targets")) is None:
            return result

        for entry in cursor:
            log = entry["data"]
            try:
                if (instrument := lookup.find_symbol(log["exchange"], log["symbol"])) is None:
                    logger.warning(f"Instrument not found for {log['symbol']} on {log['exchange']}")
                    continue

                target_size = float(log["target_position"])
                price = log.get("entry_price", None)
                options = {key: log[key] for key in ["comment", "size", "meta"] if key in log}

                target = TargetPosition(
                    time=recognize_time(log["timestamp"]),
                    instrument=instrument,
                    target_position_size=target_size,
                    entry_price=price,
                    stop_price=log.get("stop_price", None),
                    take_price=log.get("take_price", None),
                    options=options,
                )

                result.setdefault(instrument, []).append(target)
            except Exception as e:
                logger.exception(f"Failed to process target document: {e}")

        cursor.close()
        return result

    def _load_data_from_mongo(self, log_type: str) -> CommandCursor | None:
        try:
            logger.info(f"Restoring latest {self.max_restored_records} signals per symbol from MongoDB")

            pipeline = [
                {"$match": {"log_type": log_type, "strategy_name": self.strategy_name}},
                {"$sort": {"timestamp": -1}},
                {
                    "$group": {
                        "_id": {
                            "symbol": "$symbol",
                            "exchange": "$exchange",
                            "market_type": "$market_type",
                        },
                        "data": {"$push": "$$ROOT"},
                    }
                },
                {"$project": {"data": {"$slice": ["$data", self.max_restored_records]}}},
                {"$unwind": "$data"},
            ]

            return self.collection.aggregate(pipeline)
        except Exception as e:
            logger.error(
                f"Error restoring {log_type} data from MongoDB::{self.collection_name} for {self.strategy_name} : {e}"
            )
            return None
