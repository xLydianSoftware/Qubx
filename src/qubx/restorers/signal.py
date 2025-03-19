"""
Signal restorer implementations.

This module provides implementations of the ISignalRestorer interface
for restoring signals from various sources.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

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
        file_pattern: str = "*_signals.csv",
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
        self.file_pattern = file_pattern
        self.lookback_days = lookback_days
        self.strategy_name = strategy_name

        # If strategy name is provided, update the file pattern
        if strategy_name:
            self.file_pattern = f"{strategy_name}*_signals.csv"

    def restore_signals(self) -> dict[Instrument, list[TargetPosition]]:
        """
        Restore signals from the most recent run folder.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        # Find the latest run folder
        latest_run = find_latest_run_folder(self.base_dir)
        if not latest_run:
            return {}

        # Find signal files in the latest run folder
        signal_files = list(latest_run.glob(self.file_pattern))
        if not signal_files:
            logger.warning(f"No signal files matching '{self.file_pattern}' found in {latest_run}")
            return {}

        # Use the first matching file (or the only one if there's just one)
        file_path = signal_files[0]

        try:
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.info(f"Could not read signal file {file_path}: {e}")
                return {}

            if df.empty:
                logger.info(f"No signals found in {file_path}")
                return {}

            # Filter signals from the lookback period
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent_signals = df[df["timestamp"] >= cutoff_date]

            # Process the signals
            return self._restore_signals_from_df(recent_signals)

        except Exception as e:
            # Log the error and return an empty dictionary
            logger.error(f"Error restoring signals from {file_path}: {e}")
            return {}

    def _restore_signals_from_df(self, df: pd.DataFrame) -> dict[Instrument, list[TargetPosition]]:
        """
        Process signals from a DataFrame.

        Args:
            df: The DataFrame containing signal data.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        targets_by_instrument = {}

        # Group by symbol, exchange, and market_type
        for (symbol, exchange, market_type_str), group in df.groupby(["symbol", "exchange", "market_type"]):
            # Create or find the instrument
            instrument = lookup.find_symbol(exchange, symbol)
            if instrument is None:
                logger.warning(f"Instrument not found for {symbol} on {exchange}")
                continue

            target_positions = []

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
                for key in ["target_position", "comment", "size", "meta"]:
                    if key in row and not pd.isna(row[key]):
                        options[key] = row[key]

                # Determine price
                price = None
                for price_col in ["price", "reference_price"]:
                    if price_col in row and pd.notna(row[price_col]):
                        price = row[price_col]
                        break

                timestamp = recognize_time(row["timestamp"])
                target_size = row["target_position"]

                target_positions.append(
                    TargetPosition(
                        time=timestamp,
                        target_position_size=target_size,
                        signal=Signal(
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
                        ),
                    )
                )

            targets_by_instrument[instrument] = target_positions

        return targets_by_instrument
