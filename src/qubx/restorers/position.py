"""
Position restorer implementations.

This module provides implementations of the IPositionRestorer interface
for restoring positions from various sources.
"""

import os
from pathlib import Path

import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument, Position
from qubx.core.lookups import lookup
from qubx.core.utils import recognize_time
from qubx.restorers.interfaces import IPositionRestorer
from qubx.restorers.utils import find_latest_run_folder


class CsvPositionRestorer(IPositionRestorer):
    """
    Position restorer that reads positions from CSV files.

    This restorer reads the most recent positions from CSV files generated
    by the CsvFileLogsWriter.
    """

    def __init__(
        self,
        base_dir: str | None = None,
        file_pattern: str = "*_positions.csv",
        strategy_name: str | None = None,
    ):
        """
        Initialize the CSV position restorer.

        Args:
            base_dir: The base directory where log folders are stored.
                If None, defaults to the current working directory.
            file_pattern: The pattern for position CSV filenames.
                Default is "*_positions.csv" which will match any strategy's positions file.
            strategy_name: Optional strategy name to filter files.
                If provided, only files matching the strategy name will be considered.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.file_pattern = file_pattern
        self.strategy_name = strategy_name

        # If strategy name is provided, update the file pattern
        if strategy_name:
            self.file_pattern = f"{strategy_name}*_positions.csv"

    def restore_positions(self) -> dict[Instrument, Position]:
        """
        Restore positions from the most recent run folder.

        Returns:
            A dictionary mapping instruments to positions.
        """
        # Find the latest run folder
        latest_run = find_latest_run_folder(self.base_dir)
        if not latest_run:
            return {}

        # Find position files in the latest run folder
        position_files = list(latest_run.glob(self.file_pattern))
        if not position_files:
            logger.warning(f"No position files matching '{self.file_pattern}' found in {latest_run}")
            return {}

        # Use the first matching file (or the only one if there's just one)
        file_path = position_files[0]

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Get the most recent positions
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")

            # Process the positions
            return self._restore_positions_from_df(df)

        except Exception as e:
            # Log the error and return an empty dictionary
            logger.error(f"Error restoring positions from {file_path}: {e}")
            return {}

    def _restore_positions_from_df(self, df: pd.DataFrame) -> dict[Instrument, Position]:
        """
        Process positions from a DataFrame.

        Args:
            df: The DataFrame containing position data.

        Returns:
            A dictionary mapping instruments to positions.
        """
        positions = {}

        # Group by symbol, exchange, and market_type to get the latest entry for each instrument
        latest_positions = df.groupby(["symbol", "exchange", "market_type"]).last().reset_index()

        for _, row in latest_positions.iterrows():
            # Get the instrument details
            symbol = row["symbol"]
            exchange = row["exchange"]

            # Create or find the instrument
            instrument = lookup.find_symbol(exchange, symbol)
            if instrument is None:
                logger.warning(f"Instrument not found for {symbol} on {exchange}")
                continue

            # Determine quantity and price column names
            quantity_col = "quantity" if "quantity" in row else "size"
            price_col = "avg_position_price" if "avg_position_price" in row else "avg_price"

            # Create a Position object
            position = Position(
                instrument=instrument,
                quantity=row[quantity_col],
                pos_average_price=row[price_col],
                r_pnl=row.get("realized_pnl_quoted", row.get("realized_pnl", 0.0)),
            )

            timestamp = recognize_time(row["timestamp"])
            current_price = row["current_price"]

            if current_price is not None:
                position.update_market_price(timestamp, current_price, 1.0)

            positions[instrument] = position

        return positions
