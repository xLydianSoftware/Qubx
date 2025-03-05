"""
Position restorer implementations.

This module provides implementations of the IPositionRestorer interface
for restoring positions from various sources.
"""

import os
from pathlib import Path

import pandas as pd

from qubx.core.basics import AssetType, Instrument, MarketType, Position
from qubx.restorers.interfaces import IPositionRestorer


class CsvPositionRestorer(IPositionRestorer):
    """
    Position restorer that reads positions from CSV files.

    This restorer reads the most recent positions from CSV files generated
    by the CsvFileLogsWriter.
    """

    def __init__(
        self,
        base_dir: str | None = None,
        file_pattern: str = "{strategy_id}_positions.csv",
    ):
        """
        Initialize the CSV position restorer.

        Args:
            base_dir: The base directory where position CSV files are stored.
                If None, defaults to the current working directory.
            file_pattern: The pattern for position CSV filenames.
                Should include a {strategy_id} placeholder.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.file_pattern = file_pattern

    def restore_positions(self, strategy_id: str) -> dict[Instrument, Position]:
        """
        Restore positions for a strategy from CSV files.

        Args:
            strategy_id: The ID of the strategy to restore positions for.

        Returns:
            A dictionary mapping instruments to positions.
        """
        file_path = self.base_dir / self.file_pattern.format(strategy_id=strategy_id)

        if not file_path.exists():
            return {}

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Get the most recent positions
            latest_positions = df.sort_values("timestamp").groupby("instrument").last().reset_index()

            # Convert to Position objects
            positions = {}
            for _, row in latest_positions.iterrows():
                # Parse the instrument string (format: "EXCHANGE:MARKET_TYPE:SYMBOL")
                instrument_parts = row["instrument"].split(":")
                if len(instrument_parts) != 3:
                    print(f"Invalid instrument format: {row['instrument']}")
                    continue

                exchange, market_type, symbol = instrument_parts

                # Create a simplified Instrument object
                # Note: This is a simplified version with default values for many fields
                instrument = Instrument(
                    symbol=symbol,
                    asset_type=AssetType.CRYPTO,  # Default to CRYPTO
                    market_type=MarketType(market_type),
                    exchange=exchange,
                    base=symbol.split("USD")[0] if "USD" in symbol else symbol,
                    quote="USD" if "USD" in symbol else "",
                    settle="USD" if "USD" in symbol else "",
                    exchange_symbol=symbol,
                    tick_size=0.01,  # Default value
                    lot_size=0.001,  # Default value
                    min_size=0.001,  # Default value
                )

                # Create a Position object
                position = Position(
                    instrument=instrument,
                    quantity=row["size"],
                    pos_average_price=row["avg_price"],
                    r_pnl=row.get("realized_pnl", 0.0),
                )

                positions[instrument] = position

            return positions

        except Exception as e:
            # Log the error and return an empty dictionary
            print(f"Error restoring positions from {file_path}: {e}")
            return {}
