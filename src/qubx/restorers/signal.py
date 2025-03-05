"""
Signal restorer implementations.

This module provides implementations of the ISignalRestorer interface
for restoring signals from various sources.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from qubx.core.basics import AssetType, Instrument, MarketType, Signal
from qubx.restorers.interfaces import ISignalRestorer


class CsvSignalRestorer(ISignalRestorer):
    """
    Signal restorer that reads signals from CSV files.

    This restorer reads historical signals from CSV files generated
    by the CsvFileLogsWriter.
    """

    def __init__(
        self,
        base_dir: str | None = None,
        file_pattern: str = "{strategy_id}_signals.csv",
        lookback_days: int = 30,
    ):
        """
        Initialize the CSV signal restorer.

        Args:
            base_dir: The base directory where signal CSV files are stored.
                If None, defaults to the current working directory.
            file_pattern: The pattern for signal CSV filenames.
                Should include a {strategy_id} placeholder.
            lookback_days: The number of days to look back for signals.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.file_pattern = file_pattern
        self.lookback_days = lookback_days

    def restore_signals(self, strategy_id: str) -> dict[Instrument, list[Signal]]:
        """
        Restore signals for a strategy from CSV files.

        Args:
            strategy_id: The ID of the strategy to restore signals for.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        file_path = self.base_dir / self.file_pattern.format(strategy_id=strategy_id)

        if not file_path.exists():
            return {}

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Filter signals from the lookback period
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            recent_signals = df[df["timestamp"] >= cutoff_date]

            # Group by instrument
            signals_by_instrument = {}
            for instrument_str, group in recent_signals.groupby("instrument"):
                # Parse the instrument string (format: "EXCHANGE:MARKET_TYPE:SYMBOL")
                instrument_parts = instrument_str.split(":")
                if len(instrument_parts) != 3:
                    print(f"Invalid instrument format: {instrument_str}")
                    continue

                exchange, market_type, symbol = instrument_parts

                # Create a simplified Instrument object
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

                signals = []

                for _, row in group.iterrows():
                    # Create a Signal object
                    # Note: The Signal class expects 'signal' not 'side'
                    # We'll convert 'buy'/'sell' to +1/-1
                    signal_value = 1.0 if row["side"].lower() == "buy" else -1.0

                    signal = Signal(
                        instrument=instrument,
                        signal=signal_value,
                        price=row.get("price", None),
                        # We don't have stop/take in our CSV, so we'll set them to None
                        stop=None,
                        take=None,
                        # We don't have reference_price in our CSV
                        reference_price=None,
                        # We don't have group/comment in our CSV
                        group="",
                        comment="",
                        # We'll store any additional data in options
                        options={"size": row.get("size", None), "meta": row.get("meta", {})},
                    )
                    signals.append(signal)

                signals_by_instrument[instrument] = signals

            return signals_by_instrument

        except Exception as e:
            # Log the error and return an empty dictionary
            print(f"Error restoring signals from {file_path}: {e}")
            return {}
