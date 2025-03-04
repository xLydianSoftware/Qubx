"""
Base formatter interfaces and implementations.

This module provides the base formatter interface and a default implementation
for formatting trading data before it is exported to external systems.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer


class IExportFormatter(ABC):
    """
    Interface for formatting trading data before export.

    Formatters are responsible for converting trading data objects (signals, target positions, etc.)
    into a format suitable for export to external systems.
    """

    @abstractmethod
    def format_signal(self, time: dt_64, signal: Signal, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a signal for export.

        Args:
            time: Timestamp when the signal was generated
            signal: The signal to format
            account: Account viewer to get account information like total capital, leverage, etc.
        Returns:
            A dictionary containing the formatted signal data
        """
        pass

    @abstractmethod
    def format_target_position(self, time: dt_64, target: TargetPosition, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a target position for export.

        Args:
            time: Timestamp when the target position was generated
            target: The target position to format
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted target position data
        """
        pass

    @abstractmethod
    def format_position_change(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> dict[str, Any]:
        """
        Format a leverage change for export.

        Args:
            time: Timestamp when the leverage change occurred
            instrument: The instrument for which the leverage changed
            price: Price at which the leverage changed
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted leverage change data
        """
        pass


class DefaultFormatter(IExportFormatter):
    """
    Default implementation of the IExportFormatter interface.

    This formatter creates standardized JSON-serializable dictionaries for each data type.
    """

    def _format_timestamp(self, timestamp: Any) -> str:
        """
        Format a timestamp for export.

        Args:
            timestamp: The timestamp to format

        Returns:
            The formatted timestamp as a string
        """
        if timestamp is not None:
            return pd.Timestamp(timestamp).isoformat()
        else:
            return ""

    def format_signal(self, time: dt_64, signal: Signal, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a signal for export.

        Args:
            time: Timestamp when the signal was generated
            signal: The signal to format
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted signal data
        """
        return {
            "timestamp": self._format_timestamp(time),
            "instrument": signal.instrument.symbol,
            "exchange": signal.instrument.exchange,
            "direction": str(signal.signal),
            "strength": str(abs(signal.signal)),
            "reference_price": str(signal.reference_price) if signal.reference_price is not None else "",
            "group": signal.group,
            "metadata": json.dumps(signal.options) if signal.options else "",
        }

    def format_target_position(self, time: dt_64, target: TargetPosition, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a target position for export.

        Args:
            time: Timestamp when the target position was generated
            target: The target position to format
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted target position data
        """
        return {
            "timestamp": self._format_timestamp(time),
            "instrument": target.instrument.symbol,
            "exchange": target.instrument.exchange,
            "target_size": str(target.target_position_size),
            "price": str(target.price) if target.price is not None else "",
            "signal_id": str(id(target.signal)) if target.signal else "",
        }

    def format_position_change(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> dict[str, Any]:
        """
        Format a position change for export.

        Args:
            time: Timestamp when the leverage change occurred
            instrument: The instrument for which the leverage changed
            price: Price at which the leverage changed
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted leverage change data
        """
        return {
            "timestamp": self._format_timestamp(time),
            "instrument": instrument.symbol,
            "exchange": instrument.exchange,
            "price": price,
            "target_quantity": account.get_position(instrument).quantity,
        }
