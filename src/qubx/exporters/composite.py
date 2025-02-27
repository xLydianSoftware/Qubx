"""
Composite Exporter for trading data.

This module provides a composite implementation of ITradeDataExport that delegates
to multiple exporters.
"""

from typing import List

from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer, ITradeDataExport


class CompositeExporter(ITradeDataExport):
    """
    Composite exporter that delegates to multiple exporters.

    This exporter can be used to send trading data to multiple destinations
    by combining multiple exporters into one.
    """

    def __init__(self, exporters: List[ITradeDataExport]):
        """
        Initialize the Composite Exporter.

        Args:
            exporters: List of exporters to delegate to
        """
        self._exporters = exporters

    def export_signals(self, time: dt_64, signals: List[Signal], account: IAccountViewer) -> None:
        """
        Export signals to all configured exporters.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to export
            account: Account viewer to get account information
        """
        for exporter in self._exporters:
            try:
                exporter.export_signals(time, signals, account)
            except Exception as e:
                from qubx import logger

                logger.error(f"Error exporting signals to {exporter.__class__.__name__}: {e}")

    def export_target_positions(self, time: dt_64, targets: List[TargetPosition], account: IAccountViewer) -> None:
        """
        Export target positions to all configured exporters.

        Args:
            time: Timestamp when the target positions were generated
            targets: List of target positions to export
            account: Account viewer to get account information
        """
        for exporter in self._exporters:
            try:
                exporter.export_target_positions(time, targets, account)
            except Exception as e:
                from qubx import logger

                logger.error(f"Error exporting target positions to {exporter.__class__.__name__}: {e}")

    def export_position_changes(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> None:
        """
        Export position changes to all configured exporters.

        Args:
            time: Timestamp when the position change occurred
            instrument: The instrument for which the position changed
            price: Price at which the position changed
            account: Account viewer to get account information
        """
        for exporter in self._exporters:
            try:
                exporter.export_position_changes(time, instrument, price, account)
            except Exception as e:
                from qubx import logger

                logger.error(f"Error exporting position changes to {exporter.__class__.__name__}: {e}")
