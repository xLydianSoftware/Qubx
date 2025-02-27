from typing import Any

from qubx.core.basics import Instrument, dt_64
from qubx.core.interfaces import IAccountViewer
from qubx.exporters.formatters.base import DefaultFormatter


class IncrementalFormatter(DefaultFormatter):
    """
    Incremental formatter for exporting trading data.
    """

    def format_position_change(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> dict[str, Any]:
        return {}
