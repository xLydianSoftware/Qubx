from typing import Any, Optional

import numpy as np

from qubx.core.basics import TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer
from qubx.exporters.formatters.base import DefaultFormatter


class TargetPositionFormatter(DefaultFormatter):
    """
    Formatter for exporting target positions as structured messages.

    This formatter creates messages suitable for trading systems that need
    to know target position sizes with leverage calculations.
    """

    def __init__(
        self,
        alert_name: str,
        exchange_mapping: Optional[dict[str, str]] = None,
        account: Optional[IAccountViewer] = None,
    ):
        """
        Initialize the TargetPositionFormatter.

        Args:
            alert_name: The name of the alert to include in the messages
            exchange_mapping: Optional mapping of exchange names to use in messages.
                             If an exchange is not in the mapping, the instrument's exchange is used.
            account: The account viewer to get account information like total capital, leverage, etc.
        """
        super().__init__()
        self.alert_name = alert_name
        self.exchange_mapping = exchange_mapping or {}

    def format_target_position(self, time: dt_64, target: TargetPosition, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a target position for export.

        This method creates a structured message with target position information
        including leverage calculated as (notional size / total capital).

        Args:
            time: Timestamp when the target position was generated
            target: The target position to format
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted target position data
        """
        # Get price: use entry_price if available, else fallback to position's last_update_price
        price = target.entry_price
        if price is None:
            position = account.get_position(target.instrument)
            if not np.isnan(position.last_update_price):
                price = position.last_update_price
            else:
                # Cannot calculate leverage without price
                return {}

        # Calculate notional size and leverage
        notional = abs(target.target_position_size * price)
        total_capital = account.get_total_capital(exchange=target.instrument.exchange)
        leverage = notional / total_capital if total_capital > 0 else 0.0

        # Determine side
        side = "BUY" if target.target_position_size > 0 else "SELL"

        # Get the exchange name from mapping or use the instrument's exchange
        exchange = self.exchange_mapping.get(target.instrument.exchange, target.instrument.exchange)

        return {
            "type": "TARGET_POSITION",
            "data": f'{{"action":"TARGET_POSITION","alertName":"{self.alert_name}","exchange":"{exchange}","symbol":"{target.instrument.exchange_symbol.upper()}","side":"{side}","leverage":{leverage}}}',
        }
