from typing import Any, Dict, Optional

from qubx.core.basics import Instrument, dt_64
from qubx.core.interfaces import IAccountViewer
from qubx.exporters.formatters.base import DefaultFormatter


class IncrementalFormatter(DefaultFormatter):
    """
    Incremental formatter for exporting trading data.

    This formatter tracks position changes and generates entry/exit messages
    based on leverage changes.
    """

    def __init__(
            self, 
            alert_name: str, 
            exchange_mapping: Optional[Dict[str, str]] = None,
            account: Optional[IAccountViewer] = None
    ):
        """
        Initialize the IncrementalFormatter.

        Args:
            alert_name: The name of the alert to include in the messages
            exchange_mapping: Optional mapping of exchange names to use in messages.
                             If an exchange is not in the mapping, the instrument's exchange is used.
            account: The account viewer to get account information like total capital, leverage, etc.
        """
        super().__init__()
        self.alert_name = alert_name
        self.exchange_mapping = exchange_mapping or {}
        self.instrument_leverages: Dict[Instrument, float] = {}

        if account:
            self.instrument_leverages = dict(account.get_leverages())

    def format_position_change(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> dict[str, Any]:
        """
        Format a position change for export.

        This method tracks leverage changes and generates entry/exit messages
        based on the change in leverage.

        Args:
            time: Timestamp when the leverage change occurred
            instrument: The instrument for which the leverage changed
            price: Price at which the leverage changed
            account: Account viewer to get account information like total capital, leverage, etc.

        Returns:
            A dictionary containing the formatted position change data
        """
        current_leverage = account.get_leverage(instrument)
        previous_leverage = self.instrument_leverages.get(instrument, 0.0)

        # Get the exchange name from mapping or use the instrument's exchange
        exchange = self.exchange_mapping.get(instrument.exchange, instrument.exchange)

        # Update the stored leverage for this instrument
        self.instrument_leverages[instrument] = current_leverage

        # If no change in leverage, return empty dict
        if current_leverage == previous_leverage:
            return {}

        # Determine if this is an entry or exit
        if abs(current_leverage) > abs(previous_leverage):
            # Position increase (entry)

            # Check if the leverage side changed (from long to short or vice versa)
            if previous_leverage * current_leverage < 0 and previous_leverage != 0:
                # Side changed - generate entry signal with the full current leverage
                side = "BUY" if current_leverage > 0 else "SELL"
                return {
                    "type": "ENTRY",
                    "data": f"{{'action':'ENTRY','exchange':'{exchange}','alertName':'{self.alert_name}','symbol':'{instrument.symbol}','side':'{side}','leverage':{abs(current_leverage)},'entryPrice':{price}}}",
                }
            else:
                # Same side - generate entry signal with the leverage difference
                leverage_change = abs(current_leverage) - abs(previous_leverage)
                side = "BUY" if current_leverage > 0 else "SELL"
                return {
                    "type": "ENTRY",
                    "data": f"{{'action':'ENTRY','exchange':'{exchange}','alertName':'{self.alert_name}','symbol':'{instrument.symbol}','side':'{side}','leverage':{leverage_change},'entryPrice':{price}}}",
                }
        else:
            # Position decrease (exit)

            # Special case: if the leverage side changed (from long to short or vice versa)
            # This can happen if we go from positive to negative or vice versa, but with a smaller absolute value
            if previous_leverage * current_leverage < 0 and current_leverage != 0:
                # Side changed - generate entry signal with the full current leverage
                side = "BUY" if current_leverage > 0 else "SELL"
                return {
                    "type": "ENTRY",
                    "data": f"{{'action':'ENTRY','exchange':'{exchange}','alertName':'{self.alert_name}','symbol':'{instrument.symbol}','side':'{side}','leverage':{abs(current_leverage)},'entryPrice':{price}}}",
                }

            # Calculate the fraction of the position that was closed
            if previous_leverage == 0:
                exit_fraction = 0.0
            else:
                exit_fraction = (abs(previous_leverage) - abs(current_leverage)) / abs(previous_leverage)

            return {
                "type": "EXIT",
                "data": f"{{'action':'EXIT','exchange':'{exchange}','alertName':'{self.alert_name}','symbol':'{instrument.symbol}','exitFraction':{exit_fraction},'exitPrice':{price}}}",
            }
