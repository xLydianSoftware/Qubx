"""
Mock objects for exporter tests.

This module provides mock implementations of various interfaces used in exporter tests.
"""

import numpy as np

from qubx.core.basics import Position
from qubx.core.interfaces import IAccountViewer


class MockAccountViewer(IAccountViewer):
    """Mock implementation of IAccountViewer for testing.

    This class provides a simple implementation of the IAccountViewer interface
    that can be used in tests without requiring a real account.
    """

    account_id = "test_account"

    def __init__(self):
        """Initialize the mock account viewer."""
        self._leverages = {}
        self._positions = {}

    def get_base_currency(self, exchange: str | None = None):
        """Get the base currency of the account."""
        return "USD"

    def get_capital(self, exchange: str | None = None):
        """Get the capital of the account."""
        return 10000.0

    def get_total_capital(self, exchange: str | None = None):
        """Get the total capital of the account."""
        return 12000.0

    def get_balances(self, exchange: str | None = None):
        """Get the balances of the account."""
        return {}

    def get_positions(self, exchange: str | None = None):
        """Get all positions in the account."""
        return {}

    def get_position(self, instrument):
        """Get the position for a specific instrument."""
        if instrument not in self._positions:
            self._positions[instrument] = Position(instrument)
        return self._positions[instrument]

    def set_position_price(self, instrument, price):
        """Set the last_update_price for a specific instrument's position."""
        position = self.get_position(instrument)
        position.last_update_price = price

    def get_orders(self, instrument=None):
        """Get orders for a specific instrument or all orders if instrument is None."""
        return {}

    def position_report(self, exchange: str | None = None):
        """Get a report of all positions."""
        return {}

    def get_leverage(self, instrument):
        """Get the leverage for a specific instrument."""
        return self._leverages.get(instrument, 1.0)

    def set_leverage(self, instrument, leverage):
        """Set the leverage for a specific instrument."""
        self._leverages[instrument] = leverage

    def get_leverages(self, exchange: str | None = None):
        """Get all leverages."""
        return self._leverages.copy()

    def get_net_leverage(self, exchange: str | None = None):
        """Get the net leverage of the account."""
        return 1.0

    def get_gross_leverage(self, exchange: str | None = None):
        """Get the gross leverage of the account."""
        return 1.0

    def get_total_required_margin(self, exchange: str | None = None):
        """Get the total required margin."""
        return 0.0

    def get_available_margin(self, exchange: str | None = None):
        """Get the available margin."""
        return 10000.0

    def get_margin_ratio(self, exchange: str | None = None):
        """Get the margin ratio."""
        return 1.0
