"""Bitfinex CcxtConnector subclass.

Bitfinex, like OKX, splits the account feed: ``watch_orders`` carries order-status
transitions and ``watch_my_trades`` carries the fills. Bitfinex needs no balance or
clOrdId override, so this subclass is just the generic ``CcxtConnector`` plus the
shared ``_TwoStreamExecutionsMixin``.
"""

from qubx.connectors.ccxt.connector import CcxtConnector

from .._two_stream import _TwoStreamExecutionsMixin


class BitfinexCcxtConnector(_TwoStreamExecutionsMixin, CcxtConnector):
    """Bitfinex connector: split orders/fills streams, base behavior otherwise."""
