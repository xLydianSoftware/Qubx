"""Bitfinex CcxtConnector subclass.

Bitfinex, like OKX, splits the account feed: ``watch_orders`` carries order-status
transitions and ``watch_my_trades`` carries the fills. Bitfinex needs no balance or
clOrdId override, so this subclass is just the shared ``_TwoStreamCcxtConnector``.
"""

from .._two_stream import _TwoStreamCcxtConnector


class BitfinexCcxtConnector(_TwoStreamCcxtConnector):
    """Bitfinex connector: split orders/fills streams, base behavior otherwise."""
