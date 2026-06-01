"""Bitfinex CcxtConnector subclass (commit 4b).

Bitfinex, like OKX, splits the account feed: ``watch_orders`` carries order-status
transitions and ``watch_my_trades`` carries the fills. The only per-exchange behavior
the old ``BitfinexAccountProcessor`` had was that split (confirmed against
``bitfinex_account.py`` — no balance or clOrdId override), so this subclass is just
the generic ``CcxtConnector`` plus the shared ``_TwoStreamExecutionsMixin``.
"""

from qubx.connectors.ccxt.connector import CcxtConnector

from .._two_stream import _TwoStreamExecutionsMixin


class BitfinexCcxtConnector(_TwoStreamExecutionsMixin, CcxtConnector):
    """Bitfinex connector: split orders/fills streams, base behavior otherwise."""
