"""Bitfinex CcxtConnector subclass.

Bitfinex, like OKX, splits the account feed: ``watch_orders`` carries order-status
transitions and ``watch_my_trades`` carries the fills. Bitfinex needs no balance or
clOrdId override; the only addition beyond the shared ``_TwoStreamCcxtConnector``
is the documented all-None venue-figures override below.
"""

from typing import Any

from .._two_stream import _TwoStreamCcxtConnector


class BitfinexCcxtConnector(_TwoStreamCcxtConnector):
    """Bitfinex connector: split orders/fills streams, base behavior otherwise."""

    def _extract_venue_figures(
        self, raw_balance: dict[str, Any]
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Deliberately all-None: Bitfinex's ``fetch_balance`` carries no account figures.

        Its raw ``info`` is the bare wallets *list* from ``auth/r/wallets`` (the
        framework's ``BitfinexF`` exchange forces ``type='margin'`` onto it); account
        equity/margin live on a separate margin-info endpoint, out of the synchronous
        snapshot seam's reach. All-None → AM derives every metric from balances +
        positions.
        """
        return None, None, None, None
