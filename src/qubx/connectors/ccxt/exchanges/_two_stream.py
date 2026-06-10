"""Shared two-stream executions mixin for split orders/fills venues.

Most venues (Binance) carry an order's fills inline on the ``watch_orders`` report,
so the base ``CcxtConnector`` reads everything from a single stream. OKX and Bitfinex
instead split the account feed:

- ``watch_orders`` carries ONLY status transitions (Accepted / Canceled / Expired /
  Rejected, and a terminal Filled flag) — NO fill/trade detail.
- ``watch_my_trades`` carries the executions (the per-trade fills).

``_TwoStreamExecutionsMixin`` adapts the base connector to that split by staying dumb
(hybrid event model — the AccountManager correlates the two streams, not the connector):

1. Running TWO ``_run_ws_loop`` loops concurrently (one per stream), reusing the
   base's reconnect/backoff/teardown so the split path inherits the same resilience
   as the single-stream path. Only the order loop owns liveness (``mark_ready``).
2. Turning each ``watch_my_trades`` trade into a ``Deal`` and emitting one
   ``DealEvent`` per trade. A DealEvent never changes order status — it only books
   the fill into the position/ledger, deduped by trade_id.
3. Emitting plain status events off ``watch_orders`` — ``OrderPartiallyFilledEvent``
   / ``OrderFilledEvent`` with ``fill=None`` — which transition the order without
   booking anything. Whichever stream wins the race, AM's per-order trade-id dedup
   keeps the application idempotent.
"""

import asyncio
from typing import Any

from qubx import logger
from qubx.core.basics import Instrument, Order
from qubx.core.events import DealEvent, OrderFilledEvent, OrderPartiallyFilledEvent

from ..exceptions import CcxtSymbolNotRecognized
from ..utils import ccxt_convert_deal_info


class _TwoStreamExecutionsMixin:
    """Mixin overriding the base connector's execution-stream seams for split venues.

    Mixed into a ``CcxtConnector`` subclass (it relies on the base's ``_run_ws_loop``,
    ``_instrument_for_symbol``, ``_venue_to_cid`` and ``send``). Stateless: each stream
    forwards what it sees; the AccountManager correlates deals to orders.
    """

    async def _subscribe_executions(self) -> None:
        """Run the orders-status and trades streams concurrently.

        Each leg uses the base ``_run_ws_loop`` so it inherits the same
        reconnect/backoff/teardown as the single-stream base path. Only the order
        loop marks WS readiness — a trade-only feed must not flip liveness on its own.
        """
        await asyncio.gather(
            self._run_ws_loop(  # type: ignore[attr-defined]
                watch=self._em.exchange.watch_orders,  # type: ignore[attr-defined]
                handle=self._handle_ws_order,  # type: ignore[attr-defined]
                stream="orders",
                mark_ready=True,
            ),
            self._run_ws_loop(  # type: ignore[attr-defined]
                watch=self._em.exchange.watch_my_trades,  # type: ignore[attr-defined]
                handle=self._handle_ws_trade,
                stream="my_trades",
                mark_ready=False,
            ),
        )
        logger.debug(f"[{self.exchange_name}] split executions streams ended")  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Trade stream → deals
    # ------------------------------------------------------------------ #
    def _handle_ws_trade(self, raw: dict[str, Any]) -> None:
        """Convert one ``watch_my_trades`` trade to a Deal and emit a ``DealEvent``.

        Resolves the originating client_order_id from the venue id the trade carries
        (``raw['order']``) via the connector's venue->cid index. The order's status
        transitions (including the terminal FILLED) are driven by the watch_orders
        stream, never here — AM books the deal against the order whichever stream
        delivered first, deduped by trade_id.
        """
        try:
            instrument = self._instrument_for_symbol(raw["symbol"])  # type: ignore[attr-defined]
        except CcxtSymbolNotRecognized:  # unknown symbol: skip the trade
            logger.warning(
                f"[{self.exchange_name}] WS trade for unknown symbol {raw.get('symbol')}; skipped"  # type: ignore[attr-defined]
            )
            return
        deal = ccxt_convert_deal_info(raw)
        venue_order_id = raw.get("order")
        cid = self._venue_to_cid.get(venue_order_id) if venue_order_id is not None else None  # type: ignore[attr-defined]
        if cid is None:
            # The order's open/new report (which seeds the venue->cid index) may not have
            # arrived yet, the order may already be evicted (terminal status beat the
            # trade), or this is an external order. Emit routed by venue id — AM resolves
            # (or materializes) the order from it.
            logger.debug(
                f"[{self.exchange_name}] trade {deal.trade_id} for venue id {venue_order_id}: "  # type: ignore[attr-defined]
                "no cid in index; emitting deal by venue id"
            )
        self.send(  # type: ignore[attr-defined]
            DealEvent(
                instrument=instrument,
                client_order_id=cid,  # None when not indexed — AM resolves by venue id
                venue_order_id=venue_order_id,
                deal=deal,
            )
        )

    # ------------------------------------------------------------------ #
    # Order-status stream — fill branches (override the base's inline-trade reads)
    # ------------------------------------------------------------------ #
    def _handle_partial_fill_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Emit the PARTIALLY_FILLED status with no fill — the deal rides the trade stream."""
        self.send(  # type: ignore[attr-defined]
            OrderPartiallyFilledEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                fill=None,
            )
        )

    def _handle_filled_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Emit the terminal FILLED status with no fill — the deal rides the trade stream."""
        self.send(  # type: ignore[attr-defined]
            OrderFilledEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                fill=None,
            )
        )
