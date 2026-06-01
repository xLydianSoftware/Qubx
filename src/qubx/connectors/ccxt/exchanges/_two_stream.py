"""Shared two-stream executions mixin for split orders/fills venues (commit 4b).

Most venues (Binance) carry an order's fills inline on the ``watch_orders`` report,
so the base ``CcxtConnector`` reads everything from a single stream. OKX and Bitfinex
instead split the account feed:

- ``watch_orders`` carries ONLY status transitions (Accepted / Canceled / Expired /
  Rejected, and a terminal Filled flag) — NO fill/trade detail.
- ``watch_my_trades`` carries the executions (the per-trade fills).

``_TwoStreamExecutionsMixin`` adapts the base connector to that split by:

1. Running TWO ``_run_ws_loop`` loops concurrently (one per stream), reusing the
   base's reconnect/backoff/teardown so the split path inherits the same resilience
   as the single-stream path. Only the order loop owns liveness (``mark_ready``).
2. Turning each ``watch_my_trades`` trade into a ``Deal`` and emitting
   ``OrderPartiallyFilledEvent`` immediately, while remembering the LAST deal per
   client_order_id.
3. Promoting to ``OrderFilledEvent`` when the ``watch_orders`` stream reports the
   order FILLED — carrying that last remembered deal.

**Why the promotion is dedup-safe.** AccountManager dedups fills by
``deal.trade_id`` (``order.seen_trade_ids``): ``_handle_fill`` only applies the
fill amount/position when the trade_id is new, but ALWAYS transitions the order to
FILLED. So re-emitting the already-seen last deal inside the FILLED event does NOT
double-count the fill — it just drives the terminal transition. (If the two streams
race and the FILLED status beats the final trade, the trade still arrives on the
trade stream as a fresh PARTIALLY_FILLED and AM applies it; the position stays
correct, only the terminal transition is what the FILLED event guarantees.)
"""

import asyncio
from typing import Any

from qubx import logger
from qubx.core.basics import Deal, Instrument, Order
from qubx.core.events import OrderFilledEvent, OrderPartiallyFilledEvent

from ..exceptions import CcxtSymbolNotRecognized
from ..utils import ccxt_convert_deal_info


class _TwoStreamExecutionsMixin:
    """Mixin overriding the base connector's execution-stream seams for split venues.

    Mixed into a ``CcxtConnector`` subclass (it relies on the base's ``_run_ws_loop``,
    ``_instrument_for_symbol``, ``_venue_to_cid``, ``_evict`` and ``send``). The
    last-deal map is transient connector-local state — NOT account state — keyed by
    client_order_id and evicted alongside the order cache on terminal status.
    """

    # client_order_id -> last Deal seen on the trade stream, used to carry a fill into
    # the FILLED-promotion event. Transient; evicted when the order leaves the cache.
    _last_deal_by_cid: dict[str, Deal]
    # One-shot holder for the deal of the order currently being handled in
    # _handle_ws_order — captured before the base evicts it, consumed by
    # _handle_filled_status. See _handle_ws_order.
    _evicted_last_deal: Deal | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._last_deal_by_cid = {}
        self._evicted_last_deal = None

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
    # Trade stream → fills
    # ------------------------------------------------------------------ #
    def _handle_ws_trade(self, raw: dict[str, Any]) -> None:
        """Convert one ``watch_my_trades`` trade to a Deal and emit a partial fill.

        Resolves the originating client_order_id from the venue id the trade carries
        (``raw['order']``) via the connector's venue->cid index, remembers the deal as
        the last-for-cid (so the later FILLED status can carry it), and emits
        ``OrderPartiallyFilledEvent``. The order's terminal transition is driven by the
        watch_orders stream, never here.
        """
        try:
            instrument = self._instrument_for_symbol(raw["symbol"])  # type: ignore[attr-defined]
        except CcxtSymbolNotRecognized:  # mirrors base _handle_ws_order's skip
            logger.warning(
                f"[{self.exchange_name}] WS trade for unknown symbol {raw.get('symbol')}; skipped"  # type: ignore[attr-defined]
            )
            return
        deal = ccxt_convert_deal_info(raw)
        venue_order_id = raw.get("order")
        cid = self._venue_to_cid.get(venue_order_id) if venue_order_id is not None else None  # type: ignore[attr-defined]
        if cid is not None:
            self._last_deal_by_cid[cid] = deal
        else:
            # The order's open/new report (which seeds the venue->cid index) may not
            # have arrived yet, or this is an external order. Still emit the partial
            # routed by venue id — AM materializes the order from it; the FILLED
            # promotion just won't have a remembered deal and falls back to reconcile.
            logger.debug(
                f"[{self.exchange_name}] trade {deal.trade_id} for venue id {venue_order_id}: "  # type: ignore[attr-defined]
                "no cid in index yet; emitting partial by venue id"
            )
        self.send(  # type: ignore[attr-defined]
            OrderPartiallyFilledEvent(
                instrument=instrument,
                client_order_id=cid,  # type: ignore[arg-type]  # AM resolves by venue id when None
                venue_order_id=venue_order_id,
                fill=deal,
            )
        )

    # ------------------------------------------------------------------ #
    # Order-status stream — fill branches (override the base's inline-trade reads)
    # ------------------------------------------------------------------ #
    def _handle_partial_fill_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """No-op: the watch_orders report carries no trades on a split venue.

        The partial fill is emitted off the watch_my_trades stream instead
        (``_handle_ws_trade``), so there is nothing to emit from the order report.
        """

    def _handle_filled_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Promote the order to FILLED, carrying the last deal seen on the trade stream.

        The watch_orders FILLED report carries no trade, so the terminal event reuses
        the most recent deal remembered for this cid. AM dedups it by trade_id (no
        double count) but transitions the order to FILLED. If no deal was remembered
        (trade stream lagged, or an external order), there is no Deal to attach —
        ``OrderFilledEvent`` requires one — so we log and rely on AM's snapshot
        reconcile to settle the terminal state.

        The deal is read from ``_evicted_last_deal`` (stashed in ``_handle_ws_order``
        just before the base evicted the terminal order from the cache) rather than
        from the live ``_last_deal_by_cid`` map — by the time the base reaches this
        emit it has already run ``_cache_from_ws`` → ``_evict``, so the live entry is
        gone. See ``_handle_ws_order`` for the stash.
        """
        cid = order.client_order_id
        last_deal = self._evicted_last_deal
        if last_deal is None:
            logger.warning(
                f"[{self.exchange_name}] FILLED status for {cid} with no remembered trade; "  # type: ignore[attr-defined]
                "leaving terminal transition to snapshot reconcile"
            )
            return
        self.send(  # type: ignore[attr-defined]
            OrderFilledEvent(
                instrument=instrument,
                client_order_id=cid,
                venue_order_id=order.venue_order_id,
                fill=last_deal,
            )
        )

    # ------------------------------------------------------------------ #
    # Cache eviction — drop the transient last-deal alongside the order cache
    # ------------------------------------------------------------------ #
    def _handle_ws_order(self, raw: dict[str, Any]) -> None:
        """Stash this order's remembered deal before the base may evict it.

        The base ``_handle_ws_order`` runs ``_cache_from_ws`` (which calls ``_evict``
        on a terminal status, dropping ``_last_deal_by_cid[cid]``) BEFORE it emits the
        lifecycle event. So for a FILLED report we capture the remembered deal into a
        one-shot holder up front; ``_handle_filled_status`` then reads the holder. The
        holder is cleared each call so it never carries a stale deal into the next
        order's event.
        """
        cid = raw.get("clientOrderId")
        self._evicted_last_deal = self._last_deal_by_cid.get(cid) if cid is not None else None
        super()._handle_ws_order(raw)  # type: ignore[misc]
        self._evicted_last_deal = None

    def _evict(self, client_order_id: str) -> None:
        self._last_deal_by_cid.pop(client_order_id, None)
        super()._evict(client_order_id)  # type: ignore[misc]
