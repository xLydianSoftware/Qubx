"""CcxtConnector — the IConnector adapter for CCXT exchanges (read + write).

This module owns both sides of the IConnector surface:

- WRITE: submit / cancel / update + leverage / margin.
- READ: the WS account-event subscription (``watch_orders`` → typed lifecycle
  events), the full-account snapshot fetch, and single-order status reconcile —
  plus a small order cache the write side consults to fill in the ccxt
  symbol/side/type that cancel/edit require.

Design contract (see docs/account-management/account-management-design.md
"IConnector — exchange-facing only" and the rejection-boundary table):

- The connector is a pure adapter. Its only outbound surface to the framework is
  ``self.send(event)`` on the channel. It holds NO AccountManager / ProcessingManager
  reference. The single READ dependency is ``data_provider`` (quote lookup during
  framework-side payload validation).
- Rejection boundary (HARD rule): framework-side rejections (bad params, quote
  unavailable, below min-notional, read-only) RAISE synchronously from
  submit/cancel/update so the caller sees them immediately. Venue verdicts
  (insufficient funds, post-only crossing, rate-limit, auth, generic exchange
  error) are CAUGHT and EMITTED as OrderRejected/Cancel/UpdateRejected events on
  the channel — never raised — even when the venue returns them as a synchronous
  REST error.
- The order cache is connector-local venue-call metadata (symbol/side/type/ids),
  NOT framework account state. AccountManager owns the authoritative order/
  position/balance state; the cache only exists so cancel/edit can rebuild the
  venue payload and so WS updates can resolve the originating client_order_id.
"""

import asyncio
from asyncio.exceptions import CancelledError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ccxt
from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError

from qubx import logger
from qubx.core.basics import (
    Balance,
    CtrlChannel,
    Deal,
    Instrument,
    Order,
    OrderRequest,
    OrderStatus,
    Position,
    dt_64,
)
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    ChannelMessage,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import InvalidOrderParameters, ReadOnlyConnector
from qubx.core.interfaces import IDataProvider, ITimeProvider
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import to_timedelta

from .exceptions import CcxtSymbolNotRecognized
from .utils import (
    ccxt_convert_balance,
    ccxt_convert_order_info,
    ccxt_convert_positions,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
    instrument_to_ccxt_symbol,
    prepare_ccxt_order_payload,
)

if TYPE_CHECKING:
    from .exchange_manager import ExchangeManager

# Venue-verdict exceptions: every one of these is the venue refusing the order,
# so it rides the channel as a rejection event rather than being raised. Listed
# most-specific first; a bare ccxt.ExchangeError catches the long tail. (A
# ccxt.BadRequest that is genuinely a framework param error is expected to have
# been caught by the synchronous validation in submit_order; if one escapes the
# venue call it is treated as a venue verdict and emitted.)
_VENUE_VERDICT_ERRORS: tuple[type[Exception], ...] = (
    ccxt.InsufficientFunds,
    ccxt.OrderNotFillable,
    ccxt.InvalidOrder,
    ccxt.OperationRejected,
    ccxt.RateLimitExceeded,
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.ExchangeNotAvailable,
    ccxt.OnMaintenance,
    ccxt.ExchangeError,  # catch-all for venue-side ExchangeError subtypes (BadRequest, NotSupported, ...)
)
# RateLimitExceeded / ExchangeNotAvailable / OnMaintenance are ccxt NetworkError
# *subclasses* but the venue actively refused the request, so they are venue verdicts
# and listed above. The except sites MUST match this tuple BEFORE a bare ccxt.NetworkError
# catch, otherwise those three are swallowed as "transient" and never emit a reject.
# A bare NetworkError (RequestTimeout, connection reset) is a genuine UNKNOWN outcome —
# the order is left inflight for AM to reconcile, never terminal-rejected.

_CLIENT_ID_PREFIX = "qubx_"


@dataclass
class _CachedOrder:
    """Connector-local venue-call metadata for one order — NOT account state.

    Holds exactly what the venue REST/WS calls need that the framework ids alone
    don't carry: the ccxt ``symbol`` (cancel/edit require it on most venues), the
    ``side`` and ``type`` (edit_order requires them), and the venue id once known
    (so a cloid-keyed cache can still resolve a venue-id cancel). ``status`` is the
    last ``OrderStatus`` we converted from a WS update, used only to drive the
    new→ACCEPTED-once transition and terminal eviction.
    """

    instrument: Instrument
    ccxt_symbol: str
    side: str
    type: str
    venue_order_id: str | None = None
    status: OrderStatus | None = None


class CcxtConnector:
    """IConnector implementation backed by a CCXT exchange (write side).

    Construction args are intentionally NOT part of the IConnector protocol.
    """

    channel: CtrlChannel
    exchange_name: str

    def __init__(
        self,
        *,
        exchange_name: str,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        exchange_manager: "ExchangeManager",
        data_provider: IDataProvider,
        read_only: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        cancel_timeout: int = 30,
        cancel_retry_interval: int = 2,
        max_cancel_retries: int = 10,
        max_ws_retries: int = 10,
        **kwargs: Any,
    ):
        self.exchange_name = exchange_name
        self.channel = channel
        self._time = time_provider
        self._em = exchange_manager
        self._data_provider = data_provider
        self._read_only = read_only
        self._explicit_loop = loop
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries
        self.max_ws_retries = max_ws_retries

        # Order cache (read side): client_order_id -> _CachedOrder, plus a
        # venue_order_id -> client_order_id reverse index. Connector-local venue-call
        # metadata, not account state (AM owns that).
        self._orders: dict[str, _CachedOrder] = {}
        self._venue_to_cid: dict[str, str] = {}
        # Memoized ccxt-symbol -> Instrument resolution shared by the WS loop and the
        # snapshot/order-status converters (ccxt_find_instrument populates it lazily).
        self._symbol_to_instrument: dict[str, Instrument] = {}
        # WS execution-stream readiness: flips True after the first watch_orders()
        # round-trip, False on disconnect / before connect(). Polled by AM liveness.
        self._ws_ready = False
        self._executions_future: Any = None

    # ------------------------------------------------------------------ #
    # Async plumbing
    # ------------------------------------------------------------------ #
    @property
    def _loop(self) -> AsyncThreadLoop:
        """AsyncThreadLoop bound to the exchange's asyncio loop.

        The exchange owns the loop (created by the factory); the connector never
        creates one. Resolved lazily through the ExchangeManager so it survives
        exchange recreation.
        """
        loop = self._explicit_loop or self._em.exchange.asyncio_loop
        return AsyncThreadLoop(loop)

    def _spawn(self, coro: Any) -> None:
        """Fire-and-forget a coroutine on the exchange loop.

        Factored out so unit tests can drive the coroutine deterministically
        (await the public method's coroutine directly) instead of crossing a
        real thread/loop boundary.
        """
        future = self._loop.submit(coro)
        # The Future is otherwise discarded; surface any uncaught exception in the
        # coroutine (e.g. post-success emit work) instead of silently dead-lettering it.
        future.add_done_callback(self._log_spawn_error)

    def _log_spawn_error(self, future: Any) -> None:
        try:
            exc = future.exception()
        except Exception:  # noqa: BLE001 — cancelled/loop-teardown; nothing to surface
            return
        if exc is not None:
            logger.error(f"[{self.exchange_name}] background connector task failed: {exc!r}")

    def _run_sync(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the exchange loop and block for the result.

        Used by the synchronous leverage / margin / disconnect paths. Factored
        out (mirroring ``_spawn``) so tests can drive the coroutine without a
        real loop/thread boundary.
        """
        return self._loop.submit(coro).result(timeout=timeout)

    def send(self, event: ChannelMessage) -> None:
        self.channel.send(event)

    # ------------------------------------------------------------------ #
    # Order cache (connector-local venue-call metadata)
    # ------------------------------------------------------------------ #
    def _resolve_cached(
        self, client_order_id: str | None, venue_order_id: str | None
    ) -> _CachedOrder | None:
        """Resolve a cached order by cid (preferred) or venue id, else None."""
        if client_order_id is not None and client_order_id in self._orders:
            return self._orders[client_order_id]
        if venue_order_id is not None:
            cid = self._venue_to_cid.get(venue_order_id)
            if cid is not None:
                return self._orders.get(cid)
        return None

    def _index_venue_id(self, client_order_id: str | None, venue_order_id: str | None) -> None:
        """Record the venue id on the cached order and the reverse index."""
        if client_order_id is None or venue_order_id is None:
            return
        cached = self._orders.get(client_order_id)
        if cached is not None:
            cached.venue_order_id = venue_order_id
        self._venue_to_cid[venue_order_id] = client_order_id

    def _cache_from_ws(self, order: Order) -> None:
        """Insert/update the cache from a WS order update.

        Materializes EXTERNAL orders seen on the venue (so a later cancel/update of a
        manually-placed order resolves a symbol), refreshes the venue id/status, and
        evicts on terminal status.
        """
        cid = order.client_order_id
        if cid is None:
            return
        cached = self._orders.get(cid)
        if cached is None:
            cached = _CachedOrder(
                instrument=order.instrument,
                ccxt_symbol=instrument_to_ccxt_symbol(order.instrument),
                side=order.side.lower(),
                type=order.type.lower(),
            )
            self._orders[cid] = cached
        if order.venue_order_id is not None:
            self._index_venue_id(cid, order.venue_order_id)
        cached.status = order.status
        if order.status.is_terminal():
            self._evict(cid)

    def _evict(self, client_order_id: str) -> None:
        cached = self._orders.pop(client_order_id, None)
        if cached is not None and cached.venue_order_id is not None:
            self._venue_to_cid.pop(cached.venue_order_id, None)

    # ------------------------------------------------------------------ #
    # Write side — submit
    # ------------------------------------------------------------------ #
    def submit_order(self, request: OrderRequest) -> None:
        """Submit an order (fire-and-forget).

        Framework-side validation runs SYNCHRONOUSLY and RAISES on failure so the
        caller (TradingManager) sees it immediately. The venue call is then fired
        on the exchange loop; its verdict rides the channel as an event.
        """
        if self._read_only:
            raise ReadOnlyConnector(f"{self.exchange_name} connector is read_only")

        instrument = request.instrument
        if instrument is None:
            raise InvalidOrderParameters("submit_order: instrument is required")
        if request.quantity is None or abs(request.quantity) <= 0:
            raise InvalidOrderParameters(f"submit_order: quantity must be non-zero (got {request.quantity})")

        options = request.options or {}
        reduce_only = bool(options.get("reduceOnly", options.get("reduce_only", False)))

        # Quote lookup is the connector's only READ dependency; payload build raises
        # framework-side rejections (no quote, below min-notional, missing price).
        quote = self._data_provider.get_quote(instrument)
        payload = prepare_ccxt_order_payload(
            instrument=instrument,
            order_side=request.side,
            order_type=request.order_type,
            amount=request.quantity,
            price=request.price,
            client_id=request.client_id,
            time_in_force=request.time_in_force,
            quote=quote,
            reduce_only=reduce_only,
        )
        # Forward any remaining venue-specific options ccxt understands (e.g.
        # lighter_* indices) without clobbering what the payload builder set.
        for k, v in options.items():
            if k in ("reduceOnly", "reduce_only"):
                continue
            payload["params"].setdefault(k, v)

        # Cache the order BEFORE the async venue call: a cancel/update issued before
        # the create round-trips (or before its WS ack) must still be able to resolve
        # the ccxt symbol/side/type. The venue id is filled in by the REST ack or the
        # first WS update.
        if request.client_id is not None:
            self._orders[request.client_id] = _CachedOrder(
                instrument=instrument,
                ccxt_symbol=payload["symbol"],
                side=payload["side"],
                type=payload["type"],
            )

        self._spawn(self._submit_async(instrument, request.client_id, payload))

    async def _submit_async(self, instrument: Instrument, client_id: str | None, payload: dict[str, Any]) -> None:
        try:
            r = await self._em.exchange.create_order(**payload)
        except _VENUE_VERDICT_ERRORS as e:
            # Venue verdict — must precede the bare NetworkError catch (rate-limit /
            # maintenance / unavailable are NetworkError subclasses but are verdicts).
            self._emit_submit_rejected(instrument, client_id, e)
            return
        except ccxt.NetworkError as e:
            # Transient connectivity / timeout: the order may or may not have reached
            # the venue. Do NOT terminal-reject — leave it inflight so AM's inflight
            # check / snapshot reconcile resolves the true state from the venue.
            logger.warning(f"[{self.exchange_name}] Network error submitting {client_id}: {e}; leaving inflight")
            return
        except Exception as e:  # noqa: BLE001 — unexpected: still a venue-side failure, must not raise across the loop
            logger.error(f"[{self.exchange_name}] Unexpected error creating order {client_id}: {e}")
            self._emit_submit_rejected(instrument, client_id, e)
            return

        if r is None or r.get("id") is None:
            # Venue accepted but returned no id (some create_order_ws paths). The WS
            # read side will surface the OrderAcceptedEvent once the venue echoes it.
            logger.debug(f"[{self.exchange_name}] create_order for {client_id} returned no id; awaiting WS ack")
            return

        order = ccxt_convert_order_info(instrument, r)
        # Record the venue id on the cache so a subsequent cancel/update can prefer it.
        self._index_venue_id(order.client_order_id, order.venue_order_id)
        # Immediate ack from the REST response. AM dedups this against the later WS
        # OrderAcceptedEvent (same client_order_id), so emitting both is safe and the
        # strategy gets the faster of the two.
        self.send(
            OrderAcceptedEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.require_venue_id(),
                accepted_at=self._time.time(),
            )
        )

    def _emit_submit_rejected(self, instrument: Instrument, client_id: str | None, error: Exception) -> None:
        logger.warning(f"[{self.exchange_name}] Order {client_id} rejected by venue: {error}")
        self.send(
            OrderRejectedEvent(
                instrument=instrument,
                client_order_id=client_id,  # type: ignore[arg-type]
                reason=str(error),
                code=type(error).__name__,
            )
        )

    # ------------------------------------------------------------------ #
    # Write side — cancel
    # ------------------------------------------------------------------ #
    def cancel_order(self, *, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        if self._read_only:
            raise ReadOnlyConnector(f"{self.exchange_name} connector is read_only")
        if client_order_id is None and venue_order_id is None:
            raise InvalidOrderParameters("cancel_order: must provide client_order_id or venue_order_id")
        self._spawn(self._cancel_async(client_order_id, venue_order_id))

    async def _cancel_async(self, client_order_id: str | None, venue_order_id: str | None) -> None:
        """Cancel with retry/backoff. Prefers venue_order_id; falls back to cloid.

        A successful REST cancel ack emits OrderCanceledEvent immediately; the WS
        read side also emits one and AM dedups. A definitive venue cancel-rejection
        emits OrderCancelRejectedEvent. A transient network failure is an UNKNOWN
        outcome (the cancel may still have landed), so the order is left inflight for
        AM to reconcile rather than terminal-rejected.
        """
        try:
            ok, response = await self._cancel_with_retry(client_order_id, venue_order_id)
        except ccxt.NetworkError as e:
            logger.warning(
                f"[{self.exchange_name}] Network error cancelling {client_order_id or venue_order_id}: "
                f"{e}; leaving inflight"
            )
            return
        if ok:
            self._emit_canceled_from_response(client_order_id, venue_order_id, response)
        else:
            self._emit_cancel_rejected(client_order_id, venue_order_id)

    def _emit_cancel_rejected(self, client_order_id: str | None, venue_order_id: str | None) -> None:
        # Route by the real client_order_id: AM's reject handler resolves the order by
        # cid, so a venue-id-as-cid (or None) would never match and the order would
        # stick in PENDING_CANCEL. When the caller passed only a venue id (external-order
        # cancel), recover the cid from the venue→cid index; drop if still unknown.
        cid = client_order_id or (self._venue_to_cid.get(venue_order_id) if venue_order_id else None)
        if cid is None:
            logger.warning(
                f"[{self.exchange_name}] Cancel rejected for venue order {venue_order_id} with no known "
                "client id; dropping (unroutable)"
            )
            return
        self.send(
            OrderCancelRejectedEvent(
                instrument=None,
                client_order_id=cid,
                reason=f"venue rejected cancel for {venue_order_id or cid}",
            )
        )

    async def _cancel_with_retry(
        self, client_order_id: str | None, venue_order_id: str | None
    ) -> tuple[bool, dict[str, Any] | None]:
        """Cancel with retry/backoff. Prefers venue_order_id; falls back to cloid.

        Returns ``(ok, venue_response)`` for a DEFINITIVE outcome: ``(True, r)`` on a
        confirmed cancel, ``(False, None)`` on a venue refusal (→ cancel-reject). RAISES
        ``ccxt.NetworkError`` when the outcome is UNKNOWN (transient connectivity, or
        retries exhausted without a definitive answer) so the caller leaves the order
        inflight rather than terminal-rejecting a cancel that may have landed. Does NOT
        emit: the caller maps the outcome to an event. The ccxt ``symbol`` (which most
        venues — e.g. Binance — require) is supplied from the order cache; if the order
        is unknown (e.g. cancel of an external order we never submitted) we fall back to
        passing none and let ccxt resolve it from the id alone.
        """
        cached = self._resolve_cached(client_order_id, venue_order_id)
        symbol = cached.ccxt_symbol if cached is not None else None

        # cloid-only path: single attempt (Binance rejects cancel-by-cloid without
        # an orderId; retrying is useless).
        if venue_order_id is None:
            assert client_order_id is not None
            try:
                r = await self._em.exchange.cancel_order_with_client_order_id(client_order_id, symbol)
                return True, r
            except ccxt.NetworkError:
                # Transient (incl. ExchangeNotAvailable / OnMaintenance / rate-limit):
                # UNKNOWN whether the cancel landed → re-raise so the caller leaves it
                # inflight rather than terminal-rejecting.
                raise
            except (ccxt.NotSupported, ccxt.BadRequest, ccxt.ExchangeError) as e:
                logger.warning(f"[{client_order_id}] Cancel-by-client-id rejected by venue: {e}")
                return False, None
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{client_order_id}] Cancel-by-client-id unexpected error: {e}")
                return False, None

        start_time = self._time.time()
        retries = 0
        last_network_error: ccxt.NetworkError | None = None
        while True:
            try:
                r = await self._em.exchange.cancel_order(venue_order_id, symbol)
                return True, r
            except ccxt.OperationRejected as err:
                err_msg = str(err).lower()
                if "unknown order" in err_msg or "order does not exist" in err_msg or "order not found" in err_msg:
                    logger.debug(f"[{venue_order_id}] Order not found for cancellation, might retry: {err}")
                    last_network_error = None
                elif "filled" in err_msg or "partially filled" in err_msg:
                    logger.debug(f"[{venue_order_id}] Order cannot be cancelled - already executed: {err}")
                    return False, None
                else:
                    logger.debug(f"[{venue_order_id}] Could not cancel order: {err}")
                    return False, None
            except ccxt.NetworkError as e:
                # Transient connectivity (incl. ExchangeNotAvailable / rate-limit): retry,
                # and raise on exhaustion so the UNKNOWN outcome leaves the order inflight.
                if "Mandatory parameter 'orderId' was not sent" in str(e):
                    logger.warning(f"[{venue_order_id}] Cancel failed (missing/invalid orderId): {e}")
                    return False, None
                last_network_error = e
                logger.warning(f"[{venue_order_id}] Network error while cancelling: {e}")
            except ccxt.ExchangeError as e:
                # Definitive venue refusal (non-network) → cancel-reject after retries.
                if "Mandatory parameter 'orderId' was not sent" in str(e):
                    logger.warning(f"[{venue_order_id}] Cancel failed (missing/invalid orderId): {e}")
                    return False, None
                last_network_error = None
                logger.warning(f"[{venue_order_id}] Exchange error while cancelling: {e}")
            except Exception as err:  # noqa: BLE001
                logger.error(f"Unexpected error canceling order {venue_order_id}: {err}")
                return False, None

            elapsed_seconds = to_timedelta(self._time.time() - start_time).total_seconds()
            retries += 1
            if elapsed_seconds >= self.cancel_timeout or retries >= self.max_cancel_retries:
                if last_network_error is not None:
                    logger.error(f"[{venue_order_id}] Cancel exhausted retries after network errors; leaving inflight")
                    raise last_network_error
                logger.error(f"Timeout reached for canceling order {venue_order_id}")
                return False, None

            backoff_time = min(self.cancel_retry_interval * (2 ** (retries - 1)), 30)
            logger.debug(f"Retrying cancel for {venue_order_id} in {backoff_time}s (retry {retries})")
            await asyncio.sleep(backoff_time)

    def _emit_canceled_from_response(
        self, client_order_id: str | None, venue_order_id: str | None, response: dict[str, Any] | None
    ) -> None:
        instrument: Instrument | None = None
        cid = client_order_id
        vid = venue_order_id
        if isinstance(response, dict) and response.get("id") is not None:
            # ccxt echoes the order; recover ids from it when we lack them.
            vid = vid or str(response.get("id"))
            cid = cid or response.get("clientOrderId")
        self.send(
            OrderCanceledEvent(
                instrument=instrument,
                client_order_id=cid,  # type: ignore[arg-type]
                venue_order_id=vid,
            )
        )

    # ------------------------------------------------------------------ #
    # Write side — update
    # ------------------------------------------------------------------ #
    def update_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
    ) -> None:
        if self._read_only:
            raise ReadOnlyConnector(f"{self.exchange_name} connector is read_only")
        if client_order_id is None and venue_order_id is None:
            raise InvalidOrderParameters("update_order: must provide client_order_id or venue_order_id")
        self._spawn(self._update_async(client_order_id, venue_order_id, price, quantity))

    async def _update_async(
        self, client_order_id: str | None, venue_order_id: str | None, price: float | None, quantity: float | None
    ) -> None:
        """Direct editOrder where the venue supports it, else cancel+recreate.

        The cancel+recreate fallback preserves the original client_order_id so the
        strategy sees a single OrderUpdatedEvent rather than a Canceled+Accepted pair.
        """
        cached = self._resolve_cached(client_order_id, venue_order_id)
        try:
            if self._em.exchange.has.get("editOrder", False):
                r = await self._edit_order_direct(venue_order_id or client_order_id, price, quantity, cached)
            else:
                r = await self._update_via_cancel_recreate(client_order_id, venue_order_id, price, quantity)
        except _VENUE_VERDICT_ERRORS as e:
            # Venue verdict — must precede the bare NetworkError catch (see _submit_async).
            self._emit_update_rejected(client_order_id, e)
            return
        except ccxt.NetworkError as e:
            # Transient: UNKNOWN whether the edit landed. Leave the order PENDING_UPDATE
            # inflight for AM to reconcile rather than emitting a terminal reject.
            logger.warning(f"[{self.exchange_name}] Network error updating {client_order_id}: {e}; leaving inflight")
            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] Unexpected error updating order {client_order_id}: {e}")
            self._emit_update_rejected(client_order_id, e)
            return

        # The venue echoes the edited order; recover the venue id from the response
        # when we don't already have one. Instrument is left None on the event — AM
        # resolves it from its own cached order by client_order_id.
        vid = venue_order_id
        if isinstance(r, dict) and r.get("id") is not None:
            vid = vid or str(r.get("id"))
        self.send(
            OrderUpdatedEvent(
                instrument=None,
                client_order_id=client_order_id,  # type: ignore[arg-type]
                venue_order_id=vid or "",
                new_price=price,
                new_quantity=quantity,
            )
        )

    async def _edit_order_direct(
        self, order_id: str | None, price: float | None, quantity: float | None, cached: _CachedOrder | None
    ) -> dict[str, Any]:
        # editOrder requires symbol/side/type on most venues (Binance resolves the
        # market from `symbol`). These come from the order cache; if the order is
        # unknown we pass the venue-tolerant fallbacks (symbol/side None, type limit)
        # which works only on venues that can resolve the market from the id alone.
        return await self._em.exchange.edit_order(
            id=order_id,
            symbol=cached.ccxt_symbol if cached is not None else None,
            type=cached.type if cached is not None else "limit",
            side=cached.side if cached is not None else None,
            amount=abs(quantity) if quantity is not None else None,
            price=price,
            params={},
        )

    async def _update_via_cancel_recreate(
        self, client_order_id: str | None, venue_order_id: str | None, price: float | None, quantity: float | None
    ) -> dict[str, Any] | None:
        ok, _response = await self._cancel_with_retry(client_order_id, venue_order_id)
        if not ok:
            raise RuntimeError(f"failed to cancel order {venue_order_id or client_order_id} during update")
        # TODO(account-mgmt): cancel+recreate still needs the ORIGINAL order's full
        # parameters (quantity/price/tif when the update leaves one unspecified) which
        # the connector-local cache deliberately does not hold (AM owns that state).
        # Until that wiring exists, this path raises, surfacing as OrderUpdateRejected.
        raise ccxt.NotSupported("cancel+recreate update requires the original order parameters")

    def _emit_update_rejected(self, client_order_id: str | None, error: Exception) -> None:
        logger.warning(f"[{self.exchange_name}] Update for {client_order_id} rejected: {error}")
        self.send(
            OrderUpdateRejectedEvent(
                instrument=None,
                client_order_id=client_order_id,  # type: ignore[arg-type]
                reason=str(error),
                code=type(error).__name__,
            )
        )

    # ------------------------------------------------------------------ #
    # Client id
    # ------------------------------------------------------------------ #
    def make_client_id(self, suggested: str) -> str:
        """Return the framework client id, ensuring the ``qubx_`` prefix.

        ``ccxt_convert_order_info`` keys order-origin detection on the ``qubx_``
        prefix (FRAMEWORK vs EXTERNAL), so the connector guarantees it. The generic
        base impl only enforces the prefix; venue-specific sanitization (OKX char
        set / length) is overridden in the subclass.
        """
        if suggested.startswith(_CLIENT_ID_PREFIX):
            return suggested
        return _CLIENT_ID_PREFIX + suggested

    # ------------------------------------------------------------------ #
    # Leverage / margin
    # ------------------------------------------------------------------ #
    def set_instrument_leverage(self, instrument: Instrument, leverage: float) -> bool:
        try:
            symbol = instrument_to_ccxt_symbol(instrument)
            self._run_sync(self._em.exchange.set_leverage(leverage, symbol))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] Failed to set leverage {leverage} for {instrument.symbol}: {e}")
            return False

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool:
        try:
            symbol = instrument_to_ccxt_symbol(instrument)
            ex = self._em.exchange
            fn = getattr(ex, "set_margin_mode", None) or getattr(ex, "set_margin_type", None)
            if fn is None:
                logger.error(f"[{self.exchange_name}] does not support set_margin_mode")
                return False
            self._run_sync(fn(mode, symbol))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] Failed to set margin mode {mode} for {instrument.symbol}: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Read side — WS account-event subscription
    # ------------------------------------------------------------------ #
    def _instrument_for_symbol(self, ccxt_symbol: str) -> Instrument:
        """Resolve a ccxt symbol to an Instrument, memoizing the result."""
        return ccxt_find_instrument(ccxt_symbol, self._em.exchange, self._symbol_to_instrument)

    async def _subscribe_executions(self) -> None:
        """WS account-event loop (base / Binance model).

        A single ``watch_orders()`` stream carries both order-status transitions and
        their fills. The loop survives WS drops: it retries with backoff on transient
        errors and exits cleanly on cancellation / channel close. OKX and Bitfinex
        split fills into a separate ``watch_my_trades`` stream — they override this
        method by running two ``_run_ws_loop`` loops concurrently; the base assumes
        the unified Binance feed.
        """
        await self._run_ws_loop(
            watch=self._em.exchange.watch_orders,
            handle=self._handle_ws_order,
            stream="executions",
            mark_ready=True,
        )
        logger.debug(f"[{self.exchange_name}] executions stream ended")

    async def _run_ws_loop(
        self,
        *,
        watch: Any,
        handle: Any,
        stream: str,
        mark_ready: bool,
    ) -> None:
        """Generic WS subscription loop: ``await watch()`` → ``handle(raw)`` per item.

        Owns the reconnect/backoff/teardown contract shared by every account WS stream
        (the single Binance ``watch_orders`` feed and the split OKX/Bitfinex
        ``watch_orders`` + ``watch_my_trades`` feeds). ``mark_ready`` flips
        ``_ws_ready`` after the first successful round-trip — only the order stream
        owns liveness, so a trade-only stream passes ``mark_ready=False``. On exit
        (cancellation, channel close, or max retries) readiness is cleared so a
        half-open split feed can't report ready.
        """
        n_retry = 0
        while self.channel.control.is_set():
            try:
                updates = await watch()
                if mark_ready:
                    self._ws_ready = True
                n_retry = 0
                for raw in updates:
                    handle(raw)
            except CcxtSymbolNotRecognized:
                continue
            except CancelledError:
                break
            except ExchangeClosedByUser:
                logger.info(f"[{self.exchange_name}] {stream} stream stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"[{self.exchange_name}] error in {stream} stream: {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:  # noqa: BLE001
                if not self.channel.control.is_set():
                    break
                logger.error(f"[{self.exchange_name}] exception in {stream} stream: {e}")
                logger.exception(e)
                n_retry += 1
                if n_retry >= self.max_ws_retries:
                    logger.error(f"[{self.exchange_name}] max retries reached for {stream} stream")
                    break
                await asyncio.sleep(min(2**n_retry, 60))
        self._ws_ready = False

    def _handle_ws_order(self, raw: dict[str, Any]) -> None:
        """Convert one ccxt order update and emit the matching lifecycle event(s)."""
        try:
            instrument = self._instrument_for_symbol(raw["symbol"])
        except CcxtSymbolNotRecognized:
            logger.warning(f"[{self.exchange_name}] WS order for unknown symbol {raw.get('symbol')}; skipped")
            return
        order = ccxt_convert_order_info(instrument, raw)
        # Was this the first venue acknowledgement (no prior ACCEPTED in our cache)?
        cached = self._orders.get(order.client_order_id) if order.client_order_id is not None else None
        had_prior_ack = cached is not None and cached.status is not None
        self._cache_from_ws(order)
        self._emit_order_events(instrument, order, raw, had_prior_ack)

    def _emit_order_events(
        self, instrument: Instrument, order: Order, raw: dict[str, Any], had_prior_ack: bool
    ) -> None:
        """Map a converted order's status to the typed lifecycle event(s).

        Fill events carry the new Deal(s) extracted from the execution report; AM
        dedups by trade_id, so emitting one event per deal is safe even if the venue
        re-sends. We don't diff against a cached prior status beyond the new→ACCEPTED
        gate — AM is idempotent on every other transition (late/duplicate events are
        no-ops there), so the connector stays minimal.
        """
        status = order.status  # OrderStatus enum (mapped from the ccxt status by utils)

        # A fill can be the FIRST event we observe for an order (fast aggressive fills,
        # or a venue that sends no separate "open" report). Synthesize the venue ACCEPTED
        # ack before the fill so the strategy's on_order_accepted fires and the order
        # lifecycle stays ordered. AM dedups ACCEPTED, so this is safe alongside the REST
        # immediate-ack. Skipped when the venue id is missing (can't index the order).
        if not had_prior_ack and order.venue_order_id is not None and status in (
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
        ):
            self.send(OrderAcceptedEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                accepted_at=order.time,
            ))

        if status == OrderStatus.PARTIALLY_FILLED:
            self._handle_partial_fill_status(instrument, order, raw)
            return
        if status == OrderStatus.FILLED:
            self._handle_filled_status(instrument, order, raw)
            return
        if status == OrderStatus.CANCELED:
            self.send(OrderCanceledEvent(
                instrument=instrument, client_order_id=order.client_order_id, venue_order_id=order.venue_order_id))
            return
        if status == OrderStatus.EXPIRED:
            self.send(OrderExpiredEvent(
                instrument=instrument, client_order_id=order.client_order_id, venue_order_id=order.venue_order_id))
            return
        if status == OrderStatus.REJECTED:
            self.send(OrderRejectedEvent(
                instrument=instrument, client_order_id=order.client_order_id, reason="rejected by venue"))
            return
        # ACCEPTED (mapped from new/open, and the safe default for any status the
        # utils mapper couldn't recognize — it already logged that) → first venue ack
        # becomes ACCEPTED. A repeat (e.g. the venue re-broadcasting an open order) is
        # dropped here rather than re-emitting; AM would treat a duplicate ACCEPTED as
        # benign, but not emitting keeps the channel quiet.
        if not had_prior_ack:
            if order.venue_order_id is None:
                # An ACCEPTED ack with no venue id can't be indexed (AM keys the
                # venue-id index off it). The venue's open/new report effectively
                # always carries one, so this is an anomaly worth logging, not a "".
                logger.warning(
                    f"[{self.exchange_name}] open WS update for {order.client_order_id} carried no venue id; "
                    "skipping ACCEPTED emit"
                )
                return
            self.send(OrderAcceptedEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                accepted_at=order.time,
            ))

    def _handle_partial_fill_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Emit the PARTIALLY_FILLED fill(s) for a watch_orders report (base / Binance).

        Binance carries the trade inline on the order report, so the deals are
        extracted straight from ``raw``. Two-stream venues (OKX/Bitfinex) override this
        in ``_TwoStreamExecutionsMixin``: their watch_orders report carries no trades,
        and the partials arrive on the separate watch_my_trades stream instead.
        """
        self._emit_fills(instrument, order, ccxt_extract_deals_from_exec(raw), partial=True)

    def _handle_filled_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Emit the terminal FILLED fill(s) for a watch_orders report (base / Binance).

        The inline trades close the order (the last deal becomes OrderFilledEvent).
        Two-stream venues override this to promote-to-FILLED using the deal last seen
        on their watch_my_trades stream.
        """
        self._emit_fills(instrument, order, ccxt_extract_deals_from_exec(raw), partial=False)

    def _emit_fills(self, instrument: Instrument, order: Order, deals: list[Deal], *, partial: bool) -> None:
        """Emit one fill event per extracted deal (AM dedups by trade_id).

        OrderPartiallyFilledEvent / OrderFilledEvent both require a Deal, so when the
        venue update carries no per-trade detail there is nothing to emit here. The
        venues that omit ``trades`` on a ``watch_orders`` terminal report (OKX,
        Bitfinex) feed their fills through a separate ``watch_my_trades`` stream — the
        commit-4b subclass override that splits them — so the Deal still arrives.
        Binance carries the trade inline on the order report, so the unified base feed
        needs no split. (Reconcile does NOT recover a missed fill: a snapshot taken
        after a fully-filled order leaves the open-orders set would see it gone and
        transition it to CANCELED, not FILLED — so the trade stream, not reconcile, is
        the mitigation.)
        """
        if not deals:
            logger.debug(
                f"[{self.exchange_name}] {order.client_order_id} {order.status} WS update carried no trades; "
                "fill detail expected from the watch_my_trades stream"
            )
            return
        last = len(deals) - 1
        for i, deal in enumerate(deals):
            # On a full fill the LAST deal closes the order (OrderFilledEvent →
            # terminal); earlier deals are partials. On a partial-fill report every
            # deal is a partial.
            if partial or i < last:
                self.send(OrderPartiallyFilledEvent(
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    fill=deal,
                ))
            else:
                self.send(OrderFilledEvent(
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    fill=deal,
                ))

    # ------------------------------------------------------------------ #
    # Reconciliation primitives — READ side
    # ------------------------------------------------------------------ #
    def request_order_status(self, *, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        if client_order_id is None and venue_order_id is None:
            raise InvalidOrderParameters("request_order_status: must provide client_order_id or venue_order_id")
        self._spawn(self._order_status_async(client_order_id, venue_order_id))

    async def _order_status_async(self, client_order_id: str | None, venue_order_id: str | None) -> None:
        cached = self._resolve_cached(client_order_id, venue_order_id)
        symbol = cached.ccxt_symbol if cached is not None else None
        # Prefer the venue id (more reliable across venues); fall back to the cloid.
        lookup_id = venue_order_id or client_order_id
        try:
            raw = await self._em.exchange.fetch_order(lookup_id, symbol)
        except ccxt.OrderNotFound:
            self._emit_order_status_not_found(client_order_id, venue_order_id, cached)
            return
        except NetworkError as e:
            logger.warning(f"[{self.exchange_name}] Network error fetching order {lookup_id}: {e}; leaving inflight")
            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] error fetching order {lookup_id}: {e}")
            return
        if raw is None or raw.get("id") is None:
            self._emit_order_status_not_found(client_order_id, venue_order_id, cached)
            return
        self._handle_ws_order(raw)

    def _emit_order_status_not_found(
        self, client_order_id: str | None, venue_order_id: str | None, cached: _CachedOrder | None
    ) -> None:
        """Emit the reconcile not-found reject, routed by the REAL client_order_id.

        OrderRejectedEvent carries no venue id and AM routes a reject by cid, so a
        venue-id-only request (cid None) must resolve the cid from the order cache's
        venue->cid index. If it can't be resolved, the reject would be unroutable —
        log and skip rather than emit an event AM silently drops.
        """
        cid = client_order_id
        if cid is None and venue_order_id is not None:
            cid = self._venue_to_cid.get(venue_order_id)
        if cid is None:
            logger.warning(
                f"[{self.exchange_name}] order-status not-found for venue id {venue_order_id} but no cid known; "
                "skipping unroutable reject"
            )
            return
        self.send(OrderRejectedEvent(
            instrument=cached.instrument if cached is not None else None,
            client_order_id=cid,
            reason="reconcile: order not present at venue",
            code="OrderNotFound",
        ))

    def request_snapshot(self) -> None:
        self._spawn(self._snapshot_async())

    async def _snapshot_async(self) -> None:
        """Fetch open orders + positions + balances concurrently and emit a snapshot.

        Network/exchange errors are logged, not raised — AM retries on its next
        snapshot tick. ``return_exceptions=True`` keeps one failing fetch from
        sinking the others; a failed leg is simply omitted (left None) from the
        snapshot so reconcile won't wipe state it didn't actually observe.
        """
        ex = self._em.exchange
        as_of: dt_64 = self._time.time()
        results = await asyncio.gather(
            ex.fetch_open_orders(),
            ex.fetch_positions(),
            ex.fetch_balance(),
            return_exceptions=True,
        )
        raw_orders, raw_positions, raw_balance = results

        open_orders: list[Order] | None = None
        if isinstance(raw_orders, BaseException):
            logger.warning(f"[{self.exchange_name}] snapshot: fetch_open_orders failed: {raw_orders}")
        else:
            open_orders = []
            for raw in raw_orders:
                try:
                    instrument = self._instrument_for_symbol(raw["symbol"])
                except CcxtSymbolNotRecognized:
                    continue
                open_orders.append(ccxt_convert_order_info(instrument, raw))

        positions: list[Position] | None = None
        if isinstance(raw_positions, BaseException):
            logger.warning(f"[{self.exchange_name}] snapshot: fetch_positions failed: {raw_positions}")
        else:
            positions = ccxt_convert_positions(raw_positions, ex.name, ex.markets)

        balances: list[Balance] | None = None
        if isinstance(raw_balance, BaseException):
            logger.warning(f"[{self.exchange_name}] snapshot: fetch_balance failed: {raw_balance}")
        else:
            balances = self._convert_balances(raw_balance)

        self.send(AccountSnapshotEvent(
            instrument=None,
            snapshot=AccountSnapshot(
                exchange=self.exchange_name,
                as_of=as_of,
                open_orders=open_orders,
                positions=positions,
                balances=balances,
            ),
        ))

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Convert a ccxt fetch_balance response to framework Balances.

        Base impl reads ccxt's canonical ``total``/``used`` maps. OKX overrides this:
        ccxt maps OKX's ``eq`` (= cashBal + unrealizedPnL) to ``total``,
        but the framework wants the cash leg — so OKX reads per-currency ``cashBal``/
        ``frozenBal`` straight from the raw ``info.data[0].details``.
        """
        return ccxt_convert_balance(raw_balance, self.exchange_name)

    # ------------------------------------------------------------------ #
    # Lifecycle / health
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Start the WS account-event subscription and emit the initial snapshot.

        The exchange/connection itself is owned by the ExchangeManager (already
        constructed). Read-only connectors keep this read surface alive (the design's
        standalone-IAccountProcessor use case).
        """
        if self._executions_future is None or self._executions_future.done():
            self._executions_future = self._loop.submit(self._subscribe_executions())
            self._executions_future.add_done_callback(self._log_spawn_error)
        # Initial snapshot (design connect() contract case 1).
        self.request_snapshot()

    def disconnect(self) -> None:
        self._ws_ready = False
        if self._executions_future is not None and not self._executions_future.done():
            self._executions_future.cancel()
        self._executions_future = None
        try:
            self._run_sync(self._em.exchange.close(), timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.exchange_name}] Error during disconnect: {e}")

    def is_ws_ready(self) -> bool:
        return self._ws_ready

    def force_ws_reconnect_sync(self) -> bool:
        return self._em.force_recreation()

    @property
    def is_simulated_trading(self) -> bool:
        return False

    @property
    def read_only(self) -> bool:
        return self._read_only
