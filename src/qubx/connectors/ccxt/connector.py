"""CcxtConnector — the IConnector adapter for CCXT exchanges (read + write).

This module owns both sides of the IConnector surface:

- WRITE: submit / cancel / update + leverage / margin.
- READ: the WS account-event subscription (``watch_orders`` → typed lifecycle
  events), the full-account snapshot fetch, and single-order status reconcile.

The connector is STATELESS — it keeps no per-order state. cancel / update /
request_order_status receive the whole ``Order`` from the AccountManager (the single
source of order state) and read the ccxt symbol / side / type / ids straight off it.

Design contract (see docs/account-management/design.md, "Connectors (IConnector)"
and its "Rejection boundary" subsection):

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
"""

import asyncio
import math
import time
from asyncio.exceptions import CancelledError
from collections.abc import Coroutine
from typing import Any, Literal

import ccxt
import ccxt.pro
from ccxt import AuthenticationError, ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError

from qubx import connector_logger, logger
from qubx.core.basics import (
    FRAMEWORK_CID_PREFIX,
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
from qubx.core.connector import ChannelEmitter
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    DealEvent,
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
from qubx.core.exceptions import InvalidOrderParameters
from qubx.core.interfaces import IDataProvider, ITimeProvider
from qubx.core.utils import recognize_time
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import to_timedelta

from .exceptions import CcxtSymbolNotRecognized
from .exchange_manager import ExchangeManager
from .utils import (
    ccxt_convert_balance,
    ccxt_convert_deal_info,
    ccxt_convert_order_info,
    ccxt_convert_positions,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
    info_float,
    instrument_to_ccxt_symbol,
    prepare_ccxt_order_payload,
)

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


class CcxtConnector(ChannelEmitter):
    """IConnector implementation backed by a CCXT exchange (write side).

    Construction args are intentionally NOT part of the IConnector protocol.
    """

    channel: CtrlChannel
    exchange_name: str

    # Framework cid prefix as the venue echoes it back — what origin classification
    # keys on when parsing venue order data. A venue whose cid charset mangles
    # FRAMEWORK_CID_PREFIX (OKX bans "_") overrides this with the prefix its
    # make_client_id actually produces, so producer and classifier can never drift.
    cid_framework_prefix: str = FRAMEWORK_CID_PREFIX

    def __init__(
        self,
        *,
        exchange_name: str,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        exchange_manager: ExchangeManager,
        data_provider: IDataProvider,
        loop: asyncio.AbstractEventLoop | None = None,
        cancel_timeout: int = 30,
        cancel_retry_interval: int = 2,
        max_cancel_retries: int = 10,
        max_ws_retries: int = 10,
        **kwargs: Any,
    ):
        self.exchange_name = exchange_name
        # Diagnostic logger gated by QUBX_DEBUG_AREAS=connector (all) or connector.<exchange> (one)
        self._dbg = connector_logger(exchange_name)
        self.channel = channel
        self._time = time_provider
        self._em = exchange_manager
        self._data_provider = data_provider
        self._explicit_loop = loop
        self.cancel_timeout = cancel_timeout
        self.cancel_retry_interval = cancel_retry_interval
        self.max_cancel_retries = max_cancel_retries
        self.max_ws_retries = max_ws_retries

        # Memoized ccxt-symbol -> Instrument resolution shared by the WS loop and the
        # snapshot/order-status converters (ccxt_find_instrument populates it lazily).
        self._symbol_to_instrument: dict[str, Instrument] = {}
        # WS execution-stream readiness: flips True after the first watch_orders()
        # round-trip, False on disconnect / before connect(). Polled by AM liveness.
        self._ws_ready = False
        self._executions_future: Any = None

        # Re-subscribe the account WS stream + resync against venue truth after the
        # ExchangeManager swaps in a fresh exchange (the running _subscribe_executions
        # loop is bound to the PREVIOUS exchange object — see _handle_exchange_recreation).
        self._em.register_recreation_callback(self._handle_exchange_recreation)

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

    # ------------------------------------------------------------------ #
    # Write side — submit
    # ------------------------------------------------------------------ #
    def submit_order(self, request: OrderRequest) -> None:
        """Submit an order (fire-and-forget).

        Framework-side validation runs SYNCHRONOUSLY and RAISES on failure so the
        caller (TradingManager) sees it immediately. The venue call is then fired
        on the exchange loop; its verdict rides the channel as an event.
        """

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

        order = ccxt_convert_order_info(instrument, r, framework_prefix=self.cid_framework_prefix)
        # Immediate ack from the REST response. AM dedups this against the later WS
        # OrderAcceptedEvent (same client_order_id), so emitting both is safe and the
        # strategy gets the faster of the two.
        self.send(
            OrderAcceptedEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.require_venue_id(),
                last_update_time=order.last_update_time,
                accepted_at=self._time.time(),
            )
        )

    def _emit_submit_rejected(self, instrument: Instrument, client_id: str | None, error: Exception) -> None:
        logger.warning(f"[{self.exchange_name}] Order {client_id} rejected by venue: {error}")
        self.send(
            OrderRejectedEvent(
                instrument=instrument,
                client_order_id=client_id,
                reason=str(error),
                code=type(error).__name__,
            )
        )

    # ------------------------------------------------------------------ #
    # Write side — cancel
    # ------------------------------------------------------------------ #
    def cancel_order(self, order: Order) -> None:
        # Read every venue-call field off the Order SYNCHRONOUSLY — the async path must never
        # touch the live, AM-mutated object. ``symbol`` is what most venues (e.g. Binance) need.
        self._spawn(
            self._cancel_async(order.client_order_id, order.venue_order_id, instrument_to_ccxt_symbol(order.instrument))
        )

    async def _cancel_async(self, client_order_id: str | None, venue_order_id: str | None, symbol: str) -> None:
        """A successful REST cancel ack emits OrderCanceledEvent immediately; the WS
        read side also emits one and AM dedups. A definitive venue cancel-rejection
        emits OrderCancelRejectedEvent. A transient network failure is an UNKNOWN
        outcome (the cancel may still have landed), so the order is left inflight for
        AM to reconcile rather than terminal-rejected.
        """
        try:
            ok, response = await self._cancel_with_retry(client_order_id, venue_order_id, symbol)
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
        # Carry both ids: AM's reject handler resolves the order by cid first, then venue id,
        # so the order can revert out of PENDING_CANCEL regardless of which id the caller had.
        self.send(
            OrderCancelRejectedEvent(
                instrument=None,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                reason=f"venue rejected cancel for {venue_order_id or client_order_id}",
            )
        )

    async def _cancel_with_retry(
        self, client_order_id: str | None, venue_order_id: str | None, symbol: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """Cancel with retry/backoff. Prefers venue_order_id; falls back to cloid.

        Returns ``(ok, venue_response)`` for a DEFINITIVE outcome: ``(True, r)`` on a
        confirmed cancel, ``(False, None)`` on a venue refusal (→ cancel-reject). RAISES
        ``ccxt.NetworkError`` when the outcome is UNKNOWN (transient connectivity, or
        retries exhausted without a definitive answer) so the caller leaves the order
        inflight rather than terminal-rejecting a cancel that may have landed. Does NOT
        emit: the caller maps the outcome to an event. ``symbol`` (which most venues — e.g.
        Binance — require) comes straight off the order the AM passed.
        """
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
                if self._classify_cancel_error(err) == "reject":
                    logger.debug(f"[{venue_order_id}] Could not cancel order: {err}")
                    return False, None
                logger.debug(f"[{venue_order_id}] Order not found for cancellation, might retry: {err}")
                last_network_error = None
            except ccxt.NetworkError as e:
                # Transient connectivity (incl. ExchangeNotAvailable / rate-limit): retry,
                # and raise on exhaustion so the UNKNOWN outcome leaves the order inflight.
                if self._classify_cancel_error(e) == "reject":
                    logger.warning(f"[{venue_order_id}] Cancel failed (missing/invalid orderId): {e}")
                    return False, None
                last_network_error = e
                logger.warning(f"[{venue_order_id}] Network error while cancelling: {e}")
            except ccxt.ExchangeError as e:
                # Definitive venue refusal (non-network) → cancel-reject after retries.
                if self._classify_cancel_error(e) == "reject":
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

    def _classify_cancel_error(self, err: Exception) -> Literal["retry", "reject"]:
        """Venue-specific triage of one failed cancel attempt (mirrors the
        ``_extract_venue_figures`` seam): ``"retry"`` when the venue may still produce a
        definitive answer, ``"reject"`` on a definitive refusal. The base impl matches
        Binance's error strings; venue subclasses override.
        """
        if isinstance(err, ccxt.OperationRejected):
            msg = str(err).lower()
            if "unknown order" in msg or "order does not exist" in msg or "order not found" in msg:
                # Not visible at the venue yet (submit/cancel race) — may appear shortly.
                return "retry"
            # e.g. already filled — cancelling is permanently impossible.
            return "reject"
        if "Mandatory parameter 'orderId' was not sent" in str(err):
            # Binance's refusal of a missing/invalid orderId — retrying cannot help
            # (ccxt surfaces it as either NetworkError or ExchangeError).
            return "reject"
        return "retry"

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
                client_order_id=cid,
                venue_order_id=vid,
            )
        )

    # ------------------------------------------------------------------ #
    # Write side — update
    # ------------------------------------------------------------------ #
    def update_order(self, order: Order, *, price: float | None = None, quantity: float | None = None) -> None:
        # Read venue-call fields off the Order SYNCHRONOUSLY (see cancel_order); editOrder
        # needs side/type too, so pass them straight through.
        self._spawn(
            self._update_async(
                order.client_order_id,
                order.venue_order_id,
                instrument_to_ccxt_symbol(order.instrument),
                order.side.lower(),
                order.type.lower(),
                price,
                quantity,
            )
        )

    async def _update_async(
        self,
        client_order_id: str | None,
        venue_order_id: str | None,
        symbol: str,
        side: str,
        order_type: str,
        price: float | None,
        quantity: float | None,
    ) -> None:
        """Direct editOrder where the venue supports it, else cancel+recreate.

        The cancel+recreate fallback preserves the original client_order_id so the
        strategy sees a single OrderUpdatedEvent rather than a Canceled+Accepted pair.
        """
        # The order carries its venue id once the venue acked; before that it is None and the
        # edit goes through ccxt's client-order-id variant (Binance rejects a cloid passed as
        # orderId with -1102).
        vid = venue_order_id
        try:
            if self._em.exchange.has.get("editOrder", False):
                r = await self._edit_order_direct(client_order_id, vid, symbol, side, order_type, price, quantity)
            else:
                r = await self._update_via_cancel_recreate(client_order_id, venue_order_id, price, quantity)
        except _VENUE_VERDICT_ERRORS as e:
            # Venue verdict — must precede the bare NetworkError catch (see _submit_async).
            self._emit_update_rejected(client_order_id, venue_order_id, e)
            return
        except ccxt.NetworkError as e:
            # Transient: UNKNOWN whether the edit landed. Leave the order PENDING_UPDATE
            # inflight for AM to reconcile rather than emitting a terminal reject.
            logger.warning(f"[{self.exchange_name}] Network error updating {client_order_id}: {e}; leaving inflight")
            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] Unexpected error updating order {client_order_id}: {e}")
            self._emit_update_rejected(client_order_id, venue_order_id, e)
            return

        # The venue echoes the edited order; recover the venue id from the response
        # when we don't already have one. Instrument is left None on the event — AM
        # resolves it from its own cached order by client_order_id (or venue id).
        if isinstance(r, dict) and r.get("id") is not None:
            vid = vid or str(r.get("id"))
        self.send(
            OrderUpdatedEvent(
                instrument=None,
                client_order_id=client_order_id,
                venue_order_id=vid,  # str | None — never coerce to "" (AM would index a bogus id)
                new_price=price,
                new_quantity=quantity,
            )
        )

    async def _edit_order_direct(
        self,
        client_order_id: str | None,
        venue_order_id: str | None,
        symbol: str,
        side: str,
        order_type: str,
        price: float | None,
        quantity: float | None,
    ) -> dict[str, Any]:
        # editOrder requires symbol/side/type on most venues (Binance resolves the market from
        # `symbol`) — all read straight off the order the AM passed.
        amount = abs(quantity) if quantity is not None else None
        if venue_order_id is None:
            # cloid-only (venue ack never seen): ccxt's client-order-id variant sends the
            # cloid as origClientOrderId — mirroring the cancel path.
            assert client_order_id is not None
            return await self._em.exchange.edit_order_with_client_order_id(
                client_order_id, symbol, order_type, side, amount, price
            )
        return await self._em.exchange.edit_order(
            id=venue_order_id,
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            params={},
        )

    async def _update_via_cancel_recreate(
        self, client_order_id: str | None, venue_order_id: str | None, price: float | None, quantity: float | None
    ) -> dict[str, Any] | None:
        # Raise BEFORE cancelling: cancelling first and then failing to recreate would leave
        # the order DEAD at the venue while the strategy is told only "update rejected, order
        # still alive". The AM now passes the whole Order, so wiring a real cancel+recreate
        # (thread the order's side/type/tif + original price/qty) is straightforward — but it's
        # deferred to the recreate follow-up. Until then, reject without touching the live order.
        # TODO(account-mgmt): wire the recreate from the passed Order.
        raise ccxt.NotSupported("cancel+recreate update is not yet wired for editOrder-less venues")

    def _emit_update_rejected(self, client_order_id: str | None, venue_order_id: str | None, error: Exception) -> None:
        logger.warning(f"[{self.exchange_name}] Update for {client_order_id or venue_order_id} rejected: {error}")
        self.send(
            OrderUpdateRejectedEvent(
                instrument=None,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                reason=str(error),
                code=type(error).__name__,
            )
        )

    # ------------------------------------------------------------------ #
    # Client id
    # ------------------------------------------------------------------ #
    def make_client_id(self, suggested: str) -> str:
        """Return the framework client id, ensuring the ``FRAMEWORK_CID_PREFIX``.

        ``classify_origin`` keys order-origin detection on the prefix, so the
        connector guarantees it. The generic base impl only enforces the prefix;
        venue-specific sanitization (OKX char set / length) is overridden in the
        subclass.
        """
        if suggested.startswith(FRAMEWORK_CID_PREFIX):
            return suggested
        return FRAMEWORK_CID_PREFIX + suggested

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
        return ccxt_find_instrument(ccxt_symbol, self._em.exchange, self._symbol_to_instrument)

    async def _subscribe_executions(self) -> None:
        """Run the account WS loops concurrently (composed by ``_account_streams``).

        Base / Binance model: a single ``watch_orders()`` stream carries both
        order-status transitions and their fills, plus — on Binance derivatives
        venues only (D4) — a ``watch_balance`` push loop. On Binance both resolve
        off the same listenKey user-data WS, so no extra connections are opened.
        Each loop survives WS drops via ``_run_ws_loop``'s retry/backoff and exits
        cleanly on cancellation / channel close. Split-feed venues (OKX/Bitfinex)
        override ``_account_streams``, not this method.
        """
        await asyncio.gather(*self._account_streams())
        logger.debug(f"[{self.exchange_name}] account event streams ended")

    def _account_streams(self) -> list[Coroutine[Any, Any, None]]:
        """Build the account WS loops to gather — the venue stream-composition seam.

        Base = Binance model: the ``watch_orders`` loop (owns liveness via
        ``mark_ready``), plus a balance push loop gated on the Binance family + a
        derivatives venue (F26 / D4). Balance pushes are emitted as
        BalanceUpdateEvent; the reducer applies them absolutely through the
        per-currency ratchet. Position size is owned by the deal ledger and corrected
        by snapshot reconcile (no venue position push loop). Subclasses override to
        compose differently (``_TwoStreamCcxtConnector`` splits orders/trades and
        adds no push streams).
        """
        ex = self._em.exchange
        streams: list[Coroutine[Any, Any, None]] = [
            self._run_ws_loop(
                watch=ex.watch_orders,
                handle=self._handle_ws_order,
                stream="executions",
                mark_ready=True,
            )
        ]
        # D4 scope: the balance push handler parses Binance ACCOUNT_UPDATE shapes. Other
        # venues (hyperliquid, bybit, gateio) advertise the same has[] capability
        # flags but emit shapes this handler doesn't understand — keep them
        # snapshot-only until ported.
        if not isinstance(ex, ccxt.pro.binance) or not self._is_derivatives_venue():
            return streams
        if ex.has.get("watchBalance"):
            streams.append(
                self._run_ws_loop(
                    watch=ex.watch_balance,
                    handle=self._handle_ws_balances,
                    stream="balance",
                    mark_ready=False,
                    iterate=False,  # watch_balance resolves one Balances dict, not a list
                )
            )
        return streams

    def _is_derivatives_venue(self) -> bool:
        """True when the ccxt exchange trades derivatives (position/balance push
        streams only exist there). ccxt resolves the account stream from
        ``defaultType`` OR the linear/inverse ``defaultSubType`` — binanceusdm
        carries ``defaultSubType='linear'`` while ``defaultType`` stays 'spot'."""
        options = self._em.exchange.options
        return options.get("defaultType") in ("future", "delivery", "swap") or options.get("defaultSubType") in (
            "linear",
            "inverse",
        )

    async def _run_ws_loop(
        self,
        *,
        watch: Any,
        handle: Any,
        stream: str,
        mark_ready: bool,
        iterate: bool = True,
    ) -> None:
        """Generic WS subscription loop: ``await watch()`` → ``handle(raw)`` per item.

        Owns the reconnect/backoff/teardown contract shared by every account WS stream
        (the single Binance ``watch_orders`` feed and the split OKX/Bitfinex
        ``watch_orders`` + ``watch_my_trades`` feeds). ``mark_ready`` gives the loop
        liveness ownership — only the order stream carries it; every other stream
        passes ``mark_ready=False``. Readiness is OPTIMISTIC: ccxt account watch
        futures resolve only on actual traffic (binance ``watch_orders`` sends no
        subscribe message — the listenKey stream just delivers events), so a quiet
        account would never look ready if we waited for the first message. The loop
        marks ready each time it (re-)drives the watch and clears it when the watch
        raises (auth/connect failures surface as exceptions out of the await): after
        a transient drop readiness is False only for the backoff window, while a
        persistently failing watch raises promptly and keeps it False until AM
        liveness repairs it. On self-termination (channel close, max
        retries) readiness is cleared so a dead account feed can't report ready; on
        cancellation the canceller (disconnect / exchange recreation) owns readiness —
        clearing here would race the successor loop's optimistic set. Repair for a
        given-up stream is the AccountManager liveness reconnect. ``iterate=False``
        passes the resolved value to ``handle`` whole (``watch_balance`` returns a
        single Balances dict, not a list of updates).
        """
        n_retry = 0
        while self.channel.control.is_set():
            if mark_ready:
                self._ws_ready = True
            watch_started = time.monotonic()
            try:
                updates = await watch()
                n_retry = 0
                if iterate:
                    for raw in updates:
                        handle(raw)
                else:
                    handle(updates)
            except CcxtSymbolNotRecognized:
                continue
            except CancelledError:
                return
            except ExchangeClosedByUser:
                logger.info(f"[{self.exchange_name}] {stream} stream stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                if mark_ready:
                    self._ws_ready = False
                if time.monotonic() - watch_started > 60.0:
                    # A long-lived watch that dropped (routine venue disconnect) is not a
                    # persistent failure — don't let sporadic drops accumulate to max retries.
                    n_retry = 0
                n_retry += 1
                if isinstance(e, AuthenticationError):
                    logger.error(
                        f"[{self.exchange_name}] authentication failed on {stream} stream "
                        f"(retry {n_retry}/{self.max_ws_retries}) — check/rotate API keys: {e}"
                    )
                else:
                    logger.warning(
                        f"[{self.exchange_name}] {type(e).__name__} in {stream} stream "
                        f"(retry {n_retry}/{self.max_ws_retries}): {e}"
                    )
                if n_retry >= self.max_ws_retries:
                    logger.error(f"[{self.exchange_name}] max retries reached for {stream} stream")
                    break
                await asyncio.sleep(min(2**n_retry, 60))
            except Exception as e:  # noqa: BLE001
                if not self.channel.control.is_set():
                    break
                if mark_ready:
                    self._ws_ready = False
                if time.monotonic() - watch_started > 60.0:
                    n_retry = 0
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
        order = ccxt_convert_order_info(instrument, raw, framework_prefix=self.cid_framework_prefix)
        self._emit_order_events(instrument, order, raw)

    def _emit_order_events(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Map a converted order's status to the typed lifecycle event(s).

        Fill events carry the new Deal(s) extracted from the execution report; AM dedups
        by trade_id, so emitting one event per deal is safe even if the venue re-sends. The
        connector keeps no per-order state, so it emits ACCEPTED on every venue ack — AM
        dedups ACCEPTED by client_order_id, so a repeat is a benign no-op and every other
        transition is idempotent there too.
        """
        status = order.status  # OrderStatus enum (mapped from the ccxt status by utils)

        # A fill can be the FIRST event we observe for an order (fast aggressive fills,
        # or a venue that sends no separate "open" report). Synthesize the venue ACCEPTED
        # ack before the fill so the strategy's on_order sees ACCEPTED and the order
        # lifecycle stays ordered. AM dedups ACCEPTED. Skipped when the venue id is missing
        # (can't index the order).
        if order.venue_order_id is not None and status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.send(
                OrderAcceptedEvent(
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id, last_update_time=order.last_update_time,
                    accepted_at=order.submitted_at,
                )
            )

        if status == OrderStatus.PARTIALLY_FILLED:
            self._handle_partial_fill_status(instrument, order, raw)
            return
        if status == OrderStatus.FILLED:
            self._handle_filled_status(instrument, order, raw)
            return
        if status == OrderStatus.CANCELED:
            self.send(
                OrderCanceledEvent(
                    instrument=instrument, client_order_id=order.client_order_id, venue_order_id=order.venue_order_id, last_update_time=order.last_update_time
                )
            )
            return
        if status == OrderStatus.EXPIRED:
            self.send(
                OrderExpiredEvent(
                    instrument=instrument, client_order_id=order.client_order_id, venue_order_id=order.venue_order_id, last_update_time=order.last_update_time
                )
            )
            return
        if status == OrderStatus.REJECTED:
            self.send(
                OrderRejectedEvent(
                    instrument=instrument, client_order_id=order.client_order_id, reason="rejected by venue", last_update_time=order.last_update_time
                )
            )
            return
        # ACCEPTED (mapped from new/open, and the safe default for any status the utils mapper
        # couldn't recognize — it already logged that). Emitted on every venue ack; AM dedups a
        # repeat (e.g. the venue re-broadcasting an open order). Skipped only when the venue id
        # is missing — AM keys the venue-id index off it, and the venue's open/new report
        # effectively always carries one, so its absence is an anomaly worth logging, not a "".
        if order.venue_order_id is None:
            logger.warning(
                f"[{self.exchange_name}] open WS update for {order.client_order_id} carried no venue id; "
                "skipping ACCEPTED emit"
            )
            return
        self.send(
            OrderAcceptedEvent(
                instrument=instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                last_update_time=order.last_update_time,
                accepted_at=order.submitted_at,
            )
        )

    def _handle_partial_fill_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Emit the PARTIALLY_FILLED fill(s) for a watch_orders report (base / Binance).

        Binance carries the trade inline on the order report, so the deals are
        extracted straight from ``raw``. Two-stream venues (OKX/Bitfinex) override this
        in ``_TwoStreamCcxtConnector``: their watch_orders report carries no trades,
        and the partials arrive on the separate watch_my_trades stream instead.
        """
        self._emit_fills(instrument, order, ccxt_extract_deals_from_exec(raw), partial=True)

    def _handle_filled_status(self, instrument: Instrument, order: Order, raw: dict[str, Any]) -> None:
        """Emit the terminal FILLED fill(s) for a watch_orders report (base / Binance).

        The inline trades close the order (the last deal becomes OrderFilledEvent).
        Two-stream venues override this to emit a status-only OrderFilledEvent
        (``fill=None``); their deals arrive via DealEvent off the trade stream.
        """
        self._emit_fills(instrument, order, ccxt_extract_deals_from_exec(raw), partial=False)

    def _emit_fills(self, instrument: Instrument, order: Order, deals: list[Deal], *, partial: bool) -> None:
        """Emit one fill event per extracted deal (AM dedups by trade_id).

        This is the combined-stream (Binance) path: the deal rides embedded on the fill
        event. The venues that omit ``trades`` on a ``watch_orders`` report (OKX,
        Bitfinex) feed their fills through a separate ``watch_my_trades`` stream — the
        two-stream subclass overrides these seams to emit status-only fill events
        (``fill=None``) plus one ``DealEvent`` per trade.

        A FILLED report with no trades still emits a status-only fill: reconcile rescues
        an order that filled during a WS gap by re-fetching it (request_order_status →
        this path), and Binance ``fetch_order`` payloads typically carry no embedded
        trades. Limitation: only the STATUS is rescued — the executions are not re-booked
        here (no DealEvents), so position/balance converge via the next snapshot's
        position reconcile rather than per-deal booking.
        """
        if not deals:
            if partial:
                logger.debug(
                    f"[{self.exchange_name}] {order.client_order_id} {order.status} update carried no trades; "
                    "fill detail expected from the watch_my_trades stream"
                )
                return
            logger.warning(
                f"[{self.exchange_name}] {order.client_order_id} FILLED report carried no trades; "
                "emitting status-only fill — AM books the cumulative gap, snapshot re-syncs the rest"
            )
            self.send(
                OrderFilledEvent(
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id, last_update_time=order.last_update_time,
                    fill=None,
                    venue_filled_quantity=order.filled_quantity,
                    venue_avg_price=order.avg_fill_price,
                )
            )
            return
        last = len(deals) - 1
        for i, deal in enumerate(deals):
            self._dbg.debug(
                "emit fill {} amt={} cum={} tid={}",
                instrument.symbol,
                deal.amount,
                order.filled_quantity,
                deal.trade_id,
            )
            # On a full fill the LAST deal closes the order (OrderFilledEvent →
            # terminal); earlier deals are partials. On a partial-fill report every
            # deal is a partial.
            if partial or i < last:
                self.send(
                    OrderPartiallyFilledEvent(
                        instrument=instrument,
                        client_order_id=order.client_order_id,
                        venue_order_id=order.venue_order_id, last_update_time=order.last_update_time,
                        fill=deal,
                    )
                )
            else:
                self.send(
                    OrderFilledEvent(
                        instrument=instrument,
                        client_order_id=order.client_order_id,
                        venue_order_id=order.venue_order_id, last_update_time=order.last_update_time,
                        fill=deal,
                        # Cumulative venue figures so the reducer can book any fills the
                        # venue counted but never delivered as deals (dropped WS messages).
                        venue_filled_quantity=order.filled_quantity,
                        venue_avg_price=order.avg_fill_price,
                    )
                )

    # ------------------------------------------------------------------ #
    # Read side — WS balance pushes (F26)
    # ------------------------------------------------------------------ #
    def _handle_ws_balances(self, raw: dict[str, Any]) -> None:
        """Emit one BalanceUpdateEvent per asset changed by a venue balance push.

        ccxt's unified watch_balance dict is a cache of every currency ever seen, so
        the handler reads the raw venue message it carries in ``info``: a Binance
        futures ACCOUNT_UPDATE lists exactly the changed assets (``a.B``) with the
        post-change wallet total ``wb``. Futures pushes carry no free/locked split —
        free/locked ride as NaN so the reducer applies total-only, preserving locked.
        ``as_of`` is the venue event time ``E`` (same clock domain as ``Deal.time``,
        driving the per-currency ratchet and the covered-delta guards); ``reason`` is
        the venue change reason ``a.m`` (ORDER / FUNDING_FEE / ...).
        """
        info = raw.get("info")
        data = info.get("a") if isinstance(info, dict) else None
        event_time = info.get("E") if isinstance(info, dict) else None
        if not isinstance(data, dict) or event_time is None:
            logger.debug(f"[{self.exchange_name}] balance push without ACCOUNT_UPDATE payload; skipped")
            return
        as_of = recognize_time(int(event_time))
        reason = data.get("m")
        for entry in data.get("B") or []:
            currency = entry.get("a")
            total = entry.get("wb")
            if currency is None or total is None:
                continue
            self.send(
                BalanceUpdateEvent(
                    instrument=None,
                    balance=Balance(
                        exchange=self.exchange_name,
                        currency=currency,
                        free=math.nan,
                        locked=math.nan,
                        total=float(total),
                    ),
                    as_of=as_of,
                    reason=reason,
                )
            )

    # ------------------------------------------------------------------ #
    # Reconciliation primitives — READ side
    # ------------------------------------------------------------------ #
    def request_order_status(self, order: Order) -> None:
        # Read venue-call fields off the Order SYNCHRONOUSLY (see cancel_order). symbol is what
        # Binance refuses the fetch without.
        self._spawn(
            self._order_status_async(
                order.client_order_id,
                order.venue_order_id,
                instrument_to_ccxt_symbol(order.instrument),
                order.instrument,
            )
        )

    async def _order_status_async(
        self, client_order_id: str | None, venue_order_id: str | None, symbol: str, instrument: Instrument
    ) -> None:
        # Prefer the venue id; a cloid-only fetch must use ccxt's client-order-id variant —
        # Binance rejects a cloid passed as orderId with -1102 BadRequest, NOT OrderNotFound.
        vid = venue_order_id
        lookup_id = vid or client_order_id
        try:
            if vid is not None:
                raw = await self._em.exchange.fetch_order(vid, symbol)
            else:
                assert client_order_id is not None
                raw = await self._em.exchange.fetch_order_with_client_order_id(client_order_id, symbol)
        except ccxt.OrderNotFound:
            self._emit_order_status_not_found(client_order_id, venue_order_id, instrument)
            return
        except NetworkError as e:
            logger.warning(f"[{self.exchange_name}] Network error fetching order {lookup_id}: {e}; leaving inflight")
            return
        except ccxt.ExchangeError as e:
            # Venue refused the fetch itself (BadRequest/-1102 family) — the order's state
            # is still UNKNOWN, so no event; loud so an unanswerable reconcile fetch
            # (which burns the AM's give-up budget) is operator-visible.
            logger.warning(f"[{self.exchange_name}] status fetch for {lookup_id} refused by venue: {e}")
            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] error fetching order {lookup_id}: {e}")
            return
        if raw is None or raw.get("id") is None:
            self._emit_order_status_not_found(client_order_id, venue_order_id, instrument)
            return
        self._handle_ws_order(raw)

    def request_hist_deals(self, instrument: Instrument, since: dt_64) -> None:
        # Read the symbol off the instrument SYNCHRONOUSLY (see cancel_order).
        self._spawn(self._hist_deals_async(instrument, instrument_to_ccxt_symbol(instrument), since))

    async def _hist_deals_async(self, instrument: Instrument, symbol: str, since: dt_64) -> None:
        """Fetch trades since ``since`` and emit one DealEvent per trade.

        Recovers executions missed behind a position size diff (ConfirmPositionBySnapshot →
        RequestHistDeals). Routed by the venue order id the trade carries — the connector keeps
        no state, so AM resolves the originating order (and its cid). AM books each deal (deduped
        by trade id) and the confirm task consumes them as coverage. Errors are logged, not raised
        — the confirm task drops on timeout regardless. A single fetch (no pagination): the recovery
        window is the position-reconcile watermark, recent and small; a result hitting the venue cap
        is logged so a wider gap is operator-visible.
        """
        since_ms = int(since.astype("datetime64[ms]").astype("int64"))
        logger.debug(f"[{self.exchange_name}] hist-deals: fetch_my_trades {symbol} since {since}")
        try:
            raw_trades = await self._em.exchange.fetch_my_trades(symbol, since=since_ms)
        except NetworkError as e:
            logger.warning(f"[{self.exchange_name}] hist deals fetch for {symbol} since {since} failed: {e}")
            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] error fetching hist deals for {symbol}: {e}")
            return
        logger.debug(
            f"[{self.exchange_name}] hist-deals: {symbol} since {since} -> {len(raw_trades)} trade(s)"
            f"{' (emitting DealEvents)' if raw_trades else ' (nothing to recover)'}"
        )
        for raw in raw_trades:
            deal = ccxt_convert_deal_info(raw)
            logger.debug(
                f"[{self.exchange_name}] hist-deals: {symbol} trade tid={deal.trade_id} "
                f"amt={deal.amount} px={deal.price} order={raw.get('order')} t={deal.time}"
            )
            self.send(
                DealEvent(
                    instrument=instrument,
                    client_order_id=None,  # AM resolves the order by the venue id
                    venue_order_id=raw.get("order"),
                    deal=deal,
                    last_update_time=deal.time,  # venue trade ts (terminal audit order eviction)
                    historical=True,  # recovered trade -> materialize TERMINAL, not an ACCEPTED phantom
                )
            )

    def _emit_order_status_not_found(
        self, client_order_id: str | None, venue_order_id: str | None, instrument: Instrument
    ) -> None:
        """Emit the reconcile not-found reject, carrying both ids so AM routes by either."""
        self.send(
            OrderRejectedEvent(
                instrument=instrument,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                reason="reconcile: order not present at venue",
                code="OrderNotFound",
            )
        )

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
                order = ccxt_convert_order_info(instrument, raw, framework_prefix=self.cid_framework_prefix)
                open_orders.append(order)

        positions: list[Position] | None = None
        if isinstance(raw_positions, BaseException):
            logger.warning(f"[{self.exchange_name}] snapshot: fetch_positions failed: {raw_positions}")
        else:
            positions = ccxt_convert_positions(raw_positions, ex.name, ex.markets)

        balances: list[Balance] | None = None
        equity = available_margin = margin_ratio = withdrawable = None
        if isinstance(raw_balance, BaseException):
            logger.warning(f"[{self.exchange_name}] snapshot: fetch_balance failed: {raw_balance}")
        else:
            balances = self._convert_balances(raw_balance)
            equity, available_margin, margin_ratio, withdrawable = self._extract_venue_figures(raw_balance)

        self.send(
            AccountSnapshotEvent(
                instrument=None,
                snapshot=AccountSnapshot(
                    exchange=self.exchange_name,
                    as_of=as_of,
                    open_orders=open_orders,
                    positions=positions,
                    balances=balances,
                    equity=equity,
                    available_margin=available_margin,
                    margin_ratio=margin_ratio,
                    withdrawable=withdrawable,
                ),
            )
        )

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Convert a ccxt fetch_balance response to framework Balances.

        Base impl reads ccxt's canonical ``total``/``used`` maps; venue subclasses
        override when ccxt's mapping is wrong for the framework (see OKX).
        """
        return ccxt_convert_balance(raw_balance, self.exchange_name)

    def _extract_venue_figures(
        self, raw_balance: dict[str, Any]
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """(equity, available_margin, margin_ratio, withdrawable) from the venue's raw
        account payload.

        ccxt has no unified account-figures schema, so the base impl reads the
        Binance-futures account fields carried through in ``info`` (both fapi v2 and
        v3 account payloads carry them top-level): ``totalMarginBalance`` (wallet +
        unrealized PnL = account equity), ``availableBalance`` (margin available for
        new positions) and ``maxWithdrawAmount`` (maximum amount for transfer out).
        Binance reports no direct margin ratio — left None so AM derives it. Venues
        whose payload lacks these keys yield all-None and AM derives every metric;
        subclasses override for venue-specific payloads.
        """
        info = raw_balance.get("info")
        if not isinstance(info, dict):
            return None, None, None, None
        return (
            info_float(info, "totalMarginBalance"),
            info_float(info, "availableBalance"),
            None,
            info_float(info, "maxWithdrawAmount"),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle / health
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Start the WS account-event subscription and emit the initial snapshot.

        The exchange/connection itself is owned by the ExchangeManager (already
        constructed).
        """
        self._start_executions_stream()
        # Initial snapshot (design.md "connect / reconnect contract", case 1).
        self.request_snapshot()

    def _start_executions_stream(self) -> None:
        """Submit the _subscribe_executions loop on the exchange loop if not running.

        Factored out of connect() so the recreation handler can restart the stream
        against a freshly-recreated exchange without re-issuing the initial-snapshot
        side effect.
        """
        if self._executions_future is None or self._executions_future.done():
            self._executions_future = self._loop.submit(self._subscribe_executions())
            self._executions_future.add_done_callback(self._log_spawn_error)

    def _handle_exchange_recreation(self) -> None:
        """Re-subscribe the account WS stream against the freshly-recreated exchange and
        pull a snapshot.

        The running _subscribe_executions loop captured watch_orders bound to the
        *previous* exchange; after recreation that stream is dead, so restart it (the
        loop re-reads self._em.exchange.watch_orders) and resync AM against venue truth
        (design.md "connect / reconnect contract", case 2).
        """
        if self._executions_future is None:
            return  # never connected; nothing to resubscribe
        self._ws_ready = False
        if not self._executions_future.done():
            self._executions_future.cancel()
        # Drop the old (now-cancelled) future so _start_executions_stream submits a
        # fresh one — a just-cancelled future may not report done() synchronously, and
        # the start helper skips resubmission while the old future looks live.
        self._executions_future = None
        self._start_executions_stream()
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

    def reconnect(self) -> bool:
        return self._em.force_recreation()

    @property
    def is_simulated_trading(self) -> bool:
        return False
