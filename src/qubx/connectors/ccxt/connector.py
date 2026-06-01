"""CcxtConnector — the IConnector adapter for CCXT exchanges (write side).

Replaces the old ``CcxtBroker``. This module ships only the WRITE surface
(submit / cancel / update + leverage / margin) plus minimal lifecycle stubs so
the IConnector protocol is satisfied. The READ side (WS account-event handlers,
snapshot, single-order reconcile) lands in a later commit — its methods here are
deliberate ``TODO(account-mgmt)`` stubs.

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
"""

import asyncio
from typing import TYPE_CHECKING, Any

import ccxt

from qubx import logger
from qubx.core.basics import CtrlChannel, Instrument, OrderRequest
from qubx.core.events import (
    ChannelMessage,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import InvalidOrderParameters, ReadOnlyConnector
from qubx.core.interfaces import IDataProvider, ITimeProvider
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import to_timedelta

from .utils import ccxt_convert_order_info, instrument_to_ccxt_symbol, prepare_ccxt_order_payload

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
# ccxt.NetworkError (incl. RequestTimeout) is NOT an ExchangeError — it's transient
# connectivity, an UNKNOWN outcome, not a venue verdict. Handled separately: the
# order is left inflight for AM to reconcile, never terminal-rejected.

_CLIENT_ID_PREFIX = "qubx_"


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

        self._spawn(self._submit_async(instrument, request.client_id, payload))

    async def _submit_async(self, instrument: Instrument, client_id: str | None, payload: dict[str, Any]) -> None:
        try:
            r = await self._em.exchange.create_order(**payload)
        except ccxt.NetworkError as e:
            # Transient connectivity / timeout: the order may or may not have reached
            # the venue. Do NOT terminal-reject — leave it inflight so AM's inflight
            # check / snapshot reconcile resolves the true state from the venue.
            logger.warning(f"[{self.exchange_name}] Network error submitting {client_id}: {e}; leaving inflight")
            return
        except _VENUE_VERDICT_ERRORS as e:
            self._emit_submit_rejected(instrument, client_id, e)
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
        read side (commit 4) also emits one and AM dedups. A definitive venue
        cancel-rejection emits OrderCancelRejectedEvent.
        """
        ok, response = await self._cancel_with_retry(client_order_id, venue_order_id)
        if ok:
            self._emit_canceled_from_response(client_order_id, venue_order_id, response)
        else:
            # Route by the real client_order_id (TradingManager always passes it). Do
            # NOT substitute venue_order_id as the cid — AM's reject handlers resolve
            # the order by client_order_id, so a venue-id-as-cid would never match and
            # the order would stick in PENDING_CANCEL.
            self.send(
                OrderCancelRejectedEvent(
                    instrument=None,
                    client_order_id=client_order_id,  # type: ignore[arg-type]
                    reason=f"venue rejected cancel for {venue_order_id or client_order_id}",
                )
            )

    async def _cancel_with_retry(
        self, client_order_id: str | None, venue_order_id: str | None
    ) -> tuple[bool, dict[str, Any] | None]:
        """Port of CcxtBroker._cancel_order_with_retry.

        Returns ``(ok, venue_response)``. Does NOT emit: the caller decides whether
        a success becomes an OrderCanceledEvent (public cancel) or is folded into an
        OrderUpdatedEvent (update via cancel+recreate). The instrument/symbol is
        resolved from the venue response (ccxt echoes it), so no order cache is needed.
        """
        # cloid-only path: single attempt (Binance rejects cancel-by-cloid without
        # an orderId; retrying is useless).
        if venue_order_id is None:
            assert client_order_id is not None
            try:
                r = await self._em.exchange.cancel_order_with_client_order_id(client_order_id)
                return True, r
            except (
                ccxt.NotSupported,
                ccxt.BadRequest,
                ccxt.ExchangeError,
                ccxt.ExchangeNotAvailable,
                ccxt.NetworkError,
            ) as e:
                logger.warning(f"[{client_order_id}] Cancel-by-client-id failed: {e}")
                return False, None
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{client_order_id}] Cancel-by-client-id unexpected error: {e}")
                return False, None

        start_time = self._time.time()
        retries = 0
        while True:
            try:
                r = await self._em.exchange.cancel_order(venue_order_id)
                return True, r
            except ccxt.OperationRejected as err:
                err_msg = str(err).lower()
                if "unknown order" in err_msg or "order does not exist" in err_msg or "order not found" in err_msg:
                    logger.debug(f"[{venue_order_id}] Order not found for cancellation, might retry: {err}")
                elif "filled" in err_msg or "partially filled" in err_msg:
                    logger.debug(f"[{venue_order_id}] Order cannot be cancelled - already executed: {err}")
                    return False, None
                else:
                    logger.debug(f"[{venue_order_id}] Could not cancel order: {err}")
                    return False, None
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.ExchangeNotAvailable) as e:
                err_msg = str(e)
                if "Mandatory parameter 'orderId' was not sent" in err_msg:
                    logger.warning(f"[{venue_order_id}] Cancel failed (missing/invalid orderId): {e}")
                    return False, None
                logger.warning(f"[{venue_order_id}] Network or exchange error while cancelling: {e}")
            except Exception as err:  # noqa: BLE001
                logger.error(f"Unexpected error canceling order {venue_order_id}: {err}")
                return False, None

            elapsed_seconds = to_timedelta(self._time.time() - start_time).total_seconds()
            retries += 1
            if elapsed_seconds >= self.cancel_timeout or retries >= self.max_cancel_retries:
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
        try:
            if self._em.exchange.has.get("editOrder", False):
                r = await self._edit_order_direct(venue_order_id or client_order_id, price, quantity)
            else:
                r = await self._update_via_cancel_recreate(client_order_id, venue_order_id, price, quantity)
        except ccxt.NetworkError as e:
            # Transient: UNKNOWN whether the edit landed. Leave the order PENDING_UPDATE
            # inflight for AM to reconcile rather than emitting a terminal reject.
            logger.warning(f"[{self.exchange_name}] Network error updating {client_order_id}: {e}; leaving inflight")
            return
        except _VENUE_VERDICT_ERRORS as e:
            self._emit_update_rejected(client_order_id, e)
            return
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] Unexpected error updating order {client_order_id}: {e}")
            self._emit_update_rejected(client_order_id, e)
            return

        # The venue echoes the edited order; recover the venue id from the response
        # when we don't already have one. The connector holds no order cache (that
        # arrives in commit 4), so the instrument is left None on the event — AM
        # resolves it from the cached order by client_order_id.
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
        self, order_id: str | None, price: float | None, quantity: float | None
    ) -> dict[str, Any]:
        # TODO(account-mgmt): editOrder needs symbol/side/type, which require an order
        # cache the write-side connector doesn't have yet — so on venues that resolve
        # the market from `symbol` (e.g. Binance) this direct-edit fails and surfaces as
        # OrderUpdateRejectedEvent. The read-side commit adds the cache (and venue-correct
        # edit/recreate); until then direct edit is only reliable where ccxt tolerates the
        # missing fields.
        return await self._em.exchange.edit_order(
            id=order_id,
            symbol=None,
            type="limit",
            side=None,
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
        # Without an order cache the connector can't rebuild the full create payload
        # from ids alone; the WS read side (commit 4) carries the cached order needed
        # to recreate. Until then this path raises, surfacing as OrderUpdateRejected.
        raise ccxt.NotSupported("cancel+recreate update requires the order cache built in the read-side commit")

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
        set / length) is overridden in the subclass (commit 4).
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
    # Reconciliation primitives — READ side (commit 4)
    # ------------------------------------------------------------------ #
    def request_order_status(self, *, client_order_id: str | None = None, venue_order_id: str | None = None) -> None:
        # TODO(account-mgmt): implemented in the read-side commit (fetch single
        # order from the venue, emit the synthesized lifecycle event).
        raise NotImplementedError("request_order_status is implemented in the read-side commit")

    def request_snapshot(self) -> None:
        # TODO(account-mgmt): implemented in the read-side commit (fetch open
        # orders + positions + balances, emit AccountSnapshotEvent).
        raise NotImplementedError("request_snapshot is implemented in the read-side commit")

    # ------------------------------------------------------------------ #
    # Lifecycle / health (WS account subscriptions land in commit 4)
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        # TODO(account-mgmt): WS account-event subscriptions + initial snapshot
        # emission land in the read-side commit. Connection management itself is
        # owned by the ExchangeManager (already constructed/connected).
        pass

    def disconnect(self) -> None:
        try:
            self._run_sync(self._em.exchange.close(), timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.exchange_name}] Error during disconnect: {e}")

    def is_ws_ready(self) -> bool:
        # TODO(account-mgmt): real WS-readiness check in the read-side commit.
        return True

    def force_ws_reconnect_sync(self) -> bool:
        return self._em.force_recreation()

    @property
    def is_simulated_trading(self) -> bool:
        return False

    @property
    def read_only(self) -> bool:
        return self._read_only
