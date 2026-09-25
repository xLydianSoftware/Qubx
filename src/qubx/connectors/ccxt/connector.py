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
import hashlib
import math
import re
import threading
import time
import uuid
from asyncio.exceptions import CancelledError
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from typing import Any, Literal

import ccxt
import ccxt.pro
import numpy as np
from ccxt import AuthenticationError, ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError

from qubx import connector_logger, logger
from qubx.core.basics import (
    FRAMEWORK_CID_PREFIX,
    OPTION_REPRICE_IF_CROSSING,
    Balance,
    CtrlChannel,
    CurrencyConversion,
    Deal,
    Instrument,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    RejectCause,
    VenueSettingsUpdate,
    create_venue_settings_event,
    dt_64,
    resolve_reduce_only,
)
from qubx.core.connector import ChannelEmitter
from qubx.core.errors import ErrorLevel, VenueOperationError, create_error_event
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    CurrencyConversionEvent,
    DealEvent,
    FundingPaymentEvent,
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
from qubx.rate_limiting import RateLimitGateTimeout
from qubx.utils.misc import AsyncThreadLoop
from qubx.utils.time import to_timedelta

from .exceptions import CcxtSymbolNotRecognized
from .exchange_manager import ExchangeManager
from .utils import (
    FRAMEWORK_ONLY_OPTIONS,
    ccxt_convert_balance,
    ccxt_convert_deal_info,
    ccxt_convert_order_info,
    ccxt_convert_positions,
    ccxt_extract_deals_from_exec,
    ccxt_extract_leverage_settings,
    ccxt_extract_margin_modes,
    ccxt_find_instrument,
    info_float,
    instrument_to_ccxt_symbol,
    normalize_margin_mode,
    prepare_ccxt_order_payload,
    reject_cause_of,
)

# Venue-verdict exceptions: every one of these is the venue refusing the order,
# so it rides the channel as a rejection event rather than being raised. Listed
# most-specific first; a bare ccxt.ExchangeError catches the long tail. (A
# ccxt.BadRequest that is genuinely a framework param error is expected to have
# been caught by the synchronous validation in submit_order; if one escapes the
# venue call it is treated as a venue verdict and emitted.)
# How often the connector re-reads the venue's configured / maximum leverage.
LEVERAGE_REFRESH_INTERVAL_S = 3600.0
# Default bound for the synchronous venue calls below. An unbounded wait on the exchange
# loop from the strategy/account thread is the deadlock this connector must never allow.
DEFAULT_VENUE_CALL_TIMEOUT_SECONDS = 15.0
# Enforces Binance's newClientOrderId rule; ids already inside it pass through untouched.
_CID_ILLEGAL_RE = re.compile(r"[^.A-Za-z0-9:/_-]")
_CID_MAX_LEN = 36
_CID_DIGEST_LEN = 8


def with_framework_prefix(suggested: str) -> str:
    return suggested if suggested.startswith(FRAMEWORK_CID_PREFIX) else FRAMEWORK_CID_PREFIX + suggested


def conforming_client_id(cid: str) -> str:
    """Map a prefixed cid into ``[.A-Za-z0-9:/_-]{1,36}``, deterministically.

    The symbol part is stripped of illegal chars, trimmed and suffixed with a digest of the
    original (distinct symbols stay distinct); the ``_<counter>`` tail is kept whole.
    """
    if len(cid) <= _CID_MAX_LEN and not _CID_ILLEGAL_RE.search(cid):
        return cid
    body = cid[len(FRAMEWORK_CID_PREFIX) :]
    symbol, sep, tail = body.rpartition("_")
    if not sep or _CID_ILLEGAL_RE.search(tail):
        symbol, tail = body, ""
    else:
        tail = sep + tail
    room = _CID_MAX_LEN - len(FRAMEWORK_CID_PREFIX) - len(tail) - _CID_DIGEST_LEN
    if room < 0:
        return FRAMEWORK_CID_PREFIX + _digest(body)[: _CID_MAX_LEN - len(FRAMEWORK_CID_PREFIX)]
    return f"{FRAMEWORK_CID_PREFIX}{_CID_ILLEGAL_RE.sub('', symbol)[:room]}{_digest(symbol)[:_CID_DIGEST_LEN]}{tail}"


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _LeverageInfo:
    """One symbol's venue leverage state. Any field is None when that read failed or the
    venue does not report it.

    Leverages are whole numbers wherever the venue reports them so — the wire takes an
    integer — but a venue that publishes a fractional configured leverage keeps it, since
    rounding it down would pick too generous a notional tier.
    """

    configured: float | None
    maximum: int | None
    max_notional: float | None = None
    margin_mode: str | None = None


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
# The venue telling us we exceeded ITS order budget. Both are needed: Binance maps -1015
# ("too many new orders") to RateLimitExceeded, Kraken maps "EOrder:Rate limit exceeded" to
# DDoSProtection, and in ccxt the two are siblings, not parent/child.
_ORDER_RATE_LIMIT_ERRORS: tuple[type[Exception], ...] = (ccxt.RateLimitExceeded, ccxt.DDoSProtection)

# RateLimitExceeded / ExchangeNotAvailable / OnMaintenance are ccxt NetworkError
# *subclasses* but the venue actively refused the request, so they are venue verdicts
# and listed above. The except sites MUST match this tuple BEFORE a bare ccxt.NetworkError
# catch, otherwise those three are swallowed as "transient" and never emit a reject.
# A bare NetworkError (RequestTimeout, connection reset) is a genuine UNKNOWN outcome —
# the order is left inflight for AM to reconcile, never terminal-rejected.


@dataclass(frozen=True)
class VenueFigures:
    """Account-level figures a connector reads off its raw balance payload. None = the
    venue did not report it (AM derives that metric)."""

    equity: float | None = None
    available_margin: float | None = None
    margin_ratio: float | None = None
    withdrawable: float | None = None
    total_maint_margin: float | None = None
    total_initial_margin: float | None = None
    collateral_equity: float | None = None


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

    # Whether the venue's ACCOUNT_UPDATE `a.B[].wb` is the ACCOUNT wallet balance.
    # True on plain UM/CM, where the futures wallet is the account. False on venues
    # whose stream reports a SUB-wallet (Binance PM) — there an absolute push would
    # overwrite the account balance with the sub-wallet figure, so they stay
    # snapshot-only.
    _wants_ws_balance_push: bool = True

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
        self._funding_future: Any = None
        self._leverage_future: Any = None
        # - venue leverage state per ccxt symbol, refreshed by _leverage_poll_loop
        self._leverage_cache: dict[str, _LeverageInfo] = {}

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
            exc = future.exception()  # unbounded-result-ok: done-callback, the future is finished
        except Exception:  # noqa: BLE001 — cancelled/loop-teardown; nothing to surface
            return
        if exc is not None:
            # Positional arg, NOT an f-string: venue error text can contain markup (e.g. a
            # Binance HTML error page) that loguru's colorizer rejects as bad color tags —
            # only the static format string is tag-parsed, args are inserted after.
            logger.error("[{}] background connector task failed: {!r}", self.exchange_name, exc)

    def _run_sync(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the exchange loop and block for the result.

        Used by the synchronous leverage / margin / disconnect paths. Factored
        out (mirroring ``_spawn``) so tests can drive the coroutine without a
        real loop/thread boundary.

        Always bounded, and routed through ``run_sync`` so being called from the exchange loop's
        own thread raises instead of parking that loop on itself.
        """
        try:
            return self._loop.run_sync(coro, timeout=DEFAULT_VENUE_CALL_TIMEOUT_SECONDS if timeout is None else timeout)
        except RuntimeError:
            # The loop-thread guard (or a closed loop) rejected before the coroutine was ever
            # awaited; close it so it does not surface as "coroutine was never awaited".
            coro.close()
            raise

    async def _acquire_endpoint_budget(self, endpoint: str) -> None:
        """Charge whatever budget this endpoint draws beyond IP weight, which the throttle already took."""
        rate_limiter = self._em.rate_limiter
        if rate_limiter is not None:
            await rate_limiter.acquire(endpoint)

    def _report_order_limit_hit(self, error: Exception, pool_name: str = "orders") -> None:
        """Close ``pool_name``'s gate when the venue itself reports a budget breach.

        A no-op for every other error. Without it the pool is proactive-model-only, so it never
        sees the budget another actor on the same account (manual trading, a second bot) spends.

        ``pool_name`` must be the one the refused endpoint is billed against — closing ``orders``
        for a set_leverage breach would stop trading over a budget orders never spent. An
        undefined name is a no-op in the limiter.
        """
        rate_limiter = self._em.rate_limiter
        if rate_limiter is None or not isinstance(error, _ORDER_RATE_LIMIT_ERRORS):
            return
        # pool_name, never endpoint= — the latter closes every pool in the endpoint's cost list
        rate_limiter.report_limit_hit(pool_name=pool_name, reason=f"venue rate limit on {pool_name}: {error}")

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
        reduce_only = bool(resolve_reduce_only(options))
        post_only = bool(options.get("post_only", False))
        reprice_if_crossing = bool(options.get(OPTION_REPRICE_IF_CROSSING, False))

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
            post_only=post_only,
            reprice_if_crossing=reprice_if_crossing,
        )
        # Forward any remaining venue-specific options ccxt understands (e.g.
        # lighter_* indices) without clobbering what the payload builder set.
        for k, v in options.items():
            if k in FRAMEWORK_ONLY_OPTIONS:
                continue
            payload["params"].setdefault(k, v)

        self._spawn(self._submit_async(instrument, request.client_id, payload))

    async def _submit_async(self, instrument: Instrument, client_id: str | None, payload: dict[str, Any]) -> None:
        try:
            await self._acquire_endpoint_budget("create_order")
            r = await self._em.exchange.create_order(**payload)
        except RateLimitGateTimeout as e:
            self._emit_submit_rejected(instrument, client_id, e)
            return
        except _VENUE_VERDICT_ERRORS as e:
            # Venue verdict — must precede the bare NetworkError catch (rate-limit /
            # maintenance / unavailable are NetworkError subclasses but are verdicts).
            self._report_order_limit_hit(e)
            self._emit_submit_rejected(instrument, client_id, e)
            return
        except ccxt.NetworkError as e:
            # Transient connectivity / timeout: the order may or may not have reached
            # the venue. Do NOT terminal-reject — leave it inflight so AM's inflight
            # check / snapshot reconcile resolves the true state from the venue.
            self._report_order_limit_hit(e)
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
        logger.warning(f"[{self.exchange_name}] Order {client_id} rejected: {error}")
        self.send(
            OrderRejectedEvent(
                instrument=instrument,
                client_order_id=client_id,
                reason=str(error),
                code=type(error).__name__,
                cause=reject_cause_of(error),
            )
        )

    # ------------------------------------------------------------------ #
    # Write side — cancel
    # ------------------------------------------------------------------ #
    def cancel_order(self, order: Order) -> None:
        # Read every venue-call field off the Order SYNCHRONOUSLY — the async path must never
        # touch the live, AM-mutated object. ``symbol`` is what most venues (e.g. Binance) need.
        # A stop/conditional order lives on the venue's trigger surface, so the cancel must
        # target it (else Binance answers -2011 for a live order and the cancel silently fails).
        is_trigger = order.type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT)
        self._spawn(
            self._cancel_async(
                order.client_order_id,
                order.venue_order_id,
                instrument_to_ccxt_symbol(order.instrument),
                is_trigger,
            )
        )

    async def _cancel_async(
        self, client_order_id: str | None, venue_order_id: str | None, symbol: str, is_trigger: bool = False
    ) -> None:
        """A successful REST cancel ack emits OrderCanceledEvent immediately; the WS
        read side also emits one and AM dedups. A definitive venue cancel-rejection
        emits OrderCancelRejectedEvent. A transient network failure is an UNKNOWN
        outcome (the cancel may still have landed), so the order is left inflight for
        AM to reconcile rather than terminal-rejected.
        """
        try:
            # Outside _cancel_with_retry: one slot per operation, and its broad handlers would eat the timeout.
            await self._acquire_endpoint_budget("cancel_order")
            ok, response = await self._cancel_with_retry(client_order_id, venue_order_id, symbol, is_trigger)
        except RateLimitGateTimeout as e:
            self._emit_cancel_rejected(
                client_order_id,
                venue_order_id,
                reason=f"rate limited: {e}",
                code="RateLimitGateTimeout",
                cause=reject_cause_of(e),
            )
            return
        except ccxt.NetworkError as e:
            logger.warning(
                f"[{self.exchange_name}] Network error cancelling {client_order_id or venue_order_id}: "
                f"{e}; leaving inflight"
            )
            return
        if ok is None:
            # Order already gone at the venue (filled/expired/canceled before our cancel landed).
            # Emit nothing — the WS terminal + snapshot reconciler resolve the true terminal state.
            logger.debug(f"[{self.exchange_name}] cancel: {client_order_id or venue_order_id} already gone; no event")
            return
        if ok:
            self._emit_canceled_from_response(client_order_id, venue_order_id, response)
        else:
            self._emit_cancel_rejected(client_order_id, venue_order_id)

    def _emit_cancel_rejected(
        self,
        client_order_id: str | None,
        venue_order_id: str | None,
        reason: str | None = None,
        code: str | None = None,
        cause: RejectCause = RejectCause.UNKNOWN,
    ) -> None:
        # Carry both ids: AM's reject handler resolves the order by cid first, then venue id,
        # so the order can revert out of PENDING_CANCEL regardless of which id the caller had.
        self.send(
            OrderCancelRejectedEvent(
                instrument=None,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                reason=reason or f"venue rejected cancel for {venue_order_id or client_order_id}",
                code=code,
                cause=cause,
            )
        )

    async def _cancel_with_retry(
        self, client_order_id: str | None, venue_order_id: str | None, symbol: str, is_trigger: bool = False
    ) -> tuple[bool | None, dict[str, Any] | None]:
        """Cancel with retry/backoff. Prefers venue_order_id; falls back to cloid.

        Returns ``(ok, venue_response)`` for a DEFINITIVE outcome: ``(True, r)`` on a
        confirmed cancel, ``(False, None)`` on a venue refusal (→ cancel-reject),
        ``(None, None)`` when the order is already GONE at the venue (acked order the venue
        already removed → emit nothing; the WS terminal + snapshot reconciler resolve it).
        RAISES
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
                r = (
                    await self._em.exchange.cancel_order_with_client_order_id(
                        client_order_id, symbol, params={"trigger": True}
                    )
                    if is_trigger
                    else await self._em.exchange.cancel_order_with_client_order_id(client_order_id, symbol)
                )
                return True, r
            except ccxt.NetworkError as e:
                # Transient (incl. ExchangeNotAvailable / OnMaintenance / rate-limit):
                # UNKNOWN whether the cancel landed → re-raise so the caller leaves it
                # inflight rather than terminal-rejecting.
                self._report_order_limit_hit(e)
                raise
            except (ccxt.NotSupported, ccxt.BadRequest, ccxt.ExchangeError) as e:
                logger.warning(f"[{client_order_id}] Cancel-by-client-id rejected by venue: {e}")
                return False, None
            except RateLimitGateTimeout:
                # erupts from inside the venue call (the throttle hook's own gate check) — the venue
                # was never reached, so it is not a refusal; the caller emits the rate-limited event.
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{client_order_id}] Cancel-by-client-id unexpected error: {e}")
                return False, None

        start_time = self._time.time()
        retries = 0
        last_network_error: ccxt.NetworkError | None = None
        while True:
            try:
                r = (
                    await self._em.exchange.cancel_order(venue_order_id, symbol, params={"trigger": True})
                    if is_trigger
                    else await self._em.exchange.cancel_order(venue_order_id, symbol)
                )
                return True, r
            except ccxt.OperationRejected as err:
                # acked=True: we have a venue id, so an "unknown order" is GONE, not the race.
                verdict = self._classify_cancel_error(err, acked=True)
                if verdict == "reject":
                    logger.debug(f"[{venue_order_id}] Could not cancel order: {err}")
                    return False, None
                if verdict == "gone":
                    logger.debug(f"[{venue_order_id}] already gone at venue; nothing to cancel: {err}")
                    return None, None
                logger.debug(f"[{venue_order_id}] Order not found for cancellation, might retry: {err}")
                last_network_error = None
            except ccxt.NetworkError as e:
                # Transient connectivity (incl. ExchangeNotAvailable / rate-limit): retry,
                # and raise on exhaustion so the UNKNOWN outcome leaves the order inflight.
                self._report_order_limit_hit(e)
                verdict = self._classify_cancel_error(e, acked=True)
                if verdict == "reject":
                    logger.warning(f"[{venue_order_id}] Cancel failed (missing/invalid orderId): {e}")
                    return False, None
                if verdict == "gone":
                    logger.debug(f"[{venue_order_id}] already gone at venue; nothing to cancel: {e}")
                    return None, None
                last_network_error = e
                logger.warning(f"[{venue_order_id}] Network error while cancelling: {e}")
            except ccxt.ExchangeError as e:
                # Definitive venue refusal (non-network) → cancel-reject after retries.
                verdict = self._classify_cancel_error(e, acked=True)
                if verdict == "reject":
                    logger.warning(f"[{venue_order_id}] Cancel failed (missing/invalid orderId): {e}")
                    return False, None
                if verdict == "gone":
                    # Order existed (we have its id) but is already gone at the venue — the WS
                    # terminal + snapshot reconciler resolve it; don't retry, don't reject.
                    logger.debug(f"[{venue_order_id}] already gone at venue; nothing to cancel: {e}")
                    return None, None
                last_network_error = None
                logger.warning(f"[{venue_order_id}] Exchange error while cancelling: {e}")
            except RateLimitGateTimeout:
                # see the cloid-only path above: not a venue refusal, the caller emits it coded.
                raise
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

    def _classify_cancel_error(self, err: Exception, *, acked: bool = False) -> Literal["retry", "reject", "gone"]:
        """Venue-specific triage of one failed cancel attempt (mirrors the
        ``_extract_venue_figures`` seam): ``"retry"`` when the venue may still produce a
        definitive answer, ``"reject"`` on a definitive refusal, ``"gone"`` when the order
        provably no longer exists at the venue. The base impl reads ccxt's typed errors and
        falls back to Binance's error strings; venue subclasses override.

        ``acked`` = the order had a venue id (so it WAS accepted). An "unknown order" then
        means it is already gone (filled/expired/canceled), not the submit/cancel race —
        retrying is pointless. Without an ack it IS the race (the order may appear shortly).
        """
        msg = str(err).lower()
        if isinstance(err, ccxt.OrderNotFound):
            # - ccxt's per-venue error-code map is the authority on "not there any more", and its
            #   wording often misses the strings below: OKX 51400 reads "Order cancellation failed
            #   as the order has been filled, canceled or does not exist."
            return "gone" if acked else "retry"
        if "unknown order" in msg or "order does not exist" in msg or "order not found" in msg:
            return "gone" if acked else "retry"
        if isinstance(err, ccxt.OperationRejected):
            # e.g. already filled — cancelling is permanently impossible.
            return "reject"
        if "mandatory parameter 'orderid' was not sent" in msg:
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
        # The framework speaks TOTAL quantity (incl. filled); translate to the venue's
        # amend dialect here. "replacement" (declared on the exchange subclass): the venue
        # modify is a cancel+replace and its amount is the replacement's size = remaining.
        wire_price = price if price is not None else order.price
        total = quantity if quantity is not None else order.quantity
        if getattr(self._em.exchange, "AMEND_QUANTITY_DIALECT", "total") == "replacement":
            wire_amount = total - order.filled_quantity
        else:
            wire_amount = total
        if wire_amount <= 0:
            raise ValueError(
                f"update_order for {order.client_order_id}: effective amend amount "
                f"{wire_amount} <= 0 (total {total}, filled {order.filled_quantity})"
            )
        self._spawn(
            self._update_async(
                order.client_order_id,
                order.venue_order_id,
                instrument_to_ccxt_symbol(order.instrument),
                order.side.lower(),
                order.type.lower(),
                wire_price,
                wire_amount,
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
        wire_price: float | None,
        wire_amount: float | None,
        requested_price: float | None,
        requested_total: float | None,
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
            await self._acquire_endpoint_budget("edit_order")
            if self._em.exchange.has.get("editOrder", False):
                r = await self._edit_order_direct(
                    client_order_id, vid, symbol, side, order_type, wire_price, wire_amount
                )
            else:
                r = await self._update_via_cancel_recreate(client_order_id, venue_order_id, wire_price, wire_amount)
        except RateLimitGateTimeout as e:
            self._emit_update_rejected(client_order_id, venue_order_id, e)
            return
        except _VENUE_VERDICT_ERRORS as e:
            # Venue verdict — must precede the bare NetworkError catch (see _submit_async).
            self._report_order_limit_hit(e)
            self._emit_update_rejected(client_order_id, venue_order_id, e)
            return
        except ccxt.NetworkError as e:
            # Transient: UNKNOWN whether the edit landed. Leave the order PENDING_UPDATE
            # inflight for AM to reconcile rather than emitting a terminal reject.
            self._report_order_limit_hit(e)
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
        if isinstance(r, dict) and r.get("status") == "canceled":
            # Binance/Gate amend to a total at/below executedQty CANCELS the order
            # (documented, no error). Surface the truth: this is a cancel, not an update.
            logger.warning(
                f"[{self.exchange_name}] edit of {client_order_id} was a silent venue cancel "
                f"(amend total at/below executed) — emitting cancel"
            )
            self._emit_canceled_from_response(client_order_id, vid, r)
            return
        self.send(
            OrderUpdatedEvent(
                instrument=None,
                client_order_id=client_order_id,
                venue_order_id=vid,  # str | None — never coerce to "" (AM would index a bogus id)
                new_price=requested_price,
                new_quantity=requested_total,
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
        # TODO(account-mgmt): this path is itself a REPLACEMENT — when wired, size the recreate
        # at total - filled, not the total-dialect `quantity` (wire_amount) it currently receives.
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
                cause=reject_cause_of(error),
            )
        )

    # ------------------------------------------------------------------ #
    # Client id
    # ------------------------------------------------------------------ #
    def make_client_id(self, suggested: str) -> str:
        """Return the framework client id: ``FRAMEWORK_CID_PREFIX`` enforced, then mapped into
        Binance's charset/length (``conforming_client_id``).

        ``classify_origin`` keys order-origin detection on the prefix, so the connector
        guarantees it. Venues with stricter rules (OKX) override this.
        """
        return conforming_client_id(with_framework_prefix(suggested))

    # ------------------------------------------------------------------ #
    # Leverage / margin
    # ------------------------------------------------------------------ #
    def set_instrument_leverage(self, instrument: Instrument, leverage: float) -> None:
        """Request the configured leverage. Never blocks and never reports back.

        Three things happen before anything reaches the venue, none of them blocking: the
        request is clamped to the venue maximum, skipped entirely when the venue already has
        that value, and otherwise sent off-thread. A wide universe called this once per
        instrument on the ProcessorThread and each round trip cost ~1.2s; skipping the
        unchanged ones removes most calls outright, and the rest no longer hold the caller.

        With no cap and no cache entry (first tick after connect, or a venue whose meta read
        failed) both checks are skipped and the request goes as asked — the venue still
        enforces its own cap, and its refusal arrives as a VenueOperationError on on_error.
        """
        symbol = instrument_to_ccxt_symbol(instrument)
        # - whole numbers throughout: the venue takes an integer (Binance answers a float
        #   with -1102 "Mandatory parameter 'leverage' ... malformed", and ccxt types it
        #   int), and comparing int to int keeps the cache decisions readable.
        wanted = int(leverage)
        if wanted != leverage:
            logger.warning(
                f"[{self.exchange_name}] {instrument.symbol}: leverage {leverage} is not a whole "
                f"number; the venue takes integers, requesting {wanted}"
            )
        # - through the getter, not the cache: a venue whose cap is not in _leverage_cache at all
        #   (OKX reads it off the market metadata) must still clamp
        maximum = self.get_max_instrument_leverage(instrument)
        if maximum is not None and wanted > maximum:
            logger.warning(
                f"[{self.exchange_name}] {instrument.symbol}: leverage {wanted} exceeds the venue "
                f"maximum {maximum}; requesting {maximum}"
            )
            wanted = int(maximum)
        cached = self._leverage_cache.get(symbol)
        if cached is not None and cached.configured is not None and cached.configured == wanted:
            logger.info(
                f"[{self.exchange_name}] {instrument.symbol}: venue already at leverage "
                f"{cached.configured}, not sending {wanted}"
            )
            return
        logger.info(
            f"[{self.exchange_name}] {instrument.symbol}: sending leverage {wanted} "
            f"(cached {cached.configured} / max {maximum})"
            if cached is not None
            else f"[{self.exchange_name}] {instrument.symbol}: sending leverage {wanted} (max {maximum})"
        )
        self._spawn(self._do_set_leverage(instrument, symbol, wanted))

    def _report_leverage_failure(self, instrument: Instrument, leverage: int, error: Exception) -> None:
        """The caller of set_instrument_leverage is long gone, so the verdict rides the channel."""
        logger.error(f"[{self.exchange_name}] Failed to set leverage {leverage} for {instrument.symbol}: {error}")
        self.channel.send(
            create_error_event(
                VenueOperationError(
                    timestamp=self._time.time(),
                    message=f"set leverage {leverage} for {instrument.symbol}",
                    level=ErrorLevel.MEDIUM,
                    error=error,
                    operation="set_instrument_leverage",
                    instrument=instrument,
                )
            )
        )

    async def _do_set_leverage(self, instrument: Instrument, symbol: str, leverage: int) -> None:
        try:
            await self._acquire_endpoint_budget("set_leverage")
            await self._em.exchange.set_leverage(leverage, symbol)
        except _ORDER_RATE_LIMIT_ERRORS as e:
            # The write is fire-and-forget, so nothing re-sends a leverage that never landed.
            # Retry once, but only where the gate paces it — a venue with no `leverage` pool would
            # re-send instantly, and binance escalates bans for requests sent after a 429.
            self._report_order_limit_hit(e, pool_name="leverage")
            limiter = self._em.rate_limiter
            if limiter is None or not limiter.is_gate_closed("leverage"):
                self._report_leverage_failure(instrument, leverage, e)
                return
            logger.warning(f"[{self.exchange_name}] leverage {leverage} for {instrument.symbol} rate limited; retrying")
            try:
                await self._acquire_endpoint_budget("set_leverage")
                await self._em.exchange.set_leverage(leverage, symbol)
            except Exception as retry_error:  # noqa: BLE001 — report the retry's verdict, not the first
                # the gate reopened while we waited; a second refusal must re-close it
                self._report_order_limit_hit(retry_error, pool_name="leverage")
                self._report_leverage_failure(instrument, leverage, retry_error)
                return
        except Exception as e:  # noqa: BLE001 — the caller is long gone; report on the channel
            self._report_leverage_failure(instrument, leverage, e)
            return
        # - adopt what we just set, so the next call for the same value is skipped without
        #   waiting for the poller; the poller corrects it if the venue disagrees
        # The cap belongs to the bracket we just left, so the new one is read here rather than
        # carried or dropped: dropping left the position reading null for up to an hour, until
        # the sweep — a snapshot only copies a cap across when the Differ flags that position
        # for a size or margin change, which a leverage edit does not cause.
        max_notional, margin_mode = await self._read_leverage_row(symbol)
        # read after the await: an hourly sweep landing inside it must not be rolled back here
        cached = self._leverage_cache.get(symbol)
        self._leverage_cache[symbol] = _LeverageInfo(
            configured=leverage,
            maximum=cached.maximum if cached is not None else None,
            max_notional=max_notional,
            margin_mode=margin_mode or (cached.margin_mode if cached is not None else None),
        )
        # the cache is private to the connector; the Position is what every reader sees, and
        # only the snapshot writes it — so announce the ack rather than wait for one
        self.channel.send(
            create_venue_settings_event(
                VenueSettingsUpdate(instrument, leverage=float(leverage), max_notional=max_notional)
            )
        )

    async def _read_leverage_row(self, symbol: str) -> tuple[float | None, str | None]:
        """One symbol's notional cap at its current leverage, and its margin mode, from symbolConfig.

        The cap is None on any failure — and on a venue without the endpoint — which the AM reads
        as "not known", clearing the stale value rather than keeping a cap from the wrong bracket.
        """
        if not self._em.exchange.has.get("fetchLeverages"):
            return None, None
        try:
            rows = await self._em.exchange.fetch_leverages([symbol])
        except Exception as e:  # noqa: BLE001 — the ack itself already landed
            logger.warning(f"[{self.exchange_name}] cap read for {symbol} after the leverage ack: {e}")
            return None, None
        leverage_rows = list(rows.values()) if isinstance(rows, dict) else rows
        settings = ccxt_extract_leverage_settings(leverage_rows)
        return settings.get(symbol, (None, None))[1], ccxt_extract_margin_modes(leverage_rows).get(symbol)

    def _start_leverage_poller(self) -> None:
        if self._leverage_future is None or self._leverage_future.done():
            self._leverage_future = self._loop.submit(self._leverage_poll_loop())
            self._leverage_future.add_done_callback(self._log_spawn_error)

    async def _leverage_poll_loop(self) -> None:
        """Refresh on connect, then hourly. Sleeps in <=60s slices so a cancel at
        disconnect lands promptly instead of after the full interval."""
        while True:
            await self._refresh_leverage_cache()
            remaining = LEVERAGE_REFRESH_INTERVAL_S
            while remaining > 0:
                await asyncio.sleep(min(60.0, remaining))
                remaining -= 60.0

    async def _refresh_leverage_cache(self) -> None:
        """Two venue reads, both whole-universe: the symbolConfig row per symbol and the
        leverage tiers. Either may be unsupported or fail — each is caught on its own so one
        missing half does not blank the other.

        symbolConfig carries the configured leverage AND the notional cap, both with no open
        position required — which is what lets the cap be known for a flat instrument.

        The tier list is per-notional (Binance's brackets); the cached maximum is the
        largest of them, i.e. the cap at the smallest position. A request under that can
        still be refused once the position grows into a lower bracket, and that refusal
        arrives as a VenueOperationError like any other.
        """
        ex = self._em.exchange
        configured: dict[str, float] = {}
        notionals: dict[str, float] = {}
        maxima: dict[str, float] = {}
        modes: dict[str, str] = {}
        configured_read = True
        # - annotated locals: ccxt's BASE Exchange declares both methods as an unconditional
        #   `raise NotSupported`, so their inferred return type is NoReturn. Only the venue
        #   subclasses return dicts, and `ex` is typed as the base. A venue without the
        #   override does raise, and the except below is what handles it.
        try:
            rows: dict[str, Any] = await ex.fetch_leverages()
            leverage_rows = list(rows.values()) if isinstance(rows, dict) else rows
            settings = ccxt_extract_leverage_settings(leverage_rows)
            for symbol, (leverage, max_notional) in settings.items():
                if leverage is not None:
                    configured[symbol] = int(leverage)
                if max_notional is not None:
                    notionals[symbol] = max_notional
            modes = ccxt_extract_margin_modes(leverage_rows)
        except Exception as e:  # noqa: BLE001
            configured_read = False
            logger.debug(f"[{self.exchange_name}] configured-leverage read failed: {type(e).__name__}: {e}")
        try:
            tiers_by_symbol: dict[str, Any] = await ex.fetch_leverage_tiers()
            for symbol, tiers in tiers_by_symbol.items():
                levels = [t["maxLeverage"] for t in tiers if t.get("maxLeverage") is not None]
                if levels:
                    maxima[symbol] = int(max(levels))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.exchange_name}] leverage-tier read failed: {type(e).__name__}: {e}")

        if not configured and not maxima and not notionals:
            return
        # A configured value that moved since the last sweep is an EXTERNAL change (the venue UI,
        # another client) — the ack path never sees it, so without this it would reach the
        # Position only if a snapshot happened to carry it.
        for symbol, value in configured.items():
            held = self._leverage_cache.get(symbol)
            if held is not None and held.configured is not None and held.configured != value:
                # Announce every symbol the venue reports and let the AM filter: it drops a
                # non-ack update for an instrument with no position. Resolving off the memo
                # instead would silently skip a position restored at boot that this session has
                # not traded — the memo is written only by the order/deal/funding paths — which
                # is exactly the held instrument whose external change we need to deliver.
                try:
                    instrument = self._instrument_for_symbol(symbol)
                except Exception:  # noqa: BLE001 — the venue does not know it either
                    continue
                self.channel.send(
                    create_venue_settings_event(
                        VenueSettingsUpdate(instrument, leverage=float(value), max_notional=notionals.get(symbol))
                    )
                )
        rebuilt = {
            symbol: _LeverageInfo(
                configured=configured.get(symbol),
                maximum=maxima.get(symbol),
                max_notional=notionals.get(symbol),
                margin_mode=modes.get(symbol),
            )
            for symbol in configured.keys() | maxima.keys() | notionals.keys() | modes.keys()
        }
        if configured_read:
            # rebuilt wholesale, so a symbol the venue stopped reporting leaves the cache with it
            self._leverage_cache = rebuilt
        else:
            # The `configured` read failed (bybit has no fetch_leverages), so this sweep cannot
            # speak for it: a wholesale rebuild would erase what the write path adopted and evict
            # every symbol the tier read left out — on bybit, everything past its pagination cap.
            # max_notional and margin_mode ride the same failed read, so they are preserved too.
            for symbol, info in rebuilt.items():
                held = self._leverage_cache.get(symbol)
                self._leverage_cache[symbol] = (
                    replace(
                        info,
                        configured=held.configured,
                        max_notional=held.max_notional,
                        margin_mode=held.margin_mode,
                    )
                    if held is not None
                    else info
                )
        logger.info(
            f"[{self.exchange_name}] leverage cache refreshed: {len(configured)} configured, "
            f"{len(maxima)} maxima, {len(notionals)} notional caps"
        )

    def convert_currency(
        self,
        from_currency: str,
        to_currency: str,
        amount: float,
        *,
        limit_price: float | None = None,
        max_slippage_bps: float = 10.0,
    ) -> str:
        """Swap ``amount`` of one currency for another on the venue's own market for the pair.

        Returns immediately with the conversion's id — the venue round trip runs on the
        exchange loop and the outcome arrives as a ``CurrencyConversionEvent``, so the caller
        (the ProcessorThread) keeps draining its queue. EXACTLY ONE event follows every call
        that returned an id, failures included, so a caller tracking a pending conversion
        always gets its answer. Only argument mistakes raise here.

        One IOC attempt, priced to protect rather than to chase: nothing rests on the book
        afterwards, so the event is the whole outcome and there is nothing to cancel or
        reconcile. The trade is deliberately NOT registered with the AccountManager — it moves
        cash, not exposure, and the balances arrive with the next account snapshot.

        ``amount`` is denominated in ``from_currency``, whichever side of the venue's pair that
        happens to be; only one direction is normally listed (USDC/USDT, never USDT/USDC), so
        buying the base is how you spend the quote. ``limit_price`` (in the market's own quote
        terms) bounds the fill absolutely — the guard that matters when a stablecoin depegs,
        where a relative ``max_slippage_bps`` off a broken book would still convert.
        """
        if not amount > 0:
            raise ValueError(f"[{self.exchange_name}] conversion amount must be positive, got {amount}")
        if from_currency.upper() == to_currency.upper():
            raise ValueError(f"[{self.exchange_name}] cannot convert {from_currency} into itself")
        # unique per call: make_client_id adds no uniqueness (order ids get it from the
        # TradingManager's store), and this one is the venue's
        # clientOrderId — a constant would be rejected as a duplicate on the second call
        suffix = uuid.uuid4().hex[:8]
        conversion_id = self.make_client_id(f"conv{from_currency.upper()[:6]}{to_currency.upper()[:6]}{suffix}")
        self._spawn(
            self._convert_currency(conversion_id, from_currency, to_currency, amount, limit_price, max_slippage_bps)
        )
        return conversion_id

    async def _convert_currency(
        self,
        conversion_id: str,
        from_currency: str,
        to_currency: str,
        amount: float,
        limit_price: float | None,
        max_slippage_bps: float,
    ) -> None:
        """Run one conversion to a terminal record and emit it. Never raises: a failure the
        caller cannot see is a conversion it would wait on forever."""
        try:
            record = await self._run_conversion(
                conversion_id, from_currency, to_currency, amount, limit_price, max_slippage_bps
            )
            logger.info(f"[{self.exchange_name}] {record.status} conversion {record.to_dict()}")
        except Exception as e:  # noqa: BLE001 — every failure is reported, not raised
            reason = f"{type(e).__name__}: {e}"
            logger.error(f"[{self.exchange_name}] conversion {conversion_id} FAILED — {reason}")
            record = CurrencyConversion(
                conversion_id=conversion_id,
                exchange=self.exchange_name,
                from_currency=from_currency,
                to_currency=to_currency,
                requested=amount,
                filled_from=0.0,
                filled_to=0.0,
                status="FAILED",
                failure_reason=reason,
            )
        self.send(CurrencyConversionEvent(instrument=None, conversion=record))

    async def _run_conversion(
        self,
        conversion_id: str,
        from_currency: str,
        to_currency: str,
        amount: float,
        limit_price: float | None,
        max_slippage_bps: float,
    ) -> CurrencyConversion:
        ex = self._em.exchange
        if not ex.markets:
            # a trading-only connector (market data via xdata) may never have loaded them
            await ex.load_markets()
        symbol, side = self._conversion_market(from_currency, to_currency)

        if limit_price is None:
            # per-symbol, never the venue-wide fetch_bids_asks: that resolves its market type
            # from defaultType (swap on a PM venue) and comes back without the spot pair
            ticker = await ex.fetch_ticker(symbol)
            top = ticker["bid"] if side == "sell" else ticker["ask"]
            slippage = max_slippage_bps / 10_000
            limit_price = top * (1 - slippage) if side == "sell" else top * (1 + slippage)

        price = float(ex.price_to_precision(symbol, limit_price))
        # amount is in from_currency: already the base on a sell, the budget to spend on a buy
        # (bounded by the limit price, and truncated to the step so it can never exceed it).
        qty = float(ex.amount_to_precision(symbol, amount if side == "sell" else amount / price))

        market = ex.market(symbol)
        min_amount = market["limits"]["amount"]["min"] or 0.0
        min_notional = market["limits"]["cost"]["min"] or 0.0
        if qty < min_amount or qty * price < min_notional:
            raise ValueError(
                f"{amount} {from_currency} is below {symbol}'s floor "
                f"(min amount {min_amount}, min notional {min_notional})"
            )

        response = await ex.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=qty,
            price=price,
            params={"timeInForce": "IOC", "clientOrderId": conversion_id},
        )
        filled = float(response.get("filled") or 0.0)
        cost = float(response.get("cost") or 0.0)
        if filled <= 0:
            status = "UNFILLED"
        elif filled >= qty * (1 - 1e-9):
            status = "FILLED"
        else:
            status = "PARTIAL"
        if filled > 0:
            # the venue moved cash the poller has not seen yet; a caller reading balances on
            # its next tick must not act on the pre-conversion ones
            self.request_snapshot(include_orders=False)
        return CurrencyConversion(
            conversion_id=conversion_id,
            exchange=self.exchange_name,
            from_currency=from_currency,
            to_currency=to_currency,
            requested=amount,
            filled_from=filled if side == "sell" else cost,
            filled_to=cost if side == "sell" else filled,
            status=status,
            avg_price=response.get("average"),
            venue_order_id=str(response["id"]) if response.get("id") is not None else None,
        )

    def _conversion_market(self, from_currency: str, to_currency: str) -> tuple[str, str]:
        """The venue's market for the pair and the side that spends ``from_currency``."""
        ex = self._em.exchange
        for symbol, side in ((f"{from_currency}/{to_currency}", "sell"), (f"{to_currency}/{from_currency}", "buy")):
            if symbol in ex.markets:
                return symbol, side
        raise ValueError(f"no market to convert {from_currency} -> {to_currency}")

    def set_margin_mode(self, instrument: Instrument, mode: str) -> bool:
        try:
            symbol = instrument_to_ccxt_symbol(instrument)
            ex = self._em.exchange
            fn = getattr(ex, "set_margin_mode", None) or getattr(ex, "set_margin_type", None)
            if fn is None:
                logger.error(f"[{self.exchange_name}] does not support set_margin_mode")
                return False
            self._run_sync(fn(mode, symbol))
            normalized = normalize_margin_mode(mode)
            if normalized is not None:
                self.channel.send(create_venue_settings_event(VenueSettingsUpdate(instrument, margin_mode=normalized)))
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] Failed to set margin mode {mode} for {instrument.symbol}: {e}")
            return False

    # Authoritative venue pull for the per-instrument settings
    def get_instrument_leverage(self, instrument: Instrument) -> float | None:
        """The configured leverage, from the poller's cache only. Never blocks.

        Cache-only because the 5s state snapshot reads this for every universe instrument on
        the ProcessorThread: a venue round trip per instrument would park that thread for the
        whole universe. The hourly poller (``_refresh_leverage_cache``, also run on connect)
        and the write path's adopt-on-send keep the cache warm; None means it has not landed
        yet, which is the interface's "unknown".
        """
        cached = self._leverage_cache.get(instrument_to_ccxt_symbol(instrument))
        return float(cached.configured) if cached is not None and cached.configured is not None else None

    def get_max_instrument_leverage(self, instrument: Instrument) -> float | None:
        """The venue's published maximum, from the poller's cache.

        No fallback: the only per-symbol source is ``fetch_leverage_tiers``, which the
        poller already sweeps for the whole universe every hour. None here means that
        sweep has not run yet or the venue does not publish tiers — and the write path
        treats None as "send it unclamped and let the venue decide".
        """
        cached = self._leverage_cache.get(instrument_to_ccxt_symbol(instrument))
        return float(cached.maximum) if cached is not None and cached.maximum is not None else None

    def get_max_instrument_notional(self, instrument: Instrument) -> float:
        """The venue's notional cap, from the poller's cache only. Never blocks.

        Cache-only for the same reason as ``get_instrument_leverage``: the 5s state snapshot
        reads it for every universe instrument on the ProcessorThread, and the symbolConfig /
        position-row pulls this used to fall through to each blocked it on a venue round trip.
        ``inf`` is the interface's "no cap, or not populated yet".
        """
        cached = self._leverage_cache.get(instrument_to_ccxt_symbol(instrument))
        return cached.max_notional if cached is not None and cached.max_notional is not None else float("inf")

    def get_margin_mode(self, instrument: Instrument) -> str | None:
        """Cache-only, like the leverage getters; the sweep fills it from symbolConfig, which
        covers flat symbols."""
        cached = self._leverage_cache.get(instrument_to_ccxt_symbol(instrument))
        return cached.margin_mode if cached is not None else None

    def get_adl_level(self, instrument: Instrument) -> int | None:
        row = self._fetch_position_row(instrument)
        if row is None:
            return None
        raw = row.get("info") or {}
        # - v3 positionRisk renamed the field to `adl`; v2 (params.useV2) still spells it `adlQuantile`.
        adl = info_float(raw, "adl")
        if adl is None:
            adl = info_float(raw, "adlQuantile")
        return int(adl) if adl is not None else None

    async def _fill_leverage_settings(self, positions: list[Position]) -> None:
        """Fill ``leverage``/``max_notional`` from symbolConfig for positions the position
        payload left blank (Binance v3 carries neither — see ccxt_extract_leverage_settings).

        Best-effort: a failure leaves the fields None, exactly as before. Never overwrites a
        value the position payload did supply, so venues that do carry them are untouched.
        """
        if not positions or not self._em.exchange.has.get("fetchLeverages"):
            return
        if all(p.leverage is not None and p.max_notional is not None for p in positions):
            return
        try:
            rows = await self._em.exchange.fetch_leverages()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.exchange_name}] fetch_leverages failed: {e}")
            return
        settings = ccxt_extract_leverage_settings(list(rows.values()) if isinstance(rows, dict) else rows)
        for pos in positions:
            leverage, max_notional = settings.get(instrument_to_ccxt_symbol(pos.instrument), (None, None))
            if pos.leverage is None and leverage is not None:
                pos.leverage = leverage
            if pos.max_notional is None and max_notional is not None:
                pos.max_notional = max_notional

    def _fetch_position_row(self, instrument: Instrument) -> dict[str, Any] | None:
        """Blocking single-symbol position pull; None on error or when no position is held."""
        try:
            symbol = instrument_to_ccxt_symbol(instrument)
            rows = self._run_sync(self._em.exchange.fetch_positions([symbol]))
            return rows[0] if rows else None
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.exchange_name}] fetch position for {instrument.symbol}: {e}")
            return None

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
        if ex.has.get("watchBalance") and self._wants_ws_balance_push:
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
                    venue_order_id=order.venue_order_id,
                    last_update_time=order.last_update_time,
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
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    last_update_time=order.last_update_time,
                )
            )
            return
        if status == OrderStatus.EXPIRED:
            self.send(
                OrderExpiredEvent(
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    venue_order_id=order.venue_order_id,
                    last_update_time=order.last_update_time,
                )
            )
            return
        if status == OrderStatus.REJECTED:
            self.send(
                OrderRejectedEvent(
                    instrument=instrument,
                    client_order_id=order.client_order_id,
                    reason="rejected by venue",
                    last_update_time=order.last_update_time,
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
                    venue_order_id=order.venue_order_id,
                    last_update_time=order.last_update_time,
                    fill=None,
                    venue_filled_quantity=order.filled_quantity,
                    venue_avg_price=order.avg_fill_price,
                )
            )
            return
        last = len(deals) - 1
        for i, deal in enumerate(deals):
            # amt = THIS trade's qty; order_filled = the ORDER's cumulative filled/total sampled now
            # (order-level, NOT a per-trade running sum) — so a re-handed earlier trade shows the
            # order's current progress, not its own point in history.
            self._dbg.debug(
                "emit fill {} amt={} order_filled={}/{} tid={}",
                instrument.symbol,
                deal.amount,
                order.filled_quantity,
                order.quantity,
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
                        venue_order_id=order.venue_order_id,
                        last_update_time=order.last_update_time,
                        fill=deal,
                    )
                )
            else:
                self.send(
                    OrderFilledEvent(
                        instrument=instrument,
                        client_order_id=order.client_order_id,
                        venue_order_id=order.venue_order_id,
                        last_update_time=order.last_update_time,
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
        # Binance refuses the fetch without. A stop/conditional order lives on the venue's
        # trigger surface, so the fetch must target it (else Binance answers -2013 not-found
        # for a live order -> false reject).
        is_trigger = order.type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT)
        self._spawn(
            self._order_status_async(
                order.client_order_id,
                order.venue_order_id,
                instrument_to_ccxt_symbol(order.instrument),
                order.instrument,
                is_trigger,
            )
        )

    async def _order_status_async(
        self,
        client_order_id: str | None,
        venue_order_id: str | None,
        symbol: str,
        instrument: Instrument,
        is_trigger: bool = False,
    ) -> None:
        # Prefer the venue id; a cloid-only fetch must use ccxt's client-order-id variant —
        # Binance rejects a cloid passed as orderId with -1102 BadRequest, NOT OrderNotFound.
        vid = venue_order_id
        lookup_id = vid or client_order_id
        ex = self._em.exchange
        try:
            if vid is not None:
                raw = (
                    await ex.fetch_order(vid, symbol, params={"trigger": True})
                    if is_trigger
                    else await ex.fetch_order(vid, symbol)
                )
            else:
                assert client_order_id is not None
                raw = (
                    await ex.fetch_order_with_client_order_id(client_order_id, symbol, params={"trigger": True})
                    if is_trigger
                    else await ex.fetch_order_with_client_order_id(client_order_id, symbol)
                )
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

    def _start_funding_poller(self) -> None:
        if self._funding_future is None or self._funding_future.done():
            self._funding_future = self._loop.submit(self._funding_poll_loop())
            self._funding_future.add_done_callback(self._log_spawn_error)

    async def _funding_poll_loop(self) -> None:
        connect_time = self._time.time()
        while True:
            target = self._next_funding_poll_at(self._time.time())
            while (now := self._time.time()) < target:
                await asyncio.sleep(min(float((target - now) / np.timedelta64(1, "s")), 60.0))
            await self._funding_payments_async(self._funding_since(connect_time, self._time.time()))

    @staticmethod
    def _next_funding_poll_at(now: dt_64) -> dt_64:
        # nearest hh:10 strictly after now (10min past every hour, incl. the current one)
        slot = now.astype("datetime64[h]").astype("datetime64[ns]") + np.timedelta64(10, "m")
        return slot if slot > now else slot + np.timedelta64(1, "h")

    @staticmethod
    def _funding_since(connect_time: dt_64, now: dt_64) -> dt_64:
        return max(connect_time, now - np.timedelta64(2, "h"))

    async def _funding_payments_async(self, since: dt_64) -> None:
        """Fetch the account's funding settlements since ``since`` and emit one
        FundingPaymentEvent per income record (one poller cycle).

        One account-wide ``fetch_funding_history`` call (FUNDING_FEE income records) —
        overlapping poll windows are expected, the AM reducer's bucket dedup absorbs the
        duplicates. Errors are logged, not raised — the next cycle re-covers the window.
        A single page (1000 records) is orders of magnitude above any poll window;
        hitting it is logged as possible truncation.
        """
        since_ms = int(since.astype("datetime64[ms]").astype("int64"))
        logger.debug(f"[{self.exchange_name}] funding sweep: fetch_funding_history since {since}")
        try:
            raw_records = await self._em.exchange.fetch_funding_history(since=since_ms)
        except NetworkError as e:
            logger.warning(f"[{self.exchange_name}] funding history fetch since {since} failed: {e}")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.exchange_name}] error fetching funding history since {since}: {e}")
            return
        if len(raw_records) >= 1000:
            logger.warning(
                f"[{self.exchange_name}] funding history since {since} hit the 1000-record page; possible truncation"
            )
        for raw in raw_records:
            symbol, ts_ms, amount = raw.get("symbol"), raw.get("timestamp"), raw.get("amount")
            if symbol is None or ts_ms is None or amount is None:
                logger.debug(f"[{self.exchange_name}] funding record missing fields, skipped: {raw}")
                continue
            try:
                instrument = self._instrument_for_symbol(symbol)
            except CcxtSymbolNotRecognized:
                logger.debug(f"[{self.exchange_name}] funding record for unknown symbol {symbol}, skipped")
                continue
            self.send(FundingPaymentEvent(instrument=instrument, time=recognize_time(int(ts_ms)), amount=float(amount)))

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

    async def _fetch_trigger_open_orders(self) -> list[dict]:
        """Fetch untriggered stop/conditional ("algo") open orders.

        Binance USDⓂ serves these from a SEPARATE endpoint, disjoint from the regular
        open-orders one — the unified ccxt ``params={'trigger': True}`` routes to it.
        A venue with no such surface raises NotSupported/BadRequest → treated as "no
        trigger orders" ([]); transient errors propagate (the caller leaves order
        reconcile for the next tick rather than orphaning unseen stops).
        """
        try:
            return await self._em.exchange.fetch_open_orders(params={"trigger": True})
        except (ccxt.NotSupported, ccxt.BadRequest) as e:
            logger.debug(f"[{self.exchange_name}] snapshot: no trigger open-orders surface ({e}); regular only")
            return []

    def _merge_open_orders(self, raw_orders: list[dict], raw_trigger_orders: list[dict]) -> list[Order]:
        """Merge regular + trigger raw open orders into framework Orders (dedup by venue id).

        A row on an unrecognized symbol is skipped, loudly: reconcile reads the order list as
        venue truth, so a skipped row reads as cancelled at the venue.
        """
        open_orders: list[Order] = []
        seen: set[str] = set()
        for raw in [*raw_orders, *raw_trigger_orders]:
            try:
                instrument = self._instrument_for_symbol(raw["symbol"])
            except CcxtSymbolNotRecognized:
                # Venue symbol goes in as a positional arg (may contain markup loguru's
                # colorizer rejects in the format string).
                logger.error(
                    "[{}] snapshot: unrecognized symbol {}; open order skipped", self.exchange_name, raw.get("symbol")
                )
                continue
            order = ccxt_convert_order_info(instrument, raw, framework_prefix=self.cid_framework_prefix)
            key = order.venue_order_id or order.client_order_id
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            open_orders.append(order)
        return open_orders

    def _loop_tag(self) -> str:
        """Which loop and thread this call is actually running on, next to the loop the
        ccxt exchange was built on. Several loops coexist in one process (one per ccxt
        exchange manager, the control server's, one per CcxtStorage); a call reaching the
        exchange from a foreign loop is how an fd ends up registered twice
        ("File descriptor N is used by transport ...")."""
        try:
            running = id(asyncio.get_running_loop())
        except RuntimeError:
            running = 0
        own = id(getattr(self._em.exchange, "asyncio_loop", None))
        return f"loop=0x{running:x} exchange_loop=0x{own:x}{' FOREIGN' if running != own else ''} thread={threading.current_thread().name}"

    def request_snapshot(self, include_orders: bool = True) -> None:
        self._spawn(self._snapshot_async(include_orders))

    async def _snapshot_async(self, include_orders: bool = True) -> None:
        """Fetch account state concurrently and emit a snapshot.

        include_orders=True  -> open orders + algo/trigger orders + positions + balances (startup
        discovery + periodic sweep). include_orders=False -> positions + balances ONLY (steady
        state): the open-orders / algo legs carry the highest REST weight, and with Qubx rate
        limiting off they share ccxt's serialised throttler with order placement — so steady
        snapshots skip them (open_orders=None -> reconcile leaves order state untouched; orders
        are tracked via the WS stream + the periodic full sweep).

        Reconcile applies a snapshot as authoritative, so a leg that failed to fetch skips the
        emit entirely and the AM asks again on its next tick. A single row that will not convert
        is skipped instead (loudly) — one unresolvable symbol must not stall reconcile for the
        whole exchange. Errors are logged, never raised.
        """
        ex = self._em.exchange
        as_of: dt_64 = self._time.time()
        try:
            if include_orders:
                raw_orders, raw_trigger_orders, raw_positions, raw_balance = await asyncio.gather(
                    ex.fetch_open_orders(),
                    self._fetch_trigger_open_orders(),
                    ex.fetch_positions(),
                    ex.fetch_balance(),
                )
                open_orders = self._merge_open_orders(raw_orders, raw_trigger_orders)
            else:
                raw_positions, raw_balance = await asyncio.gather(ex.fetch_positions(), ex.fetch_balance())
                open_orders = None  # - not observed this tick -> reconcile skips order diffing
            positions = ccxt_convert_positions(raw_positions, ex.name, ex.markets)
            await self._fill_leverage_settings(positions)
            balances = self._convert_balances(raw_balance)
            figures = self._extract_venue_figures(raw_balance)
        except Exception as e:  # noqa: BLE001 — AM retries on its next snapshot tick
            # Venue exception text goes in as a positional arg (may contain HTML/markup that
            # loguru's colorizer rejects when it appears in the format string).
            logger.error(
                "[{}] snapshot: {}: {}; not emitting a partial snapshot [{}]",
                self.exchange_name,
                type(e).__name__,
                e,
                self._loop_tag(),
            )
            return
        self.send(
            AccountSnapshotEvent(
                instrument=None,
                snapshot=AccountSnapshot(
                    exchange=self.exchange_name,
                    as_of=as_of,
                    open_orders=open_orders,
                    positions=positions,
                    balances=balances,
                    equity=figures.equity,
                    available_margin=figures.available_margin,
                    margin_ratio=figures.margin_ratio,
                    withdrawable=figures.withdrawable,
                    total_maint_margin=figures.total_maint_margin,
                    total_initial_margin=figures.total_initial_margin,
                    collateral_equity=figures.collateral_equity,
                ),
            )
        )

    def _convert_balances(self, raw_balance: dict[str, Any]) -> list[Balance]:
        """Convert a ccxt fetch_balance response to framework Balances.

        Base impl reads ccxt's canonical ``total``/``used`` maps; venue subclasses
        override when ccxt's mapping is wrong for the framework (see OKX).
        """
        return ccxt_convert_balance(raw_balance, self.exchange_name)

    def _extract_venue_figures(self, raw_balance: dict[str, Any]) -> VenueFigures:
        """Venue account figures from the raw account payload, as a ``VenueFigures``.

        ccxt has no unified account-figures schema, so the base impl reads the
        Binance-futures account fields carried through in ``info`` (both fapi v2 and
        v3 account payloads carry them top-level): ``totalMarginBalance`` (wallet +
        unrealized PnL = account equity), ``availableBalance`` (margin available for
        new positions), ``maxWithdrawAmount`` (maximum amount for transfer out),
        ``totalMaintMargin`` and ``totalInitialMargin`` (account-level margin
        requirements; the initial figure includes open-order margin, and in single-asset
        mode both cover the USDT asset only — a non-USDT-margined position reads as zero
        there). Binance reports no direct margin ratio — left None so AM derives it.
        Venues whose payload lacks these keys yield all-None and AM derives every
        metric; subclasses override for venue-specific payloads.
        """
        info = raw_balance.get("info")
        if not isinstance(info, dict):
            return VenueFigures()
        return VenueFigures(
            equity=info_float(info, "totalMarginBalance"),
            available_margin=info_float(info, "availableBalance"),
            withdrawable=info_float(info, "maxWithdrawAmount"),
            total_maint_margin=info_float(info, "totalMaintMargin"),
            total_initial_margin=info_float(info, "totalInitialMargin"),
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
        self._start_funding_poller()
        self._start_leverage_poller()
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
        if self._funding_future is not None and not self._funding_future.done():
            self._funding_future.cancel()
        self._funding_future = None
        if self._leverage_future is not None and not self._leverage_future.done():
            self._leverage_future.cancel()
        self._leverage_future = None
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
