"""Applies a typed AccountMessage to one AccountState, driving the order state machine.

Pure state mutation: no connectors, no strategy callbacks, no clock reads (``now`` is a
parameter). The ProcessingManager fires callbacks from the returned ApplyResult; routing
the event to the right state is the AccountManager's job. Every status change goes
through ``reconcile.transition`` (the legality chokepoint), and every handler short-circuits on a
terminal order so late venue events are no-ops. None fields on ApplyResult are the
suppress signal — deduped duplicate fills, late events on terminal orders, and
rejects/lifecycle events for unknown orders all return empty results, so no callback
fires.
"""

from dataclasses import dataclass, field

import numpy as np

from qubx import area_logger
from qubx.core.account_manager import reconcile
from qubx.core.account_manager.state import AccountState
from qubx.core.account_manager.state_machine import can_transition
from qubx.core.basics import (
    EXTERNAL_CID_PREFIX,
    Balance,
    Deal,
    Instrument,
    Order,
    OrderChange,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.events import (
    AccountMessage,
    BalanceUpdateEvent,
    DealEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderLostEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)

# Module logger bound to the "account_manager" area: every line here carries that tag.
# INFO/WARNING show as usual; DEBUG only when QUBX_DEBUG_AREAS includes account_manager.
logger = area_logger("account_manager")


@dataclass
class ApplyResult:
    order: Order | None = None  # status changed -> on_order(order, change)
    order_change: OrderChange | None = None  # paired with order
    deal: Deal | None = None  # new deal applied -> downstream fill consumers
    position: Position | None = None  # position changed
    # balance push applied (the live state Balance) — internal/diff visibility only,
    # fires NO strategy callback by design (balances are read via ctx)
    balance: Balance | None = None
    # positions reconciled by a snapshot (Reconciler path) — PM fires on_position_change per entry
    positions: list[Position] = field(default_factory=list)

    def is_empty(self) -> bool:
        """All fields None — the suppress signal (see module docstring): no callback fires."""
        return (
            self.order is None
            and self.order_change is None
            and self.deal is None
            and self.position is None
            and self.balance is None
            and not self.positions
        )


def _resolve(state: AccountState, event: OrderEvent) -> Order | None:
    # Known by cid (active or terminal-history) or by the venue id it was assigned.
    if (order := state.get_order(event.client_order_id)) is not None:
        return order

    if event.venue_order_id is not None:
        return state.get_order_by_venue_id(event.venue_order_id)

    return None


def _active_order_for(state: AccountState, event: OrderEvent) -> Order | None:
    # Resolve an ACTIVE order by client id, then by venue id — so a reject addressed by
    # venue id alone (a venue-id-only cancel/update) still routes. Active-only (no
    # materialize): a reject has no order to create.
    order = state.get_active_order(event.client_order_id)
    if order is None and event.venue_order_id is not None:
        order = state.get_order_by_venue_id(event.venue_order_id)
    return order


def _materialize_external(
    state: AccountState,
    event: OrderEvent,
    instrument: Instrument,
    now: np.datetime64,
    *,
    status: OrderStatus = OrderStatus.ACCEPTED,
) -> Order:
    # Unknown to us => external order (manual UI / another bot / pre-existing, or a recovered
    # historical trade). A venue lifecycle event always carries the id the venue assigned; fall
    # back to the cid for a stable identity only in the (malformed) case where it doesn't.
    # status defaults to ACCEPTED (a live external order exists at the venue); a recovered
    # historical deal passes FILLED so the order is terminal (audit record, never chased as
    # a missing open order). Terminal orders require last_update_time for eviction.
    venue_id = event.venue_order_id
    order = Order(
        client_order_id=f"{EXTERNAL_CID_PREFIX}{venue_id or event.client_order_id}",
        venue_order_id=venue_id,
        origin=OrderOrigin.EXTERNAL,
        type=OrderType.LIMIT,
        instrument=instrument,
        submitted_at=now,
        quantity=0.0,
        price=0.0,
        side=OrderSide.BUY,
        status=status,
        time_in_force="gtc",
        last_update_time=(event.last_update_time or now) if status.is_terminal else None,
    )
    state.add_order(order)
    return order


def _resolve_or_materialize(
    state: AccountState,
    event: OrderEvent,
    now: np.datetime64,
    *,
    materialize_status: OrderStatus = OrderStatus.ACCEPTED,
) -> Order | None:
    if (order := _resolve(state, event)) is not None:
        return order

    if event.instrument is None:  # can't track a position without an instrument
        return None

    if event.client_order_id is None and event.venue_order_id is None:
        return None  # no identity at all — materializing would collide every such event on "ext:None"

    return _materialize_external(state, event, event.instrument, now, status=materialize_status)


# A late event for an already-terminal order — and any accepted/canceled/expired event
# for an order we don't know — is a benign no-op: the handlers below return an empty
# ApplyResult (the suppress signal) so the ProcessingManager fires nothing. External
# orders materialize only on money-carrying events (fill/partial-fill/deal/updated).


def _handle_accepted(state: AccountState, event: OrderAcceptedEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)

    if order is None:
        return ApplyResult()

    if order.status.is_terminal:
        # Late accept on an already-terminal order (design.md "Event model" —
        # a FILLED can arrive before its ACCEPTED): benign side-effect, no
        # transition, no phantom. Set
        # the venue id ONLY if the order is still in active_orders — an evicted
        # order's venue-id index was already dropped, so set_venue_id would
        # KeyError on active_orders[cid].
        if state.has_active_order(order.client_order_id):
            if event.venue_order_id is not None:
                state.set_venue_id(order.client_order_id, event.venue_order_id)
            order.accepted_at = event.accepted_at
        return ApplyResult()

    if event.venue_order_id is not None:
        state.set_venue_id(order.client_order_id, event.venue_order_id)

    order.accepted_at = event.accepted_at

    if order.status.is_pending:
        # A late/duplicate accept (the ccxt connector emits REST + WS acks by design)
        # racing an outstanding cancel/update must NOT wipe PENDING_*: the sweep keeps
        # polling, and a later cancel/update-rejected still reverts via the preserved
        # pre-pending capture.
        return ApplyResult()

    if not can_transition(order.status, OrderStatus.ACCEPTED):
        return ApplyResult()

    order = reconcile.transition(
        state, order.client_order_id, OrderStatus.ACCEPTED, now, update_time=event.last_update_time
    )

    return ApplyResult(order=order, order_change=OrderChange.ACCEPTED)


def _is_superseded_oid_cancel(order: Order, event: OrderEvent) -> bool:
    """True when a CANCELED addresses a venue id the order no longer holds.

    A cancel-and-replace modify (Hyperliquid, and any venue whose ``edit`` is implemented as
    atomic cancel+recreate) cancels the OLD venue order id but emits its CANCELED carrying the
    SAME client_order_id as the now-live replacement — which already moved to a NEW venue id.
    ``_resolve`` matches by client id first, so without this guard that stale CANCELED would
    cancel the live order. A CANCELED whose venue id differs from the order's current one is for
    a superseded id and must be ignored. (A cancel with no venue id, or one matching the order's
    current id, is a real cancel and passes through.)
    """
    return (
        event.venue_order_id is not None
        and order.venue_order_id is not None
        and event.venue_order_id != order.venue_order_id
    )


def _handle_canceled(state: AccountState, event: OrderCanceledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal or _is_superseded_oid_cancel(order, event):
        return ApplyResult()

    order = reconcile.transition(
        state, order.client_order_id, OrderStatus.CANCELED, now, update_time=event.last_update_time
    )
    return ApplyResult(order=order, order_change=OrderChange.CANCELED)


def _handle_lost(state: AccountState, event: OrderLostEvent, now: np.datetime64) -> ApplyResult:
    # - reconciler give-up: terminalize a never-confirmed order to LOST (terminal, so the
    #   transition is always legal). Routed through the bus so the strategy is notified.
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order = reconcile.transition(
        state, order.client_order_id, OrderStatus.LOST, now, update_time=event.last_update_time
    )
    return ApplyResult(order=order, order_change=OrderChange.LOST)


def _handle_expired(state: AccountState, event: OrderExpiredEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order = reconcile.transition(
        state, order.client_order_id, OrderStatus.EXPIRED, now, update_time=event.last_update_time
    )
    return ApplyResult(order=order, order_change=OrderChange.EXPIRED)


def _handle_rejected(state: AccountState, event: OrderRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order.rejected_reason = event.reason
    order.error_code = event.code
    order = reconcile.transition(
        state, order.client_order_id, OrderStatus.REJECTED, now, update_time=event.last_update_time
    )
    return ApplyResult(order=order, order_change=OrderChange.REJECTED)


def _covered_by_balance_push(state: AccountState, currency: str, venue_time: np.datetime64) -> bool:
    # An absolute balance push at/after this venue time (both clocks are the venue's —
    # same domain as Deal.time) already incorporated the change, so the delta leg must
    # not book on top of it. Correct under both [deal, push] and [push, deal] orderings:
    # an older push leaves the delta to book, a newer push supersedes it absolutely.
    as_of = state.get_balance_push_as_of(currency)
    return as_of is not None and as_of >= venue_time


def _reconciled_past(state: AccountState, instrument: Instrument, deal: Deal) -> bool:
    # - true if a snapshot already booked this deal into the size (deal.time <= watermark, both
    #   venue clock). Caller still records it for dedup/log; only the re-book is skipped.
    watermark = state.get_position_reconcile_as_of(instrument)
    return watermark is not None and deal.time <= watermark


def _book_deal(state: AccountState, instrument: Instrument, deal: Deal) -> Position:
    """Apply a deal's effect to the position and balances. Caller dedups first.

    Futures/swap: credits realized PnL and debits the fee to the settle-currency
    balance. Spot: debits the quote currency by the trade cost (notional + fee) and
    credits the base asset by the filled amount. Amounts are converted to the
    portfolio funded currency via state.conversion_rate (currently 1.0 — the
    multi-currency seam). Each balance leg is skipped when a venue balance push
    already covers it (see _covered_by_balance_push); position/r_pnl always book.
    """
    pos = state.ensure_position(instrument)
    _before = pos.quantity
    realized_pnl, fee = pos.update_position_by_deal(deal, state.conversion_rate(instrument))
    logger.debug("deal {} amt={} {}->{} tid={}", instrument.symbol, deal.amount, _before, pos.quantity, deal.trade_id)
    if instrument.is_futures():
        # TODO(account-mgmt): fee is folded into settle here (correct when
        # settle == portfolio base currency); revisit for instruments whose
        # settle currency differs from the portfolio base currency.
        if not _covered_by_balance_push(state, instrument.settle, deal.time):
            state.adjust_balance(instrument.settle, realized_pnl - fee)
    else:
        if not _covered_by_balance_push(state, instrument.quote, deal.time):
            state.adjust_balance(instrument.quote, -(deal.amount * deal.price + fee))
        if not _covered_by_balance_push(state, instrument.base, deal.time):
            state.adjust_balance(instrument.base, deal.amount)
    return pos


def _apply_execution(
    state: AccountState, order: Order, deal: Deal, now: np.datetime64, *, update_time: np.datetime64 | None = None
) -> Deal | None:
    """Shared booking gate for all three execution paths (DealEvent and the embedded fills
    on OrderFilled/OrderPartiallyFilled). Returns the deal to book, or None when it was a
    re-delivered trade we already saw (status transitions still proceed in the callers).
    Trade-id dedup lives in apply_fill.
    """
    cid = order.client_order_id
    return deal if state.apply_fill(cid, deal, now, update_time=update_time) else None


def _reconcile_fill_gap(state: AccountState, order: Order, event: OrderFilledEvent, now: np.datetime64) -> Deal | None:
    """Book the executions a terminal report counted (``event.venue_filled_quantity``) but that
    were never delivered as deals — dropped/coalesced WS fills — as one synthetic fill for the
    unbooked remainder at the venue's average fill price. Without it the position size only
    self-heals at the next snapshot reconcile and the realized PnL for the gap is never booked
    (reconcile is size-only). Routed through ``_apply_execution`` so an already-armed snapshot
    deficit still suppresses it (no double count), and booking it lifts ``filled_quantity`` to
    the venue figure so the next snapshot sees no raise. The synthesized fill carries no fee —
    the per-fill commissions are unknown; exact fees would need a trade fetch.

    Returns None (no gap) when the connector didn't supply the cumulative figure (sim/backtest,
    split-stream venues), so behaviour is unchanged unless a connector opts in by setting it.
    """
    venue_filled = event.venue_filled_quantity
    if venue_filled is None or order.instrument is None:
        return None
    remainder = venue_filled - order.filled_quantity
    if remainder <= reconcile.fill_qty_epsilon(order.instrument):
        return None
    price = event.venue_avg_price if event.venue_avg_price is not None else order.avg_fill_price
    if price is None:
        return None
    synthetic = Deal(
        trade_id=f"{order.venue_order_id or order.client_order_id}:fill-reconcile",
        order_id=order.venue_order_id or "",
        time=now,
        amount=remainder if order.side == OrderSide.BUY else -remainder,
        price=price,
        aggressive=False,
    )
    return _apply_execution(state, order, synthetic, now, update_time=event.last_update_time)


def _handle_fill(state: AccountState, event: OrderFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve_or_materialize(state, event, now)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state.set_venue_id(order.client_order_id, event.venue_order_id)
    # fill is None on split-stream venues (the deal arrives separately via DealEvent):
    # the terminal transition still happens, only the booking is skipped.
    deal = (
        _apply_execution(state, order, event.fill, now, update_time=event.last_update_time)
        if event.fill is not None
        else None
    )
    position = _book_deal(state, order.instrument, deal) if deal is not None and order.instrument is not None else None
    # Terminal fill-gap: book executions the venue counted but never delivered as deals so
    # position AND realized PnL converge now, not size-only at the next snapshot.
    if (gap := _reconcile_fill_gap(state, order, event, now)) is not None and order.instrument is not None:
        position = _book_deal(state, order.instrument, gap)
        deal = deal or gap
    order = reconcile.transition(
        state, order.client_order_id, OrderStatus.FILLED, now, update_time=event.last_update_time
    )
    return ApplyResult(order=order, order_change=OrderChange.FILLED, deal=deal, position=position)


def _handle_partial_fill(state: AccountState, event: OrderPartiallyFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve_or_materialize(state, event, now)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state.set_venue_id(order.client_order_id, event.venue_order_id)
    # fill is None on split-stream venues (the deal arrives separately via DealEvent):
    # the status transition still happens, only the booking is skipped.
    deal = (
        _apply_execution(state, order, event.fill, now, update_time=event.last_update_time)
        if event.fill is not None
        else None
    )
    position = _book_deal(state, order.instrument, deal) if deal is not None and order.instrument is not None else None
    # a pending cancel/update is resolved by the venue separately — don't disturb its status
    pending = order.status.is_pending
    if pending:
        # filled_quantity mirrors real, irreversible fills (and the position),
        # so it is NEVER reduced. A fill that races a pending modify can push it
        # past the new (smaller) target — surface that as a warning only; the
        # venue resolves the race (OrderUpdated, or OrderUpdateRejected because
        # it can't shrink an order below what's already filled).
        if order.status == OrderStatus.PENDING_UPDATE and order.filled_quantity > order.quantity:
            logger.warning(
                f"[{order.client_order_id}] fill during pending-update pushed "
                f"filled_quantity ({order.filled_quantity}) past target "
                f"({order.quantity}); leaving filled intact, awaiting venue verdict"
            )
        return ApplyResult(deal=deal, position=position)
    if can_transition(order.status, OrderStatus.PARTIALLY_FILLED):
        order = reconcile.transition(
            state, order.client_order_id, OrderStatus.PARTIALLY_FILLED, now, update_time=event.last_update_time
        )
        return ApplyResult(order=order, order_change=OrderChange.PARTIALLY_FILLED, deal=deal, position=position)
    return ApplyResult(deal=deal, position=position)  # no status change -> execution only


def _handle_deal(state: AccountState, event: DealEvent, now: np.datetime64) -> ApplyResult:
    # Status comes from order events; a deal only drives the ledger. A deal is
    # money-carrying, so an unknown order materializes as EXTERNAL (instrument-guarded
    # in _resolve_or_materialize). A historical (RequestHistDeals) recovery deal materializes
    # TERMINAL — the order already completed; an ACCEPTED phantom would be chased as missing.
    materialize_status = OrderStatus.FILLED if event.historical else OrderStatus.ACCEPTED
    order = _resolve_or_materialize(state, event, now, materialize_status=materialize_status)
    if order is None:
        return ApplyResult()
    if order.status.is_terminal and not state.has_active_order(order.client_order_id):
        # Evicted past the terminal grace window: the seen-trade dedup table is gone,
        # so booking here could double-count a re-delivered trade. A terminal order
        # still INSIDE the grace window falls through — the split-stream FILLED status
        # (fill=None) can beat the final trade, and that trade must still book; the
        # trade-id dedup below keeps an already-seen one a no-op.
        return ApplyResult()
    if event.venue_order_id is not None:
        state.set_venue_id(order.client_order_id, event.venue_order_id)
    deal = _apply_execution(state, order, event.deal, now, update_time=event.last_update_time)
    if deal is None or order.instrument is None:
        return ApplyResult(deal=deal)
    if _reconciled_past(state, order.instrument, deal):
        # - snapshot already owns size + balance; realize the deal's pnl only (no size/balance)
        pos = state.ensure_position(order.instrument)
        pos.update_position_by_deal(deal, state.conversion_rate(order.instrument), realize_only=True)
        return ApplyResult(deal=deal, position=pos)
    return ApplyResult(deal=deal, position=_book_deal(state, order.instrument, deal))


def _handle_updated(state: AccountState, event: OrderUpdatedEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve_or_materialize(state, event, now)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None and order.venue_order_id != event.venue_order_id:
        # set_venue_id re-keys internally: it drops the order's previous venue id.
        state.set_venue_id(order.client_order_id, event.venue_order_id)
    if event.new_price is not None:
        order.price = event.new_price
    if event.new_quantity is not None:
        order.quantity = event.new_quantity
    order.last_update_time = event.last_update_time if event.last_update_time is not None else now
    if order.status == OrderStatus.PENDING_UPDATE:
        target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
        order = reconcile.transition(state, order.client_order_id, target, now, update_time=event.last_update_time)
    return ApplyResult(order=order, order_change=OrderChange.UPDATED)


def _revert_from_pending(
    state: AccountState,
    order: Order,
    change: OrderChange,
    now: np.datetime64,
    *,
    update_time: np.datetime64 | None = None,
) -> ApplyResult:
    # Revert to the status captured on entry to PENDING_* — never inferred from
    # filled_quantity/venue_id (brittle when venues roll back partial fills). ACCEPTED
    # is the safe default for the rare order with no captured status. The transition
    # itself clears the capture (the target is non-pending).
    target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
    order = reconcile.transition(state, order.client_order_id, target, now, update_time=update_time)
    return ApplyResult(order=order, order_change=change)


def _handle_cancel_rejected(state: AccountState, event: OrderCancelRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status != OrderStatus.PENDING_CANCEL:
        return ApplyResult()
    return _revert_from_pending(state, order, OrderChange.CANCEL_REJECTED, now, update_time=event.last_update_time)


def _handle_update_rejected(state: AccountState, event: OrderUpdateRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status != OrderStatus.PENDING_UPDATE:
        return ApplyResult()
    return _revert_from_pending(state, order, OrderChange.UPDATE_REJECTED, now, update_time=event.last_update_time)


def _handle_funding_payment(state: AccountState, event: FundingPaymentEvent, now: np.datetime64) -> ApplyResult:
    payment = event.payment
    instrument = event.instrument
    if instrument is None:
        return ApplyResult()
    interval_ns = payment.funding_interval_hours * 3_600_000_000_000
    bucket = (instrument, int(payment.time) // interval_ns)
    if state.is_funding_applied(bucket):
        return ApplyResult()
    pos = state.get_position(instrument)
    if pos is None:
        return ApplyResult()
    # Funding cash is computed on the mark price; FundingPayment carries no
    # amount. If the position has no mark yet (NaN), we cannot value the
    # payment — skip WITHOUT consuming the bucket so a re-delivered event can
    # apply once a mark exists (rather than poisoning balance/PnL with NaN).
    mark = pos.last_update_price
    if np.isnan(mark):
        logger.warning(f"[{state.exchange}] funding for {instrument} skipped: no mark price yet")
        return ApplyResult()
    state.mark_funding_applied(bucket)
    amount = pos.apply_funding_payment(payment, mark)  # updates cumulative_funding/pnl
    # Skip the cash leg when a venue balance push already covers it (the venue debits
    # the wallet and pushes the new total with reason FUNDING_FEE — booking our computed
    # amount on top would double-count). cumulative_funding/pnl above always book.
    # payment.time is a venue-clock ns epoch — same domain as the push as_of.
    covered = _covered_by_balance_push(state, instrument.settle, np.datetime64(int(payment.time), "ns"))
    if not covered and state.get_balance(instrument.settle) is not None:
        # adjust only an existing settle balance (funding never creates one)
        state.adjust_balance(instrument.settle, amount)
    return ApplyResult(position=pos)


def _handle_balance_update(state: AccountState, event: BalanceUpdateEvent, now: np.datetime64) -> ApplyResult:
    # Absolute apply through the per-currency as_of ratchet; total-only pushes (futures
    # carry no free/locked split -> NaN) preserve locked. Producer contract: a push
    # without a real split MUST carry free/locked = NaN, not 0. The balance field on the
    # result is internal visibility only — no strategy callback fires (design.md: deliberately no balance callback).
    bal = event.balance
    if not state.apply_balance_push(bal.currency, bal.total, event.as_of, free=bal.free, locked=bal.locked):
        return ApplyResult()
    return ApplyResult(balance=state.get_balance(bal.currency))


_HANDLERS = {
    OrderAcceptedEvent: _handle_accepted,
    OrderPartiallyFilledEvent: _handle_partial_fill,
    OrderFilledEvent: _handle_fill,
    DealEvent: _handle_deal,
    OrderUpdatedEvent: _handle_updated,
    OrderCanceledEvent: _handle_canceled,
    OrderLostEvent: _handle_lost,
    OrderExpiredEvent: _handle_expired,
    OrderRejectedEvent: _handle_rejected,
    OrderCancelRejectedEvent: _handle_cancel_rejected,
    OrderUpdateRejectedEvent: _handle_update_rejected,
    FundingPaymentEvent: _handle_funding_payment,
    BalanceUpdateEvent: _handle_balance_update,
}


def apply(state: AccountState, event: AccountMessage, now: np.datetime64) -> ApplyResult:
    # - snapshots are owned by the Reconciler (driven from the AccountManager), never here.
    if (handler := _HANDLERS.get(type(event))) is None:
        logger.warning(f"unknown handler for AccountMessage type {type(event)} ::: {event}")
        return ApplyResult()

    return handler(state, event, now)
