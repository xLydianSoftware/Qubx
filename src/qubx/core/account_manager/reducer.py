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

from dataclasses import dataclass, replace

import numpy as np

from qubx import logger
from qubx.core.account_manager import reconcile
from qubx.core.account_manager.reconcile import ReconcileDiff
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
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    DealEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
    PositionUpdateEvent,
)


@dataclass
class ApplyResult:
    order: Order | None = None  # status changed -> on_order(order, change)
    order_change: OrderChange | None = None  # paired with order
    deal: Deal | None = None  # new deal applied -> downstream fill consumers
    position: Position | None = None  # position changed
    reconcile_diff: ReconcileDiff | None = None  # set when a snapshot reconcile applied
    # venue position push disagrees with local size — NOT applied to state; the
    # AccountManager reacts with a rate-limited snapshot request (race-safe correction)
    position_drift: Position | None = None
    # balance push applied (the live state Balance) — internal/diff visibility only,
    # fires NO strategy callback by design (balances are read via ctx)
    balance: Balance | None = None

    def is_empty(self) -> bool:
        """All fields None — the suppress signal (see module docstring): no callback fires."""
        return (
            self.order is None
            and self.order_change is None
            and self.deal is None
            and self.position is None
            and self.reconcile_diff is None
            and self.position_drift is None
            and self.balance is None
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


def _materialize_external(state: AccountState, event: OrderEvent, instrument: Instrument, now: np.datetime64) -> Order:
    # Unknown to us => external order (manual UI / another bot / pre-existing). A venue
    # lifecycle event always carries the id the venue assigned; fall back to the cid
    # for a stable identity only in the (malformed) case where it doesn't.
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
        status=OrderStatus.ACCEPTED,  # it exists at the venue
        time_in_force="gtc",
    )
    state.add_order(order)
    return order


def _resolve_or_materialize(state: AccountState, event: OrderEvent, now: np.datetime64) -> Order | None:
    if (order := _resolve(state, event)) is not None:
        return order
    if event.instrument is None:  # can't track a position without an instrument
        return None
    if event.client_order_id is None and event.venue_order_id is None:
        return None  # no identity at all — materializing would collide every such event on "ext:None"
    return _materialize_external(state, event, event.instrument, now)


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
    order = reconcile.transition(state, order.client_order_id, OrderStatus.ACCEPTED, now)
    return ApplyResult(order=order, order_change=OrderChange.ACCEPTED)


def _handle_canceled(state: AccountState, event: OrderCanceledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order = reconcile.transition(state, order.client_order_id, OrderStatus.CANCELED, now)
    return ApplyResult(order=order, order_change=OrderChange.CANCELED)


def _handle_expired(state: AccountState, event: OrderExpiredEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order = reconcile.transition(state, order.client_order_id, OrderStatus.EXPIRED, now)
    return ApplyResult(order=order, order_change=OrderChange.EXPIRED)


def _handle_rejected(state: AccountState, event: OrderRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order.rejected_reason = event.reason
    order.error_code = event.code
    order = reconcile.transition(state, order.client_order_id, OrderStatus.REJECTED, now)
    return ApplyResult(order=order, order_change=OrderChange.REJECTED)


def _covered_by_balance_push(state: AccountState, currency: str, venue_time: np.datetime64) -> bool:
    # An absolute balance push at/after this venue time (both clocks are the venue's —
    # same domain as Deal.time) already incorporated the change, so the delta leg must
    # not book on top of it. Correct under both [deal, push] and [push, deal] orderings:
    # an older push leaves the delta to book, a newer push supersedes it absolutely.
    as_of = state.get_balance_push_as_of(currency)
    return as_of is not None and as_of >= venue_time


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
    realized_pnl, fee = pos.update_position_by_deal(deal, state.conversion_rate(instrument))
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


def _apply_execution(state: AccountState, order: Order, deal: Deal, now: np.datetime64) -> Deal | None:
    """Shared already-counted gate for all three execution paths (DealEvent and the
    embedded fills on OrderFilled/OrderPartiallyFilled). Returns the deal to book —
    trimmed to the uncovered part when a snapshot partially covers it — or None when
    nothing should book (status transitions still proceed in the callers).

    Trade-id dedup stays first-line: a re-delivered trade we already saw never touches
    the deficit. Then the snapshot-fill deficit — the quantity a snapshot counted into
    filled_quantity (position/balance legs reconciled with it) that we never booked as
    deals — suppresses the arriving execution up to the counted amount: a fully covered
    one records its trade id (re-deliveries still dedup) and books nothing; a larger one
    books only the excess, fee pro-rated. Excess/remaining-deficit comparisons use the
    half-lot epsilon (reconcile.fill_qty_epsilon): the deficit is float-subtraction
    arithmetic, and an exact 0.0 gate booked ~1e-16 phantom dust deals. The rule is
    clock-free (quantity is the only signal — no venue-vs-local time comparison), so a
    genuinely-new execution that fits the deficit window is suppressed in place of the
    covered one it stands in for: quantity totals stay exact either way, only per-deal
    price/fee attribution can shift, and the next snapshot re-syncs sizes regardless.
    """
    cid = order.client_order_id
    deficit = state.get_snapshot_fill_deficit(cid)
    if deficit > 0.0 and not state.is_trade_seen(cid, deal.trade_id):
        eps = reconcile.fill_qty_epsilon(order.instrument)
        qty = abs(deal.amount)
        covered = min(qty, deficit)
        remaining = deficit - covered
        state.set_snapshot_fill_deficit(cid, 0.0 if remaining <= eps else remaining)
        state.set_snapshot_fill_suppressed(cid, state.get_snapshot_fill_suppressed(cid) + covered)
        # The suppression IS an order-state observation: bump freshness so a snapshot
        # FETCHED before it fails the per-order guard in reconcile — otherwise that
        # stale snapshot resets the suppressed marker / mis-absorbs the deficit
        # (apply_fill bumps the booked paths; this covers the suppressed ones).
        order.last_updated_at = now
        excess = qty - covered
        if excess <= eps:
            state.record_trade_id(cid, deal.trade_id)
            return None
        fee = deal.fee_amount * (excess / qty) if deal.fee_amount is not None else None
        deal = replace(deal, amount=excess if deal.amount > 0 else -excess, fee_amount=fee)
    return deal if state.apply_fill(cid, deal, now) else None


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
    return _apply_execution(state, order, synthetic, now)


def _handle_fill(state: AccountState, event: OrderFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve_or_materialize(state, event, now)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state.set_venue_id(order.client_order_id, event.venue_order_id)
    # fill is None on split-stream venues (the deal arrives separately via DealEvent):
    # the terminal transition still happens, only the booking is skipped.
    deal = _apply_execution(state, order, event.fill, now) if event.fill is not None else None
    position = _book_deal(state, order.instrument, deal) if deal is not None and order.instrument is not None else None
    # Terminal fill-gap: book executions the venue counted but never delivered as deals so
    # position AND realized PnL converge now, not size-only at the next snapshot.
    if (gap := _reconcile_fill_gap(state, order, event, now)) is not None and order.instrument is not None:
        position = _book_deal(state, order.instrument, gap)
        deal = deal or gap
    order = reconcile.transition(state, order.client_order_id, OrderStatus.FILLED, now)
    return ApplyResult(order=order, order_change=OrderChange.FILLED, deal=deal, position=position)


def _handle_partial_fill(state: AccountState, event: OrderPartiallyFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve_or_materialize(state, event, now)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state.set_venue_id(order.client_order_id, event.venue_order_id)
    # fill is None on split-stream venues (the deal arrives separately via DealEvent):
    # the status transition still happens, only the booking is skipped.
    deal = _apply_execution(state, order, event.fill, now) if event.fill is not None else None
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
        order = reconcile.transition(state, order.client_order_id, OrderStatus.PARTIALLY_FILLED, now)
        return ApplyResult(order=order, order_change=OrderChange.PARTIALLY_FILLED, deal=deal, position=position)
    return ApplyResult(deal=deal, position=position)  # no status change -> execution only


def _handle_deal(state: AccountState, event: DealEvent, now: np.datetime64) -> ApplyResult:
    # Status comes from order events; a deal only drives the ledger. A deal is
    # money-carrying, so an unknown order materializes as EXTERNAL (instrument-guarded
    # in _resolve_or_materialize).
    order = _resolve_or_materialize(state, event, now)
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
    deal = _apply_execution(state, order, event.deal, now)
    if deal is None:
        return ApplyResult()
    position = _book_deal(state, order.instrument, deal) if order.instrument is not None else None
    return ApplyResult(deal=deal, position=position)


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
    order.last_updated_at = now
    if order.status == OrderStatus.PENDING_UPDATE:
        target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
        order = reconcile.transition(state, order.client_order_id, target, now)
    return ApplyResult(order=order, order_change=OrderChange.UPDATED)


def _revert_from_pending(state: AccountState, order: Order, change: OrderChange, now: np.datetime64) -> ApplyResult:
    # Revert to the status captured on entry to PENDING_* — never inferred from
    # filled_quantity/venue_id (brittle when venues roll back partial fills). ACCEPTED
    # is the safe default for the rare order with no captured status. The transition
    # itself clears the capture (the target is non-pending).
    target = state.get_pre_pending(order.client_order_id) or OrderStatus.ACCEPTED
    order = reconcile.transition(state, order.client_order_id, target, now)
    return ApplyResult(order=order, order_change=change)


def _handle_cancel_rejected(state: AccountState, event: OrderCancelRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status != OrderStatus.PENDING_CANCEL:
        return ApplyResult()
    return _revert_from_pending(state, order, OrderChange.CANCEL_REJECTED, now)


def _handle_update_rejected(state: AccountState, event: OrderUpdateRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status != OrderStatus.PENDING_UPDATE:
        return ApplyResult()
    return _revert_from_pending(state, order, OrderChange.UPDATE_REJECTED, now)


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


def _handle_position_update(state: AccountState, event: PositionUpdateEvent, now: np.datetime64) -> ApplyResult:
    # Positions are tracked from fills (the deal ledger). A venue position push never
    # writes size — it only VERIFIES local state: on drift the AccountManager refetches
    # a snapshot to correct it.
    instrument = event.position.instrument
    last = state.get_position_push_as_of(instrument)
    if last is not None and event.as_of <= last:
        return ApplyResult()
    state.mark_position_push(instrument, event.as_of)
    local = state.get_position(instrument)
    local_qty = local.quantity if local is not None else 0.0
    if abs(event.position.quantity - local_qty) < instrument.lot_size:
        return ApplyResult(position=local)
    return ApplyResult(position_drift=event.position)


def _handle_balance_update(state: AccountState, event: BalanceUpdateEvent, now: np.datetime64) -> ApplyResult:
    # Absolute apply through the per-currency as_of ratchet; total-only pushes (futures
    # carry no free/locked split -> NaN) preserve locked. Producer contract: a push
    # without a real split MUST carry free/locked = NaN, not 0. The balance field on the
    # result is internal visibility only — no strategy callback fires (design.md: deliberately no balance callback).
    bal = event.balance
    if not state.apply_balance_push(bal.currency, bal.total, event.as_of, free=bal.free, locked=bal.locked):
        return ApplyResult()
    return ApplyResult(balance=state.get_balance(bal.currency))


def _handle_snapshot(
    state: AccountState, event: AccountSnapshotEvent, now: np.datetime64, grace: np.timedelta64
) -> ApplyResult:
    diff = reconcile.reconcile_snapshot(state, event.snapshot, now, grace)
    if diff is None:  # stale/out-of-order snapshot, rejected by the as_of ratchet
        return ApplyResult()
    return ApplyResult(reconcile_diff=diff)


_HANDLERS = {
    OrderAcceptedEvent: _handle_accepted,
    OrderPartiallyFilledEvent: _handle_partial_fill,
    OrderFilledEvent: _handle_fill,
    DealEvent: _handle_deal,
    OrderUpdatedEvent: _handle_updated,
    OrderCanceledEvent: _handle_canceled,
    OrderExpiredEvent: _handle_expired,
    OrderRejectedEvent: _handle_rejected,
    OrderCancelRejectedEvent: _handle_cancel_rejected,
    OrderUpdateRejectedEvent: _handle_update_rejected,
    FundingPaymentEvent: _handle_funding_payment,
    PositionUpdateEvent: _handle_position_update,
    BalanceUpdateEvent: _handle_balance_update,
}


def apply(
    state: AccountState, event: AccountMessage, now: np.datetime64, *, snapshot_grace: np.timedelta64
) -> ApplyResult:
    # The snapshot handler is the one dispatch target that needs config (the grace
    # window), so it is routed explicitly rather than through the uniform table.
    if isinstance(event, AccountSnapshotEvent):
        return _handle_snapshot(state, event, now, snapshot_grace)
    handler = _HANDLERS.get(type(event))
    if handler is None:
        logger.warning(f"unhandled AccountMessage: {type(event)}")
        return ApplyResult()
    return handler(state, event, now)
