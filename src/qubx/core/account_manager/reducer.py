"""Applies a typed AccountMessage to one AccountState, driving the order state machine.

Pure state mutation: no connectors, no strategy callbacks. The ProcessingManager fires
callbacks from the returned ApplyResult; routing the event to the right state is the
manager's job. Every status change goes through `_transition` (the legality chokepoint),
and every handler short-circuits on a terminal order so late venue events are no-ops.
"""

from dataclasses import dataclass

import numpy as np

from qubx import logger
from qubx.core.account_manager.events import (
    AccountMessage,
    DealEvent,
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
)
from qubx.core.account_manager.state import AccountState
from qubx.core.account_manager.state_machine import can_transition, validate_transition
from qubx.core.basics import Deal, Instrument, Order, OrderChange, OrderStatus, Position


@dataclass
class ApplyResult:
    order: Order | None = None  # status changed -> on_order(order, order_change)
    order_change: OrderChange | None = None  # paired with order
    deal: Deal | None = None  # new deal applied -> on_execution
    position: Position | None = None  # position changed -> on_position_change


def _transition(state: AccountState, cid: str, new_status: OrderStatus, now: np.datetime64) -> Order:
    order = state.get_active_order(cid)
    if order is None:
        raise KeyError(f"order {cid} not found in {state.exchange}")
    validate_transition(cid, order.status, new_status)
    return state._transition_order(cid, new_status, now)


def _resolve(state: AccountState, event: OrderEvent) -> Order | None:
    if (order := state.get_order(event.client_order_id)) is not None:
        return order
    if event.venue_order_id is not None:
        return state.get_order_by_venue_id(event.venue_order_id)
    return None


def _active_order_for(state: AccountState, event: OrderEvent) -> Order | None:
    order = state.get_active_order(event.client_order_id)
    if order is None and event.venue_order_id is not None:
        order = state.get_order_by_venue_id(event.venue_order_id)
    return order


def _handle_accepted(state: AccountState, event: OrderAcceptedEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    if not can_transition(order.status, OrderStatus.ACCEPTED):
        return ApplyResult()
    order = _transition(state, order.client_id, OrderStatus.ACCEPTED, now)
    return ApplyResult(order=order, order_change=OrderChange.ACCEPTED)


def _handle_canceled(state: AccountState, event: OrderCanceledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order = _transition(state, order.client_id, OrderStatus.CANCELED, now)
    return ApplyResult(order=order, order_change=OrderChange.CANCELED)


def _handle_expired(state: AccountState, event: OrderExpiredEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order = _transition(state, order.client_id, OrderStatus.EXPIRED, now)
    return ApplyResult(order=order, order_change=OrderChange.EXPIRED)


def _handle_rejected(state: AccountState, event: OrderRejectedEvent, now: np.datetime64) -> ApplyResult:
    order = _active_order_for(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    order.rejected_reason = event.reason
    order = _transition(state, order.client_id, OrderStatus.REJECTED, now)
    return ApplyResult(order=order, order_change=OrderChange.REJECTED)


def _book_deal(state: AccountState, instrument: Instrument, deal: Deal) -> Position:
    """Apply a deal's effect to the position and balances. Caller dedups first."""
    pos = state._ensure_position(instrument)
    realized_pnl, fee = pos.update_position_by_deal(deal, state.conversion_rate(instrument))
    if instrument.is_futures():
        state._adjust_balance(instrument.settle, realized_pnl - fee)
    else:
        state._adjust_balance(instrument.quote, -(deal.amount * deal.price + fee))
        state._adjust_balance(instrument.base, deal.amount)
    return pos


def _handle_fill(state: AccountState, event: OrderFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    new_deal = event.fill is not None and state._apply_fill(order.client_id, event.fill, now)
    deal = event.fill if new_deal else None
    position = _book_deal(state, order.instrument, deal) if deal is not None else None
    order = _transition(state, order.client_id, OrderStatus.FILLED, now)
    return ApplyResult(order=order, order_change=OrderChange.FILLED, deal=deal, position=position)


def _handle_partial_fill(state: AccountState, event: OrderPartiallyFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    new_deal = event.fill is not None and state._apply_fill(order.client_id, event.fill, now)
    deal = event.fill if new_deal else None
    position = _book_deal(state, order.instrument, deal) if deal is not None else None
    # a pending cancel/update is resolved by the venue separately — don't disturb its status
    pending = order.status in (OrderStatus.PENDING_CANCEL, OrderStatus.PENDING_UPDATE)
    if not pending and can_transition(order.status, OrderStatus.PARTIALLY_FILLED):
        order = _transition(state, order.client_id, OrderStatus.PARTIALLY_FILLED, now)
        return ApplyResult(order=order, order_change=OrderChange.PARTIALLY_FILLED, deal=deal, position=position)
    return ApplyResult(deal=deal, position=position)  # no status change -> on_execution (+ position)


def _handle_deal(state: AccountState, event: DealEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    if not state._apply_fill(order.client_id, event.deal, now):
        return ApplyResult()
    position = _book_deal(state, order.instrument, event.deal)
    return ApplyResult(deal=event.deal, position=position)  # status comes from order events


def _handle_updated(state: AccountState, event: OrderUpdatedEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None and order.venue_id != event.venue_order_id:
        state._set_venue_id(order.client_id, event.venue_order_id)
    if event.new_price is not None:
        order.price = event.new_price
    if event.new_quantity is not None:
        order.quantity = event.new_quantity
    order.last_updated_at = now
    if order.status == OrderStatus.PENDING_UPDATE:
        target = state.get_pre_pending(order.client_id) or OrderStatus.ACCEPTED
        order = _transition(state, order.client_id, target, now)
    return ApplyResult(order=order, order_change=OrderChange.UPDATED)


def _revert_from_pending(state: AccountState, order: Order, change: OrderChange, now: np.datetime64) -> ApplyResult:
    target = state.get_pre_pending(order.client_id) or OrderStatus.ACCEPTED
    order = _transition(state, order.client_id, target, now)
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
}


def apply(state: AccountState, event: AccountMessage, now: np.datetime64) -> ApplyResult:
    handler = _HANDLERS.get(type(event))
    if handler is None:
        logger.debug(f"reducer: no handler for {type(event).__name__}")
        return ApplyResult()
    return handler(state, event, now)
