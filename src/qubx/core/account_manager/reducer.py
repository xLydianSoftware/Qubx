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
    OrderEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
)
from qubx.core.account_manager.state import AccountState
from qubx.core.account_manager.state_machine import can_transition, validate_transition
from qubx.core.basics import Deal, Order, OrderChange, OrderStatus, Position


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


def _handle_fill(state: AccountState, event: OrderFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    new_deal = event.fill is not None and state._apply_fill(order.client_id, event.fill, now)
    order = _transition(state, order.client_id, OrderStatus.FILLED, now)
    return ApplyResult(order=order, order_change=OrderChange.FILLED, deal=event.fill if new_deal else None)


def _handle_partial_fill(state: AccountState, event: OrderPartiallyFilledEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    new_deal = event.fill is not None and state._apply_fill(order.client_id, event.fill, now)
    deal = event.fill if new_deal else None
    # a pending cancel/update is resolved by the venue separately — don't disturb its status
    pending = order.status in (OrderStatus.PENDING_CANCEL, OrderStatus.PENDING_UPDATE)
    if not pending and can_transition(order.status, OrderStatus.PARTIALLY_FILLED):
        order = _transition(state, order.client_id, OrderStatus.PARTIALLY_FILLED, now)
        return ApplyResult(order=order, order_change=OrderChange.PARTIALLY_FILLED, deal=deal)
    return ApplyResult(deal=deal)  # no status change -> on_execution only


def _handle_deal(state: AccountState, event: DealEvent, now: np.datetime64) -> ApplyResult:
    order = _resolve(state, event)
    if order is None or order.status.is_terminal:
        return ApplyResult()
    if event.venue_order_id is not None:
        state._set_venue_id(order.client_id, event.venue_order_id)
    if not state._apply_fill(order.client_id, event.deal, now):
        return ApplyResult()
    return ApplyResult(deal=event.deal)  # status comes from order events; on_execution only


_HANDLERS = {
    OrderAcceptedEvent: _handle_accepted,
    OrderPartiallyFilledEvent: _handle_partial_fill,
    OrderFilledEvent: _handle_fill,
    DealEvent: _handle_deal,
    OrderCanceledEvent: _handle_canceled,
    OrderExpiredEvent: _handle_expired,
    OrderRejectedEvent: _handle_rejected,
}


def apply(state: AccountState, event: AccountMessage, now: np.datetime64) -> ApplyResult:
    handler = _HANDLERS.get(type(event))
    if handler is None:
        logger.debug(f"reducer: no handler for {type(event).__name__}")
        return ApplyResult()
    return handler(state, event, now)
