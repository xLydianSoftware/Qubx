import numpy as np

from qubx import logger
from qubx.core.account_manager_config import AccountManagerConfig
from qubx.core.account_state import AccountState
from qubx.core.basics import Order, OrderOrigin, OrderStatus
from qubx.core.events import (
    AccountMessage,
    AccountSnapshotEvent,
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
from qubx.core.exceptions import InvalidOrderTransition

_LEGAL_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.INITIALIZED: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PENDING_CANCEL: {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.PENDING_UPDATE: {
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    },
}


class AccountManager:
    def __init__(self, *, pm, connectors, strategy, time, cfg: AccountManagerConfig | None = None):
        self._pm = pm
        self._connectors = connectors
        self._strategy = strategy
        self._time = time
        self._cfg = cfg or AccountManagerConfig()
        self._states = {ex: AccountState(exchange=ex) for ex in connectors}
        self._liveness_unready_since: dict[str, np.datetime64] = {}
        self._applied_funding_buckets: dict[str, set] = {}
        self._ctx = None  # set via set_context once StrategyContext is built
        # Tick registration deferred to PR 4.

    def set_context(self, ctx) -> None:
        """Wire the IStrategyContext after construction.

        AM-fired callbacks (reconcile, inflight-exhaustion) pass this ctx so their
        signature matches PM-fired callbacks — no None placeholder.
        """
        self._ctx = ctx

    def add_order(self, exchange: str, order: Order) -> None:
        self._states[exchange]._add_order(order)

    def transition_order(self, exchange: str, cid: str, new_status: OrderStatus) -> None:
        state = self._states[exchange]
        order = state.active_orders.get(cid)
        if order is None:
            raise KeyError(f"order {cid} not found in {exchange}")
        if new_status not in _LEGAL_TRANSITIONS.get(order.status, set()):
            raise InvalidOrderTransition(cid, order.status, new_status)
        if new_status in (OrderStatus.PENDING_CANCEL, OrderStatus.PENDING_UPDATE):
            order.pre_pending_status = order.status
        state._transition_order(cid, new_status, self._time.now())

    def get_state(self, exchange: str) -> AccountState:
        return self._states[exchange]

    def get_orders(self, exchange: str | None = None, origin: OrderOrigin | None = None) -> dict[str, Order]:
        if exchange is not None:
            orders = self._states[exchange].get_orders()
        else:
            orders = {cid: o for s in self._states.values() for cid, o in s.get_orders().items()}
        if origin is not None:
            return {cid: o for cid, o in orders.items() if o.origin == origin}
        return orders

    def get_order(self, client_order_id: str) -> Order | None:
        for state in self._states.values():
            if (o := state.get_order(client_order_id)) is not None:
                return o
        return None

    def get_position(self, instrument):
        state = self._states.get(instrument.exchange)
        return state.get_position(instrument) if state else None

    def get_balance(self, currency: str, exchange: str | None = None):
        if exchange is not None:
            return self._states[exchange].get_balance(currency)
        for state in self._states.values():
            if (b := state.get_balance(currency)) is not None:
                return b
        return None

    def apply(self, event: AccountMessage):
        state = self._get_state_for_event(event)
        if state is None:
            return None
        match event:
            case OrderPartiallyFilledEvent():
                return self._handle_partial_fill(state, event)
            case OrderFilledEvent():
                return self._handle_fill(state, event)
            case OrderAcceptedEvent():
                return self._handle_accepted(state, event)
            case OrderCanceledEvent():
                return self._handle_canceled(state, event)
            case OrderExpiredEvent():
                return self._handle_expired(state, event)
            case OrderUpdatedEvent():
                return self._handle_updated(state, event)
            case OrderRejectedEvent():
                return self._handle_rejected(state, event)
            case OrderCancelRejectedEvent():
                return self._handle_cancel_rejected(state, event)
            case OrderUpdateRejectedEvent():
                return self._handle_update_rejected(state, event)
            case _:
                logger.warning(f"unhandled AccountMessage: {type(event)}")
                return None

    def _get_state_for_event(self, event):
        if isinstance(event, AccountSnapshotEvent):
            return self._states.get(event.snapshot.exchange)
        if event.instrument is not None:
            return self._states.get(event.instrument.exchange)
        return None

    def _resolve_or_materialize(self, state, event):
        cid = getattr(event, "client_order_id", None)
        venue_id = getattr(event, "venue_order_id", None)
        if cid is not None and cid in state.active_orders:
            return state.active_orders[cid]
        if venue_id is not None and venue_id in state._venue_id_index:
            return state.active_orders[state._venue_id_index[venue_id]]
        if cid is not None:
            for hist in state._terminal_history:
                if hist.client_order_id == cid:
                    return hist
        return self._materialize_external(state, event)

    def _materialize_external(self, state, event):
        venue_id = getattr(event, "venue_order_id", None)
        if venue_id is None:
            # No venue id → no stable identity; all such orders would collide on
            # ext:unknown. Real venue events always carry one, so warn loudly.
            venue_id = "unknown"
            logger.warning(f"materializing EXTERNAL order with no venue_order_id: {event}")
        cid = f"ext:{venue_id}"
        order = Order(
            client_order_id=cid,
            venue_order_id=venue_id,
            origin=OrderOrigin.EXTERNAL,
            type="LIMIT",
            instrument=event.instrument,
            time=self._time.now(),
            quantity=0.0,
            price=0.0,
            side="BUY",
            status=OrderStatus.ACCEPTED,
            time_in_force="gtc",
        )
        state._add_order(order)
        return order

    def _handle_accepted(self, state, event: OrderAcceptedEvent):
        order = self._resolve_or_materialize(state, event)
        if order.status.is_terminal():
            # Late accept on an already-terminal order (design "OrderFilled before
            # OrderAccepted"): benign side-effect, no transition, no phantom. Set
            # the venue id ONLY if the order is still in active_orders — an evicted
            # order's venue-id index was already dropped, so _set_venue_id would
            # KeyError on active_orders[cid].
            if order.client_order_id in state.active_orders:
                state._set_venue_id(order.client_order_id, event.venue_order_id)
                order.accepted_at = event.accepted_at
            return order
        state._set_venue_id(order.client_order_id, event.venue_order_id)
        order.accepted_at = event.accepted_at
        if order.status == OrderStatus.PENDING_CANCEL:
            return order
        if order.status == OrderStatus.PENDING_UPDATE:
            return state._transition_order(order.client_order_id, OrderStatus.ACCEPTED, self._time.now())
        if OrderStatus.ACCEPTED in _LEGAL_TRANSITIONS.get(order.status, set()):
            return state._transition_order(order.client_order_id, OrderStatus.ACCEPTED, self._time.now())
        return order

    # Terminal-order guard (shared by the lifecycle handlers below).
    # A late event for an order that is already terminal — possibly still in
    # active_orders during the grace window, possibly already evicted to
    # _terminal_history — must be a benign no-op: NO status change (a terminal
    # state has no legal outgoing edge) and NO active_orders[cid] mutation (which
    # would KeyError for an evicted order). This generalizes the OrderAccepted
    # grace rule (design "OrderFilled before OrderAccepted" / terminal retention)
    # to every lifecycle event, and is what keeps a late OrderCanceled from
    # silently flipping a FILLED order to CANCELED.
    def _is_late_terminal(self, order) -> bool:
        return order.status.is_terminal()

    def _handle_partial_fill(self, state, event: OrderPartiallyFilledEvent):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late partial-fill on terminal {order.client_order_id}; ignoring")
            return order
        if event.venue_order_id and order.venue_order_id is None:
            state._set_venue_id(order.client_order_id, event.venue_order_id)
        state._apply_fill(order.client_order_id, event.fill, self._time.now())
        if order.status.is_pending():
            # While PENDING_UPDATE a fill may race in for the pre-modify (larger)
            # quantity. Clamp and warn — overshoot signals a real ordering race
            # the strategy should know about.
            if order.status == OrderStatus.PENDING_UPDATE and order.filled_quantity > order.quantity:
                logger.warning(
                    f"[{order.client_order_id}] fill races pre-modify quantity; "
                    f"clamping filled_quantity {order.filled_quantity} -> {order.quantity}"
                )
                order.filled_quantity = order.quantity
            return state.get_order(order.client_order_id)
        if OrderStatus.PARTIALLY_FILLED in _LEGAL_TRANSITIONS.get(order.status, set()):
            return state._transition_order(order.client_order_id, OrderStatus.PARTIALLY_FILLED, self._time.now())
        return order

    def _handle_fill(self, state, event: OrderFilledEvent):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late fill on terminal {order.client_order_id}; ignoring")
            return order
        if event.venue_order_id and order.venue_order_id is None:
            state._set_venue_id(order.client_order_id, event.venue_order_id)
        state._apply_fill(order.client_order_id, event.fill, self._time.now())
        return state._transition_order(order.client_order_id, OrderStatus.FILLED, self._time.now())

    def _handle_canceled(self, state, event):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late cancel on terminal {order.client_order_id}; ignoring")
            return order
        return state._transition_order(order.client_order_id, OrderStatus.CANCELED, self._time.now())

    def _handle_expired(self, state, event):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late expire on terminal {order.client_order_id}; ignoring")
            return order
        return state._transition_order(order.client_order_id, OrderStatus.EXPIRED, self._time.now())

    def _handle_rejected(self, state, event: OrderRejectedEvent):
        order = state.active_orders.get(event.client_order_id)
        if order is None:
            logger.warning(f"reject for unknown order {event.client_order_id}")
            return None
        if order.status.is_terminal():
            logger.debug(f"late reject on terminal {order.client_order_id}; ignoring")
            return order
        order.rejected_reason = event.reason
        return state._transition_order(order.client_order_id, OrderStatus.REJECTED, self._time.now())

    def _handle_updated(self, state, event: OrderUpdatedEvent):
        order = self._resolve_or_materialize(state, event)
        if self._is_late_terminal(order):
            logger.debug(f"late update on terminal {order.client_order_id}; ignoring")
            return order
        if order.venue_order_id != event.venue_order_id:
            if order.venue_order_id is not None:
                state._venue_id_index.pop(order.venue_order_id, None)
            state._set_venue_id(order.client_order_id, event.venue_order_id)
        if event.new_price is not None:
            order.price = event.new_price
        if event.new_quantity is not None:
            order.quantity = event.new_quantity
        order.last_updated_at = self._time.now()
        if order.status == OrderStatus.PENDING_UPDATE:
            return state._transition_order(order.client_order_id, OrderStatus.ACCEPTED, self._time.now())
        return order

    def _handle_cancel_rejected(self, state, event):
        order = state.active_orders.get(event.client_order_id)
        if order is None or order.status != OrderStatus.PENDING_CANCEL:
            logger.warning(f"cancel-rejected for unexpected state: {order}")
            return None
        return self._revert_from_pending(state, order)

    def _handle_update_rejected(self, state, event):
        order = state.active_orders.get(event.client_order_id)
        if order is None or order.status != OrderStatus.PENDING_UPDATE:
            logger.warning(f"update-rejected for unexpected state: {order}")
            return None
        return self._revert_from_pending(state, order)

    def _revert_from_pending(self, state, order):
        target = order.pre_pending_status or OrderStatus.ACCEPTED
        order.pre_pending_status = None
        return state._transition_order(order.client_order_id, target, self._time.now())
