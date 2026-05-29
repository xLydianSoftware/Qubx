import numpy as np

from qubx.core.account_manager_config import AccountManagerConfig
from qubx.core.account_state import AccountState
from qubx.core.basics import Order, OrderOrigin, OrderStatus
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
