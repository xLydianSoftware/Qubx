"""Order-lifecycle state machine for the AccountManager.

Pure and I/O-free so the transition rules can be audited and tested in isolation.
Terminalization is venue-authoritative: any live order may move straight to a terminal
state, so the table below lists only non-terminal -> non-terminal edges.
"""

from qubx.core.basics import OrderStatus
from qubx.core.exceptions import InvalidOrderTransition

TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.INITIALIZED: {OrderStatus.SUBMITTED},
    OrderStatus.SUBMITTED: {
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
    },
    OrderStatus.PENDING_CANCEL: {
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
    },
    OrderStatus.PENDING_UPDATE: {
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.PENDING_CANCEL,
    },
}


def can_transition(frm: OrderStatus, to: OrderStatus) -> bool:
    if frm.is_terminal:
        return False
    if to.is_terminal:
        return True
    return to in TRANSITIONS.get(frm, set())


def validate_transition(client_order_id: str, frm: OrderStatus, to: OrderStatus) -> None:
    if not can_transition(frm, to):
        raise InvalidOrderTransition(client_order_id, frm, to)
