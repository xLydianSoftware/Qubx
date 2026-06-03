"""Order-lifecycle state machine for the central AccountManager.

Kept as a small, pure, I/O-free module so the transition rules — the bug-magnet of order
management — can be audited and unit-tested in isolation. AccountManager is the only
consumer and routes every status change through ``validate_transition`` (the single
legality chokepoint).

Terminalization is venue-authoritative: a venue can move any live order straight to a
terminal state (FILLED / CANCELED / REJECTED / EXPIRED) from any non-terminal state —
GTD/IOC expiry, liquidation, ADL, admin cancel, a late reject after a partial fill, a
fill racing a cancel. So the explicit table governs only non-terminal → non-terminal
moves; transitions into a terminal state are allowed from any live state by rule, and
terminal states have no outgoing edges.

The non-terminal edges cover the strategy-initiated moves (accept, enter a PENDING_*
state) and the revert targets a PENDING_* order falls back to when the venue rejects the
cancel/modify or the stuck-order sweep gives up.
"""

from qubx.core.basics import OrderStatus
from qubx.core.exceptions import InvalidOrderTransition

# Non-terminal → non-terminal edges only. Terminal targets are governed by the
# venue-authoritative rule in ``can_transition`` (any live order may be terminalized).
LEGAL_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
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
    # PENDING_* revert targets (the pre-pending status) on cancel/update reject or
    # sweep give-up; PENDING_UPDATE confirms back to ACCEPTED/PARTIALLY_FILLED.
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
    """Whether ``frm → to`` is allowed by the state machine."""
    if frm.is_terminal:
        return False  # terminal states never transition
    if to.is_terminal:
        return True  # venue-authoritative terminalization from any live state
    return to in LEGAL_TRANSITIONS.get(frm, set())


def validate_transition(client_order_id: str, frm: OrderStatus, to: OrderStatus) -> None:
    """Raise ``InvalidOrderTransition`` if ``frm → to`` is illegal."""
    if not can_transition(frm, to):
        raise InvalidOrderTransition(client_order_id, frm, to)
