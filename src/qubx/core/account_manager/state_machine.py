"""
Order state machine for the new account-management design.

This is the bug-magnet of the redesign, kept as a small, pure, I/O-free module so it
can be tested exhaustively. The transition table is derived from the sequence flows in
the ``account-management`` excalidraw (submit / cancel / update / stuck-order recovery).

Design notes / judgement calls (no design doc was available; confirm against it):
  - **Terminalization is venue-authoritative**: a venue can unilaterally move *any* live
    order to a terminal state (``FILLED``/``CANCELED``/``REJECTED``) in any order — GTD/IOC
    expiry, liquidation, ADL, admin cancel, late reject after a partial fill, a fill racing
    a cancel, etc. So ``can_transition`` allows ``<any non-terminal> -> <any terminal>``.
    The explicit ``TRANSITIONS`` table governs only non-terminal -> non-terminal moves
    (accept, enter/revert pending, partial fills).
  - ``PENDING_CANCEL`` and ``PENDING_UPDATE`` may *revert* to a prior live state
    (the captured ``pre_pending_status``) when the venue rejects the cancel/modify, or
    race ahead to a fill.
  - ``STALE`` is a non-terminal quarantine for a ``SUBMITTED`` order the sweep could not
    confirm after N status queries. We do **not** auto-reject (the order may be live at the
    venue); instead it waits for an authoritative signal — a late ack/snapshot resurrects it
    to ``ACCEPTED``, a fill to ``PARTIALLY_FILLED``, and an explicit venue cancel/reject
    terminalizes it via the rule below.
  - Terminal states have no outgoing edges (not even to another terminal).
  - Idempotent no-ops (e.g. re-issuing a cancel on an already ``PENDING_CANCEL`` order)
    are handled by the caller (AccountManager), not encoded as self-edges here.
"""

from enum import Enum


class OrderState(str, Enum):
    """Lifecycle states of a managed order."""

    SUBMITTED = "SUBMITTED"  # local order created, sent to venue, no ack yet
    ACCEPTED = "ACCEPTED"  # venue acknowledged (venue_order_id known)
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # at least one fill, remainder live
    FILLED = "FILLED"  # fully filled (terminal)
    PENDING_CANCEL = "PENDING_CANCEL"  # cancel requested, awaiting venue verdict
    PENDING_UPDATE = "PENDING_UPDATE"  # modify requested, awaiting venue verdict
    STALE = "STALE"  # submitted but never confirmed; quarantined by the sweep (non-terminal)
    CANCELED = "CANCELED"  # cancel acknowledged (terminal)
    REJECTED = "REJECTED"  # venue/framework rejected (terminal)


TERMINAL_STATES: frozenset[OrderState] = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED}
)

PENDING_STATES: frozenset[OrderState] = frozenset(
    {OrderState.PENDING_CANCEL, OrderState.PENDING_UPDATE}
)


# Non-terminal -> non-terminal edges only. Transitions to terminal states are governed by
# the venue-authoritative rule in ``can_transition`` (any live order may be terminalized).
TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.PENDING_CANCEL,
            OrderState.PENDING_UPDATE,
            OrderState.PARTIALLY_FILLED,
            OrderState.STALE,  # sweep quarantine: never confirmed after N status queries
        }
    ),
    OrderState.STALE: frozenset(
        {
            # resolved by a late authoritative signal
            OrderState.ACCEPTED,  # late ack / snapshot says live
            OrderState.PARTIALLY_FILLED,  # late fill
        }
    ),
    OrderState.ACCEPTED: frozenset(
        {
            OrderState.PENDING_CANCEL,
            OrderState.PENDING_UPDATE,
            OrderState.PARTIALLY_FILLED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.PENDING_UPDATE,
        }
    ),
    OrderState.PENDING_CANCEL: frozenset(
        {
            # revert targets (pre_pending_status) on cancel-rejected / sweep give-up
            OrderState.SUBMITTED,
            OrderState.ACCEPTED,
            OrderState.PARTIALLY_FILLED,
        }
    ),
    OrderState.PENDING_UPDATE: frozenset(
        {
            # confirm or revert (pre_pending_status) on update-rejected / sweep give-up
            OrderState.SUBMITTED,
            OrderState.ACCEPTED,
            OrderState.PARTIALLY_FILLED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELED: frozenset(),
    OrderState.REJECTED: frozenset(),
}


class IllegalOrderTransition(Exception):
    """Raised when an order is asked to make a transition the state machine forbids."""

    def __init__(self, frm: OrderState, to: OrderState):
        self.frm = frm
        self.to = to
        super().__init__(f"Illegal order transition: {frm.value} -> {to.value}")


def is_terminal(state: OrderState) -> bool:
    return state in TERMINAL_STATES


def is_pending(state: OrderState) -> bool:
    return state in PENDING_STATES


def can_transition(frm: OrderState, to: OrderState) -> bool:
    if frm in TERMINAL_STATES:
        return False  # terminal states never transition
    if to in TERMINAL_STATES:
        return True  # venue-authoritative terminalization from any live state
    return to in TRANSITIONS[frm]


def transition(frm: OrderState, to: OrderState) -> OrderState:
    """Validate ``frm -> to`` and return ``to``; raise ``IllegalOrderTransition`` if illegal."""
    if not can_transition(frm, to):
        raise IllegalOrderTransition(frm, to)
    return to
