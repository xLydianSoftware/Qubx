"""Exhaustive tests for the pure order state machine.

The expected non-terminal edges below are written out INDEPENDENTLY of the module's
own table, so this acts as a true oracle: a silent change to LEGAL_TRANSITIONS fails
here. Terminalization is checked by rule (any live state may terminalize; terminal
states never transition).
"""

import pytest

from qubx.core.basics import OrderStatus
from qubx.core.exceptions import InvalidOrderTransition
from qubx.core.order_state_machine import can_transition, validate_transition

S = OrderStatus
ALL = list(OrderStatus)
TERMINAL = [S.FILLED, S.CANCELED, S.REJECTED, S.EXPIRED]
NON_TERMINAL = [S.INITIALIZED, S.SUBMITTED, S.ACCEPTED, S.PARTIALLY_FILLED, S.PENDING_CANCEL, S.PENDING_UPDATE]

# Independent reference of non-terminal -> non-terminal edges (mirrors the design, NOT
# imported from the module). Terminal targets are intentionally excluded (rule-based).
EXPECTED_NON_TERMINAL_EDGES = {
    S.INITIALIZED: {S.SUBMITTED},
    S.SUBMITTED: {S.ACCEPTED, S.PARTIALLY_FILLED, S.PENDING_CANCEL},
    S.ACCEPTED: {S.PARTIALLY_FILLED, S.PENDING_CANCEL, S.PENDING_UPDATE},
    S.PARTIALLY_FILLED: {S.PENDING_CANCEL, S.PENDING_UPDATE},
    S.PENDING_CANCEL: {S.SUBMITTED, S.ACCEPTED, S.PARTIALLY_FILLED},
    S.PENDING_UPDATE: {S.SUBMITTED, S.ACCEPTED, S.PARTIALLY_FILLED, S.PENDING_CANCEL},
}


def test_terminal_states_have_no_outgoing_edges():
    for frm in TERMINAL:
        for to in ALL:
            assert not can_transition(frm, to), f"{frm} should not transition to {to}"


def test_any_live_state_can_terminalize():
    # venue-authoritative: liquidation / ADL / admin-cancel / late-reject-after-partial.
    for frm in NON_TERMINAL:
        for term in TERMINAL:
            assert can_transition(frm, term), f"{frm} -> {term} must be allowed"


@pytest.mark.parametrize("frm", NON_TERMINAL)
@pytest.mark.parametrize("to", NON_TERMINAL)
def test_non_terminal_edges_match_reference(frm, to):
    assert can_transition(frm, to) == (to in EXPECTED_NON_TERMINAL_EDGES[frm])


def test_pending_states_revert_to_live_states():
    # cancel/update reject (and sweep give-up) revert a PENDING_* order to a live state.
    for revert_to in (S.SUBMITTED, S.ACCEPTED, S.PARTIALLY_FILLED):
        assert can_transition(S.PENDING_CANCEL, revert_to)
        assert can_transition(S.PENDING_UPDATE, revert_to)


def test_cannot_modify_before_venue_ack():
    # a not-yet-acked SUBMITTED order can be cancelled but not modified.
    assert can_transition(S.SUBMITTED, S.PENDING_CANCEL)
    assert not can_transition(S.SUBMITTED, S.PENDING_UPDATE)


def test_validate_transition_raises_with_context():
    with pytest.raises(InvalidOrderTransition) as exc:
        validate_transition("cid-7", S.FILLED, S.ACCEPTED)
    assert exc.value.client_order_id == "cid-7"
    assert exc.value.current is S.FILLED
    assert exc.value.attempted is S.ACCEPTED


def test_validate_transition_allows_legal_move():
    # returns None, does not raise
    assert validate_transition("cid-1", S.ACCEPTED, S.REJECTED) is None
