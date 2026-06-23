"""Exhaustive oracle tests for the pure order state machine.

The expected non-terminal edges below are written out INDEPENDENTLY of the module's
own table, so this acts as a true oracle: a silent change to TRANSITIONS fails here.
The terminalization rules (any live state may terminalize; terminal states never
transition) are pinned by the ported PR #302 suite in
tests/qubx/core/account_manager/state_machine_test.py.
"""

import pytest

from qubx.core.account_manager.state_machine import can_transition, validate_transition
from qubx.core.basics import OrderStatus
from qubx.core.exceptions import InvalidOrderTransition

S = OrderStatus
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


@pytest.mark.parametrize("frm", NON_TERMINAL)
@pytest.mark.parametrize("to", NON_TERMINAL)
def test_non_terminal_edges_match_reference(frm, to):
    assert can_transition(frm, to) == (to in EXPECTED_NON_TERMINAL_EDGES[frm])


def test_validate_transition_raises_with_context():
    with pytest.raises(InvalidOrderTransition) as exc:
        validate_transition("cid-7", S.FILLED, S.ACCEPTED)
    assert exc.value.client_order_id == "cid-7"
    assert exc.value.current is S.FILLED
    assert exc.value.attempted is S.ACCEPTED


def test_validate_transition_silent_on_legal():
    # the single silent-on-legal pin; covers both can_transition branches —
    # rule-based terminalization and a table edge. Returns None, does not raise.
    assert validate_transition("cid-1", S.ACCEPTED, S.REJECTED) is None
    assert validate_transition("cid-1", S.SUBMITTED, S.ACCEPTED) is None
