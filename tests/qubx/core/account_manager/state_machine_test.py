import pytest

from qubx.core.account_manager.state_machine import TRANSITIONS, can_transition, validate_transition
from qubx.core.basics import OrderStatus
from qubx.core.exceptions import InvalidOrderTransition

TERMINAL = [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
LIVE = [
    OrderStatus.INITIALIZED,
    OrderStatus.SUBMITTED,
    OrderStatus.ACCEPTED,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.PENDING_CANCEL,
    OrderStatus.PENDING_UPDATE,
]


def test_any_live_state_can_terminalize():
    for live in LIVE:
        for terminal in TERMINAL:
            assert can_transition(live, terminal), f"{live} -> {terminal}"


def test_terminal_states_have_no_outgoing_edges():
    for terminal in TERMINAL:
        assert all(not can_transition(terminal, to) for to in OrderStatus), terminal


def test_legal_non_terminal_edges_allowed():
    assert can_transition(OrderStatus.SUBMITTED, OrderStatus.ACCEPTED)
    assert can_transition(OrderStatus.ACCEPTED, OrderStatus.PENDING_CANCEL)
    assert can_transition(OrderStatus.PENDING_UPDATE, OrderStatus.ACCEPTED)


def test_illegal_non_terminal_edges_rejected():
    assert not can_transition(OrderStatus.ACCEPTED, OrderStatus.SUBMITTED)
    assert not can_transition(OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE)


def test_validate_transition_raises_on_illegal():
    with pytest.raises(InvalidOrderTransition) as exc:
        validate_transition("c1", OrderStatus.FILLED, OrderStatus.CANCELED)
    assert exc.value.current is OrderStatus.FILLED
    assert exc.value.attempted is OrderStatus.CANCELED


def test_validate_transition_silent_on_legal():
    validate_transition("c1", OrderStatus.SUBMITTED, OrderStatus.ACCEPTED)


def test_table_keys_are_non_terminal_only():
    assert all(not s.is_terminal for s in TRANSITIONS)
