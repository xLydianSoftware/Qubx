"""Rule-based terminalization sweeps + table-structure pin (ported from PR #302).

Per-edge non-terminal behavior and the validate_transition contract are pinned by
the exhaustive oracle in tests/qubx/core/test_order_state_machine.py.
"""

from qubx.core.account_manager.state_machine import TRANSITIONS, can_transition
from qubx.core.basics import OrderStatus

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


def test_table_keys_are_non_terminal_only():
    assert all(not s.is_terminal for s in TRANSITIONS)
