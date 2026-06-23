from qubx.core.basics import OrderStatus


def test_order_status_terminal_states():
    assert OrderStatus.FILLED.is_terminal
    assert OrderStatus.CANCELED.is_terminal
    assert OrderStatus.REJECTED.is_terminal
    assert OrderStatus.EXPIRED.is_terminal
    assert not OrderStatus.SUBMITTED.is_terminal
    assert not OrderStatus.ACCEPTED.is_terminal
    assert not OrderStatus.PARTIALLY_FILLED.is_terminal


def test_order_status_inflight_states():
    assert OrderStatus.SUBMITTED.is_inflight
    assert OrderStatus.PENDING_CANCEL.is_inflight
    assert OrderStatus.PENDING_UPDATE.is_inflight
    assert not OrderStatus.ACCEPTED.is_inflight
    assert not OrderStatus.FILLED.is_inflight
