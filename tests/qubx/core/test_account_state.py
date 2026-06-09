"""Ours-only AccountState coverage.

The primary AccountState suite (ported from PR #302) lives in
tests/qubx/core/account_manager/state_test.py; this file keeps only the behaviors
that suite does not pin: ACCEPTED adds not indexed in-flight, remove_order as a
full drop (vs evict_to_history), the bounded terminal-history ring buffer, and
position map reads.
"""

import numpy as np

from qubx.core.account_manager import AccountState
from qubx.core.basics import Order, OrderOrigin, OrderStatus, Position


def _make_order(cid="qubx-1", status=OrderStatus.SUBMITTED, venue_id=None, last_updated_at=None):
    return Order(
        client_order_id=cid,
        venue_order_id=venue_id,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        submitted_at=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
        last_updated_at=last_updated_at,
    )


def testadd_order_accepted_not_indexed_inflight():
    # only SUBMITTED / PENDING_* are in-flight; a resting ACCEPTED order is not swept
    state = AccountState(exchange="binance", base_currency="USDT")
    state.add_order(_make_order(status=OrderStatus.ACCEPTED))
    assert "qubx-1" not in state._inflight_index


def testremove_order_drains_indexes():
    # remove_order is the FULL drop (submit never reached the venue): unlike
    # evict_to_history, the order must NOT remain resolvable via terminal history.
    state = AccountState(exchange="binance", base_currency="USDT")
    state.add_order(_make_order())
    state.set_venue_id("qubx-1", "VENUE_ABC")
    state.remove_order("qubx-1")
    assert state.get_order("qubx-1") is None
    assert state.get_order_by_venue_id("VENUE_ABC") is None
    assert "qubx-1" not in state._inflight_index


def test_terminal_history_bounded_by_constructor_size():
    state = AccountState(exchange="binance", base_currency="USDT", terminal_history_size=2)
    now = np.datetime64("2026-05-28T00:01:00")
    for i in range(3):
        cid = f"qubx-{i}"
        state.add_order(_make_order(cid=cid))
        state.transition_order(cid, OrderStatus.FILLED, now)
        state.evict_to_history(cid)
    # ring buffer keeps only the most recent `terminal_history_size` evictions
    assert state.get_order("qubx-0") is None
    assert state.get_order("qubx-1") is not None
    assert state.get_order("qubx-2") is not None


def test_set_and_get_position():
    state = AccountState(exchange="binance", base_currency="USDT")
    inst = object()
    pos = Position.__new__(Position)
    state.set_position(inst, pos)
    assert state.get_position(inst) is pos
    assert state.get_positions()[inst] is pos
    assert state.get_position(object()) is None
