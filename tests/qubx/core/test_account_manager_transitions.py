from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.account_manager import AccountManager
from qubx.core.basics import Order, OrderOrigin, OrderStatus
from qubx.core.exceptions import InvalidOrderTransition


class _T:
    def time(self):
        return np.datetime64("2026-05-28T00:00:00")


def _am():
    return AccountManager(
        connectors={"binance": MagicMock()},
        base_currencies={"binance": "USDT"},
        time=_T(),
    )


def _make_order(status=OrderStatus.SUBMITTED, cid="cid-1"):
    return Order(
        client_order_id=cid,
        venue_order_id=None,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        submitted_at=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
        # add_order requires a terminal-at timestamp on terminal adds (eviction registration)
        last_updated_at=np.datetime64("2026-05-28T00:00:00") if status.is_terminal else None,
    )


# Representative cases exercising the PUBLIC transition_order wiring (status set vs
# raise). The exhaustive state-machine table is pinned in test_order_state_machine.py.
LEGAL = [
    (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED),
    (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED),  # live -> terminal wiring
]

ILLEGAL = [
    (OrderStatus.FILLED, OrderStatus.ACCEPTED),  # terminal: no outgoing edge
]


@pytest.mark.parametrize("frm,to", LEGAL, ids=lambda x: x.value if x else "")
def test_legal_transitions(frm, to):
    am = _am()
    am._states["binance"].add_order(_make_order(status=frm))
    am.transition_order("binance", "cid-1", to)
    assert am._states["binance"].get_order("cid-1").status is to


@pytest.mark.parametrize("frm,to", ILLEGAL, ids=lambda x: x.value if x else "")
def test_illegal_transitions(frm, to):
    am = _am()
    am._states["binance"].add_order(_make_order(status=frm))
    with pytest.raises(InvalidOrderTransition):
        am.transition_order("binance", "cid-1", to)


def test_transition_unknown_order_raises_keyerror():
    am = _am()
    with pytest.raises(KeyError):
        am.transition_order("binance", "missing", OrderStatus.ACCEPTED)


def test_get_orders_filters_by_origin():
    am = _am()
    am._states["binance"].add_order(_make_order(cid="fw"))
    ext = _make_order(cid="ext")
    ext.origin = OrderOrigin.EXTERNAL
    am._states["binance"].add_order(ext)
    fw_only = am.get_orders(origin=OrderOrigin.FRAMEWORK)
    assert set(fw_only) == {"fw"}
