import numpy as np
import pytest

from qubx.core.account_manager import AccountManager
from qubx.core.account_manager_config import AccountManagerConfig
from qubx.core.account_state import AccountState
from qubx.core.basics import Order, OrderOrigin, OrderStatus
from qubx.core.exceptions import InvalidOrderTransition


class _T:
    def time(self):
        return np.datetime64("2026-05-28T00:00:00")


def _am():
    am = AccountManager.__new__(AccountManager)
    am._states = {"binance": AccountState(exchange="binance")}
    am._cfg = AccountManagerConfig()
    am._time = _T()
    return am


def _make_order(status=OrderStatus.SUBMITTED, cid="cid-1"):
    return Order(
        client_order_id=cid,
        venue_order_id=None,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=status,
        time_in_force="gtc",
    )


LEGAL = [
    (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED),
    (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED),
    (OrderStatus.SUBMITTED, OrderStatus.PENDING_CANCEL),
    (OrderStatus.SUBMITTED, OrderStatus.FILLED),
    (OrderStatus.SUBMITTED, OrderStatus.CANCELED),
    (OrderStatus.SUBMITTED, OrderStatus.REJECTED),
    (OrderStatus.SUBMITTED, OrderStatus.EXPIRED),
    (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED),
    (OrderStatus.ACCEPTED, OrderStatus.PENDING_CANCEL),
    (OrderStatus.ACCEPTED, OrderStatus.PENDING_UPDATE),
    (OrderStatus.ACCEPTED, OrderStatus.FILLED),
    (OrderStatus.ACCEPTED, OrderStatus.CANCELED),
    (OrderStatus.ACCEPTED, OrderStatus.EXPIRED),
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL),
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_UPDATE),
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED),
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED),
    (OrderStatus.PENDING_CANCEL, OrderStatus.FILLED),
    (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED),
    (OrderStatus.PENDING_UPDATE, OrderStatus.ACCEPTED),
    (OrderStatus.PENDING_UPDATE, OrderStatus.PARTIALLY_FILLED),
    (OrderStatus.PENDING_UPDATE, OrderStatus.PENDING_CANCEL),
    (OrderStatus.PENDING_UPDATE, OrderStatus.CANCELED),
]

ILLEGAL = [
    (OrderStatus.FILLED, OrderStatus.ACCEPTED),
    (OrderStatus.CANCELED, OrderStatus.PARTIALLY_FILLED),
    (OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE),
    (OrderStatus.ACCEPTED, OrderStatus.REJECTED),
]


def test_legal_count_is_23():
    assert len(LEGAL) == 23


@pytest.mark.parametrize("frm,to", LEGAL, ids=lambda x: x.value if x else "")
def test_legal_transitions(frm, to):
    am = _am()
    am._states["binance"]._add_order(_make_order(status=frm))
    am.transition_order("binance", "cid-1", to)
    assert am._states["binance"].get_order("cid-1").status is to


@pytest.mark.parametrize("frm,to", ILLEGAL, ids=lambda x: x.value if x else "")
def test_illegal_transitions(frm, to):
    am = _am()
    am._states["binance"]._add_order(_make_order(status=frm))
    with pytest.raises(InvalidOrderTransition):
        am.transition_order("binance", "cid-1", to)


def test_illegal_transition_carries_context():
    am = _am()
    am._states["binance"]._add_order(_make_order(status=OrderStatus.FILLED))
    with pytest.raises(InvalidOrderTransition) as exc:
        am.transition_order("binance", "cid-1", OrderStatus.ACCEPTED)
    assert exc.value.client_order_id == "cid-1"
    assert exc.value.current is OrderStatus.FILLED
    assert exc.value.attempted is OrderStatus.ACCEPTED


def test_transition_unknown_order_raises_keyerror():
    am = _am()
    with pytest.raises(KeyError):
        am.transition_order("binance", "missing", OrderStatus.ACCEPTED)


def test_pre_pending_status_captured_for_pending_cancel():
    am = _am()
    am._states["binance"]._add_order(_make_order(status=OrderStatus.ACCEPTED))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    assert am._states["binance"].get_order("cid-1").pre_pending_status is OrderStatus.ACCEPTED


def test_pre_pending_status_captured_for_pending_update():
    am = _am()
    am._states["binance"]._add_order(_make_order(status=OrderStatus.PARTIALLY_FILLED))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    assert am._states["binance"].get_order("cid-1").pre_pending_status is OrderStatus.PARTIALLY_FILLED


def test_non_pending_transition_does_not_set_pre_pending():
    am = _am()
    am._states["binance"]._add_order(_make_order(status=OrderStatus.SUBMITTED))
    am.transition_order("binance", "cid-1", OrderStatus.ACCEPTED)
    assert am._states["binance"].get_order("cid-1").pre_pending_status is None


def test_get_state_returns_state():
    am = _am()
    assert am.get_state("binance") is am._states["binance"]


def test_get_orders_filters_by_origin():
    am = _am()
    am._states["binance"]._add_order(_make_order(cid="fw"))
    ext = _make_order(cid="ext")
    ext.origin = OrderOrigin.EXTERNAL
    am._states["binance"]._add_order(ext)
    fw_only = am.get_orders(origin=OrderOrigin.FRAMEWORK)
    assert set(fw_only) == {"fw"}


def test_get_order_searches_all_states():
    am = _am()
    am._states["binance"]._add_order(_make_order(cid="cid-1"))
    assert am.get_order("cid-1") is not None
    assert am.get_order("nope") is None
