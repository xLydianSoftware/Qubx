import numpy as np
import pytest

from qubx.core.account_manager import AccountManager, AccountManagerConfig, AccountState
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
        # add_order requires a terminal-at timestamp on terminal adds (eviction registration)
        last_updated_at=np.datetime64("2026-05-28T00:00:00") if status.is_terminal else None,
    )


# Representative cases exercising the PUBLIC transition_order wiring (status set vs
# raise). The exhaustive state-machine table is pinned in test_order_state_machine.py.
LEGAL = [
    (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED),
    (OrderStatus.SUBMITTED, OrderStatus.PENDING_CANCEL),
    (OrderStatus.SUBMITTED, OrderStatus.REJECTED),
    (OrderStatus.ACCEPTED, OrderStatus.PENDING_CANCEL),
    (OrderStatus.ACCEPTED, OrderStatus.PENDING_UPDATE),
    (OrderStatus.ACCEPTED, OrderStatus.FILLED),
    # venue can terminalize a live order from any state (liquidation / risk reject /
    # late reject after accept).
    (OrderStatus.ACCEPTED, OrderStatus.REJECTED),
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.EXPIRED),
    (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED),
    (OrderStatus.PENDING_CANCEL, OrderStatus.ACCEPTED),  # revert on cancel-reject
    (OrderStatus.PENDING_UPDATE, OrderStatus.ACCEPTED),
]

ILLEGAL = [
    (OrderStatus.FILLED, OrderStatus.ACCEPTED),  # terminal: no outgoing edge
    (OrderStatus.CANCELED, OrderStatus.PARTIALLY_FILLED),
    (OrderStatus.REJECTED, OrderStatus.SUBMITTED),
    (OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE),  # can't modify before venue ack
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


def test_illegal_transition_carries_context():
    am = _am()
    am._states["binance"].add_order(_make_order(status=OrderStatus.FILLED))
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
    am._states["binance"].add_order(_make_order(status=OrderStatus.ACCEPTED))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    assert am._states["binance"].get_pre_pending("cid-1") is OrderStatus.ACCEPTED


def test_pre_pending_status_captured_for_pending_update():
    am = _am()
    am._states["binance"].add_order(_make_order(status=OrderStatus.PARTIALLY_FILLED))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    assert am._states["binance"].get_pre_pending("cid-1") is OrderStatus.PARTIALLY_FILLED


def test_pre_pending_status_kept_across_pending_update_to_pending_cancel():
    # First-entry capture: cancelling while an amend is in flight must keep the
    # ORIGINAL revert target (ACCEPTED), not record the intermediate PENDING_UPDATE.
    am = _am()
    am._states["binance"].add_order(_make_order(status=OrderStatus.ACCEPTED))
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    assert am._states["binance"].get_pre_pending("cid-1") is OrderStatus.ACCEPTED


def test_non_pending_transition_does_not_set_pre_pending():
    am = _am()
    am._states["binance"].add_order(_make_order(status=OrderStatus.SUBMITTED))
    am.transition_order("binance", "cid-1", OrderStatus.ACCEPTED)
    assert am._states["binance"].get_pre_pending("cid-1") is None


def test_get_orders_filters_by_origin():
    am = _am()
    am._states["binance"].add_order(_make_order(cid="fw"))
    ext = _make_order(cid="ext")
    ext.origin = OrderOrigin.EXTERNAL
    am._states["binance"].add_order(ext)
    fw_only = am.get_orders(origin=OrderOrigin.FRAMEWORK)
    assert set(fw_only) == {"fw"}


def test_get_order_searches_all_states():
    am = _am()
    am._states["binance"].add_order(_make_order(cid="cid-1"))
    assert am.get_order("cid-1") is not None
    assert am.get_order("nope") is None
