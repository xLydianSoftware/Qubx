import numpy as np
import pytest

from qubx.core.basics import Deal, Order, OrderOrigin, OrderStatus


def _make_order(**overrides):
    defaults = dict(
        client_order_id="qubx-abc123",
        venue_order_id=None,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=None,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=OrderStatus.SUBMITTED,
        time_in_force="gtc",
    )
    defaults.update(overrides)
    return Order(**defaults)


def test_order_has_client_and_venue_order_id():
    order = _make_order()
    assert order.client_order_id == "qubx-abc123"
    assert order.venue_order_id is None


def test_order_legacy_id_property_falls_back_to_client_order_id():
    order = _make_order()
    assert order.id == "qubx-abc123"
    order.venue_order_id = "V1"
    assert order.id == "V1"


def test_order_legacy_id_setter_writes_venue_order_id():
    order = _make_order()
    order.id = "V9"
    assert order.venue_order_id == "V9"


def test_order_legacy_client_id_read_write_aliases_client_order_id():
    order = _make_order()
    assert order.client_id == "qubx-abc123"
    order.client_id = "qubx-xyz"
    assert order.client_order_id == "qubx-xyz"


def test_order_require_venue_id_raises_when_unset():
    order = _make_order()
    with pytest.raises(ValueError):
        order.require_venue_id()


def test_order_defaults_for_new_fields():
    order = _make_order()
    assert order.pre_pending_status is None
    assert order.seen_trade_ids == set()
    assert order.retry_count == 0
    assert order.last_updated_at is None


def test_deal_trade_id_field_and_legacy_id_alias():
    deal = Deal(trade_id="t1", order_id="o1",
                time=np.datetime64("2026-05-28T00:00:00"),
                amount=1.0, price=50_000.0, aggressive=True)
    assert deal.trade_id == "t1"
    assert deal.id == "t1"   # legacy read-only alias
