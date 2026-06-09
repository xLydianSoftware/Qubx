from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager
from qubx.core.basics import Deal, Instrument, MarketType, Order, OrderChange, OrderOrigin, OrderStatus
from qubx.core.events import (
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)


def _assert_empty(result):
    assert result.order is None
    assert result.order_change is None
    assert result.deal is None
    assert result.position is None


class _T:
    def time(self):
        return np.datetime64("2026-05-28T00:00:00")


def _Inst() -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="binance",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _am():
    am = AccountManager.__new__(AccountManager)
    am._init_state(
        connectors={"binance": MagicMock()},
        base_currencies={"binance": "USDT"},
        strategy=MagicMock(),
        time=_T(),
        cfg=None,
        account_id="test",
        tcc=None,
    )
    return am


def add_order(state, cid="cid-1", status=OrderStatus.SUBMITTED, instrument=None, quantity=1.0):
    state.add_order(
        Order(
            client_order_id=cid,
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=instrument,
            time=np.datetime64("2026-05-28T00:00:00"),
            quantity=quantity,
            price=50_000.0,
            side="BUY",
            status=status,
            time_in_force="gtc",
        )
    )


def _fill(trade_id="t1", amount=0.5, price=50_000.0):
    return Deal(
        trade_id=trade_id,
        order_id="V1",
        time=np.datetime64("2026-05-28T00:00:00"),
        amount=amount,
        price=price,
        aggressive=True,
    )


def test_apply_event_for_unknown_exchange_returns_empty_result():
    am = _am()

    class _Other:
        exchange = "kraken"
        symbol = "ETHUSDT"

    result = am.apply(
        OrderAcceptedEvent(
            instrument=_Other(),
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    _assert_empty(result)


def test_accepted_sets_venue_and_transitions():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), instrument=inst)
    r = am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    assert r.order is not None
    assert r.order.status is OrderStatus.ACCEPTED
    assert r.order.venue_order_id == "V1"
    assert r.order.accepted_at == np.datetime64("2026-05-28")
    assert r.order_change is OrderChange.ACCEPTED
    assert r.deal is None and r.position is None


def test_accepted_double_ack_fires_once():
    # Routine REST-ack + WS-ack double delivery: the first ACCEPTED transitions and
    # reports, the second is suppressed (empty result -> no on_order_update re-fire).
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), instrument=inst)
    evt = OrderAcceptedEvent(
        instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
    )
    r1 = am.apply(evt)
    assert r1.order is not None and r1.order_change is OrderChange.ACCEPTED
    r2 = am.apply(evt)
    _assert_empty(r2)
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.ACCEPTED


def test_accepted_during_pending_cancel_is_side_effect_only():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    r = am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    _assert_empty(r)
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.PENDING_CANCEL
    assert o.venue_order_id == "V1"


def test_accepted_during_pending_update_transitions_to_accepted():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    r = am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V2",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    assert r.order is not None and r.order_change is OrderChange.ACCEPTED
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.venue_order_id == "V2"


def test_accepted_on_terminal_order_sets_venue_without_transition():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    # late OrderAccepted on an already-filled order: venue id only, no transition, no callback
    r = am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    _assert_empty(r)
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.FILLED


def test_happy_path_accepted_partial_filled():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), instrument=inst)
    r = am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    assert r.order_change is OrderChange.ACCEPTED
    r = am.apply(
        OrderPartiallyFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=0.5))
    )
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED
    assert r.order_change is OrderChange.PARTIALLY_FILLED
    assert r.deal is not None and r.deal.trade_id == "t1"
    assert r.position is not None
    assert r.order.filled_quantity == 0.5
    r = am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t2", amount=0.5)
        )
    )
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.order_change is OrderChange.FILLED
    assert r.deal is not None and r.deal.trade_id == "t2"
    assert r.order.filled_quantity == 1.0


def test_fill_dedup_by_trade_id():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    evt = OrderPartiallyFilledEvent(
        instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.5)
    )
    r1 = am.apply(evt)
    assert r1.deal is not None
    r2 = am.apply(evt)
    _assert_empty(r2)  # deduped -> no deal/position, no status change
    assert am.get_state("binance").get_order("cid-1").filled_quantity == 0.5


def test_fill_redelivered_trade_id_promotes_status_without_deal():
    # Two-stream FILLED promotion re-delivers the already-applied deal: the order still
    # transitions to FILLED, but the deal is deduped -> delivered downstream exactly once.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    f = _fill(trade_id="t1", amount=0.5)
    r1 = am.apply(OrderPartiallyFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=f))
    assert r1.deal is not None and r1.deal.trade_id == "t1"
    r2 = am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=f))
    assert r2.order is not None and r2.order.status is OrderStatus.FILLED
    assert r2.order_change is OrderChange.FILLED
    assert r2.deal is None and r2.position is None  # already applied via the partial
    assert r2.order.filled_quantity == 0.5


def test_subsequent_partial_fill_is_execution_only():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.PARTIALLY_FILLED, instrument=inst)
    r = am.apply(
        OrderPartiallyFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t2", amount=0.2)
        )
    )
    assert r.order is None and r.order_change is None  # no status transition
    assert r.deal is not None and r.deal.trade_id == "t2"
    assert r.position is not None


def test_partial_fill_during_pending_cancel_applies_without_transition():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    r = am.apply(
        OrderPartiallyFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=0.3))
    )
    assert r.order is None  # pending status not disturbed -> execution only
    assert r.deal is not None
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.PENDING_CANCEL
    assert o.filled_quantity == 0.3


def test_pending_update_overfill_keeps_filled_intact_and_warns():
    from qubx import logger

    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        am.apply(
            OrderPartiallyFilledEvent(
                instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=1.5)
            )
        )
    finally:
        logger.remove(sink_id)

    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.PENDING_UPDATE
    assert o.filled_quantity == 1.5  # left intact — NOT clamped to quantity
    assert any("leaving filled intact" in m for m in messages)


def test_canceled_transitions_to_canceled():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    r = am.apply(OrderCanceledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    assert r.order is not None and r.order.status is OrderStatus.CANCELED
    assert r.order_change is OrderChange.CANCELED
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.CANCELED


def test_expired_transitions_to_expired():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    r = am.apply(OrderExpiredEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    assert r.order is not None and r.order.status is OrderStatus.EXPIRED
    assert r.order_change is OrderChange.EXPIRED
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.EXPIRED


def test_rejected_transitions_and_stores_reason():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.SUBMITTED, instrument=inst)
    r = am.apply(OrderRejectedEvent(instrument=inst, client_order_id="cid-1", reason="insufficient funds"))
    assert r.order is not None and r.order_change is OrderChange.REJECTED
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.REJECTED
    assert o.rejected_reason == "insufficient funds"


def test_rejected_for_unknown_order_returns_empty_result():
    am = _am()
    inst = _Inst()
    result = am.apply(OrderRejectedEvent(instrument=inst, client_order_id="nope", reason="x"))
    _assert_empty(result)


def test_unknown_order_lifecycle_events_create_no_orders():
    # Accepted/canceled/expired are resolve-only: an unknown order id must not
    # materialize a phantom EXTERNAL order just to terminalize it, and fires nothing.
    am = _am()
    inst = _Inst()
    r1 = am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="ghost", venue_order_id="VX", accepted_at=np.datetime64("2026-05-28")
        )
    )
    r2 = am.apply(OrderCanceledEvent(instrument=inst, client_order_id="ghost", venue_order_id="VX"))
    r3 = am.apply(OrderExpiredEvent(instrument=inst, client_order_id="ghost", venue_order_id="VX"))
    for r in (r1, r2, r3):
        _assert_empty(r)
    state = am.get_state("binance")
    assert state.get_orders() == {}
    assert not state.has_active_order("ext:VX")
    assert state.get_order_by_venue_id("VX") is None


def test_fill_without_instrument_for_unknown_order_is_noop():
    # Money-carrying events materialize external orders ONLY when the event carries an
    # instrument — without one there is no position to track, so nothing materializes.
    am = _am()
    r = am.apply(OrderFilledEvent(instrument=None, client_order_id="ghost", venue_order_id="VX", fill=_fill()))
    _assert_empty(r)
    assert am.get_state("binance").get_orders() == {}


def test_rejected_resolves_by_venue_id_when_cid_unknown():
    # A reject addressed by venue id only (the connector filled client_order_id with the
    # venue id because the real cid was unknown) must still resolve via the venue-id index.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.SUBMITTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    am.apply(OrderRejectedEvent(instrument=inst, client_order_id="V1", venue_order_id="V1", reason="rejected"))
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.REJECTED
    assert o.rejected_reason == "rejected"


def test_updated_in_place_modifies_fields_no_transition():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    r = am.apply(
        OrderUpdatedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", new_price=49_000.0, new_quantity=2.0
        )
    )
    assert r.order is not None
    assert r.order_change is OrderChange.UPDATED  # fires despite no status change
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.price == 49_000.0
    assert o.quantity == 2.0


def test_updated_during_pending_update_transitions_and_reindexes_venue():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    r = am.apply(
        OrderUpdatedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V2", new_price=48_000.0, new_quantity=None
        )
    )
    assert r.order is not None and r.order_change is OrderChange.UPDATED
    state = am.get_state("binance")
    o = state.get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.venue_order_id == "V2"
    assert o.price == 48_000.0
    assert state.get_order_by_venue_id("V2").client_order_id == "cid-1"
    assert state.get_order_by_venue_id("V1") is None


def test_updated_confirms_back_to_partially_filled():
    # PENDING_UPDATE confirmation reverts to the captured pre-pending status, not a
    # hardcoded ACCEPTED — a partially-filled order must keep PARTIALLY_FILLED.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.PARTIALLY_FILLED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    r = am.apply(
        OrderUpdatedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", new_price=48_000.0, new_quantity=None
        )
    )
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED
    assert r.order_change is OrderChange.UPDATED


def test_cancel_rejected_reverts_to_pre_pending_status():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    r = am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="too late"))
    assert r.order is not None and r.order_change is OrderChange.CANCEL_REJECTED
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert am.get_state("binance").get_pre_pending("cid-1") is None


def test_cancel_rejected_after_pending_update_reverts_to_original_status():
    # ACCEPTED -> PENDING_UPDATE -> PENDING_CANCEL -> cancel-rejected must revert to the
    # ORIGINAL pre-pending status (ACCEPTED). Recording the intermediate PENDING_UPDATE
    # used to strand the order in PENDING_CANCEL (illegal PENDING_CANCEL -> PENDING_UPDATE).
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="too late"))
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert am.get_state("binance").get_pre_pending("cid-1") is None


def test_cancel_rejected_reverts_to_partially_filled():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.PARTIALLY_FILLED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="too late"))
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert am.get_state("binance").get_pre_pending("cid-1") is None


def test_cancel_rejected_unexpected_state_returns_empty_result():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    result = am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="x"))
    _assert_empty(result)
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.ACCEPTED


def test_cancel_rejected_resolves_by_venue_id_when_cid_unknown():
    # Venue-id-only cancel that the venue refused: the cancel-reject (addressed by venue id)
    # must resolve the PENDING_CANCEL order via the venue-id index and revert it.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="V1", venue_order_id="V1", reason="too late"))
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED


def test_update_rejected_reverts_to_pre_pending_status():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    r = am.apply(OrderUpdateRejectedEvent(instrument=inst, client_order_id="cid-1", reason="bad price"))
    assert r.order is not None and r.order_change is OrderChange.UPDATE_REJECTED
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert am.get_state("binance").get_pre_pending("cid-1") is None


def test_materialize_external_for_unknown_cid_and_venue():
    am = _am()
    inst = _Inst()
    r = am.apply(OrderPartiallyFilledEvent(instrument=inst, client_order_id="alien", venue_order_id="VX", fill=_fill()))
    assert r.order is not None and r.order.origin is OrderOrigin.EXTERNAL
    assert r.deal is not None
    assert r.position is not None and r.position.quantity == 0.5
    state = am.get_state("binance")
    o = state.get_order_by_venue_id("VX")
    assert o is not None
    assert o.origin is OrderOrigin.EXTERNAL
    assert o.client_order_id == "ext:VX"
    assert o.filled_quantity == 0.5


def test_fill_before_accept_submitted_to_filled():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.SUBMITTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0
    assert o.venue_order_id == "V1"


def test_resolve_via_terminal_history_fallback():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    # evict the terminal order to history, then a late accept must resolve there,
    # not materialize a phantom EXTERNAL order.
    am.get_state("binance").evict_to_history("cid-1")
    am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    state = am.get_state("binance")
    # no phantom EXTERNAL order materialized
    assert not state.has_active_order("ext:V1")
    assert state.get_order("cid-1").status is OrderStatus.FILLED


def test_resolve_via_venue_id_index():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    # event carries no client_order_id match (different cid) but a known venue id
    am.apply(
        OrderPartiallyFilledEvent(instrument=inst, client_order_id="other", venue_order_id="V1", fill=_fill(amount=0.4))
    )
    o = am.get_state("binance").get_order("cid-1")
    assert o.filled_quantity == 0.4
    assert not am.get_state("binance").has_active_order("ext:V1")


def test_late_cancel_on_filled_order_is_noop():
    # C1: a late OrderCanceled on an already-FILLED order (still in active_orders
    # during the grace window) must NOT flip FILLED -> CANCELED, and fires nothing.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.FILLED
    # late cancel: benign no-op, no exception, status unchanged, empty result
    r = am.apply(OrderCanceledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    _assert_empty(r)
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.FILLED


def test_late_expired_on_terminal_order_is_noop():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderCanceledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    r = am.apply(OrderExpiredEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    _assert_empty(r)
    assert am.get_state("binance").get_order("cid-1").status is OrderStatus.CANCELED


def test_late_fill_on_filled_order_is_noop():
    # A second OrderFilled (new trade_id) on an already-FILLED order is ignored:
    # status stays FILLED and filled_quantity is unchanged.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=1.0)
        )
    )
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0
    # late fill with a brand-new trade_id: ignored, no double-count, empty result
    r = am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t2", amount=0.5)
        )
    )
    _assert_empty(r)
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0


def test_late_fill_on_evicted_order_does_not_raise():
    # C2: once an order is evicted to _terminal_history it is no longer in
    # active_orders, so any mutator doing active_orders[cid] would KeyError.
    # Late fill/cancel/update events resolving to it via cid must be benign
    # no-ops with no phantom EXTERNAL order in active_orders.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.get_state("binance").set_venue_id("cid-1", "V1")
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    am.get_state("binance").evict_to_history("cid-1")
    assert not am.get_state("binance").has_active_order("cid-1")

    # each of these resolves to the evicted order via the terminal-history fallback
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t-late", amount=0.5)
        )
    )
    am.apply(OrderCanceledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    am.apply(
        OrderUpdatedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", new_price=49_000.0, new_quantity=2.0
        )
    )

    state = am.get_state("binance")
    assert not state.has_active_order("cid-1")
    assert not state.has_active_order("ext:V1")
    assert not state.has_active_order("ext:cid-1")
    # the evicted order remains FILLED and untouched
    assert state.get_order("cid-1").status is OrderStatus.FILLED


def test_rejected_with_no_instrument_routes_via_client_order_id():
    # SimulatedConnector emits OrderRejectedEvent with instrument=None when the
    # order is not found in the OME. Verify the AM still routes the event and
    # transitions the order to REJECTED (not short-circuits to None).
    am = _am()
    # Note: instrument=None here, so the order is seeded without one.
    add_order(am.get_state("binance"), cid="cid-no-inst", status=OrderStatus.SUBMITTED, instrument=None)
    result = am.apply(
        OrderRejectedEvent(instrument=None, client_order_id="cid-no-inst", reason="order not found in OME")
    )
    o = am.get_state("binance").get_order("cid-no-inst")
    assert o is not None, "order must be retrievable after reject"
    assert o.status is OrderStatus.REJECTED
    assert o.rejected_reason == "order not found in OME"
    # apply() must return the real order, not an empty result
    assert result.order is not None
    assert result.order.client_order_id == "cid-no-inst"
    assert result.order_change is OrderChange.REJECTED


def test_cancel_rejected_with_no_instrument_reverts_pending_cancel():
    # OrderCancelRejectedEvent with instrument=None must still revert the order
    # from PENDING_CANCEL back to its pre_pending_status via the state lookup.
    am = _am()
    add_order(am.get_state("binance"), cid="cid-pc", status=OrderStatus.ACCEPTED, instrument=None)
    am.transition_order("binance", "cid-pc", OrderStatus.PENDING_CANCEL)
    result = am.apply(OrderCancelRejectedEvent(instrument=None, client_order_id="cid-pc", reason="too late"))
    o = am.get_state("binance").get_order("cid-pc")
    assert o is not None
    assert o.status is OrderStatus.ACCEPTED
    assert am.get_state("binance").get_pre_pending("cid-pc") is None
    assert result.order is not None
    assert result.order.client_order_id == "cid-pc"
    assert result.order_change is OrderChange.CANCEL_REJECTED


def test_get_order_history_records_transitions():
    # B5 audit log: every AM-driven status change is appended to Order.transitions and
    # surfaced via get_order_history (searches active orders + terminal-history buffer).
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.SUBMITTED, instrument=inst, quantity=1.0)
    am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    history = am.get_order_history("cid-1")
    assert [(t.from_status, t.to_status) for t in history] == [
        (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED),
        (OrderStatus.ACCEPTED, OrderStatus.FILLED),
    ]
    assert am.get_order_history("does-not-exist") == []


def test_get_metrics_counts_transitions_by_status():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.SUBMITTED, instrument=inst, quantity=1.0)
    am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    counts = am.get_metrics()["binance"]
    assert counts[OrderStatus.ACCEPTED.value] == 1
    assert counts[OrderStatus.FILLED.value] == 1
