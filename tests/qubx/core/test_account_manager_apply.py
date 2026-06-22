from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager
from qubx.core.basics import Deal, Instrument, MarketType, Order, OrderChange, OrderOrigin, OrderStatus
from qubx.core.events import (
    DealEvent,
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
    return AccountManager(
        connectors={"binance": MagicMock()},
        base_currencies={"binance": "USDT"},
        time=_T(),
        account_id="test",
    )


def add_order(state, cid="cid-1", status=OrderStatus.SUBMITTED, instrument=None, quantity=1.0):
    state.add_order(
        Order(
            client_order_id=cid,
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=instrument,
            submitted_at=np.datetime64("2026-05-28T00:00:00"),
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
    # reports, the second is suppressed (empty result -> no on_order re-fire).
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


def test_accepted_during_pending_update_is_side_effect_only():
    # A duplicate accept (REST + WS acks) racing a pending update must NOT wipe
    # PENDING_UPDATE — a later update-rejected still reverts via pre-pending.
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
    _assert_empty(r)
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.PENDING_UPDATE
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
    # A venue re-sends an order report whose embedded deal was already applied (combined-stream
    # re-delivery / cross-stream race): the order still transitions to FILLED, but the deal is
    # deduped -> delivered downstream exactly once.
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


def test_deal_event_books_without_status_change():
    # Split-stream: the deal drives the ledger only; status comes from order events.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    r = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=_fill(amount=0.4)))
    assert r.order is None and r.order_change is None  # never a status transition
    assert r.deal is not None and r.deal.trade_id == "t1"
    assert r.position is not None and r.position.quantity == 0.4
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.filled_quantity == 0.4


def test_deal_event_duplicate_is_noop():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    evt = DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=_fill(amount=0.4))
    r1 = am.apply(evt)
    assert r1.deal is not None
    r2 = am.apply(evt)
    _assert_empty(r2)  # deduped -> no second booking
    assert am.get_state("binance").get_order("cid-1").filled_quantity == 0.4


def test_cross_stream_dedup_deal_then_status_with_embedded_fill():
    # Trade stream wins the race: the DealEvent books; the FILLED status re-delivering
    # the same trade embedded still transitions but the deal is deduped.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    f = _fill(trade_id="t1", amount=0.5)
    r1 = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=f))
    assert r1.deal is not None and r1.deal.trade_id == "t1"
    r2 = am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=f))
    assert r2.order is not None and r2.order.status is OrderStatus.FILLED
    assert r2.order_change is OrderChange.FILLED
    assert r2.deal is None and r2.position is None  # already applied via DealEvent
    assert r2.order.filled_quantity == 0.5


def test_cross_stream_dedup_status_with_embedded_fill_then_deal():
    # Status stream wins the race: the embedded fill books with the FILLED transition;
    # the late DealEvent re-delivering the same trade is a complete no-op.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    f = _fill(trade_id="t1", amount=0.5)
    r1 = am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=f))
    assert r1.deal is not None and r1.order is not None and r1.order.status is OrderStatus.FILLED
    r2 = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=f))
    _assert_empty(r2)
    assert am.get_state("binance").get_order("cid-1").filled_quantity == 0.5


def test_fill_event_without_fill_transitions_without_booking():
    # Split-stream FILLED status: terminal transition happens, nothing books — the deal
    # arrives separately via DealEvent.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    r = am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=None))
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.order_change is OrderChange.FILLED
    assert r.deal is None and r.position is None
    assert r.order.filled_quantity == 0.0


def test_partial_fill_event_without_fill_transitions_without_booking():
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    r = am.apply(OrderPartiallyFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=None))
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED
    assert r.order_change is OrderChange.PARTIALLY_FILLED
    assert r.deal is None and r.position is None
    assert r.order.filled_quantity == 0.0


def test_late_deal_after_filled_books_when_trade_unseen():
    # Split-stream race: the FILLED status (fill=None) beats the final trade. The order
    # is terminal but still inside the eviction grace window, and the trade was never
    # seen — it must still book (ledger correctness), with no status disturbance.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=None))
    r = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=_fill(amount=1.0)))
    assert r.order is None and r.order_change is None
    assert r.deal is not None and r.deal.trade_id == "t1"
    assert r.position is not None and r.position.quantity == 1.0
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0


def test_deal_after_terminal_eviction_is_noop():
    # Past the grace window the seen-trade table is gone — a re-delivered trade must be
    # suppressed rather than risk double-booking against an evicted order.
    am = _am()
    inst = _Inst()
    state = am.get_state("binance")
    add_order(state, status=OrderStatus.ACCEPTED, instrument=inst)
    f = _fill(amount=1.0)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=f))
    state.evict_to_history("cid-1")
    r = am.apply(DealEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", deal=f))
    _assert_empty(r)
    assert state.get_position(inst).quantity == 1.0  # booked exactly once


def test_deal_event_on_unknown_order_materializes_external():
    # A deal is money-carrying: an unknown order materializes as EXTERNAL and the deal
    # books — but materialization never produces a status-change callback.
    am = _am()
    inst = _Inst()
    r = am.apply(DealEvent(instrument=inst, client_order_id="alien", venue_order_id="VX", deal=_fill(amount=0.3)))
    assert r.order is None and r.order_change is None
    assert r.deal is not None
    assert r.position is not None and r.position.quantity == 0.3
    o = am.get_state("binance").get_order_by_venue_id("VX")
    assert o is not None
    assert o.client_order_id == "ext:VX"
    assert o.origin is OrderOrigin.EXTERNAL
    assert o.status is OrderStatus.ACCEPTED
    assert o.filled_quantity == 0.3


def test_deal_event_without_instrument_for_unknown_order_is_noop():
    am = _am()
    r = am.apply(DealEvent(instrument=None, client_order_id="alien", venue_order_id="VX", deal=_fill()))
    _assert_empty(r)
    assert am.get_state("binance").get_orders() == {}


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
    assert any(m.record["level"].name == "WARNING" and "cid-1" in m for m in messages)


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
    # no code on the event -> error_code stays None
    assert o.error_code is None


def test_rejected_stores_error_code_from_event():
    # A venue reject carries the connector's error code (e.g. the ccxt error class name);
    # the handler threads it onto the order alongside rejected_reason.
    am = _am()
    inst = _Inst()
    add_order(am.get_state("binance"), status=OrderStatus.SUBMITTED, instrument=inst)
    am.apply(
        OrderRejectedEvent(
            instrument=inst, client_order_id="cid-1", reason="insufficient funds", code="InsufficientFunds"
        )
    )
    o = am.get_state("binance").get_order("cid-1")
    assert o.status is OrderStatus.REJECTED
    assert o.error_code == "InsufficientFunds"


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


# --- terminal fill-gap reconciliation (dropped WS fills) -------------------------------- #


def test_filled_books_gap_for_dropped_ws_fills():
    # The reported scenario: a market order fully fills at the venue, but some fills never
    # arrive as deals; the terminal FILLED carries fill=None plus the venue's cumulative
    # filled_quantity. The reducer books the unbooked remainder so position AND filled_quantity
    # converge now, instead of size-only at the next snapshot reconcile.
    am = _am()
    inst = _Inst()
    state = am.get_state("binance")
    add_order(state, status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
    am.apply(
        OrderPartiallyFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.3)
        )
    )
    r = am.apply(
        OrderFilledEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            fill=None,
            venue_filled_quantity=1.0,
            venue_avg_price=50_000.0,
        )
    )
    o = state.get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0  # 0.3 booked + 0.7 reconciled
    assert state.get_position(inst).quantity == 1.0  # full position, not 0.3
    assert r.deal is not None and abs(r.deal.amount - 0.7) < 1e-9  # the synthetic gap fill is surfaced
    assert r.deal.fee_amount is None  # a synthesized fill carries no fee


def test_filled_books_gap_on_top_of_a_delivered_last_deal():
    # FILLED carries the last real deal AND the venue cumulative still exceeds what we booked.
    am = _am()
    inst = _Inst()
    state = am.get_state("binance")
    add_order(state, status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
    am.apply(
        OrderPartiallyFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.3)
        )
    )
    am.apply(
        OrderFilledEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            fill=_fill(trade_id="t2", amount=0.2),
            venue_filled_quantity=1.0,
            venue_avg_price=50_000.0,
        )
    )
    o = state.get_order("cid-1")
    assert o.filled_quantity == 1.0  # 0.3 + 0.2 + 0.5 gap
    assert state.get_position(inst).quantity == 1.0


def test_filled_no_gap_when_cumulative_matches_booked():
    # When the venue cumulative equals the sum we booked, no synthetic fill is created.
    am = _am()
    inst = _Inst()
    state = am.get_state("binance")
    add_order(state, status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
    r = am.apply(
        OrderFilledEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            fill=_fill(trade_id="t1", amount=1.0),
            venue_filled_quantity=1.0,
            venue_avg_price=50_000.0,
        )
    )
    assert state.get_order("cid-1").filled_quantity == 1.0
    assert state.get_position(inst).quantity == 1.0
    assert r.deal.trade_id == "t1"  # the real delivered deal, not a synthetic gap


def test_filled_without_venue_cumulative_is_unchanged():
    # Sim/backtest and split-stream venues don't set venue_filled_quantity → no gap booking;
    # behaviour is identical to before (only the delivered fill books).
    am = _am()
    inst = _Inst()
    state = am.get_state("binance")
    add_order(state, status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.3)
        )
    )
    assert state.get_order("cid-1").filled_quantity == 0.3
    assert state.get_position(inst).quantity == 0.3


def test_gap_fill_suppressed_when_snapshot_deficit_already_covers_it():
    # If a snapshot already counted the missing fills (armed a fill deficit + corrected size),
    # the terminal gap reconciliation must NOT double-book: routed through _apply_execution,
    # the synthetic fill is suppressed up to the deficit.
    am = _am()
    inst = _Inst()
    state = am.get_state("binance")
    add_order(state, status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
    am.apply(
        OrderPartiallyFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.3)
        )
    )
    state.set_snapshot_fill_deficit("cid-1", 0.7)  # a snapshot already counted the remaining 0.7
    am.apply(
        OrderFilledEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            fill=None,
            venue_filled_quantity=1.0,
            venue_avg_price=50_000.0,
        )
    )
    assert state.get_position(inst).quantity == 0.3  # suppressed: size came from the snapshot, not re-booked
    assert state.get_snapshot_fill_deficit("cid-1") == 0.0  # deficit consumed
