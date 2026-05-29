import numpy as np

from qubx.core.account_manager import AccountManager
from qubx.core.account_state import AccountState
from qubx.core.basics import Deal, Instrument, MarketType, Order, OrderOrigin, OrderStatus
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


class _T:
    def now(self):
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
    am._states = {"binance": AccountState(exchange="binance")}
    am._time = _T()
    am._applied_funding_buckets = {}
    return am


def _add_order(state, cid="cid-1", status=OrderStatus.SUBMITTED, instrument=None, quantity=1.0):
    state._add_order(
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


def test_apply_event_for_unknown_exchange_returns_none():
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
    assert result is None


def test_accepted_sets_venue_and_transitions():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], instrument=inst)
    am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    order = am._states["binance"].get_order("cid-1")
    assert order.status is OrderStatus.ACCEPTED
    assert order.venue_order_id == "V1"
    assert order.accepted_at == np.datetime64("2026-05-28")


def test_accepted_during_pending_cancel_is_side_effect_only():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PENDING_CANCEL
    assert o.venue_order_id == "V1"


def test_accepted_during_pending_update_transitions_to_accepted():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V2",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.venue_order_id == "V2"


def test_accepted_on_terminal_order_sets_venue_without_transition():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    # late OrderAccepted on an already-filled order: venue id only, no transition
    am.apply(
        OrderAcceptedEvent(
            instrument=inst,
            client_order_id="cid-1",
            venue_order_id="V1",
            accepted_at=np.datetime64("2026-05-28"),
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.FILLED


def test_happy_path_accepted_partial_filled():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], instrument=inst)
    am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    am.apply(
        OrderPartiallyFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=0.5))
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert o.filled_quantity == 0.5
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t2", amount=0.5)
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0


def test_fill_dedup_by_trade_id():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    evt = OrderPartiallyFilledEvent(
        instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.5)
    )
    am.apply(evt)
    am.apply(evt)
    assert am._states["binance"].get_order("cid-1").filled_quantity == 0.5


def test_partial_fill_during_pending_cancel_applies_without_transition():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(
        OrderPartiallyFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=0.3))
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PENDING_CANCEL
    assert o.filled_quantity == 0.3


def test_pending_update_overfill_keeps_filled_intact_and_warns():
    from qubx import logger

    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst, quantity=1.0)
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

    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PENDING_UPDATE
    assert o.filled_quantity == 1.5  # left intact — NOT clamped to quantity
    assert any("leaving filled intact" in m for m in messages)


def test_canceled_transitions_to_canceled():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderCanceledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    assert am._states["binance"].get_order("cid-1").status is OrderStatus.CANCELED


def test_expired_transitions_to_expired():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderExpiredEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    assert am._states["binance"].get_order("cid-1").status is OrderStatus.EXPIRED


def test_rejected_transitions_and_stores_reason():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.SUBMITTED, instrument=inst)
    am.apply(OrderRejectedEvent(instrument=inst, client_order_id="cid-1", reason="insufficient funds"))
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.REJECTED
    assert o.rejected_reason == "insufficient funds"


def test_rejected_for_unknown_order_returns_none():
    am = _am()
    inst = _Inst()
    result = am.apply(OrderRejectedEvent(instrument=inst, client_order_id="nope", reason="x"))
    assert result is None


def test_updated_in_place_modifies_fields_no_transition():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am._states["binance"]._set_venue_id("cid-1", "V1")
    am.apply(
        OrderUpdatedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", new_price=49_000.0, new_quantity=2.0
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.price == 49_000.0
    assert o.quantity == 2.0


def test_updated_during_pending_update_transitions_and_reindexes_venue():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am._states["binance"]._set_venue_id("cid-1", "V1")
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    am.apply(
        OrderUpdatedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V2", new_price=48_000.0, new_quantity=None
        )
    )
    state = am._states["binance"]
    o = state.get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.venue_order_id == "V2"
    assert o.price == 48_000.0
    assert state.get_order_by_venue_id("V2").client_order_id == "cid-1"
    assert state.get_order_by_venue_id("V1") is None


def test_cancel_rejected_reverts_to_pre_pending_status():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="too late"))
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.pre_pending_status is None


def test_cancel_rejected_reverts_to_partially_filled():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.PARTIALLY_FILLED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_CANCEL)
    am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="too late"))
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert o.pre_pending_status is None


def test_cancel_rejected_unexpected_state_returns_none():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    result = am.apply(OrderCancelRejectedEvent(instrument=inst, client_order_id="cid-1", reason="x"))
    assert result is None
    assert am._states["binance"].get_order("cid-1").status is OrderStatus.ACCEPTED


def test_update_rejected_reverts_to_pre_pending_status():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.transition_order("binance", "cid-1", OrderStatus.PENDING_UPDATE)
    am.apply(OrderUpdateRejectedEvent(instrument=inst, client_order_id="cid-1", reason="bad price"))
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.ACCEPTED
    assert o.pre_pending_status is None


def test_materialize_external_for_unknown_cid_and_venue():
    am = _am()
    inst = _Inst()
    am.apply(OrderPartiallyFilledEvent(instrument=inst, client_order_id="alien", venue_order_id="VX", fill=_fill()))
    state = am._states["binance"]
    o = state.get_order_by_venue_id("VX")
    assert o is not None
    assert o.origin is OrderOrigin.EXTERNAL
    assert o.client_order_id == "ext:VX"
    assert o.filled_quantity == 0.5


def test_fill_before_accept_submitted_to_filled():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.SUBMITTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0
    assert o.venue_order_id == "V1"


def test_resolve_via_terminal_history_fallback():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am._states["binance"]._set_venue_id("cid-1", "V1")
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    # evict the terminal order to history, then a late accept must resolve there,
    # not materialize a phantom EXTERNAL order.
    am._states["binance"]._evict_to_history("cid-1")
    am.apply(
        OrderAcceptedEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", accepted_at=np.datetime64("2026-05-28")
        )
    )
    state = am._states["binance"]
    # no phantom EXTERNAL order materialized
    assert "ext:V1" not in state.active_orders
    assert state.get_order("cid-1").status is OrderStatus.FILLED


def test_resolve_via_venue_id_index():
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am._states["binance"]._set_venue_id("cid-1", "V1")
    # event carries no client_order_id match (different cid) but a known venue id
    am.apply(
        OrderPartiallyFilledEvent(instrument=inst, client_order_id="other", venue_order_id="V1", fill=_fill(amount=0.4))
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.filled_quantity == 0.4
    assert "ext:V1" not in am._states["binance"].active_orders


def test_late_cancel_on_filled_order_is_noop():
    # C1: a late OrderCanceled on an already-FILLED order (still in active_orders
    # during the grace window) must NOT flip FILLED -> CANCELED.
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    assert am._states["binance"].get_order("cid-1").status is OrderStatus.FILLED
    # late cancel: benign no-op, no exception, status unchanged
    am.apply(OrderCanceledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1"))
    assert am._states["binance"].get_order("cid-1").status is OrderStatus.FILLED


def test_late_fill_on_filled_order_is_noop():
    # A second OrderFilled (new trade_id) on an already-FILLED order is ignored:
    # status stays FILLED and filled_quantity is unchanged.
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=1.0)
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0
    # late fill with a brand-new trade_id: ignored, no double-count
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t2", amount=0.5)
        )
    )
    o = am._states["binance"].get_order("cid-1")
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 1.0


def test_late_fill_on_evicted_order_does_not_raise():
    # C2: once an order is evicted to _terminal_history it is no longer in
    # active_orders, so any mutator doing active_orders[cid] would KeyError.
    # Late fill/cancel/update events resolving to it via cid must be benign
    # no-ops with no phantom EXTERNAL order in active_orders.
    am = _am()
    inst = _Inst()
    _add_order(am._states["binance"], status=OrderStatus.ACCEPTED, instrument=inst)
    am._states["binance"]._set_venue_id("cid-1", "V1")
    am.apply(OrderFilledEvent(instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=1.0)))
    am._states["binance"]._evict_to_history("cid-1")
    assert "cid-1" not in am._states["binance"].active_orders

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

    state = am._states["binance"]
    assert "cid-1" not in state.active_orders
    assert "ext:V1" not in state.active_orders
    assert "ext:cid-1" not in state.active_orders
    # the evicted order remains FILLED and untouched
    assert state.get_order("cid-1").status is OrderStatus.FILLED
