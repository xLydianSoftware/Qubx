"""Reducer acceptance tests ported from PR #302's reducer_test.py.

Adapted to this branch's API: events live in qubx.core.events (no timestamp field on
ChannelMessage, instrument is an explicit kwarg), Order/event fields are
client_order_id/venue_order_id, state mutators are unprefixed, and reducer.apply takes
a kw-only snapshot_grace (threaded once via the local apply wrapper below).
"""

from typing import TypeVar

import numpy as np

from qubx.core.account_manager import reducer
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import (
    Balance,
    Deal,
    FundingPayment,
    Instrument,
    MarketType,
    Order,
    OrderChange,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.events import (
    BalanceUpdateEvent,
    DealEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
    PositionUpdateEvent,
)
from qubx.core.lookups import lookup

T0 = np.datetime64("2026-05-28T00:00:00", "ns")
T1 = np.datetime64("2026-05-28T00:01:00", "ns")

SNAPSHOT_GRACE = np.timedelta64(60_000, "ms")

_T = TypeVar("_T")


def _present(value: _T | None) -> _T:
    assert value is not None
    return value


def apply(state: AccountState, event, now: np.datetime64) -> reducer.ApplyResult:
    # our reducer.apply takes a kw-only snapshot_grace; irrelevant for order events
    return reducer.apply(state, event, now, snapshot_grace=SNAPSHOT_GRACE)


_btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
assert _btc is not None
BTC: Instrument = _btc


def _state() -> AccountState:
    return AccountState("binance", "USDT")


def _fill(trade_id: str = "t1", amount: float = 0.5, price: float = 100.0) -> Deal:
    return Deal(trade_id=trade_id, order_id="v1", time=T0, amount=amount, price=price, aggressive=True)


def _order(state: AccountState, cid: str = "c1", status: OrderStatus = OrderStatus.SUBMITTED, venue_id=None) -> Order:
    order = Order(
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=BTC,
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_order_id=venue_id,
        price=100.0,
        last_updated_at=T0 if status.is_terminal else None,
        origin=OrderOrigin.FRAMEWORK,
    )
    state.add_order(order)
    return order


def test_accept_transitions_and_sets_venue_id():
    state = _state()
    _order(state)
    r = apply(state, OrderAcceptedEvent(instrument=None, client_order_id="c1", venue_order_id="V1", accepted_at=T0), T1)
    assert r.order is not None
    assert r.order.status is OrderStatus.ACCEPTED
    assert r.order.venue_order_id == "V1"
    assert r.order.last_updated_at == T1
    assert r.order_change is OrderChange.ACCEPTED
    assert r.deal is None and r.position is None


def test_accept_on_terminal_is_noop():
    state = _state()
    _order(state, status=OrderStatus.FILLED)
    r = apply(state, OrderAcceptedEvent(instrument=None, client_order_id="c1", accepted_at=T0), T1)
    assert r.order is None


def test_accept_unknown_order_is_noop():
    r = apply(_state(), OrderAcceptedEvent(instrument=None, client_order_id="nope", accepted_at=T0), T1)
    assert r.order is None


def test_cancel_transitions_to_terminal():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderCanceledEvent(instrument=None, client_order_id="c1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.CANCELED
    assert r.order_change is OrderChange.CANCELED


def test_cancel_on_terminal_is_noop():
    state = _state()
    _order(state, status=OrderStatus.FILLED)
    r = apply(state, OrderCanceledEvent(instrument=None, client_order_id="c1"), T1)
    assert r.order is None


def test_expire_transitions_to_terminal():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderExpiredEvent(instrument=None, client_order_id="c1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.EXPIRED
    assert r.order_change is OrderChange.EXPIRED


def test_reject_transitions_and_records_reason():
    state = _state()
    _order(state)
    r = apply(state, OrderRejectedEvent(instrument=None, client_order_id="c1", reason="insufficient margin"), T1)
    assert r.order is not None
    assert r.order.status is OrderStatus.REJECTED
    assert r.order.rejected_reason == "insufficient margin"
    assert r.order_change is OrderChange.REJECTED


def test_reject_unknown_order_is_noop():
    r = apply(_state(), OrderRejectedEvent(instrument=None, client_order_id="nope", reason="x"), T1)
    assert r.order is None


def test_reject_resolves_by_venue_id():
    # client_order_id=None: venue-only addressing — the event resolves via the venue index.
    state = _state()
    _order(state, cid="c1", status=OrderStatus.ACCEPTED, venue_id="V1")
    r = apply(state, OrderRejectedEvent(instrument=None, client_order_id=None, venue_order_id="V1", reason="x"), T1)
    assert r.order is not None and r.order.client_order_id == "c1"
    assert r.order.status is OrderStatus.REJECTED


def test_fill_with_embedded_deal():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.5)), T1)
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.order_change is OrderChange.FILLED
    assert _present(r.deal).trade_id == "t1"
    assert r.order.filled_quantity == 0.5


def test_fill_without_deal_split_stream():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderFilledEvent(instrument=None, client_order_id="c1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.deal is None
    assert r.order.filled_quantity == 0.0  # deal arrives separately via DealEvent


def test_fill_duplicate_deal_not_double_counted():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.5)
    state.apply_fill("c1", f, T0)  # already applied (e.g. earlier DealEvent)
    r = apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=f), T1)
    assert r.order is not None and r.order.status is OrderStatus.FILLED
    assert r.deal is None  # deduped -> no on_execution
    assert r.order.filled_quantity == 0.5


def test_partial_fill_first_transitions():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderPartiallyFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.3)), T1)
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED
    assert r.order_change is OrderChange.PARTIALLY_FILLED
    assert _present(r.deal).trade_id == "t1"
    assert r.order.filled_quantity == 0.3


def test_partial_fill_before_accept():
    state = _state()
    _order(state, status=OrderStatus.SUBMITTED)
    r = apply(state, OrderPartiallyFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.3)), T1)
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED


def test_subsequent_partial_fill_is_execution_only():
    state = _state()
    _order(state, status=OrderStatus.PARTIALLY_FILLED)
    r = apply(state, OrderPartiallyFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t2", 0.2)), T1)
    assert r.order is None  # no status transition
    assert _present(r.deal).trade_id == "t2"


def test_partial_fill_while_pending_is_execution_only():
    state = _state()
    _order(state, status=OrderStatus.PENDING_CANCEL)
    r = apply(state, OrderPartiallyFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.2)), T1)
    assert r.order is None  # pending status not disturbed
    assert _present(r.deal).trade_id == "t1"
    assert _present(state.get_active_order("c1")).status is OrderStatus.PENDING_CANCEL


def test_deal_event_applies_and_reports_execution_only():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, DealEvent(instrument=None, client_order_id="c1", deal=_fill("t1", 0.4)), T1)
    assert r.order is None  # deal event never changes status
    assert _present(r.deal).trade_id == "t1"
    assert _present(state.get_active_order("c1")).filled_quantity == 0.4


def test_deal_event_with_no_cid_resolves_by_venue_id():
    # Split-stream trade arriving before the connector's cid index is seeded: cid=None,
    # venue id present — resolves via the venue index, never materializes an external.
    state = _state()
    _order(state, cid="c1", status=OrderStatus.ACCEPTED, venue_id="V1")
    r = apply(state, DealEvent(instrument=BTC, client_order_id=None, venue_order_id="V1", deal=_fill("t1", 0.4)), T1)
    assert _present(r.deal).trade_id == "t1"
    assert _present(state.get_active_order("c1")).filled_quantity == 0.4
    assert state.get_order("ext:V1") is None


def test_deal_event_duplicate_is_noop():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.4)
    apply(state, DealEvent(instrument=None, client_order_id="c1", deal=f), T1)
    r = apply(state, DealEvent(instrument=None, client_order_id="c1", deal=f), T1)
    assert r.order is None and r.deal is None


def test_dedup_across_deal_and_order_stream():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.5)
    r1 = apply(state, DealEvent(instrument=None, client_order_id="c1", deal=f), T1)
    assert _present(r1.deal).trade_id == "t1"
    r2 = apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=f), T1)
    assert r2.order is not None and r2.order.status is OrderStatus.FILLED
    assert r2.deal is None  # already applied via DealEvent
    assert r2.order.filled_quantity == 0.5


# --------------------------------------------------------------------------- #
# F26 — venue position/balance pushes
# --------------------------------------------------------------------------- #


def _venue_position(qty: float, price: float = 50_000.0) -> Position:
    return Position(BTC, quantity=qty, pos_average_price=price)


def _total_push(total: float, currency: str = "USDT") -> Balance:
    # futures pushes carry no free/locked split — total only (NaN split by contract)
    return Balance(exchange="binance", currency=currency, free=np.nan, locked=np.nan, total=total)


def test_position_push_size_equal_returns_local_position():
    # Size-equal push carries the LOCAL position (never the venue payload), so the
    # purely field-driven dispatch fires on_position_change off local state.
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.5)), T1)
    venue = _venue_position(0.5)
    r = apply(state, PositionUpdateEvent(instrument=BTC, position=venue, as_of=T1), T1)
    assert r.position is _present(state.get_position(BTC))
    assert r.position is not venue
    assert r.position_drift is None
    assert state.get_position_push_as_of(BTC) == T1
    assert _present(state.get_position(BTC)).quantity == 0.5


def test_position_push_flat_on_empty_state_creates_nothing():
    state = _state()
    r = apply(state, PositionUpdateEvent(instrument=BTC, position=_venue_position(0.0), as_of=T1), T1)
    assert r.is_empty()
    assert state.get_position(BTC) is None
    assert state.get_position_push_as_of(BTC) == T1


def test_position_push_drift_flags_without_mutation():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.5, 50_000.0)), T1)
    venue = _venue_position(0.7, 49_000.0)
    r = apply(state, PositionUpdateEvent(instrument=BTC, position=venue, as_of=T1), T1)
    assert r.position_drift is venue
    assert r.position is None and not r.is_empty()
    local = _present(state.get_position(BTC))
    assert local.quantity == 0.5  # zero mutation — the snapshot reconcile corrects size
    assert local.position_avg_price == 50_000.0


def test_position_push_stale_as_of_suppressed():
    # Ratchet-dropped pushes return empty -> NOTHING fires (no fire-through off the
    # event payload — that would deliver data older than already delivered).
    state = _state()
    apply(state, PositionUpdateEvent(instrument=BTC, position=_venue_position(0.0), as_of=T1), T1)
    # older AND same-time pushes are stale (<= ratchet), even when they would drift
    r = apply(state, PositionUpdateEvent(instrument=BTC, position=_venue_position(0.7), as_of=T0), T1)
    assert r.is_empty()
    r = apply(state, PositionUpdateEvent(instrument=BTC, position=_venue_position(0.7), as_of=T1), T1)
    assert r.is_empty()
    assert state.get_position_push_as_of(BTC) == T1


def test_balance_push_applies_absolutely_preserving_locked():
    state = _state()
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", free=80.0, locked=20.0, total=100.0))
    r = apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(150.0), as_of=T1), T1)
    bal = _present(state.get_balance("USDT"))
    assert r.balance is bal  # internal visibility only — fires NO strategy callback
    assert (bal.total, bal.free, bal.locked) == (150.0, 130.0, 20.0)


def test_balance_push_stale_ratchet_returns_empty():
    state = _state()
    apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(150.0), as_of=T1), T1)
    r = apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(90.0), as_of=T0), T1)
    assert r.is_empty()
    assert _present(state.get_balance("USDT")).total == 150.0


def test_deal_covered_by_balance_push_books_position_but_skips_cash_leg():
    # A balance push at/after the deal's venue time already carries the deal's cash
    # effect: position/r_pnl still book, the settle adjust_balance leg is skipped —
    # correct under both [deal, push] and [push, deal] orderings.
    state = _state()
    _order(state, cid="c1", status=OrderStatus.ACCEPTED)
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.5, 50_000.0)), T0)
    apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(1010.0), as_of=T1), T1)
    # closing deal at T0 (<= push as_of T1): pnl +12, fee 2 — already in the pushed total
    closing = Deal(trade_id="t2", order_id="v2", time=T0, amount=-0.5, price=50_024.0, aggressive=True, fee_amount=2.0)
    _order(state, cid="c2", status=OrderStatus.ACCEPTED)
    r = apply(state, OrderFilledEvent(instrument=None, client_order_id="c2", fill=closing), T1)
    pos = _present(r.position)
    assert pos.quantity == 0.0
    assert pos.r_pnl == 12.0
    assert _present(state.get_balance("USDT")).total == 1010.0  # the push figure stands


def test_deal_not_covered_by_older_push_still_adjusts_balance():
    state = _state()
    apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(1000.0), as_of=T0), T0)
    _order(state, status=OrderStatus.ACCEPTED)
    deal = Deal(trade_id="t1", order_id="v1", time=T1, amount=0.5, price=50_000.0, aggressive=True, fee_amount=2.0)
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=deal), T1)
    assert _present(state.get_balance("USDT")).total == 998.0  # 1000 - fee


def test_spot_deal_legs_covered_independently():
    # Spot books two legs (quote debit, base credit); each is guarded against its OWN
    # currency's push as_of.
    spot = Instrument(
        symbol="ETHUSDT",
        market_type=MarketType.SPOT,
        exchange="binance",
        base="ETH",
        quote="USDT",
        settle="USDT",
        exchange_symbol="ETHUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )
    state = _state()
    _seed_usdt(state, 1000.0)
    apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(900.0), as_of=T1), T1)
    deal = Deal(trade_id="t1", order_id="v1", time=T0, amount=0.1, price=1_000.0, aggressive=True)
    apply(state, OrderFilledEvent(instrument=spot, client_order_id="x", venue_order_id="E1", fill=deal), T1)
    assert _present(state.get_balance("USDT")).total == 900.0  # quote leg covered by the push
    assert _present(state.get_balance("ETH")).total == 0.1  # base leg uncovered -> still credited


def test_funding_covered_by_balance_push_skips_cash_leg():
    state = _state()
    pos = Position(BTC, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(T0, 50_000.0, 1.0)
    state.set_position(BTC, pos)
    _seed_usdt(state, 1000.0)
    apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(995.0), as_of=T1), T1)
    # funding venue time T0 <= push as_of T1: the venue's FUNDING_FEE debit is in the total
    payment = FundingPayment(time=int(T0.astype(np.int64)), funding_rate=0.0001, funding_interval_hours=8)
    r = apply(state, FundingPaymentEvent(instrument=BTC, payment=payment), T1)
    assert r.position is pos
    expected = -(1.0 * 50_000.0 * 0.0001)
    assert abs(pos.cumulative_funding - expected) < 1e-9  # funding still books on the position
    assert _present(state.get_balance("USDT")).total == 995.0  # cash leg skipped


def test_funding_not_covered_by_older_push_still_adjusts_balance():
    state = _state()
    pos = Position(BTC, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(T0, 50_000.0, 1.0)
    state.set_position(BTC, pos)
    _seed_usdt(state, 1000.0)
    apply(state, BalanceUpdateEvent(instrument=None, balance=_total_push(1000.0), as_of=T0), T0)
    payment = FundingPayment(time=int(T1.astype(np.int64)), funding_rate=0.0001, funding_interval_hours=8)
    apply(state, FundingPaymentEvent(instrument=BTC, payment=payment), T1)
    assert abs(_present(state.get_balance("USDT")).total - 995.0) < 1e-9  # 1000 - 5 funding paid


# --------------------------------------------------------------------------- #
# 4.3 — deal -> position / balance ledger
# --------------------------------------------------------------------------- #


def _seed_usdt(state: AccountState, amount: float = 1000.0) -> None:
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", free=amount, locked=0.0, total=amount))


def test_fill_books_position_and_sets_result_position():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.5, 50_000.0)), T1)
    assert r.position is not None
    assert r.position.quantity == 0.5
    assert _present(state.get_position(BTC)).quantity == 0.5


def test_partial_fills_accumulate_position():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    apply(state, OrderPartiallyFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t1", 0.3, 50_000.0)), T1)
    apply(state, OrderPartiallyFilledEvent(instrument=None, client_order_id="c1", fill=_fill("t2", 0.2, 50_000.0)), T1)
    assert _present(state.get_position(BTC)).quantity == 0.5


def test_duplicate_deal_not_double_booked():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    f = _fill("t1", 0.5, 50_000.0)
    apply(state, DealEvent(instrument=None, client_order_id="c1", deal=f), T1)
    r = apply(state, DealEvent(instrument=None, client_order_id="c1", deal=f), T1)
    assert r.position is None  # deduped -> no second booking
    assert _present(state.get_position(BTC)).quantity == 0.5


def test_deal_event_books_position():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, DealEvent(instrument=None, client_order_id="c1", deal=_fill("t1", 0.4, 50_000.0)), T1)
    assert r.position is not None and r.position.quantity == 0.4


def test_futures_fee_debits_settle_balance():
    state = _state()
    _seed_usdt(state, 1000.0)
    _order(state, status=OrderStatus.ACCEPTED)
    deal = Deal(trade_id="t1", order_id="v1", time=T0, amount=0.5, price=50_000.0, aggressive=True, fee_amount=2.0)
    apply(state, OrderFilledEvent(instrument=None, client_order_id="c1", fill=deal), T1)
    # futures: settle (USDT) += realized_pnl - fee = 0 - 2
    assert _present(state.get_balance("USDT")).total == 998.0


# --------------------------------------------------------------------------- #
# 4.4 — update / cancel-rejected / update-rejected
# --------------------------------------------------------------------------- #


def test_update_applies_params_and_confirms_pending_update():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)  # captures pre_pending=ACCEPTED
    r = apply(state, OrderUpdatedEvent(instrument=None, client_order_id="c1", new_price=110.0, new_quantity=2.0), T1)
    assert r.order is not None
    assert r.order.status is OrderStatus.ACCEPTED  # reverted to pre-pending
    assert r.order.price == 110.0 and r.order.quantity == 2.0
    assert r.order_change is OrderChange.UPDATED
    assert state.get_pre_pending("c1") is None  # cleared on leaving pending


def test_update_confirms_back_to_partially_filled():
    state = _state()
    _order(state, status=OrderStatus.PARTIALLY_FILLED)
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)  # pre_pending = PARTIALLY_FILLED
    r = apply(state, OrderUpdatedEvent(instrument=None, client_order_id="c1", new_price=110.0, new_quantity=None), T1)
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED


def test_update_on_non_pending_applies_params_without_status_change():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    r = apply(state, OrderUpdatedEvent(instrument=None, client_order_id="c1", new_price=120.0, new_quantity=None), T1)
    assert r.order is not None
    assert r.order.status is OrderStatus.ACCEPTED  # unchanged
    assert r.order.price == 120.0
    assert r.order_change is OrderChange.UPDATED  # fires on_order despite no status change


def test_cancel_rejected_reverts_to_pre_pending():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)
    state.transition_order("c1", OrderStatus.PENDING_CANCEL, T0)
    r = apply(state, OrderCancelRejectedEvent(instrument=None, client_order_id="c1", reason="too late"), T1)
    assert r.order is not None and r.order.status is OrderStatus.ACCEPTED
    assert r.order_change is OrderChange.CANCEL_REJECTED


def test_cancel_rejected_wrong_state_is_noop():
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED)  # not PENDING_CANCEL
    r = apply(state, OrderCancelRejectedEvent(instrument=None, client_order_id="c1", reason="x"), T1)
    assert r.order is None


def test_update_rejected_reverts_to_pre_pending():
    state = _state()
    _order(state, status=OrderStatus.PARTIALLY_FILLED)
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)
    r = apply(state, OrderUpdateRejectedEvent(instrument=None, client_order_id="c1", reason="x"), T1)
    assert r.order is not None and r.order.status is OrderStatus.PARTIALLY_FILLED
    assert r.order_change is OrderChange.UPDATE_REJECTED


def test_update_rejected_then_canceled_lands_terminal():
    # The simulated connector's failed cancel+recreate emits UpdateRejected then Canceled
    # (the original is gone at the venue): the order must converge to terminal CANCELED,
    # not strand at the reverted pre-pending status.
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED, venue_id="V1")
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)
    r = apply(state, OrderUpdateRejectedEvent(instrument=None, client_order_id="c1", reason="would trigger"), T1)
    assert r.order is not None and r.order.status is OrderStatus.ACCEPTED
    r = apply(state, OrderCanceledEvent(instrument=None, client_order_id="c1", venue_order_id="V1"), T1)
    assert r.order is not None and r.order.status is OrderStatus.CANCELED
    assert r.order_change is OrderChange.CANCELED
    assert _present(state.get_active_order("c1")).status.is_terminal  # later cancels no-op on terminal


def test_accept_during_pending_update_preserves_marker():
    # The ccxt connector emits ACCEPTED twice by design (REST ack + WS ack): a duplicate
    # accept racing a pending update must not wipe PENDING_UPDATE, so the venue's later
    # verdict (here a rejection) still reverts via the preserved pre-pending capture.
    state = _state()
    _order(state, status=OrderStatus.ACCEPTED, venue_id="V1")
    state.transition_order("c1", OrderStatus.PENDING_UPDATE, T0)
    r = apply(state, OrderAcceptedEvent(instrument=None, client_order_id="c1", venue_order_id="V1", accepted_at=T0), T1)
    assert r.order is None  # suppressed — no callback fires
    assert _present(state.get_active_order("c1")).status is OrderStatus.PENDING_UPDATE
    r = apply(state, OrderUpdateRejectedEvent(instrument=None, client_order_id="c1", reason="below filled"), T1)
    assert r.order is not None and r.order.status is OrderStatus.ACCEPTED
    assert r.order_change is OrderChange.UPDATE_REJECTED


# --------------------------------------------------------------------------- #
# 4.5 — external-order materialization
# --------------------------------------------------------------------------- #


def test_external_fill_materializes_books_and_fires_all():
    state = _state()
    _seed_usdt(state, 1000.0)
    # order unknown to us -> external
    r = apply(
        state,
        OrderFilledEvent(instrument=BTC, client_order_id="x", venue_order_id="EXT1", fill=_fill("t1", 0.5, 50_000.0)),
        T1,
    )
    assert r.order is not None
    assert r.order.origin is OrderOrigin.EXTERNAL
    assert r.order.client_order_id == "ext:EXT1"
    assert r.order.status is OrderStatus.FILLED
    assert r.deal is not None
    assert r.position is not None and r.position.quantity == 0.5


def test_materialize_skipped_without_instrument():
    state = _state()
    r = apply(state, OrderFilledEvent(instrument=None, client_order_id="x", venue_order_id="EXT1", fill=_fill()), T1)
    assert r.order is None and r.deal is None and r.position is None


def test_external_order_resolves_same_on_second_event():
    state = _state()
    apply(state, DealEvent(instrument=BTC, client_order_id="x", venue_order_id="EXT1", deal=_fill("t1", 0.3)), T1)
    apply(state, DealEvent(instrument=BTC, client_order_id="x", venue_order_id="EXT1", deal=_fill("t2", 0.2)), T1)
    order = state.get_order_by_venue_id("EXT1")
    assert order is not None and order.client_order_id == "ext:EXT1"
    assert _present(state.get_position(BTC)).quantity == 0.5  # accumulated on one external order
