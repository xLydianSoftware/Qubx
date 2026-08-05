"""AccountManager routing/aggregation acceptance tests ported from PR #302's manager_test.py.

Adapted to this branch's API: AccountManager takes kw-only connectors/base_currencies/time
(no pm -> no periodic ticks), events live in qubx.core.events with an explicit instrument
kwarg, mutators are unprefixed, and Order/event fields are client_order_id/venue_order_id.
"""

from typing import TypeVar
from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.account_manager.manager import AccountManager
from qubx.core.account_manager.reconciler import AbandonReplace, ResolveMissingOrder, SubmitReplacement
from qubx.core.account_manager.state import AccountState, ReplaceIntent
from qubx.core.basics import (
    Balance,
    Deal,
    ITimeProvider,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
)
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
)
from qubx.core.lookups import lookup
from qubx.core.series import Quote

T0 = np.datetime64("2026-05-28T00:00:00", "ns")
T1 = np.datetime64("2026-05-28T00:01:00", "ns")
NOW = np.datetime64("2026-05-28T00:02:00", "ns")

_btc = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
assert _btc is not None
BTC = _btc
EX = BTC.exchange

_T = TypeVar("_T")


def _present(value: _T | None) -> _T:
    assert value is not None
    return value


class _Time(ITimeProvider):
    def time(self) -> np.datetime64:
        return T1


def _am(*exchanges: str) -> AccountManager:
    exs = exchanges or (EX,)
    return AccountManager(
        connectors={ex: MagicMock() for ex in exs},
        base_currencies={ex: "USDT" for ex in exs},
        time=_Time(),
    )


def _order(cid: str = "c1", status: OrderStatus = OrderStatus.SUBMITTED, venue_id=None) -> Order:
    return Order(
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=BTC,
        quantity=1.0,
        side=OrderSide.BUY,
        time_in_force="gtc",
        status=status,
        venue_order_id=venue_id,
        price=100.0,
        last_update_time=T0 if status.is_terminal else None,
        origin=OrderOrigin.FRAMEWORK,
    )


def _fill(trade_id: str = "t1", amount: float = 0.5, price: float = 50_000.0) -> Deal:
    return Deal(trade_id=trade_id, order_id="v1", time=T0, amount=amount, price=price, aggressive=True)


def _seed(state: AccountState, amount: float) -> None:
    state.update_balance(
        "USDT", Balance(exchange=state.exchange, currency="USDT", free=amount, locked=0.0, total=amount)
    )


def _quote(bid: float, ask: float) -> Quote:
    return Quote(T1, bid, ask, 1.0, 1.0)


def test_apply_routes_by_instrument_and_delegates_to_reducer():
    am = _am()
    am.get_state(EX).add_order(_order("c1", OrderStatus.SUBMITTED))
    r = am.apply(OrderAcceptedEvent(instrument=BTC, client_order_id="c1", venue_order_id="V1", accepted_at=T0))
    assert r.order is not None and r.order.status is OrderStatus.ACCEPTED
    assert r.order.last_update_time == T1  # the AM's clock, not the event timestamp


def test_apply_routes_by_order_id_across_exchanges():
    am = _am(EX, "OKX")
    am.get_state("OKX").add_order(_order("c1", OrderStatus.ACCEPTED))
    r = am.apply(OrderCanceledEvent(instrument=None, client_order_id="c1"))  # no instrument, 2 exchanges
    assert r.order is not None and r.order.status is OrderStatus.CANCELED
    assert _present(am.get_state("OKX").get_order("c1")).status is OrderStatus.CANCELED


def test_is_synced_false_until_initial_snapshot_then_true():
    # is_synced drives the startup readiness gate: it must be False until the first venue
    # AccountSnapshot is applied (state watermarked), then True — so on_start/on_fit see real state.
    am = _am()
    assert am.is_synced() is False  # no snapshot applied yet
    am.apply(AccountSnapshotEvent(instrument=None, snapshot=AccountSnapshot(exchange=EX, as_of=T1)))
    assert am.is_synced() is True


def test_is_synced_requires_every_managed_exchange():
    # multi-exchange: not synced until ALL managed exchanges have applied their first snapshot.
    am = _am("BINANCE.UM", "BINANCE.CM")
    am.apply(AccountSnapshotEvent(instrument=None, snapshot=AccountSnapshot(exchange="BINANCE.UM", as_of=T1)))
    assert am.is_synced() is False  # BINANCE.CM still un-synced
    am.apply(AccountSnapshotEvent(instrument=None, snapshot=AccountSnapshot(exchange="BINANCE.CM", as_of=T1)))
    assert am.is_synced() is True


def test_apply_unroutable_returns_empty():
    am = _am(EX, "OKX")  # 2 exchanges, no fallback
    r = am.apply(OrderCanceledEvent(instrument=None, client_order_id="unknown"))
    assert r.order is None


def test_apply_fill_books_position_via_manager():
    am = _am()
    am.get_state(EX).add_order(_order("c1", OrderStatus.ACCEPTED))
    r = am.apply(OrderPartiallyFilledEvent(instrument=BTC, client_order_id="c1", fill=_fill("t1", 0.5)))
    assert r.position is not None and r.position.quantity == 0.5
    pos = _present(am.get_position(BTC))
    assert pos.quantity == 0.5
    assert pos.position_avg_price == 50_000.0
    # second fill at a new price: quantity accumulates, avg price size-weights
    am.apply(OrderFilledEvent(instrument=BTC, client_order_id="c1", fill=_fill("t2", 0.5, 51_000.0)))
    assert pos.quantity == 1.0
    assert abs(pos.position_avg_price - 50_500.0) < 1e-6


def test_get_orders_aggregates_and_filters_terminal():
    am = _am(EX, "OKX")
    am.get_state(EX).add_order(_order("a", OrderStatus.ACCEPTED))
    am.get_state("OKX").add_order(_order("b", OrderStatus.ACCEPTED))
    am.get_state("OKX").add_order(_order("c", OrderStatus.FILLED))  # terminal, retained in state
    assert set(am.get_orders().keys()) == {"a", "b"}  # terminal 'c' excluded
    assert set(am.get_orders(exchange="OKX").keys()) == {"b"}


def test_total_capital_aggregates_across_exchanges():
    am = _am(EX, "OKX")
    _seed(am.get_state(EX), 1000.0)
    _seed(am.get_state("OKX"), 500.0)
    assert am.get_total_capital() == 1500.0
    assert am.get_total_capital(exchange=EX) == 1000.0
    assert am.get_available_margin() == 1500.0  # no positions -> no initial margin


def test_on_market_quote_marks_existing_position():
    am = _am()
    pos = am.get_state(EX).ensure_position(BTC)
    pos.update_position_by_deal(_fill("t1", 0.5, 50_000.0), 1.0)  # long 0.5 @ 50000
    am.on_market_quote(BTC, _quote(50_999.0, 51_001.0))  # mid 51000
    assert pos.last_update_price == 51_000.0
    assert pos.unrealized_pnl() == 500.0  # 0.5 * (51000 - 50000)
    assert abs(pos.market_value - 500.0) < 1e-6  # futures market value tracks unrealized pnl


def test_on_market_quote_noop_without_position():
    am = _am()
    am.on_market_quote(BTC, _quote(49_999.0, 50_001.0))
    assert am.get_state(EX).get_position(BTC) is None  # a quote alone never creates a position
    # diverges from PR #302: am.get_position never returns None for a known instrument —
    # it materializes an empty Position (IAccountViewer contract), which must read flat.
    assert _present(am.get_position(BTC)).quantity == 0.0


def test_on_market_quote_unknown_exchange_is_noop():
    am = _am("OTHER")  # BTC's exchange not present
    am.on_market_quote(BTC, _quote(1.0, 2.0))  # must not raise


def test_get_order_by_exchange_and_shortcuts():
    am = _am(EX, "OKX")
    am.get_state("OKX").add_order(_order("c1", OrderStatus.ACCEPTED))
    assert _present(am.get_order("c1", exchange="OKX")).status is OrderStatus.ACCEPTED  # direct
    assert am.get_order("c1", exchange=EX) is None  # wrong exchange
    assert _present(am.get_order("c1")).client_order_id == "c1"  # multi-exchange scan fallback
    assert am.get_order("nope") is None  # unknown cid -> None (scan miss)
    # single-exchange shortcut (no scan)
    one = _am()
    one.get_state(EX).add_order(_order("x", OrderStatus.ACCEPTED))
    assert _present(one.get_order("x")).client_order_id == "x"


# --------------------------------------------------------------------------- #
# F26 — venue push handling at the manager layer
# --------------------------------------------------------------------------- #


def test_balance_push_routes_by_balance_exchange():
    am = _am(EX, "OKX")
    push = Balance(exchange="OKX", currency="USDT", free=np.nan, locked=np.nan, total=500.0)
    r = am.apply(BalanceUpdateEvent(instrument=None, balance=push, as_of=T1))
    assert r.balance is not None
    assert _present(am.get_state("OKX").get_balance("USDT")).total == 500.0
    assert am.get_state(EX).get_balance("USDT") is None  # other exchange untouched


def test_balance_push_for_unmanaged_exchange_is_dropped():
    # Strict routing: even with a single state, a push stamped for an unmanaged
    # exchange must not fall back into it.
    am = _am()
    push = Balance(exchange="KRAKEN.F", currency="USDT", free=np.nan, locked=np.nan, total=500.0)
    r = am.apply(BalanceUpdateEvent(instrument=None, balance=push, as_of=T1))
    assert r.is_empty()
    assert am.get_state(EX).get_balance("USDT") is None


def test_execute_dispatches_reconciler_actions():
    # the action executor performs the I/O the Reconciler asks for: connector status/snapshot
    # calls and routing a synthesized event back through the processing manager.
    from qubx.core.account_manager.reconciler import RequestHistDeals, RequestSnapshot, RequestStatus, RouteEvent

    am = _am()
    am.set_processing_manager(MagicMock())
    state = am.get_state(EX)
    order = _order("c1", OrderStatus.ACCEPTED, venue_id="V1")
    state.add_order(order)
    conn = am._connectors[EX]
    routed = OrderCanceledEvent(instrument=BTC, client_order_id="c1")

    am._execute(
        state,
        [
            RequestStatus(cid="c1", venue_id="V1", instrument=BTC),
            RequestSnapshot(exchange=EX),
            RouteEvent(event=routed),
            RequestHistDeals(instrument=BTC, since=T0),
        ],
    )

    conn.request_order_status.assert_called_once_with(order)
    conn.request_snapshot.assert_called_once_with(include_orders=True)
    conn.request_hist_deals.assert_called_once_with(BTC, T0)
    am._pm.process_event.assert_called_once_with(routed)


def test_execute_request_status_for_unknown_order_is_noop():
    from qubx.core.account_manager.reconciler import RequestStatus

    am = _am()
    am._execute(am.get_state(EX), [RequestStatus(cid="nope", venue_id=None, instrument=BTC)])
    am._connectors[EX].request_order_status.assert_not_called()


# --------------------------------------------------------------------------- #
# Task 7 — _execute performs the same-cid replacement or the truthful abandon
# --------------------------------------------------------------------------- #


class _SyncPM:
    """Routes process_event straight back into AM.apply — mirrors ProcessingManager's
    _dispatch_account without pulling in the whole processing stack, so tests can drive
    the real re-entrant apply()->_execute()->process_event()->apply() chain."""

    def __init__(self, am: AccountManager):
        self._am = am

    def process_event(self, event) -> None:
        self._am.apply(event)


@pytest.fixture
def manager_with_order():
    """A manager with a real (non-mock) routing PM and one BUY 10, filled 3.5,
    PENDING_UPDATE order — mirrors the state trading.py's update_order leaves behind
    after arming a replace intent and sending the internal cancel."""
    am = _am()
    state = am.get_state(EX)
    order = _order(status=OrderStatus.PARTIALLY_FILLED)
    order.quantity = 10.0
    order.filled_quantity = 3.5
    state.add_order(order)
    am.transition_order(EX, order.client_order_id, OrderStatus.PENDING_UPDATE)
    am._pm = _SyncPM(am)
    connector = am._connectors[EX]
    return am, state, order, connector


def test_execute_submit_replacement_splices_and_submits(manager_with_order):
    mgr, state, order, connector = manager_with_order  # BUY 10, filled 3.5, PENDING_UPDATE
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))
    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    req = connector.submit_order.call_args.args[0]
    assert req.client_id == order.client_order_id  # same-cid splice
    assert req.quantity == pytest.approx(6.5)  # residual 7.0 - 0.5, signed BUY
    assert req.price == 1.01
    assert order.quantity == pytest.approx(3.5 + 6.5)  # invariant restored
    assert state.get_replace_intent(order.client_order_id) is None
    assert order.status is not None and not order.status.is_terminal


def test_execute_submit_replacement_sends_unsigned_quantity_for_sell(manager_with_order):
    # OrderRequest.quantity must stay UNSIGNED (side conveys direction, exactly like
    # ccxt's create_order(amount, side) and TradingManager.trade's size_adj) — a signed
    # negative amount for a SELL replacement would misdirect or be rejected at the venue.
    mgr, state, order, connector = manager_with_order
    order.side = OrderSide.SELL
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))
    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    req = connector.submit_order.call_args.args[0]
    assert req.quantity == pytest.approx(6.5)  # unsigned, never negative
    assert req.side == OrderSide.SELL


def test_execute_submit_replacement_restamps_reduce_only_and_post_only(manager_with_order):
    # Pin manager.py's options re-stamping: reduceOnly/post_only come from the order record
    # (the source of truth), not blindly copied — a stale "reduce_only" alias in order.options
    # must not survive, but unrelated options do.
    mgr, state, order, connector = manager_with_order
    order.reduce_only = True
    order.post_only = True
    order.options = {"reduce_only": False, "some_venue_key": "keep-me"}
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))
    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    req = connector.submit_order.call_args.args[0]
    assert req.options["reduceOnly"] is True
    assert req.options["post_only"] is True
    assert "reduce_only" not in req.options  # stale alias dropped, not the stale value
    assert req.options["some_venue_key"] == "keep-me"  # unrelated options carried forward


def test_execute_submit_replacement_submit_failure_surfaces_cancel(manager_with_order):
    # A synchronous connector.submit_order raise means nothing is live at the venue (the OLD
    # order's cancel already succeeded — precondition for this decision to fire), so the truth
    # is CANCELED, not a stuck PENDING_UPDATE. No OrderUpdatedEvent must be routed, and the
    # exception must not escape _execute.
    mgr, state, order, connector = manager_with_order
    connector.submit_order.side_effect = RuntimeError("boom")
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))

    mgr._execute(state, [SubmitReplacement(order.client_order_id)])  # must not raise

    assert state.get_replace_intent(order.client_order_id) is None  # cleared exactly once
    assert order.status is OrderStatus.CANCELED  # truth surfaced, not a false PENDING_UPDATE
    assert order.quantity == 10.0  # unspliced: the failed submit never routed UPDATED


def test_execute_abandon_replace_reports_canceled_truth(manager_with_order):
    mgr, state, order, connector = manager_with_order
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 3.0, 3.0, NOW, filled_at_cancel=9.9))
    mgr._execute(state, [AbandonReplace(order.client_order_id, "residual below min")])
    assert state.get_replace_intent(order.client_order_id) is None
    assert order.status is OrderStatus.CANCELED  # truth surfaced via routed event


def test_execute_submit_replacement_without_pm_still_submits_and_clears(manager_with_order):
    # No processing manager wired: submit + clear must still happen, only the route is skipped.
    mgr, state, order, connector = manager_with_order
    mgr._pm = None
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))
    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    connector.submit_order.assert_called_once()
    assert state.get_replace_intent(order.client_order_id) is None
    assert order.quantity == 10.0  # unspliced: no routed UPDATED without a pm


def test_execute_submit_replacement_resolves_pending_missing_order_watcher(manager_with_order):
    # MANDATORY EXTRA (Task 6 review note): a REST snapshot landing during the internal-cancel
    # window can spawn a ResolveMissingOrder watcher for the mid-replace cid. The routed
    # OrderUpdatedEvent must resolve it via rec.on_event's normal OrderIn fallthrough, or the
    # watcher would retry to a spurious OrderLostEvent later.
    mgr, state, order, connector = manager_with_order
    rec = mgr._reconcilers[EX]
    rec._spawn(ResolveMissingOrder(order, NOW, wait=np.timedelta64(2, "s"), max_retries=3))
    assert rec.active_keys() == {order.client_order_id}

    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=3.5))
    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    assert rec.active_keys() == set()  # watcher resolved — no spurious retries/LOST


def test_apply_order_canceled_with_armed_intent_submits_replacement_end_to_end(manager_with_order):
    # End-to-end: drive apply() with a real suppressed-cancel event and let the reducer ->
    # reconciler -> _execute chain run for real (no direct _execute call).
    mgr, state, order, connector = manager_with_order
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(6.5, 1.02, 3.5, NOW))

    mgr.apply(OrderCanceledEvent(instrument=None, client_order_id=order.client_order_id, venue_order_id=None))

    req = connector.submit_order.call_args.args[0]
    assert req.client_id == order.client_order_id
    assert req.quantity == pytest.approx(6.5)
    # splice happens via the routed OrderUpdatedEvent hitting _handle_updated
    assert order.quantity == pytest.approx(3.5 + 6.5)
    assert order.status is not None and not order.status.is_terminal
    assert state.get_replace_intent(order.client_order_id) is None


def test_duplicate_cancel_terminal_does_not_terminalize_live_replacement(manager_with_order):
    # The ccxt connector emits TWO cancel terminals per cancel (REST ack + WS stream). The first
    # is consumed by the intent's suppression; the second resolves by cid onto an order whose
    # venue id has not re-keyed yet, so the superseded-oid guard alone cannot see it is stale —
    # without the per-cid superseded-oid marker it terminalizes an order that is LIVE at the venue.
    mgr, state, order, connector = manager_with_order
    cid = order.client_order_id
    state.set_venue_id(cid, "v_old")
    state.arm_replace_intent(cid, ReplaceIntent(6.5, 1.02, 3.5, NOW))
    cancel = OrderCanceledEvent(instrument=None, client_order_id=cid, venue_order_id="v_old", venue_filled_quantity=3.5)

    mgr.apply(cancel)  # REST ack: suppressed -> replacement submitted under the same cid
    connector.submit_order.assert_called_once()
    assert not _present(order.status).is_terminal

    mgr.apply(cancel)  # WS duplicate of the SAME terminal, replacement's accept not landed yet
    assert not _present(order.status).is_terminal
    connector.submit_order.assert_called_once()  # no second submit either

    # the replacement's accept lands under a NEW venue id -> re-key
    mgr.apply(OrderAcceptedEvent(instrument=None, client_order_id=cid, venue_order_id="v_new", accepted_at=T1))
    assert order.venue_order_id == "v_new"

    mgr.apply(cancel)  # third stale cancel for the superseded id
    assert not _present(order.status).is_terminal


def test_strategy_cancel_during_replace_window_terminalizes_truthfully(manager_with_order):
    # cancel_order on a PENDING_UPDATE order is legal (-> PENDING_CANCEL). The suppression must
    # key off the status, not merely the armed intent: otherwise the strategy's explicit cancel
    # is swallowed and the replacement resurrects an order the strategy killed.
    mgr, state, order, connector = manager_with_order
    cid = order.client_order_id
    state.arm_replace_intent(cid, ReplaceIntent(6.5, 1.02, 3.5, NOW))
    mgr.transition_order(EX, cid, OrderStatus.PENDING_CANCEL)  # strategy cancel during the window

    mgr.apply(OrderCanceledEvent(instrument=None, client_order_id=cid, venue_filled_quantity=3.5))

    assert order.status is OrderStatus.CANCELED  # truth, not a suppressed resurrection
    assert state.get_replace_intent(cid) is None
    connector.submit_order.assert_not_called()


def test_execute_submit_replacement_routes_lag_compensated_quantity(manager_with_order):
    # The venue's cancel ack counted fills whose WS deals have not booked locally yet. The
    # submit is the residual, but the ROUTED quantity must compensate for the local lag so the
    # reducer's splice lands on filled_at_cancel + residual — otherwise remaining stays
    # permanently understated once the raced deal books.
    mgr, state, order, connector = manager_with_order  # local filled 3.5
    state.arm_replace_intent(order.client_order_id, ReplaceIntent(7.0, 1.01, 3.0, NOW, filled_at_cancel=4.5))

    mgr._execute(state, [SubmitReplacement(order.client_order_id)])

    req = connector.submit_order.call_args.args[0]
    assert req.quantity == pytest.approx(5.5)  # residual 7.0 - raced 1.5 — what the venue gets
    assert order.quantity == pytest.approx(4.5 + 5.5)  # splice lands on filled_at_cancel + residual


def test_reconcile_tick_expiry_reverts_the_stuck_pending_update_order(manager_with_order):
    # Clearing the intent alone leaves the order in PENDING_UPDATE with no event able to move
    # it (an accepted status reply refuses to wipe pending, _reconcile_order skips pending), so
    # every later update_order silently no-ops. The sweep must revert it to venue-recognisable
    # truth: alive under the old terms.
    mgr, state, order, connector = manager_with_order
    cid = order.client_order_id
    state.arm_replace_intent(cid, ReplaceIntent(6.5, 1.02, 3.5, T0))  # armed 60s before the AM clock

    mgr._execute(state, mgr._reconcilers[EX].on_tick(state, T1))

    assert state.get_replace_intent(cid) is None
    assert order.status is OrderStatus.PARTIALLY_FILLED  # pre-pending truth, not stuck pending
    connector.request_order_status.assert_called_once()
