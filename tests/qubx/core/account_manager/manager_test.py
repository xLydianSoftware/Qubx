"""AccountManager routing/aggregation acceptance tests ported from PR #302's manager_test.py.

Adapted to this branch's API: AccountManager takes kw-only connectors/base_currencies/time
(no pm -> no periodic ticks), events live in qubx.core.events with an explicit instrument
kwarg, mutators are unprefixed, and Order/event fields are client_order_id/venue_order_id.
"""

from typing import TypeVar
from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager.manager import AccountManager
from qubx.core.account_manager.state import AccountState
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
    conn.request_snapshot.assert_called_once_with()
    conn.request_hist_deals.assert_called_once_with(BTC, T0)
    am._pm.process_event.assert_called_once_with(routed)


def test_execute_request_status_for_unknown_order_is_noop():
    from qubx.core.account_manager.reconciler import RequestStatus

    am = _am()
    am._execute(am.get_state(EX), [RequestStatus(cid="nope", venue_id=None, instrument=BTC)])
    am._connectors[EX].request_order_status.assert_not_called()
