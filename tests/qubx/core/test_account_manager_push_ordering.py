"""F26 ordering matrix — the load-bearing guarantee of WS push application.

Venue balance pushes are ABSOLUTE (post-change wallet totals) while deals/funding book
DELTAS, and Binance documents no reliable per-fill ordering of ORDER_TRADE_UPDATE vs
ACCOUNT_UPDATE. The covered-delta guard (skip only the adjust_balance leg when the
currency's push as_of is at/after the deal/funding venue time) must therefore make both
arrival orders converge to IDENTICAL qty / r_pnl / settle-currency balances. All clocks
compared here are venue event time (the Deal.time domain); the AM's local clock is set
to a deliberately different value to prove it never participates.
"""

from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager
from qubx.core.basics import (
    Balance,
    Deal,
    FundingPayment,
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.events import (
    AccountMessage,
    BalanceUpdateEvent,
    DealEvent,
    FundingPaymentEvent,
    OrderFilledEvent,
)

EX = "binance"

T_OPEN = np.datetime64("2026-05-28T00:00:00", "ns")  # opening fill venue time
T_EVENT = np.datetime64("2026-05-28T00:00:01", "ns")  # the matrix deal/funding venue time
T_PUSH = np.datetime64("2026-05-28T00:00:02", "ns")  # the covering push venue time (>= T_EVENT)

INST = Instrument(
    symbol="BTCUSDT",
    market_type=MarketType.SWAP,
    exchange=EX,
    base="BTC",
    quote="USDT",
    settle="USDT",
    exchange_symbol="BTCUSDT",
    tick_size=0.01,
    lot_size=0.001,
    min_size=0.001,
    contract_size=1.0,
)


class _T:
    def time(self):
        # AM local clock — a different domain (and value) than the venue stamps above
        return np.datetime64("2026-05-28T00:10:00", "ns")


def _am() -> AccountManager:
    am = AccountManager(connectors={EX: MagicMock()}, base_currencies={EX: "USDT"}, time=_T())
    am.seed_balance(EX, Balance(exchange=EX, currency="USDT", free=1000.0, locked=0.0, total=1000.0))
    return am


def _add_order(am: AccountManager, cid: str) -> None:
    am.add_order(
        Order(
            client_order_id=cid,
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type=OrderType.LIMIT,
            instrument=INST,
            submitted_at=T_OPEN,
            quantity=1.0,
            price=50_000.0,
            side=OrderSide.BUY,
            status=OrderStatus.ACCEPTED,
            time_in_force="gtc",
        )
    )


def _push(total: float, as_of: np.datetime64 = T_PUSH) -> BalanceUpdateEvent:
    # futures push: total-only, free/locked NaN by producer contract
    bal = Balance(exchange=EX, currency="USDT", free=np.nan, locked=np.nan, total=total)
    return BalanceUpdateEvent(instrument=None, balance=bal, as_of=as_of)


def _balances(am: AccountManager) -> tuple[float, float, float]:
    bal = am.get_state(EX).get_balance("USDT")
    assert bal is not None
    return (bal.total, bal.free, bal.locked)


# --------------------------------------------------------------------------- #
# [DealEvent, push] vs [push, DealEvent]
# --------------------------------------------------------------------------- #


def _closing_deal_event() -> DealEvent:
    # closes the 0.5 @ 50_000 long at 50_100: pnl +50, fee 1 -> cash leg +49
    deal = Deal(
        trade_id="t2", order_id="V2", time=T_EVENT, amount=-0.5, price=50_100.0, aggressive=True, fee_amount=1.0
    )
    return DealEvent(instrument=INST, client_order_id="c2", venue_order_id="V2", deal=deal)


def _run_fill_matrix(events: list[AccountMessage]) -> tuple[float, float, tuple[float, float, float]]:
    am = _am()
    _add_order(am, "c1")
    # opening fill, uncovered (no push seen yet): qty 0.5 @ 50_000, fee 1 -> 999 USDT
    opening = Deal(
        trade_id="t1", order_id="V1", time=T_OPEN, amount=0.5, price=50_000.0, aggressive=True, fee_amount=1.0
    )
    am.apply(OrderFilledEvent(instrument=INST, client_order_id="c1", venue_order_id="V1", fill=opening))
    _add_order(am, "c2")
    for event in events:
        am.apply(event)
    pos = am.get_state(EX).get_position(INST)
    assert pos is not None
    return pos.quantity, pos.r_pnl, _balances(am)


def test_same_fill_converges_under_both_event_orderings():
    # The venue's post-fill wallet total is 999 + 50 - 1 = 1048 either way.
    deal_first = _run_fill_matrix([_closing_deal_event(), _push(1048.0)])
    push_first = _run_fill_matrix([_push(1048.0), _closing_deal_event()])
    assert deal_first == push_first
    qty, r_pnl, (total, free, locked) = deal_first
    assert qty == 0.0
    assert r_pnl == 50.0
    assert (total, free, locked) == (1048.0, 1048.0, 0.0)


def test_deal_at_equal_venue_time_is_covered_by_push():
    # R51: the dominant Binance production case — the fill's ORDER_TRADE_UPDATE and the
    # wallet ACCOUNT_UPDATE share ONE transaction time. The covered-delta guard is `>=`:
    # with the push applied first, a deal at the SAME venue time must skip the cash leg
    # (a >= -> > regression would double-book the cash leg of essentially every fill).
    qty, r_pnl, (total, free, locked) = _run_fill_matrix(
        [_push(1048.0, as_of=T_EVENT), _closing_deal_event()]  # deal time == push as_of
    )
    assert qty == 0.0
    assert r_pnl == 50.0
    assert (total, free, locked) == (1048.0, 1048.0, 0.0)  # delta leg skipped, push figure stands


# --------------------------------------------------------------------------- #
# [FundingPaymentEvent, push] vs [push, FundingPaymentEvent]
# --------------------------------------------------------------------------- #


def _funding_event() -> FundingPaymentEvent:
    # long 1.0 marked at 50_000, rate 0.0001 -> pays 5 USDT
    payment = FundingPayment(time=int(T_EVENT.astype(np.int64)), funding_rate=0.0001, funding_interval_hours=8)
    return FundingPaymentEvent(instrument=INST, payment=payment)


def _run_funding_matrix(events: list[AccountMessage]) -> tuple[float, float, tuple[float, float, float]]:
    am = _am()
    pos = Position(INST, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(T_OPEN, 50_000.0, 1.0)  # the mark funding is valued at
    am.seed_position(pos)
    for event in events:
        am.apply(event)
    held = am.get_state(EX).get_position(INST)
    assert held is not None
    return held.cumulative_funding, held.r_pnl, _balances(am)


def test_same_funding_converges_under_both_event_orderings():
    # The venue debits the wallet and pushes 995; our computed FundingPaymentEvent books
    # cumulative_funding/r_pnl either way, the cash leg exactly once.
    funding_first = _run_funding_matrix([_funding_event(), _push(995.0)])
    push_first = _run_funding_matrix([_push(995.0), _funding_event()])
    assert funding_first == push_first
    cumulative_funding, r_pnl, (total, free, locked) = funding_first
    assert cumulative_funding == -5.0
    assert r_pnl == -5.0
    assert (total, free, locked) == (995.0, 995.0, 0.0)
