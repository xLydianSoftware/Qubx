"""One venue account serving two instrument universes: BINANCE spot on a BINANCE.PM account.

The AM keeps ONE state for the account (keyed BINANCE.UM) and routes BINANCE instruments to it
through ``exchange_aliases``; capital is counted once. Spot holdings are balances, so a snapshot
is authoritative for their size only — the average price stays the local ledger's.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.account_manager.manager import AccountManager
from qubx.core.basics import (
    Deal,
    ITimeProvider,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.events import AccountSnapshot, AccountSnapshotEvent, OrderAcceptedEvent, OrderFilledEvent
from qubx.core.lookups import lookup

T0 = np.datetime64("2026-10-12T00:00:00", "ns")
T1 = np.datetime64("2026-10-12T00:01:00", "ns")
T2 = np.datetime64("2026-10-12T00:02:00", "ns")

UM = "BINANCE.UM"
SPOT = lookup.find_symbol("BINANCE", "TRXUSDT")
PERP = lookup.find_symbol(UM, "TRXUSDT")
assert SPOT is not None and PERP is not None


class _Time(ITimeProvider):
    def __init__(self):
        self.now = T1

    def time(self) -> np.datetime64:
        return self.now


def _am() -> tuple[AccountManager, MagicMock, _Time]:
    conn = MagicMock()
    clock = _Time()
    am = AccountManager(
        connectors={UM: conn},
        base_currencies={UM: "USDT"},
        time=clock,
        exchange_aliases={"BINANCE": UM},
    )
    return am, conn, clock


def _order(instrument, cid: str, side: OrderSide = OrderSide.BUY, qty: float = 60.0) -> Order:
    return Order(
        client_order_id=cid,
        type=OrderType.MARKET,
        instrument=instrument,
        quantity=qty,
        side=side,
        time_in_force="gtc",
        status=OrderStatus.SUBMITTED,
        origin=OrderOrigin.FRAMEWORK,
    )


def _fill(am: AccountManager, instrument, cid: str, amount: float, price: float, tid: str) -> None:
    side = OrderSide.BUY if amount > 0 else OrderSide.SELL
    am.add_order(_order(instrument, cid, side, abs(amount)))
    am.apply(OrderAcceptedEvent(instrument=instrument, client_order_id=cid, venue_order_id=f"v{cid}", accepted_at=T0))
    deal = Deal(tid, f"v{cid}", T0, amount, price, True, fee_amount=0.0, fee_currency="USDT")
    am.apply(
        OrderFilledEvent(
            instrument=instrument,
            client_order_id=cid,
            venue_order_id=f"v{cid}",
            last_update_time=T0,
            fill=deal,
        )
    )


def _snapshot(as_of, positions, equity=454.0) -> AccountSnapshotEvent:
    return AccountSnapshotEvent(
        instrument=None,
        snapshot=AccountSnapshot(exchange=UM, as_of=as_of, positions=positions, balances=[], equity=equity),
    )


def _spot_pos(qty: float, price: float) -> Position:
    pos = Position(instrument=SPOT, quantity=qty, pos_average_price=price)
    pos.update_market_price(T0, price, 1)
    return pos


def test_alias_must_point_at_a_managed_exchange():
    with pytest.raises(ValueError):
        AccountManager(
            connectors={UM: MagicMock()},
            base_currencies={UM: "USDT"},
            time=_Time(),
            exchange_aliases={"BINANCE": "OKX"},
        )
    with pytest.raises(ValueError):
        AccountManager(
            connectors={UM: MagicMock(), "BINANCE": MagicMock()},
            base_currencies={UM: "USDT", "BINANCE": "USDT"},
            time=_Time(),
            exchange_aliases={"BINANCE": UM},
        )


def test_spot_and_perp_share_one_state_and_one_capital():
    am, _, _ = _am()
    assert am.resolve_exchange("BINANCE") == UM
    assert am.get_state("BINANCE") is am.get_state(UM)

    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    _fill(am, PERP, "p1", -60.0, 0.30, "t2")
    assert am.get_position(SPOT).quantity == 60.0
    assert am.get_position(PERP).quantity == -60.0

    am.apply(_snapshot(T1, [_spot_pos(60.0, 0.30)], equity=454.0))
    # one account, counted once — whichever exchange the caller names
    assert am.get_total_capital() == 454.0
    assert am.get_total_capital("BINANCE") == 454.0
    # exchange-scoped reads filter by the instrument's own exchange
    assert set(am.get_positions("BINANCE")) == {SPOT}
    assert set(am.get_positions(UM)) == {PERP}
    assert set(am.get_positions()) == {SPOT, PERP}


def test_orders_route_by_instrument_and_filter_by_exchange():
    am, _, _ = _am()
    am.add_order(_order(SPOT, "s1"))
    am.add_order(_order(PERP, "p1"))
    assert set(am.get_orders(exchange="BINANCE")) == {"s1"}
    assert set(am.get_orders(exchange=UM)) == {"p1"}
    assert am.get_order("s1", exchange="BINANCE") is not None


def test_spot_snapshot_keeps_ledger_average_price():
    am, _, clock = _am()
    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    clock.now = T2
    # holding = balance, marked at today's price: avg must stay the 0.30 the fill booked
    am.apply(_snapshot(T2, [_spot_pos(60.0, 0.35)]))
    pos = am.get_position(SPOT)
    assert pos.quantity == 60.0
    assert pos.position_avg_price == pytest.approx(0.30)


def test_spot_snapshot_corrects_size_but_not_average():
    am, _, clock = _am()
    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    clock.now = T2
    # the venue charged the buy fee in TRX: holding is a lot short of the gross fill
    am.apply(_snapshot(T2, [_spot_pos(59.9, 0.31)]))
    pos = am.get_position(SPOT)
    assert pos.quantity == pytest.approx(59.9)
    assert pos.position_avg_price == pytest.approx(0.30)


def test_restart_adopts_spot_holding_at_snapshot_price():
    am, _, _ = _am()
    am.apply(_snapshot(T1, [_spot_pos(60.0, 0.31)]))
    pos = am.get_position(SPOT)
    assert pos.quantity == 60.0
    assert pos.position_avg_price == pytest.approx(0.31)


def test_spot_sold_out_reads_flat():
    am, _, clock = _am()
    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    clock.now = T2
    am.apply(_snapshot(T2, []))
    assert am.get_position(SPOT).quantity == 0.0
