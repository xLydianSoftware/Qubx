"""One venue account trading two markets: BINANCE spot on a BINANCE.PM account.

The AM keeps ONE state for the account (its id is its own market, BINANCE.UM) and routes BINANCE
instruments to it through ``account_markets``; capital is counted once. Spot holdings are balances,
so a snapshot is authoritative for their size only — the average price stays the local ledger's.
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
        account_markets={UM: (UM, "BINANCE")},
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


def test_the_account_must_be_a_managed_exchange():
    with pytest.raises(ValueError):
        AccountManager(
            connectors={UM: MagicMock()},
            base_currencies={UM: "USDT"},
            time=_Time(),
            account_markets={"OKX": ("OKX", "BINANCE")},
        )


def test_a_market_belongs_to_one_account():
    with pytest.raises(ValueError, match="two accounts"):
        AccountManager(
            connectors={UM: MagicMock(), "OKX": MagicMock()},
            base_currencies={UM: "USDT", "OKX": "USDT"},
            time=_Time(),
            account_markets={UM: (UM, "BINANCE"), "OKX": ("OKX", "BINANCE")},
        )


def test_account_markets_list_every_account():
    am, _, _ = _am()
    assert am.get_account_markets() == {UM: (UM, "BINANCE")}
    assert am.account_of("BINANCE") == UM
    assert am.account_of(UM) == UM


def test_a_market_with_its_own_connector_executes_there_but_books_into_the_account():
    um, spot = MagicMock(), MagicMock()
    am = AccountManager(
        connectors={UM: um, "BINANCE": spot},
        base_currencies={UM: "USDT", "BINANCE": "USDT"},
        time=_Time(),
        account_markets={UM: (UM, "BINANCE")},
    )
    assert am.get_state("BINANCE") is am.get_state(UM)
    assert len(am._states) == 1
    assert am._connector_for("BINANCE") is spot
    assert am._connector_for(UM) is um


def test_spot_and_perp_share_one_state_and_one_capital():
    am, _, _ = _am()
    assert am.get_state("BINANCE") is am.get_state(UM)

    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    _fill(am, PERP, "p1", -60.0, 0.30, "t2")
    assert am.get_position(SPOT).quantity == 60.0
    assert am.get_position(PERP).quantity == -60.0

    am.apply(_snapshot(T1, [_spot_pos(60.0, 0.30)], equity=454.0))
    # one account, counted once — whichever exchange the caller names
    assert am.get_total_capital() == 454.0
    assert am.get_total_capital("BINANCE") == 454.0
    # positions and orders are per market
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


def test_snapshot_from_a_market_connector_is_ignored():
    am = AccountManager(
        connectors={UM: MagicMock(), "BINANCE": MagicMock()},
        base_currencies={UM: "USDT", "BINANCE": "USDT"},
        time=_Time(),
        account_markets={UM: (UM, "BINANCE")},
    )
    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    # - the spot connector's snapshot holds only its slice; applied, it would read the account as flat
    am.apply(
        AccountSnapshotEvent(instrument=None, snapshot=AccountSnapshot(exchange="BINANCE", as_of=T2, positions=[]))
    )
    assert am.get_position(SPOT).quantity == 60.0
    assert am.get_state(UM).get_last_snapshot_as_of() is None


def _holding_both_legs() -> AccountManager:
    am, _, _ = _am()
    _fill(am, SPOT, "s1", 60.0, 0.30, "t1")
    _fill(am, PERP, "p1", -60.0, 0.30, "t2")
    perp = Position(instrument=PERP, quantity=-60.0, pos_average_price=0.30)
    perp.update_market_price(T0, 0.30, 1)
    am.apply(_snapshot(T1, [_spot_pos(60.0, 0.30), perp], equity=454.0))
    return am


def _assert_one_entry_per_market(entries: dict) -> None:
    # - positions per market (spot and perp TRXUSDT never collide); the account's figures once
    assert set(entries) == {UM, "BINANCE"}
    assert entries[UM]["positions"]["TRXUSDT"]["quantity"] == -60.0
    assert entries["BINANCE"]["positions"]["TRXUSDT"]["quantity"] == 60.0
    assert entries[UM]["account"] == entries["BINANCE"]["account"] == UM
    assert entries[UM]["capital"]["total"] == pytest.approx(454.0)
    assert entries["BINANCE"]["capital"]["total"] == 0.0
    assert entries["BINANCE"]["balances"] == {}


def test_state_snapshot_has_an_entry_per_market_and_the_capital_once():
    from types import SimpleNamespace

    from qubx.core.mixins.processing import ProcessingManager

    am = _holding_both_legs()
    saved: dict = {}
    ctx = SimpleNamespace(
        _strategy_state=SimpleNamespace(is_on_start_called=True),
        account=am,
        exchanges=[UM],
        persistence=SimpleNamespace(save=saved.__setitem__),
    )
    pm = SimpleNamespace(_context=ctx, _account_manager=am, _time_provider=_Time(), _last_event_times=lambda ex: {})
    ProcessingManager._handle_state_snapshot(pm, None, "state_snapshot", (None, T1))  # type: ignore[arg-type]

    _assert_one_entry_per_market(saved["state"]["exchanges"])


def test_get_state_has_an_entry_per_market_and_the_capital_once():
    from qubx.control.builtin import _get_state

    am = _holding_both_legs()
    ctx = MagicMock()
    ctx.account = am
    ctx.exchanges = [UM]
    ctx.get_total_capital.return_value = am.get_total_capital()
    ctx.strategy = object()

    result = _get_state(ctx)

    assert result.status == "ok", result.error
    _assert_one_entry_per_market(result.data["exchanges"])
