import numpy as np

from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import (
    Balance,
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderStatus,
    Position,
)


class _T:
    def __init__(self, t="2026-05-28T00:00:00"):
        self.t = np.datetime64(t)

    def time(self):
        return self.t


def _instrument(symbol="BTCUSDT", exchange="binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _order(inst, cid="cid-1", venue_id="V1") -> Order:
    return Order(
        client_order_id=cid,
        venue_order_id=venue_id,
        origin=OrderOrigin.FRAMEWORK,
        type="LIMIT",
        instrument=inst,
        time=np.datetime64("2026-05-28T00:00:00"),
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        status=OrderStatus.ACCEPTED,
        time_in_force="gtc",
    )


def _am() -> SimulatedAccountManager:
    am = SimulatedAccountManager(
        connectors={"binance": object()},
        base_currencies={"binance": "USDT"},
        time=_T(),
        account_id="acc-1",
    )
    state = am._states["binance"]
    inst = _instrument()
    state.set_position(inst, Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0))
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=900.0, locked=100.0))
    state.add_order(_order(inst))
    return am


def test_get_positions_returns_instrument_position_map():
    am = _am()
    inst = _instrument()
    positions = am.get_positions()
    assert set(positions.keys()) == {inst}
    assert positions[inst].quantity == 1.0
    # filtered by exchange yields the same single position
    assert am.get_positions(exchange="binance")[inst].quantity == 1.0
    # deprecated property mirrors get_positions
    assert am.positions == positions


def test_get_balances_returns_list_of_balances():
    am = _am()
    balances = am.get_balances()
    assert len(balances) == 1
    assert balances[0].currency == "USDT"
    assert balances[0].total == 1000.0
    assert am.get_balances(exchange="binance")[0].free == 900.0


def test_get_base_currency():
    am = _am()
    assert am.get_base_currency() == "USDT"
    assert am.get_base_currency(exchange="binance") == "USDT"


def test_find_order_by_id_uses_venue_id():
    am = _am()
    found = am.find_order_by_id("V1")
    assert found is not None
    assert found.client_order_id == "cid-1"
    assert am.find_order_by_id("nope") is None


def test_find_order_by_client_id():
    am = _am()
    found = am.find_order_by_client_id("cid-1")
    assert found is not None
    assert found.venue_order_id == "V1"
    assert am.find_order_by_client_id("nope") is None


def test_get_orders_instrument_filter():
    am = _am()
    inst = _instrument()
    other = _instrument(symbol="ETHUSDT")
    assert set(am.get_orders(instrument=inst).keys()) == {"cid-1"}
    assert am.get_orders(instrument=other) == {}


def test_position_report():
    am = _am()
    report = am.position_report()
    assert "BTCUSDT" in report
    assert report["BTCUSDT"]["Qty"] == 1.0
