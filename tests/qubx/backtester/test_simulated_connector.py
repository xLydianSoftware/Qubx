from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.backtester.connector import SimulatedConnector
from qubx.backtester.simulated_exchange import get_simulated_exchange
from qubx.core.basics import ZERO_COSTS, CtrlChannel, ITimeProvider, OrderRequest
from qubx.core.connector import IConnector
from qubx.core.events import (
    AccountSnapshotEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
)
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from qubx.core.utils import recognize_time


class _TimeService(ITimeProvider):
    """Time provider exposing both `time()` (consumed by the OME/exchange) and
    `now()` (consumed by SimulatedConnector for accepted_at/as_of stamps)."""

    _time: np.datetime64 = np.datetime64(0, "ns")

    def feed(self, quote: Quote) -> Quote:
        self._time = np.datetime64(quote.time, "ns")
        return quote

    def time(self) -> np.datetime64:
        return self._time

    def now(self) -> np.datetime64:
        return self._time


def Q(when: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(when), bid, ask, 0, 0)


def _drain(channel: CtrlChannel) -> list:
    events = []
    while True:
        try:
            events.append(channel.receive(timeout=0.05))
        except Exception:
            break
    return events


@pytest.fixture
def setup():
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert instr is not None
    time = _TimeService()
    exchange = get_simulated_exchange("BINANCE.UM", time, ZERO_COSTS)
    channel = CtrlChannel("sim")
    conn = SimulatedConnector(channel=channel, exchange=exchange, time_provider=time)
    # prime the order book so the OME is ready to accept orders; consume the
    # generator so the quote is actually processed.
    list(exchange.process_market_data(instr, time.feed(Q("2020-01-01 10:00", 32000.0, 32001.0))))
    return conn, channel, exchange, instr, time


def test_submit_limit_emits_accepted_event():
    channel = CtrlChannel("sim")
    channel.start()
    exchange = MagicMock()
    exchange.exchange_id = "binance.um"
    report = MagicMock()
    report.order.status = "OPEN"
    report.order.id = "V1"
    report.order.client_id = "qubx-1"
    report.exec = None
    report.instrument = MagicMock()
    exchange.place_order.return_value = report
    time = MagicMock()
    time.now.return_value = np.datetime64("2026-05-28")
    conn = SimulatedConnector(channel=channel, exchange=exchange, time_provider=time)
    request = OrderRequest(
        client_id="qubx-1",
        instrument=report.instrument,
        quantity=1.0,
        price=50_000.0,
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    msg = channel.receive(timeout=1)
    assert isinstance(msg, OrderAcceptedEvent)
    assert msg.client_order_id == "qubx-1"
    assert msg.venue_order_id == "V1"


def test_isinstance_iconnector():
    channel = CtrlChannel("sim")
    exchange = get_simulated_exchange("BINANCE.UM", _TimeService(), ZERO_COSTS)
    conn = SimulatedConnector(channel=channel, exchange=exchange, time_provider=_TimeService())
    assert isinstance(conn, IConnector)


def test_submit_resting_limit_emits_only_accepted(setup):
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-1",
        instrument=instr,
        quantity=0.1,
        price=31000.0,  # below bid: rests in book
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], OrderAcceptedEvent)
    assert events[0].client_order_id == "qubx-1"
    assert events[0].venue_order_id.startswith("SIM-ORDER")


def test_submit_crossing_limit_emits_filled(setup):
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-2",
        instrument=instr,
        quantity=0.1,
        price=33000.0,  # above ask: crosses and fills immediately
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    events = _drain(channel)
    assert any(isinstance(e, OrderFilledEvent) for e in events)
    fill_event = next(e for e in events if isinstance(e, OrderFilledEvent))
    assert fill_event.client_order_id == "qubx-2"
    assert fill_event.fill.amount == 0.1


def test_cancel_emits_canceled(setup):
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-3",
        instrument=instr,
        quantity=0.1,
        price=31000.0,
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    accepted = _drain(channel)[0]
    conn.cancel_order(venue_order_id=accepted.venue_order_id)
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], OrderCanceledEvent)
    assert events[0].venue_order_id == accepted.venue_order_id


def test_update_emits_single_updated_event_with_stable_cid(setup):
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-4",
        instrument=instr,
        quantity=0.1,
        price=31000.0,
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    accepted = _drain(channel)[0]
    conn.update_order(venue_order_id=accepted.venue_order_id, price=30500.0, quantity=0.2)
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], OrderUpdatedEvent)
    assert events[0].client_order_id == "qubx-4"
    assert events[0].new_price == 30500.0
    assert events[0].new_quantity == 0.2


def test_process_market_data_translates_fill(setup):
    conn, channel, _exchange, instr, time = setup
    request = OrderRequest(
        client_id="qubx-5",
        instrument=instr,
        quantity=0.1,
        price=31000.0,  # resting buy below market
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    _drain(channel)  # consume the accept
    # drop the market down through the resting buy -> it should fill
    conn.process_market_data(instr, time.feed(Q("2020-01-01 10:01", 30900.0, 30901.0)))
    events = _drain(channel)
    assert any(isinstance(e, OrderFilledEvent) for e in events)
    assert next(e for e in events if isinstance(e, OrderFilledEvent)).client_order_id == "qubx-5"


def test_request_snapshot_emits_account_snapshot(setup):
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-6",
        instrument=instr,
        quantity=0.1,
        price=31000.0,
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    _drain(channel)
    conn.request_snapshot()
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], AccountSnapshotEvent)
    snapshot = events[0].snapshot
    assert snapshot.exchange == "BINANCE.UM"
    assert snapshot.open_orders is not None
    assert len(snapshot.open_orders) == 1
    assert snapshot.open_orders[0].client_order_id == "qubx-6"


def test_request_order_status_open_emits_accepted(setup):
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-7",
        instrument=instr,
        quantity=0.1,
        price=31000.0,
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    conn.submit_order(request)
    accepted = _drain(channel)[0]
    conn.request_order_status(venue_order_id=accepted.venue_order_id)
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], OrderAcceptedEvent)
    assert events[0].venue_order_id == accepted.venue_order_id


def test_request_order_status_missing_emits_rejected(setup):
    conn, channel, _exchange, _instr, _time = setup
    conn.request_order_status(venue_order_id="does-not-exist")
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], OrderRejectedEvent)


def test_identity_and_lifecycle_stubs(setup):
    conn, _channel, _exchange, instr, _time = setup
    assert conn.is_ws_ready() is True
    assert conn.is_simulated_trading is True
    assert conn.read_only is False
    assert conn.make_client_id("qubx-abc") == "qubx-abc"
    assert conn.set_instrument_leverage(instr, 5.0) is True
    assert conn.set_margin_mode(instr, "cross") is True
    assert conn.exchange_name == "BINANCE.UM"
