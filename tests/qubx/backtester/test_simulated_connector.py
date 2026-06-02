from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.backtester.connector import SimulatedConnector
from qubx.backtester.simulated_exchange import get_simulated_exchange
from qubx.core.basics import (
    OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION,
    OPTION_FILL_AT_SIGNAL_PRICE,
    ZERO_COSTS,
    CtrlChannel,
    ITimeProvider,
    OrderRequest,
)
from qubx.core.connector import IConnector
from qubx.core.events import (
    AccountSnapshotEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
)
from qubx.core.exceptions import InvalidOrder, QueueTimeout
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from qubx.core.utils import recognize_time


class _TimeService(ITimeProvider):
    """Time provider exposing `time()` consumed by both the OME/exchange and SimulatedConnector."""

    _time: np.datetime64 = np.datetime64(0, "ns")

    def feed(self, quote: Quote) -> Quote:
        self._time = np.datetime64(quote.time, "ns")
        return quote

    def time(self) -> np.datetime64:
        return self._time


def Q(when: str, bid: float, ask: float) -> Quote:
    return Quote(recognize_time(when), bid, ask, 0, 0)


def _drain(channel: CtrlChannel) -> list:
    events = []
    while True:
        try:
            events.append(channel.receive(timeout=0.05))
        except QueueTimeout:
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
    conn.cancel_order(client_order_id=accepted.client_order_id, venue_order_id=accepted.venue_order_id)
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
    conn.update_order(
        client_order_id=accepted.client_order_id, venue_order_id=accepted.venue_order_id, price=30500.0, quantity=0.2
    )
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
    conn.request_order_status(client_order_id=accepted.client_order_id, venue_order_id=accepted.venue_order_id)
    events = _drain(channel)
    assert len(events) == 1
    assert isinstance(events[0], OrderAcceptedEvent)
    assert events[0].venue_order_id == accepted.venue_order_id


def test_request_order_status_missing_emits_rejected(setup):
    conn, channel, _exchange, _instr, _time = setup
    conn.request_order_status(client_order_id="does-not-exist")
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


def test_submit_stop_order_options_forwarded_to_ome(setup):
    # A STOP_MARKET sell placed below current bid would normally raise "would trigger
    # immediately". With avoid_stop_order_price_validation=True the OME skips that check
    # and registers the order. This proves submit_order forwards request.options.
    conn, channel, exchange, instr, time = setup
    # market is 32000/32001; place a sell stop BELOW bid (would trigger without the flag)
    stop_price = 31500.0
    request = OrderRequest(
        client_id="qubx-stop-1",
        instrument=instr,
        quantity=0.1,
        price=stop_price,
        side="SELL",
        order_type="STOP_MARKET",
        time_in_force="gtc",
        options={
            OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION: True,
            OPTION_FILL_AT_SIGNAL_PRICE: True,
        },
    )
    conn.submit_order(request)
    events = _drain(channel)
    # The OME accepted the stop order (OPEN) — no rejection.
    assert any(isinstance(e, OrderAcceptedEvent) for e in events), events

    # Confirm the OME order carries the forwarded options.
    open_orders = exchange.get_open_orders()
    assert len(open_orders) == 1
    order = next(iter(open_orders.values()))
    assert order.options.get(OPTION_FILL_AT_SIGNAL_PRICE) is True
    assert order.options.get(OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION) is True


def test_submit_would_trigger_stop_emits_rejected_not_raises(setup):
    # A BUY STOP_MARKET below the ask "would trigger immediately" (OME: BUY triggers
    # when ask >= stop price) — a venue verdict. Per the connector rejection boundary
    # it must ride the channel as an OrderRejectedEvent, NOT raise out of submit_order.
    conn, channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-stop-reject",
        instrument=instr,
        quantity=0.1,
        price=31500.0,  # below ask 32001 -> BUY stop would trigger immediately
        side="BUY",
        order_type="STOP_MARKET",
        time_in_force="gtc",
    )
    conn.submit_order(request)  # must not raise
    events = _drain(channel)
    rejects = [e for e in events if isinstance(e, OrderRejectedEvent)]
    assert len(rejects) == 1, events
    assert rejects[0].client_order_id == "qubx-stop-reject"
    assert "would trigger" in rejects[0].reason.lower()
    assert not any(isinstance(e, OrderAcceptedEvent) for e in events)


def test_submit_invalid_amount_raises(setup):
    # Framework-side rejection (amount <= 0) must raise synchronously — it is the
    # caller's bug to fix, not a venue verdict that rides the channel.
    conn, _channel, _exchange, instr, _time = setup
    request = OrderRequest(
        client_id="qubx-bad",
        instrument=instr,
        quantity=-1.0,
        price=32000.0,
        side="BUY",
        order_type="LIMIT",
        time_in_force="gtc",
    )
    with pytest.raises(InvalidOrder):
        conn.submit_order(request)


def test_stop_order_fills_at_signal_price_not_bbo(setup):
    # Verify that fill_at_signal_price=True causes the fill to happen at the stop
    # price rather than the bar-close BBO when the market crosses the stop level.
    conn, channel, exchange, instr, time = setup
    # market is 32000/32001; place a resting sell stop well above current ask so it
    # rests in the book, then drive a bar that crosses it.
    stop_price = 32500.0
    request = OrderRequest(
        client_id="qubx-stop-2",
        instrument=instr,
        quantity=0.1,
        price=stop_price,
        side="BUY",
        order_type="STOP_MARKET",
        time_in_force="gtc",
        options={OPTION_FILL_AT_SIGNAL_PRICE: True},
    )
    conn.submit_order(request)
    _drain(channel)  # consume the accept

    # Drive the market up past the stop level; BBO ask is 33001 — well above stop.
    conn.process_market_data(instr, time.feed(Q("2020-01-01 10:01", 33000.0, 33001.0)))
    events = _drain(channel)

    fills = [e for e in events if isinstance(e, OrderFilledEvent)]
    assert len(fills) == 1, f"expected 1 fill event, got {events}"
    # With fill_at_signal_price the fill must be at the stop price, not the BBO ask.
    assert fills[0].fill.price == stop_price, f"expected fill at {stop_price}, got {fills[0].fill.price}"


def test_update_order_preserves_options(setup):
    # update_order cancel+recreates; the recreated order must carry the original options.
    conn, channel, exchange, instr, time = setup
    stop_price = 32500.0
    request = OrderRequest(
        client_id="qubx-stop-3",
        instrument=instr,
        quantity=0.1,
        price=stop_price,
        side="BUY",
        order_type="STOP_MARKET",
        time_in_force="gtc",
        options={OPTION_FILL_AT_SIGNAL_PRICE: True},
    )
    conn.submit_order(request)
    accepted = _drain(channel)[0]

    new_stop_price = 32600.0
    conn.update_order(
        client_order_id=accepted.client_order_id, venue_order_id=accepted.venue_order_id, price=new_stop_price
    )
    _drain(channel)  # consume the updated event

    open_orders = exchange.get_open_orders()
    assert len(open_orders) == 1
    order = next(iter(open_orders.values()))
    assert order.price == new_stop_price
    # The recreated order must still carry the original fill_at_signal_price flag.
    assert order.options.get(OPTION_FILL_AT_SIGNAL_PRICE) is True
