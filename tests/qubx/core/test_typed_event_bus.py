import dataclasses

import pytest

from qubx.core import events
from qubx.core.basics import DataType
from qubx.core.events import (
    FundingPaymentEvent,
    FundingRateEvent,
    LiquidationEvent,
    OhlcEvent,
    OpenInterestEvent,
    OrderBookEvent,
    QuoteEvent,
    ScheduledEvent,
    TradeEvent,
    data_type_for_event,
    event_for_data_type,
)
from qubx.core.series import Bar, Quote


def _market_data_event_classes():
    return [
        c
        for c in vars(events).values()
        if isinstance(c, type) and issubclass(c, events.MarketDataMessage) and c is not events.MarketDataMessage
    ]


@pytest.mark.parametrize("cls", _market_data_event_classes(), ids=lambda c: c.__name__)
def test_market_data_event_is_frozen_and_slots(cls):
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert "__slots__" in cls.__dict__


def test_scheduled_event_is_frozen_and_slots():
    assert ScheduledEvent.__dataclass_params__.frozen
    assert "__slots__" in ScheduledEvent.__dict__


def test_new_market_data_events_are_market_data_not_account():
    for cls in (OhlcEvent, FundingRateEvent, OpenInterestEvent, LiquidationEvent):
        assert issubclass(cls, events.MarketDataMessage)
        assert not issubclass(cls, events.AccountMessage)


def test_event_for_data_type_wraps_and_round_trips():
    q = Quote(0, 1.0, 2.0, 1.0, 1.0)
    ev = event_for_data_type(DataType.QUOTE, instrument=None, payload=q)
    assert isinstance(ev, QuoteEvent)
    assert ev.quote is q
    assert ev.is_historical is False
    assert data_type_for_event(ev) == DataType.QUOTE


def test_event_for_data_type_carries_is_historical():
    ev = event_for_data_type(DataType.TRADE, instrument=None, payload=object(), is_historical=True)
    assert isinstance(ev, TradeEvent)
    assert ev.is_historical is True


def test_ohlc_event_preserves_timeframe_round_trip():
    bar = Bar(0, 1.0, 2.0, 0.5, 1.5, 10.0)
    ev = event_for_data_type("ohlc(1h)", instrument=None, payload=bar)
    assert isinstance(ev, OhlcEvent)
    assert ev.bar is bar
    # the parameterized data-type string is recoverable and stable under round-trip
    dt = data_type_for_event(ev)
    again = event_for_data_type(dt, instrument=None, payload=bar)
    assert data_type_for_event(again) == dt


def test_orderbook_round_trips_to_base_type():
    ev = event_for_data_type("orderbook(0.01, 200)", instrument=None, payload=object())
    assert isinstance(ev, OrderBookEvent)
    assert data_type_for_event(ev) == DataType.ORDERBOOK


def test_funding_payment_is_account_message_hybrid():
    # funding payment rides the market-data bus but is an AccountMessage (mutates balances)
    ev = event_for_data_type(DataType.FUNDING_PAYMENT, instrument=None, payload=object())
    assert isinstance(ev, FundingPaymentEvent)
    assert isinstance(ev, events.AccountMessage)
    assert data_type_for_event(ev) == DataType.FUNDING_PAYMENT


def test_unknown_data_type_raises():
    with pytest.raises(ValueError):
        event_for_data_type(DataType.FUNDAMENTAL, instrument=None, payload=object())
