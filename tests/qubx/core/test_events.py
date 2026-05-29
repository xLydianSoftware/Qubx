import dataclasses

import pytest

from qubx.core import events


def _all_message_classes():
    classes = []
    for name in dir(events):
        obj = getattr(events, name)
        if isinstance(obj, type) and issubclass(obj, events.ChannelMessage):
            classes.append(obj)
    return classes


@pytest.mark.parametrize("cls", _all_message_classes(), ids=lambda c: c.__name__)
def test_event_class_is_frozen_and_slots(cls):
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen
    assert "__slots__" in cls.__dict__


def test_account_and_market_data_split():
    assert issubclass(events.OrderAcceptedEvent, events.AccountMessage)
    assert issubclass(events.QuoteEvent, events.MarketDataMessage)
    assert not issubclass(events.QuoteEvent, events.AccountMessage)
