import dataclasses

from qubx.core import events


def _all_message_classes():
    classes = []
    for name in dir(events):
        obj = getattr(events, name)
        if isinstance(obj, type) and issubclass(obj, events.ChannelMessage):
            classes.append(obj)
    return classes


# Implemented as a single test that iterates internally rather than a
# @pytest.mark.parametrize decorator because qubx pins pytest-lazy-fixture,
# which has an incompatibility with pytest 8.x that breaks parametrized test
# collection (see tests/qubx/connectors/ccxt/test_binance_has_restore.py).
def test_event_classes_are_frozen_and_slots():
    failures: list[str] = []
    for cls in _all_message_classes():
        if not dataclasses.is_dataclass(cls):
            failures.append(f"  {cls.__name__} is not a dataclass")
            continue
        if not cls.__dataclass_params__.frozen:
            failures.append(f"  {cls.__name__} is not frozen")
        if "__slots__" not in cls.__dict__:
            failures.append(f"  {cls.__name__} has no __slots__")

    assert not failures, "ChannelMessage subclasses must be frozen+slots dataclasses:\n" + "\n".join(failures)


def test_account_and_market_data_split():
    assert issubclass(events.OrderAcceptedEvent, events.AccountMessage)
    assert issubclass(events.QuoteEvent, events.MarketDataMessage)
    assert not issubclass(events.QuoteEvent, events.AccountMessage)
