import numpy as np

from qubx.core.basics import DataType, ITimeProvider
from qubx.core.lookups import lookup
from qubx.health.base import BaseHealthMonitor

EXCHANGE = "BINANCE.UM"


class FixedTime(ITimeProvider):
    def __init__(self) -> None:
        self.now = np.datetime64("2026-09-13T00:00:00", "ns")

    def time(self) -> np.datetime64:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + np.timedelta64(minutes, "m")


def _instrument(symbol: str):
    instr = lookup.find_symbol(EXCHANGE, symbol)
    assert instr is not None
    return instr


def _monitor():
    clock = FixedTime()
    return BaseHealthMonitor(clock), clock


def test_fresh_subscription_is_in_grace_not_stale():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.in_grace == 1
    assert status.subscribed == 0
    assert status.stale == 0


def test_becomes_eligible_and_stale_after_the_grace_window():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)  # orderbook threshold is 10min

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.in_grace == 0
    assert status.subscribed == 1
    assert status.stale == 1


def test_delivering_instrument_is_not_stale():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.stale == 0
    assert status.subscribed == 1
    assert status.last_event_time == clock.time()


def test_counts_are_per_instrument_not_max_over_the_exchange():
    """One live instrument must not mask nineteen dead ones."""
    monitor, clock = _monitor()
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc, eth}})

    assert status.subscribed == 2
    assert status.stale == 1


def test_connected_is_none_when_no_callback_registered():
    monitor, _ = _monitor()
    status = monitor.get_exchange_data_status(EXCHANGE, {})
    assert status.connected is None


def test_connected_reflects_the_registered_callback():
    monitor, _ = _monitor()
    monitor.set_is_connected(EXCHANGE, lambda: False)
    assert monitor.get_exchange_data_status(EXCHANGE, {}).connected is False


def test_unsubscribe_clears_subscribed_at():
    monitor, clock = _monitor()
    btc = _instrument("BTCUSDT")
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.unsubscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(5)

    status = monitor.get_exchange_data_status(EXCHANGE, {DataType.ORDERBOOK: {btc}})

    assert status.in_grace == 1
