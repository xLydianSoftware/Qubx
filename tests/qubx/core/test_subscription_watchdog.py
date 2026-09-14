from unittest.mock import Mock

import numpy as np
import pytest

from qubx.core.basics import DataType, ITimeProvider
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.status import ContextStatus, DegradeReason, QubxStatus
from qubx.core.subscription_watchdog import ExchangeClassification, SubscriptionWatchdog
from qubx.health.base import BaseHealthMonitor
from qubx.health.status import ExchangeDataStatus

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


def _status(**kw) -> ExchangeDataStatus:
    base = dict(
        exchange=EXCHANGE, connected=True, subscribed=0, stale=0, in_grace=0, last_event_time=None
    )
    base.update(kw)
    return ExchangeDataStatus(**base)


@pytest.fixture
def rig():
    clock = FixedTime()
    monitor = BaseHealthMonitor(clock)
    status = ContextStatus()
    monitor.set_status(status)
    provider = Mock()
    provider.is_simulation = False
    provider.exchange.return_value = EXCHANGE
    state = StrategyState()
    state.is_on_warmup_finished_called = True
    universe: dict[str, set] = {}
    reconciled: list[set] = []

    watchdog = SubscriptionWatchdog(
        data_providers=[provider],
        health_monitor=monitor,
        status=status,
        reconcile_fn=lambda refresh=frozenset(): reconciled.append(set(refresh)),
        subscriptions_fn=lambda: universe,
        strategy_state=state,
        interval_seconds=30.0,
    )
    return watchdog, monitor, status, clock, universe, reconciled


# --- classification (spec 4.2) ---

def test_classify_ok():
    assert SubscriptionWatchdog.classify(_status(subscribed=5, stale=0)) is ExchangeClassification.OK


def test_classify_dark_when_disconnected():
    s = _status(connected=False, subscribed=5, stale=0)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.DARK


def test_classify_dark_when_all_stale():
    s = _status(subscribed=5, stale=5)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.DARK


def test_classify_partial_when_some_stale():
    s = _status(subscribed=5, stale=2)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.PARTIAL


def test_single_instrument_all_stale_is_partial_not_dark():
    s = _status(subscribed=1, stale=1)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.PARTIAL


def test_connected_none_does_not_make_it_dark():
    s = _status(connected=None, subscribed=5, stale=0)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.OK


# --- maintenance publication (spec 6) ---

def test_maintenance_needs_two_consecutive_dark_ticks(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    assert status.info.status is QubxStatus.NORMAL

    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)
    assert status.info.degradations[0].reason is DegradeReason.EXCHANGE_MAINTENANCE


def test_maintenance_clears_on_first_delivering_tick(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    watchdog.tick()
    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)

    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL


def test_dark_issues_no_repair(rig):
    """The 11:16 regression: 21 simultaneous subscribes tripped the venue's
    message rate limit and cost the recovery."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    watchdog.tick()
    watchdog.tick()

    assert reconciled == []


def test_partial_never_publishes_maintenance(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    for _ in range(5):
        watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert reconciled and reconciled[0] == {eth}


def test_grace_prevents_false_dark_on_universe_swap(rig):
    """A full universe replacement leaves every instrument with no last-event
    time. Without the grace gate this halts trading on a healthy venue."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)

    watchdog.tick()
    watchdog.tick()
    watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert reconciled == []


# --- verification and backoff (spec 4.3, D5) ---

def test_verification_is_by_advancement_not_by_staleness(rig):
    """A trade feed that delivers once and goes quiet stays stale by threshold
    but must never be repaired again."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.TRADE] = {btc, eth}
    monitor.subscribe(btc, DataType.TRADE)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(31)  # trade threshold is 30min
    monitor.on_data_arrival(btc, DataType.TRADE, clock.time())

    watchdog.tick()
    assert reconciled and reconciled[-1] == {eth}

    # eth delivers once, then nothing for a long time
    monitor.on_data_arrival(eth, DataType.TRADE, clock.time())
    before = len(reconciled)
    for _ in range(31):  # must clear the 30min trade threshold from eth's single event
        clock.advance(1)
        watchdog.tick()

    assert monitor.is_stale(eth, DataType.TRADE) is True
    assert len(reconciled) == before


def test_unverified_repair_backs_off_and_never_stops(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    attempts = 0
    for _ in range(40):
        before = len(reconciled)
        watchdog.tick()
        attempts += len(reconciled) - before

    assert 2 <= attempts < 40, "repairs must back off but never stop"


def test_backoff_resets_after_recovery(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    for _ in range(6):
        watchdog.tick()

    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()
    assert watchdog._repair_state == {}


def test_repair_never_calls_health_monitor_unsubscribe(rig):
    """Popping _last_event_time would destroy the verification signal."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.unsubscribe = Mock(side_effect=AssertionError("repair must not unsubscribe in health"))

    for _ in range(5):
        watchdog.tick()


# --- gating ---

def test_does_nothing_before_warmup_finished(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    watchdog._strategy_state.is_on_warmup_finished_called = False
    btc = _instrument("BTCUSDT")
    universe[DataType.ORDERBOOK] = {btc}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    watchdog.tick()

    assert reconciled == []
    assert status.info.status is QubxStatus.NORMAL


def test_tick_survives_an_exception(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    watchdog._subscriptions_fn = Mock(side_effect=RuntimeError("boom"))
    watchdog.tick()  # must not raise
