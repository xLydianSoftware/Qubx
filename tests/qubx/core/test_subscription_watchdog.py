import time
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
    """A genuine DARK episode (maintenance actually held) issues exactly one
    full-universe re-assert on recovery - contrast with test_flapping_exchange_
    issues_no_reassert, where maintenance never gets held at all."""
    watchdog, monitor, status, clock, universe, reconciled = rig
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
    assert reconciled == [set()]  # one full re-assert (refresh=frozenset() -> whole universe)


def test_flapping_exchange_issues_no_reassert(rig):
    """The resubscribe-storm case from the 2026-09-13 incident: single-tick DARK
    blips that never reach the maintenance threshold must not repeatedly tear down
    and re-assert the whole universe."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)

    for _ in range(4):  # 8 alternating ticks total, matching the incident reproduction
        watchdog.tick()  # one DARK tick: below the 2-tick maintenance threshold
        monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
        monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
        watchdog.tick()  # recovers within the tick: OK, no maintenance was ever held
        clock.advance(11)  # both age past threshold again for the next iteration

    assert reconciled == []
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
    """A trade feed that delivers once and goes quiet is not repaired again while
    its silence stays under the staleness threshold. Once its silence genuinely
    exceeds the threshold again, it is correctly repaired again - verification is
    not a permanent exemption, only proof that a specific repair worked."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.TRADE] = {btc, eth}
    monitor.subscribe(btc, DataType.TRADE)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(31)  # trade threshold is 30min
    monitor.on_data_arrival(btc, DataType.TRADE, clock.time())

    watchdog.tick()
    assert reconciled and reconciled[-1] == {eth}

    # eth delivers once, then nothing for a long time. btc keeps delivering on every
    # tick so the exchange stays PARTIAL rather than DARK - otherwise D5b's
    # no-repair-when-DARK would suppress repair for a reason unrelated to
    # verification, and the repair path for eth would never actually run.
    monitor.on_data_arrival(eth, DataType.TRADE, clock.time())
    before = len(reconciled)
    for _ in range(29):  # stay strictly under the 30min trade threshold
        clock.advance(1)
        monitor.on_data_arrival(btc, DataType.TRADE, clock.time())
        watchdog.tick()

    assert monitor.is_stale(eth, DataType.TRADE) is False
    assert len(reconciled) == before  # not repaired again while still under threshold

    # push eth's silence past the threshold: now it is correctly repaired again
    clock.advance(2)
    monitor.on_data_arrival(btc, DataType.TRADE, clock.time())
    watchdog.tick()

    assert monitor.is_stale(eth, DataType.TRADE) is True
    assert len(reconciled) == before + 1
    assert reconciled[-1] == {eth}


def test_verification_requires_delivery_after_the_repair(rig):
    """A last_event_time that predates the repair is not evidence the repair
    worked - only a message that arrives after it proves the subscription is live.
    Distinguishes advancement from a plain is_stale/'has ever delivered' check:
    the two agree except in exactly this ordering."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)  # past the 10min orderbook grace
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())  # eth's only ever message
    clock.advance(11)  # that message is now itself stale - keep btc fresh throughout
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())

    watchdog.tick()  # eth is genuinely stale (no message in >10min): a real repair
    assert reconciled and reconciled[-1] == {eth}
    assert (eth, "orderbook") in watchdog._repair_state

    clock.advance(1)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())  # eth still gets nothing
    watchdog.tick()

    # eth's last_event_time (set before the repair) has not moved: not verified.
    assert (eth, "orderbook") in watchdog._repair_state


def test_advancement_pops_record_on_relapse_without_new_repair(rig):
    """The case that actually discriminates advancement from is_stale: a sparse
    feed delivers once after the repair (verifying it), then goes stale again by
    threshold with no intervening tick observing the fresh window. Advancement
    pops the record on relapse and issues 0 new repairs; an is_stale-only rule
    would see "still stale" and issue 1. Confirmed by temporarily swapping the
    check to is_stale-only: this assertion then fails with reconciled growing by
    one (see task report for the swap and its output)."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    eth = _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {eth}
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)  # eth never delivered: genuinely stale

    watchdog.tick()
    assert reconciled == [{eth}]

    clock.advance(1)
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())  # verifies the repair
    clock.advance(11)  # no tick during the fresh window; now stale again purely by elapse

    watchdog.tick()

    assert reconciled == [{eth}], "advancement must issue 0 new repairs on relapse"
    assert (eth, "orderbook") not in watchdog._repair_state


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


def test_repair_does_not_contaminate_other_subscription_types(rig):
    """due used to pool instruments across all subscription types, so an
    instrument due on one type got a bogus _RepairRecord stamped (and a false
    "repairing" log line) on every other type it happens to also be subscribed
    to, even one that was never stale."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    eth = _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {eth}
    universe[DataType.TRADE] = {eth}
    monitor.subscribe(eth, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(11)  # orderbook (10min threshold) now stale
    monitor.on_data_arrival(eth, DataType.TRADE, clock.time())  # trade freshly delivering

    watchdog.tick()

    assert (eth, "orderbook") in watchdog._repair_state
    assert (eth, "trade") not in watchdog._repair_state


def test_repair_state_pruned_when_instrument_leaves_universe(rig):
    """_forget_repairs only runs on the OK branch, and _repair_stale only iterates
    the current universe - without explicit pruning, an instrument dropped from
    the universe while the exchange stays PARTIAL keeps its record forever."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())  # keep btc fresh: stay PARTIAL

    watchdog.tick()
    assert (eth, "orderbook") in watchdog._repair_state

    universe[DataType.ORDERBOOK] = {btc}  # eth unsubscribed at the manager level
    clock.advance(1)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert (eth, "orderbook") not in watchdog._repair_state


# --- parameterised subscription keys and the unpoliced-type filter (spec 4.1/4.2) ---

# "orderbook(0, 1)" - what the platform actually subscribes (context.py:275). Every rig test
# above keys its universe with the bare DataType.ORDERBOOK, which passes whether
# _watchdog_subscriptions / _repair_stale / BaseHealthMonitor.get_exchange_data_status
# normalise the key through DataType.from_str(sub)[0] or do a bare-key lookup that silently
# matches nothing - the exact blind spot that let the 2026-09-13 incident's deleted monitor
# go quiet. See task-6-report.md for the revert check pinning all three sites.
PARAM_ORDERBOOK = DataType.ORDERBOOK[0, 1]


def test_repair_reaches_reconcile_with_parameterised_subscription_key(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[PARAM_ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, PARAM_ORDERBOOK)
    monitor.subscribe(eth, PARAM_ORDERBOOK)
    clock.advance(11)
    monitor.on_data_arrival(btc, PARAM_ORDERBOOK, clock.time())

    watchdog.tick()

    assert reconciled and reconciled[-1] == {eth}


def test_all_stale_classifies_dark_with_parameterised_subscription_key(rig):
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[PARAM_ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, PARAM_ORDERBOOK)
    monitor.subscribe(eth, PARAM_ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    assert status.info.status is QubxStatus.NORMAL  # one dark tick, below the maintenance threshold

    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)
    assert status.info.degradations[0].reason is DegradeReason.EXCHANGE_MAINTENANCE


def test_unpoliced_types_are_excluded_from_repair_and_classification(rig):
    """No test put an unpoliced type (ohlc, funding, ...) in the universe before this one,
    so _watchdog_subscriptions' _WATCHDOG_DATA_TYPES filter could be deleted outright and
    the suite would stay green. An ohlc instrument must never be repaired and must never be
    counted toward the exchange's subscribed/stale ratio."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc}
    universe[DataType.OHLC["1h"]] = {eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.OHLC["1h"])
    clock.advance(11)  # past the orderbook threshold; ohlc has no STALE_THRESHOLDS entry at all

    watchdog.tick()

    assert reconciled and reconciled[-1] == {btc}
    assert (eth, "ohlc") not in watchdog._repair_state
    assert status.info.status is QubxStatus.NORMAL  # a single-instrument orderbook exchange is PARTIAL, not DARK

    exchange_status = monitor.get_exchange_data_status(EXCHANGE, universe)
    assert exchange_status.subscribed == 1  # eth (ohlc) never counted


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


# --- thread lifecycle ---

def test_start_stop_start_leaves_a_live_ticking_thread(rig):
    """stop() must clear the stop event and join the thread; otherwise a later
    start() spawns a thread whose wait() returns immediately (the event is still
    set) and tick() never runs - a silently dead watchdog, which is exactly the
    22-hour failure this module exists to prevent."""
    watchdog, monitor, status, clock, universe, reconciled = rig
    watchdog._interval_seconds = 0.02  # real wall-clock ticks, kept small for the test

    watchdog.start()
    watchdog.stop()  # first cycle: must fully tear down, not just drop the handle

    tick_count = 0
    original_tick = watchdog.tick

    def counting_tick():
        nonlocal tick_count
        tick_count += 1
        original_tick()

    watchdog.tick = counting_tick
    watchdog.start()
    try:
        deadline = time.monotonic() + 2.0
        while tick_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        watchdog.stop()

    assert tick_count > 0, "second start() produced a silently dead watchdog"
