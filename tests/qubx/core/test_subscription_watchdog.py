import threading
import time
from typing import Callable
from unittest.mock import Mock

import numpy as np
import pytest

from qubx.core import subscription_watchdog as watchdog_module
from qubx.core.basics import DataType, Instrument, ITimeProvider
from qubx.core.exceptions import NotSupported
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.status import ContextStatus, DegradeReason, QubxStatus
from qubx.core.subscription_watchdog import ExchangeClassification, SubscriptionWatchdog
from qubx.health.base import BaseHealthMonitor
from qubx.health.status import ExchangeDataStatus

EXCHANGE = "BINANCE.UM"
OTHER_EXCHANGE = "BITFINEX.F"

# "orderbook(0, 1)" - what the platform actually subscribes (context.py:275). Most tests key
# their universe with the bare DataType.ORDERBOOK, which passes whether the three
# normalisation sites (the watchdog's policed filter, its repair keying, and
# BaseHealthMonitor.get_exchange_data_status) go through DataType.from_str(sub)[0] or do a
# bare-key lookup that silently matches nothing - the blind spot that let the 2026-09-13
# incident's deleted monitor go quiet. The parameterised tests below pin all three.
PARAM_ORDERBOOK = DataType.ORDERBOOK[0, 1]


class FixedTime(ITimeProvider):
    def __init__(self) -> None:
        self.now = np.datetime64("2026-09-13T00:00:00", "ns")

    def time(self) -> np.datetime64:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + np.timedelta64(minutes, "m")


class _Provider:
    """Records what the watchdog asks of it. `subscribed`/`unsubscribed` hold
    (subscription key, instruments) in call order."""

    is_simulation = False

    def __init__(self, exchange: str) -> None:
        self._exchange = exchange
        self.unsubscribed: list[tuple[str, set[Instrument]]] = []
        self.subscribed: list[tuple[str, set[Instrument]]] = []
        self.fail_unsubscribe: Exception | None = None
        self.fail_subscribe: Exception | None = None
        self.on_unsubscribe: Callable[[], None] | None = None

    def exchange(self) -> str:
        return self._exchange

    def unsubscribe(self, sub: str, instruments: set[Instrument]) -> None:
        if self.on_unsubscribe is not None:
            self.on_unsubscribe()
        if self.fail_unsubscribe is not None:
            raise self.fail_unsubscribe
        self.unsubscribed.append((sub, set(instruments)))

    def subscribe(self, sub: str, instruments: set[Instrument], reset: bool = False) -> None:
        if self.fail_subscribe is not None:
            raise self.fail_subscribe
        self.subscribed.append((sub, set(instruments)))


def _instrument(symbol: str, exchange: str = EXCHANGE) -> Instrument:
    instr = lookup.find_symbol(exchange, symbol)
    assert instr is not None
    return instr


def _repairs(provider: _Provider) -> list[set[tuple[Instrument, str]]]:
    """Each unsubscribe the watchdog issued, as the (instrument, base type) set it repaired."""
    return [{(i, str(DataType.from_str(sub)[0])) for i in instrs} for sub, instrs in provider.unsubscribed]


def _status(**kw) -> ExchangeDataStatus:
    stale = kw.pop("stale", 0)
    base = dict(
        exchange=EXCHANGE,
        connected=True,
        subscribed=0,
        stale_keys=frozenset((f"i{n}", "orderbook") for n in range(stale)),
        in_grace=0,
        last_event_time=None,
    )
    base.update(kw)
    return ExchangeDataStatus(**base)


def _make_rig(*exchanges: str, venue_of: dict[str, str] | None = None):
    """One watchdog over the given exchanges. `venue_of` maps a canonical exchange to the
    name its provider is registered under (BINANCE.UM -> BINANCE.PM), defaulting to itself."""
    venue_of = venue_of or {}
    clock = FixedTime()
    monitor = BaseHealthMonitor(clock)
    status = ContextStatus()
    monitor.set_status(status)
    providers = {ex: _Provider(venue_of.get(ex, ex)) for ex in exchanges}
    state = StrategyState()
    state.is_on_warmup_finished_called = True
    # - exchange -> subscription -> instruments, mutated by tests in place
    universe: dict[str, dict[str, set[Instrument]]] = {ex: {} for ex in exchanges}

    watchdog = SubscriptionWatchdog(
        data_providers=list(providers.values()),
        health_monitor=monitor,
        status=status,
        snapshot_fn=lambda: {ex: {s: frozenset(i) for s, i in subs.items()} for ex, subs in universe.items()},
        strategy_state=state,
        interval_seconds=30.0,
        settle_seconds=0.0,
    )
    return watchdog, monitor, status, clock, universe, providers


@pytest.fixture
def rig():
    watchdog, monitor, status, clock, universe, providers = _make_rig(EXCHANGE)
    return watchdog, monitor, status, clock, universe[EXCHANGE], providers[EXCHANGE]


@pytest.fixture
def two_venue_rig():
    return _make_rig(EXCHANGE, OTHER_EXCHANGE)


def _subscribe_stale_pair(monitor, clock, universe, sub=DataType.ORDERBOOK, deliver_btc=False):
    """btc and eth subscribed and aged past the orderbook threshold; optionally btc delivering
    so the exchange reads PARTIAL rather than DARK."""
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[sub] = {btc, eth}
    monitor.subscribe(btc, sub)
    monitor.subscribe(eth, sub)
    clock.advance(11)
    if deliver_btc:
        monitor.on_data_arrival(btc, sub, clock.time())
    return btc, eth


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


def test_disconnected_with_nothing_eligible_is_not_dark():
    """Limb (A) must have something to be dark about. is_connected() reads False for a
    provider whose callback raises too, so without this an exchange the watchdog polices
    nothing on would refuse every order - reduce-only included."""
    s = _status(connected=False, subscribed=0, stale=0, in_grace=3)
    assert SubscriptionWatchdog.classify(s) is ExchangeClassification.OK


# --- maintenance publication (spec 6) ---


def test_maintenance_needs_two_consecutive_dark_ticks(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    _subscribe_stale_pair(monitor, clock, universe)

    watchdog.tick()
    assert status.info.status is QubxStatus.NORMAL

    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)
    assert status.info.degradations[0].reason is DegradeReason.EXCHANGE_MAINTENANCE


def test_maintenance_clears_on_first_delivering_tick(rig):
    """A genuine DARK episode (maintenance actually held) issues exactly one
    full-universe re-assert on recovery - contrast with test_flapping_exchange_
    issues_no_reassert, where maintenance never gets held at all."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe)
    watchdog.tick()
    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)

    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert provider.unsubscribed == []
    assert provider.subscribed == [(DataType.ORDERBOOK, {btc, eth})]


def test_flapping_exchange_issues_no_reassert(rig):
    """The resubscribe-storm case from the 2026-09-13 incident: single-tick DARK
    blips that never reach the maintenance threshold must not repeatedly tear down
    and re-assert the whole universe."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe)

    for _ in range(4):  # 8 alternating ticks total, matching the incident reproduction
        watchdog.tick()  # one DARK tick: below the 2-tick maintenance threshold
        monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
        monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
        watchdog.tick()  # recovers within the tick: OK, no maintenance was ever held
        clock.advance(11)  # both age past threshold again for the next iteration

    assert provider.subscribed == []
    assert status.info.status is QubxStatus.NORMAL


def test_dark_issues_no_repair(rig):
    """The 11:16 regression: 21 simultaneous subscribes tripped the venue's
    message rate limit and cost the recovery."""
    watchdog, monitor, status, clock, universe, provider = rig
    _subscribe_stale_pair(monitor, clock, universe)

    watchdog.tick()
    watchdog.tick()
    watchdog.tick()

    assert provider.unsubscribed == [] and provider.subscribed == []


def test_partial_never_publishes_maintenance(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)

    for _ in range(5):
        watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert _repairs(provider)[0] == {(eth, "orderbook")}


def test_grace_prevents_false_dark_on_universe_swap(rig):
    """A full universe replacement leaves every instrument with no last-event
    time. Without the grace gate this halts trading on a healthy venue."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)

    watchdog.tick()
    watchdog.tick()
    watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert provider.unsubscribed == [] and provider.subscribed == []


def test_grace_protects_repair_not_just_classification(rig):
    """A freshly subscribed instrument has no last-event time and reads stale to
    is_stale(). In a mixed universe - one genuinely stale, one just subscribed - the
    exchange is PARTIAL and the repair path runs; the new instrument must not be torn
    down on sight. Repair iterates the health monitor's eligible stale keys, which
    exclude grace, rather than re-deriving staleness itself."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)  # btc genuinely stale
    universe[DataType.ORDERBOOK].add(eth)
    monitor.subscribe(eth, DataType.ORDERBOOK)  # eth subscribed just now

    watchdog.tick()

    assert _repairs(provider) == [{(btc, "orderbook")}]
    assert (eth, "orderbook") not in watchdog._exchanges[EXCHANGE].repairs


def test_disconnected_exchange_with_only_unpoliced_types_never_publishes(rig):
    """tick() visits every provider, not just the ones carrying a policed subscription.
    An exchange whose only feed is ohlc has zero eligible instruments, so a dropped
    connection flag alone must not halt trading on it."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc = _instrument("BTCUSDT")
    universe[DataType.OHLC["1h"]] = {btc}
    monitor.subscribe(btc, DataType.OHLC["1h"])
    monitor.set_is_connected(EXCHANGE, lambda: False)
    clock.advance(60)

    for _ in range(3):
        watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert provider.unsubscribed == [] and provider.subscribed == []


def test_disconnected_exchange_with_everything_in_grace_never_publishes(rig):
    """Mid-universe-swap: every instrument is too fresh to judge, so `subscribed` is 0.
    Combined with a down connection flag this used to publish EXCHANGE_MAINTENANCE."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    monitor.set_is_connected(EXCHANGE, lambda: False)

    for _ in range(3):
        watchdog.tick()

    assert status.info.status is QubxStatus.NORMAL
    assert provider.unsubscribed == [] and provider.subscribed == []


def test_disconnected_exchange_with_eligible_instruments_still_publishes(rig):
    """The incident case (21 eligible instruments) is untouched by the eligibility gate."""
    watchdog, monitor, status, clock, universe, _ = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe)
    monitor.set_is_connected(EXCHANGE, lambda: False)
    # - fresh data on both: only limb (A) can classify this DARK
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())

    watchdog.tick()
    assert status.info.status is QubxStatus.NORMAL

    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)


def test_maintenance_scope_is_the_canonical_exchange_not_the_venue_alias():
    """Instruments carry the canonical exchange (BINANCE.UM) and the order path checks
    is_degraded_for(instrument.exchange). A provider registered under the venue alias
    (BINANCE.PM) must publish under the canonical name, or the degradation is accepted,
    stored, and never matched by a single order."""
    watchdog, monitor, status, clock, universe, providers = _make_rig(EXCHANGE, venue_of={EXCHANGE: "BINANCE.PM"})
    btc, eth = _subscribe_stale_pair(monitor, clock, universe[EXCHANGE])
    monitor.set_is_connected("BINANCE.PM", lambda: False)  # the health monitor keys by venue
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())

    watchdog.tick()
    watchdog.tick()

    assert status.info.is_degraded_for("BINANCE.UM")
    assert not status.info.is_degraded_for("BINANCE.PM")


# --- repair mechanics (spec 3) ---


def test_repair_unsubscribes_then_resubscribes_the_full_set(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)

    watchdog.tick()

    assert provider.unsubscribed == [(DataType.ORDERBOOK, {eth})]
    assert provider.subscribed == [(DataType.ORDERBOOK, {btc, eth})]


def test_repair_is_scoped_to_the_exchange(two_venue_rig):
    """The Critical the whole-branch review found: an unscoped subscribe loop re-asserted
    every other venue's universe on each repair - forever, since unverified repairs retry
    forever - reproducing the resubscribe pressure that cost the 11:16 recovery. Every
    single-exchange rig passes whether or not the repair is scoped."""
    watchdog, monitor, status, clock, universe, providers = two_venue_rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe[EXCHANGE], deliver_btc=True)
    other = _instrument("BTCUSDT", OTHER_EXCHANGE)
    universe[OTHER_EXCHANGE][DataType.ORDERBOOK] = {other}
    monitor.subscribe(other, DataType.ORDERBOOK)
    monitor.on_data_arrival(other, DataType.ORDERBOOK, clock.time())

    watchdog.tick()

    assert _repairs(providers[EXCHANGE]) == [{(eth, "orderbook")}]
    assert providers[OTHER_EXCHANGE].unsubscribed == []
    assert providers[OTHER_EXCHANGE].subscribed == []


def test_reassert_is_scoped_to_the_exchange(two_venue_rig):
    """Recovery of one venue re-asserts that venue only. The other was never dark."""
    watchdog, monitor, status, clock, universe, providers = two_venue_rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe[EXCHANGE])
    other = _instrument("BTCUSDT", OTHER_EXCHANGE)
    universe[OTHER_EXCHANGE][DataType.ORDERBOOK] = {other}
    monitor.subscribe(other, DataType.ORDERBOOK)
    monitor.on_data_arrival(other, DataType.ORDERBOOK, clock.time())

    watchdog.tick()
    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert providers[EXCHANGE].subscribed == [(DataType.ORDERBOOK, {btc, eth})]
    assert providers[OTHER_EXCHANGE].subscribed == []


def test_reassert_covers_unpoliced_types_too(rig):
    """The connector may have lost every subscription during the outage, not only the
    policed ones. Recovery re-asserts the whole desired universe for the exchange."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe)
    universe[DataType.OHLC["1h"]] = {btc}
    watchdog.tick()
    watchdog.tick()
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())

    watchdog.tick()

    assert dict(provider.subscribed) == {DataType.ORDERBOOK: {btc, eth}, DataType.OHLC["1h"]: {btc}}


def test_repair_is_scoped_to_the_data_type(rig):
    """A wedged trade feed must not tear down that instrument's orderbook."""
    watchdog, monitor, status, clock, universe, provider = rig
    eth = _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {eth}
    universe[DataType.TRADE] = {eth}
    monitor.subscribe(eth, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(31)  # both past threshold
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())  # orderbook fine, trade wedged

    watchdog.tick()

    assert provider.unsubscribed == [(DataType.TRADE, {eth})]
    assert provider.subscribed == [(DataType.TRADE, {eth})]


def test_repair_subscribes_even_when_unsubscribe_fails(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)
    provider.fail_unsubscribe = ConnectionError("socket closed")

    watchdog.tick()

    assert provider.unsubscribed == []
    assert provider.subscribed == [(DataType.ORDERBOOK, {btc, eth})]


def test_repair_failure_is_retried_next_tick(rig):
    """Raising during repair loses nothing: intent lives above the watchdog."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)
    provider.fail_subscribe = TimeoutError("WebSocket connection not ready after 5.0s")

    watchdog.tick()
    assert provider.subscribed == []

    provider.fail_subscribe = None
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert provider.subscribed == [(DataType.ORDERBOOK, {btc, eth})]


def test_not_supported_is_remembered_and_never_retried(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)
    provider.fail_subscribe = NotSupported("no orderbook here")

    watchdog.tick()
    provider.fail_subscribe = None
    for _ in range(5):
        monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
        watchdog.tick()

    assert len(provider.unsubscribed) == 1  # the first attempt only
    assert provider.subscribed == []
    assert DataType.ORDERBOOK in watchdog._exchanges[EXCHANGE].unsupported


# --- verification and backoff (spec 4.3, D5) ---


def test_verification_is_by_advancement_not_by_staleness(rig):
    """A trade feed that delivers once and goes quiet is not repaired again while
    its silence stays under the staleness threshold. Once its silence genuinely
    exceeds the threshold again, it is correctly repaired again - verification is
    not a permanent exemption, only proof that a specific repair worked."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.TRADE] = {btc, eth}
    monitor.subscribe(btc, DataType.TRADE)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(31)  # trade threshold is 30min
    monitor.on_data_arrival(btc, DataType.TRADE, clock.time())

    watchdog.tick()
    assert _repairs(provider)[-1] == {(eth, "trade")}

    # eth delivers once, then nothing for a long time. btc keeps delivering on every
    # tick so the exchange stays PARTIAL rather than DARK - otherwise D5b's
    # no-repair-when-DARK would suppress repair for a reason unrelated to
    # verification, and the repair path for eth would never actually run.
    monitor.on_data_arrival(eth, DataType.TRADE, clock.time())
    before = len(_repairs(provider))
    for _ in range(29):  # stay strictly under the 30min trade threshold
        clock.advance(1)
        monitor.on_data_arrival(btc, DataType.TRADE, clock.time())
        watchdog.tick()

    assert monitor.is_stale(eth, DataType.TRADE) is False
    assert len(_repairs(provider)) == before  # not repaired again while still under threshold

    # push eth's silence past the threshold: now it is correctly repaired again
    clock.advance(2)
    monitor.on_data_arrival(btc, DataType.TRADE, clock.time())
    watchdog.tick()

    assert monitor.is_stale(eth, DataType.TRADE) is True
    assert len(_repairs(provider)) == before + 1
    assert _repairs(provider)[-1] == {(eth, "trade")}


def test_verification_requires_delivery_after_the_repair(rig):
    """A last_event_time that predates the repair is not evidence the repair
    worked - only a message that arrives after it proves the subscription is live.
    Distinguishes advancement from a plain is_stale/'has ever delivered' check:
    the two agree except in exactly this ordering."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc, eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)  # past the 10min orderbook grace
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())  # eth's only ever message
    clock.advance(11)  # that message is now itself stale - keep btc fresh throughout
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    repairs = watchdog._exchanges[EXCHANGE].repairs

    watchdog.tick()  # eth is genuinely stale (no message in >10min): a real repair
    assert _repairs(provider)[-1] == {(eth, "orderbook")}
    assert (eth, "orderbook") in repairs

    clock.advance(1)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())  # eth still gets nothing
    watchdog.tick()

    # eth's last_event_time (set before the repair) has not moved: not verified.
    assert (eth, "orderbook") in repairs


def test_advancement_pops_record_on_relapse_without_new_repair(rig):
    """The case that actually discriminates advancement from is_stale: a sparse
    feed delivers once after the repair (verifying it), then goes stale again by
    threshold with no intervening tick observing the fresh window. Advancement
    pops the record on relapse and issues 0 new repairs; an is_stale-only rule
    would see "still stale" and issue 1."""
    watchdog, monitor, status, clock, universe, provider = rig
    eth = _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {eth}
    monitor.subscribe(eth, DataType.ORDERBOOK)
    clock.advance(11)  # eth never delivered: genuinely stale

    watchdog.tick()
    assert _repairs(provider) == [{(eth, "orderbook")}]

    clock.advance(1)
    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())  # verifies the repair
    clock.advance(11)  # no tick during the fresh window; now stale again purely by elapse

    watchdog.tick()

    assert _repairs(provider) == [{(eth, "orderbook")}], "advancement must issue 0 new repairs on relapse"
    assert (eth, "orderbook") not in watchdog._exchanges[EXCHANGE].repairs


def test_unverified_repair_backs_off_and_never_stops(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)

    for _ in range(40):
        watchdog.tick()

    attempts = len(_repairs(provider))
    assert 2 <= attempts < 40, "repairs must back off but never stop"


def test_backoff_resets_after_recovery(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)
    for _ in range(6):
        watchdog.tick()

    monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())
    watchdog.tick()
    assert watchdog._exchanges[EXCHANGE].repairs == {}


def test_repair_never_calls_health_monitor_unsubscribe(rig):
    """Popping _last_event_time would destroy the verification signal."""
    watchdog, monitor, status, clock, universe, _ = rig
    _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)
    monitor.unsubscribe = Mock(side_effect=AssertionError("repair must not unsubscribe in health"))

    for _ in range(5):
        watchdog.tick()


def test_repair_does_not_contaminate_other_subscription_types(rig):
    """An instrument due on one type must not get a bogus record stamped (and a false
    "repairing" log line) on every other type it is also subscribed to."""
    watchdog, monitor, status, clock, universe, provider = rig
    eth = _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {eth}
    universe[DataType.TRADE] = {eth}
    monitor.subscribe(eth, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.TRADE)
    clock.advance(11)  # orderbook (10min threshold) now stale
    monitor.on_data_arrival(eth, DataType.TRADE, clock.time())  # trade freshly delivering

    watchdog.tick()

    repairs = watchdog._exchanges[EXCHANGE].repairs
    assert (eth, "orderbook") in repairs
    assert (eth, "trade") not in repairs
    assert _repairs(provider) == [{(eth, "orderbook")}]


def test_repaired_at_is_stamped_after_the_repair_returns(rig):
    """The repair sleeps at least the settle delay before it returns. Stamping
    repaired_at before it lets a message that was already in flight when the repair
    started count as proof the repair worked - so the backoff resets every tick and a
    permanently wedged instrument never escalates to ERROR."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)

    def in_flight_message_during_repair() -> None:
        clock.advance(1)
        monitor.on_data_arrival(eth, DataType.ORDERBOOK, clock.time())  # in flight pre-unsubscribe
        clock.advance(3)

    provider.on_unsubscribe = in_flight_message_during_repair
    watchdog.tick()
    repairs = watchdog._exchanges[EXCHANGE].repairs
    assert repairs[(eth, "orderbook")].repaired_at == clock.time()

    provider.on_unsubscribe = None
    clock.advance(11)  # eth silent since that mid-repair message: stale again
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    record = repairs.get((eth, "orderbook"))
    assert record is not None, "a message predating the repair must not verify it"
    assert record.interval_ticks == 2, "the backoff must escalate rather than reset"


def test_repair_state_pruned_when_instrument_leaves_universe(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, deliver_btc=True)
    repairs = watchdog._exchanges[EXCHANGE].repairs

    watchdog.tick()
    assert (eth, "orderbook") in repairs

    universe[DataType.ORDERBOOK] = {btc}  # eth unsubscribed at the manager level
    clock.advance(1)
    monitor.on_data_arrival(btc, DataType.ORDERBOOK, clock.time())
    watchdog.tick()

    assert (eth, "orderbook") not in repairs


# --- parameterised subscription keys and the unpoliced-type filter (spec 4.1/4.2) ---


def test_repair_reaches_the_provider_with_parameterised_subscription_key(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _subscribe_stale_pair(monitor, clock, universe, sub=PARAM_ORDERBOOK, deliver_btc=True)

    watchdog.tick()

    assert provider.unsubscribed == [(PARAM_ORDERBOOK, {eth})]
    assert provider.subscribed == [(PARAM_ORDERBOOK, {btc, eth})]


def test_all_stale_classifies_dark_with_parameterised_subscription_key(rig):
    watchdog, monitor, status, clock, universe, _ = rig
    _subscribe_stale_pair(monitor, clock, universe, sub=PARAM_ORDERBOOK)

    watchdog.tick()
    assert status.info.status is QubxStatus.NORMAL  # one dark tick, below the maintenance threshold

    watchdog.tick()
    assert status.info.is_degraded_for(EXCHANGE)
    assert status.info.degradations[0].reason is DegradeReason.EXCHANGE_MAINTENANCE


def test_unpoliced_types_are_excluded_from_repair_and_classification(rig):
    """An ohlc instrument must never be repaired and must never be counted toward the
    exchange's subscribed/stale ratio."""
    watchdog, monitor, status, clock, universe, provider = rig
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    universe[DataType.ORDERBOOK] = {btc}
    universe[DataType.OHLC["1h"]] = {eth}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    monitor.subscribe(eth, DataType.OHLC["1h"])
    clock.advance(11)  # past the orderbook threshold; ohlc has no STALE_THRESHOLDS entry at all

    watchdog.tick()

    assert _repairs(provider) == [{(btc, "orderbook")}]
    assert (eth, "ohlc") not in watchdog._exchanges[EXCHANGE].repairs
    assert status.info.status is QubxStatus.NORMAL  # a single-instrument orderbook exchange is PARTIAL, not DARK


# --- gating ---


def test_does_nothing_before_warmup_finished(rig):
    watchdog, monitor, status, clock, universe, provider = rig
    watchdog._strategy_state.is_on_warmup_finished_called = False
    btc = _instrument("BTCUSDT")
    universe[DataType.ORDERBOOK] = {btc}
    monitor.subscribe(btc, DataType.ORDERBOOK)
    clock.advance(11)

    watchdog.tick()
    watchdog.tick()

    assert provider.unsubscribed == [] and provider.subscribed == []
    assert status.info.status is QubxStatus.NORMAL


def test_simulation_providers_are_never_watched():
    sim = Mock()
    sim.is_simulation = True
    sim.exchange.return_value = EXCHANGE
    watchdog = SubscriptionWatchdog(
        data_providers=[sim],
        health_monitor=BaseHealthMonitor(FixedTime()),
        status=ContextStatus(),
        snapshot_fn=dict,
        strategy_state=StrategyState(),
    )
    assert watchdog._exchanges == {}


def test_tick_survives_an_exception(rig):
    watchdog, *_ = rig
    watchdog._snapshot_fn = Mock(side_effect=RuntimeError("boom"))
    watchdog.tick()  # must not raise


# --- thread lifecycle ---


def test_start_stop_start_leaves_a_live_ticking_thread(rig):
    """stop() must clear the stop event and join the thread; otherwise a later
    start() spawns a thread whose wait() returns immediately (the event is still
    set) and tick() never runs - a silently dead watchdog, which is exactly the
    22-hour failure this module exists to prevent."""
    watchdog, *_ = rig
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


def test_stop_does_not_block_on_a_wedged_tick(rig, monkeypatch):
    """StrategyContext.stop() calls this AFTER the data providers are closed, so an
    in-flight tick can be stuck in the settle sleep or blocking against a closed
    provider. The thread is a daemon: abandoning it beats hanging teardown forever."""
    watchdog, *_ = rig
    monkeypatch.setattr(watchdog_module, "_STOP_JOIN_TIMEOUT_SECONDS", 0.2)
    in_tick = threading.Event()
    release = threading.Event()

    def wedged_tick():
        in_tick.set()
        release.wait(timeout=10)

    watchdog._interval_seconds = 0.01
    watchdog.tick = wedged_tick
    watchdog.start()
    thread = watchdog._thread
    assert in_tick.wait(timeout=5)

    started = time.monotonic()
    watchdog.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"stop() blocked on the in-flight tick for {elapsed:.1f}s"
    assert thread is not None and thread.is_alive()  # abandoned, not joined
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_stop_from_inside_the_watchdog_thread_does_not_raise(rig):
    """join()ing the current thread raises RuntimeError; the loop exits on the set event
    anyway."""
    watchdog, *_ = rig
    errors: list[BaseException] = []
    captured: list[threading.Thread] = []
    done = threading.Event()

    def tick_that_stops():
        captured.append(threading.current_thread())
        try:
            watchdog.stop()
        except BaseException as e:  # pragma: no cover - asserted below
            errors.append(e)
        finally:
            done.set()

    watchdog._interval_seconds = 0.01
    watchdog.tick = tick_that_stops
    watchdog.start()

    assert done.wait(timeout=5)
    assert not errors, f"stop() from the watchdog thread raised: {errors!r}"
    captured[0].join(timeout=5)
    assert not captured[0].is_alive()
