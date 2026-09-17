import inspect
import threading
import time
from unittest.mock import Mock

import pytest

from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.mixins.subscription import SubscriptionManager, _CommitPlan
from qubx.core.status import ContextStatus
from qubx.core.subscription_watchdog import SubscriptionWatchdog
from qubx.health.dummy import DummyHealthMonitor

EXCHANGE = "BINANCE.UM"
OTHER_EXCHANGE = "BITFINEX.F"


def _instrument(symbol: str, exchange: str = EXCHANGE) -> Instrument:
    instr = lookup.find_symbol(exchange, symbol)
    assert instr is not None
    return instr


def _provider(exchange: str = EXCHANGE, simulation: bool = False) -> Mock:
    provider = Mock()
    provider.is_simulation = simulation
    provider.exchange.return_value = exchange
    provider.get_subscribed_instruments.return_value = []
    provider.get_subscriptions.return_value = []
    return provider


def _manager(*providers: Mock, status: ContextStatus | None = None, **kwargs) -> SubscriptionManager:
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    return SubscriptionManager(
        time_provider,
        list(providers),
        CtrlChannel("test"),
        DummyHealthMonitor(),
        StrategyState(),
        status if status is not None else ContextStatus(),
        **kwargs,
    )


@pytest.fixture
def manager_and_provider():
    provider = _provider()
    return _manager(provider), provider


# --- intent ownership (spec 2) ---


def test_intent_survives_provider_losing_its_registry(manager_and_provider):
    """The 2026-09-13 incident, reduced: the provider forgets everything and the
    manager must still know what the universe is."""
    manager, provider = manager_and_provider
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")

    manager.subscribe(DataType.ORDERBOOK, [btc, eth])
    manager.commit()

    # provider wipes its own bookkeeping, as LighterDataProvider.unsubscribe did
    provider.get_subscribed_instruments.return_value = []
    provider.get_subscriptions.return_value = []

    assert set(manager.get_subscribed_instruments(DataType.ORDERBOOK)) == {btc, eth}
    assert manager.has_subscription(btc, DataType.ORDERBOOK)


def test_desired_tracks_adds_and_removes(manager_and_provider):
    manager, provider = manager_and_provider
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")

    manager.subscribe(DataType.ORDERBOOK, [btc, eth])
    manager.commit()
    assert manager._desired[DataType.ORDERBOOK] == {btc, eth}

    manager.unsubscribe(DataType.ORDERBOOK, eth)
    manager.commit()
    assert manager._desired[DataType.ORDERBOOK] == {btc}
    assert not manager.has_subscription(eth, DataType.ORDERBOOK)


def test_desired_snapshot_is_keyed_by_exchange_and_is_a_copy():
    """The watchdog's only view of intent: scoped by exchange so it never has to filter,
    and detached from the live sets so mutation on the ProcessorThread cannot reach it."""
    manager = _manager(_provider(EXCHANGE), _provider(OTHER_EXCHANGE))
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    other = _instrument("BTCUSDT", OTHER_EXCHANGE)
    manager.subscribe(DataType.ORDERBOOK, [btc, eth, other])
    manager.subscribe(DataType.TRADE, btc)
    manager.commit()

    snapshot = manager.desired_snapshot()

    assert snapshot == {
        EXCHANGE: {DataType.ORDERBOOK: frozenset({btc, eth}), DataType.TRADE: frozenset({btc})},
        OTHER_EXCHANGE: {DataType.ORDERBOOK: frozenset({other})},
    }
    manager.unsubscribe(DataType.ORDERBOOK, eth)
    manager.commit()
    assert eth in snapshot[EXCHANGE][DataType.ORDERBOOK]  # the copy is unaffected


# --- the shared instance lock (spec 1) ---


def test_concurrent_apply_swap_during_snapshot_does_not_corrupt_desired(manager_and_provider):
    manager, provider = manager_and_provider
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()

    errors: list[Exception] = []
    stop = threading.Event()

    def snapshot_loop():
        try:
            while not stop.is_set():
                manager.desired_snapshot()
        except Exception as e:  # pragma: no cover - failure path asserted below
            errors.append(e)

    reader = threading.Thread(target=snapshot_loop)
    reader.start()
    time.sleep(0.01)
    manager.subscribe(DataType.ORDERBOOK, eth)
    manager.commit()
    stop.set()
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert not errors
    assert manager._desired[DataType.ORDERBOOK] == {btc, eth}


def test_concurrent_deferred_apply_swap_during_snapshot_does_not_corrupt_desired(manager_and_provider):
    """The ProcessorThread reaches _apply_swap via _apply_deferred_swap (which had no lock
    of its own before this branch), concurrently with the watchdog reading a snapshot.
    Drives _apply_deferred_swap directly rather than standing up the WarmupThread/channel
    machinery, since that IS the call the channel dispatcher makes."""
    manager, provider = manager_and_provider
    btc, eth, sol = _instrument("BTCUSDT"), _instrument("ETHUSDT"), _instrument("SOLUSDT")
    subs = [DataType.ORDERBOOK, DataType.TRADE, DataType.QUOTE, DataType.LIQUIDATION, DataType.OPEN_INTEREST]
    for sub in subs:
        manager.subscribe(sub, [btc, eth, sol])
    manager.commit()

    apply_swap_errors: list[BaseException] = []
    original_apply_swap = manager._apply_swap

    def spying_apply_swap(plan):
        try:
            return original_apply_swap(plan)
        except BaseException as e:
            apply_swap_errors.append(e)
            raise

    manager._apply_swap = spying_apply_swap
    stop = threading.Event()

    def hammer() -> None:
        toggle = True
        while not stop.is_set():
            plan = (
                _CommitPlan(stream_subscriptions={sub: {btc} for sub in subs})
                if toggle
                else _CommitPlan(stream_unsubscriptions={sub: {btc} for sub in subs})
            )
            toggle = not toggle
            with manager._warmup_inflight_lock:
                manager._warmup_inflight += 1
            manager._apply_deferred_swap(plan)

    hammer_thread = threading.Thread(target=hammer)
    hammer_thread.start()

    snapshot_errors: list[BaseException] = []
    try:
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            manager.desired_snapshot()
    except BaseException as e:
        snapshot_errors.append(e)
    finally:
        stop.set()
        hammer_thread.join(timeout=5)

    assert not hammer_thread.is_alive()
    assert not snapshot_errors, f"desired_snapshot() raised: {snapshot_errors!r}"
    assert not apply_swap_errors, f"_apply_swap raised under concurrent snapshot: {apply_swap_errors!r}"
    # - the hammer plan only ever adds/removes btc, so eth/sol must survive untouched
    for sub in subs:
        instruments = manager._desired.get(sub, set())
        assert isinstance(instruments, set)
        assert {eth, sol} <= instruments <= {btc, eth, sol}


def test_apply_swap_is_gated_by_the_shared_instance_lock(manager_and_provider):
    """Deterministic guard: _apply_swap must acquire the same instance lock @synchronized
    uses elsewhere, so the ProcessorThread-side deferred writer (_apply_deferred_swap) can
    never proceed while another synchronized method (commit, subscribe, desired_snapshot)
    is mid-flight. Checks the exclusion property directly rather than trying to catch the
    resulting corruption under the GIL's atomic dict/set ops, so removing @synchronized
    from _apply_swap fails this test every time, not just under the right interleaving."""
    manager, provider = manager_and_provider
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()  # exercises a @synchronized call so manager._synchronized_lock exists

    plan = _CommitPlan(stream_subscriptions={DataType.TRADE: {btc}})
    holder_acquired = threading.Event()
    release_holder = threading.Event()

    def hold_lock() -> None:
        with manager._synchronized_lock:
            holder_acquired.set()
            release_holder.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holder_acquired.wait(timeout=2)

    call_returned = threading.Event()

    def call_apply_swap() -> None:
        manager._apply_swap(plan)
        call_returned.set()

    caller = threading.Thread(target=call_apply_swap)
    caller.start()
    try:
        # the instance lock is held elsewhere: a synchronized _apply_swap must block
        assert not call_returned.wait(timeout=0.2)
    finally:
        release_holder.set()
        holder.join(timeout=2)

    assert call_returned.wait(timeout=2)
    caller.join(timeout=2)


# --- watchdog wiring (spec 4/6) ---


def test_watchdog_is_started_for_live_and_absent_in_simulation():
    assert isinstance(_manager(_provider())._watchdog, SubscriptionWatchdog)
    assert _manager(_provider(simulation=True))._watchdog is None


def test_watchdog_reads_the_desired_snapshot():
    manager = _manager(_provider())
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()

    assert manager._watchdog._snapshot_fn() == {EXCHANGE: {DataType.ORDERBOOK: frozenset({btc})}}


def test_old_monitor_entry_points_are_gone():
    assert not hasattr(SubscriptionManager, "_monitor_subscription_status")
    assert not hasattr(SubscriptionManager, "_monitor_loop")
    assert not hasattr(SubscriptionManager, "reconcile")


def test_watchdog_publishes_into_the_caller_s_status_object():
    """A defaulted ContextStatus would let the watchdog degrade an object the order
    path never reads - visible in the status, inert in trading."""
    status = ContextStatus()
    manager = _manager(_provider(), status=status)

    assert manager._watchdog._status is status


def test_status_is_a_required_argument():
    parameter = inspect.signature(SubscriptionManager.__init__).parameters["status"]
    assert parameter.default is inspect.Parameter.empty


def test_stop_joins_the_watchdog_thread():
    manager = _manager(_provider(), monitor_interval_seconds=0.02)  # real wall-clock ticks
    assert manager._watchdog._thread is not None
    assert manager._watchdog._thread.is_alive()

    manager.stop()

    assert manager._watchdog._thread is None


def test_stop_is_a_noop_in_simulation():
    manager = _manager(_provider(simulation=True))
    assert manager._watchdog is None
    manager.stop()  # must not raise
