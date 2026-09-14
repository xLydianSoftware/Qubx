from unittest.mock import Mock

import pytest

from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.mixins.subscription import SubscriptionManager
from qubx.core.status import ContextStatus
from qubx.health.dummy import DummyHealthMonitor

EXCHANGE = "BINANCE.UM"


def _instrument(symbol: str) -> Instrument:
    instr = lookup.find_symbol(EXCHANGE, symbol)
    assert instr is not None
    return instr


@pytest.fixture
def manager_and_provider():
    provider = Mock()
    provider.is_simulation = False
    provider.exchange.return_value = EXCHANGE
    provider.get_subscribed_instruments.return_value = []
    provider.get_subscriptions.return_value = []
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    manager = SubscriptionManager(
        time_provider, [provider], CtrlChannel("test"), DummyHealthMonitor(), StrategyState(), ContextStatus()
    )
    return manager, provider


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


def test_not_supported_is_recorded_once(manager_and_provider):
    from qubx.core.exceptions import NotSupported

    manager, provider = manager_and_provider
    provider.subscribe.side_effect = NotSupported("no orderbook here")

    manager.subscribe(DataType.ORDERBOOK, _instrument("BTCUSDT"))
    manager.commit()

    assert (EXCHANGE, DataType.ORDERBOOK) in manager._unsupported


def test_reconcile_refresh_unsubscribes_then_resubscribes_full_set(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    manager.subscribe(DataType.ORDERBOOK, [btc, eth])
    manager.commit()
    provider.reset_mock()

    manager.reconcile(refresh={btc})

    provider.unsubscribe.assert_called_once_with(DataType.ORDERBOOK, {btc})
    provider.subscribe.assert_called_once_with(DataType.ORDERBOOK, {btc, eth}, reset=True)


def test_reconcile_without_refresh_reasserts_and_never_unsubscribes(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()

    manager.reconcile()

    provider.unsubscribe.assert_not_called()
    provider.subscribe.assert_called_once_with(DataType.ORDERBOOK, {btc}, reset=True)


def test_reconcile_failure_leaves_desired_intact_and_retries(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()
    provider.subscribe.side_effect = TimeoutError("WebSocket connection not ready after 5.0s")

    manager.reconcile(refresh={btc})

    assert manager._desired[DataType.ORDERBOOK] == {btc}

    provider.subscribe.side_effect = None
    manager.reconcile(refresh={btc})
    provider.subscribe.assert_called_with(DataType.ORDERBOOK, {btc}, reset=True)


def test_reconcile_skips_unsupported_pairs(manager_and_provider):
    from qubx.core.exceptions import NotSupported

    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    provider.subscribe.side_effect = NotSupported("nope")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()
    provider.subscribe.side_effect = None

    manager.reconcile(refresh={btc})

    provider.subscribe.assert_not_called()
    provider.unsubscribe.assert_not_called()


def test_reconcile_isolates_a_failing_exchange(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.subscribe(DataType.TRADE, btc)
    manager.commit()
    provider.reset_mock()

    calls: list[str] = []

    def record(sub, instruments, reset=False):
        calls.append(sub)
        if sub == DataType.ORDERBOOK:
            raise TimeoutError("boom")

    provider.subscribe.side_effect = record
    manager.reconcile()

    assert DataType.ORDERBOOK in calls and DataType.TRADE in calls


def test_reconcile_subscribes_even_when_unsubscribe_fails(manager_and_provider):
    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.0
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()
    provider.unsubscribe.side_effect = TimeoutError("boom")

    manager.reconcile(refresh={btc})

    provider.unsubscribe.assert_called_once_with(DataType.ORDERBOOK, {btc})
    provider.subscribe.assert_called_once_with(DataType.ORDERBOOK, {btc}, reset=True)


def test_concurrent_apply_swap_during_reconcile_does_not_corrupt_desired(manager_and_provider):
    import threading
    import time as time_module

    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.05
    btc, eth = _instrument("BTCUSDT"), _instrument("ETHUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()
    provider.reset_mock()

    errors: list[Exception] = []

    def do_reconcile():
        try:
            manager.reconcile(refresh={btc})
        except Exception as e:  # pragma: no cover - failure path asserted below
            errors.append(e)

    reconciler = threading.Thread(target=do_reconcile)
    reconciler.start()
    # give reconcile time to snapshot and enter its settle sleep, so this commit's
    # _apply_swap lands while reconcile is doing I/O with the lock released
    time_module.sleep(0.01)
    manager.subscribe(DataType.ORDERBOOK, eth)
    manager.commit()
    reconciler.join(timeout=2)

    assert not reconciler.is_alive()
    assert not errors
    assert manager._desired[DataType.ORDERBOOK] == {btc, eth}


def test_concurrent_deferred_apply_swap_during_reconcile_does_not_corrupt_desired(manager_and_provider):
    """Reproduces the actual pre-fix race behind Finding 1: the ProcessorThread reaches
    _apply_swap via _apply_deferred_swap (no lock of its own before the fix), concurrently
    with reconcile() running on the watchdog thread. Drives _apply_deferred_swap directly
    rather than standing up the WarmupThread/channel machinery, since that IS the call the
    channel dispatcher makes.
    """
    import threading
    import time as time_module

    from qubx.core.mixins.subscription import _CommitPlan

    manager, provider = manager_and_provider
    manager._repair_settle_seconds = 0.02

    btc, eth, sol = _instrument("BTCUSDT"), _instrument("ETHUSDT"), _instrument("SOLUSDT")
    subs = [DataType.ORDERBOOK, DataType.TRADE, DataType.QUOTE, DataType.LIQUIDATION, DataType.OPEN_INTEREST]
    for sub in subs:
        manager.subscribe(sub, [btc, eth, sol])
    manager.commit()
    provider.reset_mock()

    # spy on _apply_swap (an instance attribute shadows the decorated class method for
    # lookups via self._apply_swap) so an exception raised inside it is observed here
    # even though _apply_deferred_swap swallows it by design before it reaches the
    # thread that called _apply_deferred_swap
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

    reconcile_errors: list[BaseException] = []
    try:
        # let the hammer thread get going before reconcile takes its snapshot, so the
        # snapshot and the settle sleep both land inside the hammering window
        time_module.sleep(0.005)
        manager.reconcile(refresh={btc, eth, sol})
    except BaseException as e:
        reconcile_errors.append(e)
    finally:
        stop.set()
        hammer_thread.join(timeout=5)

    assert not hammer_thread.is_alive()
    assert not reconcile_errors, f"reconcile() raised: {reconcile_errors!r}"
    assert not apply_swap_errors, f"_apply_swap raised under concurrent reconcile: {apply_swap_errors!r}"

    # internal consistency: the hammer plan only ever adds/removes btc, so eth/sol must
    # survive untouched in every sub, and every sub's value must still be a well-formed set
    for sub in subs:
        instruments = manager._desired.get(sub, set())
        assert isinstance(instruments, set)
        assert instruments <= {btc, eth, sol}
        assert {eth, sol} <= instruments


def test_apply_swap_is_gated_by_the_shared_instance_lock(manager_and_provider):
    """Deterministic guard for Finding 1: _apply_swap must acquire the same instance lock
    @synchronized uses elsewhere, so the ProcessorThread-side deferred writer
    (_apply_deferred_swap) can never proceed while another synchronized method (commit,
    subscribe, or reconcile's snapshot) is mid-flight. This checks the exclusion property
    directly rather than trying to catch the resulting corruption under the GIL's atomic
    dict/set ops (see test_concurrent_deferred_apply_swap_during_reconcile_does_not_corrupt_desired's
    report note: that race is not reliably observable this way), so removing @synchronized
    from _apply_swap fails this test every time, not just under the right interleaving.
    """
    import threading

    from qubx.core.mixins.subscription import _CommitPlan

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
        # rather than proceed
        assert not call_returned.wait(timeout=0.2)
    finally:
        release_holder.set()
        holder.join(timeout=2)

    assert call_returned.wait(timeout=2)
    caller.join(timeout=2)


def test_watchdog_is_started_for_live_and_absent_in_simulation():
    from qubx.core.subscription_watchdog import SubscriptionWatchdog

    live = Mock()
    live.is_simulation = False
    live.exchange.return_value = EXCHANGE
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    manager = SubscriptionManager(
        time_provider, [live], CtrlChannel("test"), DummyHealthMonitor(), StrategyState(), ContextStatus()
    )
    assert isinstance(manager._watchdog, SubscriptionWatchdog)

    sim = Mock()
    sim.is_simulation = True
    sim.exchange.return_value = EXCHANGE
    sim_manager = SubscriptionManager(
        time_provider, [sim], CtrlChannel("test"), DummyHealthMonitor(), StrategyState(), ContextStatus()
    )
    assert sim_manager._watchdog is None


def test_watchdog_sees_the_desired_universe():
    live = Mock()
    live.is_simulation = False
    live.exchange.return_value = EXCHANGE
    live.get_subscribed_instruments.return_value = []
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    manager = SubscriptionManager(
        time_provider, [live], CtrlChannel("test"), DummyHealthMonitor(), StrategyState(), ContextStatus()
    )
    btc = _instrument("BTCUSDT")
    manager.subscribe(DataType.ORDERBOOK, btc)
    manager.commit()

    assert manager._watchdog._subscriptions_fn() == {DataType.ORDERBOOK: {btc}}


def test_old_monitor_entry_points_are_gone():
    assert not hasattr(SubscriptionManager, "_monitor_subscription_status")
    assert not hasattr(SubscriptionManager, "_monitor_loop")


def test_watchdog_publishes_into_the_caller_s_status_object():
    """A defaulted ContextStatus would let the watchdog degrade an object the order
    path never reads — visible in the status, inert in trading."""
    live = Mock()
    live.is_simulation = False
    live.exchange.return_value = EXCHANGE
    time_provider = Mock()
    time_provider.time.return_value = 0.0
    status = ContextStatus()
    manager = SubscriptionManager(
        time_provider, [live], CtrlChannel("test"), DummyHealthMonitor(), StrategyState(), status
    )

    assert manager._watchdog._status is status


def test_status_is_a_required_argument():
    import inspect

    parameter = inspect.signature(SubscriptionManager.__init__).parameters["status"]
    assert parameter.default is inspect.Parameter.empty
