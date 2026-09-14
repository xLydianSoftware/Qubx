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
