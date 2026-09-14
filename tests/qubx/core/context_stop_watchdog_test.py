"""StrategyContext.stop() must join the subscription watchdog thread. Before this test
existed, SubscriptionManager had no shutdown hook at all: the watchdog (a daemon thread)
outlived every StrategyContext that created it. Process exit made that harmless, but a
context recreated in-process (e.g. a strategy restart within the same process) would leave
a stale thread ticking - still holding the dead context's ContextStatus and a reconcile()
bound to the abandoned SubscriptionManager - able to publish EXCHANGE_MAINTENANCE into a
status object nothing reads anymore.

Deliberately does NOT patch out SubscriptionManager (unlike context_state_persistence_test.py's
helper): the whole point is a REAL watchdog thread that can be observed alive, then observed
gone. The watchdog's default 30s tick interval means tick() itself never fires in this test's
lifetime, so the mocked data_provider/health_monitor never need to support a real tick.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.basics import CtrlChannel
from qubx.core.context import StrategyContext
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategy
from qubx.core.lookups import lookup
from qubx.core.subscription_watchdog import SubscriptionWatchdog
from qubx.health.dummy import DummyHealthMonitor


class _MockStrategy(IStrategy):
    """Bare strategy — relies on IStrategy's default no-op on_init/on_stop."""


@pytest.fixture
def mock_components():
    channel = CtrlChannel("test")
    data_provider = MagicMock()
    data_provider.channel = channel
    data_provider.is_simulation = False  # forces the live watchdog path
    data_provider.exchange.return_value = "BINANCE.UM"
    connector = MagicMock()
    connector.channel = channel
    time_provider = MagicMock()
    time_provider.time.return_value = np.datetime64("2023-01-01", "ns")
    return {
        "connectors": {"BINANCE.UM": connector},
        "data_provider": data_provider,
        "channel": channel,
        "account": MagicMock(),
        "scheduler": MagicMock(),
        "time_provider": time_provider,
        "instruments": [lookup.find_symbol("BINANCE.UM", "BTCUSDT")],
        "logging": MagicMock(),
        "aux_data_storage": MagicMock(),
    }


def _build_context(mock_components) -> StrategyContext:
    with (
        patch("qubx.core.context.MarketManager"),
        patch("qubx.core.context.UniverseManager"),
        patch("qubx.core.context.TradingManager"),
        patch("qubx.core.context.ProcessingManager"),
    ):
        return StrategyContext(
            strategy=_MockStrategy(),
            connectors=mock_components["connectors"],
            data_providers=[mock_components["data_provider"]],
            account_manager=mock_components["account"],
            scheduler=mock_components["scheduler"],
            channel=mock_components["channel"],
            time_provider=mock_components["time_provider"],
            instruments=mock_components["instruments"],
            logging=mock_components["logging"],
            initializer=BasicStrategyInitializer(simulation=True),
            aux_data_storage=mock_components["aux_data_storage"],
            health_monitor=DummyHealthMonitor(),
        )


def test_stop_leaves_no_live_watchdog_thread(mock_components):
    ctx = _build_context(mock_components)

    watchdog = ctx._subscription_manager._watchdog
    assert isinstance(watchdog, SubscriptionWatchdog)
    assert watchdog._thread is not None and watchdog._thread.is_alive()

    ctx.stop()

    assert watchdog._thread is None, "ctx.stop() must join the watchdog thread, not just drop the handle"
