"""Task 6 scope extension: StrategyContext wires a SafeStatePersistence (duck-typed via
last_success_age) into the health monitor at construction, and flushes it (stop()) during
StrategyContext.stop()'s fault-tolerant cleanup sequence. DummyStatePersistence (no
last_success_age/stop) must be a no-op on both paths.

Mirrors the StrategyContext construction pattern from context_initializer_test.py — the
same set of manager classes (MarketManager/UniverseManager/SubscriptionManager/
TradingManager/ProcessingManager) is patched out so construction stays cheap.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.context import StrategyContext
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategy
from qubx.core.lookups import lookup


class _MockStrategy(IStrategy):
    """Bare strategy — relies on IStrategy's default no-op on_init/on_stop."""


class _StubSafePersistence:
    """Duck-types the SafeStatePersistence surface context.py wires on: last_success_age()
    + staleness_threshold_s (health wiring) and stop() (shutdown flush)."""

    staleness_threshold_s = 60.0

    def __init__(self) -> None:
        self.stop_called = False

    def last_success_age(self) -> float | None:
        return None

    def stop(self) -> None:
        self.stop_called = True


@pytest.fixture
def mock_components():
    data_provider = MagicMock()
    time_provider = MagicMock()
    time_provider.time.return_value = np.datetime64("2023-01-01", "ns")
    return {
        "connectors": {"BINANCE.UM": MagicMock()},
        "data_provider": data_provider,
        "account": MagicMock(),
        "scheduler": MagicMock(),
        "time_provider": time_provider,
        "instruments": [lookup.find_symbol("BINANCE.UM", "BTCUSDT")],
        "logging": MagicMock(),
        "aux_data_storage": MagicMock(),
    }


def _build_context(mock_components, state_persistence, health_monitor) -> StrategyContext:
    with (
        patch("qubx.core.context.MarketManager"),
        patch("qubx.core.context.UniverseManager"),
        patch("qubx.core.context.SubscriptionManager"),
        patch("qubx.core.context.TradingManager"),
        patch("qubx.core.context.ProcessingManager"),
    ):
        return StrategyContext(
            strategy=_MockStrategy(),
            connectors=mock_components["connectors"],
            data_providers=[mock_components["data_provider"]],
            account_manager=mock_components["account"],
            scheduler=mock_components["scheduler"],
            time_provider=mock_components["time_provider"],
            instruments=mock_components["instruments"],
            logging=mock_components["logging"],
            initializer=BasicStrategyInitializer(simulation=True),
            aux_data_storage=mock_components["aux_data_storage"],
            state_persistence=state_persistence,
            health_monitor=health_monitor,
        )


def test_safe_persistence_wired_into_health_monitor_on_construction(mock_components):
    sp = _StubSafePersistence()
    health_monitor = MagicMock()
    _build_context(mock_components, sp, health_monitor)
    health_monitor.set_state_persistence.assert_called_once_with(sp)


def test_dummy_persistence_not_wired_into_health_monitor(mock_components):
    health_monitor = MagicMock()
    _build_context(mock_components, None, health_monitor)  # None -> DummyStatePersistence()
    health_monitor.set_state_persistence.assert_not_called()


def test_stop_flushes_safe_state_persistence(mock_components):
    sp = _StubSafePersistence()
    ctx = _build_context(mock_components, sp, MagicMock())
    ctx.stop()
    assert sp.stop_called is True


def test_stop_is_noop_for_dummy_state_persistence(mock_components):
    ctx = _build_context(mock_components, None, MagicMock())  # None -> DummyStatePersistence()
    ctx.stop()  # must not raise (DummyStatePersistence has no stop())


class _HealthMonitorWithoutStatePersistenceSupport:
    """A custom IHealthMonitor that predates set_state_persistence — must not AttributeError
    at construction when paired with real (SafeStatePersistence-shaped) persistence."""

    def set_status(self, status) -> None:
        pass


def test_construction_does_not_raise_for_health_monitor_missing_set_state_persistence(mock_components):
    sp = _StubSafePersistence()
    # must not raise AttributeError even though the monitor has no set_state_persistence
    _build_context(mock_components, sp, _HealthMonitorWithoutStatePersistenceSupport())
