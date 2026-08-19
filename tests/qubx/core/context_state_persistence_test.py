"""StrategyContext wires its state persistence into the health monitor at construction
(set_state_persistence is part of IHealthMonitor) and flushes it (stop(), part of
IStatePersistence) during StrategyContext.stop()'s fault-tolerant cleanup sequence.
Backends that don't track write health (DummyStatePersistence) rely on the interface
defaults: last_success_age() -> None and a no-op stop().

Mirrors the StrategyContext construction pattern from context_initializer_test.py — the
same set of manager classes (MarketManager/UniverseManager/SubscriptionManager/
TradingManager/ProcessingManager) is patched out so construction stays cheap.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.context import StrategyContext
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IHealthMonitor, IStrategy
from qubx.core.lookups import lookup
from qubx.state import DummyStatePersistence


class _MockStrategy(IStrategy):
    """Bare strategy — relies on IStrategy's default no-op on_init/on_stop."""


class _StubSafePersistence:
    """The SafeStatePersistence surface context.py relies on: last_success_age()
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


def test_dummy_persistence_is_wired_unconditionally(mock_components):
    health_monitor = MagicMock()
    ctx = _build_context(mock_components, None, health_monitor)  # None -> DummyStatePersistence()
    health_monitor.set_state_persistence.assert_called_once_with(ctx._state_persistence)
    assert isinstance(ctx._state_persistence, DummyStatePersistence)


def test_stop_flushes_safe_state_persistence(mock_components):
    sp = _StubSafePersistence()
    ctx = _build_context(mock_components, sp, MagicMock())
    ctx.stop()
    assert sp.stop_called is True


def test_stop_is_noop_for_dummy_state_persistence(mock_components):
    ctx = _build_context(mock_components, None, MagicMock())  # None -> DummyStatePersistence()
    ctx.stop()  # must not raise (DummyStatePersistence inherits the no-op stop() default)


class _MinimalHealthMonitor(IHealthMonitor):
    """A monitor that implements nothing beyond what IHealthMonitor's interface
    defaults provide — set_state_persistence must be inherited as a no-op."""

    def set_status(self, status) -> None:
        pass


def test_construction_works_for_monitor_using_interface_default(mock_components):
    sp = _StubSafePersistence()
    # must not raise: set_state_persistence is part of IHealthMonitor with a no-op default
    _build_context(mock_components, sp, _MinimalHealthMonitor())
