"""Tests for ExchangeManager functionality."""

from unittest.mock import Mock, patch

from qubx.connectors.ccxt.exchange_manager import ExchangeManager
from qubx.core.basics import Instrument, LiveTimeProvider
from qubx.core.lookups import lookup
from qubx.health.dummy import DummyHealthMonitor


def _get_test_instrument(symbol: str = "BTCUSDT", exchange: str = "BINANCE.UM") -> Instrument:
    """Get a test instrument for testing."""
    instr = lookup.find_symbol(exchange, symbol)
    assert instr is not None, f"Could not find {symbol} on {exchange}"
    return instr


class TestExchangeManager:
    def test_successful_initialization(self):
        """Test ExchangeManager initializes with exchange properly."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()

        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            check_interval_seconds=30.0,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        assert manager._exchange == mock_exchange
        assert manager._recreation_count == 0
        assert manager._check_interval == 30.0

    def test_initialization_with_provided_exchange(self):
        """Test ExchangeManager initializes with provided exchange."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()

        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        assert manager._exchange == mock_exchange
        assert manager._recreation_count == 0

    def test_exchange_property_access(self):
        """Test that ExchangeManager exposes exchange via .exchange property."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.fetch_ticker.return_value = {"symbol": "BTC/USDT"}

        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        # Access underlying exchange via .exchange property
        result = manager.exchange.fetch_ticker("BTC/USDT")
        assert result == {"symbol": "BTC/USDT"}
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

        # Verify .exchange returns the actual exchange
        assert manager.exchange == mock_exchange

    def test_self_monitoring_initialization(self):
        """Test ExchangeManager initializes self-monitoring capabilities."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()

        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            check_interval_seconds=10.0,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        # Verify stall detection parameters are set
        assert manager._check_interval == 10.0
        assert manager._last_data_times == {}
        assert not manager._monitoring_enabled

    def test_on_data_arrival(self):
        """Test that ExchangeManager uses health monitor for data tracking."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"

        # Use a real health monitor to track data
        from qubx.health.base import BaseHealthMonitor

        health_monitor = BaseHealthMonitor(LiveTimeProvider())
        health_monitor.start()

        try:
            _manager = ExchangeManager(
                exchange_name="binance",
                factory_params={"exchange": "binance", "api_key": "test"},
                initial_exchange=mock_exchange,
                health_monitor=health_monitor,
                time_provider=LiveTimeProvider(),
            )

            # Simulate data arrival through health monitor
            import pandas as pd

            instrument = _get_test_instrument()
            test_time = pd.Timestamp("2023-01-01T12:00:00.000000000", tz="UTC").asm8

            # Subscribe before recording data
            health_monitor.subscribe(instrument, "ohlc")  # Use correct DataType
            health_monitor.subscribe(instrument, "trade")

            health_monitor.on_data_arrival(instrument, "ohlc", test_time)
            health_monitor.on_data_arrival(instrument, "trade", test_time)

            # Verify data arrival is tracked in health monitor
            assert health_monitor.get_last_event_time(instrument, "ohlc") is not None
            assert health_monitor.get_last_event_time(instrument, "trade") is not None
        finally:
            health_monitor.stop()

    def test_start_stop_monitoring(self):
        """Test start/stop monitoring controls background thread."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"

        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        # Initially not monitoring
        assert not manager._monitoring_enabled
        assert manager._monitor_thread is None

        # Start monitoring
        manager.start_monitoring()
        assert manager._monitoring_enabled
        assert manager._monitor_thread is not None

        # Stop monitoring
        manager.stop_monitoring()
        assert not manager._monitoring_enabled

    def test_stall_threshold_with_parameterized_event_types(self):
        """ExchangeManager no longer maintains its own stale thresholds (delegates to health monitor)."""
        # Kept as a placeholder to document removal of ExchangeManager stale-threshold logic.
        assert True

    def test_self_monitoring_stall_detection(self):
        """Test ExchangeManager detects and handles stales with custom thresholds."""
        import numpy as np

        from qubx.core.basics import dt_64
        from qubx.health.base import BaseHealthMonitor

        mock_exchange = Mock()
        mock_exchange.name = "binance"

        # Use real health monitor for tracking
        health_monitor = BaseHealthMonitor(LiveTimeProvider())
        health_monitor.start()

        try:
            manager = ExchangeManager(
                exchange_name="binance",
                factory_params={"exchange": "binance", "api_key": "test"},
                initial_exchange=mock_exchange,
                health_monitor=health_monitor,
                time_provider=LiveTimeProvider(),
            )

            # Record orderbook data
            instrument = _get_test_instrument()
            test_time = dt_64(np.datetime64("2023-01-01T12:00:00", "ns"))
            health_monitor.on_data_arrival(instrument, "orderbook", test_time)

            # Should trigger self-recreation when data is stale
            with patch.object(manager, "force_recreation", return_value=True) as mock_recreate:
                # Manually trigger stale check (normally runs in background thread)
                manager._check_and_handle_stales()
                # Should not recreate yet since data is fresh
                mock_recreate.assert_not_called()
        finally:
            health_monitor.stop()

    def test_stale_detection_delegates_to_health_monitor(self):
        """Test that stale detection relies on health_monitor.is_exchange_stale (single source of truth)."""
        import numpy as np

        from qubx.core.basics import dt_64
        from qubx.health.base import BaseHealthMonitor

        mock_exchange = Mock()
        mock_exchange.name = "binance"

        health_monitor = BaseHealthMonitor(LiveTimeProvider())
        health_monitor.start()

        try:
            instrument = _get_test_instrument()
            manager = ExchangeManager(
                # Important: ExchangeManager looks up last event times by exchange name.
                # Use the instrument's exchange so get_last_event_times_by_exchange() returns our recorded events.
                exchange_name=instrument.exchange,
                factory_params={"exchange": instrument.exchange, "api_key": "test"},
                initial_exchange=mock_exchange,
                health_monitor=health_monitor,
                time_provider=LiveTimeProvider(),
            )

            # Must subscribe; BaseHealthMonitor only records arrivals for active subscriptions
            health_monitor.subscribe(instrument, "quote")
            health_monitor.on_data_arrival(instrument, "quote", dt_64(np.datetime64("2023-01-01T00:00:00", "ns")))

            with patch.object(health_monitor, "is_exchange_stale", return_value=True) as mock_is_stale:
                with patch.object(manager, "force_recreation", return_value=True) as mock_recreate:
                    manager._check_and_handle_stales()
                    mock_is_stale.assert_called()
                    mock_recreate.assert_called_once()
        finally:
            health_monitor.stop()

    def test_exchange_property_delegation(self):
        """Test that ExchangeManager provides access to exchange properties via .exchange."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.apiKey = "test_key"
        mock_exchange.sandbox = False

        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        # Verify properties are accessible via .exchange
        assert manager.exchange.name == "binance"
        assert manager.exchange.id == "binance"
        assert manager.exchange.apiKey == "test_key"
        assert manager.exchange.sandbox is False

    def test_stall_triggered_recreation(self):
        """Test that stall detection triggers recreation."""
        with patch("qubx.connectors.ccxt.exchange_manager.ExchangeManager._create_exchange") as mock_create:
            # Setup two different exchanges for initial and recreated
            initial_exchange = Mock()
            initial_exchange.name = "binance"
            initial_exchange.id = "binance"
            initial_exchange.asyncio_loop = Mock()
            initial_exchange.close = Mock()

            new_exchange = Mock()
            new_exchange.name = "binance"
            new_exchange.id = "binance"
            new_exchange.asyncio_loop = Mock()

            mock_create.return_value = new_exchange

            manager = ExchangeManager(
                exchange_name="binance",
                factory_params={"exchange": "binance", "api_key": "test"},
                initial_exchange=initial_exchange,
                health_monitor=DummyHealthMonitor(),
                time_provider=LiveTimeProvider(),
            )

            # Trigger recreation
            result = manager.force_recreation()

            assert result is True
            assert manager._recreation_count == 1
            assert manager._exchange == new_exchange


class TestExchangeManagerIntegration:
    """Integration tests for ExchangeManager functionality."""

    def test_complete_self_monitoring_workflow(self):
        """Test complete self-monitoring workflow from start to cleanup."""
        import numpy as np

        from qubx.core.basics import dt_64
        from qubx.health.base import BaseHealthMonitor

        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        mock_exchange.close = Mock()

        # Use real health monitor for integration test
        health_monitor = BaseHealthMonitor(LiveTimeProvider())
        health_monitor.start()

        try:
            manager = ExchangeManager(
                exchange_name="binance",
                factory_params={"exchange": "binance", "api_key": "test"},
                initial_exchange=mock_exchange,
                check_interval_seconds=5.0,
                health_monitor=health_monitor,
                time_provider=LiveTimeProvider(),
            )

            # Test complete workflow
            manager.start_monitoring()
            assert manager._monitoring_enabled

            # Record some data through health monitor
            instrument = _get_test_instrument()
            test_time = dt_64(np.datetime64("2023-01-01T12:00:00", "ns"))

            # Subscribe before recording data
            health_monitor.subscribe(instrument, "ohlc")  # Use correct DataType
            health_monitor.subscribe(instrument, "trade")

            health_monitor.on_data_arrival(instrument, "ohlc", test_time)
            health_monitor.on_data_arrival(instrument, "trade", test_time)

            # Verify data is tracked in health monitor
            assert health_monitor.get_last_event_time(instrument, "ohlc") is not None
            assert health_monitor.get_last_event_time(instrument, "trade") is not None

            # Stop monitoring
            manager.stop_monitoring()
            assert not manager._monitoring_enabled
        finally:
            health_monitor.stop()

    def test_exception_handler_is_applied(self):
        """Test that exception handler is set on exchange."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.asyncio_loop = Mock()

        _manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            health_monitor=DummyHealthMonitor(),
            time_provider=LiveTimeProvider(),
        )

        # Verify exception handler was set up
        mock_exchange.asyncio_loop.set_exception_handler.assert_called()
