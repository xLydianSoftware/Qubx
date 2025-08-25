"""Tests for ExchangeManager functionality."""

import time
from unittest.mock import Mock, patch
from qubx.connectors.ccxt.exchange_manager import ExchangeManager


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
            max_recreations=3
        )
        
        assert manager._exchange == mock_exchange
        assert manager._recreation_count == 0
            
    def test_initialization_with_provided_exchange(self):
        """Test ExchangeManager initializes with provided exchange."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"api_key": "test"},
            initial_exchange=mock_exchange
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
            initial_exchange=mock_exchange
        )
        
        # Access underlying exchange via .exchange property
        result = manager.exchange.fetch_ticker("BTC/USDT")
        assert result == {"symbol": "BTC/USDT"}
        mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")
        
        # Verify .exchange returns the actual exchange
        assert manager.exchange == mock_exchange
            
    def test_exchange_property_delegation(self):
        """Test that ExchangeManager provides access to exchange properties via .exchange."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        mock_exchange.sandbox = False
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange
        )
        
        # Property access should work via .exchange property
        assert manager.exchange.name == "binance"
        assert manager.exchange.id == "binance"
        assert manager.exchange.asyncio_loop == mock_exchange.asyncio_loop
        assert manager.exchange.sandbox is False
            
    def test_stall_triggered_recreation_success(self):
        """Test that stall detection triggers recreation successfully."""
        # Setup initial exchange
        initial_exchange = Mock()
        initial_exchange.name = "binance"
        initial_exchange.id = "binance"
        initial_exchange.asyncio_loop = Mock()
        initial_exchange.close = Mock()
        
        new_exchange = Mock()
        new_exchange.name = "binance"
        new_exchange.id = "binance" 
        new_exchange.asyncio_loop = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=initial_exchange
        )
        
        # Mock the _create_exchange method to return the new exchange
        with patch.object(manager, '_create_exchange', return_value=new_exchange):
            # Trigger stall-based recreation
            result = manager.force_recreation()
            
            assert result is True
            assert manager._recreation_count == 1
            assert manager._exchange == new_exchange
            
    def test_stall_triggered_recreation_failure(self):
        """Test that stall detection handles recreation failure."""
        # Setup initial exchange
        initial_exchange = Mock()
        initial_exchange.name = "binance"
        initial_exchange.id = "binance"
        initial_exchange.asyncio_loop = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=initial_exchange
        )
        
        # Mock _create_exchange to raise an exception
        with patch.object(manager, '_create_exchange', side_effect=RuntimeError("Recreation failed")):
            # Trigger stall-based recreation (should fail)
            result = manager.force_recreation()
            
            assert result is False
            assert manager._recreation_count == 1
            assert manager._exchange == initial_exchange  # Original exchange still there
            
    def test_recreation_limit_prevents_excessive_attempts(self):
        """Test that recreation limit prevents infinite recreation attempts."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            max_recreations=2
        )
        
        # Exhaust recreation limit
        manager._recreation_count = 2
        
        # Should not trigger recreation
        result = manager.force_recreation()
        assert result is False

    def test_recreation_count_reset_after_interval(self):
        """Test that recreation count resets after interval."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            reset_interval_hours=0.001  # Very short interval for testing
        )
        
        # Set recreation count
        manager._recreation_count = 2
        manager._last_successful_reset = time.time() - 3600  # 1 hour ago
        
        # Call reset method
        manager.reset_recreation_count_if_needed()
        
        # Recreation count should be reset
        assert manager._recreation_count == 0

    def test_exchange_close_method_access(self):
        """Test that close method can be accessed via .exchange property."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        mock_exchange.close = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"api_key": "test"},
            initial_exchange=mock_exchange
        )
        
        # Create a mock coroutine for the close method
        import asyncio
        async def close_coro():
            return None
        mock_exchange.close.return_value = close_coro()
        
        # Test close method access via .exchange property
        close_result = manager.exchange.close()
        assert asyncio.iscoroutine(close_result)
        mock_exchange.close.assert_called_once()
        
        # Clean up the coroutine
        asyncio.get_event_loop().run_until_complete(close_result)


class TestBaseHealthMonitorStallDetection:
    def test_stall_detection_disabled_by_default(self):
        """Test BaseHealthMonitor stall detection is disabled by default."""
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock
        
        mock_time_provider = Mock()
        
        # Test BaseHealthMonitor without any registered exchange managers
        health_monitor = BaseHealthMonitor(
            time_provider=mock_time_provider,
        )
        
        assert len(health_monitor._registered_exchange_managers) == 0

    def test_register_exchange_manager(self):
        """Test BaseHealthMonitor register_exchange_manager method."""
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock
        
        mock_time_provider = Mock()
        mock_exchange_manager = Mock()
        mock_exchange_manager._exchange_name = "binance"
        
        # Test BaseHealthMonitor with registered exchange manager
        health_monitor = BaseHealthMonitor(
            time_provider=mock_time_provider,
        )
        
        # Register the exchange manager
        health_monitor.register_exchange_manager(
            mock_exchange_manager,
            stall_threshold_seconds=10.0
        )
        
        assert len(health_monitor._registered_exchange_managers) == 1
        assert mock_exchange_manager in health_monitor._registered_exchange_managers
        assert health_monitor._stall_threshold == 10.0

    def test_unregister_exchange_manager(self):
        """Test BaseHealthMonitor unregister_exchange_manager method."""
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock
        
        mock_time_provider = Mock()
        mock_exchange_manager = Mock()
        mock_exchange_manager._exchange_name = "binance"
        
        health_monitor = BaseHealthMonitor(
            time_provider=mock_time_provider,
        )
        
        # Register then unregister the exchange manager
        health_monitor.register_exchange_manager(mock_exchange_manager)
        assert len(health_monitor._registered_exchange_managers) == 1
        
        health_monitor.unregister_exchange_manager(mock_exchange_manager)
        assert len(health_monitor._registered_exchange_managers) == 0

    @patch('time.time')
    def test_record_data_arrival_tracks_when_managers_registered(self, mock_time):
        """Test that record_data_arrival tracks data arrival times when exchange managers are registered."""
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock
        import numpy as np
        
        mock_time_provider = Mock()
        mock_time_provider.time.return_value = np.datetime64('2023-01-01T12:00:00.000000000', 'ns')
        mock_exchange_manager = Mock()
        mock_exchange_manager._exchange_name = "binance"
        
        mock_time.return_value = 1672574400.0  # Mock timestamp
        
        health_monitor = BaseHealthMonitor(
            time_provider=mock_time_provider,
        )
        
        # Register exchange manager
        health_monitor.register_exchange_manager(mock_exchange_manager)
        
        # Record data arrival
        event_time = np.datetime64('2023-01-01T11:59:59.000000000', 'ns')
        health_monitor.record_data_arrival("ohlcv:BTC/USDT:1m", event_time)
        
        # Check that the data time was tracked
        assert "ohlcv:BTC/USDT:1m" in health_monitor._last_data_times
        assert health_monitor._last_data_times["ohlcv:BTC/USDT:1m"] == 1672574400.0

    @patch('time.time')
    def test_record_data_arrival_no_tracking_without_managers(self, mock_time):
        """Test that record_data_arrival does not track data when no exchange managers are registered."""
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock
        import numpy as np
        
        mock_time_provider = Mock()
        mock_time_provider.time.return_value = np.datetime64('2023-01-01T12:00:00.000000000', 'ns')
        
        mock_time.return_value = 1672574400.0  # Mock timestamp
        
        health_monitor = BaseHealthMonitor(
            time_provider=mock_time_provider,
        )
        
        # Record data arrival (no exchange managers registered)
        event_time = np.datetime64('2023-01-01T11:59:59.000000000', 'ns')
        health_monitor.record_data_arrival("ohlcv:BTC/USDT:1m", event_time)
        
        # Check that the data time was NOT tracked
        assert "ohlcv:BTC/USDT:1m" not in health_monitor._last_data_times

    def test_dummy_health_monitor_no_op_methods(self):
        """Test that DummyHealthMonitor has no-op register/unregister methods."""
        from qubx.health.base import DummyHealthMonitor
        from unittest.mock import Mock
        
        dummy_monitor = DummyHealthMonitor()
        mock_exchange_manager = Mock()
        mock_exchange_manager._exchange_name = "binance"
        
        # These methods should exist and do nothing (no exceptions)
        dummy_monitor.register_exchange_manager(mock_exchange_manager)
        dummy_monitor.unregister_exchange_manager(mock_exchange_manager)
        
        # Should have the methods
        assert hasattr(dummy_monitor, 'register_exchange_manager')
        assert hasattr(dummy_monitor, 'unregister_exchange_manager')

    @patch('time.time')
    def test_full_stall_detection_flow(self, mock_time):
        """Test the complete stall detection and recreation flow."""
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock, patch
        import numpy as np
        import threading
        import time
        
        # Setup mocks
        mock_time_provider = Mock()
        mock_time_provider.time.return_value = np.datetime64('2023-01-01T12:00:00.000000000', 'ns')
        
        mock_channel = Mock()
        mock_channel._queue = Mock()
        mock_channel._queue.qsize.return_value = 0
        
        # Create ExchangeManager mock with recreation tracking
        mock_exchange_manager = Mock()
        mock_exchange_manager._exchange_name = "binance"
        mock_exchange_manager.force_recreation.return_value = True
        mock_exchange_manager.reset_recreation_count_if_needed = Mock()
        
        # Create BaseHealthMonitor with short thresholds for testing
        health_monitor = BaseHealthMonitor(
            time_provider=mock_time_provider,
            channel=mock_channel
        )
        
        # Register exchange manager with short stall threshold
        health_monitor.register_exchange_manager(
            mock_exchange_manager,
            stall_threshold_seconds=2.0,  # Very short for testing
            check_interval_seconds=0.1    # Very short for testing
        )
        
        # Start monitoring
        health_monitor.start()
        
        try:
            # Simulate data arrival at time 100
            mock_time.return_value = 100.0
            event_time = np.datetime64('2023-01-01T12:00:00.000000000', 'ns')
            health_monitor.record_data_arrival("ohlcv:BTC/USDT:1m", event_time)
            
            # Verify data was tracked
            assert "ohlcv:BTC/USDT:1m" in health_monitor._last_data_times
            assert health_monitor._last_data_times["ohlcv:BTC/USDT:1m"] == 100.0
            
            # Simulate time passing beyond stall threshold (time 105, 5 seconds later > 2s threshold)
            mock_time.return_value = 105.0
            
            # Give stall detection thread a moment to run
            time.sleep(0.2)  # Wait for background thread
            
            # Verify recreation was triggered
            mock_exchange_manager.force_recreation.assert_called()
            mock_exchange_manager.reset_recreation_count_if_needed.assert_called()
            
            # Verify data tracking was cleared after successful recreation
            # (This happens when force_recreation returns True)
            assert len(health_monitor._last_data_times) == 0
            
        finally:
            # Stop monitoring
            health_monitor.stop()

    def test_ccxt_data_provider_always_registers_exchange_manager(self):
        """Test that CcxtDataProvider always registers ExchangeManager (since get_ccxt_exchange always returns one)."""
        from qubx.connectors.ccxt.data import CcxtDataProvider
        from qubx.connectors.ccxt.factory import get_ccxt_exchange
        from qubx.health.base import BaseHealthMonitor
        from unittest.mock import Mock
        import numpy as np
        
        mock_time_provider = Mock()
        mock_time_provider.time.return_value = np.datetime64('2023-01-01T12:00:00.000000000', 'ns')
        
        mock_channel = Mock()
        mock_channel._queue = Mock()
        mock_channel._queue.qsize.return_value = 0
        
        health_monitor = BaseHealthMonitor(time_provider=mock_time_provider, channel=mock_channel)
        
        # get_ccxt_exchange always returns ExchangeManager now
        exchange_manager = get_ccxt_exchange('binance')
        assert isinstance(exchange_manager, ExchangeManager)
        
        data_provider = CcxtDataProvider(
            exchange_manager=exchange_manager,
            time_provider=mock_time_provider,
            channel=mock_channel,
            health_monitor=health_monitor
        )
        
        # Should always register since we always get ExchangeManager
        assert len(health_monitor._registered_exchange_managers) == 1
        data_provider.close()
        assert len(health_monitor._registered_exchange_managers) == 0
