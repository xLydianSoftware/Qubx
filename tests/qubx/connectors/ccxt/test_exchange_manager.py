"""Tests for ExchangeManager functionality."""

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
            max_recreations=3,
            stall_threshold_seconds=120.0,
            check_interval_seconds=30.0
        )
        
        assert manager._exchange == mock_exchange
        assert manager._recreation_count == 0
        assert manager._stall_threshold == 120.0
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
            stall_threshold_seconds=60.0,
            check_interval_seconds=10.0
        )
        
        # Verify stall detection parameters are set
        assert manager._stall_threshold == 60.0
        assert manager._check_interval == 10.0
        assert manager._last_data_times == {}
        assert not manager._monitoring_enabled
        
    def test_on_data_arrival(self):
        """Test on_data_arrival tracks data timestamps."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange
        )
        
        with patch('time.time', return_value=100.0):
            import pandas as pd
            test_time = pd.Timestamp('2023-01-01T12:00:00.000000000', tz='UTC').asm8
            manager.on_data_arrival("ohlcv", test_time)
            manager.on_data_arrival("trade", test_time)
            
        # Verify data arrival times are tracked
        assert manager._last_data_times["ohlcv"] == 100.0
        assert manager._last_data_times["trade"] == 100.0
        
    def test_start_stop_monitoring(self):
        """Test start/stop monitoring controls background thread."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange
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
        
    def test_self_monitoring_stall_detection(self):
        """Test ExchangeManager detects and handles stalls itself."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            stall_threshold_seconds=10.0
        )
        
        with patch('time.time') as mock_time:
            # Record data arrival at time 100
            mock_time.return_value = 100.0
            import pandas as pd
            test_time = pd.Timestamp('2023-01-01T12:00:00.000000000', tz='UTC').asm8
            manager.on_data_arrival("ohlcv", test_time)
            
            # Simulate stall (time 120, 20 seconds later > 10s threshold)
            mock_time.return_value = 120.0
            
            # Should trigger self-recreation
            with patch.object(manager, 'force_recreation', return_value=True) as mock_recreate:
                manager._check_and_handle_stalls()
                mock_recreate.assert_called_once()
                
                # Verify data tracking was cleared after successful recreation
                assert len(manager._last_data_times) == 0
            
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
            initial_exchange=mock_exchange
        )
        
        # Verify properties are accessible via .exchange
        assert manager.exchange.name == "binance"
        assert manager.exchange.id == "binance"
        assert manager.exchange.apiKey == "test_key"
        assert manager.exchange.sandbox == False
            
    def test_stall_triggered_recreation(self):
        """Test that stall detection triggers recreation."""
        with patch('qubx.connectors.ccxt.exchange_manager.ExchangeManager._create_exchange') as mock_create:
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
                initial_exchange=initial_exchange
            )
            
            # Trigger recreation
            result = manager.force_recreation()
            
            assert result is True
            assert manager._recreation_count == 1
            assert manager._exchange == new_exchange
            
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


class TestExchangeManagerIntegration:
    """Integration tests for ExchangeManager functionality."""
    
    def test_complete_self_monitoring_workflow(self):
        """Test complete self-monitoring workflow from start to cleanup."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.id = "binance"
        mock_exchange.asyncio_loop = Mock()
        mock_exchange.close = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange,
            stall_threshold_seconds=30.0,
            check_interval_seconds=5.0
        )
        
        # Test complete workflow
        manager.start_monitoring()
        assert manager._monitoring_enabled
        
        # Record some data
        import pandas as pd
        test_time = pd.Timestamp('2023-01-01T12:00:00.000000000', tz='UTC').asm8
        manager.on_data_arrival("ohlcv", test_time)
        manager.on_data_arrival("trade", test_time)
        assert len(manager._last_data_times) == 2
        
        # Stop monitoring
        manager.stop_monitoring()
        assert not manager._monitoring_enabled
        
    def test_exception_handler_is_applied(self):
        """Test that exception handler is set on exchange."""
        mock_exchange = Mock()
        mock_exchange.name = "binance"
        mock_exchange.asyncio_loop = Mock()
        
        manager = ExchangeManager(
            exchange_name="binance",
            factory_params={"exchange": "binance", "api_key": "test"},
            initial_exchange=mock_exchange
        )
        
        # Verify exception handler was set up
        mock_exchange.asyncio_loop.set_exception_handler.assert_called()