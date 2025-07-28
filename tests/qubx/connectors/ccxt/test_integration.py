"""
Integration tests for CCXT data provider architecture.

Tests that all components work together correctly, focusing on the integration
between the new component-based architecture.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.connectors.ccxt.subscription_orchestrator import SubscriptionOrchestrator
from qubx.connectors.ccxt.warmup_service import WarmupService
from qubx.connectors.ccxt.handlers import DataTypeHandlerFactory
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType


class TestCcxtArchitectureIntegration:
    """Integration tests for the new CCXT component architecture."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider."""
        data_provider = MagicMock()
        data_provider._exchange_id = "test_exchange"
        data_provider._last_quotes = {}
        data_provider.channel = MagicMock(spec=CtrlChannel)
        data_provider.time_provider = MagicMock()
        return data_provider

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock CCXT exchange."""
        exchange = MagicMock()
        exchange.name = "test_exchange"
        exchange.watch_ohlcv_for_symbols = AsyncMock()
        exchange.watch_trades_for_symbols = AsyncMock()
        return exchange

    @pytest.fixture
    def test_instruments(self):
        """Create test instruments."""
        return [
            Instrument(
                symbol="BTCUSDT",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="BINANCE.UM",
                base="BTC",
                quote="USDT",
                settle="USDT",
                exchange_symbol="BTC/USDT:USDT",
                tick_size=0.1,
                lot_size=0.001,
                min_size=0.001,
            )
        ]

    @pytest.fixture
    def integrated_components(self, mock_data_provider, mock_exchange, test_instruments):
        """Create a full set of integrated components."""
        # Create all components
        subscription_manager = SubscriptionManager()
        connection_manager = ConnectionManager("test_exchange", 3, subscription_manager)
        orchestrator = SubscriptionOrchestrator("test_exchange", subscription_manager, connection_manager)
        handler_factory = DataTypeHandlerFactory(mock_data_provider, mock_exchange, "test_exchange")
        
        # Mock async loop for warmup service
        mock_async_loop = MagicMock()
        mock_async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock()
        mock_async_loop.submit.return_value = mock_future
        
        warmup_service = WarmupService(
            handler_factory=handler_factory,
            channel=mock_data_provider.channel,
            exchange_id="test_exchange",
            async_loop=mock_async_loop,
            warmup_timeout=30
        )

        return {
            "subscription_manager": subscription_manager,
            "connection_manager": connection_manager,
            "orchestrator": orchestrator,
            "handler_factory": handler_factory,
            "warmup_service": warmup_service,
            "mock_async_loop": mock_async_loop,
        }

    def test_components_integration(self, integrated_components):
        """Test that all components are properly integrated."""
        components = integrated_components
        
        # Verify all components exist
        assert components["subscription_manager"] is not None
        assert components["connection_manager"] is not None
        assert components["orchestrator"] is not None
        assert components["handler_factory"] is not None
        assert components["warmup_service"] is not None

        # Verify cross-component references
        assert components["connection_manager"]._subscription_manager is components["subscription_manager"]
        assert components["orchestrator"]._subscription_manager is components["subscription_manager"]
        assert components["orchestrator"]._connection_manager is components["connection_manager"]

    def test_handler_factory_integration(self, integrated_components):
        """Test that handler factory creates handlers for all supported data types."""
        handler_factory = integrated_components["handler_factory"]
        
        supported_types = ["ohlc", "trade", "orderbook", "quote", "liquidation", "funding_rate", "open_interest"]
        
        for data_type in supported_types:
            handler = handler_factory.get_handler(data_type)
            assert handler is not None, f"No handler found for {data_type}"

    def test_subscription_lifecycle_integration(self, integrated_components, test_instruments, mock_exchange):
        """Test the integration of subscription lifecycle across components."""
        orchestrator = integrated_components["orchestrator"]
        handler_factory = integrated_components["handler_factory"]
        mock_async_loop = integrated_components["mock_async_loop"]
        
        # Mock handler
        mock_handler = handler_factory.get_handler("ohlc")
        
        # Mock subscription config
        mock_config = MagicMock()
        mock_config.subscriber_func = AsyncMock()
        mock_config.unsubscriber_func = AsyncMock()
        mock_config.stream_name = "test_stream"
        mock_handler.prepare_subscription = MagicMock(return_value=mock_config)
        
        # Mock channel
        mock_channel = MagicMock()
        
        # Execute subscription - should integrate all components
        orchestrator.execute_subscription(
            subscription_type="ohlc(1m)",
            instruments=set(test_instruments),
            handler=mock_handler,
            stream_name_generator=lambda x, **kwargs: f"{x}_stream",
            async_loop_submit=mock_async_loop.submit,
            exchange=mock_exchange,
            channel=mock_channel,
            timeframe="1m"
        )
        
        # Verify integration worked
        assert mock_handler.prepare_subscription.called
        assert mock_async_loop.submit.called

    def test_warmup_integration(self, integrated_components, test_instruments):
        """Test warmup service integration with handler factory."""
        warmup_service = integrated_components["warmup_service"]
        mock_async_loop = integrated_components["mock_async_loop"]
        
        # Execute warmup
        warmups = {
            ("ohlc", test_instruments[0]): "1h"
        }
        
        warmup_service.execute_warmup(warmups)
        
        # Verify async loop was used
        assert mock_async_loop.submit.called

    def test_component_error_isolation(self, integrated_components):
        """Test that errors in one component don't break others."""
        subscription_manager = integrated_components["subscription_manager"]
        connection_manager = integrated_components["connection_manager"]
        
        # Each component should be independent
        assert subscription_manager is not connection_manager
        assert hasattr(subscription_manager, "_pending_subscriptions")
        assert hasattr(connection_manager, "_is_stream_enabled")

    def test_architecture_separation_of_concerns(self, integrated_components):
        """Test that each component has clear responsibilities."""
        components = integrated_components
        
        # SubscriptionManager - manages subscription state
        assert hasattr(components["subscription_manager"], "setup_new_subscription")
        assert hasattr(components["subscription_manager"], "mark_subscription_active")
        
        # ConnectionManager - manages WebSocket connections
        assert hasattr(components["connection_manager"], "listen_to_stream")
        assert hasattr(components["connection_manager"], "stop_stream")
        
        # SubscriptionOrchestrator - coordinates between them
        assert hasattr(components["orchestrator"], "execute_subscription")
        assert hasattr(components["orchestrator"], "stop_subscription")
        
        # Handler Factory - creates data type handlers
        assert hasattr(components["handler_factory"], "get_handler")
        assert hasattr(components["handler_factory"], "get_supported_data_types")
        
        # WarmupService - handles warmup operations
        assert hasattr(components["warmup_service"], "execute_warmup")


class TestCcxtComponentCompatibility:
    """Test compatibility between components and existing architecture."""

    def test_subscription_manager_api(self):
        """Test that SubscriptionManager has expected API."""
        manager = SubscriptionManager()
        
        # Key methods that other components depend on
        assert hasattr(manager, "setup_new_subscription")
        assert hasattr(manager, "mark_subscription_active")
        assert hasattr(manager, "get_subscription_name")
        assert hasattr(manager, "prepare_resubscription")
        assert hasattr(manager, "complete_resubscription_cleanup")

    def test_connection_manager_api(self):
        """Test that ConnectionManager has expected API."""
        manager = ConnectionManager("test_exchange")
        
        # Key methods that other components depend on
        assert hasattr(manager, "listen_to_stream")
        assert hasattr(manager, "stop_stream")
        # stop_stream can handle both regular and resubscription scenarios
        assert hasattr(manager, "is_stream_enabled")
        assert hasattr(manager, "register_stream_future")

    def test_handler_factory_api(self):
        """Test that DataTypeHandlerFactory has expected API.""" 
        factory = DataTypeHandlerFactory(None, None, "test_exchange")
        
        # Key methods that other components depend on
        assert hasattr(factory, "get_handler")
        assert hasattr(factory, "has_handler")
        assert hasattr(factory, "get_supported_data_types")

    def test_orchestrator_api(self):
        """Test that SubscriptionOrchestrator has expected API."""
        manager = SubscriptionManager()
        connection_manager = ConnectionManager("test_exchange", subscription_manager=manager)
        orchestrator = SubscriptionOrchestrator("test_exchange", manager, connection_manager)
        
        # Key methods that facade depends on
        assert hasattr(orchestrator, "execute_subscription")
        assert hasattr(orchestrator, "stop_subscription")
        assert hasattr(orchestrator, "call_by_market_type")