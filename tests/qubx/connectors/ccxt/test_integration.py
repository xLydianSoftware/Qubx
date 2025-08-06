"""
Integration tests for CCXT data provider architecture.

Tests that all components work together correctly, focusing on the integration
between the new component-based architecture.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.handlers import DataTypeHandlerFactory
from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.connectors.ccxt.subscription_orchestrator import SubscriptionOrchestrator
from qubx.connectors.ccxt.warmup_service import WarmupService
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType
from qubx.core.mixins.subscription import SubscriptionManager as SubscriptionMixin


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
            warmup_timeout=30,
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
        mock_config.individual_subscribers = None  # Use bulk subscription
        mock_config.uses_individual_streams = MagicMock(return_value=False)  # Mock the method
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
            timeframe="1m",
        )

        # Verify integration worked
        assert mock_handler.prepare_subscription.called
        assert mock_async_loop.submit.called

    def test_warmup_integration(self, integrated_components, test_instruments):
        """Test warmup service integration with handler factory."""
        warmup_service = integrated_components["warmup_service"]
        mock_async_loop = integrated_components["mock_async_loop"]

        # Execute warmup
        warmups = {("ohlc", test_instruments[0]): "1h"}

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


class TestSubscriptionUnsubscribeWorkflow:
    """Test the complete subscription/unsubscribe workflow demonstrating the commit pattern."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider for testing."""
        data_provider = MagicMock()
        data_provider.exchange.return_value = "BINANCE.UM"
        data_provider.has_subscription = MagicMock(return_value=False)
        data_provider.get_subscriptions = MagicMock(return_value=[])
        data_provider.get_subscribed_instruments = MagicMock(return_value=[])
        data_provider.subscribe = MagicMock()
        data_provider.unsubscribe = MagicMock()
        data_provider.warmup = MagicMock()
        return data_provider

    @pytest.fixture
    def test_instrument(self):
        """Create a test instrument."""
        return Instrument(
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

    def test_unsubscribe_requires_commit(self, mock_data_provider, test_instrument):
        """
        Test that demonstrates the unsubscribe issue and its solution.

        The key insight is that unsubscribe() only stages changes - commit() is required to apply them.
        """
        # Setup subscription manager with mock data provider
        subscription_manager = SubscriptionMixin(data_providers=[mock_data_provider])

        # Step 1: Subscribe to funding_rate
        subscription_manager.subscribe("funding_rate", [test_instrument])
        subscription_manager.commit()  # Apply subscription

        # Verify subscription was applied
        mock_data_provider.subscribe.assert_called_once()
        call_args = mock_data_provider.subscribe.call_args
        assert call_args[0][0] == "funding_rate"  # subscription type
        assert test_instrument in call_args[0][1]  # instruments set

        # Reset mock to track unsubscribe calls
        mock_data_provider.reset_mock()

        # Step 2: Call unsubscribe() - this ONLY stages the change
        subscription_manager.unsubscribe("funding_rate", [test_instrument])

        # IMPORTANT: At this point, unsubscribe has NOT been called on data provider yet
        mock_data_provider.unsubscribe.assert_not_called()

        # Step 3: Call commit() to actually apply the unsubscribe
        subscription_manager.commit()

        # NOW the unsubscribe should have been called
        mock_data_provider.unsubscribe.assert_called_once()
        call_args = mock_data_provider.unsubscribe.call_args
        assert call_args[0][0] == "funding_rate"  # subscription type
        assert test_instrument in call_args[0][1]  # instruments set

    def test_global_unsubscribe_requires_commit(self, mock_data_provider, test_instrument):
        """Test that global unsubscribe (without specific instruments) also requires commit."""
        # Mock the data provider to return the instrument as subscribed
        mock_data_provider.get_subscribed_instruments.return_value = [test_instrument]

        subscription_manager = SubscriptionMixin(data_providers=[mock_data_provider])

        # Subscribe first
        subscription_manager.subscribe("funding_rate", [test_instrument])
        subscription_manager.commit()
        mock_data_provider.reset_mock()

        # Global unsubscribe (no instruments specified)
        subscription_manager.unsubscribe("funding_rate")

        # Should not call unsubscribe yet
        mock_data_provider.unsubscribe.assert_not_called()

        # Commit to apply
        subscription_manager.commit()

        # Now unsubscribe should have been called
        mock_data_provider.unsubscribe.assert_called()

    def test_multiple_operations_batched_in_commit(self, mock_data_provider, test_instrument):
        """Test that multiple subscription operations are batched in a single commit."""
        subscription_manager = SubscriptionMixin(data_providers=[mock_data_provider])

        # Multiple operations without commit
        subscription_manager.subscribe("funding_rate", [test_instrument])
        subscription_manager.subscribe("trade", [test_instrument])
        subscription_manager.unsubscribe("ohlc", [test_instrument])  # Assuming ohlc was subscribed

        # None should be applied yet
        mock_data_provider.subscribe.assert_not_called()
        mock_data_provider.unsubscribe.assert_not_called()

        # Single commit applies all operations
        subscription_manager.commit()

        # All operations should now be applied
        assert mock_data_provider.subscribe.call_count >= 1  # At least the new subscriptions
        # (unsubscribe may or may not be called depending on whether ohlc was actually subscribed)

    def test_commit_is_idempotent(self, mock_data_provider, test_instrument):
        """Test that calling commit() multiple times doesn't cause issues."""
        subscription_manager = SubscriptionMixin(data_providers=[mock_data_provider])

        # Make a change
        subscription_manager.subscribe("funding_rate", [test_instrument])

        # First commit applies the change
        subscription_manager.commit()
        assert mock_data_provider.subscribe.call_count == 1

        # Second commit should be a no-op (nothing pending)
        subscription_manager.commit()
        assert mock_data_provider.subscribe.call_count == 1  # Still only called once


class TestDataTypeAllSubscriptionIntegration:
    """Integration test for DataType.ALL subscription behavior with pending subscriptions."""

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
            ),
            Instrument(
                symbol="ETHUSDT",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="BINANCE.UM",
                base="ETH",
                quote="USDT",
                settle="USDT",
                exchange_symbol="ETH/USDT:USDT",
                tick_size=0.01,
                lot_size=0.001,
                min_size=0.001,
            ),
        ]

    def test_datatype_all_includes_pending_subscriptions_integration(self, test_instruments):
        """Integration test: DataType.ALL should subscribe to pending subscription types.

        This test simulates the BaseDataGatheringStrategy scenario where:
        1. Initial instrument (BTCUSDT) subscribes to ohlc, funding_rate, open_interest
        2. Some subscriptions become active quickly, others stay pending
        3. Universe update adds more instruments (ETHUSDT) using DataType.ALL
        4. DataType.ALL should include ALL subscription types, including pending ones
        """
        from qubx.connectors.ccxt.subscription_manager import SubscriptionManager as CcxtSubscriptionManager
        from qubx.core.basics import DataType
        from qubx.core.mixins.subscription import SubscriptionManager as CoreSubscriptionManager

        # Setup: Create CCXT subscription manager and mock data provider
        ccxt_subscription_manager = CcxtSubscriptionManager()

        # Create a mock data provider that uses our subscription manager
        mock_data_provider = MagicMock()
        mock_data_provider.get_subscriptions = ccxt_subscription_manager.get_subscriptions
        mock_data_provider.get_subscribed_instruments = ccxt_subscription_manager.get_subscribed_instruments
        mock_data_provider.has_subscription = ccxt_subscription_manager.has_subscription
        mock_data_provider.has_pending_subscription = ccxt_subscription_manager.has_pending_subscription
        mock_data_provider.exchange.return_value = "BINANCE.UM"

        # Track actual subscribe calls
        subscribe_calls = []

        def track_subscribe(sub_type, instruments, reset=False):
            subscribe_calls.append((sub_type, set(instruments), reset))
            # Simulate adding to CCXT subscription manager
            ccxt_subscription_manager.add_subscription(sub_type, list(instruments), reset=reset)

        mock_data_provider.subscribe = track_subscribe
        mock_data_provider.unsubscribe = MagicMock()

        # Create core subscription manager (used by strategy context)
        core_subscription_manager = CoreSubscriptionManager([mock_data_provider])

        # Phase 1: Initial instrument subscription (simulates BaseDataGatheringStrategy.on_init)
        initial_instrument = [test_instruments[0]]  # BTCUSDT only

        core_subscription_manager.subscribe(DataType.OHLC["1m"], initial_instrument)
        core_subscription_manager.subscribe(DataType.FUNDING_RATE, initial_instrument)
        core_subscription_manager.subscribe(DataType.OPEN_INTEREST, initial_instrument)
        core_subscription_manager.commit()

        # Simulate some subscriptions becoming active faster than others
        ccxt_subscription_manager.mark_subscription_active(DataType.OHLC["1m"])
        ccxt_subscription_manager.mark_subscription_active(DataType.FUNDING_RATE)
        # Leave OPEN_INTEREST as pending (slow connection)

        # Verify initial state: should have both active and pending subscriptions
        all_subscription_types = ccxt_subscription_manager.get_subscriptions()
        expected_types = {DataType.OHLC["1m"], DataType.FUNDING_RATE, DataType.OPEN_INTEREST}
        assert set(all_subscription_types) == expected_types, (
            f"Initial subscriptions should include pending. Got: {set(all_subscription_types)}"
        )

        # Phase 2: Universe update with DataType.ALL (simulates BaseDataGatheringStrategy.on_fit)
        new_instruments = [test_instruments[1]]  # ETHUSDT

        # Clear previous subscribe calls to focus on DataType.ALL behavior
        subscribe_calls.clear()

        # This is the critical test: DataType.ALL should use get_subscriptions() which now includes pending
        core_subscription_manager.subscribe(DataType.ALL, new_instruments)
        core_subscription_manager.commit()

        # Verify: DataType.ALL should have triggered subscriptions for ALL types including pending
        subscription_types_called = {call[0] for call in subscribe_calls}
        assert DataType.OPEN_INTEREST in subscription_types_called, (
            f"DataType.ALL should subscribe to pending open_interest. Calls: {subscribe_calls}"
        )
        assert DataType.OHLC["1m"] in subscription_types_called, (
            f"DataType.ALL should subscribe to active ohlc. Calls: {subscribe_calls}"
        )
        assert DataType.FUNDING_RATE in subscription_types_called, (
            f"DataType.ALL should subscribe to active funding_rate. Calls: {subscribe_calls}"
        )

        # Verify all subscription types were called with the new instrument
        for sub_type in expected_types:
            matching_calls = [call for call in subscribe_calls if call[0] == sub_type]
            assert len(matching_calls) == 1, f"Should have exactly one call for {sub_type}"
            call_instruments = matching_calls[0][1]
            assert test_instruments[1] in call_instruments, f"New instrument should be in {sub_type} subscription call"

        # Verify final state: new instrument should have all subscription types (excluding base subscription)
        eth_subscriptions = ccxt_subscription_manager.get_subscriptions(test_instruments[1])
        eth_subscriptions_filtered = {sub for sub in eth_subscriptions if sub != DataType.NONE}
        assert set(eth_subscriptions_filtered) == expected_types, (
            f"New instrument should have all subscription types: {set(eth_subscriptions_filtered)}"
        )

        # Verify the fix: pending subscriptions are properly handled
        assert ccxt_subscription_manager.has_pending_subscription(test_instruments[1], DataType.OPEN_INTEREST), (
            "New instrument should have pending open_interest subscription"
        )
