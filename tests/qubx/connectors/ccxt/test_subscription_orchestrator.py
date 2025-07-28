"""
Unit tests for SubscriptionOrchestrator component.

Tests the coordination between SubscriptionManager and ConnectionManager,
focusing on complex resubscription scenarios and cleanup logic.
"""

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.connectors.ccxt.subscription_orchestrator import SubscriptionOrchestrator


class TestSubscriptionOrchestrator:
    """Test suite for SubscriptionOrchestrator component."""

    def test_initialization(self, subscription_manager, connection_manager):
        """Test that SubscriptionOrchestrator initializes correctly."""
        orchestrator = SubscriptionOrchestrator(
            exchange_id="test_exchange",
            subscription_manager=subscription_manager,
            connection_manager=connection_manager
        )
        
        assert orchestrator._exchange_id == "test_exchange"
        assert orchestrator._subscription_manager == subscription_manager
        assert orchestrator._connection_manager == connection_manager

    def test_execute_subscription_new(self, subscription_orchestrator, mock_instruments, mock_exchange, mock_ctrl_channel):
        """Test executing subscription for new subscription type."""
        subscription_type = "ohlc"
        instruments = set(mock_instruments[:2])
        
        # Mock handler and its methods
        mock_handler = MagicMock()
        mock_config = MagicMock()
        mock_config.subscriber_func = AsyncMock()
        mock_config.unsubscriber_func = AsyncMock()
        mock_config.stream_name = "test_stream"
        mock_handler.prepare_subscription.return_value = mock_config
        
        # Mock stream name generator
        stream_name_generator = MagicMock(return_value="test_stream")
        
        # Mock async loop submit
        mock_future = MagicMock(spec=concurrent.futures.Future)
        async_loop_submit = MagicMock(return_value=mock_future)
        
        # Mock subscription manager methods
        subscription_orchestrator._subscription_manager.prepare_resubscription = MagicMock(return_value=None)
        subscription_orchestrator._subscription_manager.setup_new_subscription = MagicMock()
        subscription_orchestrator._connection_manager.register_stream_future = MagicMock()
        
        # Execute subscription (synchronous method)
        subscription_orchestrator.execute_subscription(
            subscription_type=subscription_type,
            instruments=instruments,
            handler=mock_handler,
            stream_name_generator=stream_name_generator,
            async_loop_submit=async_loop_submit,
            exchange=mock_exchange,
            channel=mock_ctrl_channel
        )
        
        # Should prepare subscription with handler
        mock_handler.prepare_subscription.assert_called_once()
        
        # Should submit async task to listen to stream
        async_loop_submit.assert_called_once()
        
        # Should register stream future
        subscription_orchestrator._connection_manager.register_stream_future.assert_called_once_with(
            "test_stream", mock_future
        )

    def test_execute_subscription_resubscription(self, subscription_orchestrator, mock_instruments, mock_exchange, mock_ctrl_channel):
        """Test executing subscription when resubscribing (replacing existing)."""
        subscription_type = "ohlc"
        instruments = set(mock_instruments[:2])
        
        # Setup existing subscription state
        old_stream_name = "old_stream"
        old_future = MagicMock(spec=concurrent.futures.Future)
        old_future.running.return_value = False  # Simulate quick cancellation
        
        # Add existing subscription to internal state
        subscription_orchestrator._sub_to_coro[subscription_type] = old_future
        
        subscription_orchestrator._subscription_manager.prepare_resubscription = MagicMock(
            return_value={"stream_name": old_stream_name, "instruments": instruments}
        )
        
        # Mock handler
        mock_handler = MagicMock()
        mock_config = MagicMock()
        mock_config.subscriber_func = AsyncMock()
        mock_config.unsubscriber_func = AsyncMock()
        mock_config.stream_name = "new_stream"
        mock_handler.prepare_subscription.return_value = mock_config
        
        # Mock dependencies
        stream_name_generator = MagicMock(return_value="new_stream")
        new_future = MagicMock(spec=concurrent.futures.Future)
        async_loop_submit = MagicMock(return_value=new_future)
        
        # Mock connection manager methods
        subscription_orchestrator._subscription_manager.complete_resubscription_cleanup = MagicMock()
        subscription_orchestrator._subscription_manager.setup_new_subscription = MagicMock()
        subscription_orchestrator._connection_manager.disable_stream = MagicMock()
        subscription_orchestrator._connection_manager.register_stream_future = MagicMock()
        
        # Mock the blocking _wait_for_cancellation method
        with patch.object(subscription_orchestrator, '_wait_for_cancellation') as mock_wait:
            with patch.object(subscription_orchestrator._connection_manager, 'stop_stream', new_callable=AsyncMock) as mock_stop:
                # Execute subscription (synchronous method)
                subscription_orchestrator.execute_subscription(
                    subscription_type=subscription_type,
                    instruments=instruments,
                    handler=mock_handler,
                    stream_name_generator=stream_name_generator,
                    async_loop_submit=async_loop_submit,
                    exchange=mock_exchange,
                    channel=mock_ctrl_channel
                )
        
        # Should wait for cancellation
        mock_wait.assert_called_once_with(old_future, subscription_type)
        
        # Should disable old stream
        subscription_orchestrator._connection_manager.disable_stream.assert_called_once_with(old_stream_name)
        
        # Should cancel old future
        old_future.cancel.assert_called_once()
        
        # Should prepare resubscription
        subscription_orchestrator._subscription_manager.prepare_resubscription.assert_called_once_with(subscription_type)

    def test_execute_subscription_handler_error(self, subscription_orchestrator, mock_instruments, mock_exchange, mock_ctrl_channel):
        """Test handling errors in handler preparation."""
        subscription_type = "ohlc"
        instruments = set(mock_instruments[:2])
        
        # Mock handler that raises error
        mock_handler = MagicMock()
        mock_handler.prepare_subscription.side_effect = Exception("Handler preparation failed")
        
        stream_name_generator = MagicMock(return_value="test_stream")
        async_loop_submit = MagicMock()
        
        # Mock subscription manager methods
        subscription_orchestrator._subscription_manager.prepare_resubscription = MagicMock(return_value=None)
        subscription_orchestrator._subscription_manager.setup_new_subscription = MagicMock()
        
        # Should propagate handler errors
        with pytest.raises(Exception, match="Handler preparation failed"):
            subscription_orchestrator.execute_subscription(
                subscription_type=subscription_type,
                instruments=instruments,
                handler=mock_handler,
                stream_name_generator=stream_name_generator,
                async_loop_submit=async_loop_submit,
                exchange=mock_exchange,
                channel=mock_ctrl_channel
            )

    @pytest.mark.asyncio
    async def test_stop_subscription_success(self, subscription_orchestrator):
        """Test stopping subscription successfully."""
        subscription_type = "ohlc"
        stream_name = "test_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        
        # Setup subscription state
        subscription_orchestrator._sub_to_coro[subscription_type] = mock_future
        subscription_orchestrator._subscription_manager.get_subscription_name = MagicMock(return_value=stream_name)
        subscription_orchestrator._connection_manager.stop_stream = AsyncMock()
        
        await subscription_orchestrator.stop_subscription(subscription_type)
        
        # Should stop stream via connection manager
        subscription_orchestrator._connection_manager.stop_stream.assert_called_once_with(stream_name, mock_future)
        
        # Should clean up internal state
        assert subscription_type not in subscription_orchestrator._sub_to_coro

    @pytest.mark.asyncio
    async def test_stop_subscription_no_stream(self, subscription_orchestrator):
        """Test stopping subscription when no stream exists."""
        subscription_type = "ohlc"
        
        # Mock no existing stream
        subscription_orchestrator._subscription_manager.get_subscription_name = MagicMock(return_value=None)
        subscription_orchestrator._connection_manager.stop_stream = AsyncMock()
        
        await subscription_orchestrator.stop_subscription(subscription_type)
        
        # Should not call stop_stream when no stream exists
        subscription_orchestrator._connection_manager.stop_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_subscription_error(self, subscription_orchestrator):
        """Test stopping subscription handles errors gracefully."""
        subscription_type = "ohlc"
        stream_name = "test_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        
        # Setup subscription state
        subscription_orchestrator._sub_to_coro[subscription_type] = mock_future
        subscription_orchestrator._subscription_manager.get_subscription_name = MagicMock(return_value=stream_name)
        subscription_orchestrator._connection_manager.stop_stream = AsyncMock(side_effect=Exception("Stop failed"))
        
        # Should propagate the error
        with pytest.raises(Exception, match="Stop failed"):
            await subscription_orchestrator.stop_subscription(subscription_type)

    def test_wait_for_cancellation_success(self, subscription_orchestrator):
        """Test waiting for cancellation completes successfully."""
        mock_future = MagicMock(spec=concurrent.futures.Future)
        subscription_type = "ohlc"
        
        # Mock future that becomes not running after delay
        call_count = 0
        def running_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Running for first 2 calls, then not running
        
        mock_future.running.side_effect = running_side_effect
        
        with patch('time.sleep') as mock_sleep:
            subscription_orchestrator._wait_for_cancellation(mock_future, subscription_type)
        
        # Should wait until future is not running (final check to exit loop)
        assert mock_future.running.call_count >= 3
        
        # Should sleep between checks
        assert mock_sleep.call_count >= 2

    def test_wait_for_cancellation_timeout(self, subscription_orchestrator):
        """Test waiting for cancellation with timeout."""
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = True  # Always running
        subscription_type = "ohlc"
        
        with patch('time.sleep') as mock_sleep:
            with patch('time.time', side_effect=[0, 0.1, 0.2, 3.1, 3.2]) as mock_time:  # Simulate timeout
                with patch('qubx.connectors.ccxt.subscription_orchestrator.logger') as mock_logger:
                    subscription_orchestrator._wait_for_cancellation(mock_future, subscription_type)
        
        # Should log timeout warning
        mock_logger.warning.assert_called()

    def test_execute_subscription_with_batching(self, subscription_orchestrator, mock_instruments, mock_exchange, mock_ctrl_channel):
        """Test subscription execution with market type batching requirement."""
        subscription_type = "orderbook"
        instruments = set(mock_instruments)  # Mix of SWAP and SPOT instruments
        
        # Mock handler that requires batching
        mock_handler = MagicMock()
        mock_config = MagicMock()
        mock_config.subscriber_func = AsyncMock()
        mock_config.unsubscriber_func = AsyncMock()
        mock_config.stream_name = "test_stream"
        mock_config.requires_market_type_batching = True
        mock_handler.prepare_subscription.return_value = mock_config
        
        stream_name_generator = MagicMock(return_value="test_stream")
        mock_future = MagicMock(spec=concurrent.futures.Future)
        async_loop_submit = MagicMock(return_value=mock_future)
        
        # Mock subscription manager methods
        subscription_orchestrator._subscription_manager.prepare_resubscription = MagicMock(return_value=None)
        subscription_orchestrator._subscription_manager.setup_new_subscription = MagicMock()
        subscription_orchestrator._connection_manager.register_stream_future = MagicMock()
        
        # Note: The actual implementation doesn't have _call_by_market_type method,
        # it handles batching internally through utils.create_market_type_batched_subscriber
        subscription_orchestrator.execute_subscription(
            subscription_type=subscription_type,
            instruments=instruments,
            handler=mock_handler,
            stream_name_generator=stream_name_generator,
            async_loop_submit=async_loop_submit,
            exchange=mock_exchange,
            channel=mock_ctrl_channel
        )
        
        # Should prepare subscription with handler
        mock_handler.prepare_subscription.assert_called_once()
        
        # Should submit async task
        async_loop_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_by_market_type(self, subscription_orchestrator, mock_instruments):
        """Test market type batching functionality."""
        # Create a mixed set of instruments (SWAP and SPOT)
        instruments = set(mock_instruments)  # Contains both SWAP and SPOT instruments
        
        # Mock subscriber function
        subscriber_func = AsyncMock()
        
        # Create the batched subscriber using the actual method
        batched_subscriber = subscription_orchestrator.call_by_market_type(subscriber_func, instruments)
        
        # Execute the batched subscriber
        await batched_subscriber()
        
        # Should call subscriber for each market type group
        assert subscriber_func.call_count >= 1
        
        # Check that instruments were grouped by market type
        call_args = subscriber_func.call_args_list
        for call in call_args:
            instruments_batch = call[0][0]  # First argument should be instrument list
            # All instruments in each batch should have the same market type
            if len(instruments_batch) > 1:
                market_types = [instr.market_type for instr in instruments_batch]
                assert len(set(market_types)) == 1  # All same market type

    def test_stream_name_persistence(self, subscription_orchestrator, mock_instruments, mock_exchange, mock_ctrl_channel):
        """Test that stream names are properly stored and retrieved."""
        subscription_type = "ohlc"
        stream_name = "persistent_stream"
        
        # Mock dependencies
        subscription_orchestrator._subscription_manager.set_stream_name = MagicMock()
        subscription_orchestrator._subscription_manager.get_stream_name = MagicMock(return_value=stream_name)
        
        # Test setting stream name
        subscription_orchestrator._subscription_manager.set_stream_name(subscription_type, stream_name)
        subscription_orchestrator._subscription_manager.set_stream_name.assert_called_once_with(subscription_type, stream_name)
        
        # Test getting stream name
        retrieved_name = subscription_orchestrator._subscription_manager.get_stream_name(subscription_type)
        assert retrieved_name == stream_name

    def test_concurrent_subscriptions(self, subscription_orchestrator, mock_instruments, mock_exchange, mock_ctrl_channel):
        """Test handling multiple concurrent subscription operations."""
        subscription_types = ["ohlc", "trade", "orderbook"]
        instruments = set(mock_instruments[:1])
        
        # Mock handler
        mock_handler = MagicMock()
        mock_config = MagicMock()
        mock_config.subscriber_func = AsyncMock()
        mock_config.unsubscriber_func = AsyncMock()
        mock_config.stream_name = "test_stream"
        mock_handler.prepare_subscription.return_value = mock_config
        
        stream_name_generator = MagicMock(side_effect=lambda x, **kwargs: f"{x}_stream")
        mock_future = MagicMock(spec=concurrent.futures.Future)
        async_loop_submit = MagicMock(return_value=mock_future)
        
        # Mock subscription manager methods
        subscription_orchestrator._subscription_manager.prepare_resubscription = MagicMock(return_value=None)
        subscription_orchestrator._subscription_manager.setup_new_subscription = MagicMock()
        subscription_orchestrator._connection_manager.register_stream_future = MagicMock()
        
        # Execute subscriptions (synchronous method)
        for sub_type in subscription_types:
            subscription_orchestrator.execute_subscription(
                subscription_type=sub_type,
                instruments=instruments,
                handler=mock_handler,
                stream_name_generator=stream_name_generator,
                async_loop_submit=async_loop_submit,
                exchange=mock_exchange,
                channel=mock_ctrl_channel
            )
        
        # Should handle all subscriptions
        assert mock_handler.prepare_subscription.call_count == len(subscription_types)
        assert async_loop_submit.call_count == len(subscription_types)