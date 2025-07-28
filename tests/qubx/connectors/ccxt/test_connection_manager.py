"""
Unit tests for ConnectionManager component.

Tests the WebSocket connection handling, retry logic, and stream lifecycle
management functionality in isolation.
"""

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock, patch, call
from asyncio.exceptions import CancelledError

import pytest
from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.exceptions import CcxtSymbolNotRecognized


class TestConnectionManager:
    """Test suite for ConnectionManager component."""

    def test_initialization(self, subscription_manager):
        """Test that ConnectionManager initializes with correct state."""
        manager = ConnectionManager(
            exchange_id="test_exchange",
            max_ws_retries=5,
            subscription_manager=subscription_manager
        )
        
        assert manager._exchange_id == "test_exchange"
        assert manager.max_ws_retries == 5
        assert manager._subscription_manager == subscription_manager
        assert len(manager._is_stream_enabled) == 0
        assert len(manager._stream_to_unsubscriber) == 0
        assert len(manager._stream_to_coro) == 0

    def test_is_stream_enabled_default_false(self, connection_manager):
        """Test that streams are disabled by default."""
        stream_name = "test_stream"
        
        # Should default to False
        assert connection_manager.is_stream_enabled(stream_name) is False

    def test_disable_stream(self, connection_manager):
        """Test disabling streams."""
        stream_name = "test_stream"
        
        # Initially stream should be disabled
        assert connection_manager.is_stream_enabled(stream_name) is False
        
        # Manually enable stream (this would normally happen in listen_to_stream)
        connection_manager._is_stream_enabled[stream_name] = True
        assert connection_manager.is_stream_enabled(stream_name) is True
        
        # Disable stream
        connection_manager.disable_stream(stream_name)
        assert connection_manager.is_stream_enabled(stream_name) is False

    def test_register_stream_future(self, connection_manager):
        """Test registering stream future."""
        stream_name = "test_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        
        connection_manager.register_stream_future(stream_name, mock_future)
        
        assert connection_manager.get_stream_future(stream_name) == mock_future

    @pytest.mark.asyncio
    async def test_listen_to_stream_success(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test successful stream listening without retries."""
        stream_name = "test_stream"
        
        # Mock successful subscriber
        subscriber = AsyncMock()
        
        # Mock subscription manager method
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,  
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Subscriber should be called once
        subscriber.assert_called_once()
        
        # Should log success
        mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_listen_to_stream_with_retries(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test stream listening with network error retries."""
        stream_name = "test_stream"
        
        # Mock subscriber that fails twice then succeeds
        subscriber = AsyncMock()
        subscriber.side_effect = [
            NetworkError("Connection failed"),
            ExchangeNotAvailable("Exchange temporarily unavailable"),
            None  # Success on third try
        ]
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Subscriber should be called 3 times (2 failures + 1 success)
        assert subscriber.call_count == 3
        
        # Should log retry attempts
        assert mock_logger.warning.call_count >= 2

    @pytest.mark.asyncio
    async def test_listen_to_stream_max_retries_exceeded(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test stream listening when max retries are exceeded."""
        stream_name = "test_stream"
        
        # Mock subscriber that always fails
        subscriber = AsyncMock()
        subscriber.side_effect = NetworkError("Persistent connection failure")
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Subscriber should be called max_ws_retries + 1 times
        expected_calls = connection_manager.max_ws_retries + 1
        assert subscriber.call_count == expected_calls
        
        # Should log max retries exceeded
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_listen_to_stream_symbol_not_recognized(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test handling of symbol not recognized error (no retries)."""
        stream_name = "test_stream"
        
        # Mock subscriber that fails with symbol error
        subscriber = AsyncMock()
        subscriber.side_effect = CcxtSymbolNotRecognized("Symbol not found")
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Subscriber should be called only once (no retries for symbol errors)
        subscriber.assert_called_once()
        
        # Should log error without retries
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_listen_to_stream_exchange_closed_by_user(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test handling of exchange closed by user (no retries)."""
        stream_name = "test_stream"
        
        # Mock subscriber that fails with user-closed error
        subscriber = AsyncMock()
        subscriber.side_effect = ExchangeClosedByUser("User closed connection")
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Subscriber should be called only once (no retries for user-closed)
        subscriber.assert_called_once()

    @pytest.mark.asyncio 
    async def test_listen_to_stream_cancelled_error(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test handling of cancelled error (clean shutdown)."""
        stream_name = "test_stream"
        
        # Mock subscriber that gets cancelled
        subscriber = AsyncMock()
        subscriber.side_effect = CancelledError("Task was cancelled")
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Subscriber should be called once
        subscriber.assert_called_once()
        
        # Should log clean shutdown
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_listen_to_stream_with_unsubscriber(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test stream listening with unsubscriber function."""
        stream_name = "test_stream"
        
        subscriber = AsyncMock()
        unsubscriber = AsyncMock()
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        await connection_manager.listen_to_stream(
            subscriber=subscriber,
            exchange=mock_exchange,
            channel=mock_ctrl_channel,
            stream_name=stream_name,
            unsubscriber=unsubscriber
        )
        
        # Should store unsubscriber
        assert connection_manager.get_stream_unsubscriber(stream_name) == unsubscriber

    @pytest.mark.asyncio
    async def test_stop_stream_with_unsubscriber(self, connection_manager):
        """Test stopping stream calls unsubscriber."""
        stream_name = "test_stream"
        unsubscriber = AsyncMock()
        
        # Setup stream with unsubscriber
        connection_manager.set_stream_unsubscriber(stream_name, unsubscriber)
        connection_manager.enable_stream(stream_name)
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.stop_stream(stream_name)
        
        # Should call unsubscriber
        unsubscriber.assert_called_once()
        
        # Should disable stream
        assert connection_manager.is_stream_enabled(stream_name) is False

    @pytest.mark.asyncio
    async def test_stop_stream_without_unsubscriber(self, connection_manager):
        """Test stopping stream without unsubscriber."""
        stream_name = "test_stream"
        
        # Enable stream without unsubscriber
        connection_manager.enable_stream(stream_name)
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.stop_stream(stream_name)
        
        # Should disable stream
        assert connection_manager.is_stream_enabled(stream_name) is False

    @pytest.mark.asyncio
    async def test_stop_stream_unsubscriber_error(self, connection_manager):
        """Test stopping stream handles unsubscriber errors gracefully."""
        stream_name = "test_stream"
        unsubscriber = AsyncMock()
        unsubscriber.side_effect = Exception("Unsubscriber failed")
        
        # Setup stream with failing unsubscriber
        connection_manager.set_stream_unsubscriber(stream_name, unsubscriber)
        connection_manager.enable_stream(stream_name)
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.stop_stream(stream_name)
        
        # Should still disable stream despite unsubscriber error
        assert connection_manager.is_stream_enabled(stream_name) is False
        
        # Should log error
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_stop_old_stream_success(self, connection_manager):
        """Test stopping old stream successfully."""
        old_name = "old_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = True
        mock_future.cancel.return_value = True
        
        # Setup old stream
        connection_manager.set_stream_coro(old_name, mock_future)
        connection_manager.enable_stream(old_name)
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.stop_old_stream(old_name, mock_future)
        
        # Should cancel future
        mock_future.cancel.assert_called_once()
        
        # Should disable stream
        assert connection_manager.is_stream_enabled(old_name) is False

    @pytest.mark.asyncio
    async def test_stop_old_stream_not_running(self, connection_manager):
        """Test stopping old stream that's not running."""
        old_name = "old_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = False
        
        # Setup old stream
        connection_manager.set_stream_coro(old_name, mock_future)
        connection_manager.enable_stream(old_name)
        
        with patch('qubx.connectors.ccxt.connection_manager.logger') as mock_logger:
            await connection_manager.stop_old_stream(old_name, mock_future)
        
        # Should not try to cancel if not running
        mock_future.cancel.assert_not_called()
        
        # Should still disable stream
        assert connection_manager.is_stream_enabled(old_name) is False

    @pytest.mark.asyncio
    async def test_stop_old_stream_with_timeout(self, connection_manager):
        """Test stopping old stream with cancellation timeout."""
        old_name = "old_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = True
        mock_future.cancel.return_value = True
        
        # Setup old stream
        connection_manager.set_stream_coro(old_name, mock_future)
        
        # Mock asyncio.sleep to control timeout
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await connection_manager.stop_old_stream(old_name, mock_future)
        
        # Should wait for cancellation
        mock_sleep.assert_called()

    def test_stream_state_isolation(self, connection_manager):
        """Test that stream state is properly isolated between streams."""
        stream1 = "stream1"
        stream2 = "stream2"
        
        unsubscriber1 = AsyncMock()
        unsubscriber2 = AsyncMock()
        future1 = MagicMock()
        future2 = MagicMock()
        
        # Setup different states for each stream
        connection_manager.enable_stream(stream1)
        connection_manager.set_stream_unsubscriber(stream1, unsubscriber1)
        connection_manager.set_stream_coro(stream1, future1)
        
        connection_manager.set_stream_unsubscriber(stream2, unsubscriber2)
        connection_manager.set_stream_coro(stream2, future2)
        # Note: stream2 is not enabled
        
        # Verify isolation
        assert connection_manager.is_stream_enabled(stream1) is True
        assert connection_manager.is_stream_enabled(stream2) is False
        assert connection_manager.get_stream_unsubscriber(stream1) == unsubscriber1
        assert connection_manager.get_stream_unsubscriber(stream2) == unsubscriber2
        assert connection_manager.get_stream_coro(stream1) == future1
        assert connection_manager.get_stream_coro(stream2) == future2

    @pytest.mark.asyncio
    async def test_retry_delay_increases(self, connection_manager, mock_exchange, mock_ctrl_channel):
        """Test that retry delay increases with each attempt."""
        stream_name = "test_stream"
        
        # Mock subscriber that fails multiple times
        subscriber = AsyncMock()
        subscriber.side_effect = [
            NetworkError("Fail 1"),
            NetworkError("Fail 2"), 
            None  # Success
        ]
        
        connection_manager._subscription_manager.mark_subscription_active = MagicMock()
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await connection_manager.listen_to_stream(
                subscriber=subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                stream_name=stream_name
            )
        
        # Should have called sleep with increasing delays
        assert mock_sleep.call_count == 2  # Two retries
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        
        # Delays should increase (1, 2, 4, ...)
        assert sleep_calls[0] == 1
        assert sleep_calls[1] == 2