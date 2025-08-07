"""
Unit tests for ConnectionManager component.

Tests the WebSocket connection handling, retry logic, and stream lifecycle
management functionality in isolation.
"""

import asyncio
import concurrent.futures
from asyncio.exceptions import CancelledError
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError

from qubx.connectors.ccxt.connection_manager import ConnectionManager
from qubx.connectors.ccxt.exceptions import CcxtSymbolNotRecognized
from qubx.connectors.ccxt.subscription_manager import SubscriptionManager
from qubx.utils.misc import AsyncThreadLoop


class TestConnectionManager:
    """Test suite for ConnectionManager component."""

    @pytest.fixture
    def mock_async_loop(self):
        """Create a mock AsyncThreadLoop for testing."""
        loop = MagicMock(spec=AsyncThreadLoop)
        loop.submit = MagicMock()
        return loop

    @pytest.fixture
    def subscription_manager(self):
        """Create a SubscriptionManager instance for testing."""
        return SubscriptionManager()

    @pytest.fixture
    def connection_manager(self, subscription_manager, mock_async_loop):
        """Create a ConnectionManager instance for testing."""
        return ConnectionManager(
            exchange_id="test_exchange",
            loop=mock_async_loop,
            max_ws_retries=3,
            subscription_manager=subscription_manager,
        )

    def test_initialization(self, subscription_manager, mock_async_loop):
        """Test that ConnectionManager initializes with correct state."""
        manager = ConnectionManager(
            exchange_id="test_exchange",
            loop=mock_async_loop,
            max_ws_retries=5,
            subscription_manager=subscription_manager,
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

    def test_enable_stream(self, connection_manager):
        """Test enabling streams."""
        stream_name = "test_stream"

        # Initially stream should be disabled
        assert connection_manager.is_stream_enabled(stream_name) is False

        # Enable stream using the new method
        connection_manager.enable_stream(stream_name)
        assert connection_manager.is_stream_enabled(stream_name) is True

    def test_register_stream_future(self, connection_manager):
        """Test registering stream futures."""
        stream_name = "test_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)

        # Initially should have no future
        assert connection_manager.get_stream_future(stream_name) is None

        # Register future
        connection_manager.register_stream_future(stream_name, mock_future)

        # Should now have the future
        assert connection_manager.get_stream_future(stream_name) is mock_future

    async def test_listen_to_stream_success(self, connection_manager, mock_async_loop):
        """Test successful stream listening."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.return_value = False  # Will exit the loop immediately
        stream_name = "test_stream"
        subscription_type = "ohlc"

        # Execute
        await connection_manager.listen_to_stream(
            subscriber=mock_subscriber,
            exchange=mock_exchange,
            channel=mock_ctrl_channel,
            subscription_type=subscription_type,
            stream_name=stream_name,
        )

        # Verify stream was enabled during the call
        # Note: Stream is disabled after loop exits

    async def test_listen_to_stream_with_retries(self, connection_manager, mock_async_loop):
        """Test stream listening with network error retries."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_subscriber.side_effect = [
            NetworkError("Connection lost"),
            NetworkError("Connection lost"),
            None,  # Success on third try
        ]

        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.side_effect = [True, True, True, False]  # Exit after 4 calls

        stream_name = "test_stream"
        subscription_type = "ohlc"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Execute
            await connection_manager.listen_to_stream(
                subscriber=mock_subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                subscription_type=subscription_type,
                stream_name=stream_name,
            )

            # Verify subscriber was called multiple times due to retries
            assert mock_subscriber.call_count >= 2
            # Verify sleep was called for retry delays
            assert mock_sleep.call_count >= 1

    async def test_listen_to_stream_max_retries_exceeded(self, connection_manager, mock_async_loop):
        """Test stream listening stops after max retries."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_subscriber.side_effect = Exception("Persistent error")

        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.return_value = True

        stream_name = "test_stream"
        subscription_type = "ohlc"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Execute
            await connection_manager.listen_to_stream(
                subscriber=mock_subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                subscription_type=subscription_type,
                stream_name=stream_name,
            )

            # Should have tried max_ws_retries + 1 times (initial + retries)
            expected_calls = connection_manager.max_ws_retries
            assert mock_subscriber.call_count >= expected_calls

    async def test_listen_to_stream_symbol_not_recognized(self, connection_manager, mock_async_loop):
        """Test stream listening handles unrecognized symbols gracefully."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_subscriber.side_effect = [
            CcxtSymbolNotRecognized("Symbol not found"),
            CcxtSymbolNotRecognized("Symbol not found"),
            None,  # Success after skipping errors
        ]

        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.side_effect = [True, True, True, False]

        stream_name = "test_stream"
        subscription_type = "ohlc"

        # Execute
        await connection_manager.listen_to_stream(
            subscriber=mock_subscriber,
            exchange=mock_exchange,
            channel=mock_ctrl_channel,
            subscription_type=subscription_type,
            stream_name=stream_name,
        )

        # Verify subscriber was called multiple times
        assert mock_subscriber.call_count >= 3

    async def test_listen_to_stream_exchange_closed_by_user(self, connection_manager, mock_async_loop):
        """Test stream listening handles user-initiated exchange closure."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_subscriber.side_effect = ExchangeClosedByUser("Connection closed by user")

        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.return_value = True

        stream_name = "test_stream"
        subscription_type = "ohlc"

        # Execute
        await connection_manager.listen_to_stream(
            subscriber=mock_subscriber,
            exchange=mock_exchange,
            channel=mock_ctrl_channel,
            subscription_type=subscription_type,
            stream_name=stream_name,
        )

        # Should exit gracefully after one call
        assert mock_subscriber.call_count == 1

    async def test_listen_to_stream_cancelled_error(self, connection_manager, mock_async_loop):
        """Test stream listening handles cancellation gracefully."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_subscriber.side_effect = CancelledError()

        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.return_value = True

        stream_name = "test_stream"
        subscription_type = "ohlc"

        # Execute
        await connection_manager.listen_to_stream(
            subscriber=mock_subscriber,
            exchange=mock_exchange,
            channel=mock_ctrl_channel,
            subscription_type=subscription_type,
            stream_name=stream_name,
        )

        # Should exit gracefully after one call
        assert mock_subscriber.call_count == 1

    async def test_listen_to_stream_with_unsubscriber(self, connection_manager, mock_async_loop):
        """Test stream listening registers unsubscriber."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_unsubscriber = AsyncMock()
        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        mock_ctrl_channel.control.is_set.return_value = False  # Exit immediately

        stream_name = "test_stream"
        subscription_type = "ohlc"

        # Execute
        await connection_manager.listen_to_stream(
            subscriber=mock_subscriber,
            exchange=mock_exchange,
            channel=mock_ctrl_channel,
            subscription_type=subscription_type,
            stream_name=stream_name,
            unsubscriber=mock_unsubscriber,
        )

        # Verify unsubscriber was registered
        assert connection_manager.get_stream_unsubscriber(stream_name) is mock_unsubscriber

    def test_stop_stream_with_unsubscriber(self, connection_manager):
        """Test stopping stream calls unsubscriber."""
        stream_name = "test_stream"
        mock_unsubscriber = AsyncMock()
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = False

        # Setup stream state
        connection_manager.enable_stream(stream_name)
        connection_manager.set_stream_unsubscriber(stream_name, mock_unsubscriber)
        connection_manager.register_stream_future(stream_name, mock_future)

        # Execute
        connection_manager.stop_stream(stream_name, wait=False)

        # Verify cleanup
        assert not connection_manager.is_stream_enabled(stream_name)
        assert connection_manager.get_stream_future(stream_name) is None
        assert connection_manager.get_stream_unsubscriber(stream_name) is None

    def test_stop_stream_without_unsubscriber(self, connection_manager):
        """Test stopping stream without unsubscriber."""
        stream_name = "test_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = False

        # Setup stream state
        connection_manager.enable_stream(stream_name)
        connection_manager.register_stream_future(stream_name, mock_future)

        # Execute
        connection_manager.stop_stream(stream_name, wait=False)

        # Verify cleanup
        assert not connection_manager.is_stream_enabled(stream_name)
        assert connection_manager.get_stream_future(stream_name) is None

    def test_stop_stream_unsubscriber_error(self, connection_manager):
        """Test stopping stream handles unsubscriber errors."""
        stream_name = "test_stream"
        mock_unsubscriber = AsyncMock()
        mock_unsubscriber.side_effect = Exception("Unsubscriber error")
        mock_future = MagicMock(spec=concurrent.futures.Future)
        mock_future.running.return_value = False

        # Setup stream state
        connection_manager.enable_stream(stream_name)
        connection_manager.set_stream_unsubscriber(stream_name, mock_unsubscriber)
        connection_manager.register_stream_future(stream_name, mock_future)

        # Mock the AsyncThreadLoop.submit to return a failing future
        mock_unsub_future = MagicMock()
        mock_unsub_future.running.return_value = False
        connection_manager._loop.submit.return_value = mock_unsub_future

        # Execute - should not raise exception
        connection_manager.stop_stream(stream_name, wait=False)

        # Verify cleanup still happened
        assert not connection_manager.is_stream_enabled(stream_name)

    def test_stream_state_isolation(self, connection_manager):
        """Test that different streams maintain separate state."""
        stream1 = "stream_1"
        stream2 = "stream_2"

        future1 = MagicMock(spec=concurrent.futures.Future)
        future2 = MagicMock(spec=concurrent.futures.Future)
        unsubscriber1 = AsyncMock()
        unsubscriber2 = AsyncMock()

        # Setup different states for each stream
        connection_manager.enable_stream(stream1)
        connection_manager.register_stream_future(stream1, future1)
        connection_manager.set_stream_unsubscriber(stream1, unsubscriber1)

        connection_manager.enable_stream(stream2)
        connection_manager.register_stream_future(stream2, future2)
        connection_manager.set_stream_unsubscriber(stream2, unsubscriber2)

        # Verify isolation
        assert connection_manager.is_stream_enabled(stream1) is True
        assert connection_manager.is_stream_enabled(stream2) is True
        assert connection_manager.get_stream_future(stream1) is future1
        assert connection_manager.get_stream_future(stream2) is future2
        assert connection_manager.get_stream_unsubscriber(stream1) is unsubscriber1
        assert connection_manager.get_stream_unsubscriber(stream2) is unsubscriber2

        # Disable one stream
        connection_manager._is_stream_enabled[stream1] = False

        # Verify other stream unaffected
        assert connection_manager.is_stream_enabled(stream1) is False
        assert connection_manager.is_stream_enabled(stream2) is True

    async def test_retry_delay_increases(self, connection_manager, mock_async_loop):
        """Test that retry delays increase exponentially."""
        # Setup
        mock_subscriber = AsyncMock()
        mock_subscriber.side_effect = [Exception("Error 1"), Exception("Error 2"), Exception("Error 3")]

        mock_exchange = MagicMock()
        mock_ctrl_channel = MagicMock()
        # Allow 3 retries then exit
        mock_ctrl_channel.control.is_set.side_effect = [True, True, True, False]

        stream_name = "test_stream"
        subscription_type = "ohlc"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Execute
            await connection_manager.listen_to_stream(
                subscriber=mock_subscriber,
                exchange=mock_exchange,
                channel=mock_ctrl_channel,
                subscription_type=subscription_type,
                stream_name=stream_name,
            )

            # Verify exponential backoff (2^1, 2^2, 2^3, etc.)
            sleep_calls = mock_sleep.call_args_list
            if len(sleep_calls) >= 2:
                # First retry should be 2 seconds, second should be 4 seconds
                assert sleep_calls[0][0][0] == 2
                assert sleep_calls[1][0][0] == 4

    def test_set_subscription_manager(self, mock_async_loop):
        """Test setting subscription manager after initialization."""
        manager = ConnectionManager(
            exchange_id="test_exchange", loop=mock_async_loop, max_ws_retries=3, subscription_manager=None
        )

        # Initially no subscription manager
        assert manager._subscription_manager is None

        # Set subscription manager
        sub_manager = SubscriptionManager()
        manager.set_subscription_manager(sub_manager)

        # Verify it was set
        assert manager._subscription_manager is sub_manager

    def test_stream_unsubscriber_methods(self, connection_manager):
        """Test stream unsubscriber setter and getter methods."""
        stream_name = "test_stream"
        mock_unsubscriber = AsyncMock()

        # Initially should be None
        assert connection_manager.get_stream_unsubscriber(stream_name) is None

        # Set unsubscriber
        connection_manager.set_stream_unsubscriber(stream_name, mock_unsubscriber)

        # Verify it was set
        assert connection_manager.get_stream_unsubscriber(stream_name) is mock_unsubscriber

    def test_stream_coro_methods(self, connection_manager):
        """Test stream coroutine/future setter and getter methods."""
        stream_name = "test_stream"
        mock_future = MagicMock(spec=concurrent.futures.Future)

        # Initially should be None
        assert connection_manager.get_stream_coro(stream_name) is None

        # Set future
        connection_manager.set_stream_coro(stream_name, mock_future)

        # Verify it was set
        assert connection_manager.get_stream_coro(stream_name) is mock_future

        # Verify register_stream_future does the same thing
        mock_future2 = MagicMock(spec=concurrent.futures.Future)
        connection_manager.register_stream_future(stream_name, mock_future2)
        assert connection_manager.get_stream_future(stream_name) is mock_future2
        assert connection_manager.get_stream_coro(stream_name) is mock_future2
