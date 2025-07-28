"""
Unit tests for WarmupService component.

Tests the warmup task coordination, error handling, and timeout functionality
for historical data warmup operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.connectors.ccxt.warmup_service import WarmupService


class TestWarmupService:
    """Test suite for WarmupService component."""

    def test_initialization(self, handler_factory, mock_ctrl_channel, mock_async_thread_loop):
        """Test that WarmupService initializes correctly."""
        service = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="test_exchange",
            async_loop=mock_async_thread_loop,
            warmup_timeout=60
        )
        
        assert service._handler_factory == handler_factory
        assert service._channel == mock_ctrl_channel
        assert service._exchange_id == "test_exchange"
        assert service._async_loop == mock_async_thread_loop
        assert service._warmup_timeout == 60

    def test_execute_warmup_empty(self, warmup_service):
        """Test executing warmup with empty warmup dictionary."""
        with patch('qubx.connectors.ccxt.warmup_service.logger') as mock_logger:
            warmup_service.execute_warmup({})
        
        # Should log that no warmups are needed
        mock_logger.debug.assert_called()

    def test_execute_warmup_single_type(self, warmup_service, mock_instruments):
        """Test executing warmup for single data type."""
        instrument = mock_instruments[0]
        warmups = {("ohlc", instrument): "1h"}
        
        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Mock async execution
        warmup_service._async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock()
        warmup_service._async_loop.submit.return_value = mock_future
        
        with patch('qubx.connectors.ccxt.warmup_service.logger') as mock_logger:
            warmup_service.execute_warmup(warmups)
        
        # Should get handler for data type
        warmup_service._handler_factory.get_handler.assert_called_once_with("ohlc")
        
        # Should submit warmup task
        warmup_service._async_loop.submit.assert_called_once()
        
        # Should wait for result with timeout
        mock_future.result.assert_called_once_with(warmup_service._warmup_timeout)
        
        # Should log warmup execution
        mock_logger.info.assert_called()

    def test_execute_warmup_multiple_types(self, warmup_service, mock_instruments):
        """Test executing warmup for multiple data types."""
        instrument = mock_instruments[0]
        warmups = {
            ("ohlc", instrument): "1h",
            ("trade", instrument): "30m",
            ("orderbook", instrument): "15m"
        }
        
        # Mock handlers
        mock_ohlc_handler = MagicMock()
        mock_ohlc_handler.warmup = AsyncMock()
        mock_trade_handler = MagicMock()
        mock_trade_handler.warmup = AsyncMock()
        mock_orderbook_handler = MagicMock()
        mock_orderbook_handler.warmup = AsyncMock()
        
        def get_handler_side_effect(data_type):
            handlers = {
                "ohlc": mock_ohlc_handler,
                "trade": mock_trade_handler,
                "orderbook": mock_orderbook_handler
            }
            return handlers.get(data_type)
        
        warmup_service._handler_factory.get_handler = MagicMock(side_effect=get_handler_side_effect)
        
        # Mock async execution
        warmup_service._async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock()
        warmup_service._async_loop.submit.return_value = mock_future
        
        with patch('qubx.connectors.ccxt.warmup_service.logger'):
            warmup_service.execute_warmup(warmups)
        
        # Should get handler for each data type
        assert warmup_service._handler_factory.get_handler.call_count == 3
        
        # Should submit warmup tasks (grouped by data type)
        assert warmup_service._async_loop.submit.call_count >= 1

    def test_execute_warmup_multiple_instruments_same_type(self, warmup_service, mock_instruments):
        """Test executing warmup for multiple instruments of same data type."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("ohlc", mock_instruments[1]): "1h",
            ("ohlc", mock_instruments[2]): "2h"  # Different period
        }
        
        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Mock async execution
        warmup_service._async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock()
        warmup_service._async_loop.submit.return_value = mock_future
        
        with patch('qubx.connectors.ccxt.warmup_service.logger'):
            warmup_service.execute_warmup(warmups)
        
        # Should group by data type and period
        assert warmup_service._async_loop.submit.call_count >= 2  # At least 2 groups (1h and 2h)

    def test_execute_warmup_handler_not_found(self, warmup_service, mock_instruments):
        """Test executing warmup when handler is not found."""
        instrument = mock_instruments[0]
        warmups = {("unknown_type", instrument): "1h"}
        
        # Mock handler factory returning None
        warmup_service._handler_factory.get_handler = MagicMock(return_value=None)
        
        with patch('qubx.connectors.ccxt.warmup_service.logger') as mock_logger:
            warmup_service.execute_warmup(warmups)
        
        # Should log warning about missing handler
        mock_logger.warning.assert_called()
        
        # Should not submit any tasks
        warmup_service._async_loop.submit.assert_not_called()

    def test_execute_warmup_handler_error(self, warmup_service, mock_instruments):
        """Test executing warmup when handler raises error."""
        instrument = mock_instruments[0]
        warmups = {("ohlc", instrument): "1h"}
        
        # Mock handler that raises error during warmup
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock(side_effect=Exception("Warmup failed"))
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Mock async execution that propagates the error
        warmup_service._async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock(side_effect=Exception("Warmup failed"))
        warmup_service._async_loop.submit.return_value = mock_future
        
        with patch('qubx.connectors.ccxt.warmup_service.logger') as mock_logger:
            warmup_service.execute_warmup(warmups)
        
        # Should log error
        mock_logger.error.assert_called()

    def test_execute_warmup_timeout(self, warmup_service, mock_instruments):
        """Test executing warmup with timeout."""
        instrument = mock_instruments[0]
        warmups = {("ohlc", instrument): "1h"}
        
        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Mock async execution with timeout
        warmup_service._async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock(side_effect=TimeoutError("Warmup timed out"))
        warmup_service._async_loop.submit.return_value = mock_future
        
        with patch('qubx.connectors.ccxt.warmup_service.logger') as mock_logger:
            warmup_service.execute_warmup(warmups)
        
        # Should log timeout error
        mock_logger.error.assert_called()

    def test_group_warmups_by_type_and_period(self, warmup_service, mock_instruments):
        """Test that warmups are properly grouped by data type and period."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("ohlc", mock_instruments[1]): "1h",  # Same type and period
            ("ohlc", mock_instruments[2]): "2h",  # Same type, different period
            ("trade", mock_instruments[0]): "1h", # Different type, same period
        }
        
        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Track warmup calls
        warmup_calls = []
        
        def capture_warmup_calls(coro):
            # Extract the coroutine to inspect its arguments
            warmup_calls.append(coro)
            mock_future = MagicMock()
            mock_future.result = MagicMock()
            return mock_future
        
        warmup_service._async_loop.submit = MagicMock(side_effect=capture_warmup_calls)
        
        with patch('qubx.connectors.ccxt.warmup_service.logger'):
            warmup_service.execute_warmup(warmups)
        
        # Should create separate tasks for each (type, period) combination
        # Expected groups: (ohlc, 1h), (ohlc, 2h), (trade, 1h)
        assert len(warmup_calls) >= 3

    def test_warmup_with_concurrent_execution(self, warmup_service, mock_instruments):
        """Test warmup execution with concurrent tasks."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("trade", mock_instruments[1]): "30m",
            ("orderbook", mock_instruments[2]): "15m"
        }
        
        # Mock handlers
        mock_handlers = {}
        for data_type in ["ohlc", "trade", "orderbook"]:
            handler = MagicMock()
            handler.warmup = AsyncMock()
            mock_handlers[data_type] = handler
        
        warmup_service._handler_factory.get_handler = MagicMock(
            side_effect=lambda dt: mock_handlers.get(dt)
        )
        
        # Mock async execution
        completed_tasks = []
        
        def submit_and_track(coro):
            completed_tasks.append(coro)
            mock_future = MagicMock()
            mock_future.result = MagicMock()
            return mock_future
        
        warmup_service._async_loop.submit = MagicMock(side_effect=submit_and_track)
        
        with patch('qubx.connectors.ccxt.warmup_service.logger'):
            warmup_service.execute_warmup(warmups)
        
        # Should execute all warmup tasks
        assert len(completed_tasks) == 3

    def test_warmup_progress_logging(self, warmup_service, mock_instruments):
        """Test that warmup progress is properly logged."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("trade", mock_instruments[1]): "30m"
        }
        
        # Mock handlers
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Mock async execution
        warmup_service._async_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock()
        warmup_service._async_loop.submit.return_value = mock_future
        
        with patch('qubx.connectors.ccxt.warmup_service.logger') as mock_logger:
            warmup_service.execute_warmup(warmups)
        
        # Should log warmup start and completion
        info_calls = [call for call in mock_logger.info.call_args_list]
        assert len(info_calls) >= 1  # At least log warmup execution

    def test_warmup_state_isolation(self, handler_factory, mock_ctrl_channel, mock_async_thread_loop, mock_instruments):
        """Test that multiple warmup services don't interfere with each other."""
        # Create two separate warmup services
        service1 = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="exchange1",
            async_loop=mock_async_thread_loop,
            warmup_timeout=30
        )
        
        service2 = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="exchange2",
            async_loop=mock_async_thread_loop,
            warmup_timeout=60
        )
        
        warmups1 = {("ohlc", mock_instruments[0]): "1h"}
        warmups2 = {("trade", mock_instruments[1]): "30m"}
        
        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        handler_factory.get_handler = MagicMock(return_value=mock_handler)
        
        # Mock async execution
        mock_async_thread_loop.submit = MagicMock()
        mock_future = MagicMock()
        mock_future.result = MagicMock()
        mock_async_thread_loop.submit.return_value = mock_future
        
        # Execute warmups on both services
        with patch('qubx.connectors.ccxt.warmup_service.logger'):
            service1.execute_warmup(warmups1)
            service2.execute_warmup(warmups2)
        
        # Both services should have executed independently
        assert mock_async_thread_loop.submit.call_count == 2
        
        # Each service should use its own timeout
        result_calls = mock_future.result.call_args_list
        timeouts = [call[0][0] for call in result_calls]
        assert 30 in timeouts  # service1 timeout
        assert 60 in timeouts  # service2 timeout