"""
Unit tests for WarmupService component.

Tests the warmup task coordination, error handling, and timeout functionality
for historical data warmup operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qubx.connectors.ccxt.handlers import DataTypeHandlerFactory
from qubx.connectors.ccxt.warmup_service import WarmupService
from qubx.core.basics import AssetType, CtrlChannel, Instrument, MarketType


class TestWarmupService:
    """Test suite for WarmupService component."""

    @pytest.fixture
    def mock_instruments(self):
        """Create test instruments for various markets."""
        return [
            Instrument(
                symbol="BTCUSDT",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange_manager="BINANCE.UM",
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
                exchange_manager="BINANCE.UM",
                base="ETH",
                quote="USDT",
                settle="USDT",
                exchange_symbol="ETH/USDT:USDT",
                tick_size=0.01,
                lot_size=0.001,
                min_size=0.001,
            ),
            Instrument(
                symbol="ADAUSDT",
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SPOT,
                exchange_manager="BINANCE",
                base="ADA",
                quote="USDT",
                settle="USDT",
                exchange_symbol="ADA/USDT",
                tick_size=0.0001,
                lot_size=1.0,
                min_size=1.0,
            ),
        ]

    @pytest.fixture
    def mock_ctrl_channel(self):
        """Create a mock control channel for testing."""
        channel = MagicMock(spec=CtrlChannel)
        channel.send = MagicMock()
        channel.send_async = AsyncMock()

        # Mock the control event for stream lifecycle
        control_event = MagicMock()
        control_event.is_set.return_value = True
        channel.control = control_event

        return channel

    @pytest.fixture
    def mock_async_thread_loop(self):
        """Create a mock AsyncThreadLoop for controlled testing."""

        class MockAsyncThreadLoop:
            def __init__(self):
                self.submitted_tasks = []
                self.running_futures = {}

            def submit(self, coro):
                """Submit a coroutine and return a mock Future."""
                future = MagicMock()
                future.running.return_value = True
                future.cancel = MagicMock()
                future.result = MagicMock(return_value=None)

                # Store the coroutine for inspection
                self.submitted_tasks.append(coro)
                self.running_futures[id(future)] = {"future": future, "coro": coro, "cancelled": False}

                return future

            def cancel_future(self, future):
                """Mark a future as cancelled."""
                future_id = id(future)
                if future_id in self.running_futures:
                    self.running_futures[future_id]["cancelled"] = True
                    future.running.return_value = False
                    future.cancel.return_value = True

            def cleanup(self):
                """Clean up all pending coroutines to prevent warnings."""
                for task_info in self.running_futures.values():
                    coro = task_info["coro"]
                    # Properly close coroutines
                    if asyncio.iscoroutine(coro):
                        coro.close()
                    elif hasattr(coro, "close"):
                        try:
                            coro.close()
                        except Exception:
                            pass  # Ignore cleanup errors

                # Also clean up submitted tasks directly
                for coro in self.submitted_tasks:
                    if asyncio.iscoroutine(coro):
                        coro.close()
                    elif hasattr(coro, "close"):
                        try:
                            coro.close()
                        except Exception:
                            pass  # Ignore cleanup errors

                self.submitted_tasks.clear()
                self.running_futures.clear()

        loop = MockAsyncThreadLoop()
        yield loop
        # Cleanup after test
        loop.cleanup()

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider for handler testing."""
        data_provider = MagicMock()
        data_provider._exchange_id = "test_exchange"
        data_provider._last_quotes = {}
        data_provider.channel = MagicMock()
        data_provider.time_provider = MagicMock()
        return data_provider

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock CCXT exchange for testing."""
        exchange = MagicMock()
        exchange.name = "test_exchange"
        return exchange

    @pytest.fixture
    def handler_factory(self, mock_data_provider, mock_exchange):
        """Create a DataTypeHandlerFactory instance for testing."""
        factory = DataTypeHandlerFactory(
            data_provider=mock_data_provider, exchange_manager=mock_exchange, exchange_id="test_exchange"
        )

        yield factory

        # Cleanup: clear any cached handlers to prevent issues
        factory.clear_cache()

    @pytest.fixture
    def warmup_service(self, handler_factory, mock_ctrl_channel, mock_async_thread_loop):
        """Create a WarmupService instance for testing."""
        service = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="test_exchange",
            async_loop=mock_async_thread_loop,
            warmup_timeout=30,
        )

        yield service

        # Cleanup: ensure any submitted coroutines are properly closed
        # This is handled by the mock_async_thread_loop fixture cleanup

    def test_initialization(self, handler_factory, mock_ctrl_channel, mock_async_thread_loop):
        """Test that WarmupService initializes correctly."""
        service = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="test_exchange",
            async_loop=mock_async_thread_loop,
            warmup_timeout=60,
        )

        assert service._handler_factory == handler_factory
        assert service._channel == mock_ctrl_channel
        assert service._exchange_id == "test_exchange"
        assert service._async_loop == mock_async_thread_loop
        assert service._warmup_timeout == 60

    def test_execute_warmup_empty(self, warmup_service):
        """Test executing warmup with empty warmup dictionary."""
        with patch("qubx.connectors.ccxt.warmup_service.logger") as mock_logger:
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

        with patch("qubx.connectors.ccxt.warmup_service.logger") as mock_logger:
            warmup_service.execute_warmup(warmups)

        # Should get handler for data type
        warmup_service._handler_factory.get_handler.assert_called_once_with("ohlc")

        # Should submit ONE warmup task (execute_all_warmups)
        assert len(warmup_service._async_loop.submitted_tasks) == 1

        # Should log warmup execution
        mock_logger.info.assert_called()

    def test_execute_warmup_multiple_types(self, warmup_service, mock_instruments):
        """Test executing warmup for multiple data types."""
        instrument = mock_instruments[0]
        warmups = {("ohlc", instrument): "1h", ("trade", instrument): "30m", ("orderbook", instrument): "15m"}

        # Mock handlers
        mock_ohlc_handler = MagicMock()
        mock_ohlc_handler.warmup = AsyncMock()
        mock_trade_handler = MagicMock()
        mock_trade_handler.warmup = AsyncMock()
        mock_orderbook_handler = MagicMock()
        mock_orderbook_handler.warmup = AsyncMock()

        def get_handler_side_effect(data_type):
            handlers = {"ohlc": mock_ohlc_handler, "trade": mock_trade_handler, "orderbook": mock_orderbook_handler}
            return handlers.get(data_type)

        warmup_service._handler_factory.get_handler = MagicMock(side_effect=get_handler_side_effect)

        with patch("qubx.connectors.ccxt.warmup_service.logger"):
            warmup_service.execute_warmup(warmups)

        # Should get handler for each data type
        assert warmup_service._handler_factory.get_handler.call_count == 3

        # Should submit ONE warmup task (execute_all_warmups) that gathers all tasks
        assert len(warmup_service._async_loop.submitted_tasks) == 1

    def test_execute_warmup_multiple_instruments_same_type(self, warmup_service, mock_instruments):
        """Test executing warmup for multiple instruments of same data type."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("ohlc", mock_instruments[1]): "1h",
            ("ohlc", mock_instruments[2]): "2h",  # Different period
        }

        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)

        with patch("qubx.connectors.ccxt.warmup_service.logger"):
            warmup_service.execute_warmup(warmups)

        # Should submit ONE warmup task that gathers all the grouped tasks
        assert len(warmup_service._async_loop.submitted_tasks) == 1

        # Should call warmup handler twice (once for 1h group with 2 instruments, once for 2h group with 1 instrument)
        assert mock_handler.warmup.call_count == 2

    def test_execute_warmup_handler_not_found(self, warmup_service, mock_instruments):
        """Test executing warmup when handler is not found."""
        instrument = mock_instruments[0]
        warmups = {("unknown_type", instrument): "1h"}

        # Mock handler factory returning None
        warmup_service._handler_factory.get_handler = MagicMock(return_value=None)

        with patch("qubx.connectors.ccxt.warmup_service.logger") as mock_logger:
            warmup_service.execute_warmup(warmups)

        # Should log warning about missing handler
        mock_logger.warning.assert_called()

        # Should also log about no valid handlers found
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("not supported" in call for call in warning_calls)
        assert any("No valid warmup handlers found" in call for call in warning_calls)

        # Should not submit any tasks
        assert len(warmup_service._async_loop.submitted_tasks) == 0

    def test_execute_warmup_handler_error(self, warmup_service, mock_instruments):
        """Test executing warmup when handler raises error."""
        instrument = mock_instruments[0]
        warmups = {("ohlc", instrument): "1h"}

        # Mock handler that raises error during warmup
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock(side_effect=Exception("Warmup failed"))
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)

        # Mock the submit method to simulate error propagation
        original_submit = warmup_service._async_loop.submit

        def submit_with_error(coro):
            future = original_submit(coro)
            future.result = MagicMock(side_effect=Exception("Warmup failed"))
            return future

        warmup_service._async_loop.submit = submit_with_error

        with patch("qubx.connectors.ccxt.warmup_service.logger") as mock_logger:
            with pytest.raises(Exception, match="Warmup failed"):
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

        # Mock the submit method to simulate timeout
        original_submit = warmup_service._async_loop.submit

        def submit_with_timeout(coro):
            future = original_submit(coro)
            future.result = MagicMock(side_effect=TimeoutError("Warmup timed out"))
            return future

        warmup_service._async_loop.submit = submit_with_timeout

        with patch("qubx.connectors.ccxt.warmup_service.logger") as mock_logger:
            with pytest.raises(TimeoutError, match="Warmup timed out"):
                warmup_service.execute_warmup(warmups)

        # Should log timeout error
        mock_logger.error.assert_called()

    def test_group_warmups_by_type_and_period(self, warmup_service, mock_instruments):
        """Test that warmups are properly grouped by data type and period."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("ohlc", mock_instruments[1]): "1h",  # Same type and period
            ("ohlc", mock_instruments[2]): "2h",  # Same type, different period
            ("trade", mock_instruments[0]): "1h",  # Different type, same period
        }

        # Mock handlers
        mock_ohlc_handler = MagicMock()
        mock_ohlc_handler.warmup = AsyncMock()
        mock_trade_handler = MagicMock()
        mock_trade_handler.warmup = AsyncMock()

        def get_handler_side_effect(data_type):
            handlers = {"ohlc": mock_ohlc_handler, "trade": mock_trade_handler}
            return handlers.get(data_type)

        warmup_service._handler_factory.get_handler = MagicMock(side_effect=get_handler_side_effect)

        with patch("qubx.connectors.ccxt.warmup_service.logger"):
            warmup_service.execute_warmup(warmups)

        # Should submit ONE execute_all_warmups task
        assert len(warmup_service._async_loop.submitted_tasks) == 1

        # Should call ohlc handler twice (for 1h and 2h periods)
        assert mock_ohlc_handler.warmup.call_count == 2

        # Should call trade handler once (for 1h period)
        assert mock_trade_handler.warmup.call_count == 1

        # Verify grouping: ohlc handler should be called with proper instrument sets
        ohlc_calls = mock_ohlc_handler.warmup.call_args_list

        # First call should be for 1h with instruments[0] and instruments[1]
        first_call_instruments = ohlc_calls[0][1]["instruments"]  # kwargs
        assert len(first_call_instruments) == 2
        assert mock_instruments[0] in first_call_instruments
        assert mock_instruments[1] in first_call_instruments

        # Second call should be for 2h with instruments[2]
        second_call_instruments = ohlc_calls[1][1]["instruments"]  # kwargs
        assert len(second_call_instruments) == 1
        assert mock_instruments[2] in second_call_instruments

    def test_warmup_with_concurrent_execution(self, warmup_service, mock_instruments):
        """Test warmup execution with concurrent tasks."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("trade", mock_instruments[1]): "30m",
            ("orderbook", mock_instruments[2]): "15m",
        }

        # Mock handlers
        mock_handlers = {}
        for data_type in ["ohlc", "trade", "orderbook"]:
            handler = MagicMock()
            handler.warmup = AsyncMock()
            mock_handlers[data_type] = handler

        warmup_service._handler_factory.get_handler = MagicMock(side_effect=lambda dt: mock_handlers.get(dt))

        with patch("qubx.connectors.ccxt.warmup_service.logger"):
            warmup_service.execute_warmup(warmups)

        # Should submit ONE execute_all_warmups task that gathers all handler tasks
        assert len(warmup_service._async_loop.submitted_tasks) == 1

        # Each handler should be called once
        for handler in mock_handlers.values():
            handler.warmup.assert_called_once()

    def test_warmup_state_isolation(self, handler_factory, mock_ctrl_channel, mock_async_thread_loop, mock_instruments):
        """Test that multiple warmup services don't interfere with each other."""
        # Create two separate warmup services
        service1 = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="exchange1",
            async_loop=mock_async_thread_loop,
            warmup_timeout=30,
        )

        service2 = WarmupService(
            handler_factory=handler_factory,
            channel=mock_ctrl_channel,
            exchange_id="exchange2",
            async_loop=mock_async_thread_loop,
            warmup_timeout=60,
        )

        warmups1 = {("ohlc", mock_instruments[0]): "1h"}
        warmups2 = {("trade", mock_instruments[1]): "30m"}

        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        handler_factory.get_handler = MagicMock(return_value=mock_handler)

        # Execute warmups on both services
        with patch("qubx.connectors.ccxt.warmup_service.logger"):
            service1.execute_warmup(warmups1)
            service2.execute_warmup(warmups2)

        # Both services should have executed independently (one call each)
        assert len(mock_async_thread_loop.submitted_tasks) == 2

    def test_execute_warmup_with_parameters(self, warmup_service, mock_instruments):
        """Test executing warmup with data type parameters."""
        instrument = mock_instruments[0]
        # Test with basic subscription type - the parameter format depends on DataType.from_str implementation
        warmups = {("ohlc", instrument): "1h"}

        # Mock handler
        mock_handler = MagicMock()
        mock_handler.warmup = AsyncMock()
        warmup_service._handler_factory.get_handler = MagicMock(return_value=mock_handler)

        with patch("qubx.connectors.ccxt.warmup_service.logger"):
            warmup_service.execute_warmup(warmups)

        # Should get handler for base data type
        warmup_service._handler_factory.get_handler.assert_called_once_with("ohlc")

        # Should call warmup with correct parameters
        mock_handler.warmup.assert_called_once()
        call_kwargs = mock_handler.warmup.call_args[1]
        assert call_kwargs["warmup_period"] == "1h"
        assert "instruments" in call_kwargs
        assert "channel" in call_kwargs

    def test_execute_warmup_mixed_success_and_failure(self, warmup_service, mock_instruments):
        """Test that if some handlers exist and others don't, execution still proceeds."""
        warmups = {
            ("ohlc", mock_instruments[0]): "1h",
            ("unknown_type", mock_instruments[1]): "30m",
            ("trade", mock_instruments[2]): "15m",
        }

        # Mock handlers - only ohlc and trade exist
        mock_ohlc_handler = MagicMock()
        mock_ohlc_handler.warmup = AsyncMock()
        mock_trade_handler = MagicMock()
        mock_trade_handler.warmup = AsyncMock()

        def get_handler_side_effect(data_type):
            handlers = {"ohlc": mock_ohlc_handler, "trade": mock_trade_handler}
            return handlers.get(data_type)  # Returns None for unknown_type

        warmup_service._handler_factory.get_handler = MagicMock(side_effect=get_handler_side_effect)

        with patch("qubx.connectors.ccxt.warmup_service.logger") as mock_logger:
            warmup_service.execute_warmup(warmups)

        # Should log warning for unknown type
        mock_logger.warning.assert_called()

        # Should still execute for valid handlers
        assert len(warmup_service._async_loop.submitted_tasks) == 1

        # Both valid handlers should be called
        mock_ohlc_handler.warmup.assert_called_once()
        mock_trade_handler.warmup.assert_called_once()
