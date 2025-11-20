import time
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.basics import CtrlChannel, Instrument, dt_64
from qubx.core.interfaces import IMetricEmitter, ITimeProvider, LatencyMetrics
from qubx.core.lookups import lookup
from qubx.health.base import BaseHealthMonitor
from qubx.health.dummy import DummyHealthMonitor


def _get_test_instrument(symbol: str = "BTCUSDT", exchange: str = "BINANCE.UM") -> Instrument:
    """Get a test instrument for testing."""
    instr = lookup.find_symbol(exchange, symbol)
    assert instr is not None, f"Could not find {symbol} on {exchange}"
    return instr


class MockTimeProvider(ITimeProvider):
    def __init__(self, start_time: datetime | None = None):
        self._current_time = start_time or datetime(2024, 1, 1)

    def time(self) -> dt_64:
        return np.datetime64(self._current_time)

    def advance(self, delta: timedelta):
        self._current_time += delta


class TestDummyHealthMonitor:
    def test_dummy_monitor_returns_zero_values(self) -> None:
        monitor = DummyHealthMonitor()
        test_instrument = _get_test_instrument()

        # Test all methods return expected zero/empty values
        assert monitor.get_queue_size() == 0
        assert monitor.get_data_latency("test_exchange", "test_event") == 0.0
        assert monitor.get_data_latencies("test_exchange") == {}
        assert monitor.get_order_submit_latency("test_exchange") == 0.0
        assert monitor.get_order_cancel_latency("test_exchange") == 0.0
        assert monitor.get_event_frequency(test_instrument, "test_event") == 1.0
        assert monitor.get_execution_latency("test_scope") == 0.0
        assert monitor.get_execution_latencies() == {}
        assert monitor.is_connected("test_exchange") is True
        assert monitor.get_last_event_time(test_instrument, "test_event") is None

        # Test get_exchange_latencies returns LatencyMetrics
        metrics = monitor.get_exchange_latencies("test_exchange")
        assert isinstance(metrics, LatencyMetrics)
        assert metrics.data_feed == 0.0
        assert metrics.order_submit == 0.0
        assert metrics.order_cancel == 0.0

        # Test context manager interface
        with monitor("test_event") as m:
            assert m == monitor
            time.sleep(0.1)  # Simulate some work

        # Verify no state change after operations
        assert monitor.get_data_latency("test_exchange", "test_event") == 0.0

    def test_dummy_monitor_watch_decorator(self) -> None:
        """Test that DummyHealthMonitor's watch decorator returns function unchanged."""
        monitor = DummyHealthMonitor()

        # Define a test function
        def test_function(x, y):
            return x + y

        # Get decorated function
        decorated = monitor.watch()(test_function)

        # Verify the decorated function is the original function
        assert decorated is test_function

        # Verify function behavior is unchanged
        assert decorated(1, 2) == 3

        # Test with named scope as well
        decorated_named = monitor.watch("test_scope")(test_function)
        assert decorated_named is test_function
        assert decorated_named(3, 4) == 7


class TestBaseHealthMonitor:
    @pytest.fixture
    def time_provider(self) -> MockTimeProvider:
        return MockTimeProvider()

    @pytest.fixture
    def monitor(self, time_provider: MockTimeProvider) -> Generator[BaseHealthMonitor, None, None]:
        monitor = BaseHealthMonitor(time_provider)
        monitor.start()
        yield monitor
        monitor.stop()

    @pytest.fixture
    def monitor_with_custom_interval(self, time_provider: MockTimeProvider) -> Generator[BaseHealthMonitor, None, None]:
        # Create monitor with faster interval for testing
        monitor = BaseHealthMonitor(time_provider, emit_interval="100ms")
        monitor.start()
        yield monitor
        monitor.stop()

    def test_queue_size_tracking(self, monitor: BaseHealthMonitor) -> None:
        monitor.set_event_queue_size(5)
        assert monitor.get_queue_size() == 5

        monitor.set_event_queue_size(10)
        assert monitor.get_queue_size() == 10

    def test_queue_size_overflow(self, monitor: BaseHealthMonitor) -> None:
        """Test that queue size tracking works correctly when buffer is full."""
        # Fill the buffer
        for i in range(2000):  # More than the default window size
            monitor.set_event_queue_size(i)

        # Verify we only keep the most recent values
        assert monitor.get_queue_size() == 1999  # Most recent value

    def test_event_frequency_tracking(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        instrument = _get_test_instrument()
        event_type = "test_event"

        # Subscribe first
        monitor.subscribe(instrument, event_type)

        # Record a few events
        for _ in range(3):
            monitor.on_data_arrival(instrument, event_type, time_provider.time())
            time_provider.advance(timedelta(milliseconds=100))

        # Should have non-zero frequency
        freq = monitor.get_event_frequency(instrument, event_type)
        assert freq > 0

    def test_event_frequency_window(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that event frequency only counts events in the last second."""
        instrument = _get_test_instrument()
        event_type = "test_event"

        # Subscribe first
        monitor.subscribe(instrument, event_type)

        # Record events spread over 2 seconds
        for i in range(20):
            monitor.on_data_arrival(instrument, event_type, time_provider.time())
            time_provider.advance(timedelta(milliseconds=250))  # 4 events per second

        # Should only count events in the last second (4 events)
        freq = monitor.get_event_frequency(instrument, event_type)
        assert 3.5 <= freq <= 4.5  # Allow some floating point imprecision

    def test_context_manager_timing(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        event_type = "test_event"

        with monitor(event_type):
            # Simulate some work by advancing the mock time
            time_provider.advance(timedelta(milliseconds=100))

        # Should have recorded processing time
        assert monitor.get_execution_latency(event_type) > 0

    def test_nested_context_managers(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that nested context managers work correctly."""
        outer_event = "outer_event"
        inner_event = "inner_event"

        with monitor(outer_event):
            time_provider.advance(timedelta(milliseconds=50))

            with monitor(inner_event):
                time_provider.advance(timedelta(milliseconds=25))

            # Inner event should have ~25ms latency
            assert 20 <= monitor.get_execution_latency(inner_event) <= 30

            time_provider.advance(timedelta(milliseconds=25))

        # Outer event should have ~100ms latency
        assert 95 <= monitor.get_execution_latency(outer_event) <= 105

    def test_context_manager_exception_handling(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that context managers handle exceptions correctly."""
        event_type = "test_event"

        try:
            with monitor(event_type):
                time_provider.advance(timedelta(milliseconds=50))
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still record the timing even if an exception occurred
        assert 45 <= monitor.get_execution_latency(event_type) <= 55

    def test_monitor_start_stop(self, time_provider: MockTimeProvider) -> None:
        """Test that monitor can be started and stopped multiple times."""
        monitor = BaseHealthMonitor(time_provider)

        # Start and record some events
        monitor.start()
        monitor.set_event_queue_size(5)
        assert monitor.get_queue_size() == 5

        # Stop and verify we can still read metrics
        monitor.stop()
        assert monitor.get_queue_size() == 5

        # Start again and verify we can record new events
        monitor.start()
        monitor.set_event_queue_size(10)
        assert monitor.get_queue_size() == 10

        # Clean up
        monitor.stop()

    def test_custom_emit_interval(self, time_provider: MockTimeProvider) -> None:
        """Test that emit_interval parameter is properly recognized and applied."""
        # Test with different interval formats
        monitor1 = BaseHealthMonitor(time_provider, emit_interval="1s")
        assert monitor1._emit_interval_s == 1.0

        monitor2 = BaseHealthMonitor(time_provider, emit_interval="500ms")
        assert monitor2._emit_interval_s == 0.5

        monitor3 = BaseHealthMonitor(time_provider, emit_interval="2m")
        assert monitor3._emit_interval_s == 120.0

    def test_execution_latency_tracking(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that execution latency is tracked correctly using context manager."""
        scope = "test_scope"

        # Use context manager to time an operation
        with monitor(scope):
            # Simulate some work by advancing time
            time_provider.advance(timedelta(milliseconds=75))

        # Check that we can retrieve the execution latency
        latency = monitor.get_execution_latency(scope)
        assert 70 <= latency <= 80  # Should be ~75ms

        # Check that we can retrieve all execution latencies
        latencies = monitor.get_execution_latencies()
        assert scope in latencies
        assert 70 <= latencies[scope] <= 80

    def test_execution_latency_percentiles(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that execution latency percentiles are calculated correctly."""
        scope = "test_scope"

        # Generate a series of operations with different durations
        durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms

        for duration in durations:
            with monitor(scope):
                time_provider.advance(timedelta(milliseconds=duration))

        # Check different percentiles
        assert 45 <= monitor.get_execution_latency(scope, 50) <= 55  # median should be ~50ms
        assert 85 <= monitor.get_execution_latency(scope, 90) <= 95  # 90th percentile should be ~90ms
        assert 95 <= monitor.get_execution_latency(scope, 99) <= 105  # 99th percentile should be ~100ms

    def test_multiple_execution_scopes(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test tracking of multiple execution scopes simultaneously."""
        scopes = ["scope1", "scope2", "scope3"]
        durations = [50, 100, 150]  # ms

        # Time operations in different scopes
        for scope, duration in zip(scopes, durations):
            with monitor(scope):
                time_provider.advance(timedelta(milliseconds=duration))

        # Verify each scope has its own metrics
        for scope, duration in zip(scopes, durations):
            latency = monitor.get_execution_latency(scope)
            assert duration - 5 <= latency <= duration + 5  # Should be close to the expected duration

        # Check get_execution_latencies returns all scopes
        latencies = monitor.get_execution_latencies()
        assert set(latencies.keys()) == set(scopes)

    def test_execution_latency_with_exception(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that execution latency is still tracked when an exception occurs."""
        scope = "exception_scope"

        # Use try/except to catch the exception we're going to raise
        try:
            with monitor(scope):
                time_provider.advance(timedelta(milliseconds=50))
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still record the timing even if an exception occurred
        latency = monitor.get_execution_latency(scope)
        assert 45 <= latency <= 55  # Should be ~50ms

    def test_dummy_health_monitor_execution_latency(self) -> None:
        """Test that DummyHealthMonitor returns expected values for execution latency methods."""
        monitor = DummyHealthMonitor()

        # Check that execution latency methods return expected zero/empty values
        assert monitor.get_execution_latency("test_scope") == 0.0
        assert monitor.get_execution_latencies() == {}

    def test_queue_monitoring_with_channel(self) -> None:
        """Test that the health monitor correctly monitors the queue size of a provided channel."""
        # Create a mock channel with a queue
        mock_channel = MagicMock(spec=CtrlChannel)
        mock_queue = MagicMock()
        mock_queue.qsize.return_value = 42  # Set a specific queue size
        mock_channel._queue = mock_queue

        # Create monitor with the mock channel and short monitor interval for testing
        time_provider = MockTimeProvider()
        monitor = BaseHealthMonitor(time_provider, channel=mock_channel, monitor_interval="50ms")

        # Start monitoring
        monitor.start()

        # Give it a moment to update the queue size
        time.sleep(0.2)

        # Check that the queue size was updated
        assert monitor.get_queue_size() == 42

        # Change the queue size and check it updates
        mock_queue.qsize.return_value = 100
        time.sleep(0.2)
        assert monitor.get_queue_size() == 100

        # Clean up
        monitor.stop()

    def test_custom_monitor_interval(self, time_provider: MockTimeProvider) -> None:
        """Test that monitor_interval parameter is properly recognized and applied."""
        # Test with different interval formats
        monitor1 = BaseHealthMonitor(time_provider, monitor_interval="100ms")
        assert monitor1._monitor_interval_s == 0.1

        monitor2 = BaseHealthMonitor(time_provider, monitor_interval="500ms")
        assert monitor2._monitor_interval_s == 0.5

        monitor3 = BaseHealthMonitor(time_provider, monitor_interval="1s")
        assert monitor3._monitor_interval_s == 1.0

        monitor4 = BaseHealthMonitor(time_provider, monitor_interval="50ms")
        assert monitor4._monitor_interval_s == 0.05

    def test_watch_decorator(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that the watch decorator properly times function execution."""

        # Define a function to be decorated
        @monitor.watch()
        def test_function():
            time_provider.advance(timedelta(milliseconds=75))
            return "test result"

        # Define a function with custom scope name
        @monitor.watch("custom_scope")
        def another_function():
            time_provider.advance(timedelta(milliseconds=100))
            return 42

        # Call the decorated functions
        result1 = test_function()
        result2 = another_function()

        # Verify function results are returned correctly
        assert result1 == "test result"
        assert result2 == 42

        # Verify execution latencies were recorded
        function_scope = f"{test_function.__module__}.{test_function.__qualname__}"
        assert 70 <= monitor.get_execution_latency(function_scope) <= 80  # Should be ~75ms
        assert 95 <= monitor.get_execution_latency("custom_scope") <= 105  # Should be ~100ms

        # Verify function metadata is preserved
        assert test_function.__name__ == "test_function"
        assert another_function.__name__ == "another_function"

    def test_order_submit_latency_tracking(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that order submit latency is tracked correctly."""
        exchange = "test_exchange"
        client_id = "order_123"

        # Record order submit request
        request_time = time_provider.time()
        monitor.record_order_submit_request(exchange, client_id, request_time)

        # Advance time by 50ms
        time_provider.advance(timedelta(milliseconds=50))

        # Record order submit response
        response_time = time_provider.time()
        monitor.record_order_submit_response(exchange, client_id, response_time)

        # Check that latency was recorded
        latency = monitor.get_order_submit_latency(exchange)
        assert 45 <= latency <= 55  # Should be ~50ms

    def test_order_submit_latency_multiple_orders(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test tracking multiple order submits for the same exchange."""
        exchange = "test_exchange"

        # Submit multiple orders with different latencies
        latencies_ms = [10, 20, 30, 40, 50]
        for i, latency_ms in enumerate(latencies_ms):
            client_id = f"order_{i}"
            request_time = time_provider.time()
            monitor.record_order_submit_request(exchange, client_id, request_time)

            time_provider.advance(timedelta(milliseconds=latency_ms))

            response_time = time_provider.time()
            monitor.record_order_submit_response(exchange, client_id, response_time)

        # Check that the 90th percentile is recorded correctly
        latency_p90 = monitor.get_order_submit_latency(exchange, 90)
        assert 40 <= latency_p90 <= 50  # Should be close to the 90th percentile

    def test_order_submit_missing_request(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that order submit response without request doesn't crash."""
        exchange = "test_exchange"
        client_id = "order_missing"

        # Record response without request - should not crash
        response_time = time_provider.time()
        monitor.record_order_submit_response(exchange, client_id, response_time)

        # Should return 0 for unknown exchange
        assert monitor.get_order_submit_latency(exchange) == 0.0

    def test_order_cancel_latency_tracking(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that order cancel latency is tracked correctly."""
        exchange = "test_exchange"
        client_id = "order_123"

        # Record order cancel request
        request_time = time_provider.time()
        monitor.record_order_cancel_request(exchange, client_id, request_time)

        # Advance time by 30ms
        time_provider.advance(timedelta(milliseconds=30))

        # Record order cancel response
        response_time = time_provider.time()
        monitor.record_order_cancel_response(exchange, client_id, response_time)

        # Check that latency was recorded
        latency = monitor.get_order_cancel_latency(exchange)
        assert 25 <= latency <= 35  # Should be ~30ms

    def test_order_cancel_latency_multiple_orders(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test tracking multiple order cancels for the same exchange."""
        exchange = "test_exchange"

        # Cancel multiple orders with different latencies
        latencies_ms = [5, 10, 15, 20, 25]
        for i, latency_ms in enumerate(latencies_ms):
            client_id = f"order_{i}"
            request_time = time_provider.time()
            monitor.record_order_cancel_request(exchange, client_id, request_time)

            time_provider.advance(timedelta(milliseconds=latency_ms))

            response_time = time_provider.time()
            monitor.record_order_cancel_response(exchange, client_id, response_time)

        # Check that the 90th percentile is recorded correctly
        latency_p90 = monitor.get_order_cancel_latency(exchange, 90)
        assert 18 <= latency_p90 <= 25  # Should be close to the 90th percentile

    def test_order_request_cleanup(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that orphaned order requests are cleaned up after 60 seconds."""
        exchange = "test_exchange"
        client_id_orphaned = "order_orphaned"
        client_id_completed = "order_completed"

        # Record two order submit requests
        request_time = time_provider.time()
        monitor.record_order_submit_request(exchange, client_id_orphaned, request_time)
        monitor.record_order_submit_request(exchange, client_id_completed, request_time)

        # Complete one order immediately
        time_provider.advance(timedelta(milliseconds=10))
        monitor.record_order_submit_response(exchange, client_id_completed, time_provider.time())

        # Advance time by 61 seconds (past cleanup threshold)
        time_provider.advance(timedelta(seconds=61))

        # Trigger cleanup by calling it directly
        monitor._cleanup_old_order_requests()

        # Verify the orphaned request was removed from internal tracking
        assert (exchange, client_id_orphaned) not in monitor._order_submit_requests

        # Record the same order submit request again to verify it works after cleanup
        request_time_2 = time_provider.time()
        monitor.record_order_submit_request(exchange, client_id_orphaned, request_time_2)
        assert (exchange, client_id_orphaned) in monitor._order_submit_requests

    def test_connection_status_tracking(self, monitor: BaseHealthMonitor) -> None:
        """Test that connection status tracking works correctly."""
        exchange = "test_exchange"

        # Set up a connection status callback
        is_connected_flag = True

        def get_connection_status():
            return is_connected_flag

        monitor.set_is_connected(exchange, get_connection_status)

        # Check initial connection status
        assert monitor.is_connected(exchange) is True

        # Change connection status
        is_connected_flag = False
        assert monitor.is_connected(exchange) is False

        # Test unknown exchange returns True by default
        assert monitor.is_connected("unknown_exchange") is True

    def test_last_event_time_tracking(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that last event time is tracked correctly."""
        instrument = _get_test_instrument()
        event_type = "test_event"

        # Subscribe first
        monitor.subscribe(instrument, event_type)

        # Record first event
        event_time_1 = time_provider.time()
        monitor.on_data_arrival(instrument, event_type, event_time_1)

        # Check last event time
        last_time = monitor.get_last_event_time(instrument, event_type)
        assert last_time == event_time_1

        # Advance time and record another event
        time_provider.advance(timedelta(seconds=1))
        event_time_2 = time_provider.time()
        monitor.on_data_arrival(instrument, event_type, event_time_2)

        # Check that last event time was updated
        last_time = monitor.get_last_event_time(instrument, event_type)
        assert last_time == event_time_2

        # Test unknown event returns None
        assert monitor.get_last_event_time(instrument, "unknown_event") is None

    def test_last_event_times_multiple_events(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that last event times are tracked correctly for multiple event types."""
        instrument = _get_test_instrument()
        event_types = ["event_1", "event_2", "event_3"]
        event_times = {}

        # Subscribe to all event types
        for event_type in event_types:
            monitor.subscribe(instrument, event_type)

        # Record events of different types
        for event_type in event_types:
            event_time = time_provider.time()
            event_times[event_type] = event_time
            monitor.on_data_arrival(instrument, event_type, event_time)
            time_provider.advance(timedelta(milliseconds=100))

        # Verify each event type has correct last event time
        for event_type in event_types:
            last_time = monitor.get_last_event_time(instrument, event_type)
            assert last_time is not None
            # The last event time should be the time when the monitor received it (current time at that point)

        # Test unknown event returns None
        assert monitor.get_last_event_time(instrument, "unknown_event") is None

    def test_data_latency_per_exchange(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that data latency is tracked per exchange."""
        instrument = _get_test_instrument()
        exchange = instrument.exchange
        event_type = "test_event"

        # Subscribe first
        monitor.subscribe(instrument, event_type)

        # Record events with different latencies
        current_time = time_provider.time()

        # First event: 50ms latency
        event_time_1 = current_time - np.timedelta64(50, "ms")
        monitor.on_data_arrival(instrument, event_type, event_time_1)

        # Second event: 100ms latency
        event_time_2 = current_time - np.timedelta64(100, "ms")
        monitor.on_data_arrival(instrument, event_type, event_time_2)

        # Check latency is tracked (should be percentile of both values)
        latency = monitor.get_data_latency(exchange, event_type)
        assert 45 <= latency <= 105  # Should be between 50ms and 100ms

    def test_data_latencies_all_events(self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider) -> None:
        """Test that get_data_latencies returns latencies for all event types."""
        instrument = _get_test_instrument()
        exchange = instrument.exchange
        event_types = ["event_1", "event_2", "event_3"]

        # Subscribe to all event types
        for event_type in event_types:
            monitor.subscribe(instrument, event_type)

        current_time = time_provider.time()

        # Record events of different types with different latencies
        for i, event_type in enumerate(event_types):
            latency_ms = (i + 1) * 10  # 10ms, 20ms, 30ms
            event_time = current_time - np.timedelta64(latency_ms, "ms")
            monitor.on_data_arrival(instrument, event_type, event_time)

        # Get all data latencies
        latencies = monitor.get_data_latencies(exchange)

        # Verify all event types are present
        assert set(latencies.keys()) == set(event_types)

        # Verify approximate latencies
        assert 8 <= latencies["event_1"] <= 12  # Should be ~10ms
        assert 18 <= latencies["event_2"] <= 22  # Should be ~20ms
        assert 28 <= latencies["event_3"] <= 32  # Should be ~30ms

        # Test unknown exchange returns empty dict
        assert monitor.get_data_latencies("unknown_exchange") == {}

    def test_exchange_latencies_with_order_data(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that exchange latencies include order latencies."""
        exchange = "test_exchange"

        # Record some order submit latencies
        for i in range(5):
            client_id = f"order_{i}"
            request_time = time_provider.time()
            monitor.record_order_submit_request(exchange, client_id, request_time)
            time_provider.advance(timedelta(milliseconds=10 * (i + 1)))
            monitor.record_order_submit_response(exchange, client_id, time_provider.time())

        # Record some order cancel latencies
        for i in range(5):
            client_id = f"cancel_{i}"
            request_time = time_provider.time()
            monitor.record_order_cancel_request(exchange, client_id, request_time)
            time_provider.advance(timedelta(milliseconds=5 * (i + 1)))
            monitor.record_order_cancel_response(exchange, client_id, time_provider.time())

        # Get exchange latencies
        metrics = monitor.get_exchange_latencies(exchange)

        # Verify order latency metrics are populated
        assert metrics.order_submit > 0
        assert metrics.order_cancel > 0

    def test_subscribe_adds_to_active_subscriptions(self, monitor: BaseHealthMonitor) -> None:
        """Test that subscribe adds instrument/event_type to active subscriptions."""
        instrument = _get_test_instrument()
        event_type = "ohlc"

        # Subscribe
        monitor.subscribe(instrument, event_type)

        # Verify it's in active subscriptions
        assert (instrument, event_type) in monitor._active_subscriptions

    def test_unsubscribe_removes_and_cleans_data(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that unsubscribe removes from subscriptions and cleans up stored data."""
        instrument = _get_test_instrument()
        event_type = "ohlc"

        # Subscribe and record some data
        monitor.subscribe(instrument, event_type)
        monitor.on_data_arrival(instrument, event_type, time_provider.time())

        # Verify data is stored
        assert (instrument, event_type) in monitor._last_event_time
        assert (instrument, event_type) in monitor._event_frequency

        # Unsubscribe
        monitor.unsubscribe(instrument, event_type)

        # Verify removed from active subscriptions
        assert (instrument, event_type) not in monitor._active_subscriptions

        # Verify data is cleaned up
        assert (instrument, event_type) not in monitor._last_event_time
        assert (instrument, event_type) not in monitor._event_frequency

    def test_on_data_arrival_ignores_unsubscribed(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that on_data_arrival ignores data for unsubscribed instruments."""
        instrument = _get_test_instrument()
        event_type = "quote"

        # Don't subscribe, just try to record data
        monitor.on_data_arrival(instrument, event_type, time_provider.time())

        # Verify no data is stored
        assert (instrument, event_type) not in monitor._last_event_time
        assert (instrument, event_type) not in monitor._event_frequency

    def test_on_data_arrival_tracks_subscribed(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that on_data_arrival tracks data for subscribed instruments."""
        instrument = _get_test_instrument()
        event_type = "ohlc"

        # Subscribe first
        monitor.subscribe(instrument, event_type)

        # Record data
        monitor.on_data_arrival(instrument, event_type, time_provider.time())

        # Verify data is stored
        assert (instrument, event_type) in monitor._last_event_time
        assert (instrument, event_type) in monitor._event_frequency
        assert monitor.get_last_event_time(instrument, event_type) is not None

    def test_multiple_subscriptions_independent(
        self, monitor: BaseHealthMonitor, time_provider: MockTimeProvider
    ) -> None:
        """Test that multiple subscriptions work independently."""
        instrument1 = _get_test_instrument("BTCUSDT")
        instrument2 = _get_test_instrument("ETHUSDT")
        event_type1 = "ohlc"
        event_type2 = "quote"

        # Subscribe to different combinations
        monitor.subscribe(instrument1, event_type1)
        monitor.subscribe(instrument1, event_type2)
        monitor.subscribe(instrument2, event_type1)

        # Verify all are tracked
        assert (instrument1, event_type1) in monitor._active_subscriptions
        assert (instrument1, event_type2) in monitor._active_subscriptions
        assert (instrument2, event_type1) in monitor._active_subscriptions

        # Record data for all
        monitor.on_data_arrival(instrument1, event_type1, time_provider.time())
        monitor.on_data_arrival(instrument1, event_type2, time_provider.time())
        monitor.on_data_arrival(instrument2, event_type1, time_provider.time())

        # Verify all tracked
        assert monitor.get_last_event_time(instrument1, event_type1) is not None
        assert monitor.get_last_event_time(instrument1, event_type2) is not None
        assert monitor.get_last_event_time(instrument2, event_type1) is not None

        # Unsubscribe one
        monitor.unsubscribe(instrument1, event_type1)

        # Verify only that one is removed
        assert (instrument1, event_type1) not in monitor._active_subscriptions
        assert (instrument1, event_type2) in monitor._active_subscriptions
        assert (instrument2, event_type1) in monitor._active_subscriptions
