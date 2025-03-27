import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from qubx.core.basics import dt_64
from qubx.core.interfaces import HealthMetrics, ITimeProvider
from qubx.health.base import BaseHealthMetricsMonitor, DummyHealthMetricsMonitor


class MockTimeProvider(ITimeProvider):
    def __init__(self, start_time: datetime | None = None):
        self._current_time = start_time or datetime(2024, 1, 1)

    def time(self) -> dt_64:
        return np.datetime64(self._current_time)

    def advance(self, delta: timedelta):
        self._current_time += delta


class TestDummyHealthMetricsMonitor:
    def test_dummy_monitor_returns_zero_values(self):
        monitor = DummyHealthMetricsMonitor()

        # Test all methods return expected zero/empty values
        assert monitor.get_queue_size() == 0
        assert monitor.get_latency("test_event") == 0.0
        assert monitor.get_event_frequency("test_event") == 0.0

        metrics = monitor.get_system_metrics()
        assert isinstance(metrics, HealthMetrics)

        # Test context manager interface
        with monitor("test_event") as m:
            assert m == monitor
            time.sleep(0.1)  # Simulate some work

        # Verify no state change after operations
        assert monitor.get_latency("test_event") == 0.0


class TestBaseHealthMetricsMonitor:
    @pytest.fixture
    def time_provider(self):
        return MockTimeProvider()

    @pytest.fixture
    def monitor(self, time_provider):
        monitor = BaseHealthMetricsMonitor(time_provider)
        monitor.start()
        yield monitor
        monitor.stop()

    def test_queue_size_tracking(self, monitor):
        monitor.set_event_queue_size(5)
        assert monitor.get_queue_size() == 5

        monitor.set_event_queue_size(10)
        assert monitor.get_queue_size() == 10

    def test_queue_size_overflow(self, monitor):
        """Test that queue size tracking works correctly when buffer is full."""
        # Fill the buffer
        for i in range(2000):  # More than the default window size
            monitor.set_event_queue_size(i)

        # Verify we only keep the most recent values
        assert monitor.get_queue_size() == 1999  # Most recent value
        metrics = monitor.get_system_metrics()
        # Average should be high since we filled with increasing values
        assert metrics.avg_queue_size > 1000

    def test_event_frequency_tracking(self, monitor, time_provider):
        event_type = "test_event"

        # Record a few events
        for _ in range(3):
            monitor.record_data_arrival(event_type, time_provider.time())
            time_provider.advance(timedelta(milliseconds=100))

        # Should have non-zero frequency
        freq = monitor.get_event_frequency(event_type)
        assert freq > 0

    def test_event_frequency_window(self, monitor, time_provider):
        """Test that event frequency only counts events in the last second."""
        event_type = "test_event"

        # Record events spread over 2 seconds
        for i in range(10):
            monitor.record_data_arrival(event_type, time_provider.time())
            time_provider.advance(timedelta(milliseconds=250))  # 4 events per second

        # Should only count events in the last second (4 events)
        freq = monitor.get_event_frequency(event_type)
        assert 3.5 <= freq <= 4.5  # Allow some floating point imprecision

    def test_latency_tracking(self, monitor, time_provider):
        event_type = "test_event"
        event_time = time_provider.time()

        # Advance time to simulate latency
        time_provider.advance(timedelta(milliseconds=100))

        # Record event processing stages
        monitor.record_data_arrival(event_type, event_time)
        monitor.record_start_processing(event_type, event_time)
        monitor.record_end_processing(event_type, event_time)

        # Should have non-zero latency
        latency = monitor.get_latency(event_type)
        assert latency > 0

    def test_latency_accuracy(self, monitor, time_provider):
        """Test that latency measurements are accurate."""
        event_type = "test_event"
        event_time = time_provider.time()

        # Record with precise timing
        time_provider.advance(timedelta(milliseconds=50))
        monitor.record_data_arrival(event_type, event_time)  # Should record ~50ms latency

        time_provider.advance(timedelta(milliseconds=75))
        monitor.record_start_processing(event_type, event_time)  # Should record ~125ms latency

        time_provider.advance(timedelta(milliseconds=100))
        monitor.record_end_processing(event_type, event_time)  # Should record ~225ms latency

        # Get system metrics and verify latencies
        metrics = monitor.get_system_metrics()
        assert 45 <= metrics.avg_arrival_latency <= 55  # ~50ms
        assert 120 <= metrics.avg_queue_latency <= 130  # ~125ms
        assert 220 <= metrics.avg_processing_latency <= 230  # ~225ms

    def test_context_manager_timing(self, monitor, time_provider):
        event_type = "test_event"

        with monitor(event_type):
            # Simulate some work by advancing the mock time
            time_provider.advance(timedelta(milliseconds=100))

        # Should have recorded processing time
        latency = monitor.get_latency(event_type)
        assert latency > 0

    def test_nested_context_managers(self, monitor, time_provider):
        """Test that nested context managers work correctly."""
        outer_event = "outer_event"
        inner_event = "inner_event"

        with monitor(outer_event):
            time_provider.advance(timedelta(milliseconds=50))

            with monitor(inner_event):
                time_provider.advance(timedelta(milliseconds=25))

            # Inner event should have ~25ms latency
            inner_latency = monitor.get_latency(inner_event)
            assert 20 <= inner_latency <= 30

            time_provider.advance(timedelta(milliseconds=25))

        # Outer event should have ~100ms latency
        outer_latency = monitor.get_latency(outer_event)
        assert 95 <= outer_latency <= 105

    def test_context_manager_exception_handling(self, monitor, time_provider):
        """Test that context managers handle exceptions correctly."""
        event_type = "test_event"

        try:
            with monitor(event_type):
                time_provider.advance(timedelta(milliseconds=50))
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still record the timing even if an exception occurred
        latency = monitor.get_latency(event_type)
        assert 45 <= latency <= 55

    def test_dropped_events_tracking(self, monitor):
        event_type = "test_event"

        # Record some dropped events
        for _ in range(3):
            monitor.record_event_dropped(event_type)

        # Get system metrics
        metrics = monitor.get_system_metrics()
        assert metrics.avg_dropped_events > 0

    def test_multiple_event_types(self, monitor, time_provider):
        """Test handling of multiple event types simultaneously."""
        event_types = ["type1", "type2", "type3"]
        event_time = time_provider.time()

        # Record events for each type with different latencies
        for i, event_type in enumerate(event_types):
            time_provider.advance(timedelta(milliseconds=50))
            monitor.record_data_arrival(event_type, event_time)
            monitor.record_event_dropped(event_type)

            time_provider.advance(timedelta(milliseconds=25))
            monitor.record_start_processing(event_type, event_time)

            time_provider.advance(timedelta(milliseconds=25))
            monitor.record_end_processing(event_type, event_time)

        # Verify each event type has its own metrics
        for event_type in event_types:
            assert monitor.get_latency(event_type) > 0
            assert monitor.get_event_frequency(event_type) > 0

        # System metrics should average across all event types
        metrics = monitor.get_system_metrics()
        assert metrics.avg_dropped_events == 1.0  # One drop per event type
        assert metrics.avg_arrival_latency > 0
        assert metrics.avg_queue_latency > 0
        assert metrics.avg_processing_latency > 0

    def test_system_metrics_aggregation(self, monitor, time_provider):
        event_type = "test_event"
        event_time = time_provider.time()

        # Record various metrics
        monitor.set_event_queue_size(5)
        monitor.record_event_dropped(event_type)

        # Record data arrival with some latency
        time_provider.advance(timedelta(milliseconds=50))
        monitor.record_data_arrival(event_type, event_time)

        # Record start processing with additional latency
        time_provider.advance(timedelta(milliseconds=50))
        monitor.record_start_processing(event_type, event_time)

        # Record end processing with additional latency
        time_provider.advance(timedelta(milliseconds=50))
        monitor.record_end_processing(event_type, event_time)

        # Get aggregated metrics
        metrics = monitor.get_system_metrics()

        # Verify all fields are populated
        assert metrics.avg_queue_size > 0
        assert metrics.avg_dropped_events > 0
        assert metrics.avg_arrival_latency > 0
        assert metrics.avg_queue_latency > 0
        assert metrics.avg_processing_latency > 0

    def test_monitor_start_stop(self, time_provider):
        """Test that monitor can be started and stopped multiple times."""
        monitor = BaseHealthMetricsMonitor(time_provider)

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
