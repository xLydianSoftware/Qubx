import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.basics import dt_64
from qubx.core.interfaces import HealthMetrics, IMetricEmitter, ITimeProvider
from qubx.health.base import BaseHealthMonitor, DummyHealthMonitor


class MockTimeProvider(ITimeProvider):
    def __init__(self, start_time: datetime | None = None):
        self._current_time = start_time or datetime(2024, 1, 1)

    def time(self) -> dt_64:
        return np.datetime64(self._current_time)

    def advance(self, delta: timedelta):
        self._current_time += delta


class TestDummyHealthMonitor:
    def test_dummy_monitor_returns_zero_values(self):
        monitor = DummyHealthMonitor()

        # Test all methods return expected zero/empty values
        assert monitor.get_queue_size() == 0
        assert monitor.get_arrival_latency("test_event") == 0.0
        assert monitor.get_queue_latency("test_event") == 0.0
        assert monitor.get_processing_latency("test_event") == 0.0
        assert monitor.get_latency("test_event") == 0.0
        assert monitor.get_event_frequency("test_event") == 0.0

        metrics = monitor.get_system_metrics()
        assert isinstance(metrics, HealthMetrics)

        # Test context manager interface
        with monitor("test_event") as m:
            assert m == monitor
            time.sleep(0.1)  # Simulate some work

        # Verify no state change after operations
        assert monitor.get_processing_latency("test_event") == 0.0


class TestBaseHealthMonitor:
    @pytest.fixture
    def time_provider(self):
        return MockTimeProvider()

    @pytest.fixture
    def monitor(self, time_provider):
        monitor = BaseHealthMonitor(time_provider)
        monitor.start()
        yield monitor
        monitor.stop()

    @pytest.fixture
    def monitor_with_custom_interval(self, time_provider):
        # Create monitor with faster interval for testing
        monitor = BaseHealthMonitor(time_provider, emit_interval="100ms")
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
        assert metrics.queue_size > 1000

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
        for i in range(20):
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

        # Should have non-zero latencies
        assert monitor.get_arrival_latency(event_type) > 0
        # Queue latency might be 0 if calculations are based on percentiles
        # Processing latency might be 0 based on implementation
        assert monitor.get_latency(event_type) > 0

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
        assert 45 <= metrics.p50_arrival_latency <= 55  # ~50ms

        # Verify individual latency methods
        assert 45 <= monitor.get_arrival_latency(event_type, 50) <= 55  # ~50ms arrival latency

        # The queue latency is the difference between start and arrival latency
        assert 70 <= monitor.get_queue_latency(event_type, 50) <= 80  # ~75ms queue latency (125-50)

        # The processing latency is calculated from the difference of end - start
        processing_latency = monitor.get_processing_latency(event_type, 50)
        assert 95 <= processing_latency <= 105  # ~100ms processing latency

        # End-to-end latency is the time from event to end processing
        assert 220 <= monitor.get_latency(event_type, 50) <= 230  # ~225ms end-to-end latency

    def test_context_manager_timing(self, monitor, time_provider):
        event_type = "test_event"

        with monitor(event_type):
            # Simulate some work by advancing the mock time
            time_provider.advance(timedelta(milliseconds=100))

        # Should have recorded processing time
        assert monitor.get_latency(event_type) > 0

    def test_nested_context_managers(self, monitor, time_provider):
        """Test that nested context managers work correctly."""
        outer_event = "outer_event"
        inner_event = "inner_event"

        with monitor(outer_event):
            time_provider.advance(timedelta(milliseconds=50))

            with monitor(inner_event):
                time_provider.advance(timedelta(milliseconds=25))

            # Inner event should have ~25ms latency
            assert 20 <= monitor.get_latency(inner_event) <= 30

            time_provider.advance(timedelta(milliseconds=25))

        # Outer event should have ~100ms latency
        assert 95 <= monitor.get_latency(outer_event) <= 105

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
        assert 45 <= monitor.get_latency(event_type) <= 55

    def test_dropped_events_tracking(self, monitor):
        event_type = "test_event"

        # Record some dropped events
        for _ in range(3):
            monitor.record_event_dropped(event_type)

        # Get system metrics
        metrics = monitor.get_system_metrics()
        assert metrics.drop_rate > 0

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
            assert monitor.get_arrival_latency(event_type) > 0
            assert monitor.get_latency(event_type) > 0
            assert monitor.get_event_frequency(event_type) > 0
            # Note: Queue latency and processing latency might be zero depending on the implementation

        # System metrics should aggregate across all event types
        metrics = monitor.get_system_metrics()
        assert metrics.drop_rate > 0
        assert metrics.p50_arrival_latency > 0
        assert metrics.p50_queue_latency > 0
        assert metrics.p50_processing_latency > 0

    def test_latency_percentiles(self, monitor, time_provider):
        """Test that latency percentiles are calculated correctly."""
        event_type = "test_event"

        # Generate a series of events with different latencies
        base_time = time_provider.time()
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms

        for latency in latencies:
            event_time = base_time
            current_time = base_time + np.timedelta64(latency, "ms")
            time_provider._current_time = current_time.astype(datetime)

            monitor.record_data_arrival(event_type, event_time)
            monitor.record_start_processing(event_type, event_time)
            monitor.record_end_processing(event_type, event_time)

        # Get system metrics
        metrics = monitor.get_system_metrics()

        # Check arrival latency percentiles
        assert 45 <= metrics.p50_arrival_latency <= 55  # median should be ~50ms
        assert 85 <= metrics.p90_arrival_latency <= 95  # 90th percentile should be ~90ms
        assert 95 <= metrics.p99_arrival_latency <= 105  # 99th percentile should be ~100ms

        # Check percentile methods
        assert 45 <= monitor.get_arrival_latency(event_type, 50) <= 55  # 50th percentile should be ~50ms
        assert 85 <= monitor.get_arrival_latency(event_type, 90) <= 95  # 90th percentile should be ~90ms
        assert 95 <= monitor.get_arrival_latency(event_type, 99) <= 105  # 99th percentile should be ~100ms

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
        assert metrics.queue_size > 0
        assert metrics.drop_rate > 0

        # Verify all latency metrics are populated
        assert metrics.p50_arrival_latency > 0
        assert metrics.p90_arrival_latency > 0
        assert metrics.p99_arrival_latency > 0

        assert metrics.p50_queue_latency > 0
        assert metrics.p90_queue_latency > 0
        assert metrics.p99_queue_latency > 0

        assert metrics.p50_processing_latency > 0
        assert metrics.p90_processing_latency > 0
        assert metrics.p99_processing_latency > 0

    def test_monitor_start_stop(self, time_provider):
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

    def test_custom_emit_interval(self, time_provider):
        """Test that emit_interval parameter is properly recognized and applied."""
        # Test with different interval formats
        monitor1 = BaseHealthMonitor(time_provider, emit_interval="1s")
        assert monitor1._emit_interval_s == 1.0

        monitor2 = BaseHealthMonitor(time_provider, emit_interval="500ms")
        assert monitor2._emit_interval_s == 0.5

        monitor3 = BaseHealthMonitor(time_provider, emit_interval="2m")
        assert monitor3._emit_interval_s == 120.0

    def test_dropped_rate_calculation(self, monitor, time_provider):
        """Test that drop rate is calculated correctly for each event type."""
        # Create multiple event types with different drop patterns
        event_types = ["type1", "type2", "type3"]

        # Record drops for each event type at different rates
        base_time = time_provider.time()

        # For type1: 5 drops over 1 second
        for i in range(5):
            time_provider._current_time = (base_time + np.timedelta64(i * 200, "ms")).astype(datetime)
            monitor.record_event_dropped("type1")

        # For type2: 10 drops over 1 second
        for i in range(10):
            time_provider._current_time = (base_time + np.timedelta64(i * 100, "ms")).astype(datetime)
            monitor.record_event_dropped("type2")

        # For type3: no drops

        # Verify drop rates
        assert 4.5 <= monitor._get_drop_rate("type1") <= 5.5  # ~5 drops/sec
        assert 9.5 <= monitor._get_drop_rate("type2") <= 10.5  # ~10 drops/sec
        assert monitor._get_drop_rate("type3") == 0.0  # 0 drops/sec

        # Verify system-wide drop rate is the average of all defined rates
        metrics = monitor.get_system_metrics()
        # Should be close to (5 + 10) / 2 = 7.5 drops/sec
        assert 7.0 <= metrics.drop_rate <= 8.0

    def test_metrics_emission(self, time_provider):
        """Test that metrics are emitted correctly."""
        mock_emitter = MagicMock(spec=IMetricEmitter)
        monitor = BaseHealthMonitor(time_provider, emitter=mock_emitter, emit_interval="100ms")

        # Record some test data
        event_type = "test_event"
        monitor.set_event_queue_size(5)
        monitor.record_event_dropped(event_type)
        monitor.record_data_arrival(event_type, time_provider.time())

        # Manually call emit to avoid threading complexities in tests
        monitor._emit()

        # Verify that metrics were emitted
        assert mock_emitter.emit.call_count > 0

        # Extract metric names
        metric_names = set()
        for call in mock_emitter.emit.call_args_list:
            args, kwargs = call
            if args:  # If positional args were used
                metric_names.add(args[0])  # First arg is the metric name
            elif "name" in kwargs:  # If keyword args were used
                metric_names.add(kwargs["name"])

        # Verify system-wide metrics were emitted
        assert "health.queue_size" in metric_names
        assert "health.dropped_events" in metric_names

        # Verify latency metrics were emitted
        assert "health.arrival_latency.p50" in metric_names
        assert "health.arrival_latency.p90" in metric_names
        assert "health.arrival_latency.p99" in metric_names

        assert "health.queue_latency.p50" in metric_names
        assert "health.queue_latency.p90" in metric_names
        assert "health.queue_latency.p99" in metric_names

        assert "health.processing_latency.p50" in metric_names
        assert "health.processing_latency.p90" in metric_names
        assert "health.processing_latency.p99" in metric_names

        # Extract event tags
        event_metrics = set()
        for call in mock_emitter.emit.call_args_list:
            args, kwargs = call
            if args and len(args) > 2 and isinstance(args[2], dict) and args[2].get("event_type") == event_type:
                event_metrics.add(args[0])
            elif "tags" in kwargs and kwargs["tags"].get("event_type") == event_type:
                event_metrics.add(kwargs["name"])

        assert len(event_metrics) > 0

        # Verify specific event metrics were emitted
        assert "health.event_frequency" in event_metrics
        assert "health.event_processing_latency" in event_metrics
        assert "health.event_drop_rate" in event_metrics
        assert "health.event_arrival_latency" in event_metrics
        assert "health.event_queue_latency" in event_metrics
