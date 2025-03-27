import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from qubx.core.basics import dt_64
from qubx.core.interfaces import HealthMetrics, IHealthMetricsMonitor, ITimeProvider
from qubx.utils.collections import DequeFloat64, DequeIndicator


@dataclass
class TimingContext:
    """Context for timing a code block."""

    event_type: str
    start_time: dt_64


class DummyHealthMetricsMonitor(IHealthMetricsMonitor):
    """No-op implementation of health metrics monitoring."""

    def __call__(self, event_type: str) -> "DummyHealthMetricsMonitor":
        """Support for context manager usage with event type."""
        return self

    def __enter__(self) -> "DummyHealthMetricsMonitor":
        """Enter context for timing measurement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record timing"""
        pass

    def record_event_dropped(self, event_type: str) -> None:
        """Record that an event was dropped."""
        pass

    def record_data_arrival(self, event_type: str, event_time: dt_64) -> None:
        """Record a data arrival time."""
        pass

    def record_start_processing(self, event_type: str, event_time: dt_64) -> None:
        """Record a start processing time."""
        pass

    def record_end_processing(self, event_type: str, event_time: dt_64) -> None:
        """Record a end processing time."""
        pass

    def set_event_queue_size(self, size: int) -> None:
        """Set the current event queue size."""
        pass

    def get_queue_size(self) -> int:
        """Get the current event queue size."""
        return 0

    def get_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get latency for a specific event type."""
        return 0.0

    def get_event_frequency(self, event_type: str) -> float:
        """Get the events per second for a specific event type."""
        return 0.0

    def get_system_metrics(self) -> HealthMetrics:
        """Get system-wide metrics."""
        return HealthMetrics(
            avg_queue_size=0.0,
            avg_dropped_events=0.0,
            avg_arrival_latency=0.0,
            avg_queue_latency=0.0,
            avg_processing_latency=0.0,
        )

    def start(self) -> None:
        """Start the health metrics monitor."""
        pass

    def stop(self) -> None:
        """Stop the health metrics monitor."""
        pass


class BaseHealthMetricsMonitor(IHealthMetricsMonitor):
    """Base implementation of health metrics monitoring using Deque for tracking."""

    def __init__(self, time_provider: ITimeProvider, window_size: int = 1000):
        """Initialize the health metrics monitor.

        Args:
            time_provider: Provider for getting current time
            window_size: Number of samples to keep in the sliding window
        """
        self.time_provider = time_provider
        self.window_size = window_size
        self._is_running = False
        self._monitor_thread: threading.Thread | None = None

        # Metrics storage using Deque
        self._queue_size = DequeFloat64(window_size)
        self._dropped_events = defaultdict(lambda: DequeFloat64(window_size))
        self._arrival_latency = defaultdict(lambda: DequeIndicator(window_size))
        self._queue_latency = defaultdict(lambda: DequeIndicator(window_size))
        self._processing_latency = defaultdict(lambda: DequeIndicator(window_size))
        self._event_frequency = defaultdict(lambda: DequeIndicator(window_size))

        # Thread-local storage for timing contexts
        self._thread_local = threading.local()

    def _get_timing_stack(self) -> list[TimingContext]:
        """Get or create the timing stack for the current thread."""
        if not hasattr(self._thread_local, "timing_stack"):
            self._thread_local.timing_stack = []
        return self._thread_local.timing_stack

    def __call__(self, event_type: str) -> "BaseHealthMetricsMonitor":
        """Support for context manager usage with event type."""
        self._get_timing_stack().append(TimingContext(event_type, self.time_provider.time()))
        return self

    def __enter__(self) -> "BaseHealthMetricsMonitor":
        """Enter context for timing measurement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record timing"""
        stack = self._get_timing_stack()
        if stack:
            context = stack.pop()
            current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
            start_time_ns = context.start_time.astype("datetime64[ns]").astype(int)
            duration = (current_time - start_time_ns) / 1e6  # Convert to ms
            self._processing_latency[context.event_type].push_back_fields(current_time, duration)

    def record_processing_latency(self, event_type: str, latency: float) -> None:
        """Record a processing latency value."""
        current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
        self._processing_latency[event_type].push_back_fields(current_time, latency)

    def record_event_dropped(self, event_type: str) -> None:
        """Record that an event was dropped."""
        self._dropped_events[event_type].push_back(1.0)

    def record_data_arrival(self, event_type: str, event_time: dt_64) -> None:
        """Record a data arrival time."""
        current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        latency = (current_time - event_time_ns) / 1e6  # Convert to ms
        self._arrival_latency[event_type].push_back_fields(current_time, latency)
        self._event_frequency[event_type].push_back_fields(current_time, 1.0)

    def record_start_processing(self, event_type: str, event_time: dt_64) -> None:
        """Record a start processing time."""
        current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        latency = (current_time - event_time_ns) / 1e6  # Convert to ms
        self._queue_latency[event_type].push_back_fields(current_time, latency)

    def record_end_processing(self, event_type: str, event_time: dt_64) -> None:
        """Record a end processing time."""
        current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        latency = (current_time - event_time_ns) / 1e6  # Convert to ms
        self._processing_latency[event_type].push_back_fields(current_time, latency)

    def set_event_queue_size(self, size: int) -> None:
        """Set the current event queue size."""
        self._queue_size.push_back(float(size))

    def get_queue_size(self) -> int:
        """Get the current event queue size."""
        if self._queue_size.is_empty():
            return 0
        return int(self._queue_size[0])

    def get_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get latency for a specific event type."""
        if event_type not in self._processing_latency or self._processing_latency[event_type].is_empty():
            return 0.0
        # For now, just return the most recent value
        return float(self._processing_latency[event_type][0]["value"])

    def get_event_frequency(self, event_type: str) -> float:
        """Get the events per second for a specific event type."""
        if event_type not in self._event_frequency or self._event_frequency[event_type].is_empty():
            return 0.0
        # Count events in the last second
        freq = 0.0
        current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
        one_second_ago = current_time - 1_000_000_000  # 1 second in nanoseconds

        series = self._event_frequency[event_type].to_array()
        for record in series:
            if record["timestamp"] >= one_second_ago:
                freq += record["value"]
        return freq

    def get_system_metrics(self) -> HealthMetrics:
        """Get system-wide metrics."""
        # Calculate averages over all stored values
        queue_size = np.mean(self._queue_size.to_array()) if not self._queue_size.is_empty() else 0.0

        # Dropped events average
        dropped = 0.0
        if self._dropped_events:
            dropped_counts = [
                np.mean(series.to_array()) if not series.is_empty() else 0.0 for series in self._dropped_events.values()
            ]
            dropped = sum(dropped_counts) / len(dropped_counts)

        # Latency averages
        def calc_latency_avg(latency_dict):
            if not latency_dict:
                return 0.0
            latencies = []
            for series in latency_dict.values():
                if not series.is_empty():
                    arr = series.to_array()
                    latencies.extend(arr["value"])
            return float(np.mean(latencies)) if latencies else 0.0

        arrival = calc_latency_avg(self._arrival_latency)
        queue = calc_latency_avg(self._queue_latency)
        processing = calc_latency_avg(self._processing_latency)

        return HealthMetrics(
            avg_queue_size=float(queue_size),
            avg_dropped_events=float(dropped),
            avg_arrival_latency=float(arrival),
            avg_queue_latency=float(queue),
            avg_processing_latency=float(processing),
        )

    def start(self) -> None:
        """Start the health metrics monitor."""
        if not self._is_running:
            self._is_running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the health metrics monitor."""
        self._is_running = False
        if self._monitor_thread is not None:
            self._monitor_thread.join()

    def _monitor_loop(self) -> None:
        """Background thread for monitoring metrics."""
        while self._is_running:
            # Update metrics every second
            time.sleep(1)
            # The actual metrics are updated in real-time as events occur
            # This thread can be used for additional monitoring tasks if needed
