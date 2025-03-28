import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from qubx.core.basics import dt_64
from qubx.core.interfaces import HealthMetrics, IHealthMonitor, IMetricEmitter, ITimeProvider
from qubx.core.utils import recognize_timeframe
from qubx.utils.collections import DequeFloat64, DequeIndicator


@dataclass
class TimingContext:
    """Context for timing a code block."""

    event_type: str
    start_time: dt_64


class DummyHealthMonitor(IHealthMonitor):
    """No-op implementation of health metrics monitoring."""

    def __call__(self, event_type: str) -> "DummyHealthMonitor":
        """Support for context manager usage with event type."""
        return self

    def __enter__(self) -> "DummyHealthMonitor":
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

    def get_arrival_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get arrival latency for a specific event type."""
        return 0.0

    def get_queue_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get queue latency for a specific event type."""
        return 0.0

    def get_processing_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get processing latency for a specific event type."""
        return 0.0

    def get_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get end-to-end latency for a specific event type."""
        return 0.0

    def get_event_frequency(self, event_type: str) -> float:
        """Get the events per second for a specific event type."""
        return 0.0

    def get_system_metrics(self) -> HealthMetrics:
        """Get system-wide metrics."""
        return HealthMetrics(
            avg_queue_size=0.0,
            avg_dropped_events=0.0,
            p50_arrival_latency=0.0,
            p90_arrival_latency=0.0,
            p99_arrival_latency=0.0,
            p50_queue_latency=0.0,
            p90_queue_latency=0.0,
            p99_queue_latency=0.0,
            p50_processing_latency=0.0,
            p90_processing_latency=0.0,
            p99_processing_latency=0.0,
        )

    def start(self) -> None:
        """Start the health metrics monitor."""
        pass

    def stop(self) -> None:
        """Stop the health metrics monitor."""
        pass


class BaseHealthMonitor(IHealthMonitor):
    """Base implementation of health metrics monitoring using Deque for tracking."""

    def __init__(
        self, time_provider: ITimeProvider, emitter: Optional[IMetricEmitter] = None, emit_interval: str = "1s"
    ):
        """Initialize the health metrics monitor.

        Args:
            time_provider: Provider for time information
            emitter: Optional metric emitter for sending metrics to external systems
            emit_interval: Interval to emit metrics, e.g. "1s", "500ms", "5m" (default: "1s")
        """
        self.time_provider = time_provider
        self._emitter = emitter

        # Convert emit interval to nanoseconds
        self._emit_interval_ns = recognize_timeframe(emit_interval)
        self._emit_interval_s = self._emit_interval_ns / 1_000_000_000  # Convert to seconds for sleep

        # Initialize metrics storage
        self._queue_size = DequeFloat64(1000)  # Store last 1000 queue size measurements
        self._event_frequency = defaultdict(lambda: DequeIndicator(1000))
        self._arrival_latency = defaultdict(lambda: DequeIndicator(1000))
        self._start_latency = defaultdict(lambda: DequeIndicator(1000))
        self._end_latency = defaultdict(lambda: DequeIndicator(1000))
        self._dropped_events = defaultdict(lambda: DequeIndicator(1000))

        # Initialize emission thread control
        self._stop_event = threading.Event()
        self._emission_thread = None

        # Initialize monitor thread control
        self._is_running = False
        self._monitor_thread = None

        # Thread-local storage for timing contexts
        self._thread_local = threading.local()

    def _get_timing_stack(self) -> list[TimingContext]:
        """Get or create the timing stack for the current thread."""
        if not hasattr(self._thread_local, "timing_stack"):
            self._thread_local.timing_stack = []
        return self._thread_local.timing_stack

    def __call__(self, event_type: str) -> "BaseHealthMonitor":
        """Support for context manager usage with event type."""
        self._get_timing_stack().append(TimingContext(event_type, self.time_provider.time()))
        return self

    def __enter__(self) -> "BaseHealthMonitor":
        """Enter context for timing measurement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record timing"""
        stack = self._get_timing_stack()
        if stack:
            context = stack.pop()
            current_time = self.time_provider.time()
            current_time_ns = current_time.astype("datetime64[ns]").astype(int)
            start_time_ns = context.start_time.astype("datetime64[ns]").astype(int)
            duration = (current_time_ns - start_time_ns) / 1e6  # Convert to ms

            # Record processing latency
            self._end_latency[context.event_type].push_back_fields(current_time_ns, duration)

    def record_event_dropped(self, event_type: str) -> None:
        """Record that an event was dropped."""
        current_time = self.time_provider.time().astype("datetime64[ns]").astype(int)
        self._dropped_events[event_type].push_back_fields(current_time, 1.0)

    def record_data_arrival(self, event_type: str, event_time: dt_64) -> None:
        """Record data arrival time and calculate arrival latency."""
        current_time = self.time_provider.time()
        current_time_ns = current_time.astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        arrival_latency = (current_time_ns - event_time_ns) / 1e6  # Convert to milliseconds

        # Record arrival latency and event frequency
        self._arrival_latency[event_type].push_back_fields(current_time_ns, arrival_latency)
        self._event_frequency[event_type].push_back_fields(current_time_ns, 1.0)

    def record_start_processing(self, event_type: str, event_time: dt_64) -> None:
        """Record processing start time and calculate queue latency."""
        current_time = self.time_provider.time()
        current_time_ns = current_time.astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        queue_latency = (current_time_ns - event_time_ns) / 1e6  # Convert to milliseconds

        # Record queue latency
        self._start_latency[event_type].push_back_fields(current_time_ns, queue_latency)

    def record_end_processing(self, event_type: str, event_time: dt_64) -> None:
        """Record processing end time and calculate processing latency."""
        current_time = self.time_provider.time()
        current_time_ns = current_time.astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        processing_latency = (current_time_ns - event_time_ns) / 1e6  # Convert to milliseconds

        # Record processing latency
        self._end_latency[event_type].push_back_fields(current_time_ns, processing_latency)

    def set_event_queue_size(self, size: int) -> None:
        self._queue_size.push_back(float(size))

    def get_queue_size(self) -> float:
        if self._queue_size.is_empty():
            return 0.0
        return self._queue_size.back()

    def get_arrival_latency(self, event_type: str, percentile: float = 90) -> float:
        return self._get_latency_percentile(event_type, self._arrival_latency, percentile)

    def get_queue_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get queue latency (time from arrival to start processing) for a specific event type."""
        if (
            event_type in self._arrival_latency
            and not self._arrival_latency[event_type].is_empty()
            and event_type in self._start_latency
            and not self._start_latency[event_type].is_empty()
        ):
            arrival_latency = self._get_latency_percentile(event_type, self._arrival_latency, percentile)
            start_latency = self._get_latency_percentile(event_type, self._start_latency, percentile)
            return max(0.0, start_latency - arrival_latency)
        return 0.0

    def get_processing_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get processing latency (time from start processing to end) for a specific event type."""
        if (
            event_type in self._start_latency
            and not self._start_latency[event_type].is_empty()
            and event_type in self._end_latency
            and not self._end_latency[event_type].is_empty()
        ):
            start_latency = self._get_latency_percentile(event_type, self._start_latency, percentile)
            end_latency = self._get_latency_percentile(event_type, self._end_latency, percentile)
            return max(0.0, end_latency - start_latency)
        return 0.0

    def get_latency(self, event_type: str, percentile: float = 90) -> float:
        """Get the end-to-end latency for a specific event type (for backwards compatibility)."""
        if event_type not in self._end_latency or self._end_latency[event_type].is_empty():
            return 0.0
        latencies = self._end_latency[event_type].to_array()["value"]
        return float(np.percentile(latencies, percentile))

    def get_event_frequency(self, event_type: str) -> float:
        """Get the events per second for a specific event type."""
        if event_type not in self._event_frequency or self._event_frequency[event_type].is_empty():
            return 0.0

        series = self._event_frequency[event_type].to_array()
        if len(series) < 2:  # Need at least 2 points to calculate frequency
            return float(np.sum(series["value"]))  # Return total events if only one point

        # Calculate time window between newest and oldest events
        newest_time = series[-1]["timestamp"]  # Last event in the array
        oldest_time = series[0]["timestamp"]  # First event in the array
        time_window_ns = newest_time - oldest_time

        # Avoid division by zero and ensure minimum window
        if time_window_ns <= 0:
            return float(np.sum(series["value"]))  # Return total events if all events at same time

        # Convert window to seconds for per-second frequency
        time_window_seconds = max(1.0, time_window_ns / 1e9)  # At least 1 second window

        # Sum all events in our window
        total_events = np.sum(series["value"])

        # Calculate events per second
        return float(total_events / time_window_seconds)

    def get_system_metrics(self) -> HealthMetrics:
        """Get aggregated system metrics."""
        # Calculate average queue size
        avg_queue_size = float(np.mean(self._queue_size.to_array())) if not self._queue_size.is_empty() else 0.0

        # Calculate dropped events rate
        avg_dropped_events = self._calc_dropped_rate(self._dropped_events)

        # Calculate latency percentiles
        p50_arrival_latency = p90_arrival_latency = p99_arrival_latency = 0.0
        p50_start_latency = p90_start_latency = p99_start_latency = 0.0
        p50_end_latency = p90_end_latency = p99_end_latency = 0.0
        p50_queue_latency = p90_queue_latency = p99_queue_latency = 0.0
        p50_processing_latency = p90_processing_latency = p99_processing_latency = 0.0

        # Aggregate latencies across all event types
        arrival_latencies = []
        start_latencies = []
        end_latencies = []

        for event_type in self._arrival_latency:
            if not self._arrival_latency[event_type].is_empty():
                arrival_latencies.extend(self._arrival_latency[event_type].to_array()["value"])
            if not self._start_latency[event_type].is_empty():
                start_latencies.extend(self._start_latency[event_type].to_array()["value"])
            if not self._end_latency[event_type].is_empty():
                end_latencies.extend(self._end_latency[event_type].to_array()["value"])

        if arrival_latencies:
            p50_arrival_latency = float(np.percentile(arrival_latencies, 50))
            p90_arrival_latency = float(np.percentile(arrival_latencies, 90))
            p99_arrival_latency = float(np.percentile(arrival_latencies, 99))

        if start_latencies:
            p50_start_latency = float(np.percentile(start_latencies, 50))
            p90_start_latency = float(np.percentile(start_latencies, 90))
            p99_start_latency = float(np.percentile(start_latencies, 99))
            p50_queue_latency = p50_start_latency - p50_arrival_latency
            p90_queue_latency = p90_start_latency - p90_arrival_latency
            p99_queue_latency = p99_start_latency - p99_arrival_latency

        if end_latencies:
            p50_end_latency = float(np.percentile(end_latencies, 50))
            p90_end_latency = float(np.percentile(end_latencies, 90))
            p99_end_latency = float(np.percentile(end_latencies, 99))
            p50_processing_latency = p50_end_latency - p50_start_latency
            p90_processing_latency = p90_end_latency - p90_start_latency
            p99_processing_latency = p99_end_latency - p99_start_latency

        return HealthMetrics(
            avg_queue_size=avg_queue_size,
            avg_dropped_events=avg_dropped_events,
            p50_arrival_latency=p50_arrival_latency,
            p90_arrival_latency=p90_arrival_latency,
            p99_arrival_latency=p99_arrival_latency,
            p50_queue_latency=p50_queue_latency,
            p90_queue_latency=p90_queue_latency,
            p99_queue_latency=p99_queue_latency,
            p50_processing_latency=p50_processing_latency,
            p90_processing_latency=p90_processing_latency,
            p99_processing_latency=p99_processing_latency,
        )

    def start(self) -> None:
        """Start the metrics emission thread."""
        if self._emitter is None:
            return

        def emit_metrics():
            while not self._stop_event.is_set():
                self._emit()
                time.sleep(self._emit_interval_s)  # Emit at the configured interval

        self._stop_event.clear()
        self._emission_thread = threading.Thread(target=emit_metrics, daemon=True)
        self._emission_thread.start()

    def stop(self) -> None:
        """Stop the metrics emission thread."""
        if self._emission_thread is not None:
            self._stop_event.set()
            self._emission_thread.join()
            self._emission_thread = None

    def _monitor_loop(self) -> None:
        """Background thread for monitoring metrics."""
        while self._is_running:
            # Update metrics every second
            time.sleep(1)

            # Emit metrics if emitter is configured
            if self._emitter is not None:
                self._emit()

    def _get_latency_percentile(self, event_type: str, latencies: dict, percentile: float) -> float:
        if event_type not in latencies or latencies[event_type].is_empty():
            return 0.0
        _latencies = latencies[event_type].to_array()["value"]
        return float(np.percentile(_latencies, percentile))

    def _emit(self) -> None:
        """Emit all metrics to the configured emitter."""
        if self._emitter is None:
            return

        metrics = self.get_system_metrics()
        current_time = self.time_provider.time()
        tags = {"type": "health"}

        # Emit system-wide metrics
        self._emitter.emit(name="queue_size", value=metrics.avg_queue_size, tags=tags, timestamp=current_time)
        self._emitter.emit(
            name="dropped_events",
            value=metrics.avg_dropped_events,
            tags=tags,
            timestamp=current_time,
        )

        # Emit latency metrics with percentiles
        self._emitter.emit(
            name="arrival_latency.p50",
            value=metrics.p50_arrival_latency,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="arrival_latency.p90",
            value=metrics.p90_arrival_latency,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="arrival_latency.p99",
            value=metrics.p99_arrival_latency,
            tags=tags,
            timestamp=current_time,
        )

        self._emitter.emit(
            name="queue_latency.p50",
            value=metrics.p50_queue_latency,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="queue_latency.p90",
            value=metrics.p90_queue_latency,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="queue_latency.p99",
            value=metrics.p99_queue_latency,
            tags=tags,
            timestamp=current_time,
        )

        self._emitter.emit(
            name="processing_latency.p50",
            value=metrics.p50_processing_latency,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="processing_latency.p90",
            value=metrics.p90_processing_latency,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="processing_latency.p99",
            value=metrics.p99_processing_latency,
            tags=tags,
            timestamp=current_time,
        )

        # Collect all unique event types from all metric dictionaries
        event_types = set()
        event_types.update(self._dropped_events.keys())
        event_types.update(self._arrival_latency.keys())
        event_types.update(self._start_latency.keys())
        event_types.update(self._end_latency.keys())
        event_types.update(self._event_frequency.keys())

        # Emit per-event metrics for all event types
        for event_type in event_types:
            freq = self.get_event_frequency(event_type)
            processing_latency = self.get_processing_latency(event_type)
            dropped = (
                float(np.mean(self._dropped_events[event_type].to_array()["value"]))
                if not self._dropped_events[event_type].is_empty()
                else 0.0
            )
            arrival_latency = self.get_arrival_latency(event_type)
            queue_latency = self.get_queue_latency(event_type)

            event_tags = {"type": "health", "event_type": event_type}
            self._emitter.emit("event_frequency", freq, event_tags, current_time)
            self._emitter.emit("event_dropped", dropped, event_tags, current_time)
            self._emitter.emit("event_arrival_latency", arrival_latency, event_tags, current_time)
            self._emitter.emit("event_queue_latency", queue_latency, event_tags, current_time)
            self._emitter.emit("event_processing_latency", processing_latency, event_tags, current_time)

    def _calc_weighted_latency_avg(self, latency_dict) -> tuple[float, float, float]:
        """Calculate weighted average latency and percentiles across all event types.

        Returns:
            Tuple of (weighted_average, p90, p99) latencies in milliseconds.
        """
        if not latency_dict:
            return 0.0, 0.0, 0.0

        all_latencies = []
        all_timestamps = []

        for series in latency_dict.values():
            if not series.is_empty():
                arr = series.to_array()
                all_latencies.extend(arr["value"])
                all_timestamps.extend(arr["timestamp"])

        if not all_latencies:
            return 0.0, 0.0, 0.0

        # Convert to numpy arrays for efficient computation
        latencies = np.array(all_latencies)
        timestamps = np.array(all_timestamps)

        # Sort by timestamp to maintain temporal relationship
        sort_idx = np.argsort(timestamps)
        latencies = latencies[sort_idx]

        # Calculate statistics
        avg = float(np.mean(latencies))
        p90 = float(np.percentile(latencies, 90))
        p99 = float(np.percentile(latencies, 99))

        return avg, p90, p99

    def _calc_dropped_rate(self, dropped_dict) -> float:
        """Calculate the rate of dropped events per second."""
        if not dropped_dict:
            return 0.0

        total_dropped = 0
        total_time_s = 0.0

        for series in dropped_dict.values():
            if not series.is_empty():
                arr = series.to_array()
                if len(arr) < 2:
                    total_dropped += np.sum(arr["value"])
                    continue

                # Sum drops and calculate time window
                total_dropped += np.sum(arr["value"])
                time_window_ns = arr[-1]["timestamp"] - arr[0]["timestamp"]  # newest - oldest timestamp
                total_time_s += time_window_ns / 1e9

        if total_time_s <= 0:
            # If we don't have enough time data, but we have drops, report them
            return float(total_dropped) if total_dropped > 0 else 0.0

        return float(total_dropped / max(1.0, total_time_s))  # At least 1 second window
