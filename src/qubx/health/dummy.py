from qubx.core.basics import dt_64
from qubx.core.interfaces import HealthMetrics, IHealthMonitor


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

    def record_order_submit_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record a order submit request time."""
        pass

    def record_order_submit_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record a order submit response time."""
        pass

    def record_order_cancel_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record a order cancel request time."""
        pass

    def record_order_cancel_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record a order cancel response time."""
        pass

    def record_event_dropped(self, event_type: str) -> None:
        """Record that an event was dropped."""
        pass

    def on_data_arrival(self, event_type: str, event_time: dt_64) -> None:
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

    def is_connected(self, exchange: str) -> bool:
        """Check if exchange is connected."""
        return True

    def get_last_event_time(self, exchange: str, event_type: str) -> dt_64 | None:
        """Get the last event time for a specific event type."""
        return None

    def get_last_event_times(self, exchange: str) -> dict[str, dt_64]:
        """Get the last event times for all event types."""
        return {}

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

    def get_execution_latency(self, scope: str, percentile: float = 90) -> float:
        """Get execution latency for a specific scope."""
        return 0.0

    def get_execution_latencies(self) -> dict[str, float]:
        """Get all execution latencies."""
        return {}

    def get_event_frequency(self, event_type: str) -> float:
        """Get the events per second for a specific event type."""
        return 0.0

    def get_system_metrics(self) -> HealthMetrics:
        """Get system-wide metrics."""
        return HealthMetrics(
            queue_size=0.0,
            drop_rate=0.0,
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

    def watch(self, name: str = ""):
        """No-op decorator function that returns the function unchanged.

        This is provided for API compatibility with BaseHealthMonitor.
        """

        def decorator(func):
            return func

        return decorator
