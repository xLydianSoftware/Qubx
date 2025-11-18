from typing import Callable

from qubx.core.basics import dt_64
from qubx.core.interfaces import HealthMetrics, IHealthMonitor


class DummyHealthMonitor(IHealthMonitor):
    """No-op implementation of health metrics monitoring."""

    def __call__(self, event_type: str) -> "DummyHealthMonitor":
        return self

    def __enter__(self) -> "DummyHealthMonitor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def on_data_arrival(self, exchange: str, event_type: str, event_time: dt_64) -> None:
        pass

    def record_order_submit_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        pass

    def record_order_submit_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        pass

    def record_order_cancel_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        pass

    def record_order_cancel_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        pass

    def set_event_queue_size(self, size: int) -> None:
        pass

    def set_is_connected(self, exchange: str, is_connected: Callable[[], bool]) -> None:
        pass

    def watch(self, name: str = ""):
        def decorator(func):
            return func

        return decorator

    def is_connected(self, exchange: str) -> bool:
        return True

    def get_last_event_time(self, exchange: str, event_type: str) -> dt_64 | None:
        return None

    def get_last_event_times(self, exchange: str) -> dict[str, dt_64]:
        return {}

    def get_event_frequency(self, exchange: str, event_type: str) -> float:
        return 1.0

    def get_queue_size(self) -> int:
        return 0

    def get_data_latency(self, exchange: str, event_type: str, percentile: float = 90) -> float:
        return 0.0

    def get_data_latencies(self, exchange: str, percentile: float = 90) -> dict[str, float]:
        return {}

    def get_order_submit_latency(self, exchange: str, percentile: float = 90) -> float:
        return 0.0

    def get_order_cancel_latency(self, exchange: str, percentile: float = 90) -> float:
        return 0.0

    def get_execution_latency(self, scope: str, percentile: float = 90) -> float:
        return 0.0

    def get_execution_latencies(self) -> dict[str, float]:
        return {}

    def get_system_metrics(self) -> HealthMetrics:
        return HealthMetrics(
            avg_queue_size=0.0,
            max_queue_size=0.0,
            p50_data_latency=0.0,
            p90_data_latency=0.0,
            p99_data_latency=0.0,
            p50_order_submit_latency=0.0,
            p90_order_submit_latency=0.0,
            p99_order_submit_latency=0.0,
            p50_order_cancel_latency=0.0,
            p90_order_cancel_latency=0.0,
            p99_order_cancel_latency=0.0,
        )

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
