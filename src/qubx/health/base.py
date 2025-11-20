import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, dt_64, td_64
from qubx.core.interfaces import IHealthMonitor, IMetricEmitter, ITimeProvider, LatencyMetrics
from qubx.core.utils import recognize_timeframe
from qubx.utils import convert_tf_str_td64
from qubx.utils.collections import DequeFloat64, DequeIndicator

STALE_THRESHOLDS = {
    "quote": "2min",
    "orderbook": "2min",
    "trade": "30min",
}


@dataclass
class TimingContext:
    """Context for timing a code block."""

    event_type: str
    start_time: dt_64


class BaseHealthMonitor(IHealthMonitor):
    """Base implementation of health metrics monitoring using Deque for tracking."""

    def __init__(
        self,
        time_provider: ITimeProvider,
        emitter: IMetricEmitter | None = None,
        emit_interval: str = "1s",
        channel: CtrlChannel | None = None,
        monitor_interval: str = "1s",
        buffer_size: int = 1000,
        emit_health: bool = True,
    ):
        """Initialize the health metrics monitor.

        Args:
            time_provider: Provider for time information
            emitter: Optional metric emitter for sending metrics to external systems
            emit_interval: Interval to emit metrics, e.g. "1s", "500ms", "5m" (default: "1s")
            channel: Optional data channel to monitor for queue size
            queue_monitor_interval: Interval to check queue size, e.g. "100ms", "500ms" (default: "100ms")
            buffer_size: Size of buffer for storing metrics
            emit_health: Whether to emit health metrics (default: True)
        """
        self.time_provider = time_provider
        self._emitter = emitter
        self._channel = channel
        self._emit_health = emit_health

        # Convert emit interval to nanoseconds
        self._emit_interval_ns = recognize_timeframe(emit_interval)
        self._emit_interval_s = self._emit_interval_ns / 1_000_000_000  # Convert to seconds for sleep

        # Convert queue monitor interval to seconds
        self._monitor_interval_ns = recognize_timeframe(monitor_interval)
        self._monitor_interval_s = self._monitor_interval_ns / 1_000_000_000  # Convert to seconds for sleep

        # Initialize metrics storage
        self._queue_size = DequeFloat64(buffer_size)  # Store last 1000 queue size measurements

        # Data arrival tracking (per exchange per event_type) - uses tuple keys (exchange, event_type)
        self._data_latency = defaultdict(lambda: DequeIndicator(buffer_size))
        self._event_frequency = defaultdict(lambda: DequeIndicator(buffer_size))
        self._last_event_time = {}  # dict[(instrument, event_type), dt_64]

        # Subscription tracking for filtering unsubscribed data
        self._active_subscriptions: set[tuple[Instrument, str]] = set()

        # Order tracking (per exchange)
        self._order_submit_requests = {}  # dict[(exchange, client_id), dt_64]
        self._order_submit_latencies = defaultdict(lambda: DequeIndicator(buffer_size))  # Key: exchange
        self._order_cancel_requests = {}  # dict[(exchange, client_id), dt_64]
        self._order_cancel_latencies = defaultdict(lambda: DequeIndicator(buffer_size))  # Key: exchange

        # Connection tracking (per exchange)
        self._is_connected_callbacks = {}  # dict[exchange, Callable[[], bool]]

        # Execution latency tracking (unchanged)
        self._execution_latency = defaultdict(lambda: DequeIndicator(buffer_size))

        # Cleanup mechanism for orphaned order requests (60 second timeout)
        self._cleanup_interval_ns = recognize_timeframe("60s")
        self._cleanup_interval_s = self._cleanup_interval_ns / 1_000_000_000
        self._last_cleanup_time = None

        # Initialize emission thread control
        self._stop_event = threading.Event()
        self._emission_thread = None

        # Initialize monitor thread control
        self._is_running = False
        self._monitor_thread = None

        # Thread-local storage for timing contexts
        self._thread_local = threading.local()

    def _get_timing_stack(self) -> list[TimingContext]:
        if not hasattr(self._thread_local, "timing_stack"):
            self._thread_local.timing_stack = []
        return self._thread_local.timing_stack

    def __call__(self, event_type: str) -> "BaseHealthMonitor":
        self._get_timing_stack().append(TimingContext(event_type, self.time_provider.time()))
        return self

    def __enter__(self) -> "BaseHealthMonitor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        stack = self._get_timing_stack()
        if stack:
            context = stack.pop()
            current_time = self.time_provider.time()
            current_time_ns = current_time.astype("datetime64[ns]").astype(int)
            start_time_ns = context.start_time.astype("datetime64[ns]").astype(int)
            duration = (current_time_ns - start_time_ns) / 1e6  # Convert to ms

            # Record execution latency using event_type as key, not the context object
            self._execution_latency[context.event_type].push_back_fields(current_time_ns, duration)

    def on_data_arrival(self, instrument: Instrument, event_type: str, event_time: dt_64) -> None:
        # Only track metrics for actively subscribed data types
        event_type = DataType.from_str(event_type)[0]
        if (instrument, event_type) not in self._active_subscriptions:
            return

        current_time = self.time_provider.time()
        current_time_ns = current_time.astype("datetime64[ns]").astype(int)
        event_time_ns = event_time.astype("datetime64[ns]").astype(int)
        data_latency = (current_time_ns - event_time_ns) / 1e6  # Convert to milliseconds

        # Use tuple key (exchange, event_type) for per-exchange tracking
        exchange_key = (instrument.exchange, event_type)

        # Record data latency and event frequency
        self._data_latency[exchange_key].push_back_fields(current_time_ns, data_latency)

        # Store last event time
        instrument_key = (instrument, event_type)
        self._event_frequency[instrument_key].push_back_fields(current_time_ns, 1.0)
        self._last_event_time[instrument_key] = current_time

    def subscribe(self, instrument: Instrument, event_type: str) -> None:
        """
        Register active subscription for health tracking.

        Args:
            instrument: The instrument being subscribed to
            event_type: The data type being subscribed to (e.g., 'ohlc', 'quote', 'orderbook')
        """
        self._active_subscriptions.add((instrument, DataType.from_str(event_type)[0]))

    def unsubscribe(self, instrument: Instrument, event_type: str) -> None:
        """
        Remove subscription and cleanup stored metrics.

        Args:
            instrument: The instrument being unsubscribed from
            event_type: The data type being unsubscribed from
        """
        key = (instrument, DataType.from_str(event_type)[0])
        self._active_subscriptions.discard(key)

        # Clean up stored metrics immediately
        self._last_event_time.pop(key, None)
        self._event_frequency.pop(key, None)

    def record_order_submit_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record order submit request timestamp."""
        key = (exchange, client_id)
        self._order_submit_requests[key] = event_time

    def record_order_submit_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record order submit response and calculate latency."""
        key = (exchange, client_id)
        if key in self._order_submit_requests:
            request_time = self._order_submit_requests[key]
            current_time = self.time_provider.time()
            current_time_ns = current_time.astype("datetime64[ns]").astype(int)
            request_time_ns = request_time.astype("datetime64[ns]").astype(int)
            latency = (current_time_ns - request_time_ns) / 1e6  # Convert to milliseconds

            # Record latency per exchange
            self._order_submit_latencies[exchange].push_back_fields(current_time_ns, latency)

            # Remove request from tracking
            del self._order_submit_requests[key]

    def record_order_cancel_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record order cancel request timestamp."""
        key = (exchange, client_id)
        self._order_cancel_requests[key] = event_time

    def record_order_cancel_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """Record order cancel response and calculate latency."""
        key = (exchange, client_id)
        if key in self._order_cancel_requests:
            request_time = self._order_cancel_requests[key]
            current_time = self.time_provider.time()
            current_time_ns = current_time.astype("datetime64[ns]").astype(int)
            request_time_ns = request_time.astype("datetime64[ns]").astype(int)
            latency = (current_time_ns - request_time_ns) / 1e6  # Convert to milliseconds

            # Record latency per exchange
            self._order_cancel_latencies[exchange].push_back_fields(current_time_ns, latency)

            # Remove request from tracking
            del self._order_cancel_requests[key]

    def set_is_connected(self, exchange: str, is_connected: Callable[[], bool]) -> None:
        """Set the connection status callback for an exchange."""
        self._is_connected_callbacks[exchange] = is_connected

    def set_event_queue_size(self, size: int) -> None:
        self._queue_size.push_back(float(size))

    def get_queue_size(self) -> int:
        if self._queue_size.is_empty():
            return 0
        return int(self._queue_size.back())

    def is_connected(self, exchange: str) -> bool:
        """Check if exchange is connected."""
        if exchange in self._is_connected_callbacks:
            try:
                return self._is_connected_callbacks[exchange]()
            except Exception:
                return False
        return True  # Default to True if no callback registered

    def get_last_event_time(self, instrument: Instrument, event_type: str) -> dt_64 | None:
        return self._last_event_time.get((instrument, event_type))

    def get_last_event_times_by_exchange(self, exchange: str) -> dict[str, dt_64]:
        """Get all last event times for a specific exchange."""
        result = {}
        for (instrument, event_type), event_time in self._last_event_time.items():
            if instrument.exchange == exchange:
                result[event_type] = event_time
        return result

    def is_stale(self, instrument: Instrument, event_type: str, stale_delta: str | td_64 | None = None) -> bool:
        if stale_delta is None:
            stale_delta = STALE_THRESHOLDS.get(event_type, None)
            if stale_delta is None:
                return False
        if isinstance(stale_delta, str):
            stale_delta = convert_tf_str_td64(stale_delta)
        assert isinstance(stale_delta, td_64)
        current_time = self.time_provider.time()
        last_event_time = self.get_last_event_time(instrument, event_type)
        if last_event_time is None:
            return True
        time_diff = current_time - last_event_time
        return bool(time_diff > stale_delta)

    def get_event_frequency(self, instrument: Instrument, event_type: str) -> float:
        """Get the events per second for a specific event type on an instrument."""
        key = (instrument, event_type)
        if key not in self._event_frequency or self._event_frequency[key].is_empty():
            return 0.0
        series = self._event_frequency[key].to_array()
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

    def get_data_latency(self, exchange: str, event_type: str, percentile: float = 90) -> float:
        """Get data latency for a specific event type on an exchange."""
        key = (exchange, event_type)
        if key not in self._data_latency or self._data_latency[key].is_empty():
            return 0.0
        latencies = self._data_latency[key].to_array()["value"]
        return float(np.percentile(latencies, percentile))

    def get_data_latencies(self, exchange: str, percentile: float = 90) -> dict[str, float]:
        """Get all data latencies for an exchange."""
        result = {}
        for ex, event_type in self._data_latency.keys():
            if ex == exchange:
                result[event_type] = self.get_data_latency(exchange, event_type, percentile)
        return result

    def get_order_submit_latency(self, exchange: str, percentile: float = 90) -> float:
        """Get order submit latency for an exchange."""
        if exchange not in self._order_submit_latencies or self._order_submit_latencies[exchange].is_empty():
            return 0.0
        latencies = self._order_submit_latencies[exchange].to_array()["value"]
        return float(np.percentile(latencies, percentile))

    def get_order_cancel_latency(self, exchange: str, percentile: float = 90) -> float:
        """Get order cancel latency for an exchange."""
        if exchange not in self._order_cancel_latencies or self._order_cancel_latencies[exchange].is_empty():
            return 0.0
        latencies = self._order_cancel_latencies[exchange].to_array()["value"]
        return float(np.percentile(latencies, percentile))

    def get_execution_latency(self, scope: str, percentile: float = 90) -> float:
        return self._get_latency_percentile(scope, self._execution_latency, percentile)

    def get_execution_latencies(self) -> dict[str, float]:
        scopes = self._execution_latency.keys()
        return {scope: self.get_execution_latency(scope) for scope in scopes}

    def get_exchange_latencies(self, exchange: str, percentile: float = 90) -> LatencyMetrics:
        data_latencies = []
        for (ex, _), latency in self._data_latency.items():
            if ex == exchange:
                if not latency.is_empty():
                    data_latencies.extend(latency.to_array()["value"])

        data_feed = float(np.percentile(data_latencies, percentile)) if data_latencies else 0.0

        order_submit_latencies = []
        if exchange in self._order_submit_latencies:
            latency = self._order_submit_latencies[exchange]
            if not latency.is_empty():
                order_submit_latencies.extend(latency.to_array()["value"])

        order_submit = float(np.percentile(order_submit_latencies, percentile)) if order_submit_latencies else 0.0

        order_cancel_latencies = []
        if exchange in self._order_cancel_latencies:
            latency = self._order_cancel_latencies[exchange]
            if not latency.is_empty():
                order_cancel_latencies.extend(latency.to_array()["value"])

        order_cancel = float(np.percentile(order_cancel_latencies, percentile)) if order_cancel_latencies else 0.0

        return LatencyMetrics(
            data_feed=data_feed,
            order_submit=order_submit,
            order_cancel=order_cancel,
        )

    def start(self) -> None:
        """Start the metrics emission thread and queue monitoring thread."""
        # Start queue size monitoring if channel is provided
        if self._channel is not None:
            self._is_running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

        # Start metrics emission if emitter is provided
        if self._emitter is not None:

            def emit_metrics():
                while not self._stop_event.is_set():
                    try:
                        self._emit()
                    except Exception as e:
                        logger.error(f"Error emitting metrics: {e}")
                    finally:
                        time.sleep(self._emit_interval_s)

            self._stop_event.clear()
            self._emission_thread = threading.Thread(target=emit_metrics, daemon=True)
            self._emission_thread.start()

    def stop(self) -> None:
        """Stop the metrics emission thread and queue monitoring thread."""

        # Stop queue size monitoring
        if self._monitor_thread is not None:
            self._is_running = False
            self._monitor_thread.join()
            self._monitor_thread = None

        # Stop metrics emission
        if self._emission_thread is not None:
            self._stop_event.set()
            self._emission_thread.join()
            self._emission_thread = None

    def watch(self, scope_name: str = ""):
        def decorator(func):
            nonlocal scope_name
            # Use function's qualified name if scope_name is empty
            if scope_name == "":
                scope_name = f"{func.__module__}.{func.__qualname__}"

            def wrapper(*args, **kwargs):
                with self(scope_name):
                    return func(*args, **kwargs)

            # Preserve function metadata
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            wrapper.__module__ = func.__module__
            wrapper.__qualname__ = func.__qualname__
            wrapper.__annotations__ = func.__annotations__

            return wrapper

        return decorator

    def _monitor_loop(self) -> None:
        while self._is_running:
            try:
                # Update queue size if we have a channel
                if self._channel is not None:
                    current_size = self._channel._queue.qsize()
                    self.set_event_queue_size(current_size)

                # Perform cleanup of old order requests
                self._cleanup_old_order_requests()
            except Exception as e:
                logger.error(f"Error monitoring queue size: {e}")
            finally:
                time.sleep(self._monitor_interval_s)

    def _cleanup_old_order_requests(self) -> None:
        """Clean up order requests older than 60 seconds."""
        current_time = self.time_provider.time()

        # Only run cleanup once per cleanup interval
        if self._last_cleanup_time is not None:
            time_since_last_cleanup = current_time - self._last_cleanup_time
            time_since_last_cleanup_ns = time_since_last_cleanup.astype("datetime64[ns]").astype(int)
            if time_since_last_cleanup_ns < self._cleanup_interval_ns:
                return

        self._last_cleanup_time = current_time
        current_time_ns = current_time.astype("datetime64[ns]").astype(int)
        timeout_ns = self._cleanup_interval_ns

        # Clean up submit requests
        expired_submit = []
        for (exchange, client_id), request_time in self._order_submit_requests.items():
            request_time_ns = request_time.astype("datetime64[ns]").astype(int)
            if (current_time_ns - request_time_ns) > timeout_ns:
                expired_submit.append((exchange, client_id))

        for key in expired_submit:
            del self._order_submit_requests[key]

        # Clean up cancel requests
        expired_cancel = []
        for (exchange, client_id), request_time in self._order_cancel_requests.items():
            request_time_ns = request_time.astype("datetime64[ns]").astype(int)
            if (current_time_ns - request_time_ns) > timeout_ns:
                expired_cancel.append((exchange, client_id))

        for key in expired_cancel:
            del self._order_cancel_requests[key]

    def _get_latency_percentile(self, event_type: str, latencies: dict, percentile: float) -> float:
        if event_type not in latencies or latencies[event_type].is_empty():
            return 0.0
        _latencies = latencies[event_type].to_array()["value"]
        return float(np.percentile(_latencies, percentile))

    def _get_exchanges(self) -> list[str]:
        return list(set(ex for ex, _ in self._data_latency.keys()))

    def _emit(self) -> None:
        """Emit all metrics to the configured emitter."""
        if not self._emit_health or self._emitter is None:
            return

        current_time = self.time_provider.time()

        self._emitter.emit(
            name="queue_size", value=self.get_queue_size(), tags={"type": "health"}, timestamp=current_time
        )

        exchanges = self._get_exchanges()
        for exchange in exchanges:
            self._emit_exchange_latencies(exchange)

    def _emit_exchange_latencies(self, exchange: str) -> None:
        current_time = self.time_provider.time()
        metrics = self.get_exchange_latencies(exchange, percentile=90)
        tags = {"type": "health", "exchange": exchange}
        assert self._emitter is not None
        self._emitter.emit(
            name="latency.p90.data",
            value=metrics.data_feed,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="latency.p90.order_submit",
            value=metrics.order_submit,
            tags=tags,
            timestamp=current_time,
        )
        self._emitter.emit(
            name="latency.p90.order_cancel",
            value=metrics.order_cancel,
            tags=tags,
            timestamp=current_time,
        )
