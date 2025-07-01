"""
Prometheus Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to Prometheus.
"""

from typing import Dict, List, Literal, Optional

from prometheus_client import REGISTRY, Counter, Gauge, Summary, push_to_gateway

from qubx import logger
from qubx.core.basics import Signal, dt_64
from qubx.core.interfaces import IAccountViewer, IStrategyContext
from qubx.emitters.base import BaseMetricEmitter

# Define metric types
MetricType = Literal["gauge", "counter", "summary"]


class PrometheusMetricEmitter(BaseMetricEmitter):
    """
    Emits metrics to Prometheus using the Prometheus client library.

    This emitter can push metrics to a Prometheus Pushgateway or expose them
    via an HTTP endpoint for scraping.
    """

    def __init__(
        self,
        strategy_name: str,
        pushgateway_url: Optional[str] = None,
        expose_http: bool = False,
        http_port: int = 8000,
        namespace: str = "qubx",
        stats_to_emit: Optional[List[str]] = None,
        stats_interval: str = "1m",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Prometheus Metric Emitter.

        Args:
            strategy_name: Name of the strategy (used in metric names)
            pushgateway_url: URL of the Prometheus Pushgateway (optional)
            expose_http: Whether to expose metrics via HTTP endpoint
            http_port: Port to expose HTTP metrics on
            namespace: Namespace to prefix all metric names with
            stats_to_emit: Optional list of specific stats to emit
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
        """
        # Initialize default tags with strategy name
        default_tags = tags or {}
        default_tags["strategy"] = strategy_name

        super().__init__(stats_to_emit, stats_interval, default_tags)

        self._strategy_name = strategy_name
        self._pushgateway_url = pushgateway_url
        self._namespace = namespace
        self._registry = REGISTRY

        # Cache for created metrics to avoid recreating them
        self._gauges: Dict[str, Gauge] = {}
        self._counters: Dict[str, Counter] = {}
        self._summaries: Dict[str, Summary] = {}

        if expose_http:
            from prometheus_client import start_http_server

            start_http_server(http_port)
            logger.info(f"[PrometheusMetricEmitter] Started HTTP server on port {http_port}")

        logger.info(
            f"[PrometheusMetricEmitter] Initialized for strategy '{strategy_name}'"
            f"{' with Pushgateway at ' + pushgateway_url if pushgateway_url else ''}"
        )

    def _get_or_create_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """
        Get or create a Prometheus gauge.

        Args:
            name: Name of the gauge
            labels: Dictionary of labels for the gauge

        Returns:
            Gauge: The Prometheus gauge
        """
        label_dict = labels or {}
        key = f"{name}_{sorted(label_dict.keys())}"
        if key not in self._gauges:
            self._gauges[key] = Gauge(
                f"{self._namespace}_{name}",
                f"{name.replace('_', ' ')} metric",
                list(label_dict.keys()),
                registry=self._registry,
            )
        return self._gauges[key]

    def _get_or_create_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        """
        Get or create a Prometheus counter.

        Args:
            name: Name of the counter
            labels: Dictionary of labels for the counter

        Returns:
            Counter: The Prometheus counter
        """
        label_dict = labels or {}
        key = f"{name}_{sorted(label_dict.keys())}"
        if key not in self._counters:
            self._counters[key] = Counter(
                f"{self._namespace}_{name}",
                f"{name.replace('_', ' ')} metric",
                list(label_dict.keys()),
                registry=self._registry,
            )
        return self._counters[key]

    def _get_or_create_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Summary:
        """
        Get or create a Prometheus summary.

        Args:
            name: Name of the summary
            labels: Dictionary of labels for the summary

        Returns:
            Summary: The Prometheus summary
        """
        label_dict = labels or {}
        key = f"{name}_{sorted(label_dict.keys())}"
        if key not in self._summaries:
            self._summaries[key] = Summary(
                f"{self._namespace}_{name}",
                f"{name.replace('_', ' ')} metric",
                list(label_dict.keys()),
                registry=self._registry,
            )
        return self._summaries[key]

    def _emit_impl(self, name: str, value: float, tags: Dict[str, str], timestamp: dt_64 | None = None) -> None:
        """
        Implementation of emit for Prometheus.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric (ignored in Prometheus)
        """
        try:
            # Extract metric type from tags if present, default to gauge
            metric_type = tags.pop("metric_type", "gauge") if "metric_type" in tags else "gauge"

            if metric_type == "counter":
                counter = self._get_or_create_counter(name, tags)
                counter.labels(**tags).inc(value)
            elif metric_type == "summary":
                summary = self._get_or_create_summary(name, tags)
                summary.labels(**tags).observe(value)
            else:  # Default to gauge
                gauge = self._get_or_create_gauge(name, tags)
                gauge.labels(**tags).set(value)

            # Push to gateway if configured
            if self._pushgateway_url:
                push_to_gateway(
                    self._pushgateway_url, job=f"{self._namespace}_{self._strategy_name}", registry=self._registry
                )
        except Exception as e:
            logger.error(f"[PrometheusMetricEmitter] Failed to emit metric {name}: {e}")

    def emit(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        timestamp: dt_64 | None = None,
        metric_type: MetricType = "gauge",
    ) -> None:
        """
        Emit a metric to Prometheus.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric
            timestamp: Optional timestamp for the metric (ignored in Prometheus)
            metric_type: Type of metric (gauge, counter, summary)
        """
        # Add metric type to tags
        merged_tags = self._merge_tags(tags)
        merged_tags["metric_type"] = metric_type

        # Call the implementation
        self._emit_impl(name, value, merged_tags, timestamp)

    def emit_strategy_stats(self, context: IStrategyContext) -> None:
        """
        Emit standard strategy statistics to Prometheus.

        This method overrides the base implementation to add Prometheus-specific
        functionality like pushing to the Pushgateway.

        Args:
            context: The strategy context to get statistics from
        """
        # Call the parent implementation to emit the standard stats
        super().emit_strategy_stats(context)

        # Push all metrics to Pushgateway in one go if configured
        if self._pushgateway_url:
            try:
                push_to_gateway(
                    self._pushgateway_url, job=f"{self._namespace}_{self._strategy_name}", registry=self._registry
                )
            except Exception as e:
                logger.error(f"[PrometheusMetricEmitter] Failed to push metrics to gateway: {e}")

    def emit_signals(self, time: dt_64, signals: list[Signal], account: IAccountViewer) -> None:
        """
        Emit signals as Prometheus metrics.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information
        """
        if not signals:
            return

        try:
            for signal in signals:
                # Create labels for the signal
                labels = {
                    "symbol": signal.instrument.symbol,
                    "exchange": signal.instrument.exchange,
                    "group": signal.group if signal.group else "default",
                    "is_service": str(signal.is_service).lower(),
                }

                # Emit the signal value as a gauge
                gauge = self._get_or_create_gauge("signal_value", labels)
                gauge.labels(**labels).set(signal.signal)

                # Emit price-related metrics if available
                if signal.price is not None:
                    price_gauge = self._get_or_create_gauge("signal_price", labels)
                    price_gauge.labels(**labels).set(signal.price)

                if signal.stop is not None:
                    stop_gauge = self._get_or_create_gauge("signal_stop", labels)
                    stop_gauge.labels(**labels).set(signal.stop)

                if signal.take is not None:
                    take_gauge = self._get_or_create_gauge("signal_take", labels)
                    take_gauge.labels(**labels).set(signal.take)

                if signal.reference_price is not None:
                    ref_price_gauge = self._get_or_create_gauge("signal_reference_price", labels)
                    ref_price_gauge.labels(**labels).set(signal.reference_price)

            # Push to gateway if configured
            if self._pushgateway_url:
                try:
                    push_to_gateway(
                        self._pushgateway_url, job=f"{self._namespace}_{self._strategy_name}", registry=self._registry
                    )
                except Exception as e:
                    logger.error(f"[PrometheusMetricEmitter] Failed to push signal metrics to gateway: {e}")

        except Exception as e:
            logger.error(f"[PrometheusMetricEmitter] Failed to emit signals: {e}")
