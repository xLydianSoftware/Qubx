"""
Prometheus Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to Prometheus.
"""

from typing import Dict, List, Optional

from prometheus_client import REGISTRY, Counter, Gauge, Summary, push_to_gateway

from qubx import logger
from qubx.core.interfaces import IStrategyContext
from qubx.metrics.base import BaseMetricEmitter


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
        """
        super().__init__(stats_to_emit, stats_interval)

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
                ["strategy"] + list(label_dict.keys()),
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
                ["strategy"] + list(label_dict.keys()),
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
                ["strategy"] + list(label_dict.keys()),
                registry=self._registry,
            )
        return self._summaries[key]

    def emit_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Emit a gauge metric to Prometheus.

        Args:
            name: Name of the metric
            value: Current value of the metric
            tags: Dictionary of tags/labels for the metric
        """
        try:
            tags = tags or {}
            gauge = self._get_or_create_gauge(name, tags)
            gauge.labels(strategy=self._strategy_name, **tags).set(value)

            if self._pushgateway_url:
                push_to_gateway(
                    self._pushgateway_url, job=f"{self._namespace}_{self._strategy_name}", registry=self._registry
                )
        except Exception as e:
            logger.error(f"[PrometheusMetricEmitter] Failed to emit gauge {name}: {e}")

    def emit_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Emit a counter metric to Prometheus.

        Args:
            name: Name of the metric
            value: Amount to increment the counter
            tags: Dictionary of tags/labels for the metric
        """
        try:
            tags = tags or {}
            counter = self._get_or_create_counter(name, tags)
            counter.labels(strategy=self._strategy_name, **tags).inc(value)

            if self._pushgateway_url:
                push_to_gateway(
                    self._pushgateway_url, job=f"{self._namespace}_{self._strategy_name}", registry=self._registry
                )
        except Exception as e:
            logger.error(f"[PrometheusMetricEmitter] Failed to emit counter {name}: {e}")

    def emit_summary(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Emit a summary metric to Prometheus.

        Args:
            name: Name of the metric
            value: Value to add to the summary
            tags: Dictionary of tags/labels for the metric
        """
        try:
            tags = tags or {}
            summary = self._get_or_create_summary(name, tags)
            summary.labels(strategy=self._strategy_name, **tags).observe(value)

            if self._pushgateway_url:
                push_to_gateway(
                    self._pushgateway_url, job=f"{self._namespace}_{self._strategy_name}", registry=self._registry
                )
        except Exception as e:
            logger.error(f"[PrometheusMetricEmitter] Failed to emit summary {name}: {e}")

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
