"""
Prometheus Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to Prometheus.
"""

from typing import Dict, Optional

from prometheus_client import REGISTRY, Counter, Gauge, Summary, push_to_gateway

from qubx import logger
from qubx.core.interfaces import IMetricEmitter


class PrometheusMetricEmitter(IMetricEmitter):
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
    ):
        """
        Initialize the Prometheus Metric Emitter.

        Args:
            strategy_name: Name of the strategy (used in metric names)
            pushgateway_url: URL of the Prometheus Pushgateway (optional)
            expose_http: Whether to expose metrics via HTTP endpoint
            http_port: Port to expose HTTP metrics on
            namespace: Namespace to prefix all metric names with
        """
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

    def _get_or_create_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create a gauge metric."""
        tag_keys = list(tags.keys()) if tags else []
        cache_key = f"{name}:{','.join(tag_keys)}"

        if cache_key not in self._gauges:
            self._gauges[cache_key] = Gauge(
                f"{self._namespace}_{name}", f"Gauge metric for {name}", ["strategy"] + tag_keys
            )

        return self._gauges[cache_key]

    def _get_or_create_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create a counter metric."""
        tag_keys = list(tags.keys()) if tags else []
        cache_key = f"{name}:{','.join(tag_keys)}"

        if cache_key not in self._counters:
            self._counters[cache_key] = Counter(
                f"{self._namespace}_{name}", f"Counter metric for {name}", ["strategy"] + tag_keys
            )

        return self._counters[cache_key]

    def _get_or_create_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Summary:
        """Get or create a summary metric."""
        tag_keys = list(tags.keys()) if tags else []
        cache_key = f"{name}:{','.join(tag_keys)}"

        if cache_key not in self._summaries:
            self._summaries[cache_key] = Summary(
                f"{self._namespace}_{name}", f"Summary metric for {name}", ["strategy"] + tag_keys
            )

        return self._summaries[cache_key]

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
