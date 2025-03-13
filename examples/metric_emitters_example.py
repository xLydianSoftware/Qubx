#!/usr/bin/env python
"""
Example script demonstrating how to use the metric emitters.

This script shows how to use the different metric emitters to send metrics
to various monitoring systems.
"""

import time

import pandas as pd

from qubx.emitters.base import BaseMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.emitters.prometheus import PrometheusMetricEmitter
from qubx.emitters.questdb import QuestDBMetricEmitter


def create_custom_emitter():
    """Create a custom emitter that prints metrics to the console."""

    class ConsoleMetricEmitter(BaseMetricEmitter):
        """A simple emitter that prints metrics to the console."""

        def _emit_impl(self, name, value, tags, timestamp=None):
            """Print the metric to the console."""
            timestamp_str = f" at {timestamp}" if timestamp is not None else ""
            tags_str = f" with tags {tags}" if tags else ""
            print(f"METRIC: {name} = {value}{timestamp_str}{tags_str}")

    return ConsoleMetricEmitter()


def main():
    """Run the example."""
    # Create a console emitter
    console_emitter = create_custom_emitter()

    # Create a Prometheus emitter (without actually starting the HTTP server)
    prometheus_emitter = PrometheusMetricEmitter(
        strategy_name="example_strategy",
        expose_http=False,  # Don't start the HTTP server
    )

    # Create a QuestDB emitter (without actually connecting to QuestDB)
    questdb_emitter = QuestDBMetricEmitter(
        strategy_name="example_strategy",
        host="localhost",  # This won't actually connect
        port=9000,
    )

    # Create a composite emitter that sends metrics to all emitters
    composite_emitter = CompositeMetricEmitter(
        emitters=[console_emitter, prometheus_emitter, questdb_emitter],
        tags={"environment": "example"},
    )

    # Emit some metrics
    print("Emitting metrics with the console emitter:")
    console_emitter.emit("example_metric", 42.0, {"tag1": "value1"})
    console_emitter.emit("example_counter", 1.0, {"tag2": "value2"}, pd.Timestamp("2023-01-01").to_numpy())

    # Emit metrics with the composite emitter
    print("\nEmitting metrics with the composite emitter:")
    composite_emitter.emit("composite_metric", 100.0)
    composite_emitter.emit("composite_counter", 5.0, {"counter": "true"}, pd.Timestamp("2023-01-01").to_numpy())

    # Emit metrics with different types to Prometheus
    print("\nEmitting metrics with different types to Prometheus:")
    prometheus_emitter.emit("gauge_metric", 42.0, {"type": "gauge"}, metric_type="gauge")
    prometheus_emitter.emit("counter_metric", 1.0, {"type": "counter"}, metric_type="counter")
    prometheus_emitter.emit("summary_metric", 0.5, {"type": "summary"}, metric_type="summary")

    # Emit metrics with timestamps to QuestDB
    print("\nEmitting metrics with timestamps to QuestDB:")
    current_time = pd.Timestamp(int(time.time() * 1e9)).to_numpy()
    questdb_emitter.emit("questdb_metric", 42.0, {"database": "questdb"}, current_time)


if __name__ == "__main__":
    main()
