"""
CSV Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to a CSV file.
"""

import os
from pathlib import Path

from qubx import logger
from qubx.core.basics import dt_64
from qubx.emitters.base import BaseMetricEmitter
from qubx.utils.ntp import time_now


class CSVMetricEmitter(BaseMetricEmitter):
    """
    Emits metrics to a CSV file.

    This emitter writes metrics to a CSV file with timestamp, name, and value columns.
    The file is created if it doesn't exist and metrics are appended to it.
    """

    def __init__(
        self,
        file_path: str | None = None,
        stats_to_emit: list[str] | None = None,
        stats_interval: str = "1m",
        tags: dict[str, str] | None = None,
    ):
        """
        Initialize the CSV Metric Emitter.

        Args:
            file_path: Path to the CSV file. If None, creates 'metrics.csv' in current directory
            stats_to_emit: Optional list of specific stats to emit
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
        """
        super().__init__(stats_to_emit, stats_interval, tags)

        # Set default file path if none provided
        if file_path is None:
            file_path = os.path.join(os.getcwd(), "metrics.csv")

        self._file_path = Path(file_path)
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Initialize the CSV file with headers if it doesn't exist."""
        try:
            # Create directory if it doesn't exist
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create file with headers
            with open(self._file_path, "w") as f:
                f.write("timestamp,name,value,tags\n")
            logger.info(f"[CSVMetricEmitter] Created new metrics file at {self._file_path}")
        except Exception as e:
            logger.error(f"[CSVMetricEmitter] Failed to initialize metrics file: {e}")

    def _emit_impl(self, name: str, value: float, tags: dict[str, str], timestamp: dt_64 | None = None) -> None:
        """
        Implementation of emit for CSV file.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric
        """
        try:
            # Use NTP-synchronized time if no timestamp provided
            current_timestamp = timestamp if timestamp is not None else time_now()

            # Convert tags to string representation
            tags_str = ";".join(f"{k}={v}" for k, v in sorted(tags.items()))

            # Write the metric to the CSV file
            with open(self._file_path, "a") as f:
                f.write(f"{str(current_timestamp)},{name},{value},{tags_str}\n")
        except Exception as e:
            logger.error(f"[CSVMetricEmitter] Failed to emit metric {name}: {e}")
