"""
CSV Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to a CSV file.
"""

import csv
import datetime
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from qubx import logger
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer
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
        tags: dict[str, Any] | None = None,
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
        self._record_fields: dict[str, list[str]] = {}
        self._warned_undeclared: set[str] = set()
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

    def emit_signals(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        signals: list[Signal],
        account: IAccountViewer,
        target_positions: list[TargetPosition] | None = None,
    ) -> None:
        """
        Emit signals to CSV file.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information
            target_positions: Optional list of target positions generated from the signals
        """
        if not signals:
            return

        target_positions_map: dict[Instrument, TargetPosition] = {}
        if target_positions:
            for target in target_positions:
                target_positions_map[target.instrument] = target

        try:
            # Create a signals-specific CSV file
            signals_file_path = self._file_path.parent / f"signals_{self._file_path.stem}.csv"

            # Check if file exists, if not create with headers
            if not signals_file_path.exists():
                with open(signals_file_path, "w") as f:
                    f.write(
                        "timestamp,symbol,exchange,signal,price,stop,take,reference_price,"
                        "target_position_size,group,comment,is_service\n"
                    )

            # Write each signal to the CSV file
            for signal in signals:
                signal_time = str(signal.time) if hasattr(signal.time, "__str__") else str(time)
                price = signal.price if signal.price is not None else ""
                stop = signal.stop if signal.stop is not None else ""
                take = signal.take if signal.take is not None else ""
                ref_price = signal.reference_price if signal.reference_price is not None else ""
                target = target_positions_map.get(signal.instrument)
                target_size = target.target_position_size if target is not None else ""

                with open(signals_file_path, "a") as f:
                    f.write(
                        f"{signal_time},{signal.instrument.symbol},{signal.instrument.exchange},"
                        f"{signal.signal},{price},{stop},{take},{ref_price},{target_size},"
                        f"{signal.group},{signal.comment},{signal.is_service}\n"
                    )

        except Exception as e:
            logger.error(f"[CSVMetricEmitter] Failed to emit signals: {e}")

    def emit_deals(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        instrument: Instrument,
        deals: list[Deal],
        account: IAccountViewer,
    ) -> None:
        """
        Emit deals to CSV file.

        Args:
            time: Timestamp when the deals were generated
            instrument: Instrument the deals belong to
            deals: List of deals to emit
            account: Account viewer to get account information
        """
        if not deals:
            return

        try:
            deals_file_path = self._file_path.parent / f"deals_{self._file_path.stem}.csv"

            if not deals_file_path.exists():
                with open(deals_file_path, "w") as f:
                    f.write(
                        "timestamp,symbol,exchange,amount,price,aggressive,fee_amount,fee_currency,deal_id,order_id\n"
                    )

            with open(deals_file_path, "a") as f:
                for deal in deals:
                    deal_time = str(deal.time) if hasattr(deal.time, "__str__") else str(time)
                    fee_amount = deal.fee_amount if deal.fee_amount is not None else ""
                    fee_currency = deal.fee_currency if deal.fee_currency is not None else ""
                    f.write(
                        f"{deal_time},{instrument.symbol},{instrument.exchange},"
                        f"{deal.amount},{deal.price},{deal.aggressive},"
                        f"{fee_amount},{fee_currency},{deal.trade_id},{deal.order_id}\n"
                    )

        except Exception as e:
            logger.error(f"[CSVMetricEmitter] Failed to emit deals: {e}")

    def _record_path(self, table: str) -> Path:
        return self._file_path.parent / f"{table}.csv"

    def ensure_table(
        self,
        table: str,
        columns: dict[str, str],
        symbol_columns: Sequence[str] = (),
        dedup_keys: Sequence[str] | None = None,
        partition_by: str = "DAY",
    ) -> None:
        """
        Declare a strategy-owned table as one CSV file beside the metrics file.
        """
        try:
            if isinstance(symbol_columns, str):
                raise ValueError(f"symbol_columns must be a sequence of names, not a bare string: {symbol_columns!r}")

            fields = ["timestamp", *self._default_tags]
            for name in ("run_id", "is_live", *symbol_columns, *columns):
                if name not in fields:
                    fields.append(name)

            self._record_fields[table] = fields

            path = self._record_path(table)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()
        except Exception as e:
            logger.error(f"[CSVMetricEmitter] Failed to ensure table '{table}': {e}")

    def emit_record(
        self, table: str, record: dict[str, Any], symbol_columns: Sequence[str] = (), timestamp: dt_64 | None = None
    ) -> None:
        """
        Append one row to a strategy-owned table, mirroring the QuestDB emitter's scope tags
        """
        try:
            if isinstance(symbol_columns, str):
                raise ValueError(f"symbol_columns must be a sequence of names, not a bare string: {symbol_columns!r}")
            if self._context is not None and timestamp is None:
                timestamp = self._context.time()
            row = {k: v for k, v in record.items() if v is not None}
            row.pop("timestamp", None)
            row.update(self._default_tags)
            if self._context is not None:
                row["is_live"] = not self._context.is_simulation
            row["timestamp"] = str(timestamp) if timestamp is not None else str(time_now())

            path = self._record_path(table)
            fields = self._record_fields.get(table)
            if fields is None:
                # - undeclared table: take the schema from this first row
                fields = ["timestamp", *(k for k in row if k != "timestamp")]
                self._record_fields[table] = fields
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()
            elif (unknown := set(row) - set(fields)) and table not in self._warned_undeclared:
                self._warned_undeclared.add(table)
                logger.warning(f"[CSVMetricEmitter] '{table}': dropping undeclared columns {sorted(unknown)}")

            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore").writerow(row)

        except Exception as e:
            logger.error(f"[CSVMetricEmitter] Failed to emit record to '{table}': {e}")
