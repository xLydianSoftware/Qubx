"""
QuestDB Metric Emitter.

This module provides an implementation of IMetricEmitter that exports metrics to QuestDB.
"""

import datetime
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import numpy as np
import pandas as pd
from questdb.ingress import Sender

from qubx import logger
from qubx.core.basics import Deal, Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer, IStrategyContext
from qubx.emitters.base import BaseMetricEmitter
from qubx.utils.questdb import QuestDBClient
from qubx.utils.time import to_timedelta


class QuestDBMetricEmitter(BaseMetricEmitter):
    """
    Emits metrics to QuestDB using the QuestDB ingress client.

    This emitter sends metrics to QuestDB with custom timestamps and tags.
    """

    SYMBOL_TAGS = ["symbol", "exchange", "type", "environment", "strategy"]
    RECORD_COLUMN_TYPES = {"DOUBLE", "LONG", "STRING", "BOOLEAN", "TIMESTAMP", "SYMBOL"}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        table_name: str = "qubx.metrics",
        signals_table_name: str = "qubx.signals",
        deals_table_name: str = "qubx.deals",
        stats_to_emit: list[str] | None = None,
        stats_interval: str = "1m",
        flush_interval: str = "5s",
        tags: dict[str, Any] | None = None,
        max_workers: int = 1,
    ):
        """
        Initialize the QuestDB Metric Emitter.

        Args:
            host: QuestDB server host
            port: QuestDB server port
            table_name: Name of the table to store metrics in
            signals_table_name: Name of the table to store signals in
            stats_to_emit: Optional list of specific stats to emit
            stats_interval: Interval for emitting strategy stats (default: "1m")
            tags: Dictionary of default tags/labels to include with all metrics
            max_workers: Maximum number of worker threads for QuestDB operations
        """
        # Initialize default tags with strategy name
        default_tags = tags or {}

        super().__init__(stats_to_emit, stats_interval, default_tags)

        self._host = host
        self._port = port
        self._table_name = table_name
        self._signals_table_name = signals_table_name
        self._deals_table_name = deals_table_name
        self._conn_str = f"http::addr={host}:{port};"
        self._flush_interval = to_timedelta(flush_interval)
        self._sender = self._try_get_sender()
        self._last_flush = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="questdb_emitter")
        self._stopped = False

        # Strategy-owned tables declared via ensure_table: name -> column/symbol sets.
        self._declared_columns: dict[str, set[str]] = {}
        self._declared_symbols: dict[str, set[str]] = {}
        self._warned_undeclared: set[str] = set()

        # Create signals table if it doesn't exist
        self._ensure_signals_table_exists()
        self._ensure_deals_table_exists()

    def notify(self, context: IStrategyContext) -> None:
        super().notify(context)

        if self._last_flush is None:
            self._last_flush = pd.Timestamp.now()
            return

        if pd.Timestamp.now() - self._last_flush >= self._flush_interval:
            if self._sender is not None:
                try:
                    self._executor.submit(self._flush_sender)
                except Exception as e:
                    logger.error(f"[QuestDBMetricEmitter] Failed to queue flush operation: {e}")
            self._last_flush = pd.Timestamp.now()

    def _flush_sender(self) -> None:
        """Flush the sender in a background thread."""
        if self._sender is not None:
            try:
                self._sender.flush()
            except Exception as e:
                logger.error(f"[QuestDBMetricEmitter] Failed to flush metrics: {e}")

    def _log_warning(self, msg: str) -> None:
        try:
            logger.warning(f"[QuestDBMetricEmitter] {msg}")
        except Exception:
            pass

    def __del__(self):
        self.stop()

    def stop(self) -> None:
        """Flush pending metrics and shut down the executor and sender."""
        if self._stopped:
            return
        self._stopped = True

        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            self._log_warning(f"Error during executor shutdown: {e}")

        if self._sender is not None:
            try:
                self._sender.flush()
            except Exception as e:
                self._log_warning(f"Failed to flush on stop: {e}")
            try:
                self._sender.close()
            except Exception as e:
                self._log_warning(f"Failed to close sender: {e}")
            self._sender = None

    def _convert_timestamp(self, timestamp: dt_64 | pd.Timestamp | datetime.datetime) -> datetime.datetime:
        """
        Convert input timestamp (pd.Timestamp, np.datetime64, int/float nanoseconds, or datetime.datetime)
        to a Python datetime.datetime object.
        """
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.to_pydatetime()
        if hasattr(timestamp, "astype"):  # np.datetime64 or anything array-like
            # Convert to nanoseconds since epoch
            ns = cast(np.datetime64, timestamp).astype("datetime64[ns]").item()
            return datetime.datetime.fromtimestamp(ns / 1e9)
        if isinstance(timestamp, datetime.datetime):
            return timestamp
        if isinstance(timestamp, (int, float)):
            # Treat as number of nanoseconds since epoch
            return datetime.datetime.fromtimestamp(float(timestamp) / 1e9)
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")

    def _emit_impl(
        self,
        name: str,
        value: float,
        tags: dict[str, str],
        timestamp: dt_64 | pd.Timestamp | datetime.datetime | None = None,
    ) -> None:
        """
        Implementation of emit for QuestDB.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric (already merged with default tags)
            timestamp: Optional timestamp for the metric
        """
        if self._sender is None:
            return

        try:
            # Submit the metric emission to the thread pool
            self._executor.submit(self._emit_to_questdb, name, value, tags, timestamp)
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to queue metric {name}: {e}")

    def _emit_to_questdb(
        self,
        name: str,
        value: float,
        tags: dict[str, str],
        timestamp: dt_64 | pd.Timestamp | datetime.datetime | None = None,
    ) -> None:
        """
        Send metrics to QuestDB in a background thread.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Dictionary of tags/labels for the metric
            timestamp: Optional timestamp for the metric
        """
        try:
            if self._sender is None:
                return

            # Prepare symbols (tags) and columns (values)
            symbols = self._pop_symbols(tags)
            columns: dict = {"metric_name": name, "value": round(value, 5), **tags}

            # Use the provided timestamp if available, otherwise use current time
            dt_timestamp = self._convert_timestamp(timestamp) if timestamp is not None else datetime.datetime.now()

            # Send the row to QuestDB
            self._sender.row(self._table_name, symbols=symbols, columns=columns, at=dt_timestamp)
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to emit metric {name} to QuestDB: {e}")

    def _try_get_sender(self) -> Sender | None:
        try:
            _sender = Sender.from_conf(self._conn_str)
            _sender.establish()
            logger.info(f"[QuestDBMetricEmitter] Initialized QuestDB at {self._host}:{self._port}")
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to connect to QuestDB: {e}")
            _sender = None
        return _sender

    def _ensure_signals_table_exists(self) -> None:
        """Ensure the signals table exists with the correct schema."""
        try:
            # Use the PostgreSQL interface (port 8812) for DDL operations
            client = QuestDBClient(host=self._host, port=8812)

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS "{self._signals_table_name}" (
                timestamp TIMESTAMP,
                symbol SYMBOL,
                exchange SYMBOL,
                signal DOUBLE,
                price DOUBLE,
                stop DOUBLE,
                take DOUBLE,
                reference_price DOUBLE,
                target_leverage DOUBLE,
                group_name SYMBOL,
                comment STRING,
                is_service BOOLEAN
            ) TIMESTAMP(timestamp) PARTITION BY WEEK;
            """

            client.execute(create_table_sql)
            logger.info(f"[QuestDBMetricEmitter] Ensured signals table '{self._signals_table_name}' exists")
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to create signals table: {e}")

    def _ensure_deals_table_exists(self) -> None:
        """Ensure the signals table exists with the correct schema."""
        try:
            # Use the PostgreSQL interface (port 8812) for DDL operations
            client = QuestDBClient(host=self._host, port=8812)

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS "{self._deals_table_name}" (
                timestamp TIMESTAMP,
                symbol SYMBOL,
                exchange SYMBOL,
                amount DOUBLE,
                price DOUBLE,
                aggressive BOOLEAN,
                fee_amount DOUBLE,
                fee_currency SYMBOL,
                deal_id STRING,
                order_id STRING
            ) TIMESTAMP(timestamp) PARTITION BY WEEK;
            """

            client.execute(create_table_sql)
            logger.info(f"[QuestDBMetricEmitter] Ensured signals table '{self._signals_table_name}' exists")
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to create signals table: {e}")

    def emit_signals(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        signals: list[Signal],
        account: IAccountViewer,
        target_positions: list[TargetPosition] | None = None,
    ) -> None:
        """
        Emit signals to QuestDB.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information
            target_positions: Optional list of target positions generated from the signals
        """
        if not signals or self._sender is None:
            return

        try:
            self._executor.submit(self._emit_signals_to_questdb, time, signals, account, target_positions)
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to queue signals emission: {e}")

    def emit_deals(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        instrument: Instrument,
        deals: list[Deal],
        account: "IAccountViewer",
    ) -> None:
        if not deals or self._sender is None:
            return

        try:
            self._executor.submit(self._emit_deals_to_questdb, time, instrument, deals, account)
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to queue deals emission: {e}")

    def ensure_table(
        self,
        table: str,
        columns: dict[str, str],
        symbol_columns: Sequence[str] = (),
        dedup_keys: Sequence[str] | None = None,
        partition_by: str = "DAY",
    ) -> None:
        """Create a strategy-owned table with explicit types (spec: strategy-tables §2.0)."""
        try:
            decl: dict[str, str] = {"timestamp": "TIMESTAMP"}
            for key in self._default_tags:  # scope columns, injected first
                decl[key] = "SYMBOL" if key in self.SYMBOL_TAGS else "STRING"
            decl.setdefault("run_id", "STRING")
            decl["is_live"] = "BOOLEAN"
            for name in symbol_columns:
                decl.setdefault(name, "SYMBOL")  # never overrides reserved scope columns (e.g. run_id, timestamp)
            for name, typ in columns.items():
                t = typ.upper()
                if t not in self.RECORD_COLUMN_TYPES:
                    raise ValueError(f"unsupported column type {typ!r} for column {name!r}")
                decl.setdefault(name, t)  # symbol_columns take precedence
            cols_sql = ", ".join(f'"{n}" {t}' for n, t in decl.items())
            ddl = (
                f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_sql}) '
                f"TIMESTAMP(timestamp) PARTITION BY {partition_by} WAL"
            )
            if dedup_keys:
                ddl += f" DEDUP UPSERT KEYS({', '.join(dedup_keys)})"
            ddl += ";"
            QuestDBClient(host=self._host, port=8812).execute(ddl)
            self._declared_columns[table] = set(decl)
            self._declared_symbols[table] = {n for n, t in decl.items() if t == "SYMBOL"}
            logger.info(f"[QuestDBMetricEmitter] Ensured table '{table}' exists")
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to ensure table '{table}': {e}")

    def emit_record(
        self,
        table: str,
        record: dict[str, Any],
        symbol_columns: Sequence[str] = (),
        timestamp: dt_64 | None = None,
    ) -> None:
        """Emit one row to a strategy-owned table (spec: strategy-tables §2.1)."""
        if self._sender is None:
            return
        try:
            if self._context is not None and timestamp is None:
                timestamp = self._context.time()
            row = {k: v for k, v in record.items() if v is not None}
            row.update(self._default_tags)  # scope tags overwrite caller keys
            if self._context is not None:
                row["is_live"] = not self._context.is_simulation
            declared = self._declared_columns.get(table)
            if declared is not None:
                unknown = set(row) - declared
                if unknown and table not in self._warned_undeclared:
                    self._warned_undeclared.add(table)
                    logger.warning(f"[QuestDBMetricEmitter] '{table}': undeclared columns {sorted(unknown)}")
            sym_names = self._declared_symbols.get(table)
            if sym_names is None:
                sym_names = set(symbol_columns) | (set(self.SYMBOL_TAGS) & set(row))
            symbols = {k: str(row.pop(k)) for k in list(row) if k in sym_names}
            columns: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, (pd.Timestamp, np.datetime64, datetime.datetime)):
                    columns[k] = self._convert_timestamp(v)
                elif isinstance(v, np.generic):  # numpy scalar (float64/int64/bool_/...) -> native Python
                    columns[k] = v.item()
                else:
                    columns[k] = v
            at = self._convert_timestamp(timestamp) if timestamp is not None else datetime.datetime.now()
            self._executor.submit(self._emit_record_to_questdb, table, symbols, columns, at)
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to queue record for '{table}': {e}")

    def _emit_record_to_questdb(
        self, table: str, symbols: dict[str, str], columns: dict[str, Any], at: datetime.datetime
    ) -> None:
        try:
            if self._sender is not None:
                self._sender.row(table, symbols=symbols, columns=columns, at=at)
        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to emit record to '{table}': {e}")

    def _emit_signals_to_questdb(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        signals: list[Signal],
        account: IAccountViewer,
        target_positions: list[TargetPosition] | None = None,
    ) -> None:
        if self._sender is None:
            return

        try:
            # Get total capital for leverage calculations
            total_capital = account.get_total_capital()

            # Create a mapping of instruments to target positions for easier lookup
            target_positions_map = {}

            if target_positions:
                for target in target_positions:
                    target_positions_map[target.instrument] = target

            for signal in signals:
                # Get target leverage for this instrument if available
                target_leverage = None
                if signal.instrument in target_positions_map:
                    target = target_positions_map[signal.instrument]
                    # Use signal.reference_price for notional value calculation
                    if signal.reference_price is not None and total_capital > 0:
                        notional_value = abs(target.target_position_size * signal.reference_price)
                        target_leverage = (notional_value / total_capital) * 100

                # Use _merge_tags to get properly merged tags
                merged_tags = self._merge_tags({}, signal.instrument)
                symbols = self._pop_symbols(merged_tags)

                columns = {
                    "signal": float(signal.signal),
                    "price": float(signal.price) if signal.price is not None else None,
                    "stop": float(signal.stop) if signal.stop is not None else None,
                    "take": float(signal.take) if signal.take is not None else None,
                    "reference_price": float(signal.reference_price) if signal.reference_price is not None else None,
                    "target_leverage": float(target_leverage) if target_leverage is not None else None,
                    "comment": signal.comment if signal.comment else "",
                    # "options": json.dumps(signal.options) if signal.options else "{}",
                    "is_service": bool(signal.is_service),
                    "group_name": signal.group if signal.group else "",
                    **merged_tags,
                }

                # Convert timestamp - signal.time is always dt_64, no need to check for string
                dt_timestamp = self._convert_timestamp(time)

                # Send the row to QuestDB
                self._sender.row(self._signals_table_name, symbols=symbols, columns=columns, at=dt_timestamp)

        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to emit signals to QuestDB: {e}")

    def _emit_deals_to_questdb(
        self,
        time: dt_64 | pd.Timestamp | datetime.datetime,
        instrument: Instrument,
        deals: list[Deal],
        account: IAccountViewer,
    ) -> None:
        if self._sender is None:
            return

        try:
            for deal in deals:
                # Use _merge_tags to get properly merged tags
                merged_tags = self._merge_tags({}, instrument)
                symbols = self._pop_symbols(merged_tags)

                columns = {
                    "amount": float(deal.amount),
                    "price": float(deal.price),
                    "aggressive": bool(deal.aggressive),
                    "fee_amount": float(deal.fee_amount) if deal.fee_amount is not None else None,
                    "fee_currency": deal.fee_currency if deal.fee_currency is not None else None,
                    "deal_id": deal.trade_id,
                    "order_id": deal.order_id,
                    **merged_tags,
                }

                # Convert timestamp - signal.time is always dt_64, no need to check for string
                dt_timestamp = self._convert_timestamp(time)

                # Send the row to QuestDB
                self._sender.row(self._deals_table_name, symbols=symbols, columns=columns, at=dt_timestamp)

        except Exception as e:
            logger.error(f"[QuestDBMetricEmitter] Failed to emit deals to QuestDB: {e}")

    def _pop_symbols(self, tags: dict[str, str]) -> dict[str, str]:
        symbols = {}
        for symbol_name in self.SYMBOL_TAGS:
            if symbol_name in tags:
                symbols[symbol_name] = tags.pop(symbol_name)
        return symbols
