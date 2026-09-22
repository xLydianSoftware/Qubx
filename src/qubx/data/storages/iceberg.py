"""Qubx `IStorage`/`IReader` over the datavault Iceberg lake (R2 Data Catalog).

Moved here from datavault (`dvault.storage.iceberg.qubx_storage`) so consumers need
Qubx alone; install with `qubx[iceberg]`. Read-only: datavault writes the lake.

A lake table is `<venue>_<market>.<name>`, where the name is either a canonical
raw data type or `<kernel>_<interval>`; provider and lineage are table
properties (`dvault.*`), not name segments. Namespace and properties together
decode into a Qubx `DataType` plus the exchange/market pair Qubx addresses
readers by. "All data columns" is every column except the ingest provenance
ones, with `ts_event` presented as `timestamp`.
"""

import datetime as dt
import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.expressions import AlwaysTrue
from pyiceberg.io.pyarrow import ArrowScan, _read_all_delete_files
from pyiceberg.table import DataScan, FileScanTask, Table

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.registry import storage
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.data.storages.utils import calculate_time_windows_for_chunking
from qubx.utils.time import handle_start_stop

# - column names the lake speaks in (datavault storage/iceberg/tables.py writes them)
TIME_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
RAW_TIME_COLUMN = "ts_event"
RAW_RECV_COLUMN = "ts_recv"
PROVENANCE_COLUMNS = frozenset({RAW_RECV_COLUMN, "src_month"})

# - namespaces the lake owns nothing in: scratch runs and operational tables
SKIP_ROOTS = frozenset({"scratch", "ops"})

# - a derived table's `<stem>_<interval>` name
FEATURE_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<interval>\d+[a-zA-Z]+)$")

# - pyarrow's S3 client aborts a request stalled for 3 s by default; R2 stalls that long often enough
_S3_TIMEOUTS = {"s3.connect-timeout": "10", "s3.request-timeout": "60"}

_INTERVAL_SQL = {"1h": "INTERVAL 1 HOUR", "1d": "INTERVAL 1 DAY"}
_OP_SQL = {
    "sum": 'sum("{c}")',
    "avg": 'avg("{c}")',
    "min": 'min("{c}")',
    "max": 'max("{c}")',
    "first": 'arg_min("{c}", timestamp)',
    "last": 'arg_max("{c}", timestamp)',
}


def is_lake_namespace(namespace: str, *, prefix: str = "") -> bool:
    """With a `prefix` (a scratch instance), exactly the namespaces carrying it; without one,
    everything but the reserved roots. The roots stay reserved under a prefix too."""
    if prefix and not namespace.startswith(prefix):
        return False
    return namespace[len(prefix) :].split("_", 1)[0] not in SKIP_ROOTS


@dataclass(frozen=True)
class IcebergConfig:
    uri: str
    warehouse: str
    token: str


def build_catalog(cfg: IcebergConfig, name: str = "r2") -> Catalog:
    """R2 vends data-file credentials, so the bearer token is the only credential."""
    return RestCatalog(name, uri=cfg.uri, warehouse=cfg.warehouse, token=cfg.token, **_S3_TIMEOUTS)


def aggregate(data: pa.Table, aggs: dict[str, str], interval: str) -> pa.Table:
    """Resample (timestamp, symbol, ...) rows to `interval` with one aggregation per column —
    the same SQL datavault's rollup builder writes the `_1h`/`_1d` tables with."""
    selects = ", ".join(f'{_OP_SQL[op].format(c=c)} AS "{c}"' for c, op in aggs.items())
    sql = (
        f"SELECT time_bucket({_INTERVAL_SQL[interval]}, timestamp) AS timestamp, symbol, {selects} "
        f"FROM data GROUP BY 1, 2 ORDER BY symbol, timestamp"
    )
    con = duckdb.connect()
    try:
        con.register("data", data)
        rel = con.sql(sql)
        # - duckdb >= 1.5 renames fetch_arrow_table() to to_arrow_table()
        return rel.to_arrow_table() if hasattr(rel, "to_arrow_table") else rel.fetch_arrow_table()
    finally:
        con.close()


def all_tables(catalog: Catalog, namespace: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    children = catalog.list_namespaces(namespace) if namespace else catalog.list_namespaces()
    for child in children:
        found.extend(tuple(t) for t in catalog.list_tables(child))
        found.extend(all_tables(catalog, child))
    return sorted(set(found))


# Rollup intervals Task 5 materializes; a coarser request without a sibling
# rollup table is resampled to one of these in DuckDB and nothing else.
RESAMPLE_INTERVALS = ("1h", "1d")
OHLC_AGGS = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
DISCOVERY_ROLLUP = "1d"
DISCOVERY_PARTITIONS = 7
# A day in the `_1d` rollup stands for the whole day of the minute table.
DAY_END = dt.timedelta(hours=23, minutes=59)

VENUE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("binance", "perp"): ("BINANCE.UM", "SWAP"),
    ("hyperliquid", "perp"): ("HYPERLIQUID.F", "SWAP"),
    ("okx", "perp"): ("OKX.F", "SWAP"),
    ("bybit", "perp"): ("BYBIT.F", "SWAP"),
    ("gateio", "perp"): ("GATEIO.F", "SWAP"),
    ("kraken", "perp"): ("KRAKEN.F", "SWAP"),
    ("deribit", "options"): ("DERIBIT", "OPTION"),
    ("deribit", "perp"): ("DERIBIT", "SWAP"),
    ("lighter", "perp"): ("LIGHTER", "SWAP"),
    # - venue-less provider data: the pair Qubx's QuestDB storage decodes `coingecko.fundamental` to
    ("global", "crypto"): ("COINGECKO", "FUNDAMENTAL"),
}

_FEATURE_DTYPES = {"candles": DataType.OHLC, "quotes": DataType.QUOTE}
_RAW_DTYPES = {"trades": DataType.TRADE, "quotes": DataType.QUOTE}
_CANONICAL_TF = {60: "1m", 3600: "1h", 86400: "1d"}


def _seconds(timeframe: str) -> float:
    return pd.Timedelta(timeframe).total_seconds()


def _canonical_timeframe(value: str) -> str:
    try:
        return _CANONICAL_TF.get(int(_seconds(value)), value.lower())
    except (ValueError, TypeError):
        return value.lower()


def _known_dtype(name: str) -> DataType | None:
    resolved, _ = DataType.from_str(name)
    return None if resolved == DataType.NONE else resolved


@dataclass(frozen=True)
class LakeTable:
    identifier: tuple[str, ...]
    kind: str
    provider: str
    dtype: DataType
    alias: str | None
    timeframe: str | None
    aggs: dict[str, str]
    rollups: dict[str, tuple[str, ...]]

    @property
    def time_column(self) -> str:
        return RAW_TIME_COLUMN if self.kind == "raw" else TIME_COLUMN

    @property
    def name_key(self) -> str:
        """The request name this table answers to: the kernel for a feature
        table, the data type for a raw one."""
        last = self.identifier[-1]
        if self.timeframe and last.endswith(f"_{self.timeframe}"):
            return last[: -len(self.timeframe) - 1]
        return last

    @property
    def request(self) -> str:
        base = self.alias or str(self.dtype)
        return f"{base}({self.timeframe})" if self.timeframe else base


def decode_table(identifier: tuple[str, ...], properties: dict[str, str], *, prefix: str = "") -> LakeTable | None:
    if len(identifier) != 2 or not is_lake_namespace(identifier[0], prefix=prefix):
        return None
    kind = properties.get("dvault.kind")
    if not kind:
        # - every lake table carries dvault.kind; one without it was not written by the lake
        logger.warning(f"[iceberg] {'.'.join(identifier)} carries no dvault.kind property; skipped")
        return None
    name = identifier[1]
    provider = properties.get("dvault.provider", "")
    aggs = {k.removeprefix("dvault.agg."): v for k, v in properties.items() if k.startswith("dvault.agg.")}

    if kind == "raw":
        dtype = _RAW_DTYPES.get(name, DataType.RECORD)
        alias = name if dtype == DataType.RECORD else None
        return LakeTable(tuple(identifier), kind, provider, dtype, alias, None, aggs, {})

    kernel, interval = properties.get("dvault.kernel"), properties.get("dvault.interval")
    if not kernel or not interval:
        parsed = FEATURE_NAME_RE.match(name)
        kernel = parsed["stem"] if parsed else name
        interval = parsed["interval"] if parsed else None
    dtype = _FEATURE_DTYPES.get(kernel) or _known_dtype(kernel) or DataType.RECORD
    alias = kernel if dtype == DataType.RECORD else None
    return LakeTable(tuple(identifier), kind, provider, dtype, alias, interval, aggs, {})


def _record_batch(data: pa.Table) -> pa.RecordBatch:
    batches = data.combine_chunks().to_batches()
    if len(batches) > 1:
        # A Qubx RawData holds exactly one RecordBatch, and only >2GiB of a
        # single variable-width column can force Arrow to keep several chunks.
        raise ValueError(f"{data.num_rows} rows do not fit one Arrow record batch: read a narrower range")
    if not batches:
        return pa.RecordBatch.from_pylist([], schema=data.schema)
    return batches[0]


def _raw_data(data_id: str, dtype: DataType | str, data: pa.Table) -> RawData:
    return RawData.from_record_batch(data_id, dtype, _record_batch(data))


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).isoformat()


def _bound_to_datetime64(raw: bytes) -> np.datetime64:
    """Iceberg column bounds are the type's single-value serialization: a
    timestamp is int64 microseconds, little-endian."""
    return np.datetime64(int.from_bytes(raw, "little", signed=True), "us")


def _in_list(values: list[str]) -> str:
    return f"{SYMBOL_COLUMN} in ({', '.join(repr(str(v)) for v in values)})"


def _shape(data: pa.Table, time_column: str) -> pa.Table:
    """The reader's column layout: the time column presented as `timestamp`,
    then `symbol`, then everything else in schema order."""
    if time_column != TIME_COLUMN:
        data = data.rename_columns([TIME_COLUMN if n == time_column else n for n in data.column_names])
    rest = [n for n in data.column_names if n not in (TIME_COLUMN, SYMBOL_COLUMN)]
    return data.select([TIME_COLUMN, SYMBOL_COLUMN, *rest])


def _containers(data: pa.Table, dtype: DataType | str, data_id: str | list[str]) -> Transformable:
    data = data.sort_by([(SYMBOL_COLUMN, "ascending"), (TIME_COLUMN, "ascending")])
    empty = data.drop_columns([SYMBOL_COLUMN]).slice(0, 0)
    per_symbol, offset = {}, 0
    for group in pc.value_counts(data.column(SYMBOL_COLUMN)).to_pylist():
        rows = group["counts"]
        per_symbol[group["values"]] = data.slice(offset, rows).drop_columns([SYMBOL_COLUMN])
        offset += rows

    if isinstance(data_id, str):
        return _raw_data(data_id, dtype, per_symbol.get(data_id, empty))
    return RawMultiData([_raw_data(s, dtype, part) for s, part in per_symbol.items()])


def _task_order(task: FileScanTask, field_id: int) -> tuple[int, int]:
    """Files inside a symbol partition are time-contiguous, so a file's lower
    bound on the time column orders the stream; a file without one goes last."""
    raw = (task.file.lower_bounds or {}).get(field_id)
    return (1, 0) if raw is None else (0, int.from_bytes(raw, "little", signed=True))


def _task_batches(scan: DataScan, tasks: list[FileScanTask]) -> Iterator[pa.RecordBatch]:
    arrow = ArrowScan(scan.table_metadata, scan.io, scan.projection(), scan.row_filter, scan.case_sensitive)
    for task in tasks:
        # pyiceberg 0.12's public ArrowScan.to_record_batches maps tasks through
        # an executor that materializes a task's batches as a list before it
        # yields; this private generator is the only batch-lazy entry point.
        deletes = _read_all_delete_files(scan.io, [task])
        yield from arrow._record_batches_from_scan_tasks_and_deletes([task], deletes)


def _row_chunks(batches: Iterator[pa.RecordBatch], chunksize: int) -> Iterator[pa.Table]:
    """Regroup a batch stream into tables of exactly `chunksize` rows (the last
    one shorter), holding at most one chunk plus one batch."""
    pending: list[pa.Table] = []
    rows = 0
    for batch in batches:
        table = pa.Table.from_batches([batch])
        while rows + table.num_rows >= chunksize:
            pending.append(table.slice(0, chunksize - rows))
            # different files can differ in string width, exactly as `to_table`
            # allows, so the chunk is promoted rather than schema-checked
            yield pa.concat_tables(pending, promote_options="permissive")
            table = table.slice(chunksize - rows)
            pending, rows = [], 0
        if table.num_rows:
            pending.append(table)
            rows += table.num_rows
    if rows:
        yield pa.concat_tables(pending, promote_options="permissive")


class IcebergLakeReader(IReader):
    """Reads one (exchange, market) slice of the lake."""

    def __init__(
        self,
        catalog: Catalog,
        exchange: str,
        market: str,
        tables: list[LakeTable] | Callable[[], list[LakeTable]],
    ) -> None:
        self.exchange = exchange
        self.market = market
        self._catalog = catalog
        self._source = tables
        self._resolved: list[LakeTable] | None = None
        self._index: dict[str, list[LakeTable]] = {}
        self._symbols: dict[tuple[str, ...], list[str]] = {}
        self._days: dict[tuple[str, ...], dict[str, tuple[np.datetime64, np.datetime64]]] = {}

    def _discover(self) -> None:
        """Layout discovery costs a namespace walk and one `load_table` per
        table, so it waits for the first request instead of running when a
        reader is handed out."""
        if self._resolved is not None:
            return
        resolved = self._source() if callable(self._source) else list(self._source)
        index: dict[str, list[LakeTable]] = {}
        for table in resolved:
            keys = {table.name_key}
            if table.dtype != DataType.RECORD:
                keys.add(str(table.dtype))
            for key in keys:
                index.setdefault(key, []).append(table)
        # `_resolved` is what says "discovery is done", so it is published last:
        # a second thread arriving mid-discovery must not see an empty index.
        self._index = index
        self._resolved = resolved

    @property
    def tables(self) -> list[LakeTable]:
        self._discover()
        return self._resolved  # type: ignore[return-value]

    @property
    def _lookup(self) -> dict[str, list[LakeTable]]:
        self._discover()
        return self._index

    def _resolve(self, dtype: DataType | str) -> tuple[LakeTable, str | None]:
        """Map a request to the table that answers it and the timeframe it must
        be resampled to (None when the table is already at the right one)."""
        request = str(dtype)
        name, _, params = request.partition("(")
        name = name.strip().lower()
        timeframe = _canonical_timeframe(params.rstrip(")").strip()) if params else None

        known = _known_dtype(name)
        candidates = self._lookup.get(str(known) if known is not None else name, [])
        if not candidates:
            raise ValueError(f"no lake table for {request!r} in {self.exchange}/{self.market}")

        if timeframe is None:
            native = [c for c in candidates if c.timeframe is None]
            return (native[0] if native else min(candidates, key=lambda c: _seconds(c.timeframe))), None

        exact = [c for c in candidates if c.timeframe == timeframe]
        if exact:
            return exact[0], None
        finer = [c for c in candidates if c.timeframe and _seconds(c.timeframe) < _seconds(timeframe)]
        if not finer:
            raise ValueError(f"no lake table at or below {timeframe} for {request!r}")
        table = max(finer, key=lambda c: _seconds(c.timeframe))
        if timeframe not in table.rollups and timeframe not in RESAMPLE_INTERVALS:
            raise ValueError(f"{request!r}: no {timeframe} rollup, and only {RESAMPLE_INTERVALS} can be resampled")
        return table, timeframe

    def _aggs_for(self, table: LakeTable, columns: list[str]) -> dict[str, str]:
        fixed = OHLC_AGGS if table.dtype == DataType.OHLC else {}
        aggs = {c: fixed.get(c) or table.aggs.get(c) for c in columns}
        missing = sorted(c for c, op in aggs.items() if not op)
        if missing:
            raise ValueError(f"{'.'.join(table.identifier)}: no aggregation known for {missing}")
        return aggs

    def _plan(
        self, table: LakeTable, symbols: list[str] | None, start, stop, columns: list[str] | None = None
    ) -> tuple[Table, DataScan]:
        source = self._catalog.load_table(table.identifier)
        time_column = table.time_column
        names = [n for n in source.schema().column_names if n not in PROVENANCE_COLUMNS]
        if columns is None:
            projection = tuple(names)
        else:
            missing = sorted(set(columns) - set(names))
            if missing:
                raise ValueError(f"{'.'.join(table.identifier)} has no column(s) {missing}")
            keep = {time_column, SYMBOL_COLUMN, *columns}
            projection = tuple(n for n in names if n in keep)

        predicates = []
        if start is not None:
            predicates.append(f"{time_column} >= '{_iso(start)}'")
        if stop is not None:
            predicates.append(f"{time_column} < '{_iso(stop)}'")
        if symbols:
            predicates.append(_in_list(symbols))

        scan = source.scan(
            row_filter=" and ".join(predicates) if predicates else AlwaysTrue(),
            selected_fields=projection,
        )
        return source, scan

    def _scan(
        self, table: LakeTable, symbols: list[str] | None, start, stop, columns: list[str] | None = None
    ) -> pa.Table:
        _, scan = self._plan(table, symbols, start, stop, columns)
        return _shape(scan.to_arrow(), table.time_column)

    def _block(
        self,
        table: LakeTable,
        resample: str | None,
        dtype: DataType | str,
        data_id: str | list[str],
        symbols: list[str] | None,
        start,
        stop,
        columns: list[str] | None = None,
    ) -> Transformable:
        source, to_aggregate = table, None
        if resample:
            rollup = table.rollups.get(resample)
            if rollup is not None:
                source = replace(table, identifier=rollup, timeframe=resample)
            else:
                to_aggregate = resample

        data = self._scan(source, symbols, start, stop, columns)
        if to_aggregate:
            columns = [n for n in data.column_names if n not in (TIME_COLUMN, SYMBOL_COLUMN)]
            data = aggregate(data, self._aggs_for(table, columns), to_aggregate)
        return _containers(data, dtype, data_id)

    def _stream(
        self,
        table: LakeTable,
        dtype: DataType | str,
        data_id: str | list[str],
        symbols: list[str] | None,
        start,
        stop,
        chunksize: int,
        columns: list[str] | None = None,
    ) -> Iterator[Transformable]:
        """A table with no native timeframe has no windows to cut, so `chunksize`
        counts rows: plan once, then read the files one at a time, oldest first."""
        source, scan = self._plan(table, symbols, start, stop, columns)
        field_id = source.schema().find_field(table.time_column).field_id
        tasks = sorted(scan.plan_files(), key=partial(_task_order, field_id=field_id))
        batches = _task_batches(scan, tasks)
        for chunk in _row_chunks(batches, chunksize):
            yield _containers(_shape(chunk, table.time_column), dtype, data_id)

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
        chunksize=0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        table, resample = self._resolve(dtype)
        columns = kwargs.get("columns")
        columns = list(columns) if columns is not None else None
        if isinstance(data_id, str):
            symbols = [data_id]
        else:
            symbols = list(data_id) or None

        start_ts, stop_ts = handle_start_stop(start, stop, convert=pd.Timestamp)
        if chunksize <= 0:
            return self._block(table, resample, dtype, data_id, symbols, start_ts, stop_ts, columns)

        timeframe = resample or table.timeframe or ""
        if not timeframe:
            return self._stream(table, dtype, data_id, symbols, start_ts, stop_ts, chunksize, columns)
        # An open-ended range has no windows to cut: read it as one block.
        windows = calculate_time_windows_for_chunking(start_ts, stop_ts, timeframe, chunksize) or [(start_ts, stop_ts)]
        return (self._block(table, resample, dtype, data_id, symbols, w0, w1, columns) for w0, w1 in windows)

    def _discovery_table(self, table: LakeTable) -> tuple[str, ...] | None:
        """The `_1d` rollup a feature family is discovered through: its symbols
        and their spans, one small file per year, instead of the minute table's
        whole history. A `_1d` table is its own; a raw table has neither."""
        rollup = table.rollups.get(DISCOVERY_ROLLUP)
        if rollup is not None:
            return rollup
        return table.identifier if table.timeframe == DISCOVERY_ROLLUP else None

    def _day_index(self, identifier: tuple[str, ...]) -> dict[str, tuple[np.datetime64, np.datetime64]]:
        """First and last day of every symbol in a `_1d` table, cached for the
        life of the reader: one projected scan answers every later call."""
        cached = self._days.get(identifier)
        if cached is not None:
            return cached
        source = self._catalog.load_table(identifier)
        data = source.scan(selected_fields=(SYMBOL_COLUMN, TIME_COLUMN)).to_arrow()
        spans = data.group_by(SYMBOL_COLUMN).aggregate([(TIME_COLUMN, "min"), (TIME_COLUMN, "max")])
        index = {
            row[SYMBOL_COLUMN]: (
                np.datetime64(row[f"{TIME_COLUMN}_min"]),
                np.datetime64(row[f"{TIME_COLUMN}_max"] + DAY_END),
            )
            for row in spans.to_pylist()
        }
        self._days[identifier] = index
        return index

    def _symbols_of(self, table: LakeTable) -> list[str]:
        discovery = self._discovery_table(table)
        if discovery is not None:
            return sorted(self._day_index(discovery))
        cached = self._symbols.get(table.identifier)
        if cached is not None:
            return cached
        source = self._catalog.load_table(table.identifier)
        partitions = [f.name for f in source.spec().fields]
        values = source.inspect.partitions().column("partition").to_pylist()
        if SYMBOL_COLUMN in partitions:
            found = sorted({v[SYMBOL_COLUMN] for v in values if v.get(SYMBOL_COLUMN)})
        else:
            found = sorted(self._scan_symbols(source, table, partitions, values))
        self._symbols[table.identifier] = found
        return found

    def _scan_symbols(self, source: Table, table: LakeTable, partitions: list[str], values: list[dict]) -> list[str]:
        """Fallback for a family with no `_1d` rollup. Feature tables partition
        by time, so symbols only exist in the data: read the column out of the
        most recent partitions rather than all of them."""
        days = sorted(v[partitions[0]] for v in values) if partitions else []
        days = [d for d in days if isinstance(d, dt.date)]
        row_filter = AlwaysTrue()
        if days:
            since = dt.datetime.combine(days[-DISCOVERY_PARTITIONS:][0], dt.time())
            row_filter = f"{table.time_column} >= '{since.isoformat()}'"
        data = source.scan(row_filter=row_filter, selected_fields=(SYMBOL_COLUMN,)).to_arrow()
        return pc.unique(data.column(SYMBOL_COLUMN)).to_pylist()

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        if str(dtype) == str(DataType.ALL):
            found: set[str] = set()
            for table in self.tables:
                found.update(self._symbols_of(table))
            return sorted(found)
        table, _ = self._resolve(dtype)
        return list(self._symbols_of(table))

    def get_data_types(self, data_id: str) -> list[DataType]:
        return [t.request for t in self.tables if data_id in self._symbols_of(t)]  # type: ignore[misc]

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        table, _ = self._resolve(dtype)
        discovery = self._discovery_table(table)
        if discovery is not None:
            span = self._day_index(discovery).get(data_id)
            if span is None:
                raise ValueError(f"{'.'.join(discovery)} has no data for {data_id}")
            return span

        source = self._catalog.load_table(table.identifier)
        if SYMBOL_COLUMN in [f.name for f in source.spec().fields]:
            return self._range_from_files(source, table, data_id)
        return self._range_from_scan(source, table, data_id)

    @staticmethod
    def _range_from_scan(source: Table, table: LakeTable, data_id: str) -> tuple[np.datetime64, np.datetime64]:
        """Fallback for a family with no `_1d` rollup: only the data itself says
        how far a symbol runs."""
        data = source.scan(
            row_filter=f"{SYMBOL_COLUMN} == '{data_id}'", selected_fields=(table.time_column,)
        ).to_arrow()
        if data.num_rows == 0:
            raise ValueError(f"{'.'.join(table.identifier)} has no data for {data_id}")
        column = data.column(table.time_column)
        return np.datetime64(pc.min(column).as_py()), np.datetime64(pc.max(column).as_py())

    @staticmethod
    def _range_from_files(source: Table, table: LakeTable, data_id: str) -> tuple[np.datetime64, np.datetime64]:
        """Symbol-partitioned tables answer from file statistics — no data read."""
        field_id = source.schema().find_field(table.time_column).field_id
        lows, highs = [], []
        for row in source.inspect.files().to_pylist():
            if row["partition"].get(SYMBOL_COLUMN) != data_id:
                continue
            lows.append(dict(row["lower_bounds"])[field_id])
            highs.append(dict(row["upper_bounds"])[field_id])
        if not lows:
            raise ValueError(f"{'.'.join(table.identifier)} has no files for {data_id}")
        return min(map(_bound_to_datetime64, lows)), max(map(_bound_to_datetime64, highs))

    def close(self) -> None:
        self._symbols.clear()
        self._days.clear()


@storage("iceberg")
class IcebergLakeStorage(IStorage):
    """Lake-wide entry point: `StorageRegistry.get("iceberg::r2")` or
    `IcebergLakeStorage()` build the R2 catalog from configuration, and
    `from_catalog` wraps any pyiceberg catalog directly."""

    def __init__(
        self,
        account: str = "r2",
        *,
        uri: str | None = None,
        warehouse: str | None = None,
        token: str | None = None,
        namespace_prefix: str = "",
    ) -> None:
        prefix = _resolve_namespace_prefix(account, namespace_prefix)
        catalog = build_catalog(_resolve_config(account, uri, warehouse, token), name=account)
        self._setup(catalog, namespace_prefix=prefix)

    @classmethod
    def from_catalog(cls, catalog: Catalog, *, namespace_prefix: str | None = None) -> "IcebergLakeStorage":
        """`namespace_prefix` omitted (`None`) falls back to
        `DVAULT_ICEBERG_NAMESPACE_PREFIX`, same as `__init__`; there is no
        `account` here, so `~/.qubx/config.json`'s per-account prefix does not
        apply. An explicit value -- including `""` -- always wins."""
        instance = cls.__new__(cls)
        prefix = namespace_prefix if namespace_prefix is not None else _namespace_prefix_from_env()
        instance._setup(catalog, namespace_prefix=prefix)
        return instance

    def _setup(self, catalog: Catalog, *, namespace_prefix: str = "") -> None:
        self._catalog = catalog
        self._namespace_prefix = namespace_prefix
        self._layout: dict[str, dict[str, list[LakeTable]]] | None = None

    def _structure(self) -> dict[str, dict[str, list[LakeTable]]]:
        if self._layout is not None:
            return self._layout

        by_market: dict[tuple[str, str], list[tuple[LakeTable, dict[str, str]]]] = {}
        for identifier in all_tables(self._catalog):
            # every check that can be made on the name alone comes first: a
            # load_table is a REST round trip per table
            if len(identifier) != 2 or not is_lake_namespace(identifier[0], prefix=self._namespace_prefix):
                continue
            venue, _, market = identifier[0].removeprefix(self._namespace_prefix).rpartition("_")
            qubx_names = VENUE_MAP.get((venue, market))
            if qubx_names is None:
                continue
            properties = self._catalog.load_table(identifier).properties
            decoded = decode_table(identifier, properties, prefix=self._namespace_prefix)
            if decoded is None:
                continue
            by_market.setdefault(qubx_names, []).append((decoded, properties))

        layout: dict[str, dict[str, list[LakeTable]]] = {}
        for (exchange, market), entries in sorted(by_market.items()):
            layout.setdefault(exchange, {})[market] = _attach_rollups(entries)
        self._layout = layout
        return layout

    def get_exchanges(self) -> list[str]:
        return sorted(self._structure())

    def get_market_types(self, exchange: str) -> list[str]:
        return sorted(self._structure().get(exchange.upper(), {}))

    def get_reader(self, exchange: str, market: str) -> IcebergLakeReader:
        """Handing out a reader costs nothing: the layout is discovered on the
        reader's first request."""
        name, kind = exchange.upper(), market.upper()
        return IcebergLakeReader(self._catalog, name, kind, partial(self._tables_for, exchange, market))

    def read_venues(
        self,
        exchanges: list[str],
        market: str,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None,
        stop: str | None,
    ) -> pd.DataFrame:
        """One frame across venues, `venue` first, exchanges in the given order.

        Cross-venue tables are reserved and never created, so a multi-venue
        request is served by reading each venue's own table and stacking them.
        """
        frames = []
        for exchange in exchanges:
            data = self[exchange, market].read(data_id, dtype, start, stop)
            frame = data.to_pd(id_in_index=True) if isinstance(data, RawMultiData) else data.to_pd()
            if frame.empty:
                continue
            frame.insert(0, "venue", exchange.upper())
            frames.append(frame)
        return pd.concat(frames) if frames else pd.DataFrame(columns=["venue"])

    def _tables_for(self, exchange: str, market: str) -> list[LakeTable]:
        tables = self._structure().get(exchange.upper(), {}).get(market.upper())
        if not tables:
            raise ValueError(f"no lake tables for exchange {exchange!r} and market type {market!r}")
        return tables

    def close(self) -> None:
        self._layout = None


def _attach_rollups(entries: list[tuple[LakeTable, dict[str, str]]]) -> list[LakeTable]:
    """A rollup is not a table of its own to a reader — it is a coarser
    timeframe of its base, reachable through `LakeTable.rollups`."""
    bases = {t.identifier: t for t, props in entries if not props.get("dvault.rollup_of")}
    rollups: dict[tuple[str, ...], dict[str, tuple[str, ...]]] = {}
    for table, props in entries:
        base = props.get("dvault.rollup_of")
        if not base or not table.timeframe:
            continue
        rollups.setdefault(tuple(base.split(".")), {})[table.timeframe] = table.identifier
    return [replace(t, rollups=rollups.get(i, {})) for i, t in sorted(bases.items())]


def _qubx_account(account: str) -> dict[str, str]:
    path = Path.home() / ".qubx" / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())["iceberg"][account]
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _namespace_prefix_from_env() -> str:
    return os.environ.get("DVAULT_ICEBERG_NAMESPACE_PREFIX") or ""


def _resolve_namespace_prefix(account: str, explicit: str) -> str:
    """Same precedence as `_resolve_config`'s per-key fallback: explicit arg,
    then `~/.qubx/config.json`, then `DVAULT_ICEBERG_NAMESPACE_PREFIX`."""
    configured = _qubx_account(account)
    return explicit or configured.get("namespace_prefix") or _namespace_prefix_from_env()


def _resolve_config(account: str, uri: str | None, warehouse: str | None, token: str | None) -> IcebergConfig:
    configured = _qubx_account(account)
    resolved, missing = {}, []
    for key, explicit in (("uri", uri), ("warehouse", warehouse), ("token", token)):
        value = explicit or configured.get(key) or os.environ.get(f"DVAULT_ICEBERG_{key.upper()}")
        if not value:
            missing.append(key)
        resolved[key] = value
    if missing:
        raise ValueError(
            f"iceberg account {account!r}: missing {', '.join(missing)} — pass it explicitly, "
            f"put it in ~/.qubx/config.json under iceberg.{account}, or set DVAULT_ICEBERG_*"
        )
    return IcebergConfig(**resolved)
