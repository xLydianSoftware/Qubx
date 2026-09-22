"""
Caching layer for IReader / IStorage.

Provides in-memory caching with optional prefetch to reduce database queries.
Designed as transparent wrappers following the same decorator pattern as
TimeGuardedReader / TimeGuardedStorage.

Architecture:
    ICache               — backend interface for storing/retrieving RawData
    MemoryCache(ICache)  — in-memory dict-based implementation with Arrow concat/slice
    ParquetCache(ICache) — on-disk parquet files plus a JSON range index, survives the process
    CachedReader(IReader)   — wraps IReader, caches read() results
    CachedStorage(IStorage) — wraps IStorage, returns CachedReader from get_reader()

Composition with TimeGuard:
    Strategy sees: TimeGuardedReader → CachedReader → QuestDBReader
    As storages:   TimeGuardedStorage(CachedStorage(QuestDBStorage(...)))

    CachedReader fetches ahead (stop + prefetch_period) and stores in cache.
    TimeGuardedReader on the outside clamps what the strategy can see to sim_time.
    Next sim tick, cache already has the data — zero DB queries.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.utils.time import now_utc, timedelta_to_str, to_timedelta, to_timestamp


class ICache:
    """
    Cache backend interface. Stores RawData keyed by (cache_key, data_id).
    Tracks time ranges per cache_key to know what's been fetched.
    """

    def get(self, cache_key: str, data_id: str) -> RawData | None:
        """
        Get cached RawData for a single symbol, or None on miss.
        """
        ...

    def put(self, cache_key: str, data: RawData, start: str, stop: str) -> None:
        """
        Store or extend cached RawData for a single symbol with time range.
        If data already exists for this (cache_key, data_id), Arrow batches
        are concatenated, deduplicated by time, and sorted.
        """
        ...

    def covers(self, cache_key: str, start: str | None, stop: str | None) -> bool:
        """Check if global cached range fully covers [start, stop)."""
        ...

    def check(self, cache_key: str, ids: list[str], start: str | None, stop: str | None) -> list[str]:
        """
        Return ids that are NOT covered for [start, stop).
        A symbol is uncovered if it doesn't exist in cache or its per-symbol
        range doesn't span the requested window. Empty list = all covered.
        """
        ...

    def get_ranges(self, cache_key: str) -> list[tuple[str, str]]:
        """
        Return cached time ranges for a cache key.
        """
        ...

    def get_stored_ids(self, cache_key: str) -> list[str]:
        """
        Return list of data_ids stored under this cache key.
        """
        ...

    def clear(self, cache_key: str | None = None) -> None:
        """
        Clear specific key or entire cache.
        """
        ...

    def size_bytes(self) -> int:
        """
        Return approximate total cache size in bytes.
        """
        ...

    def close(self) -> None:
        """
        Release resources (file handles, connections).
        For persistent backends, this should flush any buffered writes before releasing.
        """
        ...

    def __enter__(self) -> ICache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MemoryCache(ICache):
    """
    In-memory cache storing RawData per (cache_key, data_id).
    Handles Arrow-level concat when extending ranges, deduplication by time,
    and LRU eviction when max_size_mb is exceeded.

    Tracks time ranges at two levels:
    - Global ranges per cache_key: used for all-symbols requests
    - Per-symbol ranges per (cache_key, data_id): used to detect symbols with
      incomplete coverage (e.g. added via partial hit with a narrower window)
    """

    _data: dict[str, dict[str, RawData]]
    _ranges: dict[str, list[tuple[str, str]]]
    _symbol_ranges: dict[str, dict[str, list[tuple[str, str]]]]
    _access_order: OrderedDict[str, None]
    _max_size_bytes: int

    def __init__(self, max_size_mb: int = 1000) -> None:
        self._data = {}
        self._ranges = {}
        self._symbol_ranges = {}
        self._access_order = OrderedDict()
        self._max_size_bytes = max_size_mb * 1024 * 1024

    def get(self, cache_key: str, data_id: str) -> RawData | None:
        bucket = self._data.get(cache_key)
        if bucket is None:
            return None
        raw = bucket.get(data_id)
        if raw is not None:
            # - touch for LRU
            self._access_order.move_to_end(cache_key, last=True)
        return raw

    def put(self, cache_key: str, data: RawData, start: str, stop: str) -> None:
        if len(data) == 0:
            # - still record the range even for empty data
            self._record_range(cache_key, start, stop)
            self._record_symbol_range(cache_key, data.data_id, start, stop)
            return

        bucket = self._data.setdefault(cache_key, {})
        existing = bucket.get(data.data_id)

        if existing is not None and len(existing) > 0:
            # - merge: concat Arrow batches, deduplicate by time, sort
            merged_batch = _merge_batches(existing._raw, data._raw, existing.index)
            bucket[data.data_id] = RawData.from_record_batch(data.data_id, data.dtype, merged_batch)
        else:
            bucket[data.data_id] = data

        self._record_range(cache_key, start, stop)
        self._record_symbol_range(cache_key, data.data_id, start, stop)
        self._access_order[cache_key] = None
        self._access_order.move_to_end(cache_key, last=True)
        # - skip_key protects the key being actively populated from self-eviction;
        #   if data is larger than max_size_mb we accept going over-limit rather than
        #   thrashing (evict → re-add → evict on every put)
        self._maybe_evict(skip_key=cache_key)

    def covers(self, cache_key: str, start: str | None, stop: str | None) -> bool:
        if start is None and stop is None:
            return cache_key in self._data
        ranges = self._ranges.get(cache_key)
        if not ranges:
            return False
        return self._ranges_cover(ranges, start, stop)

    def check(self, cache_key: str, ids: list[str], start: str | None, stop: str | None) -> list[str]:
        sym_ranges = self._symbol_ranges.get(cache_key, {})
        return [did for did in ids if not self._ranges_cover(sym_ranges.get(did, []), start, stop)]

    def get_ranges(self, cache_key: str) -> list[tuple[str, str]]:
        return list(self._ranges.get(cache_key, []))

    def get_stored_ids(self, cache_key: str) -> list[str]:
        bucket = self._data.get(cache_key)
        return list(bucket.keys()) if bucket else []

    def clear(self, cache_key: str | None = None) -> None:
        if cache_key is None:
            self._data.clear()
            self._ranges.clear()
            self._symbol_ranges.clear()
            self._access_order.clear()
        else:
            self._data.pop(cache_key, None)
            self._ranges.pop(cache_key, None)
            self._symbol_ranges.pop(cache_key, None)
            self._access_order.pop(cache_key, None)

    def size_bytes(self) -> int:
        total = 0
        for bucket in self._data.values():
            for raw in bucket.values():
                total += raw._raw.nbytes
        return total

    def close(self) -> None:
        # - for in-memory cache, just clear everything
        self.clear()

    @staticmethod
    def _ranges_cover(ranges: list[tuple[str, str]], start: str | None, stop: str | None) -> bool:
        merged = _merge_time_ranges(ranges)
        req_start = to_timestamp(start) if start else pd.Timestamp.min
        req_stop = to_timestamp(stop) if stop else pd.Timestamp.max
        for rs, re_ in merged:
            if rs <= req_start and re_ >= req_stop:
                return True
        return False

    def _record_range(self, cache_key: str, start: str, stop: str) -> None:
        if cache_key not in self._ranges:
            self._ranges[cache_key] = []
        self._ranges[cache_key].append((start, stop))
        # - keep merged to avoid unbounded growth
        self._ranges[cache_key] = [(str(s), str(e)) for s, e in _merge_time_ranges(self._ranges[cache_key])]

    def _record_symbol_range(self, cache_key: str, data_id: str, start: str, stop: str) -> None:
        sym_ranges = self._symbol_ranges.setdefault(cache_key, {})
        if data_id not in sym_ranges:
            sym_ranges[data_id] = []
        sym_ranges[data_id].append((start, stop))
        sym_ranges[data_id] = [(str(s), str(e)) for s, e in _merge_time_ranges(sym_ranges[data_id])]

    def _maybe_evict(self, skip_key: str | None = None) -> None:
        while self.size_bytes() > self._max_size_bytes:
            # - find oldest key that is NOT the protected one
            evicted = False
            for oldest_key in self._access_order:
                if oldest_key != skip_key:
                    self._access_order.pop(oldest_key)
                    self._data.pop(oldest_key, None)
                    self._ranges.pop(oldest_key, None)
                    logger.debug(f"Cache evicted key: {oldest_key}")
                    evicted = True
                    break
            if not evicted:
                # - only skip_key remains; accept over-limit rather than self-evicting
                break


# - - - - - - - - - - - - -
# Arrow helpers
# - - - - - - - - - - - - -


def _merge_batches(existing: pa.RecordBatch, incoming: pa.RecordBatch, time_col_idx: int) -> pa.RecordBatch:
    """
    Concatenate two Arrow RecordBatches, deduplicate, sort by time.

    Deduplicates on timestamp + string columns (e.g. symbol, metric, asset),
    keeping the last (incoming) row. This handles:
    - OHLC per-symbol data: dedup by timestamp only (no string cols in per-symbol batch)
    - Fundamental data: dedup by timestamp + metric, preserving distinct metric rows
    - Floating-point drift from QuestDB SUM across different query windows
    """
    tbl = pa.concat_tables(
        [
            pa.Table.from_batches([existing]),
            pa.Table.from_batches([incoming]),
        ],
        promote_options="permissive",
    )
    time_col_name = tbl.schema.field(time_col_idx).name
    tbl = tbl.sort_by(time_col_name)

    merged_schema = tbl.schema
    pdf = tbl.to_pandas()
    dedup_cols = [time_col_name] + [
        f.name for f in existing.schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
    ]
    pdf = pdf.drop_duplicates(subset=dedup_cols, keep="last")

    batch = pa.RecordBatch.from_pandas(pdf, schema=merged_schema, preserve_index=False)
    return batch


def _slice_batch(batch: pa.RecordBatch, time_col_idx: int, start: str | None, stop: str | None) -> pa.RecordBatch:
    """
    Slice Arrow RecordBatch to [start, stop) using the time column.
    """
    if len(batch) == 0:
        return batch

    time_col = batch.column(time_col_idx)

    masks = []
    if start is not None:
        start_scalar = _to_arrow_timestamp(start, time_col.type)
        masks.append(pc.greater_equal(time_col, start_scalar))
    if stop is not None:
        stop_scalar = _to_arrow_timestamp(stop, time_col.type)
        masks.append(pc.less(time_col, stop_scalar))

    if not masks:
        return batch

    combined = masks[0]
    for m in masks[1:]:
        combined = pc.and_(combined, m)

    return batch.filter(combined)


def _to_arrow_timestamp(time_str: str, target_type: pa.DataType) -> pa.Scalar:
    """
    Convert time string to Arrow scalar matching the target column type.
    """
    ts = to_timestamp(time_str)
    if pa.types.is_timestamp(target_type):
        return pa.scalar(ts, type=target_type)
    # - fallback: int64 nanoseconds
    return pa.scalar(int(ts.value), type=target_type)


def _merge_time_ranges(ranges: list[tuple[str, str]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Merge overlapping or adjacent time ranges.

    Returns list of (start, stop) as pd.Timestamp, sorted and merged.
    """
    if not ranges:
        return []

    parsed = []
    for s, e in ranges:
        try:
            parsed.append((to_timestamp(s), to_timestamp(e)))
        except Exception:
            continue

    if not parsed:
        return []

    parsed.sort(key=lambda x: x[0])

    merged = [parsed[0]]
    for s, e in parsed[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return merged


class ParquetCache(ICache):
    """
    On-disk cache backend: Hive-partitioned parquet, ranges in a JSON index.

    Layout under `root`:

        cache_key=<key>/data_id=<symbol>/data.parquet
        _index/<key>.json    — covered time ranges, globally and per symbol, and each symbol's
                               data type

    DuckDB reads the layout directly, with the partition keys as columns:

        SELECT * FROM read_parquet('<root>/**/*.parquet', hive_partitioning = 1)
        WHERE data_id = 'SPY'

    Persists across processes. Range bookkeeping matches MemoryCache: a symbol counts as covered
    only when one merged range spans the whole request.
    """

    _root: Path
    _index: dict[str, dict]
    _dirty: set[str]

    def __init__(self, root: str | Path) -> None:
        self._root = Path(os.path.expanduser(str(root)))
        self._root.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._dirty = set()
        self._empty: dict[str, dict] = {}

    def get(self, cache_key: str, data_id: str) -> RawData | None:
        path = self._file(cache_key, data_id)
        if not path.exists():
            return None
        dtype = self._load_index(cache_key)["symbols"].get(data_id, {}).get("dtype", str(DataType.ALL))
        # - ParquetFile reads the file itself. pq.read_table() runs dataset discovery on the parent
        # - directories and, as they are Hive-partitioned, returns cache_key and data_id as extra
        # - columns that were never in the data.
        return RawData.from_table(data_id, dtype, pq.ParquetFile(path).read())  # type: ignore[arg-type]

    def put(self, cache_key: str, data: RawData, start: str, stop: str) -> None:
        if len(data) == 0:
            self._record_empty(cache_key, data.data_id, start, stop)
            return
        index = self._load_index(cache_key)
        existing = self.get(cache_key, data.data_id)
        batch = data._raw
        if existing is not None and len(existing) > 0:
            batch = _merge_batches(existing._raw, batch, existing.index)
        path = self._file(cache_key, data.data_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_batches([batch]), path)
        entry = index["symbols"].setdefault(data.data_id, {"ranges": [], "dtype": str(data.dtype)})
        entry["dtype"] = str(data.dtype)
        entry["ranges"] = _merged_strings(entry["ranges"] + [(start, stop)])
        index["ranges"] = _merged_strings(index["ranges"] + [(start, stop)])
        self._dirty.add(cache_key)
        self._flush(cache_key)

    def covers(self, cache_key: str, start: str | None, stop: str | None) -> bool:
        index = self._load_index(cache_key)
        seen = self._empty.get(cache_key, {})
        if start is None and stop is None:
            return bool(index["symbols"]) or bool(seen.get("symbols"))
        return _ranges_cover(_merged_strings(index["ranges"] + seen.get("ranges", [])), start, stop)

    def check(self, cache_key: str, ids: list[str], start: str | None, stop: str | None) -> list[str]:
        symbols = self._load_index(cache_key)["symbols"]
        empty = self._empty.get(cache_key, {}).get("symbols", {})
        missing = []
        for i in ids:
            ranges = _merged_strings(symbols.get(i, {}).get("ranges", []) + empty.get(i, []))
            if not _ranges_cover(ranges, start, stop):
                missing.append(i)
        return missing

    def _record_empty(self, cache_key: str, data_id: str, start: str, stop: str) -> None:
        """
        Remember a window the reader answered with no rows, for this process only.

        It is deliberately not written to the index. A window can come back empty because the
        source has nothing there, but also because a reader could not build it, and the two are
        indistinguishable here. Persisting the second kind makes it permanent: the window reads as
        covered from then on and is served empty, without a refetch and without an error. Keeping
        it in memory stops one run asking repeatedly and still lets the next run find out.
        """
        seen = self._empty.setdefault(cache_key, {"ranges": [], "symbols": {}})
        seen["symbols"][data_id] = _merged_strings(seen["symbols"].get(data_id, []) + [(start, stop)])
        seen["ranges"] = _merged_strings(seen["ranges"] + [(start, stop)])

    def get_ranges(self, cache_key: str) -> list[tuple[str, str]]:
        return [(str(s), str(e)) for s, e in self._load_index(cache_key)["ranges"]]

    def get_stored_ids(self, cache_key: str) -> list[str]:
        return list(self._load_index(cache_key)["symbols"].keys())

    def clear(self, cache_key: str | None = None) -> None:
        if cache_key is not None:
            shutil.rmtree(self._dir(cache_key), ignore_errors=True)
            self._index_file(cache_key).unlink(missing_ok=True)
        else:
            for d in self._root.iterdir():
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
        if cache_key is None:
            self._index.clear()
            self._dirty.clear()
            self._empty.clear()
        else:
            self._index.pop(cache_key, None)
            self._dirty.discard(cache_key)
            self._empty.pop(cache_key, None)

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self._root.rglob("*.parquet"))

    def close(self) -> None:
        for key in list(self._dirty):
            self._flush(key)

    @staticmethod
    def _safe_key(value: str) -> str:
        """
        Plain directory name for a cache key: `ohlc(1h)|side=bid` -> `ohlc_1h_side_bid`.

        Letters, digits, `.`, `_` and `-` survive; every other run becomes one underscore. The key
        is ours, built by `_make_cache_key`, so nothing reads it back - it only has to name a
        directory, and brackets and pipes are awkward in a shell and illegal on Windows.
        """
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "_"

    @staticmethod
    def _safe_id(value: str) -> str:
        """
        Hive partition value for a data id, percent-encoded where it has to be.

        Unlike the cache key this must stay reversible: the id is the symbol, and a DuckDB query
        reads it back with `WHERE data_id = '^GSPC'`. Only `=` and `/` are encoded - a second `=`
        in `data_id=ES=F` makes the segment unparseable - and DuckDB decodes `%3D` on the way out.
        """
        return quote(value, safe="._^-")

    def _dir(self, cache_key: str) -> Path:
        return self._root / f"cache_key={self._safe_key(cache_key)}"

    def _file(self, cache_key: str, data_id: str) -> Path:
        return self._dir(cache_key) / f"data_id={self._safe_id(data_id)}" / "data.parquet"

    def _index_file(self, cache_key: str) -> Path:
        return self._root / "_index" / f"{self._safe_key(cache_key)}.json"

    def _load_index(self, cache_key: str) -> dict:
        if cache_key not in self._index:
            try:
                self._index[cache_key] = json.loads(self._index_file(cache_key).read_text())
            except (OSError, json.JSONDecodeError):
                self._index[cache_key] = {"ranges": [], "symbols": {}}
        return self._index[cache_key]

    def _flush(self, cache_key: str) -> None:
        if cache_key not in self._dirty:
            return
        path = self._index_file(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._index[cache_key]))
        self._dirty.discard(cache_key)

    def __repr__(self) -> str:
        return f"ParquetCache({self._root}, {self.size_bytes() // 1024} KB)"


def _ranges_cover(ranges, start: str | None, stop: str | None) -> bool:
    """
    True when one merged range spans the whole request.
    """
    merged = _merge_time_ranges([tuple(r) for r in ranges])
    req_start = to_timestamp(start) if start else pd.Timestamp.min
    req_stop = to_timestamp(stop) if stop else pd.Timestamp.max
    return any(rs <= req_start and re_ >= req_stop for rs, re_ in merged)


def _merged_strings(ranges) -> list[tuple[str, str]]:
    return [(str(s), str(e)) for s, e in _merge_time_ranges([tuple(r) for r in ranges])]


class CachedReader(IReader):
    """
    Wraps an IReader with in-memory caching and optional prefetch.

    On cache miss, fetches data from the inner reader (optionally extending
    stop by prefetch_period) and stores per-symbol RawData in the cache.
    On cache hit, slices cached data to the requested [start, stop) range.

    Metadata methods (get_data_id, get_data_types) are also cached.
    """

    _reader: IReader
    _cache: ICache
    _prefetch_period: pd.Timedelta | None
    _data_id_cache: dict[str, list[str]]
    _data_types_cache: dict[str, list[DataType]]
    _time_range_cache: dict[str, tuple[np.datetime64, np.datetime64]]

    def __init__(
        self,
        reader: IReader,
        cache: ICache | None = None,
        prefetch_period: str | None = None,
    ) -> None:
        self._reader = reader
        self._cache = cache if cache is not None else MemoryCache()
        self._prefetch_period = to_timedelta(prefetch_period) if prefetch_period else None
        self._data_id_cache = {}
        self._data_types_cache = {}
        self._time_range_cache = {}

    @property
    def inner(self) -> IReader:
        """
        Access the wrapped reader.
        """
        return self._reader

    def read(
        self,
        data_id: str | list[str],
        dtype: DataType | str,
        start: str | None = None,
        stop: str | None = None,
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator[Transformable] | Transformable:
        # Normalize start/stop so that start <= stop regardless of caller ordering
        if start is not None and stop is not None and to_timestamp(start) > to_timestamp(stop):
            start, stop = stop, start

        # - detect "all symbols" request (empty collection)
        # - NOTE: do NOT expand to get_data_id() — that returns ALL symbols ever in the
        #   reader (e.g. SELECT DISTINCT asset over full history), but the actual data
        #   returned for a date range is only the subset with data in that range.
        #   Expanding would make _missing_ids() always fail for ranged reads.
        is_all_request = isinstance(data_id, (list, tuple, set)) and not data_id

        cache_kwargs = kwargs.copy()
        if is_all_request:
            cache_kwargs["__all__"] = True

        cache_key = _make_cache_key(dtype, **cache_kwargs)

        if is_all_request:
            if self._cache.covers(cache_key, start, stop):
                stored_ids = self._cache.get_stored_ids(cache_key)
                result = self._build_result(cache_key, stored_ids, False, start, stop)
                return iter([result]) if chunksize > 0 else result
        else:
            ids = data_id if isinstance(data_id, (list, tuple)) else [data_id]
            is_single = isinstance(data_id, str)

            missing = self._cache.check(cache_key, ids, start, stop)

            if not missing:
                result = self._build_result(cache_key, ids, is_single, start, stop)
                return iter([result]) if chunksize > 0 else result

            if len(missing) < len(ids):
                # - partial hit: some symbols missing or lack time coverage
                fetch_stop = self._compute_fetch_stop(stop)
                miss_result = self._reader.read(missing, dtype, start, fetch_stop, **kwargs)
                self._store_result(cache_key, miss_result, start or "", fetch_stop or "")
                result = self._build_result(cache_key, ids, is_single, start, stop)
                return iter([result]) if chunksize > 0 else result

            # - all ids missing — fall through to all-symbols fallback / full miss below

            # - fallback: check all-symbols cache (data stored via read([], ...) uses a different key)
            all_key = _make_cache_key(dtype, __all__=True, **kwargs)
            if all_key != cache_key and self._cache.covers(all_key, start, stop):
                if not self._cache.check(all_key, ids, start, stop):
                    result = self._build_result(all_key, ids, is_single, start, stop)
                    return iter([result]) if chunksize > 0 else result

        # - full miss: fetch from inner reader WITHOUT chunksize to get a full Transformable
        # - this allows the complete result to be stored in cache for subsequent hits;
        #   chunksize is intentionally omitted here — inner readers return Transformable when
        #   chunksize == 0, which is what _store_result() requires
        fetch_stop = self._compute_fetch_stop(stop)

        result = self._reader.read(data_id, dtype, start, fetch_stop, **kwargs)
        self._store_result(cache_key, result, start or "", fetch_stop or "")

        # - return sliced to originally requested [start, stop)
        if is_all_request:
            stored_ids = self._cache.get_stored_ids(cache_key)
            result = self._build_result(cache_key, stored_ids, False, start, stop)
        else:
            ids = data_id if isinstance(data_id, (list, tuple)) else [data_id]
            result = self._build_result(cache_key, ids, isinstance(data_id, str), start, stop)

        return iter([result]) if chunksize > 0 else result

    def get_data_id(self, dtype: DataType | str = DataType.ALL) -> list[str]:
        key = str(dtype)
        if key not in self._data_id_cache:
            self._data_id_cache[key] = self._reader.get_data_id(dtype)
        return self._data_id_cache[key]

    def get_data_types(self, data_id: str) -> list[DataType]:
        if data_id not in self._data_types_cache:
            self._data_types_cache[data_id] = self._reader.get_data_types(data_id)
        return self._data_types_cache[data_id]

    def get_time_range(self, data_id: str, dtype: DataType | str) -> tuple[np.datetime64, np.datetime64]:
        key = f"{data_id}:{dtype}"
        if key not in self._time_range_cache:
            self._time_range_cache[key] = self._reader.get_time_range(data_id, dtype)
        return self._time_range_cache[key]

    def close(self) -> None:
        # - close cache backend first (flush if persistent), then inner reader
        self._cache.close()
        self._reader.close()

    def __repr__(self) -> str:
        pf = f", prefetch={self._prefetch_period}" if self._prefetch_period else ""
        return f"CachedReader({self._reader!r}{pf})"

    # -- internal helpers --

    def _compute_fetch_stop(self, stop: str | None) -> str | None:
        """
        Extend stop by prefetch_period (clamped to now) for inner reader calls.
        Returns stop unchanged when prefetch is disabled.
        """
        if self._prefetch_period is not None and stop is not None:
            prefetched = to_timestamp(stop) + self._prefetch_period
            # - clamp to now_utc() to avoid recording future timestamps in cache ranges;
            #   without this, live mode gets false cache hits for data that doesn't exist yet
            return str(min(prefetched, now_utc()))
        return stop

    def _store_result(self, cache_key: str, result: Transformable, start: str, stop: str) -> None:
        if isinstance(result, RawMultiData):
            for raw in result:
                self._cache.put(cache_key, raw, start, stop)
        elif isinstance(result, RawData):
            self._cache.put(cache_key, result, start, stop)

    def _build_result(
        self,
        cache_key: str,
        ids: list[str],
        is_single: bool,
        start: str | None,
        stop: str | None,
    ) -> Transformable:
        raws = []
        for did in ids:
            cached = self._cache.get(cache_key, did)
            if cached is not None and len(cached) > 0:
                sliced_batch = _slice_batch(cached._raw, cached.index, start, stop)
                if is_single or sliced_batch.num_rows > 0:
                    # - for multi/all requests skip empty slices — matches non-cached reader
                    # - behaviour where SQL WHERE naturally excludes symbols with no data
                    # - in the requested range (e.g. newly-listed coins in a prefetch window)
                    raws.append(RawData.from_record_batch(did, cached.dtype, sliced_batch))
            elif cached is not None:
                # - empty RawData: include for single requests, skip for multi
                if is_single:
                    raws.append(cached)
            else:
                # - no cached data for this id: include placeholder for single requests only
                if is_single:
                    raws.append(
                        RawData.from_record_batch(did, DataType.ALL, pa.RecordBatch.from_pydict({"timestamp": []}))
                    )

        if is_single:
            return (
                raws[0]
                if raws
                else RawData.from_record_batch(ids[0], DataType.ALL, pa.RecordBatch.from_pydict({"timestamp": []}))
            )
        return RawMultiData(raws)


class CachedStorage(IStorage):
    """
    Wraps IStorage, returns CachedReader from get_reader().
    Each (exchange, market) pair gets its own CachedReader with independent cache.

    Cache backend is pluggable via cache_factory — a callable returning ICache.
    Default: MemoryCache(max_size_mb=1000).
    """

    _storage: IStorage
    _readers: dict[str, CachedReader]
    _prefetch_period: str | None
    _cache_factory: Callable[[], ICache]

    def __init__(
        self,
        storage: IStorage,
        prefetch_period: str | None = None,
        cache_factory: Callable[[], ICache] | None = None,
    ) -> None:
        self._storage = storage
        self._readers = {}
        self._prefetch_period = prefetch_period
        self._cache_factory = cache_factory or (lambda: MemoryCache())

    @property
    def inner(self) -> IStorage:
        """
        Access the wrapped storage.
        """
        return self._storage

    def get_exchanges(self) -> list[str]:
        return self._storage.get_exchanges()

    def get_market_types(self, exchange: str) -> list[str]:
        return self._storage.get_market_types(exchange)

    def get_reader(self, exchange: str, market: str) -> IReader:
        key = f"{exchange}:{market}"
        if key not in self._readers:
            inner = self._storage.get_reader(exchange, market)
            cache = self._cache_factory()
            self._readers[key] = CachedReader(inner, cache, self._prefetch_period)
        return self._readers[key]

    def close(self) -> None:
        # - close all cached readers (flushes caches + inner readers)
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def clear_cache(self, exchange: str | None = None, market: str | None = None) -> None:
        """
        Clear cache for specific reader or all readers.
        """
        if exchange is not None and market is not None:
            key = f"{exchange}:{market}"
            reader = self._readers.get(key)
            if reader is not None:
                reader._cache.clear()
        else:
            for reader in self._readers.values():
                reader._cache.clear()

    def __repr__(self) -> str:
        pf = f", prefetch={self._prefetch_period}" if self._prefetch_period else ""
        return f"CachedStorage({self._storage!r}{pf})"


def _canonical_dtype(dtype: DataType | str) -> str:
    """
    One spelling per data type, so `ohlc(1Min)`, `ohlc(1min)`, `ohlc(1m)` and `ohlc(60s)` share one
    cache entry instead of fetching the same bars into four.

    Only a parameter shaped like a timeframe is rewritten. `orderbook(0.01,10)` and a bare `trade`
    are left exactly as they came in.
    """
    text = str(dtype)
    head, sep, tail = text.partition("(")
    if not sep or not tail.endswith(")"):
        return text
    param = tail[:-1].strip()
    if not re.fullmatch(r"\d+\s*[A-Za-z]+", param):
        return text
    try:
        return f"{head}({timedelta_to_str(to_timedelta(param))})"
    except (ValueError, TypeError):
        return text


def _make_cache_key(dtype: DataType | str, **kwargs) -> str:
    """
    Generate a time-stripped cache key from dtype and extra kwargs.
    Time parameters (start, stop) are excluded so the same cache entry
    can serve overlapping time ranges.
    """
    parts = [_canonical_dtype(dtype)]
    for k, v in sorted(kwargs.items()):
        if k in ("start", "stop", "chunksize"):
            continue
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return "|".join(parts)
