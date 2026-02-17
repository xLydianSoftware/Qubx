"""
Caching layer for IReader / IStorage.

Provides in-memory caching with optional prefetch to reduce database queries.
Designed as transparent wrappers following the same decorator pattern as
TimeGuardedReader / TimeGuardedStorage.

Architecture:
    ICache              — backend interface for storing/retrieving RawData
    MemoryCache(ICache) — in-memory dict-based implementation with Arrow concat/slice
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

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storage import IReader, IStorage, Transformable


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
        """
        Check if cached ranges fully cover [start, stop).
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
    """

    _data: dict[str, dict[str, RawData]]
    _ranges: dict[str, list[tuple[str, str]]]
    _access_order: OrderedDict[str, None]
    _max_size_bytes: int

    def __init__(self, max_size_mb: int = 1000) -> None:
        self._data = {}
        self._ranges = {}
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
        self._access_order[cache_key] = None
        self._access_order.move_to_end(cache_key, last=True)
        self._maybe_evict()

    def covers(self, cache_key: str, start: str | None, stop: str | None) -> bool:
        if start is None and stop is None:
            return cache_key in self._data

        ranges = self._ranges.get(cache_key)
        if not ranges:
            return False

        merged = _merge_time_ranges(ranges)
        req_start = pd.Timestamp(start) if start else pd.Timestamp.min
        req_stop = pd.Timestamp(stop) if stop else pd.Timestamp.max

        for rs, re in merged:
            if rs <= req_start and re >= req_stop:
                return True
        return False

    def get_ranges(self, cache_key: str) -> list[tuple[str, str]]:
        return list(self._ranges.get(cache_key, []))

    def get_stored_ids(self, cache_key: str) -> list[str]:
        bucket = self._data.get(cache_key)
        return list(bucket.keys()) if bucket else []

    def clear(self, cache_key: str | None = None) -> None:
        if cache_key is None:
            self._data.clear()
            self._ranges.clear()
            self._access_order.clear()
        else:
            self._data.pop(cache_key, None)
            self._ranges.pop(cache_key, None)
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

    def _record_range(self, cache_key: str, start: str, stop: str) -> None:
        if cache_key not in self._ranges:
            self._ranges[cache_key] = []
        self._ranges[cache_key].append((start, stop))
        # - keep merged to avoid unbounded growth
        self._ranges[cache_key] = [(str(s), str(e)) for s, e in _merge_time_ranges(self._ranges[cache_key])]

    def _maybe_evict(self) -> None:
        while self.size_bytes() > self._max_size_bytes and self._access_order:
            # - evict least recently used
            oldest_key, _ = self._access_order.popitem(last=False)
            self._data.pop(oldest_key, None)
            self._ranges.pop(oldest_key, None)
            logger.debug(f"Cache evicted key: {oldest_key}")


# - - - - - - - - - - - - -
# Arrow helpers
# - - - - - - - - - - - - -


def _merge_batches(existing: pa.RecordBatch, incoming: pa.RecordBatch, time_col_idx: int) -> pa.RecordBatch:
    """
    Concatenate two Arrow RecordBatches, deduplicate by time column, sort by time.
    """
    tbl = pa.concat_tables(
        [
            pa.Table.from_batches([existing]),
            pa.Table.from_batches([incoming]),
        ]
    )
    # - sort by time column
    time_col_name = tbl.schema.field(time_col_idx).name
    tbl = tbl.sort_by(time_col_name)

    # - deduplicate by time: convert to pandas, drop duplicates, back to Arrow
    # (Arrow lacks native group-by dedup; pandas roundtrip is fast for cache-sized data)
    pdf = tbl.to_pandas()
    pdf = pdf.drop_duplicates(subset=[time_col_name], keep="last")
    batch = pa.RecordBatch.from_pandas(pdf, schema=existing.schema, preserve_index=False)
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
    ts = pd.Timestamp(time_str)
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
            parsed.append((pd.Timestamp(s), pd.Timestamp(e)))
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
        self._prefetch_period = pd.Timedelta(prefetch_period) if prefetch_period else None
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
        # - chunked reads bypass cache (streaming use case)
        if chunksize > 0:
            return self._reader.read(data_id, dtype, start, stop, chunksize=chunksize, **kwargs)

        cache_key = _make_cache_key(dtype, **kwargs)

        # - normalize data_id to list for uniform handling
        ids = data_id if isinstance(data_id, (list, tuple)) else [data_id]
        is_single = isinstance(data_id, str)

        if self._cache.covers(cache_key, start, stop) and self._all_ids_cached(cache_key, ids):
            # - full cache hit
            return self._build_result(cache_key, ids, is_single, start, stop)

        # - cache miss: fetch from inner reader with optional prefetch
        fetch_stop = stop
        if self._prefetch_period is not None and stop is not None:
            fetch_stop = str(pd.Timestamp(stop) + self._prefetch_period)

        result = self._reader.read(data_id, dtype, start, fetch_stop, **kwargs)
        self._store_result(cache_key, result, start or "", fetch_stop or "")

        # - return sliced to originally requested [start, stop)
        return self._build_result(cache_key, ids, is_single, start, stop)

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

    def _all_ids_cached(self, cache_key: str, ids: list[str]) -> bool:
        stored = set(self._cache.get_stored_ids(cache_key))
        return all(did in stored for did in ids)

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
                raws.append(RawData.from_record_batch(did, cached.dtype, sliced_batch))
            elif cached is not None:
                # - empty RawData, return as-is
                raws.append(cached)
            else:
                # - no cached data for this id, create empty placeholder
                raws.append(RawData.from_record_batch(did, DataType.ALL, pa.RecordBatch.from_pydict({"timestamp": []})))

        if is_single:
            return raws[0]
        return RawMultiData(raws)


class CachedStorage(IStorage):
    """
    Wraps IStorage, returns CachedReader from get_reader().
    Each (exchange, market) pair gets its own CachedReader with independent cache.
    """

    _storage: IStorage
    _readers: dict[str, CachedReader]
    _prefetch_period: str | None
    _max_size_mb: int

    def __init__(
        self,
        storage: IStorage,
        prefetch_period: str | None = None,
        max_size_mb: int = 1000,
    ) -> None:
        self._storage = storage
        self._readers = {}
        self._prefetch_period = prefetch_period
        self._max_size_mb = max_size_mb

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
            cache = MemoryCache(max_size_mb=self._max_size_mb)
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


def _make_cache_key(dtype: DataType | str, **kwargs) -> str:
    """
    Generate a time-stripped cache key from dtype and extra kwargs.
    Time parameters (start, stop) are excluded so the same cache entry
    can serve overlapping time ranges.
    """
    parts = [str(dtype)]
    for k, v in sorted(kwargs.items()):
        if k in ("start", "stop", "chunksize"):
            continue
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return "|".join(parts)
