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
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from qubx import logger
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storage import IReader, IStorage, Transformable
from qubx.utils.time import now_utc


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
    Concatenate two Arrow RecordBatches, deduplicate exact duplicate rows, sort by time.

    Deduplication is done on ALL columns (exact row match), not just the time column.
    This correctly handles data where multiple rows share a timestamp but differ in other
    columns (e.g. fundamental data with one row per metric per timestamp).
    For OHLC, overlapping rows between existing and incoming are always identical so
    exact dedup still removes them correctly.
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

    # - deduplicate exact duplicate rows across all columns, then convert back to Arrow
    # (Arrow lacks native dedup; pandas roundtrip is fast for cache-sized data)
    pdf = tbl.to_pandas()
    pdf = pdf.drop_duplicates(keep="last")
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
        # Normalize start/stop so that start <= stop regardless of caller ordering
        if start is not None and stop is not None and pd.Timestamp(start) > pd.Timestamp(stop):
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
            # - for "all" requests: range coverage is sufficient — return whatever was stored
            if self._cache.covers(cache_key, start, stop):
                stored_ids = self._cache.get_stored_ids(cache_key)
                result = self._build_result(cache_key, stored_ids, False, start, stop)
                # - when caller requests chunked iteration, wrap the in-memory result in a
                #   single-element iterator; real chunking offers no benefit once data is cached
                return iter([result]) if chunksize > 0 else result
        else:
            ids = data_id if isinstance(data_id, (list, tuple)) else [data_id]
            is_single = isinstance(data_id, str)

            if self._cache.covers(cache_key, start, stop):
                missing = self._missing_ids(cache_key, ids)

                if not missing:
                    # - full hit: all ids cached and range covered
                    result = self._build_result(cache_key, ids, is_single, start, stop)
                    return iter([result]) if chunksize > 0 else result

                if len(missing) < len(ids):
                    # - partial hit: range covered but some ids are new
                    # - fetch ONLY the missing symbols — no need to re-read already-cached ones
                    fetch_stop = self._compute_fetch_stop(stop)
                    miss_result = self._reader.read(missing, dtype, start, fetch_stop, **kwargs)
                    self._store_result(cache_key, miss_result, start or "", fetch_stop or "")
                    result = self._build_result(cache_key, ids, is_single, start, stop)
                    return iter([result]) if chunksize > 0 else result

                # - all ids missing but range was recorded by a prior fetch of different symbols
                # - fall through to all-symbols fallback / full miss below

            # - fallback: check all-symbols cache (data stored via read([], ...) uses a different key)
            all_key = _make_cache_key(dtype, __all__=True, **kwargs)
            if all_key != cache_key and self._cache.covers(all_key, start, stop):
                missing = self._missing_ids(all_key, ids)
                if not missing:
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

    def _missing_ids(self, cache_key: str, ids: list[str]) -> list[str]:
        """
        Return the subset of ids not yet stored in cache for this cache_key.
        """
        stored = set(self._cache.get_stored_ids(cache_key))
        return [did for did in ids if did not in stored]

    def _compute_fetch_stop(self, stop: str | None) -> str | None:
        """
        Extend stop by prefetch_period (clamped to now) for inner reader calls.
        Returns stop unchanged when prefetch is disabled.
        """
        if self._prefetch_period is not None and stop is not None:
            prefetched = pd.Timestamp(stop) + self._prefetch_period
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
