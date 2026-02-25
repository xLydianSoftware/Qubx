"""
Tests for ICache / MemoryCache / CachedReader / CachedStorage.

Covers:
- MemoryCache: put/get, range tracking, merge, eviction, clear
- CachedReader: cache hit/miss, prefetch, multi-symbol, metadata caching, chunked bypass
- CachedStorage: reader creation, cache isolation, clear_cache
- Arrow-level: slice, merge, dedup
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from qubx.core.basics import DataType, ITimeProvider
from qubx.data.cache import (
    CachedReader,
    CachedStorage,
    MemoryCache,
    _make_cache_key,
    _merge_batches,
    _merge_time_ranges,
    _slice_batch,
)
from qubx.data.containers import RawData, RawMultiData
from qubx.data.guards import TimeGuardedStorage
from qubx.data.storage import IReader
from qubx.data.storages.handy import HandyStorage
from qubx.data.transformers import PandasFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc(
    start: str = "2024-01-01",
    end: str = "2024-01-10",
    freq: str = "1h",
    base_price: float = 40000.0,
) -> pd.DataFrame:
    """
    Generate sample OHLC data.
    """
    idx = pd.date_range(start, end, freq=freq, name="timestamp")
    rng = np.random.default_rng(42)
    n = len(idx)
    close = base_price + rng.standard_normal(n).cumsum() * 100
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, n),
            "high": close + rng.uniform(0, 100, n),
            "low": close - rng.uniform(0, 100, n),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


def _make_funding(start: str = "2024-01-01", end: str = "2024-01-10") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="8h", name="timestamp")
    rng = np.random.default_rng(42)
    return pd.DataFrame({"funding_rate": rng.uniform(-0.001, 0.001, len(idx))}, index=idx)


def _build_storage() -> HandyStorage:
    return HandyStorage(
        {
            "BTCUSDT": [
                _make_ohlc(base_price=40000),
                _make_funding(),
            ],
            "ETHUSDT": [
                _make_ohlc(base_price=3000),
                _make_funding(),
            ],
            "SOLUSDT": [
                _make_ohlc(base_price=100),
            ],
        },
        exchange="BINANCE.UM",
    )


class FixedTimeProvider(ITimeProvider):
    def __init__(self, time: str):
        self._time = np.datetime64(time, "ns")

    def time(self) -> np.datetime64:
        return self._time

    def set_time(self, t: str):
        self._time = np.datetime64(t, "ns")


# ---------------------------------------------------------------------------
# Arrow helpers tests
# ---------------------------------------------------------------------------


class TestArrowHelpers:
    def test_slice_batch_start_stop(self):
        """
        Slice [02:00, 05:00) from hourly data starting at 00:00.
        """
        ts = pd.date_range("2024-01-01", periods=10, freq="1h")
        batch = pa.RecordBatch.from_pydict(
            {
                "timestamp": ts.values,
                "value": list(range(10)),
            }
        )
        sliced = _slice_batch(batch, 0, "2024-01-01 02:00", "2024-01-01 05:00")
        assert sliced.num_rows == 3
        assert sliced.column("value").to_pylist() == [2, 3, 4]

    def test_slice_batch_no_bounds(self):
        """
        No start/stop returns full batch.
        """
        ts = pd.date_range("2024-01-01", periods=5, freq="1h")
        batch = pa.RecordBatch.from_pydict({"timestamp": ts.values, "v": [1, 2, 3, 4, 5]})
        sliced = _slice_batch(batch, 0, None, None)
        assert sliced.num_rows == 5

    def test_slice_batch_empty(self):
        batch = pa.RecordBatch.from_pydict(
            {"timestamp": pa.array([], type=pa.timestamp("ns")), "v": pa.array([], type=pa.int64())}
        )
        sliced = _slice_batch(batch, 0, "2024-01-01", "2024-01-02")
        assert sliced.num_rows == 0

    def test_merge_batches_dedup(self):
        """
        Overlapping batches (OHLC-style: one row per timestamp) are merged and
        deduplicated. In cache merges, overlapping historical rows are always exact
        duplicates — all-column dedup removes them correctly.
        """
        ts1 = pd.date_range("2024-01-01", periods=5, freq="1h")
        ts2 = pd.date_range("2024-01-01 03:00", periods=5, freq="1h")
        b1 = pa.RecordBatch.from_pydict({"timestamp": ts1.values, "v": [10, 20, 30, 40, 50]})
        # - overlapping rows (03, 04) have identical values to b1 (cache merge scenario)
        b2 = pa.RecordBatch.from_pydict({"timestamp": ts2.values, "v": [40, 50, 60, 70, 80]})
        merged = _merge_batches(b1, b2, 0)
        # - 00, 01, 02, 03, 04, 05, 06, 07 = 8 unique rows
        assert merged.num_rows == 8
        # - should be sorted by time
        times = merged.column("timestamp").to_pylist()
        assert times == sorted(times)
        # - values should be deduplicated: b1 for 00-02, b2 for 03-07
        assert merged.column("v").to_pylist() == [10, 20, 30, 40, 50, 60, 70, 80]

    def test_merge_batches_multi_row_per_timestamp(self):
        """
        Fundamental-style data: multiple rows per timestamp (one per metric) must all
        survive merge. All-column dedup never collapses rows with different metrics.
        """
        import numpy as np

        ts = pd.date_range("2024-01-01", periods=3, freq="1d")
        # - existing: 3 days × 2 metrics = 6 rows
        b1 = pa.RecordBatch.from_pydict(
            {
                "timestamp": np.repeat(ts.values, 2),
                "metric": pa.array(["market_cap", "total_volume"] * 3),
                "value": [1000.0, 50.0, 1100.0, 55.0, 1200.0, 60.0],
            }
        )
        # - incoming extends with 2 new days (no overlap with b1)
        ts2 = pd.date_range("2024-01-04", periods=2, freq="1d")
        b2 = pa.RecordBatch.from_pydict(
            {
                "timestamp": np.repeat(ts2.values, 2),
                "metric": pa.array(["market_cap", "total_volume"] * 2),
                "value": [1300.0, 65.0, 1400.0, 70.0],
            }
        )
        merged = _merge_batches(b1, b2, 0)
        # - 5 days × 2 metrics = 10 rows, none dropped
        assert merged.num_rows == 10
        times = merged.column("timestamp").to_pylist()
        assert times == sorted(times)

    def test_merge_time_ranges(self):
        ranges = [("2024-01-01", "2024-01-05"), ("2024-01-03", "2024-01-08"), ("2024-01-10", "2024-01-12")]
        merged = _merge_time_ranges(ranges)
        assert len(merged) == 2
        assert merged[0] == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08"))
        assert merged[1] == (pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-12"))

    def test_merge_time_ranges_empty(self):
        assert _merge_time_ranges([]) == []


class TestMakeCacheKey:
    def test_basic_key(self):
        key = _make_cache_key("ohlc(1h)")
        assert key == "ohlc(1h)"

    def test_key_with_kwargs(self):
        key = _make_cache_key("ohlc(1h)", timeframe="1h")
        assert "timeframe=1h" in key

    def test_key_ignores_time_params(self):
        k1 = _make_cache_key("ohlc(1h)", start="2024-01-01", stop="2024-01-05")
        k2 = _make_cache_key("ohlc(1h)", start="2024-02-01", stop="2024-02-05")
        assert k1 == k2

    def test_key_ignores_chunksize(self):
        k1 = _make_cache_key("ohlc(1h)", chunksize=100)
        k2 = _make_cache_key("ohlc(1h)")
        assert k1 == k2


# ---------------------------------------------------------------------------
# MemoryCache tests
# ---------------------------------------------------------------------------


class TestMemoryCache:
    def _make_raw(self, data_id: str = "BTCUSDT", n: int = 10, start: str = "2024-01-01") -> RawData:
        ts = pd.date_range(start, periods=n, freq="1h")
        batch = pa.RecordBatch.from_pydict(
            {
                "timestamp": ts.values,
                "close": np.random.default_rng(42).uniform(39000, 41000, n),
            }
        )
        return RawData.from_record_batch(data_id, DataType.OHLC["1h"], batch)

    def test_put_and_get(self):
        cache = MemoryCache()
        raw = self._make_raw()
        cache.put("k1", raw, "2024-01-01", "2024-01-01 10:00")
        got = cache.get("k1", "BTCUSDT")
        assert got is not None
        assert len(got) == 10

    def test_get_miss(self):
        cache = MemoryCache()
        assert cache.get("missing", "BTCUSDT") is None

    def test_covers_full(self):
        cache = MemoryCache()
        raw = self._make_raw()
        cache.put("k1", raw, "2024-01-01", "2024-01-02")
        assert cache.covers("k1", "2024-01-01", "2024-01-01 12:00")
        assert cache.covers("k1", "2024-01-01", "2024-01-02")

    def test_covers_not(self):
        cache = MemoryCache()
        raw = self._make_raw()
        cache.put("k1", raw, "2024-01-01", "2024-01-02")
        assert not cache.covers("k1", "2024-01-01", "2024-01-03")

    def test_covers_no_time(self):
        """
        covers(key, None, None) returns True if any data is cached.
        """
        cache = MemoryCache()
        raw = self._make_raw()
        cache.put("k1", raw, "2024-01-01", "2024-01-02")
        assert cache.covers("k1", None, None)
        assert not cache.covers("missing", None, None)

    def test_merge_on_extend(self):
        """
        Putting overlapping data extends the cache, deduplicates exact duplicate rows,
        and merges ranges. Overlapping rows must have identical values (cache merge
        scenario: same historical bars fetched from DB twice).
        """
        cache = MemoryCache()
        ts1 = pd.date_range("2024-01-01", periods=5, freq="1h")  # - 00:00 … 04:00
        ts2 = pd.date_range("2024-01-01 03:00", periods=5, freq="1h")  # - 03:00 … 07:00
        # - overlapping rows at 03:00 / 04:00 have IDENTICAL close values (exact duplicates)
        b1 = pa.RecordBatch.from_pydict({"timestamp": ts1.values, "close": [10.0, 20.0, 30.0, 40.0, 50.0]})
        b2 = pa.RecordBatch.from_pydict({"timestamp": ts2.values, "close": [40.0, 50.0, 60.0, 70.0, 80.0]})
        raw1 = RawData.from_record_batch("BTCUSDT", DataType.OHLC["1h"], b1)
        raw2 = RawData.from_record_batch("BTCUSDT", DataType.OHLC["1h"], b2)
        cache.put("k1", raw1, "2024-01-01", "2024-01-01 05:00")
        cache.put("k1", raw2, "2024-01-01 03:00", "2024-01-01 08:00")

        got = cache.get("k1", "BTCUSDT")
        assert got is not None
        # - 5 + 5 with 2 exact-duplicate overlapping rows → 8 unique rows
        assert len(got) == 8
        # - ranges should be merged into one
        ranges = cache.get_ranges("k1")
        assert len(ranges) == 1

    def test_get_stored_ids(self):
        cache = MemoryCache()
        cache.put("k1", self._make_raw("BTCUSDT"), "2024-01-01", "2024-01-02")
        cache.put("k1", self._make_raw("ETHUSDT"), "2024-01-01", "2024-01-02")
        ids = cache.get_stored_ids("k1")
        assert set(ids) == {"BTCUSDT", "ETHUSDT"}

    def test_clear_specific(self):
        cache = MemoryCache()
        cache.put("k1", self._make_raw(), "2024-01-01", "2024-01-02")
        cache.put("k2", self._make_raw(), "2024-01-01", "2024-01-02")
        cache.clear("k1")
        assert cache.get("k1", "BTCUSDT") is None
        assert cache.get("k2", "BTCUSDT") is not None

    def test_clear_all(self):
        cache = MemoryCache()
        cache.put("k1", self._make_raw(), "2024-01-01", "2024-01-02")
        cache.put("k2", self._make_raw(), "2024-01-01", "2024-01-02")
        cache.clear()
        assert cache.get("k1", "BTCUSDT") is None
        assert cache.get("k2", "BTCUSDT") is None

    def test_eviction_on_size_limit(self):
        """
        When cache exceeds max_size_mb, LRU entries are evicted.
        """
        # - very small limit: 1 byte → immediate eviction of older entries
        cache = MemoryCache(max_size_mb=0)
        cache._max_size_bytes = 1  # - override to 1 byte for testing
        cache.put("k1", self._make_raw(), "2024-01-01", "2024-01-02")
        cache.put("k2", self._make_raw(), "2024-01-01", "2024-01-02")
        # - k1 should be evicted, k2 might also be evicted since both exceed 1 byte
        assert cache.get("k1", "BTCUSDT") is None

    def test_empty_data_records_range(self):
        """
        Putting empty RawData still records the time range.
        """
        cache = MemoryCache()
        empty = RawData.from_record_batch(
            "BTCUSDT",
            DataType.OHLC["1h"],
            pa.RecordBatch.from_pydict(
                {"timestamp": pa.array([], type=pa.timestamp("ns")), "close": pa.array([], type=pa.float64())}
            ),
        )
        cache.put("k1", empty, "2024-01-01", "2024-01-02")
        assert cache.covers("k1", "2024-01-01", "2024-01-02")
        assert cache.get("k1", "BTCUSDT") is None  # - no actual data stored

    def test_size_bytes(self):
        cache = MemoryCache()
        cache.put("k1", self._make_raw(n=100), "2024-01-01", "2024-01-05")
        assert cache.size_bytes() > 0


# ---------------------------------------------------------------------------
# CachedReader tests
# ---------------------------------------------------------------------------


class TestCachedReader:
    def test_non_overlapping_ranges_both_cached(self):
        """
        Cache two non-overlapping time ranges. Both should be independently queryable.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        # - first range
        r1 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        df1 = r1.transform(PandasFrame())
        assert len(df1) > 0

        # - second range (non-overlapping)
        r2 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-05", "2024-01-07")
        df2 = r2.transform(PandasFrame())
        assert len(df2) > 0

        # - re-query first range: should be cache hit
        r3 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        df3 = r3.transform(PandasFrame())
        assert len(df3) == len(df1)

        # - re-query second range: should be cache hit
        r4 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-05", "2024-01-07")
        df4 = r4.transform(PandasFrame())
        assert len(df4) == len(df2)

    def test_adjacent_ranges_merge_into_spanning_hit(self):
        """
        Cache two adjacent ranges. A query spanning both should be a cache hit.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        # - first range: [01-01, 01-03)
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        # - adjacent range: [01-03, 01-05)
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-03", "2024-01-05")

        # - spanning query: [01-01, 01-05) — should be fully covered by merged ranges
        r3 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-05")
        df3 = r3.transform(PandasFrame())
        assert df3.index[0] >= pd.Timestamp("2024-01-01")
        assert df3.index[-1] < pd.Timestamp("2024-01-05")
        # - should have data from the full 4-day range
        assert len(df3) > 48  # at least 2 days of hourly data

    def test_complex_multi_period_progressive_cache(self):
        """
        Progressive cache building: multiple overlapping requests,
        final comprehensive request should be a cache hit.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        # - build cache progressively
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-02", "2024-01-05")
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-04", "2024-01-07")

        # - comprehensive query spanning all cached ranges
        r = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-07")
        df = r.transform(PandasFrame())
        assert df.index[0] >= pd.Timestamp("2024-01-01")
        assert df.index[-1] < pd.Timestamp("2024-01-07")

        expected_hours = 6 * 24  # 6 days of hourly bars
        assert len(df) == expected_hours

    def test_symbol_subset_from_multi_symbol_cache(self):
        """
        Cache data for [BTC, ETH, SOL], then request only [BTC].
        Should be served from cache without hitting inner reader again.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        # - cache 3 symbols
        reader.read(["BTCUSDT", "ETHUSDT", "SOLUSDT"], "ohlc(1h)", "2024-01-01", "2024-01-03")

        # - request single symbol — cache should cover it
        r = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        df = r.transform(PandasFrame())
        assert len(df) > 0
        assert df.index[0] >= pd.Timestamp("2024-01-01")

    def test_overlapping_merge_removes_exact_duplicates(self):
        """
        When extending cache with overlapping data, exact duplicate rows (same values
        for ALL columns) are removed. In cache merges, overlapping rows from a DB
        re-fetch of the same historical data are always exact duplicates.

        NOTE: if DB data was corrected (different value for same timestamp), both rows
        survive and the cache goes stale — caller must clear() the cache in that case.
        """
        cache = MemoryCache()

        # - first batch: hours 0-4
        ts1 = pd.date_range("2024-01-01", periods=5, freq="1h")
        raw1 = RawData.from_record_batch(
            "BTCUSDT",
            DataType.OHLC["1h"],
            pa.RecordBatch.from_pydict({"timestamp": ts1.values, "close": [100.0, 101.0, 102.0, 103.0, 104.0]}),
        )
        cache.put("k1", raw1, "2024-01-01 00:00", "2024-01-01 05:00")

        # - second batch: hours 3-7; overlapping hours (3,4) have IDENTICAL values
        ts2 = pd.date_range("2024-01-01 03:00", periods=5, freq="1h")
        raw2 = RawData.from_record_batch(
            "BTCUSDT",
            DataType.OHLC["1h"],
            pa.RecordBatch.from_pydict({"timestamp": ts2.values, "close": [103.0, 104.0, 105.0, 106.0, 107.0]}),
        )
        cache.put("k1", raw2, "2024-01-01 03:00", "2024-01-01 08:00")

        got = cache.get("k1", "BTCUSDT")
        assert got is not None
        # - 8 unique rows: exact duplicates at hours 3,4 removed
        assert len(got) == 8

        # - check values across the full merged range
        df = got.transform(PandasFrame())
        assert df.loc[pd.Timestamp("2024-01-01 00:00"), "close"] == 100.0
        assert df.loc[pd.Timestamp("2024-01-01 03:00"), "close"] == 103.0
        assert df.loc[pd.Timestamp("2024-01-01 07:00"), "close"] == 107.0

    def test_single_symbol_cache_hit(self):
        """
        Second read with same params hits cache — inner reader not called twice.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        r1 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        df1 = r1.transform(PandasFrame())
        r2 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        df2 = r2.transform(PandasFrame())

        assert len(df1) == len(df2)
        assert df1.index[0] == df2.index[0]
        assert df1.index[-1] == df2.index[-1]

    def test_subrange_from_cache(self):
        """
        Request [01-01, 01-05) first, then [01-02, 01-03) — served from cache.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        # - populate cache with wide range
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-05")

        # - subrange should be served from cache
        r2 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-02", "2024-01-03")
        df2 = r2.transform(PandasFrame())
        assert df2.index[0] >= pd.Timestamp("2024-01-02")
        assert df2.index[-1] < pd.Timestamp("2024-01-03")

    def test_multi_symbol_read(self):
        """
        Reading multiple symbols returns RawMultiData, all cached.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        result = reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)", "2024-01-01", "2024-01-03")
        assert isinstance(result, RawMultiData)
        assert set(result.get_data_ids()) == {"BTCUSDT", "ETHUSDT"}

        # - second read should be cached
        result2 = reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)", "2024-01-01", "2024-01-03")
        assert isinstance(result2, RawMultiData)
        df_btc = result2["BTCUSDT"].transform(PandasFrame())
        df_eth = result2["ETHUSDT"].transform(PandasFrame())
        assert len(df_btc) > 0
        assert len(df_eth) > 0

    def test_prefetch_extends_stop(self):
        """
        With prefetch_period="2d", first read [01-01, 01-03) fetches [01-01, 01-05).
        Second read [01-03, 01-05) should be a cache hit.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner, prefetch_period="2d")

        # - first read: [01-01, 01-03) → fetches [01-01, 01-05) internally
        r1 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        df1 = r1.transform(PandasFrame())
        assert df1.index[-1] < pd.Timestamp("2024-01-03")

        # - second read: [01-03, 01-05) should hit cache
        r2 = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-03", "2024-01-05")
        df2 = r2.transform(PandasFrame())
        assert len(df2) > 0
        assert df2.index[0] >= pd.Timestamp("2024-01-03")

    def test_different_dtypes_separate_cache(self):
        """
        OHLC and funding data use different cache keys.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        r_ohlc = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        r_fund = reader.read("BTCUSDT", DataType.FUNDING_RATE, "2024-01-01", "2024-01-03")

        df_ohlc = r_ohlc.transform(PandasFrame())
        df_fund = r_fund.transform(PandasFrame())

        assert "close" in df_ohlc.columns
        assert "funding_rate" in df_fund.columns

    def test_chunked_read_bypasses_cache(self):
        """
        Chunked reads go directly to inner reader.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        result = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03", chunksize=10)
        # - should return iterator, not Transformable
        chunks = list(result)
        assert len(chunks) > 0

    def test_get_data_id_cached(self):
        """
        get_data_id() is cached after first call.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        ids1 = reader.get_data_id("ohlc(1h)")
        ids2 = reader.get_data_id("ohlc(1h)")
        assert ids1 == ids2
        assert set(ids1) >= {"BTCUSDT", "ETHUSDT"}

    def test_get_data_types_cached(self):
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)

        types1 = reader.get_data_types("BTCUSDT")
        types2 = reader.get_data_types("BTCUSDT")
        assert types1 == types2
        assert len(types1) > 0

    def test_close_delegates(self):
        """
        close() propagates to inner reader.
        """
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner)
        # - should not raise
        reader.close()

    def test_repr(self):
        storage = _build_storage()
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        reader = CachedReader(inner, prefetch_period="1w")
        r = repr(reader)
        assert "CachedReader" in r
        assert "prefetch" in r


# ---------------------------------------------------------------------------
# CachedStorage tests
# ---------------------------------------------------------------------------


class TestCachedStorage:
    def test_get_reader_returns_cached_reader(self):
        storage = _build_storage()
        cached = CachedStorage(storage)
        reader = cached.get_reader("BINANCE.UM", "SWAP")
        assert isinstance(reader, CachedReader)

    def test_same_reader_instance(self):
        """
        Repeated get_reader() calls return the same CachedReader instance.
        """
        storage = _build_storage()
        cached = CachedStorage(storage)
        r1 = cached.get_reader("BINANCE.UM", "SWAP")
        r2 = cached.get_reader("BINANCE.UM", "SWAP")
        assert r1 is r2

    def test_delegates_exchanges(self):
        storage = _build_storage()
        cached = CachedStorage(storage)
        assert cached.get_exchanges() == storage.get_exchanges()

    def test_delegates_market_types(self):
        storage = _build_storage()
        cached = CachedStorage(storage)
        for ex in storage.get_exchanges():
            assert cached.get_market_types(ex) == storage.get_market_types(ex)

    def test_subscript_access(self):
        """
        CachedStorage["BINANCE.UM", "SWAP"] works via __getitem__.
        """
        storage = _build_storage()
        cached = CachedStorage(storage)
        reader = cached["BINANCE.UM", "SWAP"]
        assert isinstance(reader, CachedReader)

    def test_prefetch_propagates(self):
        storage = _build_storage()
        cached = CachedStorage(storage, prefetch_period="3d")
        reader = cached.get_reader("BINANCE.UM", "SWAP")
        assert reader._prefetch_period == pd.Timedelta("3d")

    def test_clear_cache_specific(self):
        storage = _build_storage()
        cached = CachedStorage(storage)
        reader = cached.get_reader("BINANCE.UM", "SWAP")
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        # - verify data is cached
        assert reader._cache.size_bytes() > 0
        cached.clear_cache("BINANCE.UM", "SWAP")
        assert reader._cache.size_bytes() == 0

    def test_clear_cache_all(self):
        storage = _build_storage()
        cached = CachedStorage(storage)
        reader = cached.get_reader("BINANCE.UM", "SWAP")
        reader.read("BTCUSDT", "ohlc(1h)", "2024-01-01", "2024-01-03")
        cached.clear_cache()
        assert reader._cache.size_bytes() == 0

    def test_repr(self):
        storage = _build_storage()
        cached = CachedStorage(storage, prefetch_period="1w")
        r = repr(cached)
        assert "CachedStorage" in r
        assert "prefetch" in r


# ---------------------------------------------------------------------------
# Integration: CachedStorage + TimeGuardedStorage
# ---------------------------------------------------------------------------


class TestCachedWithTimeGuard:
    """
    Tests the full composition: TimeGuardedStorage(CachedStorage(HandyStorage)).
    CachedReader prefetches ahead, TimeGuardedReader clamps what strategy sees.
    """

    def test_timeguard_over_cached_basic(self):
        """
        TimeGuard clamps stop to sim time, CachedReader caches the data.
        """
        raw_storage = _build_storage()
        cached_storage = CachedStorage(raw_storage, prefetch_period="3d")
        tp = FixedTimeProvider("2024-01-03T00:00:00")
        guarded_storage = TimeGuardedStorage(cached_storage, tp)

        reader = guarded_storage.get_reader("BINANCE.UM", "SWAP")
        result = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-10")
        df = result.transform(PandasFrame())

        # - TimeGuard clamps stop to 2024-01-03, exclusive → last bar < 01-03
        assert df.index[-1] < pd.Timestamp("2024-01-03")
        assert df.index[0] >= pd.Timestamp("2024-01-01")

    def test_prefetch_visible_on_time_advance(self):
        """
        CachedReader prefetches 3d ahead. After advancing sim time,
        the prefetched data becomes visible through TimeGuard.
        """
        raw_storage = _build_storage()
        cached_storage = CachedStorage(raw_storage, prefetch_period="3d")
        tp = FixedTimeProvider("2024-01-03T00:00:00")
        guarded_storage = TimeGuardedStorage(cached_storage, tp)

        reader = guarded_storage.get_reader("BINANCE.UM", "SWAP")

        # - first read: sim time 01-03, requests up to 01-10
        # - TimeGuard clamps to 01-03, CachedReader fetches [01-01, 01-06) internally
        r1 = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-10")
        df1 = r1.transform(PandasFrame())
        assert df1.index[-1] < pd.Timestamp("2024-01-03")

        # - advance sim time to 01-05
        tp.set_time("2024-01-05T00:00:00")

        # - second read: cache already has data up to 01-06 from prefetch
        r2 = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-10")
        df2 = r2.transform(PandasFrame())
        assert df2.index[-1] < pd.Timestamp("2024-01-05")
        assert len(df2) > len(df1)

    def test_multi_symbol_with_guard(self):
        """
        Multi-symbol reads work through the full stack.
        """
        raw_storage = _build_storage()
        cached_storage = CachedStorage(raw_storage)
        tp = FixedTimeProvider("2024-01-05T00:00:00")
        guarded_storage = TimeGuardedStorage(cached_storage, tp)

        reader = guarded_storage.get_reader("BINANCE.UM", "SWAP")
        result = reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)", "2024-01-01", "2024-01-10")
        assert isinstance(result, RawMultiData)
        for did in ["BTCUSDT", "ETHUSDT"]:
            df = result[did].transform(PandasFrame())
            assert df.index[-1] < pd.Timestamp("2024-01-05")


# ---------------------------------------------------------------------------
# Regression: all-symbols prefetch + long-format data (e.g. fundamental)
# ---------------------------------------------------------------------------


def _make_long_fundamental(asset: str, start: str, end: str, freq: str = "1D") -> pd.DataFrame:
    """
    Build long-format fundamental data mimicking QuestDB schema:
      (timestamp, asset, metric, value) — one row per metric per day.
    """
    dates = pd.date_range(start, end, freq=freq, name="timestamp")
    rng = np.random.default_rng(hash(asset) & 0xFFFF)
    rows = []
    for d in dates:
        for metric in ("market_cap", "total_volume"):
            rows.append({"asset": asset, "metric": metric, "value": rng.uniform(1e6, 1e9)})
    return pd.DataFrame(rows, index=dates.repeat(2))


class _FundamentalMockReader(IReader):
    """
    In-memory reader serving long-format fundamental data (mimicking QuestDB).
    Only returns assets that have data within [start, stop) — matches SQL WHERE behaviour.
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        # - asset -> DataFrame with DatetimeIndex and columns (asset, metric, value)
        self._data = {k.upper(): v for k, v in data.items()}

    def get_data_id(self, dtype=DataType.ALL) -> list[str]:
        return list(self._data.keys())

    def get_data_types(self, data_id: str) -> list[str]:
        return ["fundamental"] if data_id.upper() in self._data else []

    def get_time_range(self, data_id: str, dtype) -> tuple[np.datetime64, np.datetime64]:
        df = self._data.get(data_id.upper())
        if df is None or df.empty:
            return (np.datetime64("NaT"), np.datetime64("NaT"))
        return (np.datetime64(df.index[0]), np.datetime64(df.index[-1]))

    def read(self, data_id, dtype, start=None, stop=None, chunksize=0, **kwargs):
        ids = (
            list(data_id)
            if isinstance(data_id, (list, tuple, set)) and data_id
            else (list(self._data.keys()) if isinstance(data_id, (list, tuple, set)) else [data_id.upper()])
        )
        raws = []
        for did in ids:
            df = self._data.get(did.upper(), pd.DataFrame())
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if stop:
                df = df[df.index < pd.Timestamp(stop)]
            if df.empty:
                # - mimic SQL: skip assets with no data in range for all-symbol requests
                if isinstance(data_id, (list, tuple, set)):
                    continue
            raws.append(RawData.from_pandas(did, "fundamental", df.reset_index()))

        if isinstance(data_id, str):
            return raws[0] if raws else RawData.from_pandas(data_id, "fundamental", pd.DataFrame())
        return RawMultiData(raws)


class TestCachedReaderFundamentalRegression:
    """
    Regression tests for the bug where CachedReader.read([], "fundamental", narrow_range)
    includes assets with empty slices (data exists only in the prefetched-but-not-requested
    window), causing to_pd(False) to raise ValueError in _pivot_to_wide.

    Root cause: prefetch fetches [start, stop+prefetch]. The extended range may contain
    assets with no data in [start, stop]. These get stored in cache. On _build_result,
    all cached assets are sliced to [start, stop], producing empty RawData. When
    combine_data calls _pivot_to_wide on ALL per-asset DataFrames (because at least one
    has non-unique timestamps), empty DataFrames raise ValueError.

    Without cache: SQL WHERE naturally excludes assets with no data in range.
    """

    _EARLY_ASSET = "BTC"  # - has data from the start
    _LATE_ASSET = "NEWCOIN"  # - only listed 20 days in

    _FULL_START = "2026-01-01"
    _FULL_END = "2026-02-20"
    _LATE_START = "2026-01-20"  # - NEWCOIN has no data before this
    _QUERY_STOP = "2026-01-15"  # - query ends BEFORE NEWCOIN has any data

    def _build_reader(self) -> _FundamentalMockReader:
        return _FundamentalMockReader(
            {
                self._EARLY_ASSET: _make_long_fundamental(self._EARLY_ASSET, self._FULL_START, self._FULL_END),
                self._LATE_ASSET: _make_long_fundamental(self._LATE_ASSET, self._LATE_START, self._FULL_END),
            }
        )

    def test_direct_read_excludes_late_asset(self):
        """
        Baseline: direct reader only returns assets with data in the requested range.
        """
        reader = self._build_reader()
        result = reader.read([], "fundamental", self._FULL_START, self._QUERY_STOP)
        assert isinstance(result, RawMultiData)
        ids = [r.data_id for r in result]
        assert self._EARLY_ASSET in ids
        # - NEWCOIN has no data before 2026-01-15, must be excluded
        assert self._LATE_ASSET not in ids

    def test_cached_all_symbols_with_prefetch_excludes_empty_slices(self):
        """
        CachedReader with prefetch stores all assets from the extended range, but
        _build_result must NOT include assets whose slice is empty for the requested window.
        Fixes: ValueError from _pivot_to_wide on empty DataFrame.
        """
        inner = self._build_reader()
        # - prefetch_period="60d" → extended fetch covers NEWCOIN's data (Jan 20+)
        reader = CachedReader(inner, prefetch_period="60d")

        result = reader.read([], "fundamental", self._FULL_START, self._QUERY_STOP)
        assert isinstance(result, RawMultiData)
        ids = [r.data_id for r in result]

        # - BTC must be present (has data in [2026-01-01, 2026-01-15))
        assert self._EARLY_ASSET in ids
        # - NEWCOIN must NOT appear (empty slice for the requested window)
        assert self._LATE_ASSET not in ids, (
            "CachedReader returned NEWCOIN with empty slice — this causes to_pd(False) to fail"
        )

    def test_to_pd_does_not_raise_with_prefetch(self):
        """
        End-to-end: calling .to_pd(False) on the all-symbols prefetched result
        must not raise ValueError ('no varying columns to pivot on').
        This was the original symptom reported during on_fit with enable_prefetch=True.
        """
        inner = self._build_reader()
        reader = CachedReader(inner, prefetch_period="60d")

        result = reader.read([], "fundamental", self._FULL_START, self._QUERY_STOP)
        # - must not raise
        df = result.to_pd(False)
        assert not df.empty
        assert self._EARLY_ASSET.upper() in df.columns or any(self._EARLY_ASSET.upper() in str(c) for c in df.columns)

    def test_specific_asset_request_still_returns_empty_on_miss(self):
        """
        For a specific asset request (not all-symbols), empty RawData is still
        returned when the slice is empty — consistent with non-cached behaviour.
        """
        inner = self._build_reader()
        reader = CachedReader(inner, prefetch_period="60d")

        # - warm the cache with extended range
        reader.read([], "fundamental", self._FULL_START, self._FULL_END)

        # - now query NEWCOIN for a range before it has data
        result = reader.read(self._LATE_ASSET, "fundamental", self._FULL_START, self._QUERY_STOP)
        assert isinstance(result, RawData)
        # - empty: NEWCOIN has no data before 2026-01-15
        assert len(result) == 0
