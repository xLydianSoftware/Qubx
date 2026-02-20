from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storage import IReader, IStorage
from qubx.data.storages.multi import MultiReader, MultiStorage

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ohlc_batch(
    timestamps: list[int],
    closes: list[float],
    data_id: str = "BTCUSDT",
) -> RawData:
    """
    \n
    Build a minimal RawData with [time, open, high, low, close] columns.
    Timestamps are int64 nanoseconds.
    \n
    """
    n = len(timestamps)
    batch = pa.RecordBatch.from_pydict(
        {
            "time": pa.array(timestamps, type=pa.int64()),
            "open": pa.array(closes, type=pa.float64()),
            "high": pa.array(closes, type=pa.float64()),
            "low": pa.array(closes, type=pa.float64()),
            "close": pa.array(closes, type=pa.float64()),
        }
    )
    return RawData.from_record_batch(data_id, DataType.OHLC["1h"], batch)


def _ts(date: str) -> int:
    """Convert ISO date string to int64 nanoseconds."""
    return int(pd.Timestamp(date).value)


def _mock_reader(result: RawData | RawMultiData | None = None) -> IReader:
    reader = MagicMock(spec=IReader)
    reader.read.return_value = result
    reader.get_data_id.return_value = []
    reader.get_data_types.return_value = []
    reader.get_time_range.return_value = (None, None)
    return reader


def _mock_storage(exchanges: list[str] = (), market_types: list[str] = (), reader: IReader | None = None) -> IStorage:
    s = MagicMock(spec=IStorage)
    s.get_exchanges.return_value = list(exchanges)
    s.get_market_types.return_value = list(market_types)
    s.get_reader.return_value = reader or MagicMock(spec=IReader)
    return s


# ---------------------------------------------------------------------------
# MultiReader — read() tests
# ---------------------------------------------------------------------------


class TestMultiReaderRead:
    def test_single_reader_passthrough(self):
        # - single reader: data should come through unchanged
        raw = _make_ohlc_batch([_ts("2024-01-01"), _ts("2024-01-02")], [100.0, 101.0])
        reader = _mock_reader(raw)

        result = MultiReader([reader]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=0)

        assert isinstance(result, RawData)
        assert len(result) == 2

    def test_two_readers_non_overlapping_merged_and_sorted(self):
        # - reader1 has Jan 1-2, reader2 has Jan 3-4 → merged result has 4 rows sorted
        r1 = _make_ohlc_batch([_ts("2024-01-01"), _ts("2024-01-02")], [1.0, 2.0])
        r2 = _make_ohlc_batch([_ts("2024-01-03"), _ts("2024-01-04")], [3.0, 4.0])

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=0)

        assert isinstance(result, RawData)
        assert len(result) == 4
        times = result.data.column("time").to_pylist()
        assert times == sorted(times)

    def test_exact_duplicate_timestamps_last_reader_wins(self):
        # - both readers have same timestamp; last reader's value should survive
        ts = _ts("2024-01-01")
        r1 = _make_ohlc_batch([ts], [10.0])
        r2 = _make_ohlc_batch([ts], [99.0])

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=0)

        assert isinstance(result, RawData)
        assert len(result) == 1
        assert result.data.column("close").to_pylist() == [99.0]

    def test_partial_overlap_deduplication(self):
        # - reader1: Jan 1-3, reader2: Jan 2-4 → merge → Jan 1-4, reader2 wins on Jan 2-3
        r1 = _make_ohlc_batch(
            [_ts("2024-01-01"), _ts("2024-01-02"), _ts("2024-01-03")],
            [1.0, 2.0, 3.0],
        )
        r2 = _make_ohlc_batch(
            [_ts("2024-01-02"), _ts("2024-01-03"), _ts("2024-01-04")],
            [20.0, 30.0, 40.0],
        )

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=0)

        assert isinstance(result, RawData)
        assert len(result) == 4
        closes = result.data.column("close").to_pylist()
        # - Jan 1 from r1, Jan 2-3 from r2 (last wins), Jan 4 from r2
        assert closes == [1.0, 20.0, 30.0, 40.0]

    def test_all_readers_return_none_raises(self):
        # - no reader could provide data: MultiReader raises ValueError (same as individual readers)
        r1 = _mock_reader(None)
        r2 = _mock_reader(None)

        with pytest.raises(ValueError, match="No data for"):
            MultiReader([r1, r2]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=0)

    def test_one_reader_fails_other_succeeds(self):
        raw = _make_ohlc_batch([_ts("2024-01-01")], [42.0])
        bad = MagicMock(spec=IReader)
        bad.read.side_effect = RuntimeError("connection refused")

        result = MultiReader([bad, _mock_reader(raw)]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=0)

        assert isinstance(result, RawData)
        assert len(result) == 1

    def test_chunked_yields_correct_chunk_sizes(self):
        ts_list = [_ts(f"2024-01-0{i}") for i in range(1, 6)]  # 5 rows
        raw = _make_ohlc_batch(ts_list, [float(i) for i in range(5)])

        chunks = list(MultiReader([_mock_reader(raw)]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=2))

        assert len(chunks) == 3  # ceil(5/2) = 3
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    def test_chunked_merged_then_chunked(self):
        r1 = _make_ohlc_batch([_ts("2024-01-01"), _ts("2024-01-03")], [1.0, 3.0])
        r2 = _make_ohlc_batch([_ts("2024-01-02"), _ts("2024-01-04")], [2.0, 4.0])

        chunks = list(
            MultiReader([_mock_reader(r1), _mock_reader(r2)]).read("BTCUSDT", DataType.OHLC["1h"], chunksize=3)
        )

        # - 4 rows merged, chunk size 3 → 2 chunks
        assert len(chunks) == 2
        all_rows = sum(len(c) for c in chunks)
        assert all_rows == 4


# ---------------------------------------------------------------------------
# MultiReader — multi-symbol path
# ---------------------------------------------------------------------------


class TestMultiReaderMultiSymbol:
    def test_multi_symbol_merged_per_id(self):
        btc1 = _make_ohlc_batch([_ts("2024-01-01")], [1.0], data_id="BTCUSDT")
        eth1 = _make_ohlc_batch([_ts("2024-01-01")], [10.0], data_id="ETHUSDT")
        btc2 = _make_ohlc_batch([_ts("2024-01-02")], [2.0], data_id="BTCUSDT")
        eth2 = _make_ohlc_batch([_ts("2024-01-02")], [20.0], data_id="ETHUSDT")

        r1 = _mock_reader(RawMultiData([btc1, eth1]))
        r2 = _mock_reader(RawMultiData([btc2, eth2]))

        result = MultiReader([r1, r2]).read(["BTCUSDT", "ETHUSDT"], DataType.OHLC["1h"])

        assert isinstance(result, RawMultiData)
        assert len(result["BTCUSDT"]) == 2
        assert len(result["ETHUSDT"]) == 2


# ---------------------------------------------------------------------------
# MultiReader — metadata methods
# ---------------------------------------------------------------------------


class TestMultiReaderMetadata:
    def test_get_data_id_union(self):
        r1, r2 = MagicMock(spec=IReader), MagicMock(spec=IReader)
        r1.get_data_id.return_value = ["BTCUSDT", "ETHUSDT"]
        r2.get_data_id.return_value = ["ETHUSDT", "SOLUSDT"]

        ids = MultiReader([r1, r2]).get_data_id(DataType.OHLC["1h"])

        assert ids == sorted({"BTCUSDT", "ETHUSDT", "SOLUSDT"})

    def test_get_data_types_union(self):
        r1, r2 = MagicMock(spec=IReader), MagicMock(spec=IReader)
        r1.get_data_types.return_value = [DataType.OHLC["1h"]]
        r2.get_data_types.return_value = [DataType.OHLC["1h"], DataType.TRADE]

        dtypes = set(MultiReader([r1, r2]).get_data_types("BTCUSDT"))

        assert DataType.OHLC["1h"] in dtypes
        assert DataType.TRADE in dtypes

    def test_get_time_range_min_max(self):
        r1, r2 = MagicMock(spec=IReader), MagicMock(spec=IReader)
        t0 = np.datetime64("2023-01-01", "ns")
        t1 = np.datetime64("2023-06-01", "ns")
        t2 = np.datetime64("2023-12-31", "ns")
        r1.get_time_range.return_value = (t0, t1)
        r2.get_time_range.return_value = (t1, t2)

        lo, hi = MultiReader([r1, r2]).get_time_range("BTCUSDT", DataType.OHLC["1h"])

        assert lo == t0
        assert hi == t2

    def test_get_time_range_all_none(self):
        r1 = MagicMock(spec=IReader)
        r1.get_time_range.return_value = (None, None)

        lo, hi = MultiReader([r1]).get_time_range("BTCUSDT", DataType.OHLC["1h"])

        assert lo is None
        assert hi is None

    def test_close_closes_all_readers(self):
        r1, r2 = MagicMock(spec=IReader), MagicMock(spec=IReader)
        MultiReader([r1, r2]).close()
        r1.close.assert_called_once()
        r2.close.assert_called_once()


# ---------------------------------------------------------------------------
# MultiReader — tolerance-based merging
# ---------------------------------------------------------------------------

# - helper: build a batch where every timestamp is offset by ``lag_secs`` from
#   the reference timestamps, useful for simulating two sources that disagree
#   slightly on event timing

def _offset_batch(
    base_timestamps: list[int],
    closes: list[float],
    lag_secs: int = 0,
    data_id: str = "BTCUSDT",
) -> RawData:
    lag_ns = lag_secs * 1_000_000_000
    shifted = [t + lag_ns for t in base_timestamps]
    return _make_ohlc_batch(shifted, closes, data_id)


class TestMultiReaderTolerance:
    """Tests for tolerance-based and keep='first' deduplication modes."""

    # - shared base timestamps: 3 events one hour apart (in nanoseconds)
    _base = [
        int(pd.Timestamp("2024-01-01 08:00").value),
        int(pd.Timestamp("2024-01-01 16:00").value),
        int(pd.Timestamp("2024-01-02 00:00").value),
    ]

    def test_exact_dedup_default_no_tolerance(self):
        # - no tolerance: identical timestamps collapsed, last wins
        r1 = _make_ohlc_batch(self._base, [1.0, 2.0, 3.0])
        r2 = _make_ohlc_batch(self._base, [10.0, 20.0, 30.0])

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)]).read(
            "BTCUSDT", DataType.OHLC["1h"]
        )

        assert len(result) == 3
        assert result.data.column("close").to_pylist() == [10.0, 20.0, 30.0]

    def test_tolerance_dedup_within_window_last_wins(self):
        # - reader 2 lags by 30 s → within 1-min window → deduplicated, last kept
        r1 = _make_ohlc_batch(self._base, [1.0, 2.0, 3.0])
        r2 = _offset_batch(self._base, [10.0, 20.0, 30.0], lag_secs=30)

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)], tolerance="1min").read(
            "BTCUSDT", DataType.OHLC["1h"]
        )

        # - 3 groups, each deduped to 1 row (r2 wins as last)
        assert len(result) == 3
        assert result.data.column("close").to_pylist() == [10.0, 20.0, 30.0]

    def test_tolerance_dedup_within_window_first_wins(self):
        # - keep="first" → r1 values survive even though r2 lags by 30 s
        r1 = _make_ohlc_batch(self._base, [1.0, 2.0, 3.0])
        r2 = _offset_batch(self._base, [10.0, 20.0, 30.0], lag_secs=30)

        result = MultiReader(
            [_mock_reader(r1), _mock_reader(r2)], tolerance="1min", keep="first"
        ).read("BTCUSDT", DataType.OHLC["1h"])

        assert len(result) == 3
        assert result.data.column("close").to_pylist() == [1.0, 2.0, 3.0]

    def test_tolerance_outside_window_not_deduped(self):
        # - 2-min lag > 1-min tolerance → all 6 rows kept
        r1 = _make_ohlc_batch(self._base, [1.0, 2.0, 3.0])
        r2 = _offset_batch(self._base, [10.0, 20.0, 30.0], lag_secs=120)

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)], tolerance="1min").read(
            "BTCUSDT", DataType.OHLC["1h"]
        )

        assert len(result) == 6

    def test_zero_tolerance_equivalent_to_exact(self):
        # - tolerance="0s" behaves the same as no tolerance: only identical timestamps merge
        ts = self._base[0]
        r1 = _make_ohlc_batch([ts], [1.0])
        r2 = _make_ohlc_batch([ts], [9.0])

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)], tolerance="0s").read(
            "BTCUSDT", DataType.OHLC["1h"]
        )

        assert len(result) == 1
        assert result.data.column("close").to_pylist() == [9.0]

    def test_multi_storage_tolerance_forwarded_to_reader(self):
        # - tolerance set on MultiStorage must reach the inner MultiReader
        from qubx.data.storages.multi import MultiReader as MR

        s1 = _mock_storage(reader=MagicMock(spec=IReader))
        s2 = _mock_storage(reader=MagicMock(spec=IReader))

        ms = MultiStorage([s1, s2], tolerance="30s", keep="first")
        reader = ms.get_reader("BINANCE.UM", "SWAP")

        assert isinstance(reader, MR)
        assert reader._tolerance == "30s"
        assert reader._keep == "first"

    def test_tolerance_non_overlapping_data_unaffected(self):
        # - 8-hour gap between readers >> any reasonable tolerance → no dedup, 6 rows
        r1 = _make_ohlc_batch(self._base[:2], [1.0, 2.0])
        r2 = _make_ohlc_batch(self._base[1:], [20.0, 3.0])  # shares _base[1]

        # - with tolerance="1min" only the shared exact timestamp at _base[1] is within window
        result_tol = MultiReader([_mock_reader(r1), _mock_reader(r2)], tolerance="1min").read(
            "BTCUSDT", DataType.OHLC["1h"]
        )
        # - _base[1] deduped (within 1min of itself), rest unique → 3 rows
        assert len(result_tol) == 3

    def test_tolerance_result_is_sorted(self):
        # - merged with tolerance must still be sorted chronologically
        r1 = _make_ohlc_batch(self._base, [1.0, 2.0, 3.0])
        r2 = _offset_batch(self._base, [10.0, 20.0, 30.0], lag_secs=30)

        result = MultiReader([_mock_reader(r1), _mock_reader(r2)], tolerance="1min").read(
            "BTCUSDT", DataType.OHLC["1h"]
        )

        times = result.data.column("time").to_pylist()
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# MultiStorage
# ---------------------------------------------------------------------------


class TestMultiStorage:
    def test_get_exchanges_union(self):
        s1 = _mock_storage(exchanges=["BINANCE.UM", "OKX"])
        s2 = _mock_storage(exchanges=["OKX", "BYBIT"])

        exchanges = MultiStorage([s1, s2]).get_exchanges()

        assert exchanges == sorted({"BINANCE.UM", "OKX", "BYBIT"})

    def test_get_market_types_union(self):
        s1 = _mock_storage(market_types=["SWAP"])
        s2 = _mock_storage(market_types=["SWAP", "SPOT"])

        markets = MultiStorage([s1, s2]).get_market_types("BINANCE.UM")

        assert "SWAP" in markets
        assert "SPOT" in markets

    def test_get_reader_single_storage_returns_reader_directly(self):
        # - only one storage → its reader is returned without wrapping
        inner_reader = MagicMock(spec=IReader)
        s = _mock_storage(reader=inner_reader)

        reader = MultiStorage([s]).get_reader("BINANCE.UM", "SWAP")

        assert reader is inner_reader

    def test_get_reader_multiple_storages_returns_multi_reader(self):
        s1 = _mock_storage(reader=MagicMock(spec=IReader))
        s2 = _mock_storage(reader=MagicMock(spec=IReader))

        reader = MultiStorage([s1, s2]).get_reader("BINANCE.UM", "SWAP")

        assert isinstance(reader, MultiReader)

    def test_get_reader_cached(self):
        # - second call for same (exchange, market) must return the exact same object
        s1 = _mock_storage(reader=MagicMock(spec=IReader))
        s2 = _mock_storage(reader=MagicMock(spec=IReader))
        ms = MultiStorage([s1, s2])

        r1 = ms.get_reader("BINANCE.UM", "SWAP")
        r2 = ms.get_reader("BINANCE.UM", "SWAP")

        assert r1 is r2

    def test_get_reader_no_matching_storage_raises(self):
        s = _mock_storage()
        s.get_reader.side_effect = ValueError("no such reader")

        with pytest.raises(ValueError, match="No reader available"):
            MultiStorage([s]).get_reader("UNKNOWN", "SPOT")

    def test_get_reader_skips_failing_storage(self):
        # - first storage fails, second succeeds → result is the second reader (not wrapped)
        good_reader = MagicMock(spec=IReader)
        bad = _mock_storage()
        bad.get_reader.side_effect = RuntimeError("down")
        good = _mock_storage(reader=good_reader)

        reader = MultiStorage([bad, good]).get_reader("BINANCE.UM", "SWAP")

        assert reader is good_reader

    def test_two_handy_storages_overlapping_ohlc(self):
        """
        Two HandyStorages carry overlapping slices of the same 24-bar OHLC dataset.

          full data : 2020-01-01 00:00 – 23:00  (24 × 1h bars)
          storage 1 : slice 00:00 – 15:00       (16 bars)
          storage 2 : slice 12:00 – 23:00       (12 bars, overlap 12:00–15:00)

        MultiStorage must reconstruct the full 24-bar series without gaps or
        duplicates, and the merged result must match the original DataFrame.
        """
        from qubx.data.storages.handy import HandyStorage

        # - generate one continuous 24-bar series as the ground truth
        idx = pd.date_range("2020-01-01 00:00", periods=24, freq="1h", name="timestamp")
        rng = np.random.default_rng(0)
        close = 10_000.0 + rng.standard_normal(24).cumsum() * 50
        full_df = pd.DataFrame(
            {
                "open": close - rng.uniform(0, 10, 24),
                "high": close + rng.uniform(0, 20, 24),
                "low": close - rng.uniform(0, 20, 24),
                "close": close,
                "volume": rng.uniform(100, 1_000, 24),
            },
            index=idx,
        )

        # - slice into two overlapping windows
        slice1 = full_df.loc[:"2020-01-01 15:00"]  # hours 00–15 (16 bars)
        slice2 = full_df.loc["2020-01-01 12:00":]  # hours 12–23 (12 bars)

        assert len(slice1) == 16
        assert len(slice2) == 12

        s1 = HandyStorage({"BTCUSDT": slice1}, exchange="BINANCE.UM")
        s2 = HandyStorage({"BTCUSDT": slice2}, exchange="BINANCE.UM")

        reader = MultiStorage([s1, s2]).get_reader("BINANCE.UM", "SWAP")
        raw = reader.read("BTCUSDT", "ohlc(1h)", "2020-01-01 00:00", "2020-01-02 00:00", chunksize=0)
        result_df = raw.to_pd()

        # - must recover all 24 unique hourly bars
        assert len(result_df) == 24

        # - index is strictly monotone ascending with no duplicates
        assert result_df.index.is_monotonic_increasing
        assert result_df.index.is_unique

        # - values must match the original ground-truth series
        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True),
            full_df.reset_index(drop=True),
            check_index_type=False,
        )

    def test_two_handy_storages_overlapping_ohlc_chunked(self):
        """
        \n
        Same overlapping-slice setup as test_two_handy_storages_overlapping_ohlc but
        reading with chunksize=3.

        MultiReader merges both sources in memory, then re-yields in chunks of 3 rows.
        Collecting all chunks and concatenating must reproduce the original full series.
        \n
        """
        from qubx.data.storages.handy import HandyStorage

        # - same ground-truth series as the non-chunked test
        idx = pd.date_range("2020-01-01 00:00", periods=24, freq="1h", name="timestamp")
        rng = np.random.default_rng(0)
        close = 10_000.0 + rng.standard_normal(24).cumsum() * 50
        full_df = pd.DataFrame(
            {
                "open": close - rng.uniform(0, 10, 24),
                "high": close + rng.uniform(0, 20, 24),
                "low": close - rng.uniform(0, 20, 24),
                "close": close,
                "volume": rng.uniform(100, 1_000, 24),
            },
            index=idx,
        )

        slice1 = full_df.loc[:"2020-01-01 15:00"]  # hours 00–15
        slice2 = full_df.loc["2020-01-01 12:00":]  # hours 12–23

        s1 = HandyStorage({"BTCUSDT": slice1}, exchange="BINANCE.UM")
        s2 = HandyStorage({"BTCUSDT": slice2}, exchange="BINANCE.UM")

        reader = MultiStorage([s1, s2]).get_reader("BINANCE.UM", "SWAP")

        chunks = list(reader.read("BTCUSDT", "ohlc(1h)", "2020-01-01 00:00", "2020-01-02 00:00", chunksize=3))  # type: ignore

        # - 24 rows / 3 per chunk → exactly 8 chunks
        assert len(chunks) == 8
        assert all(len(c) <= 3 for c in chunks)
        assert sum(len(c) for c in chunks) == 24

        # - concatenate all chunks and compare to ground truth
        result_df = pd.concat([c.to_pd() for c in chunks])
        result_df = result_df.reset_index(drop=True)

        assert result_df.index.is_unique
        pd.testing.assert_frame_equal(
            result_df,
            full_df.reset_index(drop=True),
            check_index_type=False,
        )
