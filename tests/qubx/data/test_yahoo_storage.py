"""
Yahoo storage and the parquet cache. No network: the fetcher is replaced by a stub.
"""

import pandas as pd
import pytest

from qubx.core.basics import DataType
from qubx.data.cache import ParquetCache
from qubx.data.containers import RawData
from qubx.data.registry import StorageRegistry
from qubx.data.storages.yahoo import YahooStorage, _timeframe_of, normalize


def _yf_frame(stamps, close, adjclose, ticker="SPY"):
    """
    A frame shaped the way yfinance returns one: capitalised names under a ticker level, tz-aware.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(t, unit="s", tz="America/New_York") for t in stamps])
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], [ticker]], names=["Price", "Ticker"]
    )
    data = {
        ("Open", ticker): close,
        ("High", ticker): [None if c is None else c * 1.01 for c in close],
        ("Low", ticker): [None if c is None else c * 0.99 for c in close],
        ("Close", ticker): close,
        ("Adj Close", ticker): adjclose,
        ("Volume", ticker): [1000.0] * len(close),
    }
    return pd.DataFrame(data, index=idx).reindex(columns=cols)


class StubFetcher:
    """
    Serves a fixed frame and counts how many times it was asked.
    """

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple[str, str, int, int]] = []

    def fetch(self, symbol: str, interval: str, start: int, stop: int) -> pd.DataFrame:
        self.calls.append((symbol, interval, start, stop))
        f = self.frame
        return f[(f.index >= pd.Timestamp(start, unit="s")) & (f.index <= pd.Timestamp(stop, unit="s"))].copy()


@pytest.fixture
def frame() -> pd.DataFrame:
    # - a 2-for-1 split halves the raw price; adjclose is continuous through it
    stamps = [int(pd.Timestamp(f"2024-01-{d:02d}").timestamp()) for d in range(1, 11)]
    close = [100.0] * 5 + [50.0] * 5
    adjclose = [50.0] * 10
    return normalize(_yf_frame(stamps, close, adjclose))


class TestNormalize:
    def test_columns_index_and_timezone(self, frame):
        assert list(frame.columns) == ["open", "high", "low", "close", "adjclose", "volume"]
        assert len(frame) == 10
        assert frame.index.name == "timestamp"
        assert frame.index.tz is None

    def test_drops_unfinished_bar(self):
        stamps = [int(pd.Timestamp("2024-01-01").timestamp()), int(pd.Timestamp("2024-01-02").timestamp())]
        f = normalize(_yf_frame(stamps, [100.0, None], [100.0, None]))
        assert len(f) == 1

    def test_empty_inputs_do_not_raise(self):
        assert len(normalize(None)) == 0
        assert len(normalize(pd.DataFrame())) == 0

    def test_flat_columns_are_accepted(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        flat = pd.DataFrame(
            {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0], "Volume": [10.0, 20.0]},
            index=idx,
        )
        out = normalize(flat)
        # - no adjusted column from the vendor: it falls back to the raw close
        assert out["adjclose"].tolist() == [1.0, 2.0]

    def test_timeframe_rejects_intraday_and_non_ohlc(self):
        with pytest.raises(ValueError):
            _timeframe_of("ohlc(1h)")
        with pytest.raises(ValueError):
            _timeframe_of("trade")


class TestParquetCache:
    def test_roundtrip_and_persistence(self, tmp_path, frame):
        cache = ParquetCache(tmp_path)
        raw = RawData.from_pandas("SPY", DataType.OHLC["1d"], frame)
        cache.put("ohlc(1d)", raw, "2024-01-01", "2024-01-10")
        cache.close()

        reopened = ParquetCache(tmp_path)
        got = reopened.get("ohlc(1d)", "SPY")
        assert got is not None and len(got) == 10
        assert reopened.get_stored_ids("ohlc(1d)") == ["SPY"]

    def test_covers_and_check(self, tmp_path, frame):
        cache = ParquetCache(tmp_path)
        cache.put("ohlc(1d)", RawData.from_pandas("SPY", DataType.OHLC["1d"], frame), "2024-01-01", "2024-01-10")

        assert cache.covers("ohlc(1d)", "2024-01-02", "2024-01-09")
        assert not cache.covers("ohlc(1d)", "2023-01-01", "2024-01-09")
        assert cache.check("ohlc(1d)", ["SPY"], "2024-01-02", "2024-01-09") == []
        assert cache.check("ohlc(1d)", ["SPY"], "2023-06-01", "2024-01-09") == ["SPY"]
        assert cache.check("ohlc(1d)", ["QQQ"], "2024-01-02", "2024-01-09") == ["QQQ"]

    def test_put_merges_windows_without_duplicates(self, tmp_path, frame):
        cache = ParquetCache(tmp_path)
        dt = DataType.OHLC["1d"]
        cache.put("ohlc(1d)", RawData.from_pandas("SPY", dt, frame.iloc[:6]), "2024-01-01", "2024-01-06")
        cache.put("ohlc(1d)", RawData.from_pandas("SPY", dt, frame.iloc[4:]), "2024-01-05", "2024-01-10")

        got = cache.get("ohlc(1d)", "SPY")
        assert got is not None and len(got) == 10
        assert cache.covers("ohlc(1d)", "2024-01-01", "2024-01-10")

    def test_clear_removes_data_and_the_index(self, tmp_path, frame):
        cache = ParquetCache(tmp_path)
        cache.put("ohlc(1d)", RawData.from_pandas("SPY", DataType.OHLC["1d"], frame), "2024-01-01", "2024-01-10")
        cache.clear()

        assert cache.get("ohlc(1d)", "SPY") is None
        # - a surviving index would claim coverage for data that is gone
        assert not ParquetCache(tmp_path).covers("ohlc(1d)", "2024-01-01", "2024-01-10")


class TestYahooStorage:
    def _storage(self, tmp_path, frame) -> tuple[YahooStorage, StubFetcher]:
        st = YahooStorage(str(tmp_path))
        stub = StubFetcher(frame)
        reader = st.get_reader("YAHOO", "STOCK")
        reader._inner._reader._fetcher = stub  # type: ignore[attr-defined]
        return st, stub

    def test_adjusted_is_the_default_and_scales_ohlc(self, tmp_path, frame):
        st, _ = self._storage(tmp_path, frame)
        df = st["YAHOO", "STOCK"].read("SPY", "ohlc(1d)", "2024-01-01", "2024-01-10").to_pd()

        assert "adjclose" not in df.columns
        # - the split is removed: every close is the adjusted one
        assert df["close"].nunique() == 1
        assert df["close"].iloc[0] == pytest.approx(50.0)
        # - open is scaled by the same ratio, so it tracks close
        assert df["open"].iloc[0] == pytest.approx(50.0)

    def test_unadjusted_returns_traded_prices(self, tmp_path, frame):
        st, _ = self._storage(tmp_path, frame)
        df = st["YAHOO", "STOCK"].read("SPY", "ohlc(1d)", "2024-01-01", "2024-01-10", adjusted=False).to_pd()

        assert "adjclose" not in df.columns
        assert df["close"].iloc[0] == pytest.approx(100.0)
        assert df["close"].iloc[-1] == pytest.approx(50.0)

    def test_second_read_does_not_refetch(self, tmp_path, frame):
        st, stub = self._storage(tmp_path, frame)
        reader = st["YAHOO", "STOCK"]
        reader.read("SPY", "ohlc(1d)", "2024-01-01", "2024-01-10")
        assert len(stub.calls) == 1
        reader.read("SPY", "ohlc(1d)", "2024-01-02", "2024-01-09")
        assert len(stub.calls) == 1

    def test_both_views_share_one_cache_entry(self, tmp_path, frame):
        st, stub = self._storage(tmp_path, frame)
        reader = st["YAHOO", "STOCK"]
        reader.read("SPY", "ohlc(1d)", "2024-01-01", "2024-01-10", adjusted=True)
        reader.read("SPY", "ohlc(1d)", "2024-01-01", "2024-01-10", adjusted=False)
        assert len(stub.calls) == 1

    def test_uri_resolves_through_the_registry(self, tmp_path):
        assert isinstance(StorageRegistry.get(f"yahoo::{tmp_path}"), YahooStorage)


class TestDuckDBReadsTheCache:
    def test_hive_layout_is_queryable(self, tmp_path, frame):
        duckdb = pytest.importorskip("duckdb")
        cache = ParquetCache(tmp_path)
        dt = DataType.OHLC["1d"]
        cache.put("ohlc(1d)", RawData.from_pandas("SPY", dt, frame), "2024-01-01", "2024-01-10")
        cache.put("ohlc(1d)", RawData.from_pandas("QQQ", dt, frame), "2024-01-01", "2024-01-10")
        cache.close()

        rows = duckdb.sql(
            f"SELECT data_id, count(*) AS n FROM read_parquet('{tmp_path}/**/*.parquet', hive_partitioning = 1) "
            "GROUP BY data_id ORDER BY data_id"
        ).fetchall()
        assert rows == [("QQQ", 10), ("SPY", 10)]

        cols = duckdb.sql(
            f"SELECT * FROM read_parquet('{tmp_path}/**/*.parquet', hive_partitioning = 1) LIMIT 1"
        ).columns
        # - the partition keys arrive as columns, next to the bar fields
        assert "cache_key" in cols and "data_id" in cols and "close" in cols


@pytest.mark.integration
class TestLive:
    def test_downloads_and_caches(self, tmp_path):
        pytest.importorskip("yfinance")
        st = YahooStorage(str(tmp_path))
        reader = st["YAHOO", "STOCK"]
        df = reader.read("SPY", "ohlc(1d)", "2024-01-02", "2024-03-01").to_pd()
        if not len(df):
            pytest.skip("Yahoo returned nothing — network or rate limit")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.is_monotonic_increasing
        assert (df["high"] >= df["low"]).all()

        raw = reader.read("SPY", "ohlc(1d)", "2024-01-02", "2024-03-01", adjusted=False).to_pd()
        assert (raw["close"] >= df["close"]).all()  # - dividends make the adjusted close the lower one
