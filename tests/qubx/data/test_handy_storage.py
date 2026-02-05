import numpy as np
import pandas as pd
import pytest

from qubx.core.utils import time_delta_to_str
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storages.handy import HandyStorage, _infer_dtype
from qubx.utils.time import infer_series_frequency


def _make_ohlc(symbol: str = "BTCUSDT", periods: int = 100, freq: str = "1h") -> pd.DataFrame:
    """
    Generate sample OHLC DataFrame.
    """
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq, name="timestamp")
    rng = np.random.default_rng(42)
    close = 40000 + rng.standard_normal(periods).cumsum() * 100
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, periods),
            "high": close + rng.uniform(0, 100, periods),
            "low": close - rng.uniform(0, 100, periods),
            "close": close,
            "volume": rng.uniform(100, 1000, periods),
        },
        index=idx,
    )


def _make_quotes(periods: int = 50) -> pd.DataFrame:
    """
    Generate sample quote DataFrame.
    """
    idx = pd.date_range("2024-01-01", periods=periods, freq="1s", name="timestamp")
    rng = np.random.default_rng(42)
    mid = 40000 + rng.standard_normal(periods).cumsum()
    return pd.DataFrame(
        {
            "bid": mid - 0.5,
            "ask": mid + 0.5,
            "bid_size": rng.uniform(1, 10, periods),
            "ask_size": rng.uniform(1, 10, periods),
        },
        index=idx,
    )


class TestInferDtype:
    def test_ohlc_detection(self):
        df = _make_ohlc()
        dtype = _infer_dtype(df)
        assert dtype.startswith("ohlc(")

    def test_quote_detection(self):
        df = _make_quotes()
        dtype = _infer_dtype(df)
        assert dtype == "quote"

    def test_trade_detection(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="1s", name="timestamp")
        df = pd.DataFrame({"price": range(10), "size": range(10)}, index=idx)
        assert _infer_dtype(df) == "trade"

    def test_record_fallback(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", name="timestamp")
        df = pd.DataFrame({"x": range(10), "y": range(10)}, index=idx)
        assert _infer_dtype(df) == "record"


class TestInferTimeframe:
    def test_1h(self):
        df = _make_ohlc(freq="1h")
        assert time_delta_to_str(infer_series_frequency(df).item()) == "1h"

    def test_4h(self):
        df = _make_ohlc(freq="4h")
        assert time_delta_to_str(infer_series_frequency(df).item()) == "4h"

    def test_1d(self):
        df = _make_ohlc(freq="1D")
        assert time_delta_to_str(infer_series_frequency(df).item()) == "1D"

    def test_15min(self):
        df = _make_ohlc(freq="15min")
        assert time_delta_to_str(infer_series_frequency(df).item()) == "15Min"


class TestHandyStorageForm1:
    """
    Form 1: Full instrument spec in keys.
    HandyStorage({"BINANCE.UM:SWAP:BTCUSDT": df1, "BINANCE.UM:SWAP:ETHUSDT": df2})
    """

    def test_basic(self):
        df_btc = _make_ohlc("BTCUSDT")
        df_eth = _make_ohlc("ETHUSDT")
        s = HandyStorage({"BINANCE.UM:SWAP:BTCUSDT": df_btc, "BINANCE.UM:SWAP:ETHUSDT": df_eth})

        assert s.get_exchanges() == ["BINANCE.UM"]
        assert s.get_market_types("BINANCE.UM") == ["SWAP"]

        reader = s["BINANCE.UM", "SWAP"]
        assert sorted(reader.get_data_id()) == ["BTCUSDT", "ETHUSDT"]

    def test_read_single(self):
        df_btc = _make_ohlc("BTCUSDT")
        s = HandyStorage({"BINANCE.UM:SWAP:BTCUSDT": df_btc})
        reader = s["BINANCE.UM", "SWAP"]

        raw = reader.read("BTCUSDT", "ohlc(1h)")
        assert isinstance(raw, RawData)
        assert len(raw) == 100

    def test_read_multi(self):
        df_btc = _make_ohlc("BTCUSDT")
        df_eth = _make_ohlc("ETHUSDT")
        s = HandyStorage({"BINANCE.UM:SWAP:BTCUSDT": df_btc, "BINANCE.UM:SWAP:ETHUSDT": df_eth})
        reader = s["BINANCE.UM", "SWAP"]

        multi = reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)")
        assert isinstance(multi, RawMultiData)
        assert sorted(multi.get_data_ids()) == ["BTCUSDT", "ETHUSDT"]


class TestHandyStorageForm2:
    """
    Form 2: Symbol keys + exchange parameter.
    HandyStorage({"BTCUSDT": df1, "ETHUSDT": df2}, exchange="BINANCE.UM")
    """

    def test_with_exchange_only(self):
        df_btc = _make_ohlc("BTCUSDT")
        s = HandyStorage({"BTCUSDT": df_btc}, exchange="BINANCE.UM")

        assert s.get_exchanges() == ["BINANCE.UM"]
        assert s.get_market_types("BINANCE.UM") == ["SWAP"]

    def test_with_exchange_and_market(self):
        df_btc = _make_ohlc("BTCUSDT")
        s = HandyStorage({"BTCUSDT": df_btc}, exchange="BINANCE.UM:SWAP")

        reader = s["BINANCE.UM", "SWAP"]
        assert reader.get_data_id() == ["BTCUSDT"]

    def test_no_exchange_defaults(self):
        df = _make_ohlc()
        s = HandyStorage({"BTCUSDT": df})

        assert s.get_exchanges() == ["HANDY"]
        assert s.get_market_types("HANDY") == ["DATA"]


class TestHandyStorageForm3:
    """
    Form 3: Multiple data types per symbol.
    HandyStorage({"BTCUSDT": [df_ohlc, df_quotes]}, exchange="BINANCE.UM:SWAP")
    """

    def test_multi_dtype(self):
        df_ohlc = _make_ohlc()
        df_quotes = _make_quotes()
        s = HandyStorage({"BTCUSDT": [df_ohlc, df_quotes]}, exchange="BINANCE.UM:SWAP")

        reader = s["BINANCE.UM", "SWAP"]
        dtypes = reader.get_data_types("BTCUSDT")
        assert "ohlc(1h)" in dtypes
        assert "quote" in dtypes

    def test_read_each_dtype(self):
        df_ohlc = _make_ohlc()
        df_quotes = _make_quotes()
        s = HandyStorage({"BTCUSDT": [df_ohlc, df_quotes]}, exchange="BINANCE.UM:SWAP")
        reader = s["BINANCE.UM", "SWAP"]

        raw_ohlc = reader.read("BTCUSDT", "ohlc(1h)")
        assert isinstance(raw_ohlc, RawData)
        assert len(raw_ohlc) == 100

        raw_quotes = reader.read("BTCUSDT", "quote")
        assert isinstance(raw_quotes, RawData)
        assert len(raw_quotes) == 50


class TestHandyStorageForm4:
    """
    Form 4: MultiIndex DataFrame.
    HandyStorage(multi_df, exchange="BINANCE.UM:SWAP")
    """

    def test_multiindex(self):
        df_btc = _make_ohlc("BTCUSDT", periods=50)
        df_eth = _make_ohlc("ETHUSDT", periods=50)
        df_btc["symbol"] = "BTCUSDT"
        df_eth["symbol"] = "ETHUSDT"

        combined = pd.concat([df_btc, df_eth])
        combined = combined.set_index("symbol", append=True)

        s = HandyStorage(combined, exchange="BINANCE.UM:SWAP")
        reader = s["BINANCE.UM", "SWAP"]

        assert sorted(reader.get_data_id()) == ["BTCUSDT", "ETHUSDT"]

        raw = reader.read("BTCUSDT", "ohlc(1h)")
        assert isinstance(raw, RawData)
        assert len(raw) == 50

    def test_multiindex_no_exchange(self):
        df = _make_ohlc(periods=20)
        df["symbol"] = "TEST"
        combined = df.set_index("symbol", append=True)

        s = HandyStorage(combined)
        assert s.get_exchanges() == ["HANDY"]

    def test_multiindex_rejects_flat_df(self):
        df = _make_ohlc()
        with pytest.raises(ValueError, match="MultiIndex"):
            HandyStorage(df)


class TestHandyReaderOperations:
    def test_time_range(self):
        df = _make_ohlc(periods=100, freq="1h")
        s = HandyStorage({"BTCUSDT": df}, exchange="BINANCE.UM")
        reader = s["BINANCE.UM", "SWAP"]

        t0, t1 = reader.get_time_range("BTCUSDT", "ohlc(1h)")
        assert t0 == np.datetime64("2024-01-01T00:00:00")
        assert t1 == np.datetime64("2024-01-05T03:00:00")

    def test_time_filtering(self):
        df = _make_ohlc(periods=100, freq="1h")
        s = HandyStorage({"BTCUSDT": df}, exchange="BINANCE.UM")
        reader = s["BINANCE.UM", "SWAP"]

        raw = reader.read("BTCUSDT", "ohlc(1h)", "2024-01-02", "2024-01-03")
        assert isinstance(raw, RawData)
        result_df = raw.to_pd()
        assert len(result_df) > 0
        assert len(result_df) < 100

    def test_chunked_reading(self):
        df = _make_ohlc(periods=100, freq="1h")
        s = HandyStorage({"BTCUSDT": df}, exchange="BINANCE.UM")
        reader = s["BINANCE.UM", "SWAP"]

        chunks = list(reader.read("BTCUSDT", "ohlc(1h)", chunksize=30))
        assert len(chunks) == 4
        assert all(isinstance(c, RawData) for c in chunks)

        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 100

    def test_chunked_multi_reading(self):
        df_btc = _make_ohlc(periods=50)
        df_eth = _make_ohlc(periods=50)
        s = HandyStorage({"BTCUSDT": df_btc, "ETHUSDT": df_eth}, exchange="BINANCE.UM")
        reader = s["BINANCE.UM", "SWAP"]

        chunks = list(reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)", chunksize=20))
        assert all(isinstance(c, RawMultiData) for c in chunks)

    def test_to_pd_transform(self):
        df = _make_ohlc(periods=10)
        s = HandyStorage({"BTCUSDT": df}, exchange="BINANCE.UM")
        reader = s["BINANCE.UM", "SWAP"]

        raw = reader.read("BTCUSDT", "ohlc(1h)")
        result = raw.to_pd()
        assert isinstance(result, pd.DataFrame)
        assert "open" in result.columns
        assert "close" in result.columns
        assert len(result) == 10

    def test_read_nonexistent_raises(self):
        df = _make_ohlc()
        s = HandyStorage({"BTCUSDT": df}, exchange="BINANCE.UM")
        reader = s["BINANCE.UM", "SWAP"]

        with pytest.raises(ValueError, match="No data"):
            reader.read("BTCUSDT", "quote")

    def test_get_reader_nonexistent_raises(self):
        df = _make_ohlc()
        s = HandyStorage({"BTCUSDT": df}, exchange="BINANCE.UM")

        with pytest.raises(ValueError, match="No data"):
            s["BYBIT", "SPOT"]
