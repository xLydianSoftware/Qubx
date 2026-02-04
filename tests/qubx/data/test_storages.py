import numpy as np

from qubx.core.basics import DataType
from qubx.data.containers import RawMultiData
from qubx.data.storages.csv import CsvStorage


class TestNewStorages:
    def test_csv_storage(self):
        r = CsvStorage("tests/data/storages/csv")

        assert "BINANCE.UM" in r.get_exchanges()
        assert r.get_market_types("BINANCE.UM") == ["SWAP"]

        bnc_swap = r["BINANCE.UM", "SWAP"]
        assert bnc_swap

        assert bnc_swap.get_data_types("BTCUSDT") == ["ohlc(1h)", "quote"]
        assert sorted(bnc_swap.get_data_id("ohlc(1h)")) == sorted(
            ["ETHUSDT", "BCHUSDT", "BTCUSDT", "LTCUSDT", "AAVEUSDT"]
        )

        assert bnc_swap.get_time_range("BTCUSDT", DataType.OHLC["1h"]) == (
            np.datetime64("2023-06-01T00:00:00"),
            np.datetime64("2023-08-01T00:00:00"),
        )
        assert bnc_swap.get_time_range("BTCUSDT", DataType.QUOTE) == (
            np.datetime64("2017-08-24T13:01:12"),
            np.datetime64("2017-08-24T13:01:59"),
        )

        assert len(bnc_swap.read("BTCUSDT", DataType.OHLC["1h"]).data) > 0  # type: ignore
        assert len(bnc_swap.read("BTCUSDT", DataType.QUOTE).data) > 0  # type: ignore
        assert len(bnc_swap.read("ETHUSDT", DataType.TRADE).data) > 0  # type: ignore

    def test_csv_storage_chunk_reading(self):
        stor = CsvStorage("tests/data/storages/csv")
        reader = stor.get_reader("BINANCE.UM", "SWAP")

        raws_iters = reader.read(
            ["BTCUSDT", "ETHUSDT", "LTCUSDT"], DataType.OHLC["1h"], "2023-06-01 00:00", "+12h", chunksize=10
        )

        for w in raws_iters:
            assert isinstance(w, RawMultiData)
            assert w.get_data_ids() == ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
