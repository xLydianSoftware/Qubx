import numpy as np

from qubx.core.basics import DataType
from qubx.data.storages import CsvStorage


class TestNewStorages:
    def test_csv_storage(self):
        r = CsvStorage("tests/data/storages/csv")

        assert r.get_exchanges() == ["BINANCE.UM"]
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

        assert len(bnc_swap.read("BTCUSDT", DataType.OHLC["1h"])) > 0
        assert len(bnc_swap.read("BTCUSDT", DataType.QUOTE)) > 0
        assert len(bnc_swap.read("ETHUSDT", DataType.TRADE)) > 0

    def test_csv_storage_chunk_reading(self):
        # TODO
        pass

    def test_transformations(self):
        # TODO
        pass
