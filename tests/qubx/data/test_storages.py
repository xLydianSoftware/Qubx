import numpy as np
import pandas as pd

from qubx.core.basics import DataType, TimestampedDict
from qubx.core.series import Trade
from qubx.data.storage import RawData, RawMultiData
from qubx.data.storages.csv import CsvStorage
from qubx.data.transformers import OHLCVSeries, PandasFrame, TypedRecords


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

        assert len(bnc_swap.read("BTCUSDT", DataType.OHLC["1h"]).raw) > 0  # type: ignore
        assert len(bnc_swap.read("BTCUSDT", DataType.QUOTE).raw) > 0  # type: ignore
        assert len(bnc_swap.read("ETHUSDT", DataType.TRADE).raw) > 0  # type: ignore

    def test_raw_data_container(self):
        r1 = RawData(
            "TEST1",
            ["time", "price", "size"],
            DataType.TRADE,
            [
                [1000, 100, 0.5],
                [2000, 120, 0.3],
                [3000, 190, 1.5],
            ],
        )
        assert len(r1) == 3
        assert r1.get_time_interval() == (1000, 3000)

    def test_transformations(self):
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        r1 = RawData(
            "TEST1", ["time", "price", "size"], DataType.TRADE, [[t0 + k * dt, 100 + k, k * 0.5] for k in range(24)]
        )
        t1 = r1.transform(TypedRecords())
        assert isinstance(t1[0], Trade)
        assert t1[0].price == 100.0
        assert t1[1].size == 0.5

        t2 = r1.transform(PandasFrame())
        assert isinstance(t2, pd.DataFrame)
        assert isinstance(t2.index, pd.DatetimeIndex)
        assert t2.index[0] == pd.Timestamp("2020-01-01 00:00:00")

        # - Data type is RECORD - we expect TimestampedDict after transformation
        r2 = RawData(
            "TEST1", ["time", "price", "size"], DataType.RECORD, [[t0 + k * dt, 100 + k, k * 0.5] for k in range(24)]
        )
        t2 = r2.transform(TypedRecords())
        assert isinstance(t2[0], TimestampedDict)
        assert t2[0].data
        assert t2[0].data["price"] == 100.0
        assert t2[1]["price"] == 101.0

        # - OHLC series
        r3 = RawData(
            "TEST1",
            ["time", "open", "high", "low", "close", "volume"],
            DataType.OHLC,
            [[t0 + k * dt, 100 + k, 100 + 1.02 * k, 100 - 1.02 * k, 100 + 1.01 * k, 100 * (k + 1)] for k in range(24)],
        )
        t3 = r3.transform(OHLCVSeries())
        assert len(r3) == 24
        assert t3["volume"][-1] == 100.0
        assert t3["open"][-1] == 100.0

        # - orderbook
        r4 = RawData(
            "TEST1",
            [
                "timestamp",
                "symbol",
                "top_bid",
                "top_ask",
                "tick_size",
                "bid_100",
                "ask_1",
                "bid_1",
                "ask_100",
                "ask_30",
            ],
            DataType.ORDERBOOK,
            [
                [t0 + k * dt, "BTCUSDT", 90_000.0 + k, 90_000.1 + k, 0.05, 100.0, 20.0, 10.0, 200.0, 130.0]
                for k in range(24)
            ],
        )
        t4 = r4.transform(TypedRecords())
        assert all(t4[0].bids == np.array([10.0, 100.0]))
        assert all(t4[0].asks == np.array([20.0, 130.0, 200.0]))

    def test_raw_multi_data_container(self):
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        r1 = RawData(
            "BTCUSDT",
            ["time", "open", "high", "low", "close", "volume"],
            DataType.OHLC,
            [[t0 + k * dt, 100 + k, 100 + 1.02 * k, 100 - 1.02 * k, 100 + 1.01 * k, 100 * (k + 1)] for k in range(24)],
        )
        r2 = RawData(
            "ETHUSDT",
            ["time", "open", "high", "low", "close", "volume"],
            DataType.OHLC,
            [[t0 + k * dt, 10 + k, 10 + 1.02 * k, 10 - 1.02 * k, 10 + 1.01 * k, 200 * (k + 1)] for k in range(24)],
        )

        rmd = RawMultiData([r1, r2])
        f1 = rmd.transform(PandasFrame(False))
        f2 = rmd.transform(PandasFrame(True))
        assert all(f1.columns.get_level_values(0).unique().to_numpy() == ["BTCUSDT", "ETHUSDT"])
        assert all(f2.index.get_level_values(1).unique().to_numpy() == ["BTCUSDT", "ETHUSDT"])

    def test_csv_storage_chunk_reading(self):
        # TODO
        pass
