##
# DEPRECATED tests (should be removed)
##
import numpy as np
import pandas as pd
from pytest import approx

from qubx import QubxLogConfig
from qubx.core.basics import ITimeProvider
from qubx.core.series import OHLCV, Bar, Quote, Trade
from qubx.core.utils import time_to_str
from qubx.data.helpers import TimeGuardedWrapper, loader
from qubx.data.readers import (
    STOCK_DAILY_SESSION,
    AsOhlcvSeries,
    AsPandasFrame,
    AsQuotes,
    CsvStorageDataReader,
    InMemoryDataFrameReader,
    RestoredBarsFromOHLC,
    RestoreTicksFromOHLC,
)
from qubx.pandaz.utils import srows

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)

T = lambda t: np.datetime64(t, "ns")


class TestDataReaders:
    def test_quotes_reader(self):
        r0 = CsvStorageDataReader("tests/data/csv")

        qts = r0.read("quotes")
        assert len(qts) == 500  # type: ignore

        qts = r0.read("quotes", "2017-08-24 13:09:29", transform=AsQuotes())
        assert len(qts) == 3  # type: ignore
        assert qts[-1].ask == 9.39  # type: ignore

    def test_ohlc_reader(self):
        r = CsvStorageDataReader("tests/data/csv/")
        assert "BTCUSDT_ohlcv_M1" in r.get_names()

        d0 = r.read("SPY")
        assert len(d0) == 4353

        d1 = r.read("BTCUSDT_ohlcv_M1", "2024-01-10 15:00", "2024-01-10 15:10")
        assert len(d1) == 11
        assert d1[0][0] == T("2024-01-10 15:00")
        assert d1[-1][0] == T("2024-01-10 15:10")

        # - let's read by 100 records
        _s = 0
        for x in r.read("BTCUSDT_ohlcv_M1", "2024-01-10", "2024-01-11", chunksize=100):
            _s += len(x)
            print(f"{x[0][0]} ~ {x[-1][0]} >> length is {len(x)} records")
        assert _s == 1441

        # - read as DataFrame
        d2: pd.DataFrame = r.read("SPY", transform=AsPandasFrame())  # type: ignore
        assert N(d2["close"].iloc[-1]) == 234.58999599999999

        # - read as OHLCV series
        d3 = r.read("BTCUSDT_ohlcv_M1", transform=AsOhlcvSeries("1d"))
        assert len(d3) == 46  # type: ignore

        # - DataFrame from quotes
        d4 = r.read("quotes", transform=AsPandasFrame())
        assert len(d4) == 500  # type: ignore
        assert d4["ask"].iloc[-1] == 9.39  # type: ignore

        # Let's make OHLC from quotes and make 2 Min bars out of them
        d5 = r.read("quotes", transform=AsOhlcvSeries("2Min"))
        assert len(d5) == 5

        # exact quote
        d6 = r.read("quotes", "2017-08-24 13:09:29", "2017-08-24 13:09:29")
        assert len(d6) == 1
        assert d6[0][0] == T("2017-08-24 13:09:29")

    def test_simulated_quotes_trades(self):
        r = CsvStorageDataReader("tests/data/csv/")

        tick_data = r.read(
            "BTCUSDT_ohlcv_M1", transform=RestoreTicksFromOHLC(trades=True, open_close_time_shift_secs=0)
        )
        ohlc_data = r.read("BTCUSDT_ohlcv_M1", transform=AsOhlcvSeries())

        restored_ohlc = OHLCV("restored", "1Min")
        for t in tick_data:
            if isinstance(t, Trade):
                restored_ohlc.update(t.time, t.price, t.size)
                # print(f"{time_to_str(t.time)} : {str(t)}")

            else:
                restored_ohlc.update(t.time, t.mid_price())

        assert all((restored_ohlc.pd() - ohlc_data.pd()) < 1e-10)

        d1 = r.read(
            "SPY",
            transform=RestoreTicksFromOHLC(daily_session_start_end=STOCK_DAILY_SESSION, open_close_time_shift_secs=0),
        )[:4]  # type: ignore

        assert d1[0].time == T("2000-01-03T09:30:00.100000000").item()
        assert d1[3].time == T("2000-01-03T15:59:59.900000000").item()

    def test_supported_data_id(self):
        r0 = CsvStorageDataReader("tests/data/csv/")
        assert set(r0.get_aux_data_ids()) == {"candles"}

        r1 = InMemoryDataFrameReader({"TEST1": pd.DataFrame(), "TEST2": pd.DataFrame()})
        # assert r1.get_aux_data_ids()
        assert set(r1.get_aux_data("symbols", exchange=None, dtype=None)) == {"TEST1", "TEST2"}
        try:
            r1.get_aux_data("some_arbitrary_data_id")
        except:
            assert True

    def test_aux_wrapped_loader(self):
        class _FixTimeProvider(ITimeProvider):
            def __init__(self, time: str):
                self._t_g = np.datetime64(time)

            def time(self) -> np.datetime64:
                return self._t_g

        aux_all = pd.read_csv("tests/data/csv/electricity_data.csv.gz", parse_dates=["datetime"])
        electro_aux = aux_all[
            (aux_all["stateDescription"] == "U.S. Total") & (aux_all["sectorName"] == "all sectors")
        ].set_index("datetime", drop=True)

        ldr = TimeGuardedWrapper(
            loader(
                "BINANCE.UM",
                "1h",
                electro=electro_aux,
                source="csv::tests/data/csv/",
            ),
            _FixTimeProvider("2022-06-01 05:00"),
        )
        data = ldr.get_aux_data("electro", start="2020-01-01")
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert data.index[-1] < pd.Timestamp("2022-06-01 05:00")

    def test_in_memory_loader(self):
        """
        Stress test on different loading scenarous
        """
        QubxLogConfig.set_log_level("DEBUG")

        S1 = ["ETHUSDT"]
        S2 = ["AAVEUSDT"]

        Lt = loader("BINANCE.UM", "1d", source="csv::tests/data/csv_1h/", n_jobs=1)
        L0 = loader("BINANCE.UM", "1d", source="csv::tests/data/csv_1h/", n_jobs=1)

        d10 = Lt.read(
            "BINANCE.UM:BTCUSDT", start="2023-07-20 23:00:00", stop="2023-07-27 00:00:00", transform=AsPandasFrame()
        )

        d11 = Lt.get_aux_data(
            "candles",
            symbols=S1,
            timeframe="1d",
            start="2023-07-05 00:00:00.100000",
            stop="2023-07-27 00:00:00",
        )

        d11 = Lt.get_aux_data(
            "candles",
            symbols=S2,
            timeframe="1d",
            start="2023-07-05 00:00:00.100000",
            stop="2023-07-27 00:00:00",
        )

        d10 = Lt.read(
            "BINANCE.UM:BTCUSDT",
            # start="2022-12-31 23:59:59.900000",
            start="2023-07-05 23:59:59.900000",
            stop="2023-07-27 00:00:00",
            transform=AsPandasFrame(),
        )
        d1x = Lt["BTCUSDT", "2023-07-01":"2023-07-28"]
        d2x = L0["BTCUSDT", "2023-07-01":"2023-07-28"]
        assert sum(d1x.close - d2x.close) == 0  # type: ignore

    def test_simulated_bars_from_ohlc(self):
        r = CsvStorageDataReader("tests/data/csv/")

        bars_data = r.read("BTCUSDT_ohlcv_M1", transform=RestoredBarsFromOHLC(open_close_time_shift_secs=10))
        ohlc_data = r.read("BTCUSDT_ohlcv_M1", transform=AsOhlcvSeries())

        restored_ohlc = OHLCV("restored", "1Min")
        for b in bars_data:
            if isinstance(b, Bar):
                restored_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)

        assert all((restored_ohlc.pd() - ohlc_data.pd()) < 1e-10)

    def test_inmem_chunk(self):
        r0 = CsvStorageDataReader("tests/data/csv/")
        data = r0.read("BTCUSDT_ohlcv_M1", transform=AsPandasFrame(), timeframe="1h")
        r1 = InMemoryDataFrameReader({"BTCUSDT": data})
        data2 = r1.read("BTCUSDT", transform=AsPandasFrame(), chunksize=10_000)

        res1 = []
        for c in data2:
            res1.append(c)
            assert len(c) <= 10_000

        # - read by chuncks must be the same as reading all at once
        assert all(srows(*res1) == r1.read("BTCUSDT", transform=AsPandasFrame(), chunksize=0))
