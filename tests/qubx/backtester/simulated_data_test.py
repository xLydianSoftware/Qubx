from dataclasses import dataclass

import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.simulated_data import IterableSimulationData
from qubx.core.basics import DataType
from qubx.core.lookups import lookup
from qubx.data.helpers import loader
from qubx.data.readers import InMemoryDataFrameReader


def get_event_dt(i: float, base: pd.Timestamp = pd.Timestamp("2021-01-01"), offset: str = "D") -> int:
    return (base + pd.Timedelta(i, offset)).as_unit("ns").asm8.item()  # type: ignore


@dataclass
class DummyTimeEvent:
    time: int
    data: str

    @staticmethod
    def from_dict(data: dict[str | pd.Timedelta, str], start: str) -> list["DummyTimeEvent"]:
        _t0 = pd.Timestamp(start)
        return [DummyTimeEvent((_t0 + pd.Timedelta(t)).as_unit("ns").asm8.item(), d) for t, d in data.items()]

    @staticmethod
    def from_seq(start: str, n: int, ds: str, pfx: str) -> list["DummyTimeEvent"]:
        return DummyTimeEvent.from_dict({s * pd.Timedelta(ds): pfx for s in range(n + 1)}, start)

    def __repr__(self) -> str:
        return f"{pd.Timestamp(self.time, 'ns')} -> ({self.data})"


class TestSimulatedDataStuff:
    def test_iterable_simulation_data_management(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld, "ohlc_quotes": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC["4h"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC_QUOTES["4h"], s1)

        # has subscription
        assert isd.has_subscription(s3, "ohlc(4h)")

        # has subscription
        assert not isd.has_subscription(s1, "ohlc(1d)")

        # get all instruments for ANY subscription
        assert set(isd.get_instruments_for_subscription(DataType.ALL)) == set([s1, s2, s3])

        # get subs for instrument
        assert set(isd.get_subscriptions_for_instrument(s3)) == set(
            [DataType.OHLC["1h"], DataType.OHLC["4h"], DataType.OHLC["1d"]]
        )

        assert isd.get_instruments_for_subscription(DataType.OHLC["4h"]) == [s3]
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == [s1, s2, s3]

        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], s3)
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == [s1, s2]

        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], [s1, s2, s3])
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == []

        assert set(isd.get_subscriptions_for_instrument(None)) == set(
            [DataType.OHLC["4h"], DataType.OHLC_QUOTES["4h"], DataType.OHLC["1d"]]
        )

    def test_iterable_simulation_data_queue_with_warmup(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        # set warmup period
        isd.set_warmup_period(DataType.OHLC["1d"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], [s1, s2, s3])

        _n_hist = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            t = pd.Timestamp(d[2].time, "ns")
            data_type = d[1]
            is_hist = d[3]
            if is_hist:
                _n_hist += 1
            print(t, d[0], data_type, "HIST" if is_hist else "")
        assert _n_hist == 3 * 4

    def test_iterable_simulation_custom_subscription_type(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        s4 = lookup.find_symbol("BINANCE.UM", "AAVEUSDT")
        assert s1 is not None and s2 is not None and s3 is not None and s4 is not None

        # set warmup period
        isd.set_warmup_period(DataType.OHLC["1d"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], [s1, s2, s3])

        # - custom reader
        idx = pd.date_range(start="2023-06-01 00:00", end="2023-07-30", freq="1h", name="timestamp")
        c_data = pd.DataFrame({"value1": np.random.randn(len(idx)), "value2": np.random.randn(len(idx))}, index=idx)
        custom_reader_2 = InMemoryDataFrameReader({"BINANCE.UM:AAVEUSDT": c_data})

        _n_r, _got_hist, got_live = 0, False, False
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            t = pd.Timestamp(d[2].time, "ns")
            data_type = d[1]
            is_hist = d[3]
            _n_r += 1

            if _n_r == 20:
                print("- Subscribe on some shit here -")
                isd.set_typed_reader("some_my_custom_data", custom_reader_2)
                isd.set_warmup_period("some_my_custom_data", "24h")
                isd.add_instruments_for_subscription("some_my_custom_data", [s4])
            print(t, d[0], data_type, "HIST" if is_hist else "")

            if "some_my_custom_data" == data_type:
                got_live = True
                if is_hist:
                    got_hist = True

        assert got_live
        assert got_hist

    def test_iterable_simulation_data_last_historical_search(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        # set warmup period
        isd.set_warmup_period(DataType.OHLC["4h"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["4h"], [s1, s2])

        # iteration is not statred yet - so history must be empty
        assert not isd.peek_historical_data(s1, DataType.OHLC["4h"])

        _n = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            instr, data_type, t, is_hist = d[0], d[1], pd.Timestamp(d[2].time, "ns"), d[3]
            _n += 1
            logger.info(
                f"[{_n}] <y>{instr.symbol}</y> <g>{t.strftime('%H:%M:%S.%f')}</g> {d[0]} {data_type} {'<r>HIST</r>' if is_hist else ''}"
            )
            if _n == 60:
                isd.add_instruments_for_subscription(DataType.OHLC["4h"], [s3])
                h_data = isd.peek_historical_data(s3, DataType.OHLC["4h"])
                s_data = "\n".join([f"\t - {np.datetime64(v.time, 'ns')}({v})" for v in h_data])
                logger.info(f"History {len(h_data)}: \n{s_data}")
                assert len(h_data) == 25
                assert h_data[-1].time < isd._current_time  # type: ignore

    def test_iterable_simulation_data_last_historical_search_no_warmup(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["4h"], [s1, s2])

        _n = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            instr, data_type, t, is_hist = d[0], d[1], pd.Timestamp(d[2].time, unit="ns"), d[3]
            _n += 1
            logger.info(
                f"[{_n}] <y>{instr.symbol}</y> <g>{t.strftime('%H:%M:%S.%f')}</g> {d[0]} {data_type} {'<r>HIST</r>' if is_hist else ''}"
            )
            if _n == 10:
                isd.add_instruments_for_subscription(DataType.OHLC["4h"], [s3])
                h_data = isd.peek_historical_data(s3, DataType.OHLC["4h"])
                assert len(h_data) == 0
