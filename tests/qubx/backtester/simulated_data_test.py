from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from qubx import QubxLogConfig, logger
from qubx.backtester.simulated_data import IterableSimulationData
from qubx.core.basics import DataType
from qubx.core.lookups import lookup
from qubx.data.helpers import loader
from qubx.data.readers import DataReader, DataTransformer, InMemoryDataFrameReader
from qubx.utils.time import handle_start_stop


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


class _DummyTestRisingOHLCDataReader(DataReader):
    """
    Test data reader that generates incrementing OHLC values.
    Each symbol has a different base value (A=100, B=200, C=300, etc.)
    so we can easily identify which instrument's data we're looking at.
    """

    def __init__(self, instruments: list[str], start: str, end: str, freq: str = "1h"):
        self.instruments = instruments
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.freq = freq
        self._data = {}

        # - Assign different base values per instrument
        for idx, instr in enumerate(instruments):
            base = 100.0 + (idx * 100.0)  # - A=100, B=200, C=300, etc.
            self._data[instr] = self._generate_rising_ohlc(base)

    def _generate_rising_ohlc(self, base: float) -> pd.DataFrame:
        """
        Generate OHLC data where values increment by 1 for each bar.
        """
        idx = pd.date_range(start=self.start, end=self.end, freq=self.freq, name="timestamp")
        n = len(idx)

        # - Use explicit array construction to avoid length mismatch
        data = pd.DataFrame(
            {
                "open": base + np.arange(n, dtype=float),
                "high": base + 0.3 + np.arange(n, dtype=float),
                "low": base - 0.2 + np.arange(n, dtype=float),
                "close": base + 0.1 + np.arange(n, dtype=float),
                "volume": np.ones(n) * 1000.0,
            },
            index=idx,
        )

        return data

    def get_names(self, **kwargs) -> list[str]:
        return self.instruments

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 0,
        **kwargs,
    ) -> Iterator | list:
        start, stop = handle_start_stop(start, stop)  # type: ignore

        if data_id not in self._data:
            raise ValueError(f"No data found for {data_id}")

        _stored_data = self._data[data_id]
        _sliced_data = _stored_data.loc[start:stop].copy().reset_index()

        columns = list(_sliced_data.columns)
        values = _sliced_data.values

        transform.start_transform(data_id, columns, start=start, stop=stop)
        transform.process_data(values)
        result = transform.collect()

        # - Return an iterator, not a list
        return iter([result])

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        return self.instruments


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
            t = pd.Timestamp(d[2].time, unit="ns")
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

    def test_iterable_simulation_data_remove_readd_instrument_peek_data(self):
        """
        Test that peek_historical_data returns correct data when an instrument is removed and re-added.

        Scenario:
            1. Subscribe to instruments A, B, C
            2. Iterate through data
            3. Remove instrument A at iteration 20
            4. Continue iterating (A not receiving data)
            5. Re-add instrument A at iteration 40
            6. Verify peek_historical_data returns data from CURRENT time, not stale data from before removal
        """
        QubxLogConfig.set_log_level("INFO")

        # - Create test data reader with rising OHLC values (A=100+, B=200+, C=300+)
        test_reader = _DummyTestRisingOHLCDataReader(
            instruments=["BINANCE.UM:BTCUSDT", "BINANCE.UM:ETHUSDT", "BINANCE.UM:LTCUSDT"],
            start="2023-07-01",
            end="2023-07-03",
            freq="1h",
        )

        isd = IterableSimulationData({"ohlc": test_reader}, open_close_time_indent_secs=1)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")  # - A (base=100)
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")  # - B (base=200)
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")  # - C (base=300)
        assert s1 is not None and s2 is not None and s3 is not None

        # - Subscribe to all 3 instruments initially
        isd.set_warmup_period(DataType.OHLC["1h"], "2h")
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2, s3])

        _n = 0
        _last_s1_close_before_removal = None
        _time_when_removed = None
        _time_when_readded = None

        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            instr, data_type, event, is_hist = d[0], d[1], d[2], d[3]
            _n += 1

            t = pd.Timestamp(event.time, unit="ns")
            close_price = event.close if hasattr(event, "close") else None

            logger.info(
                f"[{_n}] <y>{instr.symbol}</y> <g>{t.strftime('%Y-%m-%d %H:%M')}</g> "
                f"close={close_price:.1f} {'<r>HIST</r>' if is_hist else ''}"
            )

            # - Track last close price for s1 before removal
            if instr == s1 and _n < 20:
                _last_s1_close_before_removal = close_price

            # - Remove instrument A (s1) at iteration 20
            if _n == 20:
                logger.info(f"<r>>>> Removing instrument {s1.symbol} at iteration {_n}</r>")
                _time_when_removed = isd._current_time
                isd.remove_instruments_from_subscription(DataType.OHLC["1h"], s1)

            # - Re-add instrument A (s1) at iteration 40
            if _n == 40:
                logger.info(f"<g>>>> Re-adding instrument {s1.symbol} at iteration {_n}</g>")
                _time_when_readded = isd._current_time
                isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1])

                # - Peek historical data for s1
                h_data = isd.peek_historical_data(s1, DataType.OHLC["1h"])

                logger.info(f"Historical data length: {len(h_data)}")
                if h_data:
                    first_bar = h_data[0]
                    last_bar = h_data[-1]
                    logger.info(f"First bar: {pd.Timestamp(first_bar.time, unit='ns')} close={first_bar.close:.1f}")
                    logger.info(f"Last bar: {pd.Timestamp(last_bar.time, unit='ns')} close={last_bar.close:.1f}")

                    # - CRITICAL CHECK: The last bar's close price should be GREATER than
                    # - the last close price we saw before removal, because time has passed
                    # - If we're getting stale data, last_bar.close would be <= _last_s1_close_before_removal
                    logger.info(f"Last close before removal: {_last_s1_close_before_removal:.1f}")
                    logger.info(f"Last close in peek_historical_data: {last_bar.close:.1f}")  # type: ignore

                    # - This assertion will FAIL if peek_historical_data returns stale data
                    assert last_bar.close > _last_s1_close_before_removal, (  # type: ignore
                        f"peek_historical_data returned STALE data! "
                        f"Expected close > {_last_s1_close_before_removal:.1f}, "
                        f"but got {last_bar.close:.1f}"  # type: ignore
                    )

                    # - Also verify the last bar is before current time
                    assert last_bar.time < _time_when_readded, (  # type: ignore
                        f"peek_historical_data returned future data! "
                        f"Last bar time {last_bar.time} >= current time {_time_when_readded}"
                    )

                logger.info("<g>✓ peek_historical_data returned CORRECT data (not stale)</g>")
                break  # - Test complete


class TestSimulatedDataStorages:
    """
    Analog tests for IterableSimulationDataV2 using CsvStorage (IReader/IStorage).
    Each test mirrors a TestSimulatedDataStuff test and must produce same results.
    """

    CSV_STORAGE_PATH = "tests/data/storages/csv"

    @staticmethod
    def _make_csv_reader():
        from qubx.data.storages.csv import CsvStorage

        storage = CsvStorage(TestSimulatedDataStorages.CSV_STORAGE_PATH)
        return storage["BINANCE.UM", "SWAP"]

    @staticmethod
    def _make_v2(readers: dict, **kwargs):
        from qubx.backtester.simulated_data_new import IterableSimulationDataV2

        return IterableSimulationDataV2(readers=readers, **kwargs)

    def test_v2_data_management(self):
        """
        Mirrors test_iterable_simulation_data_management.
        Tests subscribe/unsubscribe/query methods produce same results.
        """
        from qubx.backtester.simulated_data_new import IterableSimulationDataV2

        reader = self._make_csv_reader()
        isd = IterableSimulationDataV2(
            readers={"ohlc": reader, "ohlc_quotes": reader},
            open_close_time_indent_secs=300,
        )

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC["4h"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC["1d"], s3)
        isd.add_instruments_for_subscription(DataType.OHLC_QUOTES["4h"], s1)

        # - has subscription
        assert isd.has_subscription(s3, "ohlc(4h)")

        # - has subscription (negative)
        assert not isd.has_subscription(s1, "ohlc(1d)")

        # - get all instruments for ANY subscription
        all_instr = isd.get_instruments_for_subscription(DataType.ALL)
        assert s1 in all_instr and s2 in all_instr and s3 in all_instr

        # - get subs for instrument
        assert set(isd.get_subscriptions_for_instrument(s3)) == set(
            [DataType.OHLC["1h"], DataType.OHLC["4h"], DataType.OHLC["1d"]]
        )

        assert isd.get_instruments_for_subscription(DataType.OHLC["4h"]) == [s3]

        instr_1h = isd.get_instruments_for_subscription(DataType.OHLC["1h"])
        assert s1 in instr_1h and s2 in instr_1h and s3 in instr_1h

        # - remove s3 from 1h
        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], s3)
        instr_1h_after = isd.get_instruments_for_subscription(DataType.OHLC["1h"])
        assert s3 not in instr_1h_after
        assert s1 in instr_1h_after and s2 in instr_1h_after

        # - remove all from 1h
        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], [s1, s2, s3])
        assert isd.get_instruments_for_subscription(DataType.OHLC["1h"]) == []

        # - remaining subscriptions
        assert set(isd.get_subscriptions_for_instrument(None)) == set(
            [DataType.OHLC["4h"], DataType.OHLC_QUOTES["4h"], DataType.OHLC["1d"]]
        )

    def test_v2_queue_with_warmup(self):
        """
        Mirrors test_iterable_simulation_data_queue_with_warmup.
        Uses 1h timeframe (csv storage has 1h data).
        24h warmup = 24 bars * 4 emulated events = 96 historical events per symbol.
        3 symbols * 96 = 288 total historical events.
        """
        reader = self._make_csv_reader()
        isd = self._make_v2({"ohlc": reader}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        # - set warmup period
        isd.set_warmup_period(DataType.OHLC["1h"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2, s3])

        _n_hist = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            is_hist = d[3]
            if is_hist:
                _n_hist += 1

        # - 3 symbols * 96 emulated events in 24h warmup (24 bars * 4 events/bar)
        assert _n_hist == 3 * 96

    def test_v2_historical_search_with_warmup(self):
        """
        Mirrors test_iterable_simulation_data_last_historical_search.
        Uses 1h timeframe. At iteration 60, dynamically add s3 with warmup.
        Must get len(h_data) == 96 (24h warmup * 4 events/bar) and last time < current_time.
        """
        reader = self._make_csv_reader()
        isd = self._make_v2({"ohlc": reader}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        # - set warmup period
        isd.set_warmup_period(DataType.OHLC["1h"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        # - iteration not started yet — history must be empty
        assert not isd.peek_historical_data(s1, DataType.OHLC["1h"])

        _n = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            _n += 1
            if _n == 60:
                isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s3])
                h_data = isd.peek_historical_data(s3, DataType.OHLC["1h"])
                assert len(h_data) == 96
                assert h_data[-1].time < isd._current_time
                break

    def test_v2_historical_search_no_warmup(self):
        """
        Mirrors test_iterable_simulation_data_last_historical_search_no_warmup.
        Uses 1h timeframe. At iteration 10, dynamically add s3 without warmup.
        peek must return empty.
        """
        reader = self._make_csv_reader()
        isd = self._make_v2({"ohlc": reader}, open_close_time_indent_secs=300)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")
        assert s1 is not None and s2 is not None and s3 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        _n = 0
        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            _n += 1
            if _n == 10:
                isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s3])
                h_data = isd.peek_historical_data(s3, DataType.OHLC["1h"])
                assert len(h_data) == 0
                break

    def test_v2_remove_readd_instrument_peek_data(self):
        """
        Mirrors test_iterable_simulation_data_remove_readd_instrument_peek_data.
        Remove instrument A at iteration 20, re-add at iteration 40.
        peek_historical_data must return FRESH data (not stale from before removal).
        """
        from qubx.backtester.simulated_data_new import IterableSimulationDataV2

        # - Build HandyStorage with rising OHLC (same as _DummyTestRisingOHLCDataReader)
        from qubx.data.storages.handy import HandyStorage

        instruments_ids = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
        idx = pd.date_range(start="2023-07-01", end="2023-07-03", freq="1h", name="timestamp")
        n = len(idx)
        data = {}
        for i, sym in enumerate(instruments_ids):
            base = 100.0 + (i * 100.0)
            data[sym] = pd.DataFrame(
                {
                    "open": base + np.arange(n, dtype=float),
                    "high": base + 0.3 + np.arange(n, dtype=float),
                    "low": base - 0.2 + np.arange(n, dtype=float),
                    "close": base + 0.1 + np.arange(n, dtype=float),
                    "volume": np.ones(n) * 1000.0,
                },
                index=idx,
            )

        storage = HandyStorage(data, exchange="BINANCE.UM:SWAP")
        reader = storage["BINANCE.UM", "SWAP"]

        isd = IterableSimulationDataV2(readers={"ohlc": reader}, open_close_time_indent_secs=1)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")  # - A (base=100)
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")  # - B (base=200)
        s3 = lookup.find_symbol("BINANCE.UM", "LTCUSDT")  # - C (base=300)
        assert s1 is not None and s2 is not None and s3 is not None

        # - Subscribe to all 3 instruments initially
        isd.set_warmup_period(DataType.OHLC["1h"], "2h")
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2, s3])

        _n = 0
        _last_s1_close_before_removal = None
        _time_when_readded = None

        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            instr, data_type, event, is_hist = d[0], d[1], d[2], d[3]
            _n += 1

            close_price = event.close if hasattr(event, "close") else None

            # - Track last close price for s1 before removal
            if instr == s1 and _n < 20:
                _last_s1_close_before_removal = close_price

            # - Remove instrument A (s1) at iteration 20
            if _n == 20:
                isd.remove_instruments_from_subscription(DataType.OHLC["1h"], s1)

            # - Re-add instrument A (s1) at iteration 40
            if _n == 40:
                _time_when_readded = isd._current_time
                isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1])

                # - Peek historical data for s1
                h_data = isd.peek_historical_data(s1, DataType.OHLC["1h"])

                if h_data:
                    last_bar = h_data[-1]

                    # - CRITICAL: last bar's close should be GREATER than before removal
                    assert last_bar.close > _last_s1_close_before_removal, (
                        f"peek_historical_data returned STALE data! "
                        f"Expected close > {_last_s1_close_before_removal:.1f}, "
                        f"but got {last_bar.close:.1f}"
                    )

                    # - Also verify the last bar is before current time
                    assert last_bar.time < _time_when_readded, (
                        f"peek_historical_data returned future data! "
                        f"Last bar time {last_bar.time} >= current time {_time_when_readded}"
                    )

                break  # - Test complete

    def test_v2_dual_subscription_mixed_warmup(self):
        """
        Two subscriptions ohlc(1h) + ohlc(4h) with different warmup configs:
          - ohlc(1h) has 24h warmup → produces historical events before sim start
          - ohlc(4h) has NO warmup  → all events are live (no historical)

        Verifies:
          1. Both subscriptions produce events
          2. 1h produces historical events (from warmup), 4h does not
          3. Events interleave in chronological order
          4. Separate pumps are created (different access keys)
          5. All instruments appear in both subscriptions
        """
        from qubx.backtester.simulated_data_new import IterableSimulationDataV2
        from qubx.data.storages.handy import HandyStorage

        # - Build 1h and 4h OHLC data for 2 symbols
        symbols = ["BTCUSDT", "ETHUSDT"]
        idx_1h = pd.date_range(start="2023-07-01", end="2023-07-04", freq="1h", name="timestamp")
        idx_4h = pd.date_range(start="2023-07-01", end="2023-07-04", freq="4h", name="timestamp")

        data = {}
        for i, sym in enumerate(symbols):
            base = 100.0 + (i * 100.0)
            n1 = len(idx_1h)
            n4 = len(idx_4h)

            df_1h = pd.DataFrame(
                {
                    "open": base + np.arange(n1, dtype=float),
                    "high": base + 0.5 + np.arange(n1, dtype=float),
                    "low": base - 0.3 + np.arange(n1, dtype=float),
                    "close": base + 0.1 + np.arange(n1, dtype=float),
                    "volume": np.ones(n1) * 1000.0,
                },
                index=idx_1h,
            )
            df_4h = pd.DataFrame(
                {
                    "open": base + np.arange(n4, dtype=float) * 4,
                    "high": base + 2.0 + np.arange(n4, dtype=float) * 4,
                    "low": base - 1.0 + np.arange(n4, dtype=float) * 4,
                    "close": base + 0.5 + np.arange(n4, dtype=float) * 4,
                    "volume": np.ones(n4) * 5000.0,
                },
                index=idx_4h,
            )
            # - Form 3: list of DataFrames per symbol (both timeframes)
            data[sym] = [df_1h, df_4h]

        storage = HandyStorage(data, exchange="BINANCE.UM:SWAP")
        reader = storage["BINANCE.UM", "SWAP"]

        isd = IterableSimulationDataV2(readers={"ohlc": reader}, open_close_time_indent_secs=1)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        assert s1 is not None and s2 is not None

        # - 1h with 24h warmup
        isd.set_warmup_period(DataType.OHLC["1h"], "24h")
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        # - 4h with NO warmup
        isd.add_instruments_for_subscription(DataType.OHLC["4h"], [s1, s2])

        # - verify separate pumps exist
        assert len(isd._pumps) == 2, f"Expected 2 pumps (1h + 4h), got {len(isd._pumps)}"

        # - verify subscriptions are tracked correctly
        assert isd.has_subscription(s1, DataType.OHLC["1h"])
        assert isd.has_subscription(s1, DataType.OHLC["4h"])
        assert isd.has_subscription(s2, DataType.OHLC["1h"])
        assert isd.has_subscription(s2, DataType.OHLC["4h"])

        # - counters: EmulatedBarSequence emits 4 bars per OHLC row:
        # -   bars 1-3 have volume=0, bar 4 (final) has real volume.
        # -   1h final bars → vol=1000, 4h final bars → vol=5000.
        hist_1h_final, hist_4h_final = 0, 0
        live_1h_final, live_4h_final = 0, 0
        total_hist, total_live = 0, 0
        prev_time = 0
        monotonic_violations = 0

        for d in isd.create_iterable("2023-07-02", "2023-07-03"):
            instr, data_type, event, is_hist = d[0], d[1], d[2], d[3]

            # - all events produce "ohlc" data type (both are EmulatedBarSequence)
            assert data_type == "ohlc", f"Expected 'ohlc', got '{data_type}'"

            # - check monotonic time (non-decreasing)
            if event.time < prev_time:
                monotonic_violations += 1
            prev_time = event.time

            if is_hist:
                total_hist += 1
            else:
                total_live += 1

            # - classify by final-bar volume (only 4th bar in each group has vol > 0)
            vol = event.volume
            if vol == 1000.0:
                if is_hist:
                    hist_1h_final += 1
                else:
                    live_1h_final += 1
            elif vol == 5000.0:
                if is_hist:
                    hist_4h_final += 1
                else:
                    live_4h_final += 1

        # - 1h warmup produced historical final bars (vol=1000)
        assert hist_1h_final > 0, f"1h should have historical final bars (warmup=24h), got {hist_1h_final}"

        # - 4h has NO warmup → zero historical final bars
        assert hist_4h_final == 0, f"4h should have NO historical final bars (no warmup), got {hist_4h_final}"

        # - both produce live final bars
        assert live_1h_final > 0, f"1h should produce live final bars, got {live_1h_final}"
        assert live_4h_final > 0, f"4h should produce live final bars, got {live_4h_final}"

        # - 1h produces more final bars than 4h (4x more bars per day)
        assert live_1h_final > live_4h_final, (
            f"1h should produce more live final bars than 4h: {live_1h_final} vs {live_4h_final}"
        )

        # - total live should exceed historical (1-day sim with 24h warmup + 4h contributing)
        assert total_live > total_hist, f"Total live ({total_live}) should exceed historical ({total_hist})"

        # - time ordering must be monotonically non-decreasing
        assert monotonic_violations == 0, f"Events not in chronological order: {monotonic_violations} violations"

        logger.info(
            f"Dual subscription: hist_1h_final={hist_1h_final}, hist_4h_final={hist_4h_final}, "
            f"live_1h_final={live_1h_final}, live_4h_final={live_4h_final}, "
            f"total_hist={total_hist}, total_live={total_live}"
        )

    def test_v2_add_remove_add_produces_events(self):
        """
        Verify that add → remove → add cycle works correctly:
          1. Subscribe A, B — iterate, both produce events
          2. Remove A — iterate, only B produces events
          3. Re-add A — iterate, BOTH produce events again (A not stuck)
          4. A's events after re-add have timestamps > removal time (fresh data, not stale)
        """
        from qubx.backtester.simulated_data_new import IterableSimulationDataV2
        from qubx.data.storages.handy import HandyStorage

        # - Build rising OHLC data for 2 symbols
        instruments_ids = ["BTCUSDT", "ETHUSDT"]
        idx = pd.date_range(start="2023-07-01", end="2023-07-03", freq="1h", name="timestamp")
        n = len(idx)
        data = {}
        for i, sym in enumerate(instruments_ids):
            base = 100.0 + (i * 100.0)
            data[sym] = pd.DataFrame(
                {
                    "open": base + np.arange(n, dtype=float),
                    "high": base + 0.3 + np.arange(n, dtype=float),
                    "low": base - 0.2 + np.arange(n, dtype=float),
                    "close": base + 0.1 + np.arange(n, dtype=float),
                    "volume": np.ones(n) * 1000.0,
                },
                index=idx,
            )

        storage = HandyStorage(data, exchange="BINANCE.UM:SWAP")
        reader = storage["BINANCE.UM", "SWAP"]

        isd = IterableSimulationDataV2(readers={"ohlc": reader}, open_close_time_indent_secs=1)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")  # - A
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")  # - B
        assert s1 is not None and s2 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        _n = 0
        _phase = "both"  # - both | only_b | both_again
        _time_at_remove = None

        # - counters per phase
        phase1_syms: set[str] = set()
        phase2_syms: set[str] = set()
        phase3_syms: set[str] = set()
        phase3_s1_times: list[int] = []

        for d in isd.create_iterable("2023-07-01", "2023-07-02"):
            instr, data_type, event, is_hist = d[0], d[1], d[2], d[3]
            _n += 1

            if _phase == "both":
                phase1_syms.add(instr.symbol)
                if _n == 20:
                    _time_at_remove = isd._current_time
                    isd.remove_instruments_from_subscription(DataType.OHLC["1h"], s1)
                    _phase = "only_b"

            elif _phase == "only_b":
                phase2_syms.add(instr.symbol)
                if _n == 40:
                    isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1])
                    _phase = "both_again"

            elif _phase == "both_again":
                phase3_syms.add(instr.symbol)
                if instr == s1:
                    phase3_s1_times.append(event.time)
                if _n == 80:
                    break

        # - Phase 1: both instruments produced events
        assert "BTCUSDT" in phase1_syms, "A should produce events in phase 1"
        assert "ETHUSDT" in phase1_syms, "B should produce events in phase 1"

        # - Phase 2: only B produces events (A was removed)
        assert "BTCUSDT" not in phase2_syms, "A should NOT produce events after removal"
        assert "ETHUSDT" in phase2_syms, "B should produce events after A's removal"

        # - Phase 3: both produce events again (A re-added)
        assert "BTCUSDT" in phase3_syms, "A should produce events after re-add"
        assert "ETHUSDT" in phase3_syms, "B should produce events after A's re-add"

        # - A's events after re-add must be fresh (timestamps > removal time)
        assert phase3_s1_times, "A should have produced events in phase 3"
        for t in phase3_s1_times:
            assert t >= _time_at_remove, (
                f"A produced stale event after re-add: event time {t} < removal time {_time_at_remove}"
            )


class TestBatchPerformance:
    """
    Performance comparison: old IterableSimulationData (per-symbol reads)
    vs new IterableSimulationDataV2 (batched reads via IReader/IStorage).

    Scenario:
        1. Subscribe 50 symbols, iterate ~500 events
        2. Mid-sim: subscribe another 50 symbols, iterate ~500 more
        3. Mid-sim: unsubscribe 25 from batch 1 + 25 from batch 2, iterate to end

    Uses QuestDB (mqdb::quantlab) for real batched SQL comparison.
    Marked as integration tests — require running QuestDB instance.
    """

    # - date range: ~2 weeks of 1h data
    START = "2025-01-01"
    STOP = "2025-01-30"
    N_BATCH_1 = 100
    N_BATCH_2 = 100
    N_REMOVE = 50  # - remove from each batch

    # - cached validated symbols (class-level to avoid re-reading across calls)
    _validated_symbols: list[str] | None = None

    @classmethod
    def _get_validated_symbols(cls) -> list[str]:
        """
        Get symbols that actually have data in the test date range.
        Caches result at class level to avoid redundant reads.
        """
        if cls._validated_symbols is not None:
            return cls._validated_symbols

        from qubx.data.containers import RawData, RawMultiData
        from qubx.data.registry import StorageRegistry

        storage = StorageRegistry.get("qdb::quantlab")
        reader = storage["BINANCE.UM", "SWAP"]
        all_ids = sorted(reader.get_data_id("ohlc(1h)"))

        # - quick validation: read 1-day window to find symbols with actual data
        valid_ids: set[str] = set()
        for chunk in reader.read(all_ids, "ohlc(1h)", cls.START, cls.STOP, chunksize=5000):
            if isinstance(chunk, RawMultiData):
                for raw in chunk:
                    valid_ids.add(raw.data_id)
            elif isinstance(chunk, RawData):
                valid_ids.add(chunk.data_id)

        cls._validated_symbols = sorted(valid_ids)
        logger.info(
            f"[TestBatchPerformance] :: Validated {len(cls._validated_symbols)}/{len(all_ids)} "
            f"symbols with data in {cls.START} → {cls.STOP}"
        )
        return cls._validated_symbols

    @classmethod
    def _get_symbols_and_instruments(cls, n: int, offset: int = 0) -> tuple[list[str], list]:
        """
        Get n valid symbols from QuestDB starting at offset.
        Only returns symbols that have actual data in the test date range.
        Returns (symbol_list, instrument_list).
        """
        valid_ids = cls._get_validated_symbols()

        symbols, instruments = [], []
        for sym in valid_ids[offset:]:
            instr = lookup.find_symbol("BINANCE.UM", sym)
            if instr:
                symbols.append(sym)
                instruments.append(instr)
            if len(symbols) >= n:
                break

        return symbols, instruments

    @staticmethod
    def _run_old_approach(
        batch1_instruments,
        batch2_instruments,
        remove_from_1,
        remove_from_2,
        start,
        stop,
    ) -> dict:
        """
        Run iteration using old IterableSimulationData with per-symbol DataReader (loader).
        """
        import time

        ld = loader("BINANCE.UM", "1h", source="mqdb::quantlab", n_jobs=1)
        isd = IterableSimulationData({"ohlc": ld}, open_close_time_indent_secs=1)

        # - phase 1: subscribe batch 1
        t0 = time.perf_counter()
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], batch1_instruments)
        t_sub1 = time.perf_counter() - t0

        isd.create_iterable(start, stop)

        # - iterate phase 1
        t0 = time.perf_counter()
        count, phase1_count = 0, 0
        for instr, dtype, event, is_hist in isd:
            count += 1
            if count == 500:
                break
        phase1_count = count
        t_iter1 = time.perf_counter() - t0

        # - phase 2: subscribe batch 2
        t0 = time.perf_counter()
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], batch2_instruments)
        t_sub2 = time.perf_counter() - t0

        for instr, dtype, event, is_hist in isd:
            count += 1
            if count == 1000:
                break
        phase2_count = count - phase1_count
        t_iter2 = time.perf_counter() - t0

        # - phase 3: unsubscribe 25+25
        t0 = time.perf_counter()
        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], remove_from_1)
        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], remove_from_2)
        t_unsub = time.perf_counter() - t0

        for instr, dtype, event, is_hist in isd:
            count += 1
            if count == 1500:
                break
        phase3_count = count - phase1_count - phase2_count
        t_iter3 = time.perf_counter() - t0

        return {
            "total_events": count,
            "phase1_events": phase1_count,
            "phase2_events": phase2_count,
            "phase3_events": phase3_count,
            "t_subscribe_batch1": t_sub1,
            "t_iterate_phase1": t_iter1,
            "t_subscribe_batch2": t_sub2,
            "t_iterate_phase2": t_iter2,
            "t_unsubscribe": t_unsub,
            "t_iterate_phase3": t_iter3,
        }

    @staticmethod
    def _run_new_approach(
        batch1_instruments,
        batch2_instruments,
        remove_from_1,
        remove_from_2,
        start,
        stop,
    ) -> dict:
        """
        Run iteration using new IterableSimulationDataV2 with batched IReader reads.
        """
        import time

        from qubx.backtester.simulated_data_new import IterableSimulationDataV2
        from qubx.data.registry import StorageRegistry

        storage = StorageRegistry.get("qdb::quantlab")
        reader = storage["BINANCE.UM", "SWAP"]

        isd = IterableSimulationDataV2(readers={"ohlc": reader}, open_close_time_indent_secs=1)

        # - phase 1: subscribe batch 1
        t0 = time.perf_counter()
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], batch1_instruments)
        t_sub1 = time.perf_counter() - t0

        isd.create_iterable(start, stop)

        # - iterate phase 1
        t0 = time.perf_counter()
        count, phase1_count = 0, 0
        for instr, dtype, event, is_hist in isd:
            count += 1
            if count == 500:
                break
        phase1_count = count
        t_iter1 = time.perf_counter() - t0

        # - phase 2: subscribe batch 2
        t0 = time.perf_counter()
        isd.add_instruments_for_subscription(DataType.OHLC["1h"], batch2_instruments)
        t_sub2 = time.perf_counter() - t0

        for instr, dtype, event, is_hist in isd:
            count += 1
            if count == 1000:
                break
        phase2_count = count - phase1_count
        t_iter2 = time.perf_counter() - t0

        # - phase 3: unsubscribe 25+25
        t0 = time.perf_counter()
        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], remove_from_1)
        isd.remove_instruments_from_subscription(DataType.OHLC["1h"], remove_from_2)
        t_unsub = time.perf_counter() - t0

        for instr, dtype, event, is_hist in isd:
            count += 1
            if count == 1500:
                break
        phase3_count = count - phase1_count - phase2_count
        t_iter3 = time.perf_counter() - t0

        return {
            "total_events": count,
            "phase1_events": phase1_count,
            "phase2_events": phase2_count,
            "phase3_events": phase3_count,
            "t_subscribe_batch1": t_sub1,
            "t_iterate_phase1": t_iter1,
            "t_subscribe_batch2": t_sub2,
            "t_iterate_phase2": t_iter2,
            "t_unsubscribe": t_unsub,
            "t_iterate_phase3": t_iter3,
        }

    @pytest.mark.integration
    def test_v1_vs_v2_batch_performance(self):
        """
        Compare old (per-symbol reads) vs new (batched reads) on QuestDB.
        Scenario: 50 symbols → +50 → -25-25, iterate 500 events per phase.
        """
        # - prepare instruments
        _, batch1 = self._get_symbols_and_instruments(self.N_BATCH_1, offset=0)
        _, batch2 = self._get_symbols_and_instruments(self.N_BATCH_2, offset=self.N_BATCH_1)

        remove_from_1 = batch1[: self.N_REMOVE]
        remove_from_2 = batch2[: self.N_REMOVE]

        print(f"Batch 1: {len(batch1)} instruments")
        print(f"Batch 2: {len(batch2)} instruments")
        print(f"Remove: {len(remove_from_1)} + {len(remove_from_2)}")

        # - run old approach
        print("\n--- Running OLD approach (per-symbol reads) ---")
        old_results = self._run_old_approach(batch1, batch2, remove_from_1, remove_from_2, self.START, self.STOP)

        # - run new approach
        print("\n--- Running NEW approach (batched IReader) ---")
        new_results = self._run_new_approach(batch1, batch2, remove_from_1, remove_from_2, self.START, self.STOP)

        # - report
        old_total = (
            old_results["t_subscribe_batch1"]
            + old_results["t_iterate_phase1"]
            + old_results["t_subscribe_batch2"]
            + old_results["t_iterate_phase2"]
            + old_results["t_unsubscribe"]
            + old_results["t_iterate_phase3"]
        )
        new_total = (
            new_results["t_subscribe_batch1"]
            + new_results["t_iterate_phase1"]
            + new_results["t_subscribe_batch2"]
            + new_results["t_iterate_phase2"]
            + new_results["t_unsubscribe"]
            + new_results["t_iterate_phase3"]
        )

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                  PERFORMANCE COMPARISON                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Phase                    │    OLD (s)  │    NEW (s)  │  Speedup ║
╠──────────────────────────┼─────────────┼─────────────┼──────────╣
║ Subscribe 50 symbols     │ {old_results["t_subscribe_batch1"]:>10.3f}  │ {new_results["t_subscribe_batch1"]:>10.3f}  │ {old_results["t_subscribe_batch1"] / max(new_results["t_subscribe_batch1"], 1e-6):>7.1f}x ║
║ Iterate phase 1 (500)   │ {old_results["t_iterate_phase1"]:>10.3f}  │ {new_results["t_iterate_phase1"]:>10.3f}  │ {old_results["t_iterate_phase1"] / max(new_results["t_iterate_phase1"], 1e-6):>7.1f}x ║
║ Subscribe +50 symbols    │ {old_results["t_subscribe_batch2"]:>10.3f}  │ {new_results["t_subscribe_batch2"]:>10.3f}  │ {old_results["t_subscribe_batch2"] / max(new_results["t_subscribe_batch2"], 1e-6):>7.1f}x ║
║ Iterate phase 2 (500)   │ {old_results["t_iterate_phase2"]:>10.3f}  │ {new_results["t_iterate_phase2"]:>10.3f}  │ {old_results["t_iterate_phase2"] / max(new_results["t_iterate_phase2"], 1e-6):>7.1f}x ║
║ Unsubscribe 25+25        │ {old_results["t_unsubscribe"]:>10.3f}  │ {new_results["t_unsubscribe"]:>10.3f}  │ {old_results["t_unsubscribe"] / max(new_results["t_unsubscribe"], 1e-6):>7.1f}x ║
║ Iterate phase 3 (500)   │ {old_results["t_iterate_phase3"]:>10.3f}  │ {new_results["t_iterate_phase3"]:>10.3f}  │ {old_results["t_iterate_phase3"] / max(new_results["t_iterate_phase3"], 1e-6):>7.1f}x ║
╠──────────────────────────┼─────────────┼─────────────┼──────────╣
║ TOTAL                    │ {old_total:>10.3f}  │ {new_total:>10.3f}  │ {old_total / max(new_total, 1e-6):>7.1f}x ║
╠──────────────────────────┼─────────────┼─────────────┼──────────╣
║ Events: P1               │ {old_results["phase1_events"]:>10d}  │ {new_results["phase1_events"]:>10d}  │          ║
║ Events: P2               │ {old_results["phase2_events"]:>10d}  │ {new_results["phase2_events"]:>10d}  │          ║
║ Events: P3               │ {old_results["phase3_events"]:>10d}  │ {new_results["phase3_events"]:>10d}  │          ║
║ Events: Total            │ {old_results["total_events"]:>10d}  │ {new_results["total_events"]:>10d}  │          ║
╚══════════════════════════════════════════════════════════════════╝
"""
        print(report)

        # - both should produce events (sanity)
        assert old_results["total_events"] == 1500, f"Old: expected 1500 events, got {old_results['total_events']}"
        assert new_results["total_events"] == 1500, f"New: expected 1500 events, got {new_results['total_events']}"
