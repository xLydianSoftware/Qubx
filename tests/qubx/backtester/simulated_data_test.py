from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

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
