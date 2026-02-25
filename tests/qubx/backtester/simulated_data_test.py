from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.data import SimulatedDataProvider
from qubx.backtester.simulated_data import EmulatedTickSequence, SimulatedDataIterator
from qubx.backtester.utils import SimulatedTimeProvider
from qubx.core.basics import DataType
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, time_as_nsec
from qubx.data.storages.csv import CsvStorage
from qubx.data.storages.handy import HandyStorage
from qubx.pandaz.utils import ohlc_resample


class TestSimulatedDataIterator:
    """
    Analog tests for SimulatedDataIterator using CsvStorage (IReader/IStorage).
    Each test mirrors a TestSimulatedDataStuff test and must produce same results.
    """

    CSV_STORAGE_PATH = "tests/data/storages/csv"

    @staticmethod
    def _make_csv_storage():
        return CsvStorage(TestSimulatedDataIterator.CSV_STORAGE_PATH)

    @staticmethod
    def _make_isd(storage, **kwargs):
        return SimulatedDataIterator(storage=storage, **kwargs)

    def test_data_management(self):
        """
        Mirrors test_iterable_simulation_data_management.
        Tests subscribe/unsubscribe/query methods produce same results.
        """

        storage = self._make_csv_storage()
        isd = SimulatedDataIterator(
            storage=storage,
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

    def test_queue_with_warmup(self):
        """
        Mirrors test_iterable_simulation_data_queue_with_warmup.
        Uses 1h timeframe (csv storage has 1h data).
        24h warmup = 24 bars * 4 emulated events = 96 historical events per symbol.
        3 symbols * 96 = 288 total historical events.
        """
        storage = self._make_csv_storage()
        isd = self._make_isd(storage, open_close_time_indent_secs=300)

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

    def test_historical_search_with_warmup(self):
        """
        Mirrors test_iterable_simulation_data_last_historical_search.
        Uses 1h timeframe. At iteration 60, dynamically add s3 with warmup.
        Must get len(h_data) == 96 (24h warmup * 4 events/bar) and last time < current_time.
        """
        storage = self._make_csv_storage()
        isd = self._make_isd(storage, open_close_time_indent_secs=300)

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

    def test_historical_search_no_warmup(self):
        """
        Mirrors test_iterable_simulation_data_last_historical_search_no_warmup.
        Uses 1h timeframe. At iteration 10, dynamically add s3 without warmup.
        peek must return empty.
        """
        storage = self._make_csv_storage()
        isd = self._make_isd(storage, open_close_time_indent_secs=300)

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

    def test_remove_readd_instrument_peek_data(self):
        """
        Mirrors test_iterable_simulation_data_remove_readd_instrument_peek_data.
        Remove instrument A at iteration 20, re-add at iteration 40.
        peek_historical_data must return FRESH data (not stale from before removal).
        """

        # - Build HandyStorage with rising OHLC (same as _DummyTestRisingOHLCDataReader)
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

        isd = SimulatedDataIterator(storage=storage, open_close_time_indent_secs=1)

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

    def test_dual_subscription_mixed_warmup(self):
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

        isd = SimulatedDataIterator(storage=storage, open_close_time_indent_secs=1)

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

    def test_add_remove_add_produces_events(self):
        """
        Verify that add → remove → add cycle works correctly:
          1. Subscribe A, B — iterate, both produce events
          2. Remove A — iterate, only B produces events
          3. Re-add A — iterate, BOTH produce events again (A not stuck)
          4. A's events after re-add have timestamps > removal time (fresh data, not stale)
        """

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

        isd = SimulatedDataIterator(storage=storage)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")  # - A
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")  # - B
        assert s1 is not None and s2 is not None

        isd.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        # - setup indent time to 5 sec after pumps creation
        isd.update_emulation_time_indent_seconds(5.0)

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

            if _n == 0:
                assert pd.Timestamp(event.time).second == 5, "Update must be +5 sec shifted"
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

    def test_emulated_updates_subscription(self):
        storage = CsvStorage(self.CSV_STORAGE_PATH)
        data_iter = SimulatedDataIterator(storage=storage)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        assert s1 is not None and s2 is not None

        data_iter.add_instruments_for_subscription(DataType.OHLC_QUOTES["1h"], [s1, s2])

        res = {
            s1: OHLCV(s1.symbol, "1h"),
            s2: OHLCV(s2.symbol, "1h"),
        }
        for d in data_iter.create_iterable("2023-07-01", "2023-07-02"):
            q = d[2]
            res[d[0]].update(q.time, q.mid_price())

        r_s1 = (
            storage.get_reader(s1.exchange, s1.market_type)
            .read(s1.symbol, "ohlc(1h)", "2023-07-01", "2023-07-02")
            .to_pd()[["open", "high", "low", "close"]]  # type: ignore
        )
        r_s2 = (
            storage.get_reader(s2.exchange, s2.market_type)
            .read(s2.symbol, "ohlc(1h)", "2023-07-01", "2023-07-02")
            .to_pd()[["open", "high", "low", "close"]]  # type: ignore
        )
        # - r_s1 index is datetime64[s] (Arrow/CSV path), res[s1].pd() index is datetime64[ns]
        # - (OHLCV stores timestamps as int64 nanoseconds) — same values, different resolution.
        # - assert_frame_equal with check_index_type=False compares timestamp values numerically.
        pd.testing.assert_frame_equal(
            r_s1,
            res[s1].pd()[["open", "high", "low", "close"]],
            check_index_type=False,
        )
        pd.testing.assert_frame_equal(
            r_s2,
            res[s2].pd()[["open", "high", "low", "close"]],
            check_index_type=False,
        )

    def test_emulated_updates_subscription_with_trading_session(self):
        from qubx.backtester.utils import STOCK_DAILY_SESSION

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert s1 is not None

        rd = CsvStorage(self.CSV_STORAGE_PATH).get_reader("BINANCE.UM", "SWAP")

        storage = HandyStorage(
            {"BTCUSDT": ohlc_resample(rd.read(s1.symbol, "ohlc(1h)", "2023-07-01", "2023-07-10").to_pd(), "1d")},  # type: ignore
            exchange="BINANCE.UM:SWAP",
        )

        data_iter = SimulatedDataIterator(
            storage=storage, trading_session={"BINANCE.UM": STOCK_DAILY_SESSION}, open_close_time_indent_secs=0
        )

        data_iter.add_instruments_for_subscription(DataType.OHLC_QUOTES["1D"], [s1])

        # - two full 1D bars needed: EmulatedTickSequence requires n_rows >= 2
        _n = 0
        for d in data_iter.create_iterable("2023-07-01", "2023-07-03"):
            q = d[2]
            if _n == 0:
                assert pd.Timestamp(q.time).time().hour == 9
                assert pd.Timestamp(q.time).time().minute == 30
                assert pd.Timestamp(q.time).time().second == 0

            if _n == 3:
                assert pd.Timestamp(q.time).time().hour == 15
                assert pd.Timestamp(q.time).time().minute == 59
                assert pd.Timestamp(q.time).time().second == 59
            _n += 1


class TestSimulatedDataProvider:
    """
    Tests for SimulatedDataProvider — thin wrapper over SimulatedDataIterator.
    Focuses on get_ohlc() which uses time_provider for simulated time
    and delegates to SimulatedDataIterator.get_ohlc() + _process_bar_records().
    """

    CSV_STORAGE_PATH = "tests/data/storages/csv"

    @staticmethod
    def _make_provider(
        data_iter: SimulatedDataIterator,
        sim_time: str,
        exchange_id: str = "BINANCE.UM",
    ) -> SimulatedDataProvider:
        """
        Create SimulatedDataProvider with real data_source and time_provider,
        mock the rest (channel, scheduler, account) — not needed for get_ohlc().
        """
        time_provider = SimulatedTimeProvider(np.datetime64(sim_time, "ns"))
        return SimulatedDataProvider(
            exchange_id=exchange_id,
            channel=MagicMock(),
            time_provider=time_provider,
            account=MagicMock(),
            data_source=data_iter,
        )

    def test_get_ohlc_with_csv_storage(self):
        """
        Test get_ohlc returns correct bars for BTCUSDT and ETHUSDT with 1H OHLC from CsvStorage.

        Setup:
        - CsvStorage with 1H OHLC data (2023-06-01 to 2023-08-01)
        - open_close_time_indent_secs = 5 (appropriate for 1h bars)
        - sim_time = 2023-07-15 12:00:00
        - nbarsback = 10

        Cut logic with indent=5s, 1h bars, sim_time=12:00:00:
        - Bar at 11:00 → effective_close = 11:00 + 1h - 5s = 11:59:55
          → 11:00 <= 12:00 AND 12:00 < 11:59:55? → NO → bar INCLUDED ✅
        - Bar at 12:00 → effective_close = 12:00 + 1h - 5s = 12:59:55
          → 12:00 <= 12:00 AND 12:00 < 12:59:55? → YES → BREAK (excluded) ✅

        Expected: 10 bars from 02:00 to 11:00, bar at 12:00 excluded.
        """
        indent_secs = 5
        storage = CsvStorage(self.CSV_STORAGE_PATH)
        data_iter = SimulatedDataIterator(storage=storage, open_close_time_indent_secs=indent_secs)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        assert s1 is not None and s2 is not None

        # - subscribe both instruments to ohlc(1h)
        data_iter.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        # - create provider with sim_time at 2023-07-15 12:00:00
        sim_time_str = "2023-07-15T12:00:00"
        provider = self._make_provider(data_iter, sim_time_str)
        sim_time_ns = time_as_nsec(np.datetime64(sim_time_str, "ns"))

        nbarsback = 10

        for instrument in [s1, s2]:
            bars = provider.get_ohlc(instrument, "1h", nbarsback)

            # - must return exactly 10 bars
            assert len(bars) == nbarsback, f"{instrument.symbol}: expected {nbarsback} bars, got {len(bars)}"

            # - all bar timestamps must be strictly before sim_time
            for bar in bars:
                assert bar.time < sim_time_ns, (
                    f"{instrument.symbol}: bar at {pd.Timestamp(bar.time)} >= sim_time {sim_time_str}"
                )

            # - bars must be in chronological order (strictly increasing)
            for i in range(1, len(bars)):
                assert bars[i].time > bars[i - 1].time, f"{instrument.symbol}: bars not chronological at index {i}"

            # - first bar should start at 02:00 (12:00 - 10h)
            first_bar_expected = pd.Timestamp("2023-07-15 02:00:00").value
            assert bars[0].time == first_bar_expected, (
                f"{instrument.symbol}: first bar at {pd.Timestamp(bars[0].time)}, expected 02:00"
            )

            # - last bar should start at 11:00 (bar at 12:00 is excluded by indent cut)
            last_bar_expected = pd.Timestamp("2023-07-15 11:00:00").value
            assert bars[-1].time == last_bar_expected, (
                f"{instrument.symbol}: last bar at {pd.Timestamp(bars[-1].time)}, expected 11:00"
            )

            # - all bars must have valid OHLC prices (non-zero, positive)
            for bar in bars:
                assert bar.open > 0 and bar.high > 0 and bar.low > 0 and bar.close > 0, (
                    f"{instrument.symbol}: invalid OHLC at {pd.Timestamp(bar.time)}"
                )
                assert bar.high >= bar.low, f"{instrument.symbol}: high < low at {pd.Timestamp(bar.time)}"

            # - all bars should have volume data
            assert any(bar.volume > 0 for bar in bars), f"{instrument.symbol}: no bars have volume data"

        # - BTC and ETH should have different price levels
        btc_bars = provider.get_ohlc(s1, "1h", nbarsback)
        eth_bars = provider.get_ohlc(s2, "1h", nbarsback)

        assert btc_bars[0].close != eth_bars[0].close, "BTC and ETH should have different prices"

        # - BTC should be significantly more expensive than ETH in Jul 2023
        assert btc_bars[0].close > eth_bars[0].close * 5, (
            f"BTC ({btc_bars[0].close:.0f}) should be > 5x ETH ({eth_bars[0].close:.0f})"
        )

    def test_get_ohlc_just_before_hour_boundary(self):
        """
        Test that at sim_time=11:59:59 the 11:00 bar is fully included (closed)
        and the 12:00 bar is NOT visible.

        With indent=5s, 1h bars:
        - Bar at 11:00 → effective_close = 11:59:55
          → 11:00 <= 11:59:59 AND 11:59:59 < 11:59:55? → NO (11:59:59 >= 11:59:55)
          → bar is INCLUDED — it's fully closed at this point ✅
        - Bar at 12:00 → not in reader range (stop=11:59:59 < 12:00) ✅

        Note: nbarsback=10 with sim_time=11:59:59 → end=01:59:59.
        Reader's searchsorted picks the 01:00 bar (contains 01:59:59), so we get
        11 bars (01:00..11:00) — one extra vs the 12:00:00 test. The key check is
        that 11:00 IS included (complete) and 12:00 is NOT visible.
        """
        indent_secs = 5
        storage = CsvStorage(self.CSV_STORAGE_PATH)
        data_iter = SimulatedDataIterator(storage=storage, open_close_time_indent_secs=indent_secs)

        s1 = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        s2 = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        assert s1 is not None and s2 is not None

        data_iter.add_instruments_for_subscription(DataType.OHLC["1h"], [s1, s2])

        sim_time_str = "2023-07-15T11:59:59"
        provider = self._make_provider(data_iter, sim_time_str)
        sim_time_ns = time_as_nsec(np.datetime64(sim_time_str, "ns"))

        nbarsback = 10

        for instrument in [s1, s2]:
            bars = provider.get_ohlc(instrument, "1h", nbarsback)

            # - last bar must be 11:00 — fully closed (effective_close=11:59:55 < 11:59:59)
            last_bar_expected = pd.Timestamp("2023-07-15 11:00:00").value
            assert bars[-1].time == last_bar_expected, (
                f"{instrument.symbol}: last bar at {pd.Timestamp(bars[-1].time)}, expected 11:00"
            )

            # - NO bar at 12:00 or later
            bar_at_12 = pd.Timestamp("2023-07-15 12:00:00").value
            for bar in bars:
                assert bar.time < bar_at_12, (
                    f"{instrument.symbol}: bar at {pd.Timestamp(bar.time)} should not be visible at sim_time {sim_time_str}"
                )

            # - all bars must be before sim_time
            for bar in bars:
                assert bar.time < sim_time_ns, f"{instrument.symbol}: bar at {pd.Timestamp(bar.time)} >= sim_time"

            # - 11:00 bar must have real close price and volume (fully closed bar)
            assert bars[-1].close > 0, f"{instrument.symbol}: 11:00 bar has no close price"
            assert bars[-1].volume > 0, f"{instrument.symbol}: 11:00 bar has no volume"
