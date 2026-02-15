"""
Tests for TimeGuardedReader and TimeGuardedStorage.

Verifies that time-guarding correctly clamps the stop parameter at read level,
preventing look-ahead bias in simulation. Tests cover:
- Stop clamped to sim time for all data types (OHLC, trade, funding, etc.)
- Caller-provided stop preserved when earlier than guard time
- TimeGuardedStorage reader caching and delegation
- Out-of-range reads return empty data
"""

import numpy as np
import pandas as pd

from qubx.core.basics import ITimeProvider
from qubx.data.guards import TimeGuardedReader, TimeGuardedStorage
from qubx.data.registry import StorageRegistry
from qubx.data.storages.handy import HandyStorage, Transformable


class FixedTimeProvider(ITimeProvider):
    """
    Test time provider returning a fixed time.
    """

    def __init__(self, time: str | np.datetime64):
        self._time = np.datetime64(time, "ns") if isinstance(time, str) else time

    def time(self) -> np.datetime64:
        return self._time

    def set_time(self, t: str):
        self._time = np.datetime64(t, "ns")


def _make_ohlc(
    start: str = "2024-01-01",
    end: str = "2024-01-05",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Generate sample OHLC data from start to end (inclusive).
    """
    idx = pd.date_range(start, end, freq=freq, name="timestamp")
    rng = np.random.default_rng(42)
    n = len(idx)
    close = 40000 + rng.standard_normal(n).cumsum() * 100
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, n),
            "high": close + rng.uniform(0, 100, n),
            "low": close - rng.uniform(0, 100, n),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


def _make_trades(start: str = "2024-01-01", end: str = "2024-01-01 03:19") -> pd.DataFrame:
    """
    Generate sample trade data from start to end at 1min intervals.
    """
    idx = pd.date_range(start, end, freq="1min", name="timestamp")
    rng = np.random.default_rng(42)
    n = len(idx)
    return pd.DataFrame(
        {
            "price": 40000 + rng.standard_normal(n).cumsum() * 10,
            "size": rng.uniform(0.01, 1.0, n),
            "side": rng.choice(["buy", "sell"], n),
        },
        index=idx,
    )


def _make_funding(start: str = "2024-01-01", end: str = "2024-01-10 16:00") -> pd.DataFrame:
    """
    Generate sample funding rate data from start to end at 8h intervals.
    """
    idx = pd.date_range(start, end, freq="8h", name="timestamp")
    rng = np.random.default_rng(42)
    n = len(idx)
    return pd.DataFrame(
        {
            "funding_rate": rng.uniform(-0.001, 0.001, n),
        },
        index=idx,
    )


def _build_storage(start: str = "2024-01-01", stop: str = "2024-01-05") -> HandyStorage:
    """
    Build a HandyStorage with OHLC (1h), trade, and funding data for BTCUSDT and ETHUSDT.
    """
    return HandyStorage(
        {
            "BTCUSDT": [
                _make_ohlc(start=start, end=stop),
                _make_trades(start=start, end=stop),
                _make_funding(start=start, end=stop),
            ],
            "ETHUSDT": [
                _make_ohlc(start=start, end=stop),
            ],
        },
        exchange="BINANCE.UM",
    )


# ---------------------------------------------------------------------------
# TimeGuardedReader
# ---------------------------------------------------------------------------


class TestTimeGuardedReader:
    """
    Tests that TimeGuardedReader correctly clamps stop parameter.
    Uses HandyStorage as inner reader for realistic end-to-end testing.
    """

    def test_ohlc_clamped_to_sim_time(self):
        """
        With sim time at 2024-01-01 12:00, reading ohlc(1h) should clamp
        stop to 12:00. The bar at 12:00 is included (its open timestamp <= sim time).
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        raw = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-05")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - last bar should be at 12:00 (sim time), not beyond
        assert df.index[-1] == pd.Timestamp("2024-01-01 12:00:00")

    def test_ohlc_4h_clamped(self):
        """
        With ohlc(4h), stop clamped to sim time directly.
        Sim time 2024-01-01 16:00 -> last visible bar at 16:00.
        """
        ohlc_4h = _make_ohlc(start="2024-01-01", end="2024-01-10", freq="4h")
        storage = HandyStorage({"BTCUSDT": ohlc_4h}, exchange="BINANCE.UM")
        tp = FixedTimeProvider("2024-01-01T16:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        raw = reader.read("BTCUSDT", "ohlc(4h)", start="2024-01-01", stop="2024-01-10")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - last visible bar at 16:00 (sim time)
        assert df.index[-1] == pd.Timestamp("2024-01-01 16:00:00")

    def test_ohlc_1d_clamped(self):
        """
        With ohlc(1d), sim time 2024-01-05 00:00 -> last visible bar at 2024-01-05.
        """
        ohlc_1d = _make_ohlc(start="2024-01-01", end="2024-01-30", freq="1d")
        storage = HandyStorage({"BTCUSDT": ohlc_1d}, exchange="BINANCE.UM")
        tp = FixedTimeProvider("2024-01-05T00:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        inner = storage.get_reader("BINANCE.UM", "SWAP")
        stored_dtype = inner.get_data_types("BTCUSDT")[0]

        raw = reader.read("BTCUSDT", stored_dtype, start="2024-01-01", stop="2024-01-10")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - last visible bar at 2024-01-05 (sim time)
        assert df.index[-1] == pd.Timestamp("2024-01-05")

    def test_trade_clamped_to_sim_time(self):
        """
        Trade data: stop clamped to sim time directly.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T01:30:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        raw = reader.read("BTCUSDT", "trade", start="2024-01-01", stop="2024-01-05")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - last trade should be at or before 01:30
        assert df.index[-1] <= pd.Timestamp("2024-01-01 01:30:00")

    def test_caller_stop_earlier_preserved(self):
        """
        When the caller's stop is earlier than the guard time, the caller's
        stop should be used (we never widen the range).
        """
        storage = _build_storage()
        # - sim time is far in the future — guard won't constrain
        tp = FixedTimeProvider("2024-12-31T00:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        # - caller requests stop at 2024-01-01 05:00
        raw = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-01 05:00")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - should respect caller's stop, not expand to sim time
        assert df.index[-1] <= pd.Timestamp("2024-01-01 05:00:00")

    def test_guard_tighter_than_caller_stop(self):
        """
        When guard time is earlier than caller's stop, guard time wins.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T05:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        # - caller requests wide range but guard clamps
        raw = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-05")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - clamped to 05:00 (sim time)
        assert df.index[-1] == pd.Timestamp("2024-01-01 05:00:00")

    def test_delegates_get_data_id(self):
        """
        get_data_id should pass through to the inner reader.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        guarded = TimeGuardedReader(inner, tp)

        assert guarded.get_data_id() == inner.get_data_id()

    def test_delegates_get_data_types(self):
        """
        get_data_types should pass through to the inner reader.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        guarded = TimeGuardedReader(inner, tp)

        assert guarded.get_data_types("BTCUSDT") == inner.get_data_types("BTCUSDT")

    def test_delegates_get_time_range(self):
        """
        get_time_range should pass through to the inner reader (unguarded).
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        inner = storage.get_reader("BINANCE.UM", "SWAP")
        guarded = TimeGuardedReader(inner, tp)

        for dtype_str in inner.get_data_types("BTCUSDT"):
            assert guarded.get_time_range("BTCUSDT", dtype_str) == inner.get_time_range("BTCUSDT", dtype_str)

    def test_repr(self):
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)
        assert "TimeGuardedReader" in repr(reader)

    def test_multiple_symbols_guarded(self):
        """
        Guard applies to each symbol independently — same stop clamp for all.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T10:00:00")
        reader = TimeGuardedReader(storage.get_reader("BINANCE.UM", "SWAP"), tp)

        from qubx.data.transformers import PandasFrame

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            raw = reader.read(symbol, "ohlc(1h)", start="2024-01-01", stop="2024-01-05")
            assert isinstance(raw, Transformable)
            df = raw.transform(PandasFrame())
            # - clamped to 10:00 (sim time)
            assert df.index[-1] == pd.Timestamp("2024-01-01 10:00:00"), f"Failed for {symbol}"


# ---------------------------------------------------------------------------
# TimeGuardedStorage
# ---------------------------------------------------------------------------


class TestTimeGuardedStorage:
    def test_get_reader_returns_guarded(self):
        """
        get_reader() should return a TimeGuardedReader instance.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        guarded_storage = TimeGuardedStorage(storage, tp)

        reader = guarded_storage.get_reader("BINANCE.UM", "SWAP")
        assert isinstance(reader, TimeGuardedReader)

    def test_reader_caching(self):
        """
        Repeated get_reader() calls for same (exchange, market) should
        return the same TimeGuardedReader instance.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        guarded_storage = TimeGuardedStorage(storage, tp)

        r1 = guarded_storage.get_reader("BINANCE.UM", "SWAP")
        r2 = guarded_storage.get_reader("BINANCE.UM", "SWAP")
        assert r1 is r2

    def test_getitem_shorthand(self):
        """
        storage["BINANCE.UM", "SWAP"] should work as get_reader shorthand.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T12:00:00")
        guarded_storage = TimeGuardedStorage(storage, tp)

        reader = guarded_storage["BINANCE.UM", "SWAP"]
        assert isinstance(reader, TimeGuardedReader)

    def test_guarded_read_through_storage(self):
        """
        End-to-end: get reader from guarded storage, read OHLC, verify clamped.
        """
        storage = _build_storage()
        tp = FixedTimeProvider("2024-01-01T08:00:00")
        guarded_storage = TimeGuardedStorage(storage, tp)

        reader = guarded_storage.get_reader("BINANCE.UM", "SWAP")

        from qubx.data.transformers import PandasFrame

        raw = reader.read("BTCUSDT", "ohlc(1h)", start="2024-01-01", stop="2024-01-05")
        assert isinstance(raw, Transformable)
        df = raw.transform(PandasFrame())

        # - clamped to 08:00 (sim time)
        assert df.index[-1] == pd.Timestamp("2024-01-01 08:00:00")

    def test_out_of_range_read(self):
        """
        When guard time (2021-01-01) is before all CSV data (2023-06-01..2023-08-01),
        the clamped stop falls before data range — result must be empty.
        """
        stor = StorageRegistry.get("csv::tests/data/storages/csv/")

        c_time = FixedTimeProvider("2021-01-01")
        gstor = TimeGuardedStorage(stor, c_time)
        gr = gstor.get_reader("BINANCE.UM", "SWAP")

        g_data = gr.read("BTCUSDT", "ohlc(1h)", "2020-10-01", "2026-01-01").to_pd()
        assert g_data.empty
