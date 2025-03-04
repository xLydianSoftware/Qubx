import numpy as np
import pytest

from qubx.core.series import OHLCV, TimeSeries, TradeArray
from qubx.core.utils import recognize_time
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader
from qubx.ta.indicators import psar, swings
from tests.qubx.ta.utils_for_testing import N, push


class TestCoreSeries:
    def test_basic_series(self):
        ts = TimeSeries("test", "10Min")
        ts.update(recognize_time("2024-01-01 00:00"), 1)
        ts.update(recognize_time("2024-01-01 00:01"), 5)
        ts.update(recognize_time("2024-01-01 00:06"), 2)
        ts.update(recognize_time("2024-01-01 00:12"), 3)
        ts.update(recognize_time("2024-01-01 00:21"), 4)
        ts.update(recognize_time("2024-01-01 00:22"), 5)
        ts.update(recognize_time("2024-01-01 00:31"), 6)
        ts.update(recognize_time("2024-01-01 00:33"), 7)
        ts.update(recognize_time("2024-01-01 00:45"), -12)
        ts.update(recognize_time("2024-01-01 00:55"), 12)
        ts.update(recognize_time("2024-01-01 01:00"), 12)
        assert np.array_equal(ts.to_series().values, np.array([2, 3, 5, 7, -12, 12, 12]))

    def test_ohlc_series(self):
        ohlc = OHLCV("BTCUSDT", "1Min")
        push(
            ohlc,
            [
                ("2024-01-01 00:00", 9),
                ("2024-01-01 00:00", 1),
                ("2024-01-01 00:01", 2),
                ("2024-01-01 00:01", 3),
                ("2024-01-01 00:01", 2),
                ("2024-01-01 00:02", 3),
                ("2024-01-01 00:03", 4),
                ("2024-01-01 00:04", 5),
                ("2024-01-01 00:04", 5.1),
                ("2024-01-01 00:04:20", 5),
                ("2024-01-01 00:05", 6),
                ("2024-01-01 00:05", 7),
                ("2024-01-01 00:05", 6),
                ("2024-01-01 00:07", 8),
                ("2024-01-01 00:07", -1),
                ("2024-01-01 00:07", 8),
                ("2024-01-01 00:08", 8),
                ("2024-01-01 00:09", 8),
                ("2024-01-01 00:10", 12),
                ("2024-01-01 00:10:01", 21),
                ("2024-01-01 00:10:30", 1),
                ("2024-01-01 00:10:31", 5),
                ("2024-01-01 00:11", 13),
                ("2024-01-01 00:12", 14),
                ("2024-01-01 00:13", 15),
                ("2024-01-01 00:14", 17),
                ("2024-01-01 00:15", 4),
            ],
            1,
        )

        assert len(ohlc) == 15

        r = ohlc.to_series()

        ri = r.loc["2024-01-01 00:10:00"]
        assert ri.open == 12
        assert ri.high == 21
        assert ri.low == 1
        assert ri.close == 5
        assert ri.volume == 4

    def test_series_locator(self):
        ts = TimeSeries("test", "1Min")
        push(
            ts,
            [
                ("2024-01-01 00:00", 9),
                ("2024-01-01 00:01", 2),
                ("2024-01-01 00:02", 3),
                ("2024-01-01 00:03", 4),
                ("2024-01-01 00:04", 5),
                ("2024-01-01 00:05", 6),
                ("2024-01-01 00:07", 8),
                ("2024-01-01 00:08", 8),
                ("2024-01-01 00:09", 8),
                ("2024-01-01 00:10", 12),
                ("2024-01-01 00:11", 13),
                ("2024-01-01 00:12", 14),
                ("2024-01-01 00:13", 15),
                ("2024-01-01 00:14", 17),
                ("2024-01-01 00:15", 18),
            ],
        )
        assert ts[:] == ts.loc[:][:]
        assert len(ts.loc["2024-01-01 00:00":"2024-01-01 00:03"]) == 4
        assert len(ts.loc["2024-01-01 00:00":]) == 15
        assert len(ts.loc[:"2024-01-01 00:10"]) == 10
        assert len(ts.loc[0:4]) == 4
        assert len(ts.loc[0:100]) == 15
        assert ts.loc["2024-01-01 00:10"] == (np.datetime64("2024-01-01 00:10"), 12.0)
        assert len(ts.loc[:]) == 15

    def test_indicator_locator(self):
        ohlc = CsvStorageDataReader("tests/data/csv").read(
            "BTCUSDT_ohlcv_M1", start="2024-01-01", stop="2024-01-15", transform=AsOhlcvSeries("15Min")
        )
        assert isinstance(ohlc, OHLCV)
        sw = swings(ohlc, psar)

        cln1 = sw.loc[:]

        assert all(sw.tops.pd() == cln1.tops.pd())
        assert all(sw.bottoms.pd() == cln1.bottoms.pd())

        # - slices
        assert len(sw.loc["2024-01-01 00:30:00":"2024-01-01 11:30:00"]) == 45
        assert len(sw.loc[:"2024-01-02 00:00:00"]) == 97


class TestTradeArray:
    def test_trade_array_empty(self):
        trades = TradeArray()
        assert len(trades) == 0
        assert trades.total_size == 0
        assert trades.buy_size == 0
        assert trades.sell_size == 0
        assert trades.min_buy_price == float("inf")
        assert trades.max_buy_price == float("-inf")
        assert trades.min_sell_price == float("inf")
        assert trades.max_sell_price == float("-inf")

    def test_trade_array_add(self):
        trades = TradeArray()

        # Add some trades
        trades.add(1000000, 100.0, 1.0, 1)  # buy
        trades.add(1000001, 101.0, 2.0, -1)  # sell
        trades.add(1000002, 99.0, 1.5, 1)  # buy
        trades.add(1000003, 102.0, 0.5, -1)  # sell

        assert len(trades) == 4
        assert trades.total_size == 5.0
        assert trades.buy_size == 2.5
        assert trades.sell_size == 2.5
        assert trades.min_buy_price == 99.0
        assert trades.max_buy_price == 100.0
        assert trades.min_sell_price == 101.0
        assert trades.max_sell_price == 102.0
        assert trades.time == 1000003

        # Test Trade object access
        first_trade = trades[0]
        assert first_trade.time == 1000000
        assert first_trade.price == 100.0
        assert first_trade.size == 1.0
        assert first_trade.taker == 1

        # Test array access using trades attribute
        slice_data = trades.trades[1:3]
        assert np.array_equal(slice_data["size"], np.array([2.0, 1.5]))

        # Test clear
        trades.clear()
        assert len(trades) == 0
        assert trades.total_size == 0

    def test_trade_array_from_array(self):
        # Create test data
        data = np.array(
            [(1000000, 100.0, 1.0, 1), (1000001, 101.0, 2.0, -1), (1000002, 99.0, 1.5, 1), (1000003, 102.0, 0.5, -1)],
            dtype=[("time", "i8"), ("price", "f8"), ("size", "f8"), ("side", "i1")],
        )

        trades = TradeArray(data)

        # Verify statistics were calculated correctly
        assert len(trades) == 4
        assert trades.total_size == 5.0
        assert trades.buy_size == 2.5
        assert trades.sell_size == 2.5
        assert trades.min_buy_price == 99.0
        assert trades.max_buy_price == 100.0
        assert trades.min_sell_price == 101.0
        assert trades.max_sell_price == 102.0
        assert trades.time == 1000003

        # Verify Trade object access
        trade = trades[1]
        assert trade.time == 1000001
        assert trade.price == 101.0
        assert trade.size == 2.0
        assert trade.taker == -1

        # Test array access using trades attribute
        slice_data = trades.trades[1:3]
        assert len(slice_data) == 2
        assert np.array_equal(slice_data["time"], np.array([1000001, 1000002]))

    def test_trade_array_invalid_input(self):
        # Test invalid array type
        with pytest.raises(TypeError, match="data must be a numpy array"):
            TradeArray([(1, 100.0, 1.0, 1)])  # List instead of numpy array

        # Test invalid dtype
        invalid_data = np.array([(1, 100.0)], dtype=[("time", "i8"), ("price", "f8")])
        with pytest.raises(ValueError, match="Cannot convert input array to required dtype"):
            TradeArray(invalid_data)
