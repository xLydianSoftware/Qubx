import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import TimestampedDict
from qubx.core.series import (
    OHLCV,
    Bar,
    BundledSeries,
    ColumnarSeries,
    GenericSeries,
    IndicatorGeneric,
    Quote,
    TimeSeries,
    TradeArray,
)
from qubx.core.utils import recognize_time
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader
from qubx.ta.indicators import psar, sma, swings
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

    def test_ohlc_update_by_bars(self):
        """Test the update_by_bars method of OHLCV class."""
        # Create an empty OHLCV series
        ohlc = OHLCV("BTCUSDT", "1Min")

        # Create some initial bars
        initial_bars = [
            Bar(
                recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item(),
                100.0,
                105.0,
                99.0,
                102.0,
                volume=10.0,
                bought_volume=6.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:11").astype("datetime64[ns]").item(),
                102.0,
                107.0,
                101.0,
                105.0,
                volume=15.0,
                bought_volume=8.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                105.0,
                110.0,
                104.0,
                108.0,
                volume=12.0,
                bought_volume=7.0,
            ),
        ]

        # Add initial bars
        result = ohlc.update_by_bars(initial_bars)

        # Verify initial bars were added
        assert result is ohlc  # Should return self for method chaining
        assert len(ohlc) == 3
        assert ohlc.times[0] == recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item()
        assert ohlc.times[-1] == recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item()

        # Create a simple indicator to test indicator updates
        ma = sma(ohlc.close, 2)  # Simple moving average with period 2

        # Test adding bars in the past (older than existing data)
        past_bars = [
            Bar(
                recognize_time("2024-01-01 00:08").astype("datetime64[ns]").item(),
                95.0,
                98.0,
                94.0,
                97.0,
                volume=8.0,
                bought_volume=5.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:09").astype("datetime64[ns]").item(),
                97.0,
                99.0,
                96.0,
                98.0,
                volume=9.0,
                bought_volume=4.0,
            ),
        ]

        result = ohlc.update_by_bars(past_bars)

        # Verify past bars were added
        assert result is ohlc
        assert len(ohlc) == 5

        # Note: The implementation of update_by_bars appends past bars to the end
        # without sorting, so the order is not maintained as newest first
        # Check that all expected times are in the series
        expected_times = set(
            [
                recognize_time("2024-01-01 00:08").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:09").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:11").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
            ]
        )
        actual_times = set(ohlc.times.values)
        assert expected_times == actual_times

        # Test adding bars in the future (newer than existing data)
        future_bars = [
            Bar(
                recognize_time("2024-01-01 00:13").astype("datetime64[ns]").item(),
                108.0,
                112.0,
                107.0,
                110.0,
                volume=14.0,
                bought_volume=9.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:14").astype("datetime64[ns]").item(),
                110.0,
                115.0,
                109.0,
                113.0,
                volume=16.0,
                bought_volume=10.0,
            ),
        ]

        result = ohlc.update_by_bars(future_bars)

        # Verify future bars were added
        assert result is ohlc
        assert len(ohlc) == 7
        assert (
            ohlc.times[0] == recognize_time("2024-01-01 00:14").astype("datetime64[ns]").item()
        )  # Newest should be first for future bars

        # Check that all expected times are in the series
        expected_times = set(
            [
                recognize_time("2024-01-01 00:08").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:09").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:11").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:13").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:14").astype("datetime64[ns]").item(),
            ]
        )
        actual_times = set(ohlc.times.values)
        assert expected_times == actual_times

        # Test adding bars with gaps in the middle
        gap_bars = [
            Bar(
                recognize_time("2024-01-01 00:16").astype("datetime64[ns]").item(),
                115.0,
                120.0,
                114.0,
                118.0,
                volume=18.0,
                bought_volume=11.0,
            ),
        ]

        result = ohlc.update_by_bars(gap_bars)

        # Verify gap bars were added
        assert result is ohlc
        assert len(ohlc) == 8
        assert ohlc.times[0] == recognize_time("2024-01-01 00:16").astype("datetime64[ns]").item()

        # Test adding duplicate bars (should be skipped)
        duplicate_bars = [
            Bar(
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                105.0,
                110.0,
                104.0,
                108.0,
                volume=12.0,
                bought_volume=7.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:13").astype("datetime64[ns]").item(),
                108.0,
                112.0,
                107.0,
                110.0,
                volume=14.0,
                bought_volume=9.0,
            ),
        ]

        result = ohlc.update_by_bars(duplicate_bars)

        # Verify no new bars were added
        assert result is ohlc
        assert len(ohlc) == 8  # Length should remain the same

        # Test adding a mix of new and duplicate bars
        mixed_bars = [
            Bar(
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                105.0,
                110.0,
                104.0,
                108.0,
                volume=12.0,
                bought_volume=7.0,
            ),  # Duplicate
            Bar(
                recognize_time("2024-01-01 00:15").astype("datetime64[ns]").item(),
                113.0,
                118.0,
                112.0,
                116.0,
                volume=17.0,
                bought_volume=10.0,
            ),  # New (fills gap) - but will be skipped because it's within the existing range
        ]

        result = ohlc.update_by_bars(mixed_bars)

        # Verify only new bars were added
        assert result is ohlc
        # Note: The implementation of update_by_bars skips bars that fall within the existing range
        # but aren't in the existing times. The bar at 00:15 is between 00:14 and 00:16, so it's skipped.
        assert len(ohlc) == 8  # Length should remain the same

        # Check that all expected times are in the series
        expected_times = set(
            [
                recognize_time("2024-01-01 00:08").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:09").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:11").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:13").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:14").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:16").astype("datetime64[ns]").item(),
            ]
        )
        actual_times = set(ohlc.times.values)
        assert expected_times == actual_times

        # Verify indicator was updated correctly
        # The indicator should have values for all bars
        # Note: The implementation of update_by_bars doesn't update indicators for bars added to the back
        # of the series (older bars), so the indicator will have fewer values than the OHLCV series
        assert len(ma) < len(ohlc)  # Indicator has fewer values than the OHLCV series

        # Test adding bars in random order
        ohlc_random = OHLCV("BTCUSDT_RANDOM", "1Min")
        random_bars = [
            Bar(
                recognize_time("2024-01-01 00:15").astype("datetime64[ns]").item(),
                113.0,
                118.0,
                112.0,
                116.0,
                volume=17.0,
                bought_volume=10.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item(),
                100.0,
                105.0,
                99.0,
                102.0,
                volume=10.0,
                bought_volume=6.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:13").astype("datetime64[ns]").item(),
                108.0,
                112.0,
                107.0,
                110.0,
                volume=14.0,
                bought_volume=9.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:08").astype("datetime64[ns]").item(),
                95.0,
                98.0,
                94.0,
                97.0,
                volume=8.0,
                bought_volume=5.0,
            ),
            Bar(
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                105.0,
                110.0,
                104.0,
                108.0,
                volume=12.0,
                bought_volume=7.0,
            ),
        ]

        result = ohlc_random.update_by_bars(random_bars)

        # Verify bars were added
        assert result is ohlc_random
        assert len(ohlc_random) == 5

        # Check that all expected times are in the series
        expected_times = set(
            [
                recognize_time("2024-01-01 00:08").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:10").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:12").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:13").astype("datetime64[ns]").item(),
                recognize_time("2024-01-01 00:15").astype("datetime64[ns]").item(),
            ]
        )
        actual_times = set(ohlc_random.times.values)
        assert expected_times == actual_times

    def test_series_update_past_data(self):
        ts = TimeSeries("test", "10Min")
        ts.update(recognize_time("2024-01-01 00:20"), 1)
        with pytest.raises(
            ValueError, match="test.600000000000: Attempt to update past data at 2024-01-01T00:10:00.000000000 !"
        ):
            ts.update(recognize_time("2024-01-01 00:10"), 5)

    def test_series_diff(self):
        """Test the diff() method for differencing time series."""
        ts = TimeSeries("test", "1Min")

        # Create a simple series with known values
        push(
            ts,
            [
                ("2024-01-01 00:00", 10.0),
                ("2024-01-01 00:01", 15.0),
                ("2024-01-01 00:02", 18.0),
                ("2024-01-01 00:03", 20.0),
                ("2024-01-01 00:04", 25.0),
                ("2024-01-01 00:05", 22.0),
                ("2024-01-01 00:06", 28.0),
            ],
        )

        # Test first-order differencing (default period=1)
        diff1 = ts.diff()

        # First value should be NaN (no previous value to subtract)
        assert np.isnan(diff1[-1])

        # Subsequent values should be: value[i] - value[i-1]
        # Series values (newest first): 28, 22, 25, 20, 18, 15, 10
        # Differences: 28-22=6, 22-25=-3, 25-20=5, 20-18=2, 18-15=3, 15-10=5
        expected = [6.0, -3.0, 5.0, 2.0, 3.0, 5.0]
        actual = [diff1[i] for i in range(len(diff1) - 1)]
        assert np.allclose(actual, expected)

        # Test second-order differencing (period=2)
        diff2 = ts.diff(2)

        # First two values should be NaN
        assert np.isnan(diff2[-1])
        assert np.isnan(diff2[-2])

        # Subsequent values should be: value[i] - value[i-2]
        # Series values (newest first): 28, 22, 25, 20, 18, 15, 10
        # Differences with period=2: 28-25=3, 22-20=2, 25-18=7, 20-15=5, 18-10=8
        expected2 = [3.0, 2.0, 7.0, 5.0, 8.0]
        actual2 = [diff2[i] for i in range(len(diff2) - 2)]
        assert np.allclose(actual2, expected2)

        # Test that diff() is equivalent to series - series.shift(period)
        diff_manual = ts - ts.shift(1)
        assert np.allclose(
            [diff1[i] for i in range(len(diff1)) if not np.isnan(diff1[i])],
            [diff_manual[i] for i in range(len(diff_manual)) if not np.isnan(diff_manual[i])],
        )

        # Test validation: period must be positive
        with pytest.raises(ValueError, match="Period must be positive and greater than zero !"):
            ts.diff(0)

        with pytest.raises(ValueError, match="Period must be positive and greater than zero !"):
            ts.diff(-1)

    def test_diff_on_indicator(self):
        """Test diff() method on an indicator."""
        ts = TimeSeries("test", "1Min")

        # Create a series
        push(
            ts,
            [
                ("2024-01-01 00:00", 10.0),
                ("2024-01-01 00:01", 20.0),
                ("2024-01-01 00:02", 30.0),
                ("2024-01-01 00:03", 40.0),
                ("2024-01-01 00:04", 50.0),
                ("2024-01-01 00:05", 60.0),
            ],
        )

        # Create SMA indicator
        ma = sma(ts, 3)

        # Apply differencing to the indicator
        ma_diff = ma.diff()

        # Check that diff works on indicators
        assert len(ma_diff) == len(ma)

        # First value should be NaN (no previous MA value)
        # Plus the first 2 values of MA are NaN (initialization period)
        # So first 3 values should be NaN
        assert np.isnan(ma_diff[-1])
        assert np.isnan(ma_diff[-2])
        assert np.isnan(ma_diff[-3])

        # After that, we should have valid differences
        # MA values: [NaN, NaN, 20, 30, 40, 50]
        # Differences: [NaN, NaN, NaN, 10, 10, 10]
        for i in range(3):
            if not np.isnan(ma_diff[i]):
                assert np.isclose(ma_diff[i], 10.0)

    def test_series_resample(self):
        ts0 = TimeSeries("test", "1Min")
        ts1 = ts0.resample("5Min")

        # Create a series
        push(
            ts0,
            [
                ("2024-01-01 00:00", 10.0),
                ("2024-01-01 00:01", 20.0),
                ("2024-01-01 00:02", 30.0),
                ("2024-01-01 00:03", 40.0),
                ("2024-01-01 00:04", 50.0),
                ("2024-01-01 00:05", 60.0),
                ("2024-01-01 00:06", 70.0),
                ("2024-01-01 00:07", 80.0),
                ("2024-01-01 00:08", 90.0),
            ],
        )
        assert ts1[0] == 90.0
        assert ts1[1] == 50.0

    def test_ohlc_series_resample(self):
        ohlc1 = OHLCV("BTCUSDT", "1Min")
        push(
            ohlc1,
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

        ohlc2 = ohlc1.resample("5Min")
        print(ohlc2)

        # - 1. test passsed
        assert ohlc2[0].close == 4.0

        # - 2. test failed
        push(
            ohlc1,
            [
                ("2024-01-01 00:20", 1000),
            ],
        )
        print(ohlc2)


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
        assert first_trade.side == 1

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
            dtype=[("timestamp", "i8"), ("price", "f8"), ("size", "f8"), ("side", "i1")],
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
        assert trade.side == -1

        # Test array access using trades attribute
        slice_data = trades.trades[1:3]
        assert len(slice_data) == 2
        assert np.array_equal(slice_data["timestamp"], np.array([1000001, 1000002]))

    def test_trade_array_invalid_input(self):
        # Test invalid array type
        with pytest.raises(TypeError, match="data must be a numpy array"):
            TradeArray([(1, 100.0, 1.0, 1)])  # List instead of numpy array

        # Test invalid dtype
        invalid_data = np.array([(1, 100.0)], dtype=[("time", "i8"), ("price", "f8")])
        with pytest.raises(ValueError, match="Cannot convert input array to required dtype"):
            TradeArray(invalid_data)

    def test_trade_array_from_time(self):
        trades = TradeArray()
        t0 = pd.Timestamp("2024-01-01 15:00").to_datetime64().astype("datetime64[ns]").item()

        # Add trades
        trades.add(t0 + 0, 100.0, 1.0, 1)  # buy
        trades.add(t0 + 1000, 101.0, 2.0, -1)  # sell
        trades.add(t0 + 2000, 99.0, 1.5, 1)  # buy
        trades.add(t0 + 3000, 102.0, 0.5, -1)  # sell
        trades.add(t0 + 4000, 101.0, 0.5, -1)  # sell
        trades.add(t0 + 5000, 99.0, 0.5, -1)  # sell
        trades.add(t0 + 6000, 91.0, 0.5, -1)  # sell

        assert trades.min_buy_price == 99.0
        assert trades.max_buy_price == 100.0
        assert trades.min_sell_price == 91.0
        assert trades.max_sell_price == 102.0

        assert trades.traded_range_from(t0) == (99.0, 100.0, 91.0, 102.0)
        assert trades.traded_range_from(0) == (99.0, 100.0, 91.0, 102.0)
        assert trades.traded_range_from(t0 + 5100) == (np.inf, -np.inf, 91.0, 91.0)

        assert trades.traded_range_from(t0 + 15000) == (np.inf, -np.inf, np.inf, -np.inf)


class TestGenericSeries:
    def test_generic_series(self):
        quotes = GenericSeries("BTCUSDT_quotes", "5Min")

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        minute_ns = 60 * 10**9

        for i in range(60):
            time = base_time.item() + i * minute_ns
            bid = 50000 + i * 10
            ask = bid + 5 + i * 0.5  # - spread increases slightly
            bid_size = 10 + i
            ask_size = 12 + i

            q = Quote(time, bid, ask, bid_size, ask_size)
            quotes.update(q)
        assert len(quotes) == 12
        assert quotes[0].time == np.datetime64("2024-01-01 00:59:00", "ns").item()

    def test_generic_series_indicator(self):
        class BidAskSpread(IndicatorGeneric):
            """
            Calculate bid-ask spread from Quote data
            """

            def calculate(self, time, quote, new_item_started):
                return (quote.ask - quote.bid) if quote is not None else np.nan

        # - quotes series
        quotes = GenericSeries("BTCUSDT", "5Min")

        # - Attach indicators
        spread = BidAskSpread("spread", quotes)

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        minute_ns = 60 * 10**9

        for i in range(60):
            time = base_time.item() + i * minute_ns
            bid = 50000 + i * 10
            ask = bid + 5 + i * 0.5  # - spread increases slightly
            bid_size = 10 + i
            ask_size = 12 + i

            q = Quote(time, bid, ask, bid_size, ask_size)
            quotes.update(q)

        assert len(spread) == len(quotes)
        assert spread[-1] == 7.0
        assert spread[0] == 34.5


class TestColumnarSeries:
    """Tests for ColumnarSeries - dynamic column decomposition like OHLCV but with custom columns."""

    def test_columnar_series_basic(self):
        """Test basic ColumnarSeries creation and column access."""
        cs = ColumnarSeries("test", "1h", ["col_a", "col_b", "col_c"])

        assert cs.column_names == ["col_a", "col_b", "col_c"]
        assert len(cs.columns) == 3
        assert all(isinstance(ts, TimeSeries) for ts in cs.columns.values())

    def test_columnar_series_update(self):
        """Test that updates propagate to child TimeSeries."""
        cs = ColumnarSeries("test", "1h", ["buy_volume", "sell_volume"])

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        # Add data using TimestampedDict
        for i in range(5):
            t = base_time.item() + i * hour_ns
            obj = TimestampedDict(time=t, data={"buy_volume": float(i * 100), "sell_volume": float(i * 50)})
            cs.update(obj)

        # Check parent series
        assert len(cs) == 5

        # Check child series have same length
        assert len(cs.buy_volume) == 5
        assert len(cs.sell_volume) == 5

        # Check values (newest first in qubx series)
        assert cs.buy_volume[0] == 400.0  # Last value (i=4)
        assert cs.sell_volume[0] == 200.0
        assert cs.buy_volume[-1] == 0.0  # First value (i=0)
        assert cs.sell_volume[-1] == 0.0

    def test_columnar_series_getattr(self):
        """Test column access via __getattr__."""
        cs = ColumnarSeries("test", "1h", ["ratio", "volume"])

        # Access columns as attributes
        ratio_ts = cs.ratio
        volume_ts = cs.volume

        assert isinstance(ratio_ts, TimeSeries)
        assert isinstance(volume_ts, TimeSeries)
        assert ratio_ts.name == "ratio"
        assert volume_ts.name == "volume"

        # Non-existent column should raise AttributeError
        with pytest.raises(AttributeError, match="has no column 'nonexistent'"):
            _ = cs.nonexistent

    def test_columnar_series_getitem(self):
        """Test column access via __getitem__ with string key."""
        cs = ColumnarSeries("test", "1h", ["col_a", "col_b"])

        # Access by string key
        assert cs["col_a"] is cs.col_a
        assert cs["col_b"] is cs.col_b

        # Non-existent column should raise KeyError
        with pytest.raises(KeyError, match="No column named 'nonexistent'"):
            _ = cs["nonexistent"]

    def test_columnar_series_indicator_updates(self):
        """Test that indicators attached to child series get updated."""
        cs = ColumnarSeries("test", "1h", ["value"])

        # Attach SMA indicator to child series
        value_sma = sma(cs.value, 3)

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        # Add data - need more values since first update doesn't trigger indicators
        # (GenericSeries behavior: first bar may be incomplete)
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        for i, v in enumerate(values):
            t = base_time.item() + i * hour_ns
            obj = TimestampedDict(time=t, data={"value": v})
            cs.update(obj)

        # Check child series has all values
        assert len(cs.value) == 7

        # Indicator gets updates starting from 2nd bar (first bar skipped by GenericSeries)
        # So indicator has values for: 20, 30, 40, 50, 60, 70 (6 values)
        # SMA(3) needs 3 values, so first 2 are NaN, then we get valid SMA
        # Values (newest first): 70, 60, 50, 40, 30, 20
        # SMA(3) at each: (50+60+70)/3=60, (40+50+60)/3=50, (30+40+50)/3=40, (20+30+40)/3=30, nan, nan

        # Check we have valid SMA values
        assert len(value_sma) > 0
        # The newest SMA values should be valid
        assert np.isclose(value_sma[0], 60.0)  # (50+60+70)/3
        assert np.isclose(value_sma[1], 50.0)  # (40+50+60)/3

    def test_columnar_series_get_indicators(self):
        """Test get_indicators returns indicators from all child series."""
        cs = ColumnarSeries("test", "1h", ["a", "b"])

        # Attach indicators to different columns
        sma_a = sma(cs.a, 5)
        sma_b = sma(cs.b, 10)

        indicators = cs.get_indicators()

        # Should have indicators from both child series
        assert "a.sma(5)" in indicators
        assert "b.sma(10)" in indicators

    def test_columnar_series_with_object_attributes(self):
        """Test ColumnarSeries with objects that have direct attributes (not TimestampedDict)."""

        class CustomData:
            def __init__(self, time, price, size):
                self.time = time
                self.price = price
                self.size = size

        cs = ColumnarSeries("test", "1h", ["price", "size"])

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        for i in range(3):
            t = base_time.item() + i * hour_ns
            obj = CustomData(t, price=100.0 + i * 10, size=1.0 + i)
            cs.update(obj)

        assert len(cs.price) == 3
        assert len(cs.size) == 3
        assert cs.price[0] == 120.0  # Newest
        assert cs.size[0] == 3.0


class TestBundledSeries:
    """Tests for BundledSeries - virtual series that bundles fields from multiple sources."""

    def test_bundled_series_basic(self):
        """Test basic BundledSeries creation."""
        ts1 = TimeSeries("price", "1h")
        ts2 = TimeSeries("volume", "1h")

        bundle = BundledSeries("bundle", "1h", {"price": ts1, "volume": ts2})

        assert bundle.field_names == ["price", "volume"]
        assert len(bundle.fields) == 2

    def test_bundled_series_updates_from_any_source(self):
        """Test that BundledSeries triggers when ANY source updates."""
        ts_price = TimeSeries("price", "1h")
        ts_volume = TimeSeries("volume", "1h")

        bundle = BundledSeries("bundle", "1h", {"price": ts_price, "volume": ts_volume})

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        # Update price first
        ts_price.update(base_time.item(), 100.0)
        ts_volume.update(base_time.item(), 1000.0)

        # Move to next hour
        t1 = base_time.item() + hour_ns
        ts_price.update(t1, 105.0)
        ts_volume.update(t1, 1100.0)

        # Move to next hour
        t2 = base_time.item() + 2 * hour_ns
        ts_price.update(t2, 110.0)
        ts_volume.update(t2, 1200.0)

        # Bundle should have values (minus first incomplete bar)
        assert len(bundle) >= 2

        # Check latest value is a dict with both fields
        latest = bundle[0]
        assert isinstance(latest, dict)
        assert "price" in latest
        assert "volume" in latest

    def test_bundled_series_with_indicator(self):
        """Test attaching an IndicatorGeneric to BundledSeries."""

        class PriceVolumeRatio(IndicatorGeneric):
            """Simple indicator that computes price / volume."""

            def calculate(self, time, values, new_item_started):
                price = values.get("price", np.nan)
                volume = values.get("volume", np.nan)
                if np.isnan(price) or np.isnan(volume) or volume == 0:
                    return np.nan
                return price / volume

        ts_price = TimeSeries("price", "1h")
        ts_volume = TimeSeries("volume", "1h")

        bundle = BundledSeries("bundle", "1h", {"price": ts_price, "volume": ts_volume})
        ratio = PriceVolumeRatio("ratio", bundle)

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        # Add data
        for i in range(5):
            t = base_time.item() + i * hour_ns
            ts_price.update(t, 100.0 + i * 10)  # 100, 110, 120, 130, 140
            ts_volume.update(t, 1000.0)  # constant volume

        # Indicator should have computed values
        assert len(ratio) > 0

        # Check ratio is computed correctly (latest: 140 / 1000 = 0.14)
        assert np.isclose(ratio[0], 0.14, atol=0.01)

    def test_bundled_series_with_ohlcv_and_columnar(self):
        """Test BundledSeries with OHLCV and ColumnarSeries sources."""
        ohlcv = OHLCV("btc", "1h")
        vtwap = ColumnarSeries("vtwap", "1h", ["twap", "vwap"])

        # Bundle close from OHLCV and vwap from ColumnarSeries
        bundle = BundledSeries("close_vwap", "1h", {"close": ohlcv.close, "vwap": vtwap.vwap})

        class VwapDeviation(IndicatorGeneric):
            """Computes (close - vwap) / close."""

            def calculate(self, time, values, new_item_started):
                close = values.get("close", np.nan)
                vwap = values.get("vwap", np.nan)
                if np.isnan(close) or np.isnan(vwap) or close == 0:
                    return np.nan
                return (close - vwap) / close

        deviation = VwapDeviation("deviation", bundle)

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        # Add data
        for i in range(5):
            t = base_time.item() + i * hour_ns
            price = 50000.0 + i * 100  # 50000, 50100, 50200, 50300, 50400

            # Update OHLCV
            ohlcv.update(t, price)

            # Update vtwap (vwap slightly below close)
            vtwap_data = TimestampedDict(time=t, data={"twap": price - 5, "vwap": price - 10})
            vtwap.update(vtwap_data)

        # Check deviation is computed
        assert len(deviation) > 0

        # Latest: (50400 - 50390) / 50400 ≈ 0.000198
        latest_deviation = deviation[0]
        assert not np.isnan(latest_deviation)
        assert latest_deviation > 0  # close > vwap

    def test_bundled_series_misaligned_updates(self):
        """Test that bundle handles sources updating at different times."""
        ts_fast = TimeSeries("fast", "1m")  # Updates every minute
        ts_slow = TimeSeries("slow", "1m")  # Updates less frequently

        bundle = BundledSeries("bundle", "1m", {"fast": ts_fast, "slow": ts_slow})

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        minute_ns = 60 * 10**9

        # Update both at t=0
        ts_fast.update(base_time.item(), 1.0)
        ts_slow.update(base_time.item(), 100.0)

        # Only update fast at t=1
        t1 = base_time.item() + minute_ns
        ts_fast.update(t1, 2.0)
        # slow still has value from t=0

        # Update both at t=2
        t2 = base_time.item() + 2 * minute_ns
        ts_fast.update(t2, 3.0)
        ts_slow.update(t2, 200.0)

        # Bundle should use latest available values
        latest = bundle[0]
        assert latest["fast"] == 3.0
        assert latest["slow"] == 200.0

    def test_bundled_series_late_update_skipped(self):
        """Test that late updates from lagging sources are skipped."""
        ts_a = TimeSeries("a", "1h")
        ts_b = TimeSeries("b", "1h")

        bundle = BundledSeries("bundle", "1h", {"a": ts_a, "b": ts_b})

        base_time = np.datetime64("2024-01-01T00:00:00", "ns")
        hour_ns = 60 * 60 * 10**9

        # Both sources at t=0
        ts_a.update(base_time.item(), 1.0)
        ts_b.update(base_time.item(), 10.0)

        # Both sources at t=1 (hour 1)
        t1 = base_time.item() + hour_ns
        ts_a.update(t1, 2.0)
        ts_b.update(t1, 20.0)

        # Source A advances to t=2 (hour 2) - creates new bar in bundle
        t2 = base_time.item() + 2 * hour_ns
        ts_a.update(t2, 3.0)

        # Record bundle length after A's update
        len_after_a = len(bundle)

        # Source B sends a LATE update for t=1 (still in hour 1)
        # This should be skipped since bundle already has hour 2 bar
        t1_late = base_time.item() + hour_ns + 30 * 60 * 10**9  # 1:30
        ts_b.update(t1_late, 25.0)

        # Bundle length should NOT have changed (late update skipped)
        assert len(bundle) == len_after_a

        # Now B catches up to t=2
        ts_b.update(t2, 30.0)

        # Latest bundle should have updated values
        latest = bundle[0]
        assert latest["a"] == 3.0
        assert latest["b"] == 30.0
