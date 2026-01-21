"""
Tests for data transformers handling None values in optional columns.
"""

import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import DataType
from qubx.data.transformers import OHLCVSeries, TypedRecords


class TestOHLCVSeriesNoneHandling:
    """
    Tests for OHLCVSeries transformer handling None values in optional columns.
    """

    def test_ohlcv_series_with_none_optional_values(self):
        """
        Test that OHLCVSeries handles None values in count, taker_buy_volume, etc.
        """
        # - create sample data with None values in optional columns
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        hour_ns = 3600 * 1_000_000_000

        raw_data = [
            np.array(
                [base_time, "BTCUSD", 42175.0, 42308.0, 42076.0, 42246.0, 32.29, 1363488.1, None, None, None],
                dtype=object,
            ),
            np.array(
                [base_time + hour_ns, "BTCUSD", 42246.0, 42246.0, 41969.0, 41981.0, 0.2055, 8634.663, None, None, None],
                dtype=object,
            ),
            np.array(
                [
                    base_time + 2 * hour_ns,
                    "BTCUSD",
                    41981.0,
                    42440.0,
                    41981.0,
                    42440.0,
                    2.6476,
                    111593.29,
                    None,
                    None,
                    None,
                ],
                dtype=object,
            ),
        ]

        names = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]

        transformer = OHLCVSeries(timestamp_units="ns")
        ohlc = transformer.process_data("BTCUSD", DataType.OHLC["1h"], raw_data, names, index=0)

        assert len(ohlc) == 3
        # - verify last bar values (OHLCV stores newest first, so index -1 is oldest)
        assert ohlc.open[-1] == 42175.0
        assert ohlc.close[-1] == 42246.0
        assert ohlc.volume[-1] == pytest.approx(32.29, rel=1e-3)
        # - None values should be converted to 0
        assert ohlc.trade_count[-1] == 0
        assert ohlc.trade_count[0] == 0  # - all bars should have 0 trade_count

    def test_ohlcv_series_with_all_valid_values(self):
        """
        Test that OHLCVSeries works correctly when all values are present.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        hour_ns = 3600 * 1_000_000_000

        raw_data = [
            np.array(
                [base_time, "BTCUSD", 42175.0, 42308.0, 42076.0, 42246.0, 32.29, 1363488.1, 100, 16.0, 680000.0],
                dtype=object,
            ),
            np.array(
                [base_time + hour_ns, "BTCUSD", 42246.0, 42246.0, 41969.0, 41981.0, 0.2055, 8634.663, 50, 0.1, 4200.0],
                dtype=object,
            ),
        ]

        names = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]

        transformer = OHLCVSeries(timestamp_units="ns")
        ohlc = transformer.process_data("BTCUSD", DataType.OHLC["1h"], raw_data, names, index=0)

        assert len(ohlc) == 2
        # - OHLCV stores newest first, so -1 is oldest (first record), 0 is newest (second record)
        assert ohlc.trade_count[-1] == 100
        assert ohlc.trade_count[0] == 50


class TestTypedRecordsNoneHandling:
    """
    Tests for TypedRecords transformer handling None values when creating Bar objects.
    """

    def test_typed_records_bar_with_none_optional_values(self):
        """
        Test that TypedRecords handles None values when creating Bar objects.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        hour_ns = 3600 * 1_000_000_000

        raw_data = [
            np.array(
                [base_time, 42175.0, 42308.0, 42076.0, 42246.0, 32.29, None, None, None, None],
                dtype=object,
            ),
            np.array(
                [base_time + hour_ns, 42246.0, 42246.0, 41969.0, 41981.0, 0.2055, None, None, None, None],
                dtype=object,
            ),
        ]

        names = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bought_volume",
            "volume_quote",
            "bought_volume_quote",
            "trade_count",
        ]

        transformer = TypedRecords(timestamp_units="ns")
        bars = transformer.process_data("BTCUSD", DataType.OHLC["1h"], raw_data, names, index=0)

        assert len(bars) == 2
        # - verify first bar
        bar = bars[0]
        assert bar.open == 42175.0
        assert bar.close == 42246.0
        assert bar.volume == pytest.approx(32.29, rel=1e-3)
        # - None values should be converted to defaults
        assert bar.bought_volume == 0.0
        assert bar.volume_quote == 0.0
        assert bar.bought_volume_quote == 0.0
        assert bar.trade_count == 0
