"""
Tests for data transformers handling None values in optional columns.
"""

import numpy as np
import pandas as pd
import pytest

from qubx.core.basics import DataType
from qubx.core.series import OrderBook, Quote, Trade
from qubx.data.containers import RawData
from qubx.data.transformers import OHLCVSeries, TickSeries, TypedGenericSeries, TypedRecords


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

        r_data = [
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

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1h"], pd.DataFrame(r_data, columns=names))

        transformer = OHLCVSeries(timestamp_units="ns")
        ohlc = transformer.process_data(raw_data)

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

        r_data = [
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

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1h"], pd.DataFrame(r_data, columns=names))
        ohlc = transformer.process_data(raw_data)

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

        r_data = [
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

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1h"], pd.DataFrame(r_data, columns=names))

        transformer = TypedRecords(timestamp_units="ns")
        bars = transformer.process_data(raw_data)

        assert len(bars) == 2
        # - verify first bar
        bar = bars[0]
        assert bar.open == 42175.0  # type: ignore
        assert bar.close == 42246.0  # type: ignore
        assert bar.volume == pytest.approx(32.29, rel=1e-3)  # type: ignore

        # - None values should be converted to defaults
        assert bar.bought_volume == 0.0  # type: ignore
        assert bar.volume_quote == 0.0  # type: ignore
        assert bar.bought_volume_quote == 0.0  # type: ignore
        assert bar.trade_count == 0  # type: ignore


class TestOrderbookTransformation:
    """
    Tests for orderbook data transformation using TypedRecords and TypedGenericSeries.
    """

    def test_typed_records_orderbook(self):
        """
        Test that TypedRecords correctly transforms orderbook data into OrderBook snapshots.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        sec_ns = 1_000_000_000

        # - orderbook data: level > 0 for asks, level < 0 for bids
        # - build_snapshots collects snapshot when time changes
        # - so we need 3 timestamps to get 2 snapshots
        r_data = [
            # - first snapshot at base_time
            [base_time, -1, 100.0, 1.0],  # - bid level 1
            [base_time, -2, 99.0, 2.0],  # - bid level 2
            [base_time, 1, 101.0, 1.5],  # - ask level 1
            [base_time, 2, 102.0, 2.5],  # - ask level 2
            # - second snapshot (triggers collection of first)
            [base_time + sec_ns, -1, 100.5, 1.2],
            [base_time + sec_ns, 1, 101.5, 1.8],
            # - third timestamp (triggers collection of second)
            [base_time + 2 * sec_ns, -1, 100.2, 1.1],
            [base_time + 2 * sec_ns, 1, 101.2, 1.6],
        ]

        names = ["timestamp", "level", "price", "size"]
        df = pd.DataFrame(r_data, columns=names)

        raw_data = RawData.from_pandas("BTCUSD", DataType.ORDERBOOK, df)

        transformer = TypedRecords(timestamp_units="ns")
        snapshots = transformer.process_data(raw_data)

        # - should produce 3 snapshots (including final)
        assert len(snapshots) == 3
        assert isinstance(snapshots[0], OrderBook)

        # - verify first snapshot
        ob = snapshots[0]
        assert ob.top_bid == 100.0
        assert ob.top_ask == 101.0
        assert ob.tick_size == pytest.approx(1.0)  # - tick_size stores the spread
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        assert ob.bids[0] == 1.0  # - size at bid level 1
        assert ob.bids[1] == 2.0  # - size at bid level 2
        assert ob.asks[0] == 1.5  # - size at ask level 1
        assert ob.asks[1] == 2.5  # - size at ask level 2

        # - verify second snapshot
        ob2 = snapshots[1]
        assert ob2.top_bid == 100.5
        assert ob2.top_ask == 101.5

        # - verify third (final) snapshot
        ob3 = snapshots[2]
        assert ob3.top_bid == 100.2
        assert ob3.top_ask == 101.2

    def test_typed_generic_series_orderbook(self):
        """
        Test that TypedGenericSeries correctly transforms orderbook data into GenericSeries.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        sec_ns = 1_000_000_000

        r_data = [
            [base_time, -1, 100.0, 1.0],
            [base_time, 1, 101.0, 1.5],
            [base_time + sec_ns, -1, 100.5, 1.2],
            [base_time + sec_ns, 1, 101.5, 1.8],
            [base_time + 2 * sec_ns, -1, 100.2, 1.1],
            [base_time + 2 * sec_ns, 1, 101.2, 1.6],
        ]

        names = ["timestamp", "level", "price", "size"]
        df = pd.DataFrame(r_data, columns=names)

        raw_data = RawData.from_pandas("BTCUSD", DataType.ORDERBOOK, df)

        transformer = TypedGenericSeries(timestamp_units="ns")
        gens = transformer.process_data(raw_data)

        # - should have collected snapshots
        assert len(gens) >= 1


class TestTickSeriesTransformation:
    """
    Tests for TickSeries transformer that converts OHLC bars to simulated ticks.
    """

    def test_tick_series_quotes_only(self):
        """
        Test TickSeries generates quotes from OHLC data.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        hour_ns = 3600 * 1_000_000_000

        r_data = [
            [base_time, 100.0, 102.0, 98.0, 101.0, 1000.0],
            [base_time + hour_ns, 101.0, 103.0, 99.0, 102.0, 1500.0],
        ]

        names = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(r_data, columns=names)

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1h"], df)

        transformer = TickSeries(trades=False, quotes=True, spread=2.0)
        ticks = transformer.process_data(raw_data)

        # - 4 quotes per bar (open, mid1, mid2, close)
        assert len(ticks) == 4 * 2
        assert all(isinstance(t, Quote) for t in ticks)

        # - verify first quote (opening)
        q0 = ticks[0]
        assert q0.bid == pytest.approx(99.0)  # - open - spread/2
        assert q0.ask == pytest.approx(101.0)  # - open + spread/2

    def test_tick_series_with_trades(self):
        """
        Test TickSeries generates both quotes and trades from OHLC data.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        hour_ns = 3600 * 1_000_000_000

        r_data = [
            [base_time, 100.0, 102.0, 98.0, 101.0, 1000.0],
            [base_time + hour_ns, 101.0, 103.0, 99.0, 100.0, 1500.0],
        ]

        names = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(r_data, columns=names)

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1h"], df)

        transformer = TickSeries(trades=True, quotes=True, spread=2.0)
        ticks = transformer.process_data(raw_data)

        # - 4 quotes + 3 trades per bar = 7 ticks per bar
        assert len(ticks) == 7 * 2

        # - verify we have both quotes and trades
        quotes = [t for t in ticks if isinstance(t, Quote)]
        trades = [t for t in ticks if isinstance(t, Trade)]
        assert len(quotes) == 4 * 2
        assert len(trades) == 3 * 2

    def test_tick_series_trades_only(self):
        """
        Test TickSeries generates trades only from OHLC data.
        """
        base_time = pd.Timestamp("2022-03-23 10:00:00").value
        hour_ns = 3600 * 1_000_000_000

        r_data = [
            [base_time, 100.0, 102.0, 98.0, 101.0, 1000.0],
            [base_time + hour_ns, 101.0, 103.0, 99.0, 100.0, 1500.0],
        ]

        names = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(r_data, columns=names)

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1h"], df)

        transformer = TickSeries(trades=True, quotes=False, spread=2.0)
        ticks = transformer.process_data(raw_data)

        # - 3 trades per bar
        assert len(ticks) == 3 * 2
        assert all(isinstance(t, Trade) for t in ticks)

    def test_tick_series_stock_daily_session(self):
        """
        Test TickSeries with STOCKS daily session generates quotes at correct times.
        For daily bars, quotes should be generated within stock market hours (9:30-16:00).
        """
        # - daily bars
        base_time = pd.Timestamp("2022-03-23 00:00:00").value
        day_ns = 24 * 3600 * 1_000_000_000

        r_data = [
            [base_time, 100.0, 102.0, 98.0, 101.0, 1000.0],
            [base_time + day_ns, 101.0, 103.0, 99.0, 102.0, 1500.0],
        ]

        names = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(r_data, columns=names)

        raw_data = RawData.from_pandas("BTCUSD", DataType.OHLC["1d"], df)

        # - use STOCKS session (9:30 - 16:00)
        transformer = TickSeries(trades=False, quotes=True, daily_session_start_end="STOCKS")
        ticks = transformer.process_data(raw_data)

        # - first quote should be near 9:30 AM
        first_quote = ticks[0]
        first_time = pd.Timestamp(first_quote.time, unit="ns")
        assert first_time.hour == 9
        assert first_time.minute >= 30

        # - last quote of first bar should be near 16:00 (15:59:xx)
        last_quote_bar1 = ticks[3]  # - 4th quote is closing quote of first bar
        last_time = pd.Timestamp(last_quote_bar1.time, unit="ns")
        assert last_time.hour == 15
        assert last_time.minute == 59
