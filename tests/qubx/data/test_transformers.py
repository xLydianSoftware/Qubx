"""
Tests for data transformers handling None values in optional columns.
"""

import numpy as np
import pandas as pd
import pytest

from qubx.backtester.simulated_data import EmulatedBarSequence, EmulatedTickSequence
from qubx.backtester.utils import STOCK_DAILY_SESSION
from qubx.core.basics import Bar, DataType, TimestampedDict
from qubx.core.series import OrderBook, Quote, Trade
from qubx.data.containers import RawData, RawMultiData
from qubx.data.transformers import OHLCVSeries, PandasFrame, TypedGenericSeries, TypedRecords


class TestsTransformations:
    def test_raw_data_container(self):
        r1 = RawData.from_pandas(
            "TEST1",
            DataType.TRADE,
            pd.DataFrame({"price": [100, 120, 190], "time": [1000, 2000, 3000], "size": [0.5, 0.3, 1.5]}),
        )

        assert len(r1) == 3
        assert r1.index == 1
        assert r1.get_time_interval() == (1000, 3000)

    def test_transformations(self):
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        # - Trade data
        r1 = RawData.from_pandas(
            "TEST1",
            DataType.TRADE,
            pd.DataFrame(
                {
                    "time": [t0 + k * dt for k in range(24)],
                    "price": [100 + k for k in range(24)],
                    "size": [k * 0.5 for k in range(24)],
                }
            ),
        )
        t1 = r1.transform(TypedRecords())
        assert isinstance(t1[0], Trade)
        assert t1[0].price == 100.0
        assert t1[1].size == 0.5

        t2 = r1.transform(PandasFrame())
        assert isinstance(t2, pd.DataFrame)
        assert isinstance(t2.index, pd.DatetimeIndex)
        assert t2.index[0] == pd.Timestamp("2020-01-01 00:00:00")

        # - Data type is RECORD - we expect TimestampedDict after transformation
        r2 = RawData.from_pandas(
            "TEST1",
            DataType.RECORD,
            pd.DataFrame(
                {
                    "time": [t0 + k * dt for k in range(24)],
                    "price": [100 + k for k in range(24)],
                    "size": [k * 0.5 for k in range(24)],
                }
            ),
        )
        t2 = r2.transform(TypedRecords())
        assert isinstance(t2[0], TimestampedDict)
        assert t2[0].data
        assert t2[0].data["price"] == 100.0
        assert t2[1]["price"] == 101.0

        # - OHLC series
        r3 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0 + k * dt for k in range(24)],
                    "open": [100 + k for k in range(24)],
                    "high": [100 + 1.02 * k for k in range(24)],
                    "low": [100 - 1.02 * k for k in range(24)],
                    "close": [100 + 1.01 * k for k in range(24)],
                    "volume": [100 * (k + 1) for k in range(24)],
                }
            ),
        )
        t3 = r3.transform(OHLCVSeries())
        assert len(r3) == 24
        assert t3["volume"][-1] == 100.0
        assert t3["open"][-1] == 100.0

    def test_raw_multi_data_container(self):
        times = pd.date_range("2020-01-01", periods=24, freq="1h")

        r1 = RawData.from_pandas(
            "BTCUSDT",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": times,
                    "open": [100 + k for k in range(24)],
                    "high": [100 + 1.02 * k for k in range(24)],
                    "low": [100 - 1.02 * k for k in range(24)],
                    "close": [100 + 1.01 * k for k in range(24)],
                    "volume": [100 * (k + 1) for k in range(24)],
                }
            ),
        )
        r2 = RawData.from_pandas(
            "ETHUSDT",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": times,
                    "open": [10 + k for k in range(24)],
                    "high": [10 + 1.02 * k for k in range(24)],
                    "low": [10 - 1.02 * k for k in range(24)],
                    "close": [10 + 1.01 * k for k in range(24)],
                    "volume": [200 * (k + 1) for k in range(24)],
                }
            ),
        )

        rmd = RawMultiData([r1, r2])
        f1 = rmd.transform(PandasFrame(False))
        f2 = rmd.transform(PandasFrame(True))
        assert all(f1.columns.get_level_values(0).unique().to_numpy() == ["BTCUSDT", "ETHUSDT"])
        assert all(f2.index.get_level_values(1).unique().to_numpy() == ["BTCUSDT", "ETHUSDT"])


class TestPandasFrameLongFormat:
    """
    Tests for PandasFrame.combine_data auto-pivot of long-format data.

    Long-format data (e.g. FUNDAMENTAL from QuestDB) returns rows like
    (timestamp, asset, metric, value) — multiple rows per timestamp per symbol.
    PandasFrame(False).combine_data must pivot to wide before column-concat.
    """

    def _make_fundamental_raw(self, symbol: str, dates: pd.DatetimeIndex, metrics: dict) -> RawData:
        """
        Build a RawData in long format: (timestamp, asset, metric, value).
        """
        rows = []
        for ts in dates:
            for metric, values in metrics.items():
                rows.append({"timestamp": ts, "asset": symbol, "metric": metric, "value": values[ts]})
        df = pd.DataFrame(rows)
        return RawData.from_pandas(symbol, DataType.FUNDAMENTAL, df)

    def test_to_pd_false_pivots_long_format(self):
        """
        to_pd(False) on long-format multi-symbol data should auto-pivot to wide
        and return a DataFrame with (symbol, metric) MultiIndex columns.
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="1D")
        btc_mktcap = {ts: float(i * 1e12) for i, ts in enumerate(dates)}
        btc_vol = {ts: float(i * 1e10) for i, ts in enumerate(dates)}
        eth_mktcap = {ts: float(i * 0.5e12) for i, ts in enumerate(dates)}
        eth_vol = {ts: float(i * 0.5e10) for i, ts in enumerate(dates)}

        btc = self._make_fundamental_raw("BTC", dates, {"market_cap": btc_mktcap, "total_volume": btc_vol})
        eth = self._make_fundamental_raw("ETH", dates, {"market_cap": eth_mktcap, "total_volume": eth_vol})
        multi = RawMultiData([btc, eth])

        df = multi.to_pd(False)

        # - should have unique DatetimeIndex
        assert df.index.is_unique
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 5

        # - columns should be MultiIndex (symbol, metric)
        assert isinstance(df.columns, pd.MultiIndex)
        symbols = df.columns.get_level_values(0).unique().tolist()
        assert set(symbols) == {"BTC", "ETH"}
        metrics = df.columns.get_level_values(1).unique().tolist()
        assert set(metrics) == {"market_cap", "total_volume"}

    def test_to_pd_false_handles_duplicate_timestamp_metric(self):
        """
        When DB has duplicate (timestamp, metric) rows (data re-ingestion / corrections),
        pivot_table's aggfunc='last' should resolve them without raising.
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="1D")
        # - duplicate row: same (timestamp, asset, metric) with different values
        rows = []
        for ts in dates:
            rows.append({"timestamp": ts, "asset": "BTC", "metric": "market_cap", "value": 1.0})
            rows.append({"timestamp": ts, "asset": "BTC", "metric": "market_cap", "value": 2.0})  # - duplicate
            rows.append({"timestamp": ts, "asset": "BTC", "metric": "total_volume", "value": 3.0})
        df = pd.DataFrame(rows)
        btc = RawData.from_pandas("BTC", DataType.FUNDAMENTAL, df)
        multi = RawMultiData([btc])

        result = multi.to_pd(False)

        # - should resolve to unique index, last value wins for the duplicate
        assert result.index.is_unique
        assert result["BTC"]["market_cap"].iloc[0] == 2.0
        assert result["BTC"]["total_volume"].iloc[0] == 3.0


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

    def test_orderbook(self):
        # - orderbook
        p0 = 100
        rows = []
        for k, t in enumerate(pd.date_range("2020-01-02 00:00:00", periods=10000, freq="1min")):
            for lvl in range(-200, 200):  # - 200 levels for each side
                if lvl == 0:
                    continue
                rows.append(
                    {
                        "timestamp": t,
                        "symbol": "BTCUSDT",
                        "level": lvl,
                        "price": p0 + k / 10 + lvl / 10,
                        "size": (k + abs(lvl)) * 10,
                    }
                )
        r4 = RawData.from_pandas("TEST1", DataType.ORDERBOOK, pd.DataFrame(rows))
        t4 = r4.transform(TypedRecords())
        assert all(t4[0].bids[:5] == np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        assert all(t4[0].asks[:5] == np.array([10.0, 20.0, 30.0, 40.0, 50.0]))


class TestEmulatedSequenceTransformation:
    """
    Tests for EmulatedTickSequence transformer that converts OHLC bars to simulated ticks.
    """

    def test_tick_series_quotes_only(self):
        """
        Test EmulatedTickSequence generates quotes from OHLC data.
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

        transformer = EmulatedTickSequence(trades=False, quotes=True, spread=2.0)
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
        Test EmulatedTickSequence generates both quotes and trades from OHLC data.
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

        transformer = EmulatedTickSequence(trades=True, quotes=True, spread=2.0)
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
        Test EmulatedTickSequence generates trades only from OHLC data.
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

        transformer = EmulatedTickSequence(trades=True, quotes=False, spread=2.0)
        ticks = transformer.process_data(raw_data)

        # - 3 trades per bar
        assert len(ticks) == 3 * 2
        assert all(isinstance(t, Trade) for t in ticks)

    def test_tick_series_stock_daily_session(self):
        """
        Test EmulatedTickSequence with STOCKS daily session generates quotes at correct times.
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
        transformer = EmulatedTickSequence(trades=False, quotes=True, daily_session_start_end=STOCK_DAILY_SESSION)
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

    def test_ticks_transformations(self):
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()
        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0 + k * dt for k in range(2)],
                    "open": [100 + k for k in range(2)],
                    "high": [100 + 1.02 * k for k in range(2)],
                    "low": [100 - 1.02 * k for k in range(2)],
                    "close": [100 + 1.01 * k for k in range(2)],
                    "volume": [100 * (k + 1) for k in range(2)],
                }
            ),
        )
        # - only quotes
        t1 = r1.transform(EmulatedTickSequence(trades=False, spread=2, default_ask_size=100, default_bid_size=200))
        assert len(t1) == 4 * 2
        assert isinstance(t1[0], Quote)
        assert t1[0].mid_price() == 100.0
        assert t1[0].ask_size == 100
        assert t1[0].bid_size == 200

        # - quotes with trades
        t2 = r1.transform(EmulatedTickSequence(trades=True, spread=2))
        assert len(t2) == 2 * (4 + 3)  # - 4 quotes and 3 trades per bar
        assert isinstance(t2[1], Trade)
        assert t2[1].price == 99.0

    def test_emulated_bars_transformations(self):
        """
        Test EmulatedBarSequence generates 4 bars per OHLC record with correct
        progressive price updates (opening -> mid1 -> mid2 -> final).
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0 + k * dt for k in range(3)],
                    "open": [100.0, 105.0, 110.0],
                    "high": [102.0, 108.0, 115.0],
                    "low": [98.0, 103.0, 107.0],
                    "close": [101.0, 104.0, 112.0],
                    "volume": [1000.0, 1500.0, 2000.0],
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence())

        # - 4 bars per OHLC record
        assert len(bars) == 4 * 3
        assert all(isinstance(b, Bar) for b in bars)

    def test_emulated_bars_bullish(self):
        """
        Test bullish bar (close >= open): open -> low -> high -> close sequence.
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        # - single bullish bar: close(101) >= open(100)
        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0, t0 + dt],
                    "open": [100.0, 100.0],
                    "high": [105.0, 105.0],
                    "low": [95.0, 95.0],
                    "close": [103.0, 103.0],
                    "volume": [1000.0, 1000.0],
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence())
        # - take first bar's 4 updates
        b0, b1, b2, b3 = bars[0], bars[1], bars[2], bars[3]

        # - opening bar: o,o,o,o
        assert b0.open == 100.0
        assert b0.high == 100.0
        assert b0.low == 100.0
        assert b0.close == 100.0
        assert b0.volume == 0

        # - mid1 (bullish): o,o,l,l
        assert b1.open == 100.0
        assert b1.high == 100.0
        assert b1.low == 95.0
        assert b1.close == 95.0
        assert b1.volume == 0

        # - mid2 (bullish): o,h,l,h
        assert b2.open == 100.0
        assert b2.high == 105.0
        assert b2.low == 95.0
        assert b2.close == 105.0
        assert b2.volume == 0

        # - final bar: o,h,l,c with full volume
        assert b3.open == 100.0
        assert b3.high == 105.0
        assert b3.low == 95.0
        assert b3.close == 103.0
        assert b3.volume == 1000.0

    def test_emulated_bars_bearish(self):
        """
        Test bearish bar (close < open): open -> high -> low -> close sequence.
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        # - single bearish bar: close(97) < open(100)
        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0, t0 + dt],
                    "open": [100.0, 100.0],
                    "high": [105.0, 105.0],
                    "low": [95.0, 95.0],
                    "close": [97.0, 97.0],
                    "volume": [1000.0, 1000.0],
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence())
        b0, b1, b2, b3 = bars[0], bars[1], bars[2], bars[3]

        # - opening bar: o,o,o,o
        assert b0.open == 100.0
        assert b0.high == 100.0
        assert b0.low == 100.0
        assert b0.close == 100.0

        # - mid1 (bearish): o,h,o,h
        assert b1.open == 100.0
        assert b1.high == 105.0
        assert b1.low == 100.0
        assert b1.close == 105.0

        # - mid2 (bearish): o,h,l,l
        assert b2.open == 100.0
        assert b2.high == 105.0
        assert b2.low == 95.0
        assert b2.close == 95.0

        # - final bar: o,h,l,c with full volume
        assert b3.open == 100.0
        assert b3.high == 105.0
        assert b3.low == 95.0
        assert b3.close == 97.0
        assert b3.volume == 1000.0

    def test_emulated_bars_with_volume_columns(self):
        """
        Test EmulatedBarSequence propagates all volume fields on final bar.
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0, t0 + dt],
                    "open": [100.0, 100.0],
                    "high": [105.0, 105.0],
                    "low": [95.0, 95.0],
                    "close": [103.0, 103.0],
                    "volume": [1000.0, 2000.0],
                    "bought_volume": [600.0, 1200.0],
                    "volume_quote": [50000.0, 100000.0],
                    "bought_volume_quote": [30000.0, 60000.0],
                    "trade_count": [150, 300],
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence())

        # - check first bar's final update (index 3)
        final = bars[3]
        assert final.volume == pytest.approx(1000.0)
        assert final.bought_volume == pytest.approx(600.0)
        assert final.volume_quote == pytest.approx(50000.0)
        assert final.bought_volume_quote == pytest.approx(30000.0)
        assert final.trade_count == 150

        # - intermediate bars should have zero volume
        for b in bars[:3]:
            assert b.volume == 0

        # - check second bar's final update (index 7)
        final2 = bars[7]
        assert final2.volume == pytest.approx(2000.0)
        assert final2.trade_count == 300

    def test_emulated_bars_without_volume(self):
        """
        Test EmulatedBarSequence works when no volume columns are present.
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0, t0 + dt],
                    "open": [100.0, 105.0],
                    "high": [102.0, 108.0],
                    "low": [98.0, 103.0],
                    "close": [101.0, 107.0],
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence())
        assert len(bars) == 4 * 2

        # - final bar should have zero volume when no volume column
        final = bars[3]
        assert final.volume == 0
        assert final.bought_volume == 0
        assert final.trade_count == 0

    def test_emulated_bars_timestamps_are_monotonic(self):
        """
        Test that emulated bar timestamps are strictly increasing within each OHLC record
        and across records.
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        dt = pd.Timedelta("1h").asm8.item()

        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC,
            pd.DataFrame(
                {
                    "time": [t0 + k * dt for k in range(5)],
                    "open": [100.0 + k for k in range(5)],
                    "high": [102.0 + k for k in range(5)],
                    "low": [98.0 + k for k in range(5)],
                    "close": [101.0 + k for k in range(5)],
                    "volume": [1000.0] * 5,
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence())

        # - all timestamps should be strictly monotonically increasing
        times = [b.time for b in bars]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], f"Timestamp at index {i} not strictly increasing"

    def test_emulated_bars_daily_session(self):
        """
        Test EmulatedBarSequence with daily bars and STOCKS session.
        Opening bar should be near 9:30, closing bar near 16:00.
        """
        t0 = np.datetime64("2020-01-01", "ns").item()
        day = pd.Timedelta("1d").asm8.item()

        r1 = RawData.from_pandas(
            "TEST1",
            DataType.OHLC["1d"],
            pd.DataFrame(
                {
                    "time": [t0, t0 + day],
                    "open": [100.0, 105.0],
                    "high": [110.0, 115.0],
                    "low": [90.0, 95.0],
                    "close": [105.0, 110.0],
                    "volume": [5000.0, 6000.0],
                }
            ),
        )

        bars = r1.transform(EmulatedBarSequence(daily_session_start_end=STOCK_DAILY_SESSION))

        # - first bar (opening) should be near 9:30
        first_time = pd.Timestamp(bars[0].time, unit="ns")
        assert first_time.hour == 9
        assert first_time.minute >= 30

        # - last bar of first record (index 3 = final) should be near 15:59
        last_time = pd.Timestamp(bars[3].time, unit="ns")
        assert last_time.hour == 15
        assert last_time.minute == 59
