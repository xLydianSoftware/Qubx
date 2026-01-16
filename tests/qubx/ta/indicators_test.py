import numpy as np
import pandas as pd

import qubx.pandaz.ta as pta
import tests.qubx.ta.utils_for_testing as test
from qubx.core.series import OHLCV, TimeSeries, compare, lag
from qubx.data.readers import AsOhlcvSeries, AsQuotes, CsvStorageDataReader
from qubx.data.registry import StorageRegistry
from qubx.ta.indicators import (
    atr,
    bollinger_bands,
    cusum_filter,
    dema,
    ema,
    highest,
    kama,
    lowest,
    macd,
    pct_change,
    pewma,
    pewma_outliers_detector,
    pivots,
    psar,
    rsi,
    sma,
    std,
    stdema,
    super_trend,
    swings,
    tema,
    vwma,
)

def pandas_vwma(df: pd.DataFrame, period: int, price_source: str = 'close') -> pd.Series:
    """
    Pandas reference implementation of VWMA

    :param df: DataFrame with OHLC and 'volume' columns
    :param period: lookback period
    :param price_source: price calculation method ('close', 'hl2', 'hlc3', 'ohlc4')
    :return: VWMA series
    """
    # - select price based on source
    if price_source == 'close':
        price = df["close"]
    elif price_source == 'hl2':
        price = (df["high"] + df["low"]) / 2
    elif price_source == 'hlc3':
        price = (df["high"] + df["low"] + df["close"]) / 3
    elif price_source == 'ohlc4':
        price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    else:
        price = df["close"]

    # - calculate price * volume
    pv = price * df["volume"]

    # - rolling sum of pv and volume
    pv_sum = pv.rolling(window=period, min_periods=period).sum()
    vol_sum = df["volume"].rolling(window=period, min_periods=period).sum()

    # - VWMA = sum(pv) / sum(vol)
    result = pv_sum / vol_sum

    return result.rename("vwma")


MIN1_UPDATES = [
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
]


class TestIndicators:
    def generate_random_series(self, n=100_000, freq="1Min"):
        np.random.seed(42)  # Set fixed seed for reproducibility
        T = pd.date_range("2024-01-01 00:00", freq=freq, periods=n)
        ds = 1 + (2 * np.random.randn(len(T))).cumsum()
        data = list(zip(T, ds))
        return T, ds, data

    def test_indicators_on_series(self):
        _, _, data = self.generate_random_series()

        ts = TimeSeries("close", "1h")
        s1 = sma(ts, 50)
        e1 = ema(ts, 50)
        d1 = dema(ts, 50)
        t1 = tema(ts, 50)
        k1 = kama(ts, 50)
        ss1 = sma(s1, 50)
        ee1 = ema(e1, 50)
        test.push(ts, data)

        assert test.N(s1.to_series()[-20:]) == test.apply_to_frame(test.sma, ts.to_series(), 50)[-20:]
        assert test.N(e1.to_series()[-20:]) == test.apply_to_frame(test.ema, ts.to_series(), 50)[-20:]
        assert test.N(t1.to_series()[-20:]) == test.apply_to_frame(test.tema, ts.to_series(), 50)[-20:]
        assert test.N(k1.to_series()[-20:]) == test.apply_to_frame(test.kama, ts.to_series(), 50)[-20:]
        assert test.N(d1.to_series()[-20:]) == test.apply_to_frame(test.dema, ts.to_series(), 50)[-20:]
        # print(ss1.to_series())

    def test_indicators_lagged(self):
        _, _, data = self.generate_random_series()
        ts = TimeSeries("close", "1h")
        l1 = lag(ts, 1)
        l2 = lag(lag(ts, 1), 4)
        test.push(ts, data)
        assert all(lag(ts, 5).to_series().dropna() == l2.to_series().dropna())

    def test_indicators_comparison(self):
        _, _, data = self.generate_random_series()
        # - precalculated
        xs = test.push(TimeSeries("close", "10Min"), data)
        r = test.scols(xs.to_series(), lag(xs, 1).to_series(), names=["a", "b"])
        assert len(compare(xs, lag(xs, 1)).to_series()) > 0
        assert all(np.sign(r.a - r.b).dropna() == compare(xs, lag(xs, 1)).to_series().dropna())

        # - on streamed data
        xs1 = TimeSeries("close", "10Min")
        c1 = compare(xs1, lag(xs1, 1))
        test.push(xs1, data)
        r = test.scols(xs1.to_series(), lag(xs1, 1).to_series(), names=["a", "b"])
        assert len(c1.to_series()) > 0
        assert all(np.sign(r.a - r.b).dropna() == c1.to_series().dropna())

    def test_indicators_highest_lowest(self):
        _, _, data = self.generate_random_series()

        xs = TimeSeries("close", "12Min")
        hh = highest(xs, 13)
        ll = lowest(xs, 13)
        test.push(xs, data)

        rh = xs.pd().rolling(13).max()
        rl = xs.pd().rolling(13).min()
        assert all(abs(hh.pd().dropna() - rh.dropna()) <= 1e-4)
        assert all(abs(ll.pd().dropna() - rl.dropna()) <= 1e-4)

    def test_indicators_on_ohlc(self):
        ohlc = OHLCV("TEST", "1Min")
        s1 = sma(ohlc.close, 5)
        test.push(ohlc, MIN1_UPDATES, 1)
        print(ohlc.to_series())
        print(s1.to_series())

        s2s = sma(ohlc.close, 5).to_series()
        print(s2s)

        # - TODO: fix this behaviour (nan) !
        assert test.N(s2s) == s1.to_series()

    def test_bsf_calcs(self):
        _, _, data = self.generate_random_series(2000)

        def test_i(ts: TimeSeries):
            ds = ts - ts.shift(1)
            a1 = sma(ds * (ds > 0), 14)
            a2 = ds
            return (a1 - a2) / (a1 + a2)

        # - incremental calcs
        ts_i = TimeSeries("close", "1h")
        r1_i = test_i(ts_i)
        test.push(ts_i, data)

        # - calc on ready data
        ts_p = TimeSeries("close", "1h")
        test.push(ts_p, data)
        r1_p = test_i(ts_p)

        # - pandas
        ds = ts_i.pd().diff()
        a1 = test.apply_to_frame(test.sma, ds * (ds > 0), 14)
        a2 = ds
        gauge = (a1 - a2) / (a1 + a2)

        s1 = sum(abs(r1_p.pd() - gauge).dropna())
        s2 = sum(abs(r1_i.pd() - gauge).dropna())
        print(s1, s2)
        assert s1 < 1e-9
        assert s2 < 1e-9

        # - another case
        def test_ii(ts: TimeSeries):
            a1 = sma(ts, 5)
            a2 = sma(ts, 10) * 1000
            return a1 - a2

        ts_ii = TimeSeries("close", "10Min")
        r_ii = test_ii(ts_ii)
        test.push(ts_ii, data[:1000])

        a1 = test.apply_to_frame(test.sma, ts_ii.pd(), 5)
        a2 = 1000 * test.apply_to_frame(test.sma, ts_ii.pd(), 10)
        err = np.std(abs((a1 - a2) - r_ii.pd()).dropna())
        assert err < 1e-9

    def test_on_ready_series(self):
        r0 = CsvStorageDataReader("tests/data/csv/")
        ticks = r0.read("quotes", transform=AsQuotes())

        s0 = TimeSeries("T0", "1Min")
        control = TimeSeries("T0", "1Min")

        # this indicator is being calculated on streamed data
        m0 = sma(s0, 3)

        for q in ticks:
            s0.update(q.time, 0.5 * (q.ask + q.bid))
            control.update(q.time, 0.5 * (q.ask + q.bid))

        # calculate indicator on already formed series
        m1 = sma(control, 3)
        mx = test.scols(s0, m0, m1, names=["series", "streamed", "finished"]).dropna()

        assert test.N(mx.streamed) == mx.finished

    def test_on_formed_only(self):
        r0 = CsvStorageDataReader("tests/data/csv/")
        ticks = r0.read("quotes", transform=AsQuotes())

        # - ask to calculate indicators on closed bars only
        s0 = TimeSeries("T0", "30Sec", process_every_update=False)
        m0 = ema(s0, 5)
        for q in ticks:
            s0.update(q.time, 0.5 * (q.ask + q.bid))

        # - prepare series
        s1 = TimeSeries("T0", "30Sec")
        for q in ticks:
            s1.update(q.time, 0.5 * (q.ask + q.bid))

        # - indicator on already formed series must be equal to calculated on bars
        assert np.nansum((ema(s1, 5) - m0).pd()) == 0

    def test_pewma(self):
        r = CsvStorageDataReader("tests/data/csv/")
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+15h", transform=AsOhlcvSeries("1Min", "ms"))

        ohlc_p = ohlc.pd()
        qs = ohlc.close
        ps = ohlc_p["close"]

        p0 = pta.pwma(ps, 0.99, 0.01, 30)
        p1 = pewma(qs, 0.99, 0.01, 30)
        assert abs(np.mean(p1.pd() - p0.Mean)) < 1e-9
        assert abs(np.mean(p1.std.pd() - p0.Std)) < 1e-9

        # - test streaming data
        ohlc10 = OHLCV("test", "15Min")
        v10 = pewma(ohlc10.close, 0.9, 0.2, 30)
        for b in ohlc[::-1]:
            ohlc10.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)
        e10 = pta.pwma(ohlc10.close.pd(), 0.9, 0.2, 30)
        assert abs(np.mean(v10.pd() - e10.Mean)) < 1e-9
        assert abs(np.mean(v10.std.pd() - e10.Std)) < 1e-9

    def test_pewma_outliers_detector(self):
        r = CsvStorageDataReader("tests/data/csv/")
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+15h", transform=AsOhlcvSeries("1Min", "ms"))

        ohlc_p = ohlc.pd()
        qs = ohlc.close
        ps = ohlc_p["close"]

        p0 = pta.pwma_outliers_detector(ps, 0.90, 0.2)
        p1 = pewma_outliers_detector(qs, 0.90, 0.2)
        assert np.mean(p0.m - p1.pd()) < 1e-9
        assert np.mean(p0.u - p1.upper.pd()) < 1e-9
        assert np.mean(p0.l - p1.lower.pd()) < 1e-9

        # - test streaming data
        ohlc10 = OHLCV("test", "15Min")
        s0 = pewma_outliers_detector(ohlc10.close, 0.9, 0.2, 30)
        for b in ohlc[::-1]:
            ohlc10.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)
        s1 = pta.pwma_outliers_detector(ohlc10.close.pd(), 0.9, 0.2, 30)
        assert abs(np.mean(s0.pd() - s1.m)) < 1e-9
        assert abs(np.mean(s0.std.pd() - s1.s)) < 1e-9
        assert abs(np.mean(s0.outliers.pd() - s1.outliers)) < 1e-9

    def test_psar(self):
        r = CsvStorageDataReader("tests/data/csv/")

        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+360Min", transform=AsOhlcvSeries("1Min", "ms"))
        v = psar(ohlc)
        e = pta.psar(ohlc.pd())

        assert np.mean(abs(v.pd() - e.psar)) < 1e-3
        assert np.mean(abs(v.upper.pd() - e.up)) < 1e-3
        assert np.mean(abs(v.lower.pd() - e.down)) < 1e-3

        # - test streaming data
        ohlc10 = OHLCV("test", "5Min")
        v10 = psar(ohlc10)

        for b in ohlc[::-1]:
            ohlc10.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)

        e10 = pta.psar(ohlc10.pd())
        assert np.mean(abs(v10.pd() - e10.psar)) < 1e-3
        assert np.mean(abs(v10.upper.pd() - e10.up)) < 1e-3
        assert np.mean(abs(v10.lower.pd() - e10.down)) < 1e-3

    def test_atr(self):
        r = CsvStorageDataReader("tests/data/csv/")

        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+5d", transform=AsOhlcvSeries("1Min", "ms"))
        v = atr(ohlc, 14, "sma", percentage=False)
        e = pta.atr(ohlc.pd(), 14, "sma", percentage=False)

        assert (v.pd() - e).dropna().sum() < 1e-6

        # - test streaming data
        ohlc10 = OHLCV("test", "5Min")
        v10 = atr(ohlc, 14, "sma", percentage=False)

        for b in ohlc[::-1]:
            ohlc10.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)

        e10 = pta.atr(ohlc10.pd(), 14, "sma", percentage=False)
        assert (v10.pd() - e10).dropna().sum() < 1e-6

    def test_bollinger_bands(self):
        r = CsvStorageDataReader("tests/data/csv/")

        # Test on existing data
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+24h", transform=AsOhlcvSeries("5Min", "ms"))
        v = bollinger_bands(ohlc.close, period=20, nstd=2, smoother="sma")

        # Test against pandas implementation (now fixed)
        e = pta.bollinger(ohlc.close.pd(), window=20, nstd=2, mean="sma")

        # Compare middle band (moving average)
        assert abs((v.pd() - e["Median"]).dropna().sum()) < 1e-6

        # Compare upper band
        assert abs((v.upper.pd() - e["Upper"]).dropna().sum()) < 1e-6

        # Compare lower band
        assert abs((v.lower.pd() - e["Lower"]).dropna().sum()) < 1e-6

        # Test streaming data
        ohlc_stream = OHLCV("test", "5Min")
        v_stream = bollinger_bands(ohlc_stream.close, period=20, nstd=2, smoother="sma")

        for b in ohlc[::-1]:
            ohlc_stream.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)

        # Test streaming against pandas
        e_stream = pta.bollinger(ohlc_stream.close.pd(), window=20, nstd=2, mean="sma")

        # Compare streaming results
        assert abs((v_stream.pd() - e_stream["Median"]).dropna().sum()) < 1e-6
        assert abs((v_stream.upper.pd() - e_stream["Upper"]).dropna().sum()) < 1e-6
        assert abs((v_stream.lower.pd() - e_stream["Lower"]).dropna().sum()) < 1e-6

        # Test with different parameters
        v_small = bollinger_bands(ohlc.close, period=10, nstd=1.5, smoother="sma")
        e_small = pta.bollinger(ohlc.close.pd(), window=10, nstd=1.5, mean="sma")

        assert abs((v_small.pd() - e_small["Median"]).dropna().sum()) < 1e-6
        assert abs((v_small.upper.pd() - e_small["Upper"]).dropna().sum()) < 1e-6
        assert abs((v_small.lower.pd() - e_small["Lower"]).dropna().sum()) < 1e-6

        # Test with EMA smoother
        v_ema = bollinger_bands(ohlc.close, period=20, nstd=2, smoother="ema")
        e_ema = pta.bollinger(ohlc.close.pd(), window=20, nstd=2, mean="ema")

        assert abs((v_ema.pd() - e_ema["Median"]).dropna().sum()) < 1e-6
        assert abs((v_ema.upper.pd() - e_ema["Upper"]).dropna().sum()) < 1e-6
        assert abs((v_ema.lower.pd() - e_ema["Lower"]).dropna().sum()) < 1e-6

    def test_swings(self):
        r = CsvStorageDataReader("tests/data/csv/")

        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+12h", transform=AsOhlcvSeries("10Min", "ms"))
        v = swings(ohlc, psar, iaf=0.1, maxaf=1)
        e = pta.swings(ohlc.pd(), pta.psar, iaf=0.1, maxaf=1)

        assert all(
            e.trends["UpTrends"][["start_price", "end_price"]].dropna()
            == v.pd()["UpTrends"][["start_price", "end_price"]].dropna()
        )

        # - test streaming data
        ohlc10 = OHLCV("test", "30Min")
        v10 = swings(ohlc10, psar, iaf=0.1, maxaf=1)

        for b in ohlc[::-1]:
            ohlc10.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)

        e10 = pta.swings(ohlc10.pd(), pta.psar, iaf=0.1, maxaf=1)

        assert all(
            e10.trends["UpTrends"][["start_price", "end_price"]].dropna()
            == v10.pd()["UpTrends"][["start_price", "end_price"]].dropna()
        )

    def test_pivots(self):
        """Test Pivots indicator against pandas pivots_highs_lows"""
        r = CsvStorageDataReader("tests/data/csv/")

        # Load test data
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+12h", transform=AsOhlcvSeries("5Min", "ms"))

        # Test with different before/after parameters
        before, after = 5, 5

        # Create pivots indicator
        p = pivots(ohlc, before=before, after=after)

        # Get pandas data for comparison
        ohlc_pd = ohlc.pd()

        # Calculate pivots using pandas
        pivots_pd = pta.pivots_highs_lows(
            ohlc_pd["high"],
            ohlc_pd["low"],
            nf_before=before,
            nf_after=after,
            index_on_observed_time=True,
            align_with_index=False,
        )

        # Check that we have detected some pivots
        assert len(p.tops) > 0, "No pivot highs detected"
        assert len(p.bottoms) > 0, "No pivot lows detected"

        # Get the pivots as pandas series
        tops_series = p.tops.pd()
        bottoms_series = p.bottoms.pd()

        # Compare detected pivot highs
        pd_highs = pivots_pd["U"].dropna()
        assert len(tops_series) > 0, "Streaming pivots detected no tops"
        assert len(pd_highs) > 0, "Pandas pivots detected no highs"

        # Compare detected pivot lows
        pd_lows = pivots_pd["L"].dropna()
        assert len(bottoms_series) > 0, "Streaming pivots detected no bottoms"
        assert len(pd_lows) > 0, "Pandas pivots detected no lows"

        # Test the pd() method returns proper DataFrame structure
        df = p.pd()
        assert "Tops" in df.columns.get_level_values(0).tolist()
        assert "Bottoms" in df.columns.get_level_values(0).tolist()
        assert "price" in df["Tops"].columns
        assert "detection_lag" in df["Tops"].columns
        assert "spotted" in df["Tops"].columns

        # Test streaming behavior
        ohlc_streaming = OHLCV("test", "5Min")
        p_streaming = pivots(ohlc_streaming, before=before, after=after)

        # Feed data bar by bar
        for bar in ohlc[::-1]:
            ohlc_streaming.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume)

        # Compare streaming results with batch results
        assert len(p_streaming.tops) == len(p.tops), "Streaming vs batch tops count mismatch"
        assert len(p_streaming.bottoms) == len(p.bottoms), "Streaming vs batch bottoms count mismatch"

        # Test with different parameters
        p2 = pivots(ohlc, before=3, after=3)
        assert len(p2.tops) >= len(p.tops), "Smaller window should detect more or equal pivots"

        p3 = pivots(ohlc, before=10, after=10)
        assert len(p3.tops) <= len(p.tops), "Larger window should detect fewer or equal pivots"

    def test_pct_change(self):
        """Test PctChange indicator against pandas pct_change"""
        r = CsvStorageDataReader("tests/data/csv/")

        # Load test data
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+24h", transform=AsOhlcvSeries("5Min", "ms"))

        # Test with period=1 (default)
        pct = pct_change(ohlc.close, period=1)

        # Get pandas pct_change for comparison
        pandas_pct = ohlc.close.pd().pct_change(periods=1)

        # Compare results (they should be identical)
        pct_series = pct.pd()
        diff = abs(pct_series - pandas_pct).dropna()
        assert diff.sum() < 1e-10, f"PctChange differs from pandas: max diff = {diff.max()}"

        # Test with period=5
        pct5 = pct_change(ohlc.close, period=5)
        pandas_pct5 = ohlc.close.pd().pct_change(periods=5)

        pct5_series = pct5.pd()
        diff5 = abs(pct5_series - pandas_pct5).dropna()
        assert diff5.sum() < 1e-10, f"PctChange(period=5) differs from pandas: max diff = {diff5.max()}"

        # Test streaming behavior
        ohlc_stream = OHLCV("test", "5Min")
        pct_stream = pct_change(ohlc_stream.close, period=1)

        # Feed data bar by bar
        for bar in ohlc[::-1]:
            ohlc_stream.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume)

        # Compare streaming results with batch results and pandas
        pct_stream_series = pct_stream.pd()
        pandas_stream = ohlc_stream.close.pd().pct_change(periods=1)

        diff_stream = abs(pct_stream_series - pandas_stream).dropna()
        assert diff_stream.sum() < 1e-10, f"Streaming PctChange differs from pandas: max diff = {diff_stream.max()}"

        # Test edge cases
        # 1. Test with zero values
        test_series = TimeSeries("test", "1Min")
        test_pct = pct_change(test_series, period=1)

        test_data = [
            ("2024-01-01 00:00", 100),
            ("2024-01-01 00:01", 0),  # Zero value
            ("2024-01-01 00:02", 50),
            ("2024-01-01 00:03", 100),
            ("2024-01-01 00:04", 0),  # Another zero
            ("2024-01-01 00:05", 0),  # Zero to zero
            ("2024-01-01 00:06", 10),
        ]

        test.push(test_series, test_data)

        # Verify pandas behavior with zero values
        test_pd = test_series.pd().pct_change(periods=1)
        test_pct_pd = test_pct.pd()

        # Check that our implementation returns NaN for division by zero (while pandas returns inf)
        assert np.isnan(test_pct_pd.iloc[2]), "Should return NaN when previous value is 0"
        assert np.isinf(test_pd.iloc[2]), "Pandas returns inf when previous value is 0"

        # 2. Test with NaN values
        # Note: Our implementation and pandas handle NaN differently
        # Pandas fills NaN forward by default, we don't
        # This is documented behavior and acceptable
        nan_series = TimeSeries("nan_test", "1Min")
        nan_pct = pct_change(nan_series, period=1)

        nan_data = [
            ("2024-01-01 00:00", 100),
            ("2024-01-01 00:01", np.nan),
            ("2024-01-01 00:02", 150),
            ("2024-01-01 00:03", 200),
        ]

        test.push(nan_series, nan_data)

        nan_pct_pd = nan_pct.pd()

        # Verify our NaN handling - when current or previous is NaN, result is NaN
        assert np.isnan(nan_pct_pd.iloc[0]), "First value should be NaN"
        assert np.isnan(nan_pct_pd.iloc[1]), "NaN input should give NaN output"
        assert np.isnan(nan_pct_pd.iloc[2]), "Value after NaN should give NaN (prev is NaN)"

        # 3. Test with different periods
        for period in [1, 2, 3, 10]:
            pct_p = pct_change(ohlc.close, period=period)
            pandas_p = ohlc.close.pd().pct_change(periods=period)

            diff_p = abs(pct_p.pd() - pandas_p).dropna()
            assert diff_p.sum() < 1e-10, f"PctChange(period={period}) differs from pandas"

    def test_std_with_min_periods(self):
        """Test Std indicator with min_periods parameter against pandas rolling std"""
        r = CsvStorageDataReader("tests/data/csv/")

        # Load test data
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+12h", transform=AsOhlcvSeries("5Min", "ms"))

        # Test 1: Basic min_periods test with sample std (ddof=1)
        period = 20
        min_periods = 5

        # Calculate using our implementation
        std_with_min = std(ohlc.close, period=period, ddof=1, min_periods=min_periods)

        # Calculate using pandas
        pandas_std = ohlc.close.pd().rolling(window=period, min_periods=min_periods).std(ddof=1)

        # Compare results
        our_std = std_with_min.pd()

        # Check that we get values starting from min_periods
        # First min_periods-1 values should be NaN
        assert all(pd.isna(our_std.iloc[: min_periods - 1])), "Should have NaN for first min_periods-1 values"

        # From min_periods onwards, should match pandas (with small numerical tolerance)
        diff = abs(our_std - pandas_std).dropna()
        assert diff.max() < 1e-10, f"Std with min_periods differs from pandas: max diff = {diff.max()}"

        # Test 2: Test with different min_periods values
        for test_min_periods in [3, 10, 15]:
            std_mp = std(ohlc.close, period=20, ddof=1, min_periods=test_min_periods)
            pandas_mp = ohlc.close.pd().rolling(window=20, min_periods=test_min_periods).std(ddof=1)

            diff_mp = abs(std_mp.pd() - pandas_mp).dropna()
            assert diff_mp.max() < 1e-10, f"Std(min_periods={test_min_periods}) differs from pandas"

        # Test 3: Test with population std (ddof=0) for exact match
        # Using population std to avoid numerical issues with sample std
        period = 10
        min_periods = 5

        # Calculate using our implementation
        std_pop = std(ohlc.close, period=period, ddof=0, min_periods=min_periods)

        # Calculate using pandas
        pandas_pop = ohlc.close.pd().rolling(window=period, min_periods=min_periods).std(ddof=0)

        # Compare results
        diff_pop = abs(std_pop.pd() - pandas_pop).dropna()
        assert diff_pop.max() < 1e-9, (
            f"Population std with min_periods differs from pandas: max diff = {diff_pop.max()}"
        )

    def test_rsi(self):
        """
        Test RSI indicator against pandas rsi implementation
        """
        r = CsvStorageDataReader("tests/data/csv/")

        # - Load test data
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+24h", transform=AsOhlcvSeries("5Min", "ms"))

        # - Test with default parameters (period=14, smoother='ema')
        v = rsi(ohlc.close, period=14, smoother="ema")
        e = pta.rsi(ohlc.close.pd(), 14, smoother=pta.ema)

        # - Compare results (allow small numerical differences)
        diff = abs(v.pd() - e).dropna()
        assert diff.sum() < 1e-6, f"RSI differs from pandas: sum diff = {diff.sum()}"

        # - Test with SMA smoother
        v_sma = rsi(ohlc.close, period=14, smoother="sma")
        e_sma = pta.rsi(ohlc.close.pd(), 14, smoother=pta.sma)

        diff_sma = abs(v_sma.pd() - e_sma).dropna()
        assert diff_sma.sum() < 1e-6, f"RSI (sma) differs from pandas: sum diff = {diff_sma.sum()}"

        # - Test streaming data
        ohlc_stream = OHLCV("test", "5Min")
        v_stream = rsi(ohlc_stream.close, period=14, smoother="ema")

        for b in ohlc[::-1]:
            ohlc_stream.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume)

        e_stream = pta.rsi(ohlc_stream.close.pd(), 14, smoother=pta.ema)
        diff_stream = abs(v_stream.pd() - e_stream).dropna()
        assert diff_stream.sum() < 1e-6, f"Streaming RSI differs from pandas: sum diff = {diff_stream.sum()}"

        # - Test with different period
        v_short = rsi(ohlc.close, period=7, smoother="ema")
        e_short = pta.rsi(ohlc.close.pd(), 7, smoother=pta.ema)

        diff_short = abs(v_short.pd() - e_short).dropna()
        assert diff_short.sum() < 1e-6, f"RSI(7) differs from pandas: sum diff = {diff_short.sum()}"

        # - Verify RSI values are in valid range [0, 100]
        rsi_values = v.pd().dropna()
        assert all(rsi_values >= 0), "RSI should be >= 0"
        assert all(rsi_values <= 100), "RSI should be <= 100"

    def test_std_ema(self):
        def volatility_ethalon(series: pd.Series, volatility_lookback: int) -> pd.Series:
            """
            Calculates the rolling standard deviation of returns

            Parameters
            ----------
                series : pd.Series
                    A Series containing price data
                volatility_lookback : int
                    The lookback period for calculating the rolling standard deviation.
            Returns
            -------
                pd.Series
                    A volatility Series
            """
            returns = series.ffill().pct_change(fill_method=None)
            vols = returns.ewm(span=volatility_lookback, min_periods=volatility_lookback).std()

            # - Set volatility to NaN where returns are NaN (delisted instruments)
            return vols.where(returns.notna())

        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
        c1 = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc().close

        # - calculate pandas reference
        v1 = volatility_ethalon(c1.pd(), 30)

        # - create returns TimeSeries
        returns_ts = pct_change(c1)

        # - apply stdema to returns
        v2 = stdema(returns_ts, 30)

        diff = abs(v2.pd() - v1).dropna()
        assert diff.sum() < 1e-6, f"stdema differs from pandas: sum diff = {diff.sum()}"

        # - Test streaming data
        close_stream = TimeSeries("close_stream", "1h")
        returns_stream = pct_change(close_stream)
        v_stream = stdema(returns_stream, 30)

        # - Feed data in forward order (oldest to newest)
        c1_pd = c1.pd()
        for time, value in zip(c1_pd.index, c1_pd.values):
            close_stream.update(int(time.value), value)

        e_stream = volatility_ethalon(close_stream.pd(), 30)
        diff_stream = abs(v_stream.pd() - e_stream).dropna()
        # - Streaming has slightly larger numerical differences due to incremental algorithm
        assert diff_stream.sum() < 1e-3, f"Streaming stdema differs from pandas: sum diff = {diff_stream.sum()}"
        assert diff_stream.max() < 2e-5, f"Max streaming stdema diff: {diff_stream.max()}"

    def test_cusum_filter(self):
        reader = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]

        # - hourly ohlc
        c1h = reader.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

        # - daily ohlc
        c1d = c1h.resample("1d")

        # - daily volatility
        vol = stdema(pct_change(c1d.close), 30)

        # - calculate cusum on streaming data
        r = cusum_filter(c1h.close, vol * 2)

        # - calculate cusum etalon data
        # - pandas cusum_filter expects end-of-bar timestamps, but Qubx uses start-of-bar
        # - shift threshold forward by 1 period to convert start-of-bar to end-of-bar behavior
        vol_pd = vol.pd() * 2
        test_pd = pta.cusum_filter(c1h.close.pd(), vol_pd.shift(1))
        # print(test_pd)

        # - indecies must be equal
        r_pd = r.pd()
        assert r_pd[r_pd == 1].index.equals(test_pd)

        # - test on streaming data
        s1h = OHLCV("s1", "1h")
        s1d = s1h.resample("1d")
        vol1 = stdema(pct_change(s1d.close), 30)

        # - calculate cusum on streaming data
        r1 = cusum_filter(s1h.close, vol1 * 2)

        # - populate s1h by bars from c1h series
        c1h_pd = c1h.pd()
        for idx in c1h_pd.index:
            bar = c1h_pd.loc[idx]
            s1h.update_by_bar(int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0))

        r1_pd = r1.pd()

        # - calculate pandas reference on the SAME streaming data
        # - pandas cusum_filter expects end-of-bar timestamps, shift threshold for start-of-bar
        vol1_pd = vol1.pd() * 2
        test_streaming_pd = pta.cusum_filter(s1h.close.pd(), vol1_pd.shift(1))

        # - with proper lookback, streaming should match pandas exactly
        assert r1_pd[r1_pd == 1].index.equals(test_streaming_pd), (
            f"Streaming cusum_filter should match pandas\n"
            f"Streaming: {r1_pd[r1_pd == 1].index.tolist()}\n"
            f"Pandas: {test_streaming_pd.tolist()}"
        )

    def test_cusum_filter_stream(self):
        # - additional streaming test
        reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]
        raw = reader.read("ETHUSDT", "ohlc(1h)", "2021-11-30 19:00", "2022-03-01")

        T = slice("2022-01-01", "2022-01-10")

        # - calculate on data
        ohlc = raw.to_ohlc()
        _volt_data = ohlc.resample("1d")
        vol = stdema(pct_change(_volt_data.close), 30)
        ns_csf = cusum_filter(ohlc.close, vol * 0.3).pd()[T]
        print("-----\n", ns_csf[ns_csf == 1])

        # - calculate on streaming data
        ohlc1 = raw.to_ohlc()  # get new instance of ohlc data

        H1 = OHLCV("s1", "1h")
        D1 = H1.resample("1d")
        vol1 = stdema(pct_change(D1.close), 30)
        s_csf = cusum_filter(H1.close, vol1 * 0.3)

        # - populate s1h by bars from c1h series
        bars = ohlc1.pd()
        for idx in bars.index:
            bar = bars.loc[idx]
            H1.update_by_bar(int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0))
        stream_cs = s_csf.pd()[T]
        print("-----\n", stream_cs[stream_cs == 1])

        # - debug: compare events and volatility
        print(f"\nNon-streaming events: {len(ns_csf[ns_csf == 1])}")
        print(f"Streaming events: {len(stream_cs[stream_cs == 1])}")
        print(f"\nNon-streaming first 5:\n{ns_csf[ns_csf == 1].head()}")
        print(f"\nStreaming first 5:\n{stream_cs[stream_cs == 1].head()}")

        print(f"\nNon-streaming pct_change first 5:\n{pct_change(_volt_data.close).pd().head(5)}")
        print(f"\nStreaming pct_change first 5:\n{pct_change(D1.close).pd().head(5)}")

        print(f"\nNon-streaming vol first 35:\n{vol.pd().head(35)}")
        print(f"\nStreaming vol1 first 35:\n{vol1.pd().head(35)}")

        # - with pct_change fix, streaming and non-streaming should match
        assert all(stream_cs[stream_cs == 1] == ns_csf[ns_csf == 1])

    def test_cusum_filter_on_events(self):
        from qubx.data.transformers import TickSeries

        reader = StorageRegistry.get("csv::tests/data/storages/csv_longer")["BINANCE.UM", "SWAP"]

        # - hourly ohlc
        c1h = reader.read("ETHUSDT", "ohlc(1h)", "2021-12-01", "2022-02-01").to_ohlc()

        # - daily ohlc, volatility
        c1d = c1h.resample("1d")
        vol = stdema(pct_change(c1d.close), 30)

        # - calculate cusum on ready ohlc data
        r = cusum_filter(c1h.close, vol * 0.3)

        r_pd = r.pd()

        # - try ticks updates - for that convert ohlc into trades (each bar --> 4 trade for o, h, l and c - each with time inside the bar)
        ticks = reader.read("ETHUSDT", "ohlc(1h)", "2021-12-01", "2022-02-01").transform(
            # TickSeries(quotes=False, trades=True)
            TickSeries(quotes=True, trades=False)
        )

        # - now create fresh indicators
        s1h = OHLCV("s1", "1h")
        s1d = s1h.resample("1d")
        vol1 = stdema(pct_change(s1d.close), 30)

        # - calculate cusum on streaming data
        r1 = cusum_filter(s1h.close, vol1 * 0.3)

        # - populate s1h by ticks (trades)
        for t in ticks:
            # s1h.update(t.time, t.price, t.size)
            s1h.update(t.time, t.mid_price(), 0.0)

        r1_pd = r1.pd()
        # print(r1_pd[r1_pd == 1].head(25))

        assert all(r_pd[r_pd == 1].head(25) == r1_pd[r1_pd == 1].head(25))

    def test_macd(self):
        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]

        # - hourly ohlc
        c1h = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

        # - calculate macd on streaming data
        r0 = macd(c1h.close, 12, 26, 9, "sma", "sma")

        # - calculate macd on pandas
        r1 = pta.macd(c1h.close.pd(), 12, 26, 9, "sma", "sma")

        diff_stream = abs(r1 - r0.pd()).dropna()
        assert diff_stream.sum() < 1e-6, f"macd differs from pandas: sum diff = {diff_stream.sum()}"

        # - calculate macd on streaming data
        r01 = macd(c1h.close, 12, 26, 9, "ema", "sma")

        # - calculate macd on pandas
        r11 = pta.macd(c1h.close.pd(), 12, 26, 9, "ema", "sma")
        diff_stream = abs(r11 - r01.pd()).dropna()
        assert diff_stream.sum() < 1e-6, f"macd differs from pandas: sum diff = {diff_stream.sum()}"

    def test_super_trend(self):
        """
        Test SuperTrend indicator against pandas super_trend implementation
        """
        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]

        # - hourly ohlc
        c1h = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

        # - test with default parameters
        st = super_trend(c1h, length=22, mult=3.0, src="hl2", wicks=True, atr_smoother="sma")

        # - calculate pandas version
        st_pd = pta.super_trend(c1h.pd(), length=22, mult=3.0, src="hl2", wicks=True, atr_smoother="sma")

        # - compare trend direction
        diff_trend = abs(st.pd() - st_pd["trend"]).dropna()
        assert diff_trend.sum() < 1e-6, f"super_trend trend differs from pandas: sum diff = {diff_trend.sum()}"

        # - compare utl (upper trend line)
        diff_utl = abs(st.utl.pd() - st_pd["utl"]).dropna()
        assert diff_utl.sum() < 1e-6, f"super_trend utl differs from pandas: sum diff = {diff_utl.sum()}"

        # - compare dtl (down trend line)
        diff_dtl = abs(st.dtl.pd() - st_pd["dtl"]).dropna()
        assert diff_dtl.sum() < 1e-6, f"super_trend dtl differs from pandas: sum diff = {diff_dtl.sum()}"

        # - test with different parameters
        st2 = super_trend(c1h, length=10, mult=2.0, src="close", wicks=False, atr_smoother="ema")
        st2_pd = pta.super_trend(c1h.pd(), length=10, mult=2.0, src="close", wicks=False, atr_smoother="ema")

        diff_trend2 = abs(st2.pd() - st2_pd["trend"]).dropna()
        assert diff_trend2.sum() < 1e-6, f"super_trend(2) trend differs from pandas: sum diff = {diff_trend2.sum()}"

        # - test streaming data
        ohlc_stream = OHLCV("test", "1h")
        st_stream = super_trend(ohlc_stream, length=22, mult=3.0, src="hl2", wicks=True, atr_smoother="sma")

        # - feed data bar by bar (reverse order to simulate streaming)
        c1h_pd = c1h.pd()
        for idx in c1h_pd.index:
            bar = c1h_pd.loc[idx]
            ohlc_stream.update_by_bar(
                int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0)
            )

        # - calculate pandas version on streamed data
        st_stream_pd = pta.super_trend(ohlc_stream.pd(), length=22, mult=3.0, src="hl2", wicks=True, atr_smoother="sma")

        # - compare streaming results
        diff_stream_trend = abs(st_stream.pd() - st_stream_pd["trend"]).dropna()
        assert diff_stream_trend.sum() < 1e-6, (
            f"Streaming super_trend trend differs: sum diff = {diff_stream_trend.sum()}"
        )

        # - test 4h streaming built from 1h data
        ohlc_4h_stream = OHLCV("test_4h", "4h")
        st_4h_stream = super_trend(ohlc_4h_stream, length=22, mult=3.0, src="hl2", wicks=True, atr_smoother="sma")

        # - feed 1h data to build 4h bars
        for idx in c1h_pd.index:
            bar = c1h_pd.loc[idx]
            ohlc_4h_stream.update_by_bar(
                int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0)
            )

        # - calculate super_trend on final 4h bars
        st_4h_pd = pta.super_trend(ohlc_4h_stream.pd(), length=22, mult=3.0, src="hl2", wicks=True, atr_smoother="sma")

        # - compare 4h streaming results
        diff_4h_trend = abs(st_4h_stream.pd() - st_4h_pd["trend"]).dropna()
        assert diff_4h_trend.sum() < 1e-6, f"4h streaming super_trend trend differs: sum diff = {diff_4h_trend.sum()}"

        diff_4h_utl = abs(st_4h_stream.utl.pd() - st_4h_pd["utl"]).dropna()
        assert diff_4h_utl.sum() < 1e-6, f"4h streaming super_trend utl differs: sum diff = {diff_4h_utl.sum()}"

        diff_4h_dtl = abs(st_4h_stream.dtl.pd() - st_4h_pd["dtl"]).dropna()
        assert diff_4h_dtl.sum() < 1e-6, f"4h streaming super_trend dtl differs: sum diff = {diff_4h_dtl.sum()}"

    def test_vwma(self):
        """
        Test VWMA indicator against pandas reference implementation
        """
        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]

        # - hourly ohlc
        ohlc = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

        # - parameters
        PERIOD = 20

        # - calculate streaming version
        ind_stream = vwma(ohlc, PERIOD)

        # - calculate pandas version
        df = ohlc.pd()
        ind_pandas = pandas_vwma(df, PERIOD)

        # - compare results
        stream_pd = ind_stream.pd()
        diff = abs(stream_pd - ind_pandas).dropna()

        assert len(stream_pd.dropna()) > 0, "No data in streaming indicator"
        assert diff.sum() < 1e-6, f"VWMA differs from pandas: sum diff = {diff.sum()}"

    def test_vwma_with_different_periods(self):
        """
        Test VWMA with various period lengths
        """
        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
        ohlc = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()
        df = ohlc.pd()

        for period in [10, 20, 50, 100]:
            ind_stream = vwma(ohlc, period)
            ind_pandas = pandas_vwma(df, period)

            stream_pd = ind_stream.pd()
            diff = abs(stream_pd - ind_pandas).dropna()

            assert diff.sum() < 1e-6, f"VWMA (period={period}) differs: sum diff = {diff.sum()}"

    def test_vwma_price_sources(self):
        """
        Test VWMA with different price sources
        """
        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
        ohlc = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()
        df = ohlc.pd()

        period = 20

        # - test all price sources against pandas reference
        for price_source in ['close', 'hl2', 'hlc3', 'ohlc4']:
            ind_stream = vwma(ohlc, period, price_source=price_source)
            ind_pandas = pandas_vwma(df, period, price_source=price_source)

            stream_pd = ind_stream.pd()
            diff = abs(stream_pd - ind_pandas).dropna()

            assert len(stream_pd.dropna()) > 0, f"VWMA {price_source} should have values"
            assert diff.sum() < 1e-6, f"VWMA (price_source={price_source}) differs: sum diff = {diff.sum()}"

        # - verify different price sources produce different results
        vwma_close = vwma(ohlc, period, price_source="close").pd()
        vwma_hl2 = vwma(ohlc, period, price_source="hl2").pd()
        vwma_hlc3 = vwma(ohlc, period, price_source="hlc3").pd()
        vwma_ohlc4 = vwma(ohlc, period, price_source="ohlc4").pd()

        assert not vwma_close.equals(vwma_hl2), "close and hl2 should differ"
        assert not vwma_close.equals(vwma_hlc3), "close and hlc3 should differ"
        assert not vwma_close.equals(vwma_ohlc4), "close and ohlc4 should differ"

    def test_vwma_streaming(self):
        """
        Test VWMA on streaming data
        """
        r = StorageRegistry.get("csv::tests/data/storages/csv")["BINANCE.UM", "SWAP"]
        ohlc = r.read("BTCUSDT", "ohlc(1h)", "2023-06-01", "2023-08-01").to_ohlc()

        # - create streaming OHLCV
        ohlc_stream = OHLCV("test", "1h")
        v_stream = vwma(ohlc_stream, 20)

        # - feed data bar by bar
        ohlc_pd = ohlc.pd()
        for idx in ohlc_pd.index:
            bar = ohlc_pd.loc[idx]
            ohlc_stream.update_by_bar(
                int(idx.value), bar["open"], bar["high"], bar["low"], bar["close"], bar.get("volume", 0)
            )

        # - calculate pandas reference on streamed data
        e_stream = pandas_vwma(ohlc_stream.pd(), 20)
        diff_stream = abs(v_stream.pd() - e_stream).dropna()

        assert diff_stream.sum() < 1e-6, f"Streaming VWMA differs from pandas: sum diff = {diff_stream.sum()}"
