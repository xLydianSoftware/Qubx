import numpy as np
import pandas as pd

import qubx.pandaz.ta as pta
import tests.qubx.ta.utils_for_testing as test
from qubx.core.series import OHLCV, TimeSeries, compare, lag
from qubx.data.readers import AsOhlcvSeries, AsQuotes, CsvStorageDataReader
from qubx.ta.indicators import (
    atr,
    dema,
    ema,
    highest,
    kama,
    lowest,
    pewma,
    pewma_outliers_detector,
    psar,
    sma,
    swings,
    tema,
)

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
