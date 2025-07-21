import numpy as np
import pandas as pd
from numpy import nan
from pytest import approx

import qubx.pandaz.ta as pta
from qubx.pandaz import ohlc_resample

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


class TestPandasTaIndicators:
    def test_ema(self):
        r1 = pta.ema(np.array([1, 5, 10, 4, 3]), 3, init_mean=True)

        assert N(r1) == np.array([[nan], [nan], [5.33333333], [4.66666667], [3.83333333]]), (
            "EMA init_mean are not equal"
        )

        r3 = pta.ema(np.array([[1, 5, 10, 4, 3, 4, 5], [nan, nan, 1, 5, 10, 4, 3]]).T, 3, init_mean=True)

        assert N(r3) == np.array(
            [
                [nan, nan],
                [nan, nan],
                [5.33333333, nan],
                [4.66666667, nan],
                [3.83333333, 5.33333333],
                [3.91666667, 4.66666667],
                [4.45833333, 3.83333333],
            ]
        ), "EMA multiple columns are not equal"

    def test_kama(self):
        x = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1]]).T
        x2 = np.array([[11.0, 22, 33, 44, 55, 66, 77, 88, 99, 88, 77, 66, 55, 44, 33, 22, 11]]).T

        our_data_1 = pta.kama(x, 5, 2, 30)
        our_data_2 = pta.kama(x2, 5, 2, 30)

        talib_data_1 = np.array(
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                5.44444444,
                6.13580247,
                6.96433471,
                7.86907484,
                7.89281303,
                7.86227426,
                7.79857496,
                7.29116216,
                5.82842342,
                4.57134635,
                3.42852575,
                2.34918097,
            ]
        )
        talib_data_2 = np.array(
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                59.88888889,
                67.49382716,
                76.60768176,
                86.5598232,
                86.82094329,
                86.48501684,
                85.78432457,
                80.20278377,
                64.11265765,
                50.28480981,
                37.71378323,
                25.84099068,
            ]
        )
        assert N(our_data_1.flatten()) == talib_data_1
        assert N(our_data_2.flatten()) == talib_data_2

    def test_dema(self):
        x = np.array([1.0, 5, 10, 4, 3, 4, 5, 11, 10, 8, 8, 18, 21, 1, 5])
        our_data = pta.dema(x, 3)
        talib_data = np.array(
            [
                nan,
                nan,
                nan,
                nan,
                3.05555556,
                3.56944444,
                4.55555556,
                9.41319444,
                10.27430556,
                8.92100694,
                8.35243056,
                15.62217882,
                20.28407118,
                6.62852648,
                4.80750868,
            ]
        )

        assert N(our_data.flatten()) == talib_data

    def test_tema(self):
        x = np.array([1.0, 5, 10, 4, 3, 4, 5, 11, 10, 8, 8, 18, 21, 1, 5])
        our_data = pta.tema(x, 3)
        talib_data = np.array(
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                4.7037037,
                10.2806713,
                10.5708912,
                8.6087963,
                8.02010995,
                16.64492911,
                21.15341073,
                4.24893302,
                3.71395761,
            ]
        )

        assert N(our_data.flatten()) == talib_data

    def test_denoised_trend(self):
        o1 = np.array([10.0, 11, 12, 13, 14, 15, 16])

        c1 = o1 + 0.5
        data = pd.DataFrame(
            {"open": o1, "close": c1}, index=pd.date_range("2000-01-01 00:00", periods=len(o1), freq="1h")
        )

        assert N(pta.denoised_trend(data, 6).iloc[-1]) == c1[6] - o1[0]
        assert N(pta.denoised_trend(data, 6, bar_returns=False).iloc[-1]) == c1[6] - c1[0]

        c1 = o1 + 1.5
        data = pd.DataFrame(
            {"open": o1, "close": c1}, index=pd.date_range("2000-01-01 00:00", periods=len(o1), freq="1h")
        )

        np.testing.assert_equal(pta.denoised_trend(data, 6).iloc[-1] < c1[6] - o1[0], True)
        np.testing.assert_almost_equal(pta.denoised_trend(data, 6, bar_returns=False).iloc[-1], c1[6] - c1[0])

    def test_pivot_point(self):
        prices = np.arange(50)

        data = pd.DataFrame(
            {"ask": prices, "bid": prices}, index=pd.date_range("2019-01-01 00:00", periods=len(prices), freq="1h")
        )

        r = pta.pivot_point(data, timezone="EET")

        first_day = ohlc_resample(data, "1D", resample_tz="EET").loc["2018-12-31 22:00:00"]
        assert N(r.loc["2019-01-01 22:00"]["P"]) == (first_day.high + first_day.low + first_day.close) / 3

    def test_pivots_highs_lows(self):
        high = pd.Series([1, 2, 3, 4, 5, 6, 5, 4, 3])
        low = pd.Series([5, 4, 3, 2, 3, 3, 1, 3, 4])
        r = pta.pivots_highs_lows(high, low, 3, 2, actual_time=0, align_with_index=False)

        assert all(r.U.dropna() == pd.Series([6], index=[5]))
        assert all(r.L.dropna() == pd.Series([2, 1], index=[3, 6]))

    def test_fdi(self):
        def fdi_classic(x: pd.Series | pd.DataFrame, e_period=30):
            """
            Let's keep it here just for reference
            """
            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = x.values  # type: ignore
            fdi_result = None
            for work_data in pta.running_view(x, e_period, 0):
                price_max = work_data.T.max(axis=0)
                price_min = work_data.T.min(axis=0)
                diff = (work_data.T - price_min) / (price_max - price_min)
                length = np.power(np.power(np.diff(diff.T).T, 2.0) + (1.0 / np.power(e_period, 2.0)), 0.5)
                length = np.sum(length[1:], 0)
                fdi_vs = 1.0 + (np.log(length) + np.log(2.0)) / np.log(2 * e_period)

                if type(fdi_vs) != np.array:
                    fdi_vs = np.array([fdi_vs])

                if fdi_result is None:
                    fdi_result = fdi_vs.copy()
                else:
                    fdi_result = np.vstack([fdi_result, fdi_vs])
            fdi_result[np.isinf(fdi_result)] = 0  # type: ignore
            fdi_result = np.vstack(
                (np.full([e_period, x.shape[-1] if len(x.shape) == 2 else 1], np.nan), fdi_result[1:])
            )
            return fdi_result

        mt4data = pd.read_csv("tests/data/csv/FDI_test.csv", parse_dates=True, header=0, index_col=["time"]).replace(
            -1, np.nan
        )

        q_fdi_reference = fdi_classic(mt4data.close, 30)
        q_fdi = pta.fdi(mt4data.close, 30)

        print(len(mt4data))
        print(len(q_fdi_reference))
        print(len(q_fdi))

        np.testing.assert_almost_equal(q_fdi_reference.flatten(), q_fdi.values, decimal=3)  # type: ignore

        np.testing.assert_almost_equal(mt4data.fdi.values, q_fdi_reference.flatten(), decimal=3)  # type: ignore

        np.testing.assert_almost_equal(
            mt4data.fdi.values,
            q_fdi.values,
            decimal=3,  # type: ignore
        )  # type: ignore
