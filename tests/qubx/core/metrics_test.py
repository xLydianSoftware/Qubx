import numpy as np
import pandas as pd

from qubx.core.metrics import YEARLY, absmaxdd, aggregate_returns, cagr, sharpe_ratio, sortino_ratio
from tests.qubx.ta.utils_for_testing import N


class TestMetrics:
    d_rets = pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )

    # test np array
    def test_np(self):
        res = absmaxdd(np.array([1, 2, 3, 1]))
        assert res[0] == 2
        assert res[1] == 2
        assert res[3] == 3

    def test_indexes(self):
        x = [1, 2, 3, 1]
        mdd, i0, i1, i2, dd = absmaxdd(x)
        assert x[i2] == 1

    # test pd series
    def test_pd(self):
        rng = pd.date_range("1/1/2011", periods=6, freq="h")
        ts = pd.Series([1, 2, 3, 1, 8, 11], index=rng)
        res = absmaxdd(ts)
        assert res[0] == 2
        assert res[1] == 2
        assert res[3] == 4
        assert str(res[4].index[1]) == "2011-01-01 01:00:00"

    # test increasing
    def test_increasing(self):
        res = absmaxdd((np.array([1, 2, 3, 4])))
        assert res[0] == 0
        assert res[1] == 0
        assert res[2] == 0

    # test Lowering
    def test_lowering(self):
        res = absmaxdd((np.array([10, 9, 5, 4])))
        assert res[0] == 6
        assert res[1] == 0
        assert res[3] == 3

    # test incorrect data
    def test_incorrect(self):
        try:
            res = absmaxdd("string")
        except TypeError as ex:
            cur_exception = ex
        assert type(cur_exception) == TypeError
        cur_exception = None

        try:
            res = absmaxdd(10)
        except TypeError as ex:
            cur_exception = ex
        assert type(cur_exception) == TypeError

    def test_cagr(self):
        assert N(cagr(self.d_rets) / 100) == 1.913593

        year_returns = pd.Series(
            np.array([3.0, 3.0, 3.0]) / 100, index=pd.date_range("2000-1-30", periods=3, freq="YE")
        )
        assert N(cagr(year_returns, YEARLY) / 100) == 0.03

    def test_aggregate(self):
        data = pd.Series(
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        )
        assert N(aggregate_returns(data, convert_to="Y")) == [0.04060401]
        assert N(aggregate_returns(data, convert_to="M")) == [0.01, 0.030301]
        assert N(aggregate_returns(data, convert_to="W")) == [0.0, 0.04060401, 0.0]
        assert N(aggregate_returns(data, convert_to="D")) == data.values

    def test_sharpe(self):
        benchmark = pd.Series(
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        )

        assert N(sharpe_ratio(self.d_rets, 0.0)) == 1.7238613961706866
        assert N(sharpe_ratio(self.d_rets, benchmark)) == 0.34111411441060574

    def test_sortino(self):
        assert N(sortino_ratio(self.d_rets, 0.0)) == 2.605531251673693

        incr_returns = pd.Series(
            np.array([np.nan, 1.0, 10.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        )
        assert sortino_ratio(incr_returns, 0.0) == np.inf

        zero_returns = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        assert np.isnan(sortino_ratio(zero_returns, 0.0)) == True
