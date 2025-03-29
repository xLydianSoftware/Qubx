import numpy as np
import pandas as pd
cimport numpy as np
from scipy.special.cython_special import ndtri, stdtrit, gamma
from collections import deque

from qubx.core.series cimport TimeSeries, Indicator, IndicatorOHLC, RollingSum, nans, OHLCV, Bar
from qubx.core.utils import time_to_str
from qubx.pandaz.utils import scols, srows


cdef extern from "math.h":
    float INFINITY


cdef class Sma(Indicator):
    """
    Simple Moving Average indicator
    """

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.summator = RollingSum(period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double r = self.summator.update(value, new_item_started)
        return np.nan if self.summator.is_init_stage else r / self.period


def sma(series:TimeSeries, period: int): 
    return Sma.wrap(series, period)


cdef class Ema(Indicator):
    """
    Exponential moving average
    """

    def __init__(self, str name, TimeSeries series, int period, init_mean=True):
        self.period = period

        # when it's required to initialize this ema by mean on first period
        self.init_mean = init_mean
        if init_mean:
            self.__s = nans(period)
            self.__i = 0

        self._init_stage = 1
        self.alpha = 2.0 / (1.0 + period)
        self.alpha_1 = (1 - self.alpha)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef int prev_bar_idx = 0 if new_item_started else 1 

        if self._init_stage:
            if np.isnan(value): return np.nan

            if new_item_started:
                self.__i += 1
                if self.__i > self.period - 1:
                    self._init_stage = False
                    return self.alpha * value + self.alpha_1 * self[prev_bar_idx]

            if self.__i == self.period - 1:
                self.__s[self.__i] = value 
                return np.nansum(self.__s) / self.period

            self.__s[self.__i] = value 
            return np.nan

        if len(self) == 0:
            return value

        return self.alpha * value + self.alpha_1 * self[prev_bar_idx]


def ema(series:TimeSeries, period: int, init_mean: bool = True):
    return Ema.wrap(series, period, init_mean=init_mean)


cdef class Tema(Indicator):

    def __init__(self, str name, TimeSeries series, int period, init_mean=True):
        self.period = period
        self.init_mean = init_mean
        self.ser0 = TimeSeries('ser0', series.timeframe, series.max_series_length)
        self.ema1 = ema(self.ser0, period, init_mean)
        self.ema2 = ema(self.ema1, period, init_mean)
        self.ema3 = ema(self.ema2, period, init_mean)
        super().__init__(name, series)
        
    cpdef double calculate(self, long long time, double value, short new_item_started):
        self.ser0.update(time, value)
        return 3 * self.ema1[0] - 3 * self.ema2[0] + self.ema3[0]


def tema(series:TimeSeries, period: int, init_mean: bool = True):
    return Tema.wrap(series, period, init_mean=init_mean)


cdef class Dema(Indicator):

    def __init__(self, str name, TimeSeries series, int period, init_mean=True):
        self.period = period
        self.init_mean = init_mean
        self.ser0 = TimeSeries('ser0', series.timeframe, series.max_series_length)
        self.ema1 = ema(self.ser0, period, init_mean)
        self.ema2 = ema(self.ema1, period, init_mean)
        super().__init__(name, series)
        
    cpdef double calculate(self, long long time, double value, short new_item_started):
        self.ser0.update(time, value)
        return 2 * self.ema1[0] - self.ema2[0]


def dema(series:TimeSeries, period: int, init_mean: bool = True):
    return Dema.wrap(series, period, init_mean=init_mean)


cdef class Kama(Indicator):
    # cdef int period
    # cdef int fast_span
    # cdef int slow_span
    # cdef double _S1 
    # cdef double _K1 
    # cdef _x_past
    # cdef RollingSum summator

    def __init__(self, str name, TimeSeries series, int period, int fast_span=2, int slow_span=30):
        self.period = period
        self.fast_span = fast_span
        self.slow_span = slow_span
        self._S1 = 2.0 / (slow_span + 1)
        self._K1 = 2.0 / (fast_span + 1) - self._S1
        self._x_past = deque(nans(period+1), period+1)
        self.summator = RollingSum(period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if new_item_started:
            self._x_past.append(value)
        else:
            self._x_past[-1] = value

        cdef double rs = self.summator.update(abs(value - self._x_past[-2]), new_item_started)
        cdef double er = (abs(value - self._x_past[0]) / rs) if rs != 0.0 else 1.0
        cdef double sc = (er * self._K1 + self._S1) ** 2

        if self.summator.is_init_stage:
            if not np.isnan(self._x_past[1]):
                return value
            return np.nan

        return sc * value + (1 - sc) * self[0 if new_item_started else 1]


def kama(series:TimeSeries, period: int, fast_span:int=2, slow_span:int=30):
    return Kama.wrap(series, period, fast_span, slow_span)


cdef class Highest(Indicator):

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.queue = deque([np.nan] * period, maxlen=period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        """
        Not a most effictive algo but simplest and can handle updated last value
        """
        cdef float r = np.nan

        if not np.isnan(value):
            if new_item_started:
                self.queue.append(value)
            else:
                self.queue[-1] = value

        if not np.isnan(self.queue[0]):
            r = max(self.queue) 

        return r


def highest(series:TimeSeries, period:int):
    return Highest.wrap(series, period)


cdef class Lowest(Indicator):

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.queue = deque([np.nan] * period, maxlen=period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        """
        Not a most effictive algo but simplest and can handle updated last value
        """
        cdef float r = np.nan

        if not np.isnan(value):
            if new_item_started:
                self.queue.append(value)
            else:
                self.queue[-1] = value

        if not np.isnan(self.queue[0]):
            r = min(self.queue) 

        return r


def lowest(series:TimeSeries, period:int):
    return Lowest.wrap(series, period)


cdef class Std(Indicator):
    """
    Streaming Standard Deviation indicator
    """

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.rolling_sum = RollingSum(period)
        self.rolling_sum_sq = RollingSum(period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # Update the rolling sum with the new value
        cdef double _sum = self.rolling_sum.update(value, new_item_started)
        cdef double _sum_sq = self.rolling_sum_sq.update(value * value, new_item_started)

        # If we're still in the initialization stage, return NaN
        if self.rolling_sum.is_init_stage or self.rolling_sum_sq.is_init_stage:
            return np.nan

        # Calculate the mean from the rolling sum
        cdef double _mean = _sum / self.period

        # Update the variance sum with the squared deviation from the mean
        cdef double _variance = _sum_sq / self.period - _mean * _mean

        # Return the square root of the variance (standard deviation)
        return np.sqrt(_variance)


def std(series: TimeSeries, period: int, mean=0):
    return Std.wrap(series, period)


cdef double norm_pdf(double x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


cdef double lognorm_pdf(double x, double s):
    return np.exp(-np.log(x) ** 2 / (2 * s ** 2)) / (x * s * np.sqrt(2 * np.pi))


cdef double student_t_pdf(double x, double df):
    """Compute the PDF of the Student's t-distribution."""
    gamma_df = gamma(df / 2.0)
    gamma_df_plus_1 = gamma((df + 1) / 2.0)
    
    # Normalization constant
    normalization = gamma_df_plus_1 / (np.sqrt(df * np.pi) * gamma_df)
    
    # PDF calculation
    term = (1 + (x ** 2) / df) ** (-(df + 1) / 2.0)
    pdf_value = normalization * term
    
    return pdf_value


cdef class Pewma(Indicator):

    def __init__(self, str name, TimeSeries series, double alpha, double beta, int T):
        self.alpha = alpha 
        self.beta = beta
        self.T = T

        # - local variables
        self._i = 0
        self.std = TimeSeries('std', series.timeframe, series.max_series_length)
        super().__init__(name, series)

    def _store(self):
        self.mean = self._mean
        self.vstd = self._vstd
        self.var = self._var

    def _restore(self):
        self._mean = self.mean
        self._vstd = self.vstd
        self._var = self.var

    def _get_alpha(self, p_t):
        if self._i - 1 > self.T:
            return self.alpha * (1.0 - self.beta * p_t)
        return 1.0 - 1.0 / self._i

    cpdef double calculate(self, long long time, double x, short new_item_started):
        cdef double diff, p_t, a_t, incr

        if len(self.series) <= 1:
            self._mean = x
            self._vstd = 0.0
            self._var = 0.0
            self._store()
            self.std.update(time, self.vstd)
            return self.mean

        if new_item_started:
            self._i += 1
            self._restore()
        else:
            self._store()

        diff = x - self.mean
        # prob of observing diff
        p_t = norm_pdf(diff / self.vstd) if self.vstd != 0.0 else 0.0  

        # weight to give to this point
        a_t = self._get_alpha(p_t)  
        incr = (1.0 - a_t) * diff
        self.mean += incr
        self.var = a_t * (self.var + diff * incr)
        self.vstd = np.sqrt(self.var)
        self.std.update(time, self.vstd)

        return self.mean


def pewma(series:TimeSeries, alpha: float, beta: float, T:int=30):
    """
    Implementation of probabilistic exponential weighted ma (https://sci-hub.shop/10.1109/SSP.2012.6319708)
    See pandas version here: qubx.pandaz.ta::pwma 
    """
    return Pewma.wrap(series, alpha, beta, T)


cdef class PewmaOutliersDetector(Indicator):

    def __init__(
        self,
        str name,
        TimeSeries series,
        double alpha,
        double beta,
        int T,
        double threshold,
        str dist = "normal",
        double student_t_df = 3.0
    ):
        self.alpha = alpha 
        self.beta = beta
        self.T = T
        self.threshold = threshold
        self.dist = dist
        self.student_t_df = student_t_df

        # - series
        self.upper = TimeSeries('uba', series.timeframe, series.max_series_length)
        self.lower = TimeSeries('lba', series.timeframe, series.max_series_length)
        self.std = TimeSeries('std', series.timeframe, series.max_series_length)
        self.outliers = TimeSeries('outliers', series.timeframe, series.max_series_length)

        # - local variables
        self._i = 0
        self._z_thr = self._get_z_thr()

        super().__init__(name, series)

    def _store(self):
        self.mean = self._mean
        self.vstd = self._vstd
        self.variance = self._variance

    def _restore(self):
        self._mean = self.mean
        self._vstd = self.vstd
        self._variance = self.variance
    
    cdef double _get_z_thr(self):
        if self.dist == 'normal':
            return ndtri(1 - self.threshold / 2)
        elif self.dist == 'student_t':
            return stdtrit(self.student_t_df, 1 - self.threshold / 2)
        else:
            raise ValueError('Invalid distribution type')

    cdef double _get_alpha(self, double p_t):
        if self._i + 1 >= self.T:
            return self.alpha * (1.0 - self.beta * p_t)
        return 1.0 - 1.0 / (self._i + 1.0)

    cdef double _get_mean(self, double x, double alpha_t):
        return alpha_t * self.mean + (1.0 - alpha_t) * x

    cdef double _get_variance(self, double x, double alpha_t):
        return alpha_t * self.variance + (1.0 - alpha_t) * np.square(x)

    cdef double _get_std(self, double variance, double mean):
        return np.sqrt(max(variance - np.square(mean), 0.0))

    cdef double _get_p(self, double x):
        cdef double z_t = ((x - self.mean) / self.vstd) if (self.vstd != 0 and not np.isnan(x)) else 0.0
        if self.dist == 'normal':
            p_t = norm_pdf(z_t)
        elif self.dist == 'student_t':
            p_t = student_t_pdf(z_t, self.student_t_df)
        # elif self.dist == 'cauchy':
        #     p_t = (1 / (np.pi * (1 + np.square(z_t))))
        else:
            raise ValueError('Invalid distribution type')
        return p_t

    cpdef double calculate(self, long long time, double x, short new_item_started):
        # - first bar - just use it as initial value
        if len(self.series) <= 1:
            self._mean = x
            self._variance = x ** 2
            self._vstd = 0.0
            self._store()
            self.std.update(time, self.vstd)
            self.upper.update(time, x)
            self.lower.update(time, x)
            return self._mean

        # - new bar is started use n-1 values for calculate innovations
        if new_item_started:
            self._i += 1
            self._restore()
        else:
            self._store()

        cdef double p_t = self._get_p(x)
        cdef double alpha_t = self._get_alpha(p_t)
        self.mean = self._get_mean(x, alpha_t)
        self.variance = self._get_variance(x, alpha_t)
        self.vstd = self._get_std(self.variance, self.mean)
        cdef double ub = self.mean + self._z_thr * self.vstd
        cdef double lb = self.mean - self._z_thr * self.vstd

        self.upper.update(time, ub)
        self.lower.update(time, lb)
        self.std.update(time, self.vstd)

        # - check if it's outlier
        if p_t < self.threshold:
            self.outliers.update(time, x)
        else:
            self.outliers.update(time, np.nan)
        return self.mean


def pewma_outliers_detector(
    series: TimeSeries,
    alpha: float,
    beta: float,
    T:int=30,
    threshold=0.05,
    dist: str = "normal",
    **kwargs
):
    """
    Outliers detector based on pwma
    """
    return PewmaOutliersDetector.wrap(series, alpha, beta, T, threshold, dist=dist, **kwargs)


cdef class Psar(IndicatorOHLC):

    def __init__(self, name, series, iaf, maxaf):
        self.iaf = iaf
        self.maxaf = maxaf
        self.upper = TimeSeries('upper', series.timeframe, series.max_series_length)
        self.lower = TimeSeries('lower', series.timeframe, series.max_series_length)
        super().__init__(name, series)

    cdef _store(self):
        self.bull = self._bull
        self.af = self._af
        self.psar = self._psar
        self.lp = self._lp
        self.hp = self._hp

    cdef _restore(self):
        self._bull = self.bull
        self._af = self.af
        self._psar = self.psar
        self._lp = self.lp
        self._hp = self.hp

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        cdef short reverse = 1

        if len(self.series) <= 2:
            self._bull = 1
            self._af = self.iaf
            self._psar = bar.close

            if len(self.series) == 1:
                self._lp = bar.low
                self._hp = bar.high
            self._store()
            return self._psar

        if not new_item_started:
            self._store()
        else:
            self._restore()

        bar1 = self.series[1]
        bar2 = self.series[2]
        cdef double h0 = bar.high
        cdef double l0 = bar.low
        cdef double h1 = bar1.high
        cdef double l1 = bar1.low
        cdef double h2 = bar2.high
        cdef double l2 = bar2.low

        if self.bull:
            self.psar += self.af * (self.hp - self.psar)
        else:
            self.psar += self.af * (self.lp - self.psar)

        reverse = 0
        if self.bull:
            if l0 < self.psar:
                self.bull = 0
                reverse = 1
                self.psar = self.hp
                self.lp = l0
                self.af = self.iaf
        else:
            if h0 > self.psar:
                self.bull = 1
                reverse = 1
                self.psar = self.lp
                self.hp = h0
                self.af = self.iaf

        if not reverse:
            if self.bull:
                if h0 > self.hp:
                    self.hp = h0
                    self.af = min(self.af + self.iaf, self.maxaf)
                if l1 < self.psar:
                    self.psar = l1
                if l2 < self.psar:
                    self.psar = l2
            else:
                if l0 < self.lp:
                    self.lp = l0
                    self.af = min(self.af + self.iaf, self.maxaf)
                if h1 > self.psar:
                    self.psar = h1
                if h2 > self.psar:
                    self.psar = h2

        if self.bull:
            self.lower.update(time, self.psar)
            self.upper.update(time, np.nan)
        else:
            self.upper.update(time, self.psar)
            self.lower.update(time, np.nan)

        return self.psar


def psar(series: OHLCV, iaf: float=0.02, maxaf: float=0.2):
    if not isinstance(series, OHLCV):
        raise ValueError('Series must be OHLCV !')

    return Psar.wrap(series, iaf, maxaf)


# List of smoothing functions
_smoothers = {f.__name__: f for f in [pewma, ema, sma, kama, tema, dema]}


def smooth(TimeSeries series, str smoother, *args, **kwargs) -> Indicator:
    """
    Handy utility function to smooth series
    """
    _sfn = _smoothers.get(smoother)
    if _sfn is None:
        raise ValueError(f"Smoother {smoother} not found!")
    return _sfn(series, *args, **kwargs)


cdef class Atr(IndicatorOHLC):

    def __init__(self, str name, OHLCV series, int period, str smoother, short percentage):
        self.percentage = percentage
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        self.ma = smooth(self.tr, smoother, period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        if len(self.series) <= 1:
            return np.nan

        cdef double c1 = self.series[1].close
        cdef double h_l = abs(bar.high - bar.low)
        cdef double h_pc = abs(bar.high - c1)
        cdef double l_pc = abs(bar.low - c1)
        self.tr.update(time, max(h_l, h_pc, l_pc))
        return (100 * self.ma[0] / c1) if self.percentage else self.ma[0]


def atr(series: OHLCV, period: int = 14, smoother="sma", percentage: bool = False):
    if not isinstance(series, OHLCV):
        raise ValueError("Series must be OHLCV !")
    return Atr.wrap(series, period, smoother, percentage)


cdef class Zscore(Indicator):
    """
    Z-score normalization using rolling SMA and Std
    """

    def __init__(self, str name, TimeSeries series, int period, str smoother):
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        self.ma = smooth(self.tr, smoother, period)
        self.std = std(self.tr, period)
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        self.tr.update(time, value)
        if len(self.ma) < 1 or len(self.std) < 1 or np.isnan(self.ma[0]) or np.isnan(self.std[0]) or self.std[0] == 0:
            return np.nan
        return (value - self.ma[0]) / self.std[0]


def zscore(series: TimeSeries, period: int = 20, smoother="sma"):
    return Zscore.wrap(series, period, smoother)


cdef class Swings(IndicatorOHLC):

    def __init__(self, str name, OHLCV series, trend_indicator, **indicator_args):
        self.base = OHLCV("base", series.timeframe, series.max_series_length)
        self.trend = trend_indicator(self.base, **indicator_args)

        self.tops = TimeSeries("tops", series.timeframe, series.max_series_length)
        self.tops_detection_lag = TimeSeries("tops_lag", series.timeframe, series.max_series_length)

        self.bottoms = TimeSeries("bottoms", series.timeframe, series.max_series_length)
        self.bottoms_detection_lag = TimeSeries("bottoms_lag", series.timeframe, series.max_series_length)

        self.middles = TimeSeries("middles", series.timeframe, series.max_series_length)
        self.deltas = TimeSeries("deltas", series.timeframe, series.max_series_length)

        # - store parameters for copying
        self._trend_indicator = trend_indicator
        self._indicator_args = indicator_args

        self._min_l = +np.inf
        self._max_h = -np.inf
        self._max_t = 0
        self._min_t = 0
        super().__init__(name, series)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        self.base.update_by_bar(time, bar.open, bar.high, bar.low, bar.close, bar.volume)
        cdef int _t = 0

        if len(self.trend.upper) > 0:
            _u = self.trend.upper[0]
            _d = self.trend.lower[0]

            if not np.isnan(_u):
                if self._max_t > 0:
                    self.tops.update(self._max_t, self._max_h)
                    self.tops_detection_lag.update(self._max_t, time - self._max_t)
                    if len(self.bottoms) > 0:
                        self.middles.update(time, (self.tops[0] + self.bottoms[0]) / 2)
                        self.deltas.update(time, self.tops[0] - self.bottoms[0])

                if bar.low <= self._min_l:
                    self._min_l = bar.low
                    self._min_t = time

                self._max_h = -np.inf
                self._max_t = 0
                _t = -1
            elif not np.isnan(_d):
                if self._min_t > 0:
                    self.bottoms.update(self._min_t, self._min_l)
                    self.bottoms_detection_lag.update(self._min_t, time - self._min_t)
                    if len(self.tops) > 0:
                        self.middles.update(time, (self.tops[0] + self.bottoms[0]) / 2)
                        self.deltas.update(time, self.tops[0] - self.bottoms[0])

                if bar.high >= self._max_h:
                    self._max_h = bar.high
                    self._max_t = time

                self._min_l = +np.inf
                self._min_t = 0
                _t = +1

        return _t

    def get_current_trend_end(self):
        if np.isfinite(self._min_l):
            return pd.Timestamp(self._min_t, 'ns'), self._min_l
        elif np.isfinite(self._max_h):
            return pd.Timestamp(self._max_t, 'ns'), self._max_h
        return (None, None)

    def copy(self, int start, int stop):
        n_ts = Swings(self.name, OHLCV("base", self.series.timeframe), self._trend_indicator, **self._indicator_args)

        # - copy main series
        for i in range(start, stop):
            n_ts._add_new_item(self.times.values[i], self.values.values[i])
            n_ts.trend._add_new_item(self.trend.times.values[i], self.trend.values.values[i])

        # - copy internal series
        (
            n_ts.tops, 
            n_ts.tops_detection_lag,
            n_ts.bottoms,
            n_ts.bottoms_detection_lag,
            n_ts.middles,
            n_ts.deltas
        ) = self._copy_internal_series(start, stop, 
            self.tops, 
            self.tops_detection_lag,
            self.bottoms,
            self.bottoms_detection_lag,
            self.middles,
            self.deltas
        )

        return n_ts

    def pd(self) -> pd.DataFrame:
        _t, _d = self.get_current_trend_end()
        tps, bts = self.tops.pd(), self.bottoms.pd()
        tpl, btl = self.tops_detection_lag.pd(), self.bottoms_detection_lag.pd()
        if _t is not None:
            if bts.index[-1] < tps.index[-1]:
                bts = srows(bts, pd.Series({_t: _d}))
                btl = srows(btl, pd.Series({_t: 0}))  # last lag is 0
            else:
                tps = srows(tps, pd.Series({_t: _d}))
                tpl = srows(tpl, pd.Series({_t: 0})) # last lag is 0

        # - convert tpl / btl to timedeltas
        tpl, btl = tpl.apply(lambda x: pd.Timedelta(x, unit='ns')), btl.apply(lambda x: pd.Timedelta(x, unit='ns'))

        eid = pd.Series(tps.index, tps.index)
        mx = scols(bts, tps, eid, names=["start_price", "end_price", "end"])
        dt = scols(mx["start_price"], mx["end_price"].shift(-1), mx["end"].shift(-1))  # .dropna()
        dt = dt.assign(
            delta = dt["end_price"] - dt["start_price"], 
            spotted = pd.Series(bts.index + btl, bts.index)
        )

        eid = pd.Series(bts.index, bts.index)
        mx = scols(tps, bts, eid, names=["start_price", "end_price", "end"])
        ut = scols(mx["start_price"], mx["end_price"].shift(-1), mx["end"].shift(-1))  # .dropna()
        ut = ut.assign(
            delta = ut["end_price"] - ut["start_price"], 
            spotted = pd.Series(tps.index + tpl, tps.index)
        )

        return scols(ut, dt, keys=["DownTrends", "UpTrends"])


def swings(series: OHLCV, trend_indicator, **indicator_args):
    """
    Swing detector based on provided trend indicator.
    """
    if not isinstance(series, OHLCV):
        raise ValueError("Series must be OHLCV !")
    return Swings.wrap(series, trend_indicator, **indicator_args)