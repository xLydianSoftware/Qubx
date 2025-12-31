import numpy as np
import pandas as pd
cimport numpy as np
from scipy.special.cython_special import ndtri, stdtrit, gamma
from collections import deque

from qubx.core.series cimport TimeSeries, Indicator, IndicatorOHLC, RollingSum, nans, OHLCV, Bar, SeriesCachedValue
from qubx.core.utils import time_to_str
from qubx.pandaz.utils import scols, srows


cdef extern from "math.h":
    float INFINITY


cdef inline long long floor_t64(long long time, long long dt):
    """
    Floor timestamp by dt
    """
    return time - time % dt


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
    Streaming Standard Deviation indicator (uses population std by default)
    Supports min_periods for early estimates like pandas rolling().std()
    """

    def __init__(self, str name, TimeSeries series, int period, int ddof=0, min_periods=None):
        self.period = period
        self.ddof = ddof  # degrees of freedom (0 for population, 1 for sample)
        # If min_periods not specified, use period (default behavior)
        self.min_periods = min_periods if min_periods is not None else period
        # Ensure min_periods is at least ddof + 1 to avoid division by zero
        if self.min_periods < self.ddof + 1:
            self.min_periods = self.ddof + 1

        # Use deque to track values for proper calculation
        self.values_deque = deque(maxlen=period)
        self.count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double old_value
        cdef double _variance
        cdef double _mean
        cdef int n

        if new_item_started:
            # New bar: add value to deque
            if len(self.values_deque) == self.period:
                # Remove oldest value from sums
                old_value = self.values_deque[0]
                if not np.isnan(old_value):
                    self._sum -= old_value
                    self._sum_sq -= old_value * old_value
                    self.count -= 1

            # Add new value
            self.values_deque.append(value)
            if not np.isnan(value):
                self._sum += value
                self._sum_sq += value * value
                self.count += 1
        else:
            # Update current bar: replace last value
            if len(self.values_deque) > 0:
                old_value = self.values_deque[-1]
                if not np.isnan(old_value):
                    self._sum -= old_value
                    self._sum_sq -= old_value * old_value
                    self.count -= 1

                self.values_deque[-1] = value
                if not np.isnan(value):
                    self._sum += value
                    self._sum_sq += value * value
                    self.count += 1

        # Check if we have enough values
        n = len(self.values_deque)
        # We need at least min_periods values in the deque and enough non-NaN values for calculation
        if n < self.min_periods or self.count < max(self.min_periods, self.ddof + 1):
            return np.nan

        # Use actual count of non-NaN values for calculation
        if self.count == 0:
            return np.nan

        _mean = self._sum / self.count

        # Calculate variance
        if self.ddof == 0:
            # Population variance
            _variance = self._sum_sq / self.count - _mean * _mean
        else:
            # Sample variance (Bessel's correction)
            if self.count <= self.ddof:
                return np.nan
            _variance = (self._sum_sq - self._sum * self._sum / self.count) / (self.count - self.ddof)

        # Handle numerical errors that might make variance negative
        if _variance < 0:
            _variance = 0

        # Return the square root of the variance (standard deviation)
        return np.sqrt(_variance)


def std(series: TimeSeries, period: int, ddof: int = 0, min_periods=None):
    return Std.wrap(series, period, ddof, min_periods)


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


cdef class BollingerBands(Indicator):
    """
    Bollinger Bands indicator using rolling SMA/other smoother and Std
    """

    def __init__(self, str name, TimeSeries series, int period, double nstd, str smoother):
        self.period = period
        self.nstd = nstd
        self.smoother = smoother
        
        # Create temporary series for tracking values
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        
        # Create the moving average based on smoother type
        self.ma = smooth(self.tr, smoother, period)
        
        # Create standard deviation indicator with ddof=1 for sample std (matching pandas)
        self.std = std(self.tr, period, ddof=1)
        
        # Create upper and lower band series
        self.upper = TimeSeries("upper", series.timeframe, series.max_series_length)
        self.lower = TimeSeries("lower", series.timeframe, series.max_series_length)
        
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        # Update the tracking series
        self.tr.update(time, value)
        
        # Get the current moving average
        cdef double ma_value = self.ma[0] if len(self.ma) > 0 else np.nan
        
        # Get the current standard deviation
        cdef double std_value = self.std[0] if len(self.std) > 0 else np.nan
        
        # Declare band variables
        cdef double upper_band
        cdef double lower_band
        
        # Calculate bands if we have valid values
        if not np.isnan(ma_value) and not np.isnan(std_value):
            upper_band = ma_value + self.nstd * std_value
            lower_band = ma_value - self.nstd * std_value
            
            # Update the band series
            self.upper.update(time, upper_band)
            self.lower.update(time, lower_band)
            
            # Return the middle band (moving average)
            return ma_value
        else:
            # Update with NaN if we don't have enough data
            self.upper.update(time, np.nan)
            self.lower.update(time, np.nan)
            return np.nan


def bollinger_bands(series: TimeSeries, period: int = 20, nstd: float = 2.0, smoother: str = "sma"):
    """
    Bollinger Bands indicator
    
    :param series: Input time series
    :param period: Period for moving average and standard deviation (default 20)
    :param nstd: Number of standard deviations for bands (default 2.0)
    :param smoother: Type of smoother to use for middle band (default "sma")
    :return: BollingerBands indicator with middle, upper, and lower bands
    """
    return BollingerBands.wrap(series, period, nstd, smoother)


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


cdef class Pivots(IndicatorOHLC):
    """
    Pivot points detector that identifies local highs and lows using
    lookback (before) and lookahead (after) windows.
    """

    def __init__(self, str name, OHLCV series, int before, int after):
        self.before = before
        self.after = after
        
        # Deque to store completed bars for pivot detection
        self.bars_buffer = deque(maxlen=before + after + 1)
        
        # Keep track of the current unfinished bar separately
        self.current_bar = None
        self.current_bar_time = 0
        
        # TimeSeries for pivot points
        self.tops = TimeSeries("tops", series.timeframe, series.max_series_length)
        self.bottoms = TimeSeries("bottoms", series.timeframe, series.max_series_length)
        self.tops_detection_lag = TimeSeries("tops_lag", series.timeframe, series.max_series_length)
        self.bottoms_detection_lag = TimeSeries("bottoms_lag", series.timeframe, series.max_series_length)
        
        super().__init__(name, series)
    
    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        cdef int pivot_idx, i
        cdef double pivot_high, pivot_low
        cdef long long pivot_time
        cdef short is_pivot_high, is_pivot_low
        
        if new_item_started:
            # If we have a previous bar that was being updated, add it to the buffer as completed
            if self.current_bar is not None and self.current_bar_time > 0:
                self.bars_buffer.append((self.current_bar_time, self.current_bar))
            
            # Start tracking the new unfinished bar
            self.current_bar = bar
            self.current_bar_time = time
            
            # Check if we have enough completed bars to detect a pivot
            # We need exactly before + after + 1 bars in the buffer
            if len(self.bars_buffer) < self.before + self.after + 1:
                return np.nan
            
            # The pivot candidate is at index 'before'
            pivot_idx = self.before
            pivot_time = self.bars_buffer[pivot_idx][0]
            pivot_bar = self.bars_buffer[pivot_idx][1]
            pivot_high = pivot_bar.high
            pivot_low = pivot_bar.low
            
            # Check for pivot high: pivot high must be > all other highs in window
            is_pivot_high = 1
            for i in range(len(self.bars_buffer)):
                if i != pivot_idx:
                    if self.bars_buffer[i][1].high >= pivot_high:
                        is_pivot_high = 0
                        break
            
            # Check for pivot low: pivot low must be < all other lows in window
            is_pivot_low = 1
            for i in range(len(self.bars_buffer)):
                if i != pivot_idx:
                    if self.bars_buffer[i][1].low <= pivot_low:
                        is_pivot_low = 0
                        break
            
            # Record pivot high if found
            if is_pivot_high:
                self.tops.update(pivot_time, pivot_high)
                # Detection time is now (when we actually detect it)
                self.tops_detection_lag.update(pivot_time, time - pivot_time)
            
            # Record pivot low if found
            if is_pivot_low:
                self.bottoms.update(pivot_time, pivot_low)
                # Detection time is now (when we actually detect it)
                self.bottoms_detection_lag.update(pivot_time, time - pivot_time)
            
            # Return 1 for pivot high, -1 for pivot low, 0 for both, nan for neither
            if is_pivot_high and is_pivot_low:
                return 0
            elif is_pivot_high:
                return 1
            elif is_pivot_low:
                return -1
            else:
                return np.nan
        else:
            # Just update the current unfinished bar
            self.current_bar = bar
            # Note: current_bar_time stays the same since we're updating the same bar
            return np.nan

    def pd(self) -> pd.DataFrame:
        """
        Return DataFrame with pivot points and detection lags.
        
        Returns a multi-column DataFrame with:
        - Tops: price, detection_lag, spotted (time when pivot was detected)
        - Bottoms: price, detection_lag, spotted
        """
        from qubx.pandaz.utils import scols
        
        tps = self.tops.pd()
        bts = self.bottoms.pd()
        tpl = self.tops_detection_lag.pd()
        btl = self.bottoms_detection_lag.pd()
        
        # Convert lags to timedeltas
        if len(tpl) > 0:
            tpl = tpl.apply(lambda x: pd.Timedelta(x, unit='ns'))
        if len(btl) > 0:
            btl = btl.apply(lambda x: pd.Timedelta(x, unit='ns'))
        
        # Create DataFrames for tops and bottoms
        if len(tps) > 0:
            tops_df = pd.DataFrame({
                'price': tps,
                'detection_lag': tpl,
                'spotted': pd.Series(tps.index + tpl.values, index=tps.index)
            })
        else:
            tops_df = pd.DataFrame(columns=['price', 'detection_lag', 'spotted'])
        
        if len(bts) > 0:
            bottoms_df = pd.DataFrame({
                'price': bts,
                'detection_lag': btl,
                'spotted': pd.Series(bts.index + btl.values, index=bts.index)
            })
        else:
            bottoms_df = pd.DataFrame(columns=['price', 'detection_lag', 'spotted'])
        
        return scols(tops_df, bottoms_df, keys=["Tops", "Bottoms"])


def pivots(series: OHLCV, before: int = 5, after: int = 5):
    """
    Pivot points detector using lookback/lookahead windows.

    :param series: OHLCV series
    :param before: Number of bars to look back
    :param after: Number of bars to look ahead
    :return: Pivots indicator with tops and bottoms
    """
    if not isinstance(series, OHLCV):
        raise ValueError("Series must be OHLCV!")
    return Pivots.wrap(series, before, after)


cdef class PctChange(Indicator):
    """
    Percentage change indicator that calculates the percentage change
    between current value and value from n periods ago.

    Note: Returns percentage as a decimal (0.01 for 1%), matching pandas behavior.
    """
    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        if period <= 0:
            raise ValueError("Period must be positive and greater than zero")

        # Buffer to store past values for the specified period
        self.past_values = deque(maxlen=period + 1)
        self._count = 0

        # - store/restore pattern for handling bar updates
        self.stored_past_values = None
        self.stored_count = 0

        super().__init__(name, series)

    cdef void _store(self):
        """Store current state before bar update"""
        self.stored_past_values = deque(self.past_values, maxlen=self.period + 1)
        self.stored_count = self._count

    cdef void _restore(self):
        """Restore state for bar update"""
        if self.stored_past_values is not None:
            self.past_values = deque(self.stored_past_values, maxlen=self.period + 1)
            self._count = self.stored_count

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double prev_value
        cdef double result

        # - If this is the very first value (indicator created on empty series),
        # - treat it as a new item regardless of new_item_started flag
        if len(self.past_values) == 0:
            new_item_started = True

        # - store/restore pattern for bar updates
        if not new_item_started:
            self._restore()

        if new_item_started:
            # New bar started, add value to history
            self.past_values.append(value)
            self._count += 1
            # Store state for potential bar updates
            self._store()
        else:
            # Updating existing bar - update the last value
            if len(self.past_values) > 0:
                self.past_values[-1] = value

        # Calculate percentage change if we have enough history
        if len(self.past_values) > self.period:
            # Get the value from 'period' bars ago
            prev_value = self.past_values[-(self.period + 1)]

            # Handle zero or NaN previous value
            if np.isnan(prev_value) or prev_value == 0:
                return np.nan

            # Calculate percentage change (as decimal, like pandas)
            result = (value - prev_value) / prev_value
            return result
        else:
            # Not enough history yet
            return np.nan


def pct_change(series: TimeSeries, period: int = 1):
    """
    Calculate percentage change between current value and value from n periods ago.

    Returns the percentage change as a decimal (e.g., 0.01 for 1% increase),
    matching the behavior of pandas.DataFrame.pct_change().

    :param series: Input time series
    :param period: Number of periods to shift for calculating percentage change (default 1)
    :return: PctChange indicator
    """
    return PctChange.wrap(series, period)


cdef class Rsi(Indicator):
    """
    Relative Strength Index indicator

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Formula:
        U = max(0, price_change)
        D = max(0, -price_change)
        RS = smooth(U) / smooth(D)
        RSI = 100 - (100 / (1 + RS))

    Or equivalently:
        RSI = 100 * smooth(U) / (smooth(U) + smooth(D))
    """

    def __init__(self, str name, TimeSeries series, int period, str smoother="ema"):
        self.period = period
        self.prev_value = np.nan

        # - create internal series for up and down moves
        self.up_moves = TimeSeries("up_moves", series.timeframe, series.max_series_length)
        self.down_moves = TimeSeries("down_moves", series.timeframe, series.max_series_length)

        # - create smoothers for up and down moves
        self.smooth_up = smooth(self.up_moves, smoother, period)
        self.smooth_down = smooth(self.down_moves, smoother, period)

        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double diff, up_move, down_move, smooth_u, smooth_d

        # - for the first value, store it and return NaN
        if np.isnan(self.prev_value):
            self.prev_value = value
            self.up_moves.update(time, 0.0)
            self.down_moves.update(time, 0.0)
            return np.nan

        # - calculate price change
        if new_item_started:
            diff = value - self.prev_value
            self.prev_value = value
        else:
            # - when updating the same bar, calculate diff from previous closed bar
            diff = value - self.prev_value

        # - calculate up and down moves
        if diff > 0:
            up_move = diff
            down_move = 0.0
        elif diff < 0:
            up_move = 0.0
            down_move = -diff  # - make it positive
        else:
            up_move = 0.0
            down_move = 0.0

        # - update internal series
        self.up_moves.update(time, up_move)
        self.down_moves.update(time, down_move)

        # - get smoothed values
        smooth_u = self.smooth_up[0]
        smooth_d = self.smooth_down[0]

        # - check if we have valid smoothed values
        if np.isnan(smooth_u) or np.isnan(smooth_d):
            return np.nan

        # - avoid division by zero
        if smooth_u + smooth_d == 0:
            return 50.0  # - neutral value when there's no movement

        # - calculate RSI
        return 100.0 * smooth_u / (smooth_u + smooth_d)


def rsi(series: TimeSeries, period: int = 14, smoother: str = "ema"):
    """
    Relative Strength Index indicator

    The RSI is a momentum oscillator that measures the speed and magnitude
    of recent price changes to evaluate overbought or oversold conditions.

    :param series: Input time series
    :param period: Number of periods for smoothing (default 14)
    :param smoother: Smoothing method: 'sma', 'ema', 'tema', 'dema', 'kama' (default 'ema')
    :return: RSI indicator (values range from 0 to 100)
    """
    return Rsi.wrap(series, period, smoother) # type: ignore


cdef class StdEma(Indicator):
    """
    Standard deviation using exponential weighted moving average.

    Calculates the EWM standard deviation of input series values.
    Works on any TimeSeries (returns, price differences, etc.).
    Matches pandas.Series.ewm(span=period).std() behavior (default adjust=True).

    Uses incremental algorithm for O(1) updates instead of O(n).
    """

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.count = 0

        # - incremental EWM accumulators (for adjust=True mode)
        self.ewm_mean_numer = 0.0  # - sum(w_i * x_i)
        self.ewm_mean_denom = 0.0  # - sum(w_i)
        self.ewm_var_numer = 0.0   # - sum(w_i * (x_i - mean)^2)
        self.ewm_var_denom = 0.0   # - sum(w_i) for variance
        self.prev_mean = 0.0

        # - store/restore pattern for handling bar updates
        self.stored_count = 0
        self.stored_ewm_mean_numer = 0.0
        self.stored_ewm_mean_denom = 0.0
        self.stored_ewm_var_numer = 0.0
        self.stored_ewm_var_denom = 0.0
        self.stored_prev_mean = 0.0

        super().__init__(name, series)

    cdef void _store(self):
        """Store current state before bar update"""
        self.stored_count = self.count
        self.stored_ewm_mean_numer = self.ewm_mean_numer
        self.stored_ewm_mean_denom = self.ewm_mean_denom
        self.stored_ewm_var_numer = self.ewm_var_numer
        self.stored_ewm_var_denom = self.ewm_var_denom
        self.stored_prev_mean = self.prev_mean

    cdef void _restore(self):
        """Restore state for bar update (don't restore count)"""
        # - Note: We don't restore count because it should only increment on new bars
        self.ewm_mean_numer = self.stored_ewm_mean_numer
        self.ewm_mean_denom = self.stored_ewm_mean_denom
        self.ewm_var_numer = self.stored_ewm_var_numer
        self.ewm_var_denom = self.stored_ewm_var_denom
        self.prev_mean = self.stored_prev_mean

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double w_decay, new_weight, delta, delta_new
        cdef double current_mean, old_value_estimate, numer_without_last, var_without_last

        # - skip NaN values
        if np.isnan(value):
            return np.nan

        # - If this is the very first value (indicator created on empty series),
        # - treat it as a new item regardless of new_item_started flag
        if self.count == 0:
            new_item_started = True

        # - store/restore pattern for bar updates
        if not new_item_started:
            self._restore()

        # - weight decay factor: all existing weights get multiplied by (1-alpha)
        w_decay = 1.0 - self.alpha

        # - new observation gets weight 1.0
        new_weight = 1.0

        if new_item_started:
            # Store state BEFORE adding new bar
            self._store()

            # - decay all previous weights
            self.ewm_mean_numer = w_decay * self.ewm_mean_numer + new_weight * value
            self.ewm_mean_denom = w_decay * self.ewm_mean_denom + new_weight

            # - calculate current mean
            current_mean = self.ewm_mean_numer / self.ewm_mean_denom if self.ewm_mean_denom > 0 else 0.0

            # - for variance, we need to update with the new delta
            # - when adding a new point, we update variance incrementally
            delta = value - self.prev_mean
            delta_new = value - current_mean

            self.ewm_var_numer = w_decay * self.ewm_var_numer + new_weight * delta * delta_new
            self.ewm_var_denom = w_decay * self.ewm_var_denom + new_weight

            self.prev_mean = current_mean
            self.count += 1
        else:
            # - bar update: after restore, we're back to state BEFORE current bar
            # - apply same logic as new item (decay and add), but don't increment count
            self.ewm_mean_numer = w_decay * self.ewm_mean_numer + new_weight * value
            self.ewm_mean_denom = w_decay * self.ewm_mean_denom + new_weight

            # - calculate current mean
            current_mean = self.ewm_mean_numer / self.ewm_mean_denom if self.ewm_mean_denom > 0 else 0.0

            # - update variance
            delta = value - self.prev_mean
            delta_new = value - current_mean

            self.ewm_var_numer = w_decay * self.ewm_var_numer + new_weight * delta * delta_new
            self.ewm_var_denom = w_decay * self.ewm_var_denom + new_weight

            self.prev_mean = current_mean

        # - need at least `period` values before we can calculate
        if self.count < self.period:
            return np.nan

        # - calculate EWM variance with bias correction
        # - for adjust=True mode, we need sum(w_i^2) / sum(w_i)
        # - for weights w_i = (1-alpha)^(n-1-i), we can compute this incrementally
        cdef double sum_weights = self.ewm_var_denom
        cdef double r_sq = w_decay * w_decay
        cdef double sum_weights_sq
        cdef int n
        cdef double bias_correction
        cdef double ewm_var

        # - sum of squared weights: sum((1-alpha)^(2*(n-1-i)))
        # - this is a geometric series: 1 + r^2 + r^4 + ... where r = (1-alpha)
        # - for large n, this converges to 1 / (1 - r^2) = 1 / (1 - (1-alpha)^2) = 1 / (alpha * (2-alpha))
        # - but we need exact value for finite n

        # - use the asymptotic approximation (valid for n >> period)
        if self.count > self.period * 2:
            # - use asymptotic formula for efficiency
            sum_weights_sq = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else sum_weights
        else:
            # - for early values, compute exactly
            # - sum_sq = sum((1-alpha)^(2i)) for i=0 to n-1 = (1 - r^(2n)) / (1 - r^2)
            n = self.count
            sum_weights_sq = (1.0 - r_sq ** n) / (1.0 - r_sq) if r_sq < 1.0 else <double>n

        # - bias correction factor
        bias_correction = sum_weights - sum_weights_sq / sum_weights if sum_weights > 0 else 1.0
        ewm_var = self.ewm_var_numer / bias_correction if bias_correction > 0 else 0.0

        # - handle numerical errors
        if ewm_var < 0:
            ewm_var = 0.0

        # - return standard deviation
        return np.sqrt(ewm_var)


def stdema(series: TimeSeries, period: int):
    """
    Calculate exponential weighted moving standard deviation. This is equivalent of 
    ```
    series.ewm(span=period, min_periods=period).std()
    ```

    Parameters
    ----------
    series : TimeSeries
        Input series (returns, price differences, etc.)
    period : int
        EWM span parameter

    Returns
    -------
    StdEma
        Standard deviation indicator
    """
    return StdEma.wrap(series, period) # type: ignore


cdef class CusumFilter(Indicator):
    """
    CUSUM filter on streaming data

    Detects significant changes in a time series by tracking cumulative deviations.
    Returns 1 when an event is detected (cumulative change exceeds threshold), 0 otherwise.

    The algorithm tracks both positive and negative cumulative sums and triggers events
    when either exceeds the threshold. The threshold is dynamically calculated from a
    target series (e.g., volatility) multiplied by the current value.
    """

    def __init__(self, str name, TimeSeries series, TimeSeries target):
        self.target_cache = SeriesCachedValue(target)
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.prev_value = np.nan
        self.last_value = np.nan  # - Track last processed value
        self.prev_bar_event = 0.0  # - Event from previous completed bar (what we return)
        self.current_bar_event = 0.0  # - Event for current bar
        super().__init__(name, series)

    cdef void _store(self):
        """
        Store state before processing update
        """
        self.saved_s_pos = self.s_pos
        self.saved_s_neg = self.s_neg
        self.saved_prev_value = self.prev_value

    cdef void _restore(self):
        """
        Restore state when bar is updated (not new)
        Restores cumulative sums but NOT prev_value
        prev_value must remain as previous bar's final value for correct diff calculation
        """
        self.s_pos = self.saved_s_pos
        self.s_neg = self.saved_s_neg
        # - DON'T restore prev_value - keep it as previous bar's final value

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double diff, threshold, target_value
        cdef int event = 0
        cdef double return_value

        # - first value - just store it
        if np.isnan(self.prev_value):
            self.prev_value = value
            self.last_value = value
            self._store()
            self.current_bar_event = 0.0
            return 0.0

        # - for new bar, update prev_value to last bar's final value
        if new_item_started:
            self.prev_value = self.last_value

        # - for intrabar updates, restore state
        if not new_item_started:
            self._restore()
        else:
            # - for new bar: check if this is OPEN tick (value ~ last_value) or direct CLOSE
            # - if OPEN tick, skip processing and just store state for intrabar updates
            # - if direct CLOSE (static OHLC), process normally
            # - use relative threshold (0.01% of price) to handle different price ranges
            threshold = abs(self.last_value * 0.0001) + 0.001  # 0.01% + small absolute
            if abs(value - self.last_value) < threshold:  # OPEN tick (value ~ previous CLOSE)
                # - store current state without processing
                self._store()
                self.last_value = value
                self.prev_bar_event = self.current_bar_event
                self.current_bar_event = 0.0
                return self.prev_bar_event

        # - calculate diff
        diff = value - self.prev_value

        # - update cumulative sums (do this regardless of threshold availability)
        self.s_pos = max(0.0, self.s_pos + diff)
        self.s_neg = min(0.0, self.s_neg + diff)

        # - get threshold from target series (it can be from higher timeframe)
        # - to get the FINAL value from previous period (not partial value from same time yesterday),
        # - we need to look up just before the start of current period
        # - this ensures we use yesterday's FINAL volatility for today's CUSUM calculations
        cdef long long current_period_start = floor_t64(time, self.target_cache.ser.timeframe)
        cdef long long lookup_time = current_period_start - 1  # - 1 nanosecond before current period
        target_value = self.target_cache.value(lookup_time)

        # - check for events if threshold is available
        if not np.isnan(target_value):
            threshold = abs(target_value * value)

            # - check for events (comparisons with valid threshold)
            if self.s_neg < -threshold:
                self.s_neg = 0.0
                event = 1
            elif self.s_pos > threshold:
                self.s_pos = 0.0
                event = 1

        # - Store state and return event
        # - Both static OHLC and tick-based CLOSE should return current event
        if new_item_started:
            # - new bar starting (static OHLC with CLOSE value, or tick-based OPEN - but OPEN was skipped above)
            self.last_value = value
            self._store()
            return float(event)
        else:
            # - intrabar update (tick-based: HIGH, LOW, or CLOSE ticks)
            self.last_value = value
            # - DO NOT store: all intrabar updates restore to bar start
            # - return current event (CLOSE tick event)
            return float(event)


def cusum_filter(series: TimeSeries, target: TimeSeries):
    """
    Cusum filter implementation for streaming data.

    Detects significant changes in a time series by tracking cumulative deviations.
    Returns a TimeSeries with 1 at event points (when cumulative change exceeds threshold)
    and 0 elsewhere.

    Parameters
    ----------
    series : TimeSeries
        Input series (price series, etc.)
    target : TimeSeries
        Target threshold series (typically volatility, can be from higher timeframe like daily)
        The threshold is calculated as target * current_price

    Returns
    -------
    CusumFilter
        CusumFilter indicator that returns 1 when event is detected, 0 otherwise

    Examples
    --------
    >>> # - daily volatility
    >>> vol = stdema(pct_change(daily_close), 30)
    >>> # - detect significant moves on hourly data using daily volatility
    >>> events = cusum_filter(hourly_close, vol * 2)
    """
    return CusumFilter.wrap(series, target) # type: ignore


cdef class Macd(Indicator):
    """
    Moving Average Convergence Divergence (MACD) indicator

    MACD is calculated as:
    1. MACD Line = fast_ma(price) - slow_ma(price)
    2. Signal Line = signal_ma(MACD Line)

    The returned value is the Signal Line (the smoothed MACD).
    """
    def __init__(self, str name, TimeSeries series, fast=12, slow=26, signal=9, method="ema", signal_method="ema"):
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self.method = method
        self.signal_method = signal_method

        # - create internal copy of input series to track values
        self.input_series = TimeSeries("input", series.timeframe, series.max_series_length)

        # - create fast and slow moving averages on the internal series
        self.fast_ma = smooth(self.input_series, method, fast)
        self.slow_ma = smooth(self.input_series, method, slow)

        # - create internal series for MACD line (fast - slow)
        self.macd_line_series = TimeSeries("macd_line", series.timeframe, series.max_series_length)

        # - create signal line (smoothed MACD line)
        self.signal_line = smooth(self.macd_line_series, signal_method, signal)

        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        cdef double fast_value, slow_value, macd_value

        # - update internal input series first
        self.input_series.update(time, value)

        # - get fast and slow MA values (they are automatically calculated)
        fast_value = self.fast_ma[0] if len(self.fast_ma) > 0 else np.nan
        slow_value = self.slow_ma[0] if len(self.slow_ma) > 0 else np.nan

        # - calculate MACD line (fast - slow)
        if np.isnan(fast_value) or np.isnan(slow_value):
            macd_value = np.nan
        else:
            macd_value = fast_value - slow_value

        # - update MACD line series
        self.macd_line_series.update(time, macd_value)

        # - return signal line value (smoothed MACD)
        return self.signal_line[0] if len(self.signal_line) > 0 else np.nan


def macd(series: TimeSeries, fast=12, slow=26, signal=9, method="ema", signal_method="ema"):
    """
    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices. The MACD is calculated by subtracting the 26-day slow moving average from the
    12-day fast MA. A nine-day MA of the MACD, called the "signal line", is then plotted on top of the MACD,
    functioning as a trigger for buy and sell signals.

    :param series: input data
    :param fast: fast MA period
    :param slow: slow MA period
    :param signal: signal MA period
    :param method: used moving averaging method (sma, ema, tema, dema, kama)
    :param signal_method: used method for averaging signal (sma, ema, tema, dema, kama)
    :return: macd indicator
    """
    return Macd.wrap(series, fast, slow, signal, method, signal_method) # type: ignore


cdef class SuperTrend(IndicatorOHLC):
    """
    SuperTrend indicator - a trend-following indicator based on ATR

    The SuperTrend indicator provides trend direction and support/resistance levels.
    It uses Average True Range (ATR) to calculate dynamic bands around price.

    Returns:
        trend: +1 for uptrend, -1 for downtrend
        utl (upper trend line): longstop values during uptrend
        dtl (down trend line): shortstop values during downtrend
    """

    def __init__(self, str name, OHLCV series, int length, double mult, str src, short wicks, str atr_smoother):
        self.length = length
        self.mult = mult
        self.src = src
        self.wicks = wicks
        self.atr_smoother = atr_smoother

        # - working state variables (updated during calculation)
        self._prev_longstop = np.nan
        self._prev_shortstop = np.nan
        self._prev_direction = np.nan

        # - saved state variables (for handling partial bar updates)
        self.prev_longstop = np.nan
        self.prev_shortstop = np.nan
        self.prev_direction = np.nan

        # - create internal TR series and smooth it for ATR calculation
        self.tr = TimeSeries("tr", series.timeframe, series.max_series_length)
        self.atr_ma = smooth(self.tr, atr_smoother, length)

        # - create output series for upper and down trend lines
        self.utl = TimeSeries("utl", series.timeframe, series.max_series_length)
        self.dtl = TimeSeries("dtl", series.timeframe, series.max_series_length)

        super().__init__(name, series)

    cdef _store(self):
        """Store current working state to saved state"""
        self.prev_longstop = self._prev_longstop
        self.prev_shortstop = self._prev_shortstop
        self.prev_direction = self._prev_direction

    cdef _restore(self):
        """Restore saved state to working state"""
        self._prev_longstop = self.prev_longstop
        self._prev_shortstop = self.prev_shortstop
        self._prev_direction = self.prev_direction

    cdef double calc_src(self, Bar bar):
        """Calculate source value based on src parameter"""
        if self.src == "close":
            return bar.close
        elif self.src == "hl2":
            return (bar.high + bar.low) / 2.0
        elif self.src == "hlc3":
            return (bar.high + bar.low + bar.close) / 3.0
        elif self.src == "ohlc4":
            return (bar.open + bar.high + bar.low + bar.close) / 4.0
        else:
            return (bar.high + bar.low) / 2.0  # - default to hl2

    cpdef double calculate(self, long long time, Bar bar, short new_item_started):
        cdef double atr_value, src_value, longstop, shortstop
        cdef double high_price, low_price, p_high_price, p_low_price
        cdef short is_doji4price
        cdef double direction
        cdef double saved_prev_longstop, saved_prev_shortstop
        cdef double tr_value

        # - need at least 2 bars for prev calculations
        if len(self.series) < 2:
            # - initialize on first bar
            self._prev_longstop = np.nan
            self._prev_shortstop = np.nan
            self._prev_direction = np.nan
            self._store()
            return np.nan

        # - handle store/restore for partial bar updates
        if new_item_started:
            self._store()
        else:
            self._restore()

        # - calculate True Range
        cdef Bar prev_bar = self.series[1]
        cdef double c1 = prev_bar.close
        cdef double h_l = abs(bar.high - bar.low)
        cdef double h_pc = abs(bar.high - c1)
        cdef double l_pc = abs(bar.low - c1)
        tr_value = max(h_l, h_pc, l_pc)

        # - update TR series (this will automatically update ATR via smooth)
        self.tr.update(time, tr_value)

        # - get ATR value
        atr_value = self.atr_ma[0]
        if np.isnan(atr_value):
            return np.nan

        atr_value = abs(self.mult) * atr_value

        # - calculate source value
        src_value = self.calc_src(bar)

        # - determine which prices to use for wicks
        if self.wicks:
            high_price = bar.high
            low_price = bar.low
        else:
            high_price = bar.close
            low_price = bar.close

        # - get previous bar's prices (prev_bar already retrieved above for TR)
        if self.wicks:
            p_high_price = prev_bar.high
            p_low_price = prev_bar.low
        else:
            p_high_price = prev_bar.close
            p_low_price = prev_bar.close

        # - check for doji4price (all prices equal)
        is_doji4price = (bar.open == bar.close) and (bar.open == bar.low) and (bar.open == bar.high)

        # - calculate basic stops
        longstop = src_value - atr_value
        shortstop = src_value + atr_value

        # - save previous bar's stops for direction comparison
        saved_prev_longstop = self._prev_longstop
        saved_prev_shortstop = self._prev_shortstop

        # - adjust longstop based on previous value
        if np.isnan(self._prev_longstop):
            self._prev_longstop = longstop

        if longstop > 0:
            if is_doji4price:
                longstop = self._prev_longstop
            else:
                if p_low_price > self._prev_longstop:
                    longstop = max(longstop, self._prev_longstop)
                # - else: keep calculated longstop
        else:
            longstop = self._prev_longstop

        # - adjust shortstop based on previous value
        if np.isnan(self._prev_shortstop):
            self._prev_shortstop = shortstop

        if shortstop > 0:
            if is_doji4price:
                shortstop = self._prev_shortstop
            else:
                if p_high_price < self._prev_shortstop:
                    shortstop = min(shortstop, self._prev_shortstop)
                # - else: keep calculated shortstop
        else:
            shortstop = self._prev_shortstop

        # - determine direction based on price breaking PREVIOUS bar's stops
        # - only check for breaks if we have valid previous stops
        if np.isnan(saved_prev_longstop) or np.isnan(saved_prev_shortstop):
            # - not enough data yet, forward fill previous direction or return NaN
            direction = self._prev_direction if not np.isnan(self._prev_direction) else np.nan
        elif low_price < saved_prev_longstop:
            direction = -1.0
        elif high_price > saved_prev_shortstop:
            direction = 1.0
        else:
            # - no break, keep previous direction (forward fill)
            direction = self._prev_direction if not np.isnan(self._prev_direction) else np.nan

        # - update working state for next iteration
        self._prev_longstop = longstop
        self._prev_shortstop = shortstop
        self._prev_direction = direction

        # - update utl and dtl series based on direction
        # - only update when we have a valid direction
        if direction == 1.0:
            self.utl.update(time, longstop)
        elif direction == -1.0:
            self.dtl.update(time, shortstop)

        # - return trend direction
        return direction


def super_trend(series: OHLCV, length: int = 22, mult: float = 3.0, src: str = "hl2",
                wicks: bool = True, atr_smoother: str = "sma"):
    """
    SuperTrend indicator - a trend-following indicator based on ATR

    The SuperTrend indicator uses Average True Range (ATR) to calculate dynamic support
    and resistance levels. It provides clear trend direction and can be used for
    entry/exit signals.

    :param series: OHLCV input series
    :param length: ATR period (default 22)
    :param mult: ATR multiplier (default 3.0)
    :param src: source calculation - 'close', 'hl2', 'hlc3', 'ohlc4' (default 'hl2')
    :param wicks: whether to use high/low (True) or close (False) for calculations (default True)
    :param atr_smoother: smoothing method for ATR - 'sma', 'ema', etc (default 'sma')
    :return: SuperTrend indicator with trend, utl, and dtl series
    """
    if not isinstance(series, OHLCV):
        raise ValueError("Series must be OHLCV !")
    return SuperTrend.wrap(series, length, mult, src, wicks, atr_smoother) # type: ignore