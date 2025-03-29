cimport numpy as np
from qubx.core.series cimport Indicator, IndicatorOHLC, RollingSum, TimeSeries, OHLCV, Bar

cdef class Sma(Indicator):
    cdef unsigned int period
    cdef RollingSum summator

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Ema(Indicator):
    cdef int period
    cdef np.ndarray __s
    cdef int __i
    cdef double alpha
    cdef double alpha_1
    cdef unsigned short init_mean 
    cdef unsigned short _init_stage

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Tema(Indicator):
    cdef int period
    cdef unsigned short init_mean 
    cdef TimeSeries ser0
    cdef Ema ema1
    cdef Ema ema2
    cdef Ema ema3
    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Dema(Indicator):
    cdef int period
    cdef unsigned short init_mean 
    cdef TimeSeries ser0
    cdef Ema ema1
    cdef Ema ema2

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Kama(Indicator):
    cdef int period
    cdef int fast_span
    cdef int slow_span
    cdef double _S1 
    cdef double _K1 
    cdef _x_past
    cdef RollingSum summator

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Highest(Indicator):
    cdef int period
    cdef object queue
    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Lowest(Indicator):
    cdef int period
    cdef object queue
    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Std(Indicator):
    cdef int period
    cdef RollingSum rolling_sum, rolling_sum_sq
    cpdef double calculate(self, long long time, double value, short new_item_started)


cdef class Zscore(Indicator):
    cdef TimeSeries tr
    cdef Indicator ma, std
    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Pewma(Indicator):
    cdef public TimeSeries std
    cdef double alpha, beta
    cdef int T

    cdef double _mean, _vstd, _var
    cdef double mean, vstd, var
    cdef long _i

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class PewmaOutliersDetector(Indicator):
    cdef public TimeSeries upper, lower, outliers, std
    cdef double alpha, beta, threshold
    cdef int T
    cdef str dist

    cdef double student_t_df
    cdef long _i
    cdef double mean, vstd, variance
    cdef double _mean, _vstd, _variance, _z_thr

    cpdef double calculate(self, long long time, double x, short new_item_started)

    cdef double _get_z_thr(self)
    cdef double _get_alpha(self, double p_t)
    cdef double _get_mean(self, double x, double alpha_t)
    cdef double _get_variance(self, double x, double alpha_t)
    cdef double _get_std(self, double variance, double mean)
    cdef double _get_p(self, double x)

cdef class Psar(IndicatorOHLC):
    cdef int _bull
    cdef double _af
    cdef double _psar
    cdef double _lp
    cdef double _hp

    cdef int bull
    cdef double af
    cdef double psar
    cdef double lp
    cdef double hp

    cdef public TimeSeries upper
    cdef public TimeSeries lower

    cdef double iaf
    cdef double maxaf

    cdef _store(self)
    cdef _restore(self)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started)

cdef class Atr(IndicatorOHLC):
    cdef short percentage
    cdef TimeSeries tr
    cdef Indicator ma

    cpdef double calculate(self, long long time, Bar bar, short new_item_started)

cdef class Swings(IndicatorOHLC):
    cdef double _min_l
    cdef long long _min_t
    cdef double _max_h
    cdef long long _max_t
    cdef OHLCV base
    cdef Indicator trend

    cdef object _trend_indicator
    cdef object _indicator_args
    # tops contain upper pivot point prices
    # tops_detection_lag contain time lag when top was actually spotted
    cdef public TimeSeries tops, tops_detection_lag
    cdef public TimeSeries bottoms, bottoms_detection_lag
    cdef public TimeSeries middles, deltas

    cpdef double calculate(self, long long time, Bar bar, short new_item_started)
