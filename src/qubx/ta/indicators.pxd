cimport numpy as np
from qubx.core.series cimport Indicator, IndicatorOHLC, RollingSum, TimeSeries, OHLCV, Bar, SeriesCachedValue

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
    cdef int ddof
    cdef int min_periods
    cdef object values_deque
    cdef int count
    cdef double _sum
    cdef double _sum_sq
    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Zscore(Indicator):
    cdef TimeSeries tr
    cdef Indicator ma, std
    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class BollingerBands(Indicator):
    cdef int period
    cdef double nstd
    cdef str smoother
    cdef TimeSeries tr
    cdef Indicator ma, std
    cdef public TimeSeries upper, lower
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

cdef class Pivots(IndicatorOHLC):
    cdef int before, after
    cdef object bars_buffer
    cdef Bar current_bar
    cdef long long current_bar_time
    cdef public TimeSeries tops, bottoms, tops_detection_lag, bottoms_detection_lag
    
    cpdef double calculate(self, long long time, Bar bar, short new_item_started)

cdef class PctChange(Indicator):
    cdef int period
    cdef object past_values
    cdef int _count
    cdef object stored_past_values
    cdef int stored_count

    cdef void _store(self)
    cdef void _restore(self)

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Rsi(Indicator):
    cdef int period
    cdef object up_moves
    cdef object down_moves
    cdef object smooth_up
    cdef object smooth_down
    cdef double prev_value

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class StdEma(Indicator):
    cdef int period
    cdef double alpha
    cdef int count
    cdef double ewm_mean_numer
    cdef double ewm_mean_denom
    cdef double ewm_var_numer
    cdef double ewm_var_denom
    cdef double prev_mean
    cdef int stored_count
    cdef double stored_ewm_mean_numer
    cdef double stored_ewm_mean_denom
    cdef double stored_ewm_var_numer
    cdef double stored_ewm_var_denom
    cdef double stored_prev_mean

    cdef void _store(self)
    cdef void _restore(self)

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class CusumFilter(Indicator):
    cdef double s_pos, s_neg
    cdef double prev_value
    cdef double last_value  # - Last value processed (used to update prev_value on new bar)
    cdef double saved_s_pos, saved_s_neg, saved_prev_value
    cdef double prev_bar_event  # - Event from previous completed bar
    cdef double current_bar_event  # - Event for current bar being calculated
    cdef SeriesCachedValue target_cache

    cdef void _store(self)

    cdef void _restore(self)

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class Macd(Indicator):
    cdef int fast_period
    cdef int slow_period
    cdef int signal_period
    cdef str method
    cdef str signal_method
    cdef object input_series
    cdef object fast_ma
    cdef object slow_ma
    cdef object macd_line_series
    cdef object signal_line

    cpdef double calculate(self, long long time, double value, short new_item_started)

cdef class SuperTrend(IndicatorOHLC):
    cdef int length
    cdef double mult
    cdef str src
    cdef short wicks
    cdef str atr_smoother

    cdef double _prev_longstop
    cdef double _prev_shortstop
    cdef double _prev_direction

    cdef double prev_longstop
    cdef double prev_shortstop
    cdef double prev_direction

    cdef TimeSeries tr
    cdef object atr_ma
    cdef public TimeSeries utl
    cdef public TimeSeries dtl

    cdef _store(self)
    cdef _restore(self)
    cdef double calc_src(self, Bar bar)

    cpdef double calculate(self, long long time, Bar bar, short new_item_started)