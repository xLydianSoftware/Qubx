import numpy as np
cimport numpy as np


cdef np.ndarray nans(int dims)


cdef class Indexed:
    cdef public list values
    cdef public float max_series_length
    cdef unsigned short _is_empty


cdef class Locator:
    cdef TimeSeries _series


cdef class TimeSeries:
    cdef public long long timeframe
    cdef public Indexed times
    cdef public Indexed values
    cdef public float max_series_length
    cdef public Locator loc
    cdef unsigned short _is_new_item
    cdef public str name
    cdef dict indicators        # it's used for indicators caching
    cdef list calculation_order # calculation order as list: [ (input_id, indicator_obj, indicator_id) ]
    cdef double _process_every_update
    cdef double _last_bar_update_value
    cdef long long _last_bar_update_time

    cdef _update_indicators(TimeSeries self, long long time, object value, short new_item_started)


cdef class Indicator(TimeSeries):
    cdef public TimeSeries series
    cdef public TimeSeries parent
    cdef unsigned short _is_initial_recalculate


cdef class RollingSum:
    """
    Rolling fast summator
    """
    cdef unsigned int period
    cdef np.ndarray __s
    cdef unsigned int __i
    cdef double rsum
    cdef public unsigned short is_init_stage 

    cpdef double update(RollingSum self, double value, short new_item_started)


cdef class Bar:
    cdef public long long time
    cdef public double open
    cdef public double high
    cdef public double low
    cdef public double close
    cdef public double volume               # total volume (in base asset)
    cdef public double bought_volume        # volume bought (in base asset) if presented
    cdef public double volume_quote         # total volume (in quote asset)
    cdef public double bought_volume_quote  # volume bought (in quote asset) if presented
    cdef public int trade_count             # number of trades in this bar

    cpdef Bar update(Bar self, double price, double volume, double volume_quote=*, double bought_volume=*, double bought_volume_quote=*, int trade_count=*)

    cpdef dict to_dict(Bar self, unsigned short skip_time=*)


cdef class OHLCV(TimeSeries):
    cdef public TimeSeries open
    cdef public TimeSeries high
    cdef public TimeSeries low
    cdef public TimeSeries close
    cdef public TimeSeries volume
    cdef public TimeSeries bvolume
    cdef public TimeSeries volume_quote
    cdef public TimeSeries bvolume_quote
    cdef public TimeSeries trade_count

    cpdef short update(OHLCV self, long long time, double price, double volume=*, double bvolume=*)

    cpdef short update_by_bar(
        OHLCV self, long long time, double open, double high, double low, double close, 
        double vol_incr=*, 
        double b_vol_incr=*, 
        double volume_quote=*, 
        double bought_volume_quote=*, 
        int trade_count=*, 
        short is_incremental=*
    )

    cpdef object update_by_bars(OHLCV self, list bars)

    cpdef _update_indicators(OHLCV self, long long time, object value, short new_item_started)

    cpdef object append_data(
        OHLCV self, 
        np.ndarray times, 
        np.ndarray opens,
        np.ndarray highs, 
        np.ndarray lows,
        np.ndarray closes, 
        np.ndarray volumes,
        np.ndarray bvolumes,
        np.ndarray volume_quotes,
        np.ndarray bought_volume_quotes,
        np.ndarray trade_counts
    )


cdef class Trade:
    cdef public long long time
    cdef public double price
    cdef public double size
    cdef public short side
    cdef public long long trade_id


cdef class Quote:
    cdef public long long time
    cdef public double bid
    cdef public double ask
    cdef public double bid_size
    cdef public double ask_size

    cpdef double mid_price(Quote self)


cdef class OrderBook:
    cdef public long long time
    cdef public double top_bid
    cdef public double top_ask
    cdef public double tick_size
    cdef public np.ndarray bids
    cdef public np.ndarray asks

    cpdef Quote to_quote(OrderBook self)
    cpdef double mid_price(OrderBook self)


cdef class TradeArray:
    cdef public:
        np.ndarray trades
        int size
        long long time
        double total_size
        double buy_size
        double sell_size
        double min_buy_price
        double max_buy_price
        double min_sell_price
        double max_sell_price
    
    cdef int _capacity
    
    cdef void _calculate_statistics(self, int start_idx, int end_idx)
    cdef void _ensure_capacity(self, int required_size)
    cpdef void add(self, long long time, double price, double size, short side)
    cpdef void clear(self)
    cpdef tuple traded_range_from(self, long long time)

cdef class IndicatorOHLC(Indicator):
    pass

