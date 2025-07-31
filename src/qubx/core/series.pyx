import pandas as pd
import numpy as np
cimport numpy as np
from cython cimport abs
from typing import Union
from qubx.core.utils import time_to_str, time_delta_to_str, recognize_timeframe
from qubx.utils.time import infer_series_frequency


cdef extern from "math.h":
    float INFINITY


cdef np.ndarray nans(int dims):
    """
    nans(n) is an n length array of NaNs.
    
    :param dims: array size
    :return: nans matrix 
    """
    return np.nan * np.ones(dims)


cdef inline long long floor_t64(long long time, long long dt):
    """
    Floor timestamp by dt
    """
    return time - time % dt


cpdef long long time_as_nsec(time):
    """
    Tries to recognize input time and convert it to nanosec
    """
    if isinstance(time, np.datetime64):
        return time.astype('<M8[ns]').item()
    elif isinstance(time, pd.Timestamp):
        return time.asm8
    elif isinstance(time, str):
        return np.datetime64(time).astype('<M8[ns]').item()
    return time


cdef class RollingSum:
    """
    Rolling fast summator
    """

    def __init__(self, int period):
        self.period = period
        self.__s = np.zeros(period)
        self.__i = 0
        self.rsum = 0.0
        self.is_init_stage = 1

    cpdef double update(self, double value, short new_item_started):
        if np.isnan(value):
            return np.nan
        sub = self.__s[self.__i]
        if new_item_started:
            self.__i += 1
            if self.__i >= self.period:
                self.__i = 0
                self.is_init_stage = 0
            sub = self.__s[self.__i]
        self.__s[self.__i] = value
        self.rsum -= sub
        self.rsum += value 
        return self.rsum

    def __str__(self):
        return f"rs[{self.period}] = {self.__s} @ {self.__i} -> {self.is_init_stage}"


cdef class Indexed:

    def __init__(self, max_series_length=INFINITY):
        self.max_series_length = max_series_length
        self.values = list()
        self._is_empty = 1

    def __len__(self) -> int:
        return len(self.values)

    def empty(self) -> bool:
        return self._is_empty

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.values[self._get_index(i)] for i in range(*idx.indices(len(self.values)))]
        return self.values[self._get_index(idx)]

    def _get_index(self, idx: int) -> int:
        n_len = len(self)
        if n_len == 0 or (idx > 0 and idx > (n_len - 1)) or (idx < 0 and abs(idx) > n_len):
            raise IndexError(f"Can't find record at index {idx}")
        return (n_len - idx - 1) if idx >= 0 else abs(1 + idx)

    def add(self, v):
        self.values.append(v)
        self._is_empty = 0
        if len(self.values) >= self.max_series_length:
            self.values.pop(0)

    def update_last(self, v):
        if self.values:
            self.values[-1] = v
        else:
            self.append(v)
        self._is_empty = 0

    def set_values(self, new_values: list):
        self._is_empty = False
        self.values = new_values

    def clear(self):
        self.values.clear()
        self._is_empty = 1

    def lookup_idx(self, value, str method) -> int:
        """
        Find value's index in series using specified method (ffill: previous index, bfill: next index)
        """
        cdef int i0
        if method == 'ffill':
            i0 = int(np.searchsorted(self.values, value, side='right'))
            return max(-1, i0 - 1)
        elif method == 'bfill':
            i0 = int(np.searchsorted(self.values, value, side='left'))
            return -1 if i0 >= len(self.values) else i0
        else:
            raise ValueError(f"Unsupported method {method}")


global _plot_func


cdef class Locator:
    """
    Locator service class for TimeSeries
    """

    def __init__(self, TimeSeries series):
        self._series = series 

    def __getitem__(self, idx):
        cdef int _nlen = len(self._series)
        cdef int _ix

        if isinstance(idx, slice):

            # - check slice has the same type or None
            if not ((type(idx.start) == type(idx.stop)) or idx.start is None or idx.stop is None):
                raise TypeError(f"Cannot do slice indexing with indexers of different types: [{idx.start} : {idx.stop}]")

            start_idx = 0 if idx.start is None else idx.start

            if isinstance(idx.start, str):
                # - even if start is not found we still want to start from first record
                start_idx = max(self._series.times.lookup_idx(np.datetime64(start_idx, 'ns').item(), 'ffill'), 0)

            if idx.stop is None:
                stop_idx = _nlen
            else: 
                if isinstance(idx.stop, str):
                    _ix = self._series.times.lookup_idx(np.datetime64(idx.stop, 'ns').item(), 'ffill')

                    if _ix < 0 or _ix < start_idx:
                        raise IndexError(f"Stop index {idx.stop} is not found or before start index {idx.start}")
                    
                    stop_idx = min(max(_ix, 0) + 1, _nlen)
                else:
                    stop_idx = min(idx.stop, _nlen)
            
            # print(f" >>>> LOC[{start_idx} : {stop_idx}] stop={stop_idx}")
            return self._series.copy(start_idx, stop_idx)

        elif isinstance(idx, str):
            # - handle single timestamp string
            return self.find(idx)

        return self._series.values[idx]

    def find(self, t: str):
        ix = self._series.times.lookup_idx(np.datetime64(t, 'ns').item(), 'ffill')
        return np.datetime64(self._series.times.values[ix], 'ns'), self._series.values.values[ix]


cdef class TimeSeries:

    def __init__(
        self, str name, timeframe, max_series_length=INFINITY, 
        process_every_update=True, # calculate indicators on every update (tick) - by default
    ) -> None:
        self.name = name
        self.max_series_length = max_series_length
        self.timeframe = recognize_timeframe(timeframe)
        self.times = Indexed(max_series_length)
        self.values = Indexed(max_series_length)
        self.indicators = dict()
        self.calculation_order = []
        # - locator service
        self.loc = Locator(self)

        # - processing every update
        self._process_every_update = process_every_update
        self._last_bar_update_value = np.nan
        self._last_bar_update_time = -1

    def __len__(self) -> int:
        return len(self.times)

    def _clone_empty(self, str name, long long timeframe, float max_series_length):
        """
        Make empty TimeSeries instance (no data and indicators)
        """
        return TimeSeries(name, timeframe, max_series_length)

    def copy(self, int start, int stop):
        ts_copy = self._clone_empty(self.name, self.timeframe, self.max_series_length)
        for i in range(start, stop):
            ts_copy._add_new_item(self.times.values[i], self.values.values[i])
        return ts_copy

    def clone(self):
        """
        Clone TimeSeries instance with data without indcators attached
        """
        return self.loc[:]

    def _on_attach_indicator(self, indicator: Indicator, indicator_input: TimeSeries):
        self.calculation_order.append((
            id(indicator_input), indicator, id(indicator)
        ))

    def __getitem__(self, idx):
        return self.values[idx]

    def _add_new_item(self, long long time, double value):
        self.times.add(time)
        self.values.add(value)
        self._is_new_item = True

    def _update_last_item(self, long long time, double value):
        self.times.update_last(time)
        self.values.update_last(value)
        self._is_new_item = False

    def update(self, long long time, double value) -> bool:
        item_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(item_start_time, value)

            # - disable first notification because first item may be incomplete
            self._is_new_item = False

        elif (_dt := time - self.times[0]) >= self.timeframe:
            # - add new item
            self._add_new_item(item_start_time, value)

            # - if it's needed to process every tick in indicator
            if self._process_every_update:
                self._update_indicators(item_start_time, value, True)
            else:
                # - it's required to update indicators only on closed (formed) bar
                self._update_indicators(self._last_bar_update_time, self._last_bar_update_value, True)

            # - store last data
            self._last_bar_update_time = item_start_time
            self._last_bar_update_value = value

            return self._is_new_item
        else:
            if _dt < 0:
                raise ValueError(f"{self.name}.{self.timeframe}: Attempt to update past data at {time_to_str(time)} !")
            self._update_last_item(item_start_time, value)

        # - update indicators by new data
        if self._process_every_update:
            self._update_indicators(item_start_time, value, False)

        # - store last data
        self._last_bar_update_time = item_start_time
        self._last_bar_update_value = value

        return self._is_new_item

    cdef _update_indicators(self, long long time, value, short new_item_started):
        mem = dict()              # store calculated values during this update
        mem[id(self)] = value     # initail value - new data from itself
        for input, indicator, iid in self.calculation_order:
            if input not in mem:
                raise ValueError("> No input data - something wrong in calculation order !")
            mem[iid] = indicator.update(time, mem[input], new_item_started)

    def shift(self, int period):
        """
        Returns shifted series by period
        """
        if period < 0:
            raise ValueError("Only positive shift (from past) period is allowed !")
        return lag(self, period)

    def __add__(self, other: Union[TimeSeries, float, int]):
        return plus(self, other)

    def __sub__(self, other: Union[TimeSeries, float, int]):
        return minus(self, other)

    def __mul__(self, other: Union[TimeSeries, float, int]):
        return mult(self, other)

    def __truediv__(self, other: Union[TimeSeries, float, int]):
        return divide(self, other)

    def __lt__(self, other: Union[TimeSeries, float, int]):
        return lt(self, other)

    def __le__(self, other: Union[TimeSeries, float, int]):
        return le(self, other)

    def __gt__(self, other: Union[TimeSeries, float, int]):
        return gt(self, other)

    def __ge__(self, other: Union[TimeSeries, float, int]):
        return ge(self, other)

    def __eq__(self, other: Union[TimeSeries, float, int]):
        return eq(self, other)

    def __ne__(self, other: Union[TimeSeries, float, int]):
        return ne(self, other)

    def __neg__(self):
        return neg(self)

    def __abs__(self):
        return series_abs(self)

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        return dict(zip(ts, self.values[::-1]))

    def to_series(self, length: int | None = None):
        if length is not None:
            # Efficiently extract only the last 'length' values
            total_length = len(self.values)
            if total_length == 0:
                return pd.Series([], index=pd.DatetimeIndex([]), name=self.name, dtype=float)
            
            start_idx = max(0, total_length - length)
            
            # Direct array slicing - no full series creation
            values_subset = self.values.values[start_idx:]
            times_subset = self.times.values[start_idx:]
            
            return pd.Series(
                values_subset, 
                index=pd.DatetimeIndex(times_subset), 
                name=self.name, 
                dtype=float
            )
        else:
            # Current behavior for backward compatibility
            return pd.Series(self.values.values, index=pd.DatetimeIndex(self.times.values), name=self.name, dtype=float)

    def pd(self, length: int | None = None):
        return self.to_series(length)

    def get_indicators(self) -> dict:
        return self.indicators

    def plot(self, *args, **kwargs):
        _timeseries_plot_func(self, *args, **kwargs)

    def __str__(self):
        nl = len(self)
        r = f"{self.name}[{time_delta_to_str(self.timeframe)}] | {nl} records\n"
        hd, tl = 3, 3 
        if nl <= hd + tl:
            hd, tl = nl, 0
        
        for n in range(hd):
            r += f"  {time_to_str(self.times[n], 'ns')} {str(self[n])}\n"
        
        if tl > 0:
            r += "   .......... \n"
            for n in range(-tl, 0):
                r += f"  {time_to_str(self.times[n], 'ns')} {str(self[n])}\n"

        return r

    def __repr__(self):
        return repr(self.pd())


def _wrap_indicator(series: TimeSeries, clz, *args, **kwargs):
    aw = ','.join([a.name if isinstance(a, TimeSeries) else str(a) for a in args])
    if kwargs:
        aw += ',' + ','.join([f"{k}={str(v)}" for k,v in kwargs.items()])
    nn = clz.__name__.lower() + "(" + aw + ")"
    inds = series.get_indicators()
    if nn in inds:
        return inds[nn]
    return clz(nn, series, *args, **kwargs) 


cdef class Indicator(TimeSeries):
    """
    Basic class for indicator that can be attached to TimeSeries
    """

    def __init__(self, str name, TimeSeries series):
        if not name:
            raise ValueError(f" > Name must not be empty for {self.__class__.__name__}!")
        super().__init__(name, series.timeframe, series.max_series_length)
        series.indicators[name] = self
        self.name = name

        # - initialize the initial recalculation flag
        self._is_initial_recalculate = 0

        # - we need to make a empty copy and fill it 
        self.series = self._clone_empty(series.name, series.timeframe, series.max_series_length)
        self.parent = series 
        
        # - notify the parent series that indicator has been attached
        self._on_attach_indicator(self, series)

        # - recalculate indicator on data as if it would being streamed
        self._is_initial_recalculate = 1
        self._initial_data_recalculate(series)
        self._is_initial_recalculate = 0

    def _on_attach_indicator(self, indicator: Indicator, indicator_input: TimeSeries):
        self.parent._on_attach_indicator(indicator, indicator_input)

    def _initial_data_recalculate(self, TimeSeries series):
        for t, v in zip(series.times[::-1], series.values[::-1]):
            self.update(t, v, True)

    def update(self, long long time, value, short new_item_started) -> object:
        if new_item_started or len(self) == 0:
            self.series._add_new_item(time, value)
            iv = self.calculate(time, value, new_item_started)
            self._add_new_item(time, iv)
        else:
            self.series._update_last_item(time, value)
            iv = self.calculate(time, value, new_item_started)
            self._update_last_item(time, iv)

        return iv

    def calculate(self, long long time, value, short new_item_started) -> object:
        raise ValueError("Indicator must implement calculate() method")

    @classmethod
    def wrap(clz, series:TimeSeries, *args, **kwargs):
        return _wrap_indicator(series, clz, *args, **kwargs)

    @property
    def is_initial_recalculate(self) -> bool:
        """
        Returns True if the indicator is currently in the initial data recalculation phase,
        False otherwise.
        """
        return self._is_initial_recalculate == 1


cdef class IndicatorOHLC(Indicator):
    """
    Extension of indicator class to be used for OHLCV series
    """
    def _clone_empty(self, str name, long long timeframe, float max_series_length):
        return OHLCV(name, timeframe, max_series_length)

    def _copy_internal_series(self, int start, int stop, *origins):
        """
        Helper method to copy internal series data
        """
        t0, t1 = self.times.values[start], self.times.values[stop - 1]
        return [
            o.loc[
                o.times.lookup_idx(t0, 'bfill') : o.times.lookup_idx(t1, 'ffill') + 1
            ] for o in origins
        ]

    def calculate(self, long long time, Bar value, short new_item_started) -> object:
        raise ValueError("Indicator must implement calculate() method")


cdef class Lag(Indicator):
    cdef int period

    def __init__(self, str name, TimeSeries series, int period):
        self.period = period
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if len(self.series) <= self.period:
            return np.nan
        return self.series[self.period]
     
     
def lag(series:TimeSeries, period: int):
    return Lag.wrap(series, period)


cdef class Abs(Indicator):

    def __init__(self, str name, TimeSeries series):
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        return abs(self.series[0])


def series_abs(series:TimeSeries):
    return Abs.wrap(series)


cdef class Compare(Indicator):
    cdef TimeSeries to_compare 
    cdef double comparable_scalar
    cdef short _cmp_to_series

    def __init__(self, name: str,  original: TimeSeries, comparable: Union[TimeSeries, float, int]):
        if isinstance(comparable, TimeSeries):
            if comparable.timeframe != original.timeframe:
                raise ValueError("Series must be of the same timeframe for performing operation !")
            self.to_compare = comparable
            self._cmp_to_series = 1
        else:
            self.comparable_scalar = comparable
            self._cmp_to_series = 0
        super().__init__(name, original)

    cdef double _operation(self, double a, double b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return +1 if a > b else -1 if a < b else 0

    def _initial_data_recalculate(self, TimeSeries series):
        if self._cmp_to_series:
            r = pd.concat((series.to_series(), self.to_compare.to_series()), axis=1)
            for t, (a, b) in zip(r.index, r.values):
                self.series._add_new_item(t.asm8, a)
                self._add_new_item(t.asm8, self._operation(a, b))
        else:
            r = series.to_series()
            for t, a in zip(r.index, r.values):
                self.series._add_new_item(t.asm8, a)
                self._add_new_item(t.asm8, self._operation(a, self.comparable_scalar))

    cpdef double calculate(self, long long time, double value, short new_item_started):
        if self._cmp_to_series:
            if len(self.to_compare) == 0 or len(self.series) == 0 or time != self.to_compare.times[0]:
                return np.nan
            return self._operation(value, self.to_compare[0])
        else:
            if len(self.series) == 0:
                return np.nan
            return self._operation(value, self.comparable_scalar)


def compare(series0:TimeSeries, series1:TimeSeries):
    return Compare.wrap(series0, series1)


cdef class Plus(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a + b


cdef class Minus(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a - b


cdef class Mult(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a * b


cdef class Divide(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a / b


cdef class EqualTo(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a == b


cdef class NotEqualTo(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a != b


cdef class LessThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a < b


cdef class LessEqualThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a <= b


cdef class GreaterThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a > b


cdef class GreaterEqualThan(Compare):

    def __init__(self, name: str, original:TimeSeries, comparable: Union[TimeSeries, float, int]):
        super().__init__(name, original, comparable)

    cdef double _operation(self, double a, double b):
        return a >= b


cdef class Neg(Indicator):

    def __init__(self, name: str, series:TimeSeries):
        super().__init__(name, series)

    cpdef double calculate(self, long long time, double value, short new_item_started):
        return -value


def plus(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Plus.wrap(series0, series1)


def minus(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Minus.wrap(series0, series1)


def mult(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Mult.wrap(series0, series1)


def divide(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return Divide.wrap(series0, series1)


def eq(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return EqualTo.wrap(series0, series1)
    

def ne(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return NotEqualTo.wrap(series0, series1)


def lt(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return LessThan.wrap(series0, series1)


def le(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return LessEqualThan.wrap(series0, series1)


def gt(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return GreaterThan.wrap(series0, series1)


def ge(series0:TimeSeries, series1:Union[TimeSeries, float, int]):
    return GreaterEqualThan.wrap(series0, series1)


def neg(series: TimeSeries):
    return Neg.wrap(series)


cdef class Trade:
    def __init__(self, time, double price, double size, short side=0, long long trade_id=0):
        self.time = time_as_nsec(time)
        self.price = price
        self.size = size
        self.side = side
        self.trade_id = trade_id

    def __repr__(self):
        return "[%s]\t%.5f (%.2f) %s %s" % ( 
            time_to_str(self.time, 'ns'), self.price, self.size, 
            'buy' if self.side == 1 else 'sell' if self.side == -1 else '???',
            str(self.trade_id) if self.trade_id > 0 else ''
        ) 


cdef class Quote:
    def __init__(self, time, double bid, double ask, double bid_size, double ask_size):
        self.time = time_as_nsec(time)
        self.bid = bid
        self.ask = ask
        self.bid_size = bid_size
        self.ask_size = ask_size

    cpdef double mid_price(self):
        return 0.5 * (self.ask + self.bid)

    def __repr__(self):
        return "[%s]\t%.5f (%.2f) | %.5f (%.2f)" % (
            time_to_str(self.time, 'ns'), self.bid, self.bid_size, self.ask, self.ask_size
        )


cdef class Bar:

    def __init__(
        self, long long time, double open, double high, double low, double close, 
        double volume, 
        double bought_volume=0.0, 
        double volume_quote=0.0, 
        double bought_volume_quote=0.0, 
        int trade_count=0
    ) -> None:
        self.time = time
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.bought_volume = bought_volume
        self.volume_quote = volume_quote
        self.bought_volume_quote = bought_volume_quote
        self.trade_count = trade_count

    cpdef Bar update(self, double price, double volume, double bought_volume=0, double volume_quote=0, double bought_volume_quote=0, int trade_count=0):
        self.close = price
        self.high = max(price, self.high)
        self.low = min(price, self.low)
        self.volume += volume
        self.bought_volume += bought_volume
        self.volume_quote += volume_quote
        self.bought_volume_quote += bought_volume_quote
        self.trade_count = trade_count  # Use latest trade count (cumulative)
        return self

    cpdef dict to_dict(self, unsigned short skip_time=0):
        if skip_time:
            return {
                'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close,
                'volume': self.volume, 'bought_volume': self.bought_volume, 
                'volume_quote': self.volume_quote, 'bought_volume_quote': self.bought_volume_quote,
                'trade_count': self.trade_count,
            }
        return {
            'timestamp': np.datetime64(self.time, 'ns'), 
            'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close, 
            'volume': self.volume,
            'bought_volume': self.bought_volume,
            'volume_quote': self.volume_quote,
            'bought_volume_quote': self.bought_volume_quote,
            'trade_count': self.trade_count,
        }

    def __repr__(self):
        return "[%s] {o:%f | h:%f | l:%f | c:%f | v:%f}" % (
            time_to_str(self.time, 'ns'), self.open, self.high, self.low, self.close, self.volume
        )


cdef class OrderBook:

    def __init__(self, long long time, top_bid: float, top_ask: float, tick_size: float, bids: np.ndarray, asks: np.ndarray):
        self.time = time
        self.top_bid = top_bid
        self.top_ask = top_ask
        self.tick_size = tick_size
        self.bids = bids
        self.asks = asks
    
    def __repr__(self):
        return f"[{time_to_str(self.time, 'ns')}] {self.top_bid} ({self.bids[0]}) | {self.top_ask} ({self.asks[0]})"
    
    cpdef Quote to_quote(self):
        return Quote(self.time, self.top_bid, self.top_ask, self.bids[0], self.asks[0])
    
    cpdef double mid_price(self):
        return 0.5 * (self.top_ask + self.top_bid)


cdef class TradeArray:
    """
    Array-based container for trades with efficient statistics tracking.
    """
    
    def __init__(self, data=None, int initial_capacity=1000):
        # Statistics fields
        self.time = 0                    # last trade time
        self.total_size = 0.0           # total traded volume
        self.buy_size = 0.0            # total buy volume
        self.sell_size = 0.0           # total sell volume
        self.min_buy_price = INFINITY   # minimum buy price
        self.max_buy_price = -INFINITY  # maximum buy price
        self.min_sell_price = INFINITY  # minimum sell price
        self.max_sell_price = -INFINITY # maximum sell price

        # Initialize from numpy array if provided
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("data must be a numpy array")
            
            expected_dtype = np.dtype([
                ('timestamp', 'i8'),      # timestamp in nanoseconds
                ('price', 'f8'),     # trade price
                ('size', 'f8'),      # trade size
                ('side', 'i1'),      # trade side (1: buy, -1: sell)
            ])
            
            if data.dtype != expected_dtype:
                # Try to convert the input array to our dtype
                try:
                    data = data.astype(expected_dtype)
                except:
                    raise ValueError(f"Cannot convert input array to required dtype: {expected_dtype}")
            
            self.trades = data
            self.size = len(data)
            self._capacity = len(data)
            
            # Calculate initial statistics using optimized C method
            self._calculate_statistics(0, self.size)
            
        else:
            # Create new array only if no data provided
            self.trades = np.zeros(initial_capacity, dtype=[
                ('timestamp', 'i8'),      # timestamp in nanoseconds
                ('price', 'f8'),     # trade price
                ('size', 'f8'),      # trade size
                ('side', 'i1'),      # trade side (1: buy, -1: sell)
            ])
            self.size = 0
            self._capacity = initial_capacity

    cdef void _calculate_statistics(self, int start_idx, int end_idx):
        """
        Calculate statistics for trades in range [start_idx, end_idx)
        Using pure C types for maximum performance
        """
        cdef int i
        cdef double price, size
        cdef char side
        cdef long long t
        
        # Reset statistics if starting from beginning
        if start_idx == 0:
            self.total_size = 0.0
            self.buy_size = 0.0
            self.sell_size = 0.0
            self.min_buy_price = INFINITY
            self.max_buy_price = -INFINITY
            self.min_sell_price = INFINITY
            self.max_sell_price = -INFINITY
            self.time = 0
        
        for i in range(start_idx, end_idx):
            t = self.trades[i]['timestamp']
            price = self.trades[i]['price']
            size = self.trades[i]['size']
            side = self.trades[i]['side']
            
            self.total_size += size
            
            if side > 0:  # Buy trade
                self.buy_size += size
                if price < self.min_buy_price:
                    self.min_buy_price = price
                if price > self.max_buy_price:
                    self.max_buy_price = price
            else:  # Sell trade
                self.sell_size += size
                if price < self.min_sell_price:
                    self.min_sell_price = price
                if price > self.max_sell_price:
                    self.max_sell_price = price
        
        if end_idx > start_idx:
            self.time = self.trades[end_idx - 1]['timestamp']

    cpdef tuple traded_range_from(self, long long time):
        """
        Calculate min and max prices for trades from given time
        Returns tuple of (min_buy_price, max_buy_price, min_sell_price, max_sell_price)
        """
        cdef int i
        cdef char side
        cdef float price
        cdef float min_buy_price = INFINITY
        cdef float max_buy_price = -INFINITY
        cdef float min_sell_price = INFINITY
        cdef float max_sell_price = -INFINITY

        # - speedup if time is before first trade
        if time <= self.trades[0]['timestamp']:
            return self.min_buy_price, self.max_buy_price, self.min_sell_price, self.max_sell_price

        if time <= self.trades[self.size - 1]['timestamp']:
            for i in range(0, self.size):
                t = self.trades[i]['timestamp']
                if t < time:
                    continue

                price = self.trades[i]['price']
                side = self.trades[i]['side']
                if side > 0:  # buy trade
                    min_buy_price = min(min_buy_price, price)
                    max_buy_price = max(max_buy_price, price)
                else:  # sell trade
                    min_sell_price = min(min_sell_price, price)
                    max_sell_price = max(max_sell_price, price)

        return min_buy_price, max_buy_price, min_sell_price, max_sell_price

    cdef void _ensure_capacity(self, int required_size):
        if required_size >= self._capacity:
            new_capacity = max(self._capacity * 2, required_size + 1)
            new_trades = np.zeros(new_capacity, dtype=self.trades.dtype)
            new_trades[:self.size] = self.trades[:self.size]
            self.trades = new_trades
            self._capacity = new_capacity
    
    cpdef void add(self, long long time, double price, double size, short side):
        self._ensure_capacity(self.size + 1)
        
        # Add trade to array
        self.trades[self.size] = (time, price, size, side)
        self.size += 1
        
        # Update statistics using the optimized method for single trade
        self._calculate_statistics(self.size - 1, self.size)

    cpdef void clear(self):
        """Reset the trade array and all statistics"""
        self.size = 0
        self.time = 0
        self.total_size = 0.0
        self.buy_size = 0.0
        self.sell_size = 0.0
        self.min_buy_price = INFINITY
        self.max_buy_price = -INFINITY
        self.min_sell_price = INFINITY
        self.max_sell_price = -INFINITY
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """Get a single trade by index. For array/slice access, use the trades attribute directly."""
        if isinstance(idx, slice):
            raise TypeError("Slice access not supported. Use trades attribute for array access.")
        if idx < 0:
            idx = self.size + idx
        if idx >= self.size:
            raise IndexError("Trade index out of range")
        # Convert numpy record to Trade object
        record = self.trades[idx]
        return Trade(record['timestamp'], record['price'], record['size'], record['side'])
    
    def __repr__(self):
        _s = time_to_str(self.trades[0][0]) if len(self.trades) > 0 else ''
        _e = time_to_str(self.trades[self.size - 1][0]) if len(self.trades) > 0 else ''
        return f"TradeArray({_s} - {_e}, size={self.size}, volume={self.total_size:.1f}, buys={self.buy_size:.1f}, sells={self.sell_size:.1f})"

    
cdef long long _bar_time_key(Bar bar):
    return bar.time


cdef class OHLCV(TimeSeries):

    def __init__(self, str name, timeframe, max_series_length=INFINITY) -> None:
        super().__init__(name, timeframe, max_series_length)
        self.open = TimeSeries('open', timeframe, max_series_length)
        self.high = TimeSeries('high', timeframe, max_series_length)
        self.low = TimeSeries('low', timeframe, max_series_length)
        self.close = TimeSeries('close', timeframe, max_series_length)
        self.volume = TimeSeries('volume', timeframe, max_series_length)
        self.bvolume = TimeSeries('bvolume', timeframe, max_series_length)
        self.volume_quote = TimeSeries('volume_quote', timeframe, max_series_length)
        self.bvolume_quote = TimeSeries('bvolume_quote', timeframe, max_series_length)
        self.trade_count = TimeSeries('trade_count', timeframe, max_series_length)

    cpdef object append_data(self, 
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
                ):
        cdef long long t
        cdef short _conv
        cdef short _upd_inds, _has_vol, _has_vol_quote, _has_bvol_quote, _has_trade_count
        cdef Bar b 

        # - check if volume data presented
        _has_vol = len(volumes) > 0
        _has_bvol = len(bvolumes) > 0
        _has_vol_quote = len(volume_quotes) > 0
        _has_bvol_quote = len(bought_volume_quotes) > 0
        _has_trade_count = len(trade_counts) > 0

        # - check if need to convert time to nanosec
        _conv = 0
        if not isinstance(times[0].item(), long):
            _conv = 1

        # - check if need to update any indicators
        _upd_inds = 0
        if (
            len(self.indicators) > 0 or 
            len(self.open.indicators) > 0 or 
            len(self.high.indicators) > 0 or
            len(self.low.indicators) > 0 or 
            len(self.close.indicators) > 0 or
            len(self.volume.indicators) > 0
        ):
            _upd_inds = 1

        for i in range(len(times)):
            if _conv:
                t = times[i].astype('datetime64[ns]').item()
            else:
                t = times[i].item()

            b = Bar(
                t, opens[i], highs[i], lows[i], closes[i], 
                volume=volumes[i] if _has_vol else 0, 
                bought_volume=bvolumes[i] if _has_bvol else 0,
                volume_quote=volume_quotes[i] if _has_vol_quote else 0,
                bought_volume_quote=bought_volume_quotes[i] if _has_bvol_quote else 0,
                trade_count=trade_counts[i] if _has_trade_count else 0
            )
            self._add_new_item(t, b)

            if _upd_inds:
                self._update_indicators(t, b, True)

        return self

    def _clone_empty(self, str name, long long timeframe, float max_series_length):
        return OHLCV(name, timeframe, max_series_length)

    def _add_new_item(self, long long time, Bar value):
        self.times.add(time)
        self.values.add(value)
        self.open._add_new_item(time, value.open)
        self.high._add_new_item(time, value.high)
        self.low._add_new_item(time, value.low)
        self.close._add_new_item(time, value.close)
        self.volume._add_new_item(time, value.volume)
        self.bvolume._add_new_item(time, value.bought_volume)
        self.volume_quote._add_new_item(time, value.volume_quote)
        self.bvolume_quote._add_new_item(time, value.bought_volume_quote)
        self.trade_count._add_new_item(time, value.trade_count)
        self._is_new_item = True

    def _update_last_item(self, long long time, Bar value):
        self.times.update_last(time)
        self.values.update_last(value)
        self.open._update_last_item(time, value.open)
        self.high._update_last_item(time, value.high)
        self.low._update_last_item(time, value.low)
        self.close._update_last_item(time, value.close)
        self.volume._update_last_item(time, value.volume)
        self.bvolume._update_last_item(time, value.bought_volume)
        self.volume_quote._update_last_item(time, value.volume_quote)
        self.bvolume_quote._update_last_item(time, value.bought_volume_quote)
        self.trade_count._update_last_item(time, value.trade_count)
        self._is_new_item = False

    cpdef short update(self, long long time, double price, double volume=0.0, double bvolume=0.0):
        cdef Bar b
        bar_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(bar_start_time, Bar(bar_start_time, price, price, price, price, volume=volume, bought_volume=bvolume))

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False

        elif (_dt := time - self.times[0]) >= self.timeframe:
            b = Bar(bar_start_time, price, price, price, price, volume=volume, bought_volume=bvolume)

            # - add new item
            self._add_new_item(bar_start_time, b)

            # - update indicators
            self._update_indicators(bar_start_time, b, True)

            return self._is_new_item
        else:
            if _dt < 0:
                raise ValueError(f"Attempt to update past data at {time_to_str(time)} !")

            self._update_last_item(bar_start_time, self[0].update(price, volume, bvolume))

        # - update indicators by new data
        self._update_indicators(bar_start_time, self[0], False)

        return self._is_new_item

    cpdef short update_by_bar(self, long long time, double open, double high, double low, double close, double vol_incr=0.0, double b_vol_incr=0.0, double volume_quote=0.0, double bought_volume_quote=0.0, int trade_count=0, short is_incremental=1):
        cdef Bar b
        cdef Bar l_bar
        bar_start_time = floor_t64(time, self.timeframe)

        if not self.times:
            self._add_new_item(bar_start_time, Bar(bar_start_time, open, high, low, close, volume=vol_incr, bought_volume=b_vol_incr, volume_quote=volume_quote, bought_volume_quote=bought_volume_quote, trade_count=trade_count))

            # Here we disable first notification because first item may be incomplete
            self._is_new_item = False

        elif time - self.times[0] >= self.timeframe:
            b = Bar(bar_start_time, open, high, low, close, volume=vol_incr, bought_volume=b_vol_incr, volume_quote=volume_quote, bought_volume_quote=bought_volume_quote, trade_count=trade_count)

            # - add new item
            self._add_new_item(bar_start_time, b)

            # - update indicators
            self._update_indicators(bar_start_time, b, True)

            return self._is_new_item
        else:
            l_bar = self[0]
            l_bar.high = max(high, l_bar.high)
            l_bar.low = min(low, l_bar.low)
            l_bar.close = close
            if is_incremental:
                l_bar.volume += vol_incr
                l_bar.volume_quote += volume_quote
                l_bar.bought_volume += b_vol_incr
                l_bar.bought_volume_quote += bought_volume_quote
                l_bar.trade_count += trade_count
            else:
                l_bar.volume = vol_incr
                l_bar.volume_quote = volume_quote
                l_bar.bought_volume = b_vol_incr
                l_bar.bought_volume_quote = bought_volume_quote
                l_bar.trade_count = trade_count

            self._update_last_item(bar_start_time, l_bar)

        # # - update indicators by new data
        self._update_indicators(bar_start_time, self[0], False)

        return self._is_new_item
    
    cpdef object update_by_bars(self, list bars):
        """
        Update the OHLCV series with a list of bars, handling both new bars and updates to existing bars.
        
        This method efficiently handles historical data by:
        1. For non-historical data: simply using update_by_bar for each bar
        2. For historical data:
           a. Separating bars into historical (before newest existing) and future (after newest existing)
           b. Creating a new temporary series with historical + existing bars
           c. Replacing the original series buffers with the temporary series buffers
           d. Updating the original series with future bars to ensure indicators are updated
        
        This approach avoids the need to clone indicators and recalculate them from scratch.
        
        Args:
            bars: List of Bar objects to add or update
        
        Returns:
            self: Returns self for method chaining
        """
        if not bars:
            return self
        
        # Sort bars by time (oldest first)
        cdef list new_bars = sorted(bars, key=_bar_time_key)
        cdef Bar bar
        
        # If no new bars, return early
        if not new_bars:
            return self
        
        if len(self.times) == 0:
            for bar in new_bars:
                self.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.bought_volume, bar.volume_quote, bar.bought_volume_quote, bar.trade_count)
            return self
        
        # Check if we have historical bars (bars older than our newest data)
        cdef bint has_historical_bars = False

        cdef long long newest_time = self.times[0]
        cdef long long oldest_time = self.times[-1]

        for bar in new_bars:
            if bar.time < newest_time:
                has_historical_bars = True
                break
        
        # If we don't have historical bars, use the standard update method for efficiency
        if not has_historical_bars:
            for bar in new_bars:
                self.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.bought_volume, bar.volume_quote, bar.bought_volume_quote, bar.trade_count)
            return self
        
        # We have historical bars, so we need a more complex approach
        
        # 1. Separate historical bars from future bars
        cdef list historical_bars = []
        cdef list future_bars = []
        
        for bar in new_bars:
            if len(self.times) == 0 or bar.time < oldest_time:
                historical_bars.append(bar)
            elif bar.time >= newest_time:
                future_bars.append(bar)
        
        # 2. Create a new temporary series
        cdef OHLCV temp_series = OHLCV(self.name, self.timeframe, self.max_series_length)
        
        # 3. Add historical bars to the temporary series
        for bar in historical_bars:
            temp_series.update_by_bar(
                bar.time, bar.open, bar.high, bar.low, bar.close, 
                bar.volume, 
                bar.bought_volume,
                bar.volume_quote,
                bar.bought_volume_quote,
                bar.trade_count
            )
        
        # 4. Add existing bars to the temporary series
        df = self.to_series()
        temp_series.append_data(
            df.index.values,
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values,
            df['bought_volume'].values,
            df['volume_quote'].values,
            df['bought_volume_quote'].values,
            df['trade_count'].values
        )
        
        # 5. Replace the original series buffers with the temporary series buffers
        for field in [
            self, self.open, self.high, self.low, self.close, 
            self.volume, self.bvolume, self.volume_quote, self.bvolume_quote, self.trade_count
        ]:
            field.times.clear()
            field.values.clear()
        
        # Set the new data using a loop for all fields
        for field, temp_field in [
            (self, temp_series),
            (self.open, temp_series.open),
            (self.high, temp_series.high),
            (self.low, temp_series.low),
            (self.close, temp_series.close),
            (self.volume, temp_series.volume),
            (self.bvolume, temp_series.bvolume),
            (self.volume_quote, temp_series.volume_quote),
            (self.bvolume_quote, temp_series.bvolume_quote),
            (self.trade_count, temp_series.trade_count),
        ]:
            field.times.set_values(temp_field.times.values)
            field.values.set_values(temp_field.values.values)
        
        # 6. Update with future bars to ensure indicators are updated
        for bar in future_bars:
            self.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.bought_volume, bar.volume_quote, bar.bought_volume_quote, bar.trade_count)
        
        return self

    # - TODO: need to check if it's safe to drop value series (series of Bar) to avoid duplicating data
    # def __getitem__(self, idx):
    #     if isinstance(idx, slice):
    #         return [
    #             Bar(self.times[i], self.open[i], self.high[i], self.low[i], self.close[i], self.volume[i])
    #         for i in range(*idx.indices(len(self.times)))
    #     ]
    #     return Bar(self.times[idx], self.open[idx], self.high[idx], self.low[idx], self.close[idx], self.volume[idx])

    cpdef _update_indicators(self, long long time, value, short new_item_started):
        TimeSeries._update_indicators(self, time, value, new_item_started)
        if new_item_started:
            self.open._update_indicators(time, value.open, new_item_started)
        self.close._update_indicators(time, value.close, new_item_started)
        self.high._update_indicators(time, value.high, new_item_started)
        self.low._update_indicators(time, value.low, new_item_started)
        self.volume._update_indicators(time, value.volume, new_item_started)
        self.bvolume._update_indicators(time, value.bought_volume, new_item_started)
        self.volume_quote._update_indicators(time, value.volume_quote, new_item_started)
        self.bvolume_quote._update_indicators(time, value.bought_volume_quote, new_item_started)
        self.trade_count._update_indicators(time, value.trade_count, new_item_started)

    def to_series(self, length: int | None = None) -> pd.DataFrame:
        df = pd.DataFrame({
            'open': self.open.to_series(length),                         # Each handles its own slicing
            'high': self.high.to_series(length),
            'low': self.low.to_series(length),
            'close': self.close.to_series(length),
            'volume': self.volume.to_series(length),                     # total volume
            'bought_volume': self.bvolume.to_series(length),             # bought volume
            'volume_quote': self.volume_quote.to_series(length),         # quote asset volume
            'bought_volume_quote': self.bvolume_quote.to_series(length), # bought quote volume
            'trade_count': self.trade_count.to_series(length),           # number of trades
        })
        df.index.name = 'timestamp'
        return df

    def pd(self, length: int | None = None) -> pd.DataFrame:
        return self.to_series(length)

    @staticmethod
    def from_dataframe(object df_p, str name="ohlc"):
        if not isinstance(df_p, pd.DataFrame):
            ValueError(f"Input must be a pandas DataFrame, got {type(df_p).__name__}")

        _ohlc = OHLCV(name, infer_series_frequency(df_p).item())
        for t in df_p.itertuples():
            _ohlc.update_by_bar(
                t.Index.asm8, t.open, t.high, t.low, t.close, 
                getattr(t, "volume", 0.0), 
                getattr(t, "taker_buy_volume", 0.0), 
                getattr(t, "quote_volume", 0.0), 
                getattr(t, "taker_buy_quote_volume", 0.0), 
                getattr(t, "count", 0.0)
            )
        return _ohlc

    def to_records(self) -> dict:
        ts = [np.datetime64(t, 'ns') for t in self.times[::-1]]
        bs = [v.to_dict(skip_time=True) for v in self.values[::-1]]
        return dict(zip(ts, bs))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - this should be done in separate module -
def _plot_mpl(series: TimeSeries, *args, **kwargs):
    import matplotlib.pyplot as plt
    include_indicators = kwargs.pop('with_indicators', False)
    no_labels = kwargs.pop('no_labels', False)

    plt.plot(series.pd(), *args, **kwargs, label=series.name)
    if include_indicators:
        for k, vi in series.get_indicators().items():
            plt.plot(vi.pd(), label=k)
    if not no_labels:
        plt.legend(loc=2)

_timeseries_plot_func = _plot_mpl
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 