from typing import Any, Tuple, Union

import numpy as np
cimport numpy as np

import pandas as pd

class Bar:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    bought_volume: float
    def __init__(self, time, open, high, low, close, volume, bought_volume=0): ...
    def update(self, price: float, volume: float, bought_volume: float = 0) -> "Bar": ...
    def to_dict(self, skip_time: bool = False) -> dict: ...

class Quote:
    time: int
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __init__(self, time, bid, ask, bid_size, ask_size): ...
    def mid_price(self) -> float: ...

class Trade:
    time: int
    price: float
    size: float
    taker: int
    trade_id: int
    def __init__(self, time, price, size, taker=-1, trade_id=0): ...

class OrderBook:
    time: int
    top_bid: float
    top_ask: float
    tick_size: float
    bids: np.ndarray
    asks: np.ndarray

    def __init__(self, time, top_bid, top_ask, tick_size, bids, asks): ...
    def to_quote(self) -> Quote: ...
    def mid_price(self) -> float: ...

class Locator:
    def __getitem__(self, idx): ...
    def find(self, t: str) -> Tuple[np.datetime64, Any]: ...

class Indexed:
    def __getitem__(self, idx): ...
    def lookup_idx(self, value, method: str) -> int: ...

class TimeSeries:
    name: str
    loc: Locator
    timeframe: int
    max_series_length: int
    times: Indexed
    values: Indexed
    def __init__(self, name, timeframe, max_series_length=np.inf, process_every_update=True) -> None: ...
    def __getitem__(self, idx): ...
    def __len__(self) -> int: ...
    def update(self, time: int, value: float) -> bool: ...
    def copy(self, start: int, stop: int) -> "TimeSeries": ...
    def shift(self, period: int) -> TimeSeries: ...
    def get_indicators(self) -> dict: ...
    def plot(self, *args, **kwargs): ...
    def to_records(self) -> dict: ...
    def to_series(self) -> pd.Series: ...
    def pd(self) -> pd.Series: ...

class OHLCV(TimeSeries):
    open: TimeSeries
    high: TimeSeries
    low: TimeSeries
    close: TimeSeries
    volume: TimeSeries
    bvolume: TimeSeries

    def __init__(self, name, timeframe, max_series_length: int | float = np.inf) -> None: ...
    def __len__(self) -> int: ...
    def update(self, time: int, price: float, volume: float = 0.0, bvolume: float = 0.0) -> bool: ...
    def update_by_bar(
        self,
        time: int,
        open: float,
        high: float,
        low: float,
        close: float,
        vol_incr: float = 0.0,
        b_vol_incr: float = 0.0,
    ) -> bool: ...
    def to_records(self) -> dict: ...
    def pd(self) -> pd.DataFrame: ...
    @staticmethod
    def from_dataframe(df_p: pd.DataFrame, name: str = "ohlc") -> "OHLCV": ...

class Indicator(TimeSeries):
    name: str
    series: TimeSeries

    def update(self, time: int, value, new_item_started: bool) -> object: ...
    @classmethod
    def wrap(cls, series: TimeSeries, *args, **kwargs) -> "Indicator": ...

class IndicatorOHLC(Indicator):
    series: OHLCV
    def _copy_internal_series(self, start: int, stop: int, *origins): ...

def time_as_nsec(time: Any) -> int: ...

class RollingSum:
    is_init_stage: bool
    def __init__(self, period: int) -> None: ...
    def update(self, value: float, new_item_started: bool) -> float: ...

def compare(series0: TimeSeries, series1: TimeSeries) -> "Indicator":
    """
    Compare two time series and return an indicator that shows the relationship between them.
    Returns +1 when series0 > series1, -1 when series0 < series1, and 0 when equal.
    Both series must have the same timeframe.
    """
    ...

def lag(series: TimeSeries, period: int) -> "Indicator":
    """
    Returns a new time series shifted by the specified period.
    Only positive shift (from past) period is allowed.
    """
    ...
