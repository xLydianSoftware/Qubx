from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided as stride

from qubx.core.series import OHLCV
from qubx.utils.misc import Struct


def has_columns(x, *args):
    return isinstance(x, pd.DataFrame) and sum(x.columns.isin(args)) == len(args)


def check_frame_columns(x, *args):
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Input data must be DataFrame but {type(x)} received !")

    if sum(x.columns.isin(args)) != len(args):
        required = [y for y in args if y not in x.columns]
        raise ValueError(f"> Required {required} columns not found in dataframe !")


def rolling_forward_test_split(
    x: pd.Series | pd.DataFrame,
    training_period: int,
    test_period: int,
    units: str | None = None,
):
    """
    Split data into training and testing **rolling** periods.

    Example:

    >>> for train_idx, test_idx in rolling_forward_test_split(np.array(range(15)), 5, 3):
    >>>     print('Train:', train_idx, ' Test:', test_idx)

    > Train: [1 2 3 4 5]  Test: [6 7 8]
      Train: [4 5 6 7 8]  Test: [9 10 11]
      Train: [7 8 9 10 11]  Test: [12 13 14]

    Also it allows splitting using calendar periods (see units for that).
    Example of 2w / 1w splitting:

    >>> Y = pd.Series(np.arange(30), index=pd.date_range('2000-01-01', periods=30))
    >>> for train_idx, test_idx in rolling_forward_test_split(Y, 2, 1, units='W'):
    >>>     print('Train:', Y.loc[train_idx], '\\n Test:', Y.loc[test_idx])

    :param x: data
    :param training_period: number observations for learning period
    :param test_period: number observations for learning period
    :param units: period units if training_period and test_period is the period date: {'H', 'D', 'W', 'M', 'Q', 'Y'}
    :return:
    """
    # unit formats from pd.TimeDelta and formats for pd.resample
    units_format = {"H": "h", "D": "d", "W": "W", "M": "MS", "Q": "QS", "Y": "AS", "MIN": "min"}

    if units:
        if units.upper() not in units_format:
            raise ValueError(
                'Wrong value for "units" parameter. Only %s values are valid' % ",".join(units_format.keys())
            )
        else:
            if not isinstance(x, (pd.Series, pd.DataFrame)) or not isinstance(x.index, pd.DatetimeIndex):
                raise ValueError('Data must be passed as pd.DataFrame or pd.Series when "units" specified')

            if isinstance(x, pd.Series):
                x = x.to_frame()

            resampled = x.resample(units_format[units.upper()]).mean().index
            resampled = resampled - pd.DateOffset(seconds=1)

            for i in range(0, len(resampled), test_period):
                if len(resampled) - 1 < i + training_period or resampled[i + training_period] > x.index[-1]:
                    # no data for next training period
                    break
                training_df = x[resampled[i] : resampled[i + training_period]]
                whole_period = i + training_period + test_period
                if len(resampled) - 1 < whole_period or resampled[whole_period] > x.index[-1]:
                    # if there is not all data for test period or it's just last month,
                    # we don't need restrict the end date
                    test_df = x[resampled[i + training_period] :]
                else:
                    test_df = x[resampled[i + training_period] : resampled[whole_period]]

                if training_df.empty or test_df.empty:
                    continue
                yield (np.array(training_df.index), np.array(test_df.index))
    else:
        n_obs = x.shape[0]
        i_shift = (n_obs - training_period - test_period) % test_period
        for i in range(i_shift + training_period, n_obs, test_period):
            yield (np.array(range(i - training_period, i)), np.array(range(i, i + test_period)))


def generate_equal_date_ranges(start: str | pd.Timestamp, end: str | pd.Timestamp, freq, units):
    """
    Generator for date ranges:

    for s,e in generate_ranges('2019-01-01', '2022-05-17', 1, 'Y'):
        print(s, e)
    ------------------
    2019-01-01 2019-12-31
    2020-01-01 2020-12-31
    2021-01-01 2021-12-31
    2022-01-01 2022-05-17
    """
    _as_f = lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
    if units in ["M", "Y"]:
        offset = pd.DateOffset(years=freq) if units == "Y" else pd.DateOffset(months=freq)
    else:
        offset = pd.Timedelta(f"{freq}{units}")

    if pd.Timestamp(end) - offset < pd.Timestamp(start):
        b = [start, end]

    for a, b in rolling_forward_test_split(pd.Series(0, pd.date_range(start, end)), freq, freq, units=units):
        yield _as_f(a[0]), _as_f(a[-1])

    yield _as_f(b[0]), _as_f(b[-1])


def drop_duplicated_indexes(df: pd.DataFrame, keep: Literal["first", "last", False] = "first"):
    """
    Drops duplicated indexes in dataframe/series
    Keeps either first or last occurence (parameter keep)
    """
    return df[~df.index.duplicated(keep=keep)]


def process_duplicated_indexes(data: pd.DataFrame | pd.Series, ns=1) -> pd.DataFrame | pd.Series:
    """
    Finds duplicated indexes in frame/series and add shift (in nS) to every repeating one
    :param data: time indexed dataframe/series
    :param ns: shift constant in nanosec
    :return: return dataframe with all no duplicated rows (each duplicate has own unique index)
    """
    values = data.index.duplicated(keep="first").astype(float)
    values[values == 0] = np.NaN

    missings = np.isnan(values)
    cumsum = np.cumsum(~missings)
    diff = np.diff(np.concatenate(([0.0], cumsum[missings])))
    values[missings] = -diff

    # set new index (1 ms)
    data.index = data.index.values + np.cumsum(values).astype(np.timedelta64) * ns
    return data


def scols(*xs, keys=None, names=None, keep="all") -> pd.DataFrame:
    """
    Concat dataframes/series from xs into single dataframe by axis 1
    :param keys: keys of new dataframe (see pd.concat's keys parameter)
    :param names: new column names or dict with replacements
    :return: combined dataframe

    Example
    -------
    >>>  scols(
            pd.DataFrame([1,2,3,4,-4], list('abcud')),
            pd.DataFrame([111,21,31,14], list('xyzu')),
            pd.DataFrame([11,21,31,124], list('ertu')),
            pd.DataFrame([11,21,31,14], list('WERT')),
            names=['x', 'y', 'z', 'w'])
    """
    r = pd.concat((xs), axis=1, keys=keys)
    if names:
        if isinstance(names, (list, tuple)):
            if len(names) == len(r.columns):
                r.columns = names
            else:
                raise ValueError(
                    f"if 'names' contains new column names it must have same length as resulting df ({len(r.columns)})"
                )
        elif isinstance(names, dict):
            r = r.rename(columns=names)
    return r


def srows(*xs, keep="all", sort=True) -> Union[pd.DataFrame, pd.Series]:
    """
    Concat dataframes/series from xs into single dataframe by axis 0
    :param sort: if true it sorts resulting dataframe by index (default)
    :param keep: how to deal with duplicated indexes.
                 If set to 'all' it doesn't do anything (default). Otherwise keeps first or last occurences
    :return: combined dataframe

    Example
    -------
    >>>  srows(
            pd.DataFrame([1,2,3,4,-4], list('abcud')),
            pd.DataFrame([111,21,31,14], list('xyzu')),
            pd.DataFrame([11,21,31,124], list('ertu')),
            pd.DataFrame([11,21,31,14], list('WERT')),
            sort=True, keep='last')
    """
    r = pd.concat((xs), axis=0)
    r = r.sort_index() if sort else r
    if keep != "all":
        r = drop_duplicated_indexes(r, keep=keep)
    return r


def retain_columns_and_join(data: Dict[str, pd.DataFrame], columns: str | List[str]) -> pd.DataFrame:
    """
    Retains given columns from every value of data dictionary and concatenate them into single data frame

    from qube.datasource import DataSource
    from qube.analysis.tools import retain_columns_and_join

    ds = DataSource('yahoo::daily')
    data = ds.load_data(['aapl', 'msft', 'spy'], '2000-01-01', 'now')

    closes = retain_columns_and_join(data, 'close')
    hi_lo = retain_columns_and_join(data, ['high', 'low'])

    :param data: dictionary with dataframes
    :param columns: columns names need to be retained
    :return: data frame
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be passed as dictionary")

    return pd.concat([data[k][columns] for k in data.keys()], axis=1, keys=data.keys())


def continuous_periods(xs: pd.Series, cond: pd.Series) -> Struct:
    """
    Detect continues periods on series xs based on condition cond
    """
    df = scols(xs, cond, keys=["_XS_", "sig"])
    df["block"] = (df.sig.shift(1) != df.sig).astype(int).cumsum()
    idx_col_name = xs.index.name

    blk = df[df.sig].reset_index().groupby("block")[idx_col_name].apply(np.array)
    starts = blk.apply(lambda x: x[0])
    ends = blk.apply(lambda x: x[-1])
    se_info = scols(starts, ends, keys=["start", "end"])
    return Struct(blocks=blk.reset_index(drop=True), periods=se_info)


def roll(df: pd.DataFrame, w: int, **kwargs):
    """
    Rolling window on dataframe using multiple columns

    >>> roll(pd.DataFrame(np.random.randn(10,3), index=list('ABCDEFGHIJ')), 3).apply(print)

    or alternatively

    >>> pd.DataFrame(np.random.randn(10,3), index=list('ABCDEFGHIJ')).pipe(roll, 3).apply(lambda x: print(x[2]))

    :param df: pandas DataFrame
    :param w: window size (only integers)
    :return: rolling window
    """
    if w > len(df):
        raise ValueError("Window size exceeds number of rows !")

    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides
    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))
    rolled_df = pd.concat({row: pd.DataFrame(values, columns=df.columns) for row, values in zip(df.index, a)})

    return rolled_df.groupby(level=0, **kwargs)


def dict_to_frame(x: dict, index_type=None, orient="index", columns=None, column_types=dict()) -> pd.DataFrame:
    """
    Utility for convert dictionary to indexed DataFrame
    It's possible to pass columns names and type of index
    """
    y = pd.DataFrame().from_dict(x, orient=orient)
    if index_type:
        if index_type in ["ns", "nano"]:
            index_type = "M8[ns]"
        y.index = y.index.astype(index_type)

    # rename if needed
    if columns:
        columns = [columns] if not isinstance(columns, (list, tuple, set)) else columns
        if len(columns) == len(y.columns):
            y.rename(columns=dict(zip(y.columns, columns)), inplace=True)
        else:
            raise ValueError("dict_to_frame> columns argument must contain %d elements" % len(y.columns))

    # if additional conversion is required
    if column_types:
        _existing_cols_conversion = {c: v for c, v in column_types.items() if c in y.columns}
        y = y.astype(_existing_cols_conversion)

    return y


def select_column_and_join(data: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """
    Select given column from every value of data dictionary and concatenate them into single data frame

    from qube.datasource import DataSource
    from qube.analysis.tools import retain_columns_and_join

    ds = DataSource('yahoo::daily')
    data = ds.load_data(['aapl', 'msft', 'spy'], '2000-01-01', 'now')

    closes = select_column_and_join(data, 'close')
    hi_lo = select_column_and_join(data, ['high', 'low'])

    :param data: dictionary with dataframes
    :param columns: column name need to be selected
    :return: pandas data frame
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be passed as dictionary of pandas dataframes")

    return pd.concat([data[k][column] for k in data.keys()], axis=1, keys=data.keys())


def merge_columns_by_op(x: pd.DataFrame, y: pd.DataFrame, op):
    """
    Merge 2 dataframes into one and performing operation on intersected columns

    merge_columns_by_op(
        pd.DataFrame({'A': [1,2,3], 'B': [100,200,300]}),
        pd.DataFrame({'B': [5,6,7], 'C': [10,20,30]}),
        lambda x,y: x + y
    )

        B 	    A   C
    0 	105 	1 	10
    1 	206 	2 	20
    2 	307 	3 	30

    """
    if x is None or x.empty:
        return y
    if y is None:
        return x
    r = []
    uc = set(x.columns & y.columns)
    for c in uc:
        r.append(op(x[c], y[c]))

    for c in set(x.columns) - uc:
        r.append(x[c])

    for c in set(y.columns) - uc:
        r.append(y[c])

    return scols(*r)


def bands_signals(
    prices: pd.DataFrame,
    score: pd.DataFrame,
    entry,
    exit,
    stop: Optional[float] = np.inf,
    entry_method="cross-out",  # 'cross-in', 'revert-to-band'
    position_sizes_fn=lambda time, score, prices, side: np.zeros(len(p)),
) -> pd.DataFrame:
    """
    Generate trading signals using score and entry / exit thresholds
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame")

    _as_series = lambda xs, index, name: pd.Series(xs, index=index, name=name) if isscalar(xs) else xs

    # - entry function
    ent_fn: Callable[[float, float, float, float], int] = lambda t, s2, s1, s0: 0

    match entry_method:
        case "cross-in":
            ent_fn = lambda t, s2, s1, s0: (
                +1 if (s2 < -t and s1 <= -t and s0 > -t) else -1 if (s2 > +t and s1 >= +t and s0 < +t) else 0
            )

        case "cross-out":
            ent_fn = lambda t, s2, s1, s0: (
                +1 if (s2 >= -t and s1 >= -t and s0 < -t) else -1 if (s2 <= +t and s1 <= +t and s0 > +t) else 0
            )

        case "revert-to-band":
            ent_fn = lambda t, s2, s1, s0: (
                +1 if (s1 <= -t and s0 < -t and s0 > s1) else -1 if (s1 >= +t and s0 > +t and s0 < s1) else 0
            )

        case _:
            raise ValueError(
                f"Unknown entry_method: {entry_method}, supported methods are: cross-out, cross-in, revert-to-band"
            )

    entry = abs(_as_series(entry, score.index, "entry"))
    exit = _as_series(exit, score.index, "exit")
    stop = abs(_as_series(stop, score.index, "stop"))

    mx = scols(prices, scols(score.rename("score"), entry, exit, stop), keys=["price", "service"]).dropna()
    px: pd.DataFrame = mx["price"]
    sx: pd.DataFrame = mx["service"]
    P0 = np.zeros(px.shape[1])  # zero position

    signals = {}
    pos = 0
    s1, s2 = np.nan, np.nan

    for t, pi, (s0, en, ex, stp) in zip(px.index, px.values, sx.values):
        if pos == 0:  # no positions yet
            # - only if there are at least 2 past scores available
            if not np.isnan(s1) and not np.isnan(s2):
                if (side := ent_fn(en, s2, s1, s0)) != 0:
                    signals[t] = position_sizes_fn(t, s0, pi, side)
                    pos = side  # - sell or buy spread

        elif pos == -1:
            # - check for stop or exit
            if s0 >= +stp or s0 <= +ex:
                signals[t] = P0
                pos = 0

        elif pos == +1:
            # - check for stop or exit
            if s0 <= -stp or s0 >= -ex:
                signals[t] = P0
                pos = 0

        s1, s2 = s0, s1

    return pd.DataFrame.from_dict(signals, orient="index", columns=px.columns)


def ohlc_resample(
    df: pd.DataFrame | pd.Series | Dict[str, pd.DataFrame],
    new_freq: str = "1h",
    vmpt: bool = False,
    resample_tz=None,
    non_ohlc_columns_aggregator="sum",
) -> pd.DataFrame | dict:
    """
    Resample OHLCV/tick series to new timeframe.

    Example:
    >>> d = pd.DataFrame({
    >>>          'open' : np.random.randn(30),
    >>>          'high' : np.random.randn(30),
    >>>          'low' : np.random.randn(30),
    >>>          'close' : np.random.randn(30)
    >>>         }, index=pd.date_range('2000-01-01 00:00', freq='5Min', periods=30))
    >>>
    >>> ohlc_resample(d, '15Min')
    >>>
    >>> # if we need to resample quotes
    >>> from qube.datasource import DataSource
    >>> with DataSource('kdb::dukas') as ds:
    >>>     quotes = ds.load_data(['EURUSD', 'GBPUSD'], '2018-05-07', '2018-05-11')
    >>> ohlc_resample(quotes, '1Min', vmpt=True)

    :param df: input ohlc or bid/ask quotes or dict
    :param new_freq: how to resample rule (see pandas.DataFrame::resample)
    :param vmpt: use volume weighted price for quotes (if false midprice will be used)
    :param resample_tz: timezone for resample. For example, to create daily bars in the EET timezone
    :param non_ohlc_columns_aggregator: how aggregate unknown columns
    :return: resampled ohlc / dict
    """

    def __mx_rsmpl(d, freq: str, is_vmpt: bool = False, resample_tz=None) -> pd.DataFrame:
        _cols = d.columns
        _source_tz = d.index.tz

        # if we have trades
        if all([i in _cols for i in ["price", "side", "amount"]]):
            result = _tz_convert(d.price, resample_tz, _source_tz)
            result = result.resample(freq).agg("ohlc")
            result["volume"] = d.amount.resample(freq).sum()
            result["quote_volume"] = (d.amount * d.price).resample(freq).sum()
            result["taker_buy_volume"] = d[d.side == "buy"].amount.resample(freq).sum()
            result["taker_buy_quote_volume"] = (
                (d[d.side == "buy"].amount * d[d.side == "buy"].price).resample(freq).sum()
            )
            return result if not resample_tz else result.tz_convert(_source_tz)

        # if we have bid/ask frame
        if "ask" in _cols and "bid" in _cols:
            # if sizes are presented we can calc vmpt if needed
            if is_vmpt and "askvol" in _cols and "bidvol" in _cols:
                mp = (d.ask * d.bidvol + d.bid * d.askvol) / (d.askvol + d.bidvol)
                return mp.resample(freq).agg("ohlc")

            # if there is only asks and bids and we don't need vmpt
            result = _tz_convert(d[["ask", "bid"]].mean(axis=1), resample_tz, _source_tz)
            result = result.resample(freq).agg("ohlc")
            # Convert timezone to back if it changed
            return result if not resample_tz else result.tz_convert(_source_tz)

        # for OHLC case or just simple series
        if all([i in _cols for i in ["open", "high", "low", "close"]]) or isinstance(d, pd.Series):
            ohlc_rules = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "ask_vol": "sum",
                "bid_vol": "sum",
                "volume": "sum",
            }
            result = _tz_convert(d, resample_tz, _source_tz)
            # result = result.resample(freq).apply(dict(i for i in ohlc_rules.items() if i[0] in d.columns)).dropna()
            #  25-Oct-2022: we allow other columns to be included in transformation (just use last value)
            result = (
                result.resample(freq)
                .apply({c: ohlc_rules.get(c, non_ohlc_columns_aggregator) for c in d.columns})
                .dropna()
            )

            # Convert timezone to back if it changed
            return result if not resample_tz else result.tz_convert(_source_tz)

        raise ValueError("Can't recognize structure of input data !")

    def _tz_convert(df, tz, source_tz):
        if tz:
            if not source_tz:
                df = df.tz_localize("GMT")
            return df.tz_convert(tz)
        else:
            return df

    if isinstance(df, (pd.DataFrame, pd.Series)):
        return __mx_rsmpl(df, new_freq, vmpt, resample_tz)
    elif isinstance(df, dict):
        return {k: __mx_rsmpl(v, new_freq, vmpt, resample_tz) for k, v in df.items()}
    else:
        raise ValueError("Type [%s] is not supported in ohlc_resample" % str(type(df)))


def shift_series(
    sigs: pd.Series | pd.DataFrame,
    forward: str | None = None,
    days=0,
    hours=0,
    minutes=0,
    seconds=0,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Shift timeseries into future.

    :return: shifted frame or series
    """
    n_sigs = sigs.copy()
    f0 = pd.Timedelta(forward if forward is not None else 0)
    n_sigs.index = n_sigs.index + f0 + pd.DateOffset(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return n_sigs


def _frame_to_str(data: pd.DataFrame | pd.Series, name: str, start=4, end=4, time_info=True) -> str:
    r = ""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        t_info = f"{len(data)} records"
        if time_info:
            t_info += f" | {data.index[0]}:{data.index[-1]}"
        hdr = f"- - -({name} {t_info} records)- - -"
        sep = " -" * 50
        r += hdr[: len(sep)] + "\n"
        r += data.head(start).to_string(header=True) + "\n"
        if start < len(data) and end > 0:
            r += "\t. . . . . . \n"
            _s = data.tail(end).to_string(header=True)
            r += "\n".join(_s.split("\n")[2:]) + "\n"
    else:
        r = str(data)
    return r


class OhlcDict(dict):
    """
    Extended dictionary with method to perform resampling on OHLCV (Open, High, Low, Close, Volume) data.

    Example:
    -------
    >>> index=pd.date_range('2020-01-01', periods=10, freq='15min')
        nc = np.random.rand(10).cumsum()
        d = OhlcDict({
            "A": pd.DataFrame({"open": nc, "high": nc,"low": nc,"close": nc }, index=index),
            "B": pd.DataFrame({"open": nc, "high": nc,"low": nc,"close": nc }, index=index),
        })
        print(d.close)
        print(d("1h").close)
        # - just show full info about this dict
        print(str(d))

        # - get item by keys
        print(d["A", "B"].close)
    """

    _orig: dict[str, pd.DataFrame | pd.Series | OHLCV]
    _fields: Set[str]

    def __init__(self, orig: dict[str, pd.DataFrame | pd.Series | OHLCV]):
        self._orig = orig
        _o_copy = {}
        _lst = []
        if isinstance(orig, dict):
            for k, o in orig.items():
                if not isinstance(o, (pd.DataFrame | pd.Series | OHLCV)):
                    raise ValueError(
                        f"All values in the dictionary must be pandas Series, DataFrames or OHLCV, but {k} is {type(o)}"
                    )

                if isinstance(o, OHLCV):
                    o = o.pd()

                # - skip empty data
                if not o.empty:
                    _o_copy[k] = o
                    _lst.extend(o.columns.values)

        self._fields = set(_lst)
        super().__init__(_o_copy)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if args:
            try:
                return OhlcDict(ohlc_resample(self, pd.Timedelta(args[0]), **kwds))
            except Exception as e:
                raise ValueError(str(e))
        return self

    def __getattribute__(self, name: str) -> Any:
        if name != "_fields":
            if name in self._fields:
                return retain_columns_and_join(self, name)
        return super().__getattribute__(name)

    def display(self, heads=-1, tails=0) -> str:
        r = ""
        _h, _t = (len(self), 0) if heads == -1 else (heads, tails)
        for k, v in self.items():
            r += _frame_to_str(v, name=k, start=_h, end=_t) + "\n"
        return r

    def __str__(self) -> str:
        return self.display(3, 3)

    def __repr__(self) -> str:
        return self.display(3, 3)

    def __reduce__(self):
        """
        For joblib Parallel compatibility - defines how to pickle the object
        """
        # Return the class, constructor arguments, and additional state
        data = {k: v for k, v in self.items()}
        return (self.__class__, (data,), {"_orig": self._orig})

    def __setstate__(self, state):
        """Restore object from pickle state"""
        # Recreate the object using the constructor
        self.__init__(state["_orig"])

    def __getitem__(self, key: str | list | tuple) -> "pd.DataFrame | pd.Series | OhlcDict":
        match key:
            case str():
                return super().__getitem__(key)
            case list() | tuple():
                return OhlcDict({k: self[k] for k in key})
        raise KeyError(f"Unknown key: {key}")
