import types
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from pytest import approx

from qubx.core.series import TimeSeries
from qubx.core.utils import recognize_time

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)


def drop_duplicated_indexes(df, keep="first"):
    return df[~df.index.duplicated(keep=keep)]


def scols(*xs, keys=None, names=None, keep="all"):
    r = pd.concat([x.to_series() if isinstance(x, TimeSeries) else x for x in xs], axis=1, keys=keys)
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


def srows(*xs, keep="all", sort=True):
    r = pd.concat((xs), axis=0)
    r = r.sort_index() if sort else r
    if keep != "all":
        r = drop_duplicated_indexes(r, keep=keep)
    return r


def push(series: TimeSeries, ds: List[Tuple], v=None) -> TimeSeries:
    """
    Update series by data from the input
    """
    for t, d in ds:
        if isinstance(t, str):
            t = recognize_time(t)
        elif isinstance(t, pd.Timestamp):
            t = t.asm8
        if isinstance(d, (list, tuple)):
            series.update(t, d[0], d[1])
        else:
            series.update(t, d) if v is None else series.update(t, d, v)
    return series


def shift(xs: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = fill
        e[n:] = xs[:-n]
    else:
        e[n:] = fill
        e[:n] = xs[-n:]
    return e


def column_vector(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    return np.reshape(x, (x.shape[0], -1))


def sink_nans_down(x_in, copy=False) -> Tuple[np.ndarray, np.ndarray]:
    x = np.copy(x_in) if copy else x_in
    n_ix = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        f_n = np.where(~np.isnan(x[:, i]))[0]
        if len(f_n) > 0:
            if f_n[0] != 0:
                x[:, i] = np.concatenate((x[f_n[0] :, i], x[: f_n[0], i]))
            n_ix[i] = f_n[0]
    return x, n_ix


def lift_nans_up(x_in, n_ix, copy=False) -> np.ndarray:
    x = np.copy(x_in) if copy else x_in
    for i in range(0, x.shape[1]):
        f_n = int(n_ix[i])
        if f_n != 0:
            x[:, i] = np.concatenate((nans(f_n), x[:-f_n, i]))
    return x


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    for i in range(0, x.shape[1]):
        ret = np.nancumsum(x[:, i])
        ret[n:] = ret[n:] - ret[:-n]
        x[:, i] = np.concatenate((nans(n - 1), ret[n - 1 :]))
    return x


def nans(dims):
    return np.nan * np.ones(dims)


def apply_to_frame(func, x, *args, **kwargs):
    _keep_names = False
    if "keep_names" in kwargs:
        _keep_names = kwargs.pop("keep_names")

    if func is None or not isinstance(func, types.FunctionType):
        raise ValueError(str(func) + " must be callable object")

    xp = column_vector(func(x, *args, **kwargs))
    _name = None
    if not _keep_names:
        _name = func.__name__ + "_" + "_".join([str(i) for i in args])

    if isinstance(x, pd.DataFrame):
        c_names = x.columns if _keep_names else ["%s_%s" % (c, _name) for c in x.columns]
        return pd.DataFrame(xp, index=x.index, columns=c_names)
    elif isinstance(x, pd.Series):
        return pd.Series(xp.flatten(), index=x.index, name=_name)

    return xp


def sma(x, period):
    """
    Classical simple moving average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :return: smoothed values
    """
    if period <= 0:
        raise ValueError("Period must be positive and greater than zero !!!")

    x = column_vector(x)
    x, ix = sink_nans_down(x, copy=True)
    s = rolling_sum(x, period) / period
    return lift_nans_up(s, ix)


def _calc_ema(x, span, init_mean=True, min_periods=0):
    alpha = 2.0 / (1 + span)
    x = x.astype(np.float64)
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        x_s = x[:, i][nan_start:]
        a_1 = 1 - alpha
        s = np.zeros(x_s.shape)

        start_i = 1
        if init_mean:
            s += np.nan
            if span - 1 >= len(s):
                x[:, :] = np.nan
                continue
            s[span - 1] = np.mean(x_s[:span])
            start_i = span
        else:
            s[0] = x_s[0]

        for n in range(start_i, x_s.shape[0]):
            s[n] = alpha * x_s[n] + a_1 * s[n - 1]

        if min_periods > 0:
            s[: min_periods - 1] = np.nan

        x[:, i] = np.concatenate((nans(nan_start), s))

    return x


def ema(x, span, init_mean=True, min_periods=0) -> np.ndarray:
    return _calc_ema(column_vector(x), span, init_mean, min_periods)


def tema(x, n: int, init_mean=True):
    e1 = ema(x, n, init_mean=init_mean)
    e2 = ema(e1, n, init_mean=init_mean)
    return 3 * e1 - 3 * e2 + ema(e2, n, init_mean=init_mean)


def kama(xs, period=10, period_fast=2, period_slow=30):
    # Efficiency Ratio
    change = abs(xs - xs.shift(period))
    volatility = (abs(xs - xs.shift())).rolling(period).sum()
    er = change / volatility

    # Smoothing Constant
    sc_fatest = 2 / (period_fast + 1)
    sc_slowest = 2 / (period_slow + 1)
    sc = (er * (sc_fatest - sc_slowest) + sc_slowest) ** 2

    # KAMA
    kama = np.zeros_like(xs)
    kama[period - 1] = xs.iloc[period - 1]
    for i in range(period, len(xs)):
        kama[i] = kama[i - 1] + sc.iloc[i] * (xs.iloc[i] - kama[i - 1])
    kama[kama == 0] = np.nan

    return kama


def __empty_smoother(x, *args, **kwargs):
    return column_vector(x)


def smooth(x, stype: Union[str, types.FunctionType], *args, **kwargs) -> pd.Series:
    """
    Smooth series using either given function or find it by name from registered smoothers
    """
    smoothers = {
        "sma": sma,
        "ema": ema,
        "tema": tema,
        "dema": dema,
        "kama": kama,
        # 'zlema': zlema, 'jma': jma, 'wma': wma, 'mcginley': mcginley, 'hma': hma
    }

    f_sm = __empty_smoother
    if isinstance(stype, str):
        if stype in smoothers:
            f_sm = smoothers.get(stype)
        else:
            raise ValueError(
                f"Smoothing method '{stype}' is not supported, supported methods: {list(smoothers.keys())}"
            )

    if isinstance(stype, types.FunctionType):
        f_sm = stype

    # smoothing
    x_sm = f_sm(x, *args, **kwargs)
    return x_sm if isinstance(x_sm, pd.Series) else pd.Series(x_sm.flatten(), index=x.index)


def dema(x, n: int, init_mean=True):
    e1 = ema(x, n, init_mean=init_mean)
    return 2 * e1 - ema(e1, n, init_mean=init_mean)


def rsi(x, periods, smoother=sma):
    """
    U = X_t - X_{t-1}, D = 0 when X_t > X_{t-1}
    D = X_{t-1} - X_t, U = 0 when X_t < X_{t-1}
    U = 0, D = 0,            when X_t = X_{t-1}

    RSI = 100 * E[U, n] / (E[U, n] + E[D, n])
    """
    xx = pd.concat((x, x.shift(1)), axis=1, keys=["c", "p"])
    df = xx.c - xx.p
    mu = smooth(df.where(df > 0, 0), smoother, periods)
    md = smooth(abs(df.where(df < 0, 0)), smoother, periods)
    return 100 * mu / (mu + md)


def stochastic(x, period, smooth_period, smoother="sma"):
    """
    Classical stochastic oscillator indicator
    :param x: series or OHLC dataframe
    :param period: indicator's period
    :param smooth_period: period of smoothing
    :param smoother: smoothing method (sma by default)
    :return: K and D series as DataFrame
    """
    hi, li, xi = x, x, x

    hh = hi.rolling(period).max()
    ll = li.rolling(period).min()
    k = 100 * (xi - ll) / (hh - ll)
    d = smooth(k, smoother, smooth_period)
    return scols(k, d, names=["K", "D"])
