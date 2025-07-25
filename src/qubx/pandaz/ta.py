import types
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit
from statsmodels.regression.linear_model import OLS

from qubx.pandaz.utils import check_frame_columns, continuous_periods, has_columns, ohlc_resample, scols, srows
from qubx.utils.misc import Struct
from qubx.utils.time import infer_series_frequency


def __apply_to_frame(func: types.FunctionType, x: pd.Series | pd.DataFrame, *args, **kwargs):
    """
    Utility applies given function to x and converts result to incoming type

    >>> from qubx.ta.pandaz import ema
    >>> apply_to_frame(ema, data['EURUSD'], 50)
    >>> apply_to_frame(lambda x, p1: x + p1, data['EURUSD'], 1)

    :param func: function to map
    :param x: input data
    :param args: arguments of func
    :param kwargs: named arguments of func (if it contains keep_names=True it won't change source columns names)
    :return: result of function's application
    """
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


def column_vector(x) -> np.ndarray:
    """
    Convert any vector to column vector. Matrices remain unchanged.

    :param x: vector
    :return: column vector
    """
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    return np.reshape(x, (x.shape[0], -1))


@njit
def shift(xs: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    """
    Shift data in numpy array (aka lag function):

    shift(np.array([[1.,2.],
                    [11.,22.],
                    [33.,44.]]), 1)

    >> array([[ nan,  nan],
              [  1.,   2.],
              [ 11.,  22.]])

    :param xs:
    :param n:
    :param fill: value to use for
    :return:
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = fill
        e[n:] = xs[:-n]
    else:
        e[n:] = fill
        e[:n] = xs[-n:]
    return e


def sink_nans_down(x_in, copy=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move all starting nans 'down to the bottom' in every column.

    NaN = np.nan
    x = np.array([[NaN, 1, NaN],
                  [NaN, 2, NaN],
                  [NaN, 3, NaN],
                  [10,  4, NaN],
                  [20,  5, NaN],
                  [30,  6, 100],
                  [40,  7, 200]])

    x1, nx = sink_nans_down(x)
    print(x1)

    >> [[  10.    1.  100.]
        [  20.    2.  200.]
        [  30.    3.   nan]
        [  40.    4.   nan]
        [  nan    5.   nan]
        [  nan    6.   nan]
        [  nan    7.   nan]]

    :param x_in: numpy 1D/2D array
    :param copy: set if need to make copy input to prevent being modified [False by default]
    :return: modified x_in and indexes
    """
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
    """
    Move all ending nans 'up to top' of every column.

    NaN = np.nan
    x = np.array([[NaN, 1, NaN],
                  [NaN, 2, NaN],
                  [NaN, 3, NaN],
                  [10,  4, NaN],
                  [20,  5, NaN],
                  [30,  6, 100],
                  [40, 7, 200]])

    x1, nx = sink_nans_down(x)
    print(x1)

    >> [[  10.    1.  100.]
        [  20.    2.  200.]
        [  30.    3.   nan]
        [  40.    4.   nan]
        [  nan    5.   nan]
        [  nan    6.   nan]
        [  nan    7.   nan]]

    x2 = lift_nans_up(x1, nx)
    print(x2)

    >> [[  nan    1.   nan]
        [  nan    2.   nan]
        [  nan    3.   nan]
        [  10.    4.   nan]
        [  20.    5.   nan]
        [  30.    6.  100.]
        [  40.    7.  200.]]

    :param x_in: numpy 1D/2D array
    :param n_ix: indexes for every column
    :param copy: set if need to make copy input to prevent being modified [False by default]
    :return: modified x_in
    """
    x = np.copy(x_in) if copy else x_in
    for i in range(0, x.shape[1]):
        f_n = int(n_ix[i])
        if f_n != 0:
            x[:, i] = np.concatenate((nans(f_n), x[:-f_n, i]))
    return x


@njit
def nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.

    :param dims: dimensions tuple
    :return: nans matrix
    """
    return np.nan * np.ones(dims)


@njit
def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    """
    Fast running sum for numpy array (matrix) along columns.

    Example:
    >>> rolling_sum(column_vector(np.array([[1,2,3,4,5,6,7,8,9], [11,22,33,44,55,66,77,88,99]]).T), n=5)

    array([[  nan,   nan],
       [  nan,   nan],
       [  nan,   nan],
       [  nan,   nan],
       [  15.,  165.],
       [  20.,  220.],
       [  25.,  275.],
       [  30.,  330.],
       [  35.,  385.]])

    :param x: input data
    :param n: rolling window size
    :return: rolling sum for every column preceded by nans
    """
    xs = nans(x.shape)
    for i in range(0, x.shape[1]):
        ret = np.nancumsum(x[:, i])
        ret[n:] = ret[n:] - ret[:-n]
        xs[(n - 1) :, i] = ret[n - 1 :]
    return xs


def __wrap_dataframe_decorator(func):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], (pd.Series, pd.DataFrame)):
            return __apply_to_frame(func, *args, **kwargs)
        else:
            return func(*args)

    return wrapper


def __empty_smoother(x, *args, **kwargs):
    return column_vector(x)


def running_view(arr, window, axis=-1):
    """
    Produces running view (lagged matrix) from given array.

    Example:

    > running_view(np.array([1,2,3,4,5,6]), 3)

    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])

    :param arr: array of numbers
    :param window: window length
    :param axis:
    :return: lagged matrix
    """
    shape = list(arr.shape)
    shape[axis] -= window - 1
    return np.lib.index_tricks.as_strided(arr, shape + [window], arr.strides + (arr.strides[axis],))


def detrend(y, order):
    """
    Removes linear trend from the series y.
    detrend computes the least-squares fit of a straight line to the data
    and subtracts the resulting function from the data.

    :param y:
    :param order:
    :return:
    """
    if order == -1:
        return y
    return OLS(y, np.vander(np.linspace(-1, 1, len(y)), order + 1)).fit().resid


def moving_detrend(y: pd.Series, order: int, window: int) -> pd.DataFrame:
    """
    Removes linear trend from the series y by using sliding window.
    :param y: series (ndarray or pd.DataFrame/Series)
    :param order: trend's polinome order
    :param window: sliding window size
    :return: (residual, rsquatred, betas)
    """
    yy = running_view(column_vector(y).T[0], window=window)
    n_pts = len(y)
    resid = nans((n_pts))
    r_sqr = nans((n_pts))
    betas = nans((n_pts, order + 1))
    for i, p in enumerate(yy):
        n = len(p)
        lr = OLS(p, np.vander(np.linspace(-1, 1, n), order + 1)).fit()
        r_sqr[n - 1 + i] = lr.rsquared
        resid[n - 1 + i] = lr.resid[-1]
        betas[n - 1 + i, :] = lr.params

    # return pandas frame if input is series/frame
    r = pd.DataFrame({"resid": resid, "r2": r_sqr}, index=y.index, columns=["resid", "r2"])
    betas_fr = pd.DataFrame(betas, index=y.index, columns=["b%d" % i for i in range(order + 1)])
    return pd.concat((r, betas_fr), axis=1)


def moving_ols(y, x, window):
    """
    Function for calculating moving linear regression model using sliding window
        y = B*x + err
    returns array of betas, residuals and standard deviation for residuals
    residuals = y - yhat, where yhat = betas * x

    Example:

    x = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randn(100).cumsum())
    m = moving_ols(y, x, 5)
    lr_line = (x * m).sum(axis=1)

    :param y: dependent variable (vector)
    :param x: exogenous variables (vector or matrix)
    :param window: sliding windowsize
    :return: array of betas, residuals and standard deviation for residuals
    """
    # if we have any indexes
    idx_line = y.index if isinstance(y, (pd.Series, pd.DataFrame)) else None
    x_col_names = x.columns if isinstance(y, (pd.Series, pd.DataFrame)) else None

    x = column_vector(x)
    y = column_vector(y)
    nx = len(x)
    if nx != len(y):
        raise ValueError("Series must contain equal number of points")

    if y.shape[1] != 1:
        raise ValueError("Response variable y must be column array or series object")

    if window > nx:
        raise ValueError("Window size must be less than number of observations")

    betas = nans(x.shape)
    err = nans((nx))
    sd = nans((nx))

    for i in range(window, nx + 1):
        ys = y[(i - window) : i]
        xs = x[(i - window) : i, :]
        lr = OLS(ys, xs).fit()
        betas[i - 1, :] = lr.params
        err[i - 1] = y[i - 1] - (x[i - 1, :] * lr.params).sum()
        sd[i - 1] = lr.resid.std()

    # convert to dataframe if need
    if x_col_names is not None and idx_line is not None:
        _non_empy = lambda c, idx: c if c else idx
        _bts = pd.DataFrame({_non_empy(c, i): betas[:, i] for i, c in enumerate(x_col_names)}, index=idx_line)
        return pd.concat((_bts, pd.DataFrame({"error": err, "stdev": sd}, index=idx_line)), axis=1)
    else:
        return betas, err, sd


def holt_winters_second_order_ewma(x: pd.Series, span: int, beta: float) -> pd.DataFrame:
    """
    The Holt-Winters second order method (aka double exponential smoothing) attempts to incorporate the estimated
    trend into the smoothed data, using a {b_{t}} term that keeps track of the slope of the original signal.
    The smoothed signal is written to the s_{t} term.

    :param x: series values (DataFrame, Series or numpy array)
    :param span: number of data points taken for calculation
    :param beta: trend smoothing factor, 0 < beta < 1
    :return: tuple of smoothed series and smoothed trend
    """
    if span < 0:
        raise ValueError("Span value must be positive")

    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values

    x = np.reshape(x, (x.shape[0], -1))
    alpha = 2.0 / (1 + span)
    r_alpha = 1 - alpha
    r_beta = 1 - beta
    s = np.zeros(x.shape)
    b = np.zeros(x.shape)
    s[0, :] = x[0, :]
    for i in range(1, x.shape[0]):
        s[i, :] = alpha * x[i, :] + r_alpha * (s[i - 1, :] + b[i - 1, :])
        b[i, :] = beta * (s[i, :] - s[i - 1, :]) + r_beta * b[i - 1, :]
    return pd.DataFrame({"smoothed": s, "trend": b}, index=x.index)


@__wrap_dataframe_decorator
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


@njit
def _calc_kama(x, period, fast_span, slow_span):
    x = x.astype(np.float64)
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        x_s = x[:, i][nan_start:]
        if period >= len(x_s):
            raise ValueError("Wrong value for period. period parameter must be less than number of input observations")
        abs_diff = np.abs(x_s - shift(x_s, 1))
        er = np.abs(x_s - shift(x_s, period)) / rolling_sum(np.reshape(abs_diff, (len(abs_diff), -1)), period)[:, 0]
        sc = np.square((er * (2.0 / (fast_span + 1) - 2.0 / (slow_span + 1.0)) + 2 / (slow_span + 1.0)))
        ama = nans(sc.shape)

        # here ama_0 = x_0
        ama[period - 1] = x_s[period - 1]
        for n in range(period, len(ama)):
            ama[n] = ama[n - 1] + sc[n] * (x_s[n] - ama[n - 1])

        # drop 1-st kama value (just for compatibility with ta-lib)
        ama[period - 1] = np.nan

        x[:, i] = np.concatenate((nans(nan_start), ama))

    return x


@__wrap_dataframe_decorator
def kama(x, period, fast_span=2, slow_span=30):
    """
    Kaufman Adaptive Moving Average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :param fast_span: fast period (default is 2 as in canonical impl)
    :param slow_span: slow period (default is 30 as in canonical impl)
    :return: smoothed values
    """
    x = column_vector(x)
    return _calc_kama(x, period, fast_span, slow_span)


@njit
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


@__wrap_dataframe_decorator
def ema(x, span, init_mean=True, min_periods=0) -> np.ndarray:
    """
    Exponential moving average

    :param x: data to be smoothed
    :param span: number of data points for smooth
    :param init_mean: use average of first span points as starting ema value (default is true)
    :param min_periods: minimum number of observations in window required to have a value (0)
    :return:
    """
    x = column_vector(x)
    return _calc_ema(x, span, init_mean, min_periods)


@__wrap_dataframe_decorator
def zlema(x: np.ndarray, n: int, init_mean=True):
    """
    'Zero lag' moving average
    :type x: np.array
    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    x = column_vector(x)
    return ema(2 * x - shift(x, n), n, init_mean=init_mean)


@__wrap_dataframe_decorator
def dema(x, n: int, init_mean=True):
    """
    Double EMA

    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    e1 = ema(x, n, init_mean=init_mean)
    return 2 * e1 - ema(e1, n, init_mean=init_mean)


@__wrap_dataframe_decorator
def tema(x, n: int, init_mean=True):
    """
    Triple EMA

    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    e1 = ema(x, n, init_mean=init_mean)
    e2 = ema(e1, n, init_mean=init_mean)
    return 3 * e1 - 3 * e2 + ema(e2, n, init_mean=init_mean)


@__wrap_dataframe_decorator
def wma(x, period: int, weights=None):
    """
    Weighted moving average

    :param x: values to be averaged
    :param period: period (used for standard WMA (weights = [1,2,3,4, ... period]))
    :param weights: custom weights array
    :return: weighted values
    """
    x = column_vector(x)

    # if weights are set up
    if weights is None or not weights:
        w = np.arange(1, period + 1)
    else:
        w = np.array(weights)
        period = len(w)

    if period > len(x):
        raise ValueError(f"Period for wma must be less than number of rows. {period}, {len(x)}")

    w = (w / np.sum(w))[::-1]  # order fixed !
    y = x.astype(np.float64).copy()
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        y_s = y[:, i][nan_start:]
        wm = np.concatenate((nans(period - 1), np.convolve(y_s, w, "valid")))
        y[:, i] = np.concatenate((nans(nan_start), wm))

    return y


@__wrap_dataframe_decorator
def hma(x, period: int):
    """
    Hull moving average

    :param x: values to be averaged
    :param period: period
    :return: weighted values
    """
    return wma(2 * wma(x, period // 2) - wma(x, period), int(np.sqrt(period)))


@__wrap_dataframe_decorator
def bidirectional_ema(x: pd.Series, span: int, smoother="ema") -> np.ndarray:
    """
    EMA function is really appropriate for stationary data, i.e., data without trends or seasonality.
    In particular, the EMA function resists trends away from the current mean that it’s already “seen”.
    So, if you have a noisy hat function that goes from 0, to 1, and then back to 0, then the EMA function will return
    low values on the up-hill side, and high values on the down-hill side.
    One way to circumvent this is to smooth the signal in both directions, marching forward,
    and then marching backward, and then average the two.

    :param x: data
    :param span: span for smoothing
    :param smoother: smoothing function (default 'ema' or 'tema')
    :return: smoohted data
    """
    if smoother == "tema":
        fwd = tema(x, span, init_mean=False)  # take TEMA in forward direction
        bwd = tema(x[::-1], span, init_mean=False)  # take TEMA in backward direction
    else:
        fwd = ema(x, span=span, init_mean=False)  # take EMA in forward direction
        bwd = ema(x[::-1], span=span, init_mean=False)  # take EMA in backward direction
    return (fwd + bwd[::-1]) / 2.0


def series_halflife(series: pd.Series | np.ndarray) -> float:
    """
    Tries to find half-life time for this series.

    Example:
    >>> series_halflife(np.array([1,0,2,3,2,1,-1,-2,0,1]))
    >>> 2.0

    :param series: series data (np.array or pd.Series)
    :return: half-life value rounded to integer
    """
    ser = column_vector(series)
    if ser.shape[1] > 1:
        raise ValueError("Nultimple series is not supported")

    lag = ser[1:]
    dY = -np.diff(ser, axis=0)
    m = OLS(dY, sm.add_constant(lag, prepend=False))
    reg = m.fit()

    return np.ceil(-np.log(2) / reg.params[0])


def rolling_std_with_mean(x: pd.Series, mean: float | pd.Series, window: int):
    """
    Calculates rolling standard deviation for data from x and already calculated mean series
    :param x: series data
    :param mean: calculated mean
    :param window: window
    :return: rolling standard deviation
    """
    return np.sqrt((((x - mean) ** 2).rolling(window=window).sum() / (window - 1)))


def rolling_zscore(x: pd.Series, window: int, std_window: int | None = None):
    """
    Calculates rolling z-score for data from x
    :param x: series data
    :param window: window
    :return: rolling z-score
    """
    if std_window is None:
        std_window = window
    return (x - x.rolling(window=window).mean()) / x.rolling(window=std_window).std(ddof=0)


def bollinger(x: pd.Series, window=14, nstd=2, mean="sma") -> pd.DataFrame:
    """
    Bollinger Bands indicator

    :param x: input data
    :param window: lookback window
    :param nstd: number of standard devialtions for bands
    :param mean: method for calculating mean: sma, ema, tema, dema, zlema, kama
    :param as_frame: if true result is returned as DataFrame
    :return: mean, upper and lower bands
    """
    rolling_mean = smooth(x, mean, window)
    rolling_std = rolling_std_with_mean(x, rolling_mean, window)

    upper_band = rolling_mean + (rolling_std * nstd)
    lower_band = rolling_mean - (rolling_std * nstd)

    _bb = rolling_mean, upper_band, lower_band
    return pd.concat(_bb, axis=1, keys=["Median", "Upper", "Lower"])


def bollinger_atr(x: pd.DataFrame, window=14, atr_window=14, natr=2, mean="sma", atr_mean="ema") -> pd.DataFrame:
    """
    Bollinger Bands indicator where ATR is used for bands range estimating
    :param x: input data
    :param window: window size for averaged price
    :param atr_window: atr window size
    :param natr:  number of ATRs for bands
    :param mean: method for calculating mean: sma, ema, tema, dema, zlema, kama
    :param atr_mean:  method for calculating mean for atr: sma, ema, tema, dema, zlema, kama
    :param as_frame: if true result is returned as DataFrame
    :return: mean, upper and lower bands
    """
    check_frame_columns(x, "open", "high", "low", "close")

    b = bollinger(x["close"], window, 0, mean)
    a = natr * atr(x, atr_window, atr_mean)
    m = b["Median"]
    return scols(*(m, m + a, m - a), names=["Median", "Upper", "Lower"])


def macd(x: pd.Series, fast=12, slow=26, signal=9, method="ema", signal_method="ema") -> pd.Series:
    """
    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices. The MACD is calculated by subtracting the 26-day slow moving average from the
    12-day fast MA. A nine-day MA of the MACD, called the "signal line", is then plotted on top of the MACD,
    functioning as a trigger for buy and sell signals.

    :param x: input data
    :param fast: fast MA period
    :param slow: slow MA period
    :param signal: signal MA period
    :param method: used moving averaging method (sma, ema, tema, dema, zlema, kama)
    :param signal_method: used method for averaging signal (sma, ema, tema, dema, zlema, kama)
    :return: macd signal
    """
    x_diff = smooth(x, method, fast) - smooth(x, method, slow)

    # averaging signal
    return smooth(x_diff, signal_method, signal).rename("macd")


def atr(x: pd.DataFrame, window=14, smoother="sma", percentage=False) -> pd.Series:
    """
    Average True Range indicator

    :param x: input series
    :param window: smoothing window size
    :param smoother: smooting method: sma, ema, zlema, tema, dema, kama
    :param percentage: if True ATR presented as percentage to close price: 100*ATR[i]/close[i]
    :return: average true range
    """
    check_frame_columns(x, "open", "high", "low", "close")

    close = x["close"].shift(1)
    h_l = abs(x["high"] - x["low"])
    h_pc = abs(x["high"] - close)
    l_pc = abs(x["low"] - close)
    tr = pd.concat((h_l, h_pc, l_pc), axis=1).max(axis=1)

    # smoothing
    a = smooth(tr, smoother, window).rename("atr")
    return (100 * a / close) if percentage else a


def rolling_atr(x, window, periods, smoother="sma") -> pd.Series:
    """
    Average True Range indicator calculated on rolling window

    :param x:
    :param window: windiw size as Timedelta or string
    :param periods: number periods for smoothing (applied if > 1)
    :param smoother: smoother (sma is default)
    :return: ATR
    """
    check_frame_columns(x, "open", "high", "low", "close")

    window = pd.Timedelta(window) if isinstance(window, str) else window
    tf_orig = pd.Timedelta(infer_series_frequency(x))

    if window < tf_orig:
        raise ValueError("window size must be great or equal to OHLC series timeframe !!!")

    wind_delta = window + tf_orig
    n_min_periods = wind_delta // tf_orig
    _c_1 = x.rolling(wind_delta, min_periods=n_min_periods).close.apply(lambda y: y[0])
    _l = x.rolling(window, min_periods=n_min_periods - 1).low.apply(lambda y: np.nanmin(y))
    _h = x.rolling(window, min_periods=n_min_periods - 1).high.apply(lambda y: np.nanmax(y))

    # calculate TR
    _tr = pd.concat((abs(_h - _l), abs(_h - _c_1), abs(_l - _c_1)), axis=1).max(axis=1)

    if smoother and periods > 1:
        _tr = smooth(_tr.ffill(), smoother, periods * max(1, (n_min_periods - 1)))

    return _tr


def trend_detector(
    data, period, nstd, method="bb", exit_on_mid=False, avg="sma", atr_period=12, atr_avg="kama"
) -> pd.DataFrame:
    """
    Trend detector method

    :param data: input series/frame
    :param period: bb period
    :param method: how to calculate range: bb (bollinger), bbatr (bollinger_atr),
                   hilo (rolling highest high ~ lower low -> SuperTrend regime)
    :param nstd: bb num of stds
    :param avg: averaging ma type
    :param exit_on_mid: trend is over when x crosses middle of bb
    :param atr_period: ATR period (used only when use_atr is True)
    :param atr_avg: ATR smoother (used only when use_atr is True)
    :return: frame
    """
    # flatten list lambda
    flatten = lambda l: [item for sublist in l for item in sublist]

    # just taking close prices
    x = data.close if isinstance(data, pd.DataFrame) else data

    if method in ["bb", "bb_atr", "bbatr", "bollinger", "bollinger_atr"]:
        if "atr" in method:
            _bb = bollinger_atr(data, period, atr_period, nstd, avg, atr_avg)
        else:
            _bb = bollinger(x, period, nstd, avg)

        midle, smax, smin = _bb["Median"], _bb["Upper"], _bb["Lower"]
    elif method in ["hl", "hilo", "hhll"]:
        check_frame_columns(data, ["high", "low", "close"])

        midle = (data["high"].rolling(period).max() + data["low"].rolling(period).min()) / 2
        smax = midle + nstd * atr(data, period)
        smin = midle - nstd * atr(data, period)
    else:
        raise ValueError(f"Unsupported method {method}")

    trend = (((x > smax.shift(1)) + 0.0) - ((x < smin.shift(1)) + 0.0)).replace(0, np.nan)

    # some special case if we want to exit when close is on the opposite side of median price
    if exit_on_mid:
        lom, him = ((x < midle).values, (x > midle).values)
        t = 0
        _t = trend.values.tolist()
        for i in range(len(trend)):
            t0 = _t[i]
            t = t0 if np.abs(t0) == 1 else t
            if (t > 0 and lom[i]) or (t < 0 and him[i]):
                t = 0
            _t[i] = t
        trend = pd.Series(_t, trend.index)
    else:
        trend = trend.ffill(axis=0).fillna(0.0)

    # making resulting frame
    m = x.to_frame().copy()
    m["trend"] = trend
    m["blk"] = (m.trend.shift(1) != m.trend).astype(int).cumsum()
    m["x"] = abs(m.trend) * (smax * (-m.trend + 1) - smin * (1 + m.trend)) / 2
    _g0 = m.reset_index().groupby(["blk", "trend"])
    m["x"] = flatten(abs(_g0["x"].apply(np.array).transform(np.minimum.accumulate).values))
    m["utl"] = m.x.where(m.trend > 0)
    m["dtl"] = m.x.where(m.trend < 0)

    # signals
    tsi = pd.DatetimeIndex(_g0["time"].apply(lambda x: x.values[0]).values)
    m["uts"] = m.loc[tsi].utl
    m["dts"] = m.loc[tsi].dtl

    return m.filter(items=["uts", "dts", "trend", "utl", "dtl"])


def denoised_trend(x: pd.DataFrame, period: int, window=0, mean: str = "kama", bar_returns: bool = True) -> pd.Series:
    """
    Returns denoised trend (T_i).

    ----

    R_i = C_i - O_i

    D_i = R_i - R_{i - period}

    P_i = sum_{k=i-period-1}^{i} abs(R_k)

    T_i = D_i * abs(D_i) / P_i

    ----

    :param x: OHLC dataset (must contain .open and .close columns)
    :param period: period of filtering
    :param window: smothing window size (default 0)
    :param mean: smoother
    :param bar_returns: if True use R_i = close_i - open_i
    :return: trend with removed noise
    """
    check_frame_columns(x, "open", "close")

    if bar_returns:
        ri = x.close - x.open
        di = x.close - x.open.shift(period)
    else:
        ri = x.close - x.close.shift(1)
        di = x.close - x.close.shift(period)
        period -= 1

    abs_di = abs(di)
    si = abs(ri).rolling(window=period + 1).sum()
    # for open - close there may be gaps
    if bar_returns:
        si = np.max(np.concatenate((abs_di.values[:, np.newaxis], si.values[:, np.newaxis]), axis=1), axis=1)
    filtered_trend = abs_di * (di / si)
    filtered_trend = filtered_trend.replace([np.inf, -np.inf], 0.0)

    if window > 0 and mean is not None:
        filtered_trend = smooth(filtered_trend, mean, window)

    return filtered_trend


def rolling_percentiles(
    x, window, pctls=(0, 1, 2, 3, 5, 10, 15, 25, 45, 50, 55, 75, 85, 90, 95, 97, 98, 99, 100)
) -> pd.DataFrame:
    """
    Calculates percentiles from x on rolling window basis

    :param x: series data
    :param window: window size
    :param pctls: percentiles
    :return: calculated percentiles as DataFrame indexed by time.
             Every pctl. is denoted as Qd (where d is taken from pctls)
    """
    r = nans((len(x), len(pctls)))
    i = window - 1

    for v in running_view(column_vector(x).flatten(), window):
        r[i, :] = np.percentile(v, pctls)
        i += 1

    return pd.DataFrame(r, index=x.index, columns=["Q%d" % q for q in pctls])


def trend_locker(y: pd.Series, order, window, lock_forward_window=1, use_projections=False) -> pd.DataFrame:
    """
    Trend locker indicator based on OLS.

    :param y: series data
    :param order: OLS order (1 - linear, 2 - squared etc)
    :param window: rolling window for regression
    :param lock_forward_window: how many forward points to lock (default is 1)
    :param use_projections: if need to get regression projections as well (False)
    :param as_frame: true if need to force converting result to DataFrame (True)
    :return: (residuals, projections, r2, betas)
    """

    if lock_forward_window < 1:
        raise ValueError("lock_forward_window must be positive non zero integer")

    n = window + lock_forward_window
    yy = running_view(column_vector(y).T[0], window=n)
    n_pts = len(y)
    resid = nans((n_pts, lock_forward_window))
    proj = nans((n_pts, lock_forward_window)) if use_projections else None
    r_sqr = nans(n_pts)
    betas = nans((n_pts, order + 1))

    for i, p in enumerate(yy):
        x = np.vander(np.linspace(-1, 1, n), order + 1)
        lr = OLS(p[:window], x[:window, :]).fit()

        r_sqr[window - 1 + i] = lr.rsquared
        betas[window - 1 + i, :] = lr.params

        pl = p[-lock_forward_window:]
        xl = x[-lock_forward_window:, :]
        fwd_prj = np.sum(lr.params * xl, axis=1)
        fwd_data = pl - fwd_prj

        # store forward data
        np.fill_diagonal(resid[window + i : n + i + 1, :], fwd_data)

        # if we asked for projections
        if use_projections:
            np.fill_diagonal(proj[window + i : n + i + 1, :], fwd_prj)

    # return pandas frame if input is series/frame
    y = pd.Series(y, name="X")
    y_idx = y.index
    f_res = pd.DataFrame(data=resid, index=y_idx, columns=["R%d" % i for i in range(1, lock_forward_window + 1)])
    f_prj = None
    if use_projections:
        f_prj = pd.DataFrame(data=proj, index=y_idx, columns=["L%d" % i for i in range(1, lock_forward_window + 1)])
    r = pd.DataFrame({"r2": r_sqr}, index=y_idx, columns=["r2"])
    betas_fr = pd.DataFrame(betas, index=y_idx, columns=["b%d" % i for i in range(order + 1)])
    return pd.concat((y, f_res, f_prj, r, betas_fr), axis=1)


def __slope_ols(x):
    x = x[~np.isnan(x)]
    xs = 2 * (x - min(x)) / (max(x) - min(x)) - 1
    m = OLS(xs, np.vander(np.linspace(-1, 1, len(xs)), 2)).fit()
    return m.params[0]


def __slope_angle(p, t):
    return 180 * np.arctan(p / t) / np.pi


def rolling_series_slope(x: pd.Series, period: int, method="ols", scaling="transform", n_bins=5) -> pd.Series:
    """
    Rolling slope indicator. May be used as trend indicator

    :param x: time series
    :param period: period for OLS window
    :param n_bins: number of bins used for scaling
    :param method: method used for metric of regression line slope: ('ols' or 'angle')
    :param scaling: how to scale slope 'transform' / 'binarize' / nothing
    :return: series slope metric
    """

    def __binarize(_x: pd.Series, n, limits=(None, None), center=False):
        n0 = n // 2 if center else 0
        _min = np.min(_x) if limits[0] is None else limits[0]
        _max = np.max(_x) if limits[1] is None else limits[1]
        return pd.Series(np.floor(n * (_x - _min) / (_max - _min)) - n0, index=_x.index)

    def __scaling_transform(x: pd.Series, n=5, need_round=True, limits=None):
        if limits is None:
            _lmax = max(abs(x))
            _lmin = -_lmax
        else:
            _lmax = max(limits)
            _lmin = min(limits)

        if need_round:
            ni = np.round(np.interp(x, (_lmin, _lmax), (-2 * n, +2 * n))) / 2
        else:
            ni = np.interp(x, (_lmin, _lmax), (-n, +n))
        return pd.Series(ni, index=x.index)

    if method == "ols":
        slp_meth = lambda z: __slope_ols(z)
        _lmts = (-1, 1)
    elif method == "angle":
        slp_meth = lambda z: __slope_angle(z[-1] - z[0], len(z))
        _lmts = (-90, 90)
    else:
        raise ValueError("Unknown Method %s" % method)

    _min_p = period
    if isinstance(period, str):
        _min_p = pd.Timedelta(period).days

    roll_slope = x.rolling(period, min_periods=_min_p).apply(slp_meth)

    if scaling == "transform":
        return __scaling_transform(roll_slope, n=n_bins, limits=_lmts)
    elif scaling == "binarize":
        return __binarize(roll_slope, n=(n_bins - 1) * 4, limits=_lmts, center=True) / 2

    return roll_slope


def adx(ohlc: pd.DataFrame, period: int, smoother="kama") -> pd.DataFrame:
    """
    Average Directional Index.

    ADX = 100 * MA(abs((+DI - -DI) / (+DI + -DI)))

    Where:
    -DI = 100 * MA(-DM) / ATR
    +DI = 100 * MA(+DM) / ATR

    +DM: if UPMOVE > DWNMOVE and UPMOVE > 0 then +DM = UPMOVE else +DM = 0
    -DM: if DWNMOVE > UPMOVE and DWNMOVE > 0 then -DM = DWNMOVE else -DM = 0

    DWNMOVE = L_{t-1} - L_t
    UPMOVE = H_t - H_{t-1}

    :param ohlc: DataFrame with ohlc data
    :param period: indicator period
    :param smoother: smoothing function (kama is default)
    :param as_frame: set to True if DataFrame needed as result (default false)
    :return: adx, DIp, DIm or DataFrame
    """
    check_frame_columns(ohlc, "open", "high", "low", "close")

    h, l = ohlc["high"], ohlc["low"]
    _atr = atr(ohlc, period, smoother=smoother)

    Mu, Md = h.diff(), -l.diff()
    DMp = Mu * (((Mu > 0) & (Mu > Md)) + 0)
    DMm = Md * (((Md > 0) & (Md > Mu)) + 0)
    DIp = 100 * smooth(DMp, smoother, period) / _atr
    DIm = 100 * smooth(DMm, smoother, period) / _atr
    _adx = 100 * smooth(abs((DIp - DIm) / (DIp + DIm)), smoother, period)

    return pd.concat((_adx.rename("ADX"), DIp.rename("DIp"), DIm.rename("DIm")), axis=1)


def rsi(x, periods, smoother=sma) -> pd.Series:
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


def pivot_point(data: pd.DataFrame, method="classic", timeframe="D", timezone=None) -> pd.DataFrame:
    """
    Pivot points indicator based for daily/weekly/monthly levels.
    It supports 'classic', 'woodie' and 'camarilla' species.
    """
    if timeframe not in ["D", "W", "M"]:
        raise ValueError(f"Unsupported pivots timeframe {timeframe}: only 'D', 'W', 'M' allowed !")

    tf_resample = f"1{timeframe}"
    x: pd.DataFrame | dict = ohlc_resample(data, tf_resample, resample_tz=timezone)

    pp = pd.DataFrame()
    if method == "classic":
        pvt = (x["high"] + x["low"] + x["close"]) / 3
        _range = x["high"] - x["low"]

        pp["R4"] = pvt + 3 * _range
        pp["R3"] = pvt + 2 * _range
        pp["R2"] = pvt + _range
        pp["R1"] = pvt * 2 - x["low"]
        pp["P"] = pvt
        pp["S1"] = pvt * 2 - x["high"]
        pp["S2"] = pvt - _range
        pp["S3"] = pvt - 2 * _range
        pp["S4"] = pvt - 3 * _range

        # rearrange
        pp = pp[["R4", "R3", "R2", "R1", "P", "S1", "S2", "S3", "S4"]]

    elif method == "woodie":
        pvt = (x.high + x.low + x.open + x.open) / 4
        _range = x.high - x.low

        pp["R3"] = x.high + 2 * (pvt - x.low)
        pp["R2"] = pvt + _range
        pp["R1"] = pvt * 2 - x.low
        pp["P"] = pvt
        pp["S1"] = pvt * 2 - x.high
        pp["S2"] = pvt - _range
        pp["S3"] = x.low + 2 * (x.high - pvt)
        pp = pp[["R3", "R2", "R1", "P", "S1", "S2", "S3"]]

    elif method == "camarilla":
        """
        R4 = C + RANGE * 1.1/2
        R3 = C + RANGE * 1.1/4
        R2 = C + RANGE * 1.1/6
        R1 = C + RANGE * 1.1/12
        PP = (HIGH + LOW + CLOSE) / 3
        S1 = C - RANGE * 1.1/12
        S2 = C - RANGE * 1.1/6
        S3 = C - RANGE * 1.1/4
        S4 = C - RANGE * 1.1/2
        """
        pvt = (x.high + x.low + x.close) / 3
        _range = x.high - x.low

        pp["R4"] = x.close + _range * 1.1 / 2
        pp["R3"] = x.close + _range * 1.1 / 4
        pp["R2"] = x.close + _range * 1.1 / 6
        pp["R1"] = x.close + _range * 1.1 / 12
        pp["P"] = pvt
        pp["S1"] = x.close - _range * 1.1 / 12
        pp["S2"] = x.close - _range * 1.1 / 6
        pp["S3"] = x.close - _range * 1.1 / 4
        pp["S4"] = x.close - _range * 1.1 / 2
        pp = pp[["R4", "R3", "R2", "R1", "P", "S1", "S2", "S3", "S4"]]
    else:
        raise ValueError("Unknown method %s. Available methods are: 'classic', 'woodie', 'camarilla'" % method)

    pp.index = pp.index + pd.Timedelta("1D")
    return data.combine_first(pp).ffill(axis=0)[pp.columns]


def intraday_min_max(data: pd.DataFrame, timezone="UTC") -> pd.DataFrame:
    """
    Intradeay min and max values
    :param data: ohlcv series
    :param timezone: timezone (default EET) used for find day's start/end (EET for forex data)
    :return: series with min and max values intraday
    """
    check_frame_columns(data, "open", "high", "low", "close")

    def _day_min_max(d):
        _d_min = np.minimum.accumulate(d.low)
        _d_max = np.maximum.accumulate(d.high)
        return scols(_d_min, _d_max, keys=["Min", "Max"])

    source_tz = data.index.tz
    if not source_tz:
        x = data.tz_localize("GMT")
    else:
        x = data

    x = x.tz_convert(timezone)
    return x.groupby(x.index.date).apply(_day_min_max).tz_convert(source_tz)


def stochastic(x: pd.Series | pd.DataFrame, period: int, smooth_period: int, smoother="sma") -> pd.DataFrame:
    """
    Classical stochastic oscillator indicator
    :param x: series or OHLC dataframe
    :param period: indicator's period
    :param smooth_period: period of smoothing
    :param smoother: smoothing method (sma by default)
    :return: K and D series as DataFrame
    """
    # in case we received HLC data
    if has_columns(x, "close", "high", "low"):
        hi, li, xi = x["high"], x["low"], x["close"]
    else:
        hi, li, xi = x, x, x

    hh = hi.rolling(period).max()
    ll = li.rolling(period).min()
    k = 100 * (xi - ll) / (hh - ll)
    d = smooth(k, smoother, smooth_period)
    return scols(k, d, names=["K", "D"])


@njit
def _laguerre_calc(xx, g):
    l0, l1, l2, l3, f = np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx))
    for i in range(1, len(xx)):
        l0[i] = (1 - g) * xx[i] + g * l0[i - 1]
        l1[i] = -g * l0[i] + l0[i - 1] + g * l1[i - 1]
        l2[i] = -g * l1[i] + l1[i - 1] + g * l2[i - 1]
        l3[i] = -g * l2[i] + l2[i - 1] + g * l3[i - 1]
        f[i] = (l0[i] + 2 * l1[i] + 2 * l2[i] + l3[i]) / 6
    return f


def laguerre_filter(x, gamma=0.8) -> pd.Series:
    """
    Laguerre 4 pole IIR filter
    """
    return pd.Series(_laguerre_calc(x.values.flatten(), gamma), x.index)


@njit
def _lrsi_calc(xx, g):
    l0, l1, l2, l3, f = np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx))
    for i in range(1, len(xx)):
        l0[i] = (1 - g) * xx[i] + g * l0[i - 1]
        l1[i] = -g * l0[i] + l0[i - 1] + g * l1[i - 1]
        l2[i] = -g * l1[i] + l1[i - 1] + g * l2[i - 1]
        l3[i] = -g * l2[i] + l2[i - 1] + g * l3[i - 1]

        _cu, _cd = 0, 0
        _d0 = l0[i] - l1[i]
        _d1 = l1[i] - l2[i]
        _d2 = l2[i] - l3[i]

        if _d0 >= 0:
            _cu = _d0
        else:
            _cd = np.abs(_d0)

        if _d1 >= 0:
            _cu += _d1
        else:
            _cd += np.abs(_d1)

        if _d2 >= 0:
            _cu += _d2
        else:
            _cd += np.abs(_d2)

        f[i] = 100 * _cu / (_cu + _cd) if (_cu + _cd) != 0 else 0

    return f


def lrsi(x, gamma=0.5) -> pd.Series:
    """
    Laguerre RSI
    """
    return pd.Series(_lrsi_calc(x.values.flatten(), gamma), x.index)


@njit
def calc_ema_time(t, vv, period, min_time_quant, with_correction=True):
    index = np.empty(len(vv) - 1, dtype=np.int64)
    values = np.empty(len(vv) - 1, dtype=np.float64)
    dt = np.diff(t)
    dt[dt == 0] = min_time_quant
    a = dt / period
    u = np.exp(-a)
    _ep = vv[0]

    if with_correction:
        v = (1 - u) / a
        c1 = v - u
        c2 = 1 - v
        for i in range(0, len(vv) - 1):
            _ep = u[i] * _ep + c1[i] * vv[i] + c2[i] * vv[i + 1]
            index[i] = t[i + 1]
            values[i] = _ep
    else:
        v = 1 - u
        for i in range(0, len(vv) - 1):
            _ep = _ep + v[i] * (vv[i + 1] - _ep)
            index[i] = t[i + 1]
            values[i] = _ep
    return index, values


def ema_time(
    x: pd.Series, period: str | pd.Timedelta, min_time_quant=pd.Timedelta("1ms"), with_correction=True
) -> pd.Series:
    """
    EMA on non consistent time series

    https://stackoverflow.com/questions/1023860/exponential-moving-average-sampled-at-varying-times
    """
    if not isinstance(x, pd.Series):
        raise ValueError("Input series must be instance of pandas Series class")

    t = x.index.values
    vv = x.values

    if isinstance(period, str):
        period = pd.Timedelta(period)

    index, values = calc_ema_time(t.astype("int64"), vv, period.value, min_time_quant.value, with_correction)

    old_ser_name = "UnknownSeries" if x.name is None else x.name
    res = pd.Series(values, pd.to_datetime(index), name="EMAT_%d_sec_%s" % (period.seconds, old_ser_name))
    res = res.loc[~res.index.duplicated(keep="first")]
    return res


@njit
def _rolling_rank(x, period, pctls):
    x = np.reshape(x, (x.shape[0], -1)).flatten()
    r = nans((len(x)))
    for i in range(period, len(x)):
        v = x[i - period : i]
        r[i] = np.argmax(np.sign(np.append(np.percentile(v, pctls), np.inf) - x[i]))
    return r


def rolling_rank(x, period, pctls=(25, 50, 75)):
    """
    Calculates percentile rank (number of percentile's range) on rolling window basis from series of data

    :param x: series or frame of data
    :param period: window size
    :param pctls: percentiles (25,50,75) are default.
                  Function returns 0 for values hit 0...25, 1 for 25...50, 2 for 50...75 and 3 for 75...100
                  on rolling window basis
    :return: series/frame of ranks
    """
    if period > len(x):
        raise ValueError(f"Period {period} exceeds number of data records {len(x)} ")

    if isinstance(x, pd.DataFrame):
        z = pd.DataFrame.from_dict({c: rolling_rank(s, period, pctls) for c, s in x.items()})
    elif isinstance(x, pd.Series):
        z = pd.Series(_rolling_rank(np.ascontiguousarray(x.values), period, pctls), x.index, name=x.name)
    else:
        z = _rolling_rank(np.ascontiguousarray(x), period, pctls)
    return z


def rolling_vwap(ohlc, period):
    """
    Calculate rolling volume weighted average price using specified rolling period
    """
    check_frame_columns(ohlc, "close", "volume")

    if period > len(ohlc):
        raise ValueError(f"Period {period} exceeds number of data records {len(ohlc)} ")

    def __rollsum(x, window):
        rs = pd.DataFrame(rolling_sum(column_vector(x.values.copy()), window), index=x.index)
        rs[rs < 0] = np.nan
        return rs

    return __rollsum((ohlc.volume * ohlc.close), period) / __rollsum(ohlc.volume, period)


def pivots_highs_lows(
    high: pd.Series,
    low: pd.Series,
    nf_before: int,
    nf_after: int,
    index_on_observed_time=True,
    align_with_index=False,
    include_actual_time=False,
) -> pd.DataFrame:
    """
    Calculates pivots indicator (pivot highs/lows) for asymmetric lookback/lookahead.

    Example:
    >>> high = pd.Series([1, 2, 3, 4, 5, 6, 5, 4, 3])
    >>> low = pd.Series([5, 4, 3, 2, 3, 3, 1, 3, 4])
    >>> pta.pivots_highs_lows(high, low, 3, 2, index_on_observed_time=True, align_with_index=False)

    :param high: high series
    :param low: low series
    :param nf_before: number of bars to look backward
    :param nf_after: number of bars to look forward
    :param index_on_observed_time: If True, pivot is placed at the bar where it becomes visible
    :param align_with_index: If True, result is reindexed to input data index
    :param include_actual_time: If True, include time of bar where pivot occured in result
    :return: DataFrame with 'U' (upper pivots), 'L' (lower pivots). Result is multi-index if include_actual_time is True
    """
    if len(high) != len(low):
        raise ValueError("High and low series must have the same length !")

    if any(high.index != low.index):
        raise ValueError("High and low series must have the same index !")

    idx = high.index

    # For upper fractals: current high must be greater than all highs in window
    highs_before = pd.concat([high.shift(i) for i in range(1, nf_before + 1)], axis=1)
    highs_after = pd.concat([high.shift(-i) for i in range(1, nf_after + 1)], axis=1)
    upper_cond = (highs_before.lt(high, axis=0).all(axis=1)) & (highs_after.lt(high, axis=0).all(axis=1))
    upF = pd.Series(+1, idx[upper_cond])

    # For lower fractals: current low must be less than all lows in window
    lows_before = pd.concat([low.shift(i) for i in range(1, nf_before + 1)], axis=1)
    lows_after = pd.concat([low.shift(-i) for i in range(1, nf_after + 1)], axis=1)
    lower_cond = (lows_before.gt(low, axis=0).all(axis=1)) & (lows_after.gt(low, axis=0).all(axis=1))
    dwF = pd.Series(-1, idx[lower_cond])

    # Fractal is observed only after the 'future' bars, so shift by nf_after if actual_time
    shift_forward = nf_after if index_on_observed_time else 0

    ht = high.loc[upF.index].reindex(idx).shift(shift_forward)
    lt = low.loc[dwF.index].reindex(idx).shift(shift_forward)

    if not align_with_index:
        ht = ht.dropna()
        lt = lt.dropna()

    if include_actual_time:
        _l_times = pd.Series(dwF.index, index=dwF.index).reindex(idx).shift(shift_forward)
        _h_times = pd.Series(upF.index, index=upF.index).reindex(idx).shift(shift_forward)
        if not align_with_index:
            _h_times = _h_times.dropna()
            _l_times = _l_times.dropna()

        # return pd.DataFrame({"U": ht, "L": lt, "U_time": _h_times, "L_time": _l_times})
        return scols(
            pd.DataFrame({"price": ht, "time": _h_times}),
            pd.DataFrame({"price": lt, "time": _l_times}),
            keys=["U", "L"],
        )

    return pd.DataFrame({"U": ht, "L": lt})


def fractals(
    data: pd.DataFrame,
    nf: int,
    index_on_observed_time: bool = True,
    align_with_index: bool = False,
    include_actual_time=False,
) -> pd.DataFrame:
    """
    Calculates fractals indicator

    :param data: OHLC bars series
    :param nf: fractals lookback/foreahed parameter
    :param index_on_observed_time: if true fractals timestamps at bars where they would be observed
    :param align_with_index: if true result will be reindexed as input ohlc data
    :param include_actual_time: If True, include time of bar where fractals occured in result
    :return: pd.DataFrame with U (upper) and L (lower) fractals columns
    """
    check_frame_columns(data, "high", "low")
    return pivots_highs_lows(
        data["high"],
        data["low"],
        nf,
        nf,
        index_on_observed_time,
        align_with_index,
        include_actual_time=include_actual_time,
    )


@njit
def _jma(x, period, phase, power):
    x = x.astype(np.float64)

    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = np.power(beta, power)
    phase = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5

    for i in range(0, x.shape[1]):
        xs = x[:, i]
        nan_start = np.where(~np.isnan(xs))[0][0]
        r = np.zeros(xs.shape[0])
        det0 = det1 = 0
        ma1 = ma2 = jm = xs[0]

        for k, xi in enumerate(xs):
            ma1 = (1 - alpha) * xi + alpha * ma1
            det0 = (xi - ma1) * (1 - beta) + beta * det0
            ma2 = ma1 + phase * det0
            det1 = (ma2 - jm) * np.power(1 - alpha, 2) + np.power(alpha, 2) * det1
            jm = jm + det1
            r[k] = jm
        x[:, i] = np.concatenate((nans(nan_start), r))
    return x


@__wrap_dataframe_decorator
def jma(x, period, phase=0, power=2):
    """
    Jurik MA (code from https://www.tradingview.com/script/nZuBWW9j-Jurik-Moving-Average/)
    :param x: data
    :param period: period
    :param phase: phase
    :param power: power
    :return: jma
    """
    x = column_vector(x)
    if len(x) < period:
        raise ValueError("Not enough data for calculate jma !")

    return _jma(x, period, phase, power)


def super_trend(
    data, length: int = 22, mult: float = 3, src: str = "hl2", wicks: bool = True, atr_smoother="sma"
) -> pd.DataFrame:
    """
    SuperTrend indicator (implementation from https://www.tradingview.com/script/VLWVV7tH-SuperTrend/)

    :param data: OHLC data
    :param length: ATR Period
    :param mult: ATR Multiplier
    :param src: Source: close, hl2, hlc3, ohlc4. For example hl2 = (high + low) / 2, etc
    :param wicks: Take Wicks
    :param atr_smoother: ATR smoothing function (default sma)
    :return:
    """
    check_frame_columns(data, "open", "high", "low", "close")

    def calc_src(data, src):
        if src == "close":
            return data["close"]
        elif src == "hl2":
            return (data["high"] + data["low"]) / 2
        elif src == "hlc3":
            return (data["high"] + data["low"] + data["close"]) / 3
        elif src == "ohlc4":
            return (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        else:
            raise ValueError("unsupported src: %s" % src)

    atr_data = abs(mult) * atr(data, length, smoother=atr_smoother)
    src_data = calc_src(data, src)
    high_price = data["high"] if wicks else data["close"]
    low_price = data["low"] if wicks else data["close"]
    doji4price = (data["open"] == data["close"]) & (data["open"] == data["low"]) & (data["open"] == data["high"])

    p_high_price = high_price.shift(1)
    p_low_price = low_price.shift(1)

    longstop = src_data - atr_data
    shortstop = src_data + atr_data

    prev_longstop = np.nan
    prev_shortstop = np.nan

    longstop_d = {}
    shortstop_d = {}
    for i, ls, ss, lp, hp, d4 in zip(
        src_data.index, longstop.values, shortstop.values, p_low_price.values, p_high_price.values, doji4price.values
    ):
        # longs
        if np.isnan(prev_longstop):
            prev_longstop = ls

        if ls > 0:
            if d4:
                longstop_d[i] = prev_longstop
            else:
                longstop_d[i] = max(ls, prev_longstop) if lp > prev_longstop else ls
        else:
            longstop_d[i] = prev_longstop

        prev_longstop = longstop_d[i]

        # shorts
        if np.isnan(prev_shortstop):
            prev_shortstop = ss

        if ss > 0:
            if d4:
                shortstop_d[i] = prev_shortstop
            else:
                shortstop_d[i] = min(ss, prev_shortstop) if hp < prev_shortstop else ss
        else:
            shortstop_d[i] = prev_shortstop

        prev_shortstop = shortstop_d[i]

    longstop = pd.Series(longstop_d)
    shortstop = pd.Series(shortstop_d)

    direction = pd.Series(np.nan, src_data.index)
    direction.iloc[(low_price < longstop.shift(1))] = -1
    direction.iloc[(high_price > shortstop.shift(1))] = 1
    # direction.fillna(method='ffill', inplace=True) # deprecated
    direction.ffill(inplace=True)

    longstop_res = pd.Series(np.nan, src_data.index)
    shortstop_res = pd.Series(np.nan, src_data.index)

    shortstop_res[direction == -1] = shortstop[direction == -1]
    longstop_res[direction == 1] = longstop[direction == 1]

    return scols(longstop_res, shortstop_res, direction, names=["utl", "dtl", "trend"])


def choppiness(
    data,
    period,
    upper=61.8,
    lower=38.2,
    volatility_estimator="t",
    volume_adjusting=False,
    identification="strong",
    with_raw_indicator=False,
) -> pd.Series:
    """
    Calculate market choppiness index using volatility-based formula.

    Volatile market leads to false breakouts and does not respect support/resistance levels (being choppy).
    We cannot know whether we are in a trend or in a range.

    Values above 61.8% indicate a choppy market that is bound to breakout. We should be ready for some directional movement.
    Values below 38.2% indicate a strong trending market that is bound to stabilize.

    When volatility_estimator="t" (true range), this gives the classical Dreiss choppiness formula.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data frame containing 'open', 'high', 'low', 'close', 'volume' columns
    period : int
        Lookback period for calculations
    upper : float, default 61.8
        Upper threshold - values above indicate choppy market
    lower : float, default 38.2
        Lower threshold - values below indicate trending market
    volatility_estimator : str, default 't'
        Method used for volatility calculation (see volatility() function)
    volume_adjusting : bool, default False
        If True, adjusts volatility by relative volume changes
    identification : str, default 'strong'
        Identification method:
        - 'mid': Identifies three states: choppy (1) when crossing upper threshold, trending (-1) when crossing lower threshold,
                 and neutral (0) when between thresholds
        - 'strong': Binary classification focused on trending - returns 1 when entering trending regime (crossing below lower threshold),
                    0 when exiting trending regime (crossing above lower threshold)
        - 'weak': Binary classification focused on choppiness - returns 1 when entering choppy regime (crossing above upper threshold),
                 0 when entering trending regime (crossing below lower threshold)
    with_raw_indicator : bool, default False
        If True, returns the raw indicator value instead of the classification
    Returns
    -------
    pd.Series
        Boolean series where True indicates choppy market and False indicates trending market
    """

    if volume_adjusting:
        check_frame_columns(data, "open", "high", "low", "close", "volume")
        xr = data[["open", "high", "low", "close", "volume"]]
    else:
        check_frame_columns(data, "open", "high", "low", "close")
        xr = data[["open", "high", "low", "close"]]

    rng = (
        xr["high"].rolling(window=period, min_periods=period).max()
        - xr["low"].rolling(window=period, min_periods=period).min()
    )

    if volatility_estimator == "atr":
        a = atr(xr, period, smoother="ema")
        vol = pd.Series(rolling_sum(column_vector(a.copy()), period).flatten(), a.index).replace(0, np.nan)
    else:
        vol = period * volatility(
            xr, period, method=volatility_estimator, volume_adjusting=volume_adjusting, percentage=False
        )
    ci = 100 * (1 / np.log(period)) * np.log(vol / rng)

    f0 = pd.Series(np.nan, ci.index, dtype=int)
    match identification:
        case "mid":
            f0[(ci > lower) & (ci < upper)] = 0
            f0[ci > upper] = 1
            f0[ci < lower] = -1
        case "strong":
            f0[ci > lower] = 1
            f0[ci < lower] = 0
        case "weak":
            f0[ci > upper] = 1
            f0[ci < lower] = 0
        case _:
            raise ValueError(f"Invalid identification: {identification}")

    return f0 if not with_raw_indicator else ci


@njit
def __psar(close, high, low, iaf, maxaf):
    """
    PSAR loop in numba
    """
    length = len(close)
    psar = close[0 : len(close)].copy()

    psarbull, psarbear = np.ones(length) * np.nan, np.ones(length) * np.nan

    af, ep, hp, lp, bull = iaf, low[0], high[0], low[0], True

    for i in range(2, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    return psar, psarbear, psarbull


def psar(ohlc, iaf=0.02, maxaf=0.2) -> pd.DataFrame:
    """
    Parabolic SAR indicator
    """
    check_frame_columns(ohlc, "high", "low", "close")

    # do cycle in numba
    psar_i, psarbear, psarbull = __psar(ohlc["close"].values, ohlc["high"].values, ohlc["low"].values, iaf, maxaf)

    return pd.DataFrame({"psar": psar_i, "up": psarbear, "down": psarbull}, index=ohlc.index)


@__wrap_dataframe_decorator
def fdi(x: pd.Series | pd.DataFrame, e_period=30) -> np.ndarray:
    """
    The Fractal Dimension Index determines the amount of market volatility. Value of 1.5 suggests the market is
    acting in a completely random fashion. As the indicator deviates from 1.5, the opportunity for earning profits
    is increased in proportion to the amount of deviation.

    The indicator < 1.5 when the market is in a trend. And it > 1.5 when there is a high volatility.
    When the FDI crosses 1.5 upward it means that a trend is finishing, the market becomes erratic and
    a high volatility is present.

    For more information, see
    http://www.forex-tsd.com/suggestions-trading-systems/6119-tasc-03-07-fractal-dimension-index.html

    :param x: input series (pd.Series)
    :param e_period: period of indicator (30 is default)
    """
    len_shape = 2
    data = x.copy()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    if len(data.shape) == 1:
        len_shape = 1
        data = data.reshape(data.shape[0], 1)
    fdi_result: np.ndarray | None = None
    for work_data in running_view(data, e_period, 0):
        if fdi_result is None:
            fdi_result = _fdi(work_data, e_period, len_shape)
        else:
            fdi_result = np.vstack([fdi_result, _fdi(work_data, e_period, len_shape)])
    fdi_result[np.isinf(fdi_result)] = 0
    fdi_result = np.vstack((np.full([e_period, x.shape[-1] if len(x.shape) == 2 else 1], np.nan), fdi_result[1:]))
    return fdi_result


@njit
def _fdi(work_data, e_period=30, shape_len=1) -> np.ndarray:
    idx = np.argmax(work_data, -1)
    flat_idx = np.arange(work_data.size, step=work_data.shape[-1]) + idx.ravel()
    price_max = work_data.ravel()[flat_idx].reshape(*work_data.shape[:-1])
    idx = np.argmin(work_data, -1)
    flat_idx = np.arange(work_data.size, step=work_data.shape[-1]) + idx.ravel()
    price_min = work_data.ravel()[flat_idx].reshape(*work_data.shape[:-1])

    length = 0

    if shape_len == 1:
        diffs = (work_data - price_min) / (price_max - price_min)
        length = np.power(np.power(np.diff(diffs).T, 2.0) + (1.0 / np.power(e_period, 2.0)), 0.5)
    else:
        diffs = (work_data.T - price_min) / (price_max - price_min)
        length = np.power(np.power(np.diff(diffs.T).T, 2.0) + (1.0 / np.power(e_period, 2.0)), 0.5)
    length = np.sum(length[1:], 0)

    fdi_vs = 1.0 + (np.log(length) + np.log(2.0)) / np.log(2 * e_period)

    return fdi_vs


@njit
def __mcginley(xs, es, period):
    g = 0.0
    gs = []
    for x, e in zip(xs, es):
        if g == 0.0:
            g = e
        g = g + (x - g) / (period * np.power(x / g, 4))
        gs.append(g)
    return gs


def mcginley(xs: pd.Series, period: int) -> pd.Series:
    """
    McGinley dynamic moving average
    """
    x = column_vector(xs)
    es0 = ema(xs, period, init_mean=False)
    return pd.Series(__mcginley(x.reshape(1, -1)[0], es0.reshape(1, -1)[0], period), index=xs.index)


def stdev(x, window: int) -> pd.Series:
    """
    Standard deviation of x on rolling basis period (as in tranding view)
    """
    x = x.copy()
    a = smooth(np.power(x, 2), "sma", window)
    sm = pd.Series(rolling_sum(column_vector(x), window).reshape(1, -1)[0], x.index)
    b = np.power(sm, 2) / np.power(window, 2)
    return np.sqrt(a - b)


def waexplosion(
    data: pd.DataFrame,
    fastLength=20,
    slowLength=40,
    channelLength=20,
    sensitivity=150,
    mult=2.0,
    source="close",
    tuning_F1=3.7,
    tuning_atr_period=100,
    tuning_macd_smoother="ema",
) -> pd.DataFrame:
    """
    Waddah Attar Explosion indicator (version from TradingView)

    Example:
    - - - -

    wae = waexplosion(ohlc)
    LookingGlass(ohlc, {
        'WAE': [
            'dots', wae.dead_zone, 'line', wae.explosion, 'area', 'green', wae.trend_up, 'orange', 'area', wae.trend_down
            ],
        }).look('2023-Feb-01', '2023-Feb-17').hover()

    """
    check_frame_columns(data, "open", "high", "low", "close")
    x = data[source]

    dead_zone = tuning_F1 * atr(data, 2 * tuning_atr_period + 1, "ema")
    macd1 = smooth(x, tuning_macd_smoother, fastLength) - smooth(x, tuning_macd_smoother, slowLength)
    t1 = sensitivity * macd1.diff()

    e1 = 2 * mult * stdev(x, channelLength)
    trend_up = t1.where(t1 > 0, 0)
    trend_dw = -t1.where(t1 < 0, 0)

    return scols(dead_zone, e1, trend_up, trend_dw, names=["dead_zone", "explosion", "trend_up", "trend_down"])


def rad_indicator(x: pd.DataFrame, period: int, mult: float = 2, smoother="sma") -> pd.DataFrame:
    """
    RAD chandelier indicator
    """
    check_frame_columns(x, "open", "high", "low", "close")

    a = atr(x, period, smoother=smoother)

    hh = x.high.rolling(window=period).max()
    ll = x.low.rolling(window=period).min()

    rad_long = hh - a * mult
    rad_short = ll + a * mult

    brk_d = x[(x.close.shift(1) > rad_long.shift(1)) & (x.close < rad_long)].index
    brk_u = x[(x.close.shift(1) < rad_short.shift(1)) & (x.close > rad_short)].index

    sw = pd.Series(np.nan, x.index)
    sw.loc[brk_d] = +1
    sw.loc[brk_u] = -1
    sw = sw.ffill()

    radU = rad_short[sw[sw > 0].index]
    radD = rad_long[sw[sw < 0].index]
    # rad = srows(radU, radD)

    # stop level
    mu, md = -np.inf, np.inf
    rs = {}
    for t, s in zip(sw.index, sw.values):
        if s < 0:
            mu = max(mu, rad_long.loc[t])
            rs[t] = mu
            md = np.inf
        if s > 0:
            md = min(md, rad_short.loc[t])
            rs[t] = md
            mu = -np.inf

    return scols(pd.Series(rs), rad_long, rad_short, radU, radD, names=["rad", "long", "short", "U", "D"])


@njit
def __calc_qqe_mod_core_fast(rs_index_values, newlongband_values, newshortband_values) -> list:
    ri1, lb1, sb1, tr1, lb2, sb2 = np.nan, 0, 0, np.nan, np.nan, np.nan
    c1 = 0
    fast_atr_rsi_tl = []

    for ri, nl, ns in zip(rs_index_values, newlongband_values, newshortband_values):
        if not np.isnan(ri1):
            c1 = ((ri1 <= lb2) & (ri > lb1)) | ((ri1 >= lb2) & (ri < lb1))
            tr = ((ri1 <= sb2) & (ri > sb1)) | ((ri1 >= sb2) & (ri < sb1))
            tr1 = 1 if tr else -1 if c1 else tr1

            lb2, sb2 = lb1, sb1
            lb1 = max(lb1, nl) if (ri1 > lb1) and ri > lb1 else nl
            sb1 = min(sb1, ns) if (ri1 < sb1) and ri < sb1 else ns

            fast_atr_rsi_tl.append(lb1 if tr1 == 1 else sb1)
        else:
            fast_atr_rsi_tl.append(np.nan)

        ri1 = ri

    return fast_atr_rsi_tl


def _calc_qqe_mod_core(src: pd.Series, rsi_period, sf, qqe) -> Tuple[pd.Series, pd.Series]:
    to_ser = lambda xs, name=None: pd.Series(xs, name=name)
    wilders_period = rsi_period * 2 - 1

    rsi_i = rsi(src, wilders_period, smoother=ema)
    rsi_ma = smooth(rsi_i, "ema", sf)

    atr_rsi = abs(rsi_ma.diff())
    ma_atr_rsi = smooth(atr_rsi, "ema", wilders_period)
    dar = smooth(ma_atr_rsi, "ema", wilders_period) * qqe

    delta_fast_atr_rsi = dar
    rs_index = rsi_ma

    newshortband = rs_index + delta_fast_atr_rsi
    newlongband = rs_index - delta_fast_atr_rsi

    fast_atr_rsi_tl = __calc_qqe_mod_core_fast(rs_index.values, newlongband.values, newshortband.values)
    fast_atr_rsi_tl = pd.Series(fast_atr_rsi_tl, index=rs_index.index, name="fast_atr_rsi_tl")

    # - old approach - 10 times slower
    # ri1, lb1, sb1, tr1, lb2, sb2 = np.nan, 0, 0, np.nan, np.nan, np.nan
    # c1 = 0
    # fast_atr_rsi_tl = {}

    # for t, ri, nl, ns in zip(rs_index.index,
    #                          rs_index.values, newlongband.values, newshortband.values):
    #     if not np.isnan(ri1):
    #         c1 = ((ri1 <= lb2) & (ri > lb1)) | ((ri1 >= lb2) & (ri < lb1))
    #         tr = ((ri1 <= sb2) & (ri > sb1)) | ((ri1 >= sb2) & (ri < sb1))
    #         tr1 = 1 if tr else -1 if c1 else tr1

    #         lb2, sb2 = lb1, sb1
    #         lb1 = max(lb1, nl) if (ri1 > lb1) and ri > lb1 else nl
    #         sb1 = min(sb1, ns) if (ri1 < sb1) and ri < sb1 else ns

    #         fast_atr_rsi_tl[t] = lb1 if tr1 == 1 else sb1

    #     # ri1, nl1, ns1 = ri, nl, ns
    #     ri1 = ri
    # fast_atr_rsi_tl = to_ser(fast_atr_rsi_tl, name='fast_atr_rsi_tl')

    return rsi_ma, fast_atr_rsi_tl


def qqe_mod(
    data: pd.DataFrame,
    rsi_period=6,
    sf=5,
    qqe=3,
    source="close",
    length=50,
    mult=0.35,
    source2="close",
    rsi_period2=6,
    sf2=5,
    qqe2=1.61,
    threshhold2=3,
) -> pd.DataFrame:
    """
    QQE_MOD indicator
    """
    check_frame_columns(data, "open", "high", "low", "close")

    src = data[source]
    rsi_ma, fast_atr_rsi_tl = _calc_qqe_mod_core(src, rsi_period, sf, qqe)

    basis = smooth(fast_atr_rsi_tl - 50, "sma", length)
    dev = mult * stdev(fast_atr_rsi_tl - 50, length)
    upper = basis + dev
    lower = basis - dev

    src2 = data[source2]
    rsi_ma2, fast_atr_rsi_tl2 = _calc_qqe_mod_core(src2, rsi_period2, sf2, qqe2)

    qqe_line = fast_atr_rsi_tl2 - 50
    histo1 = rsi_ma - 50
    histo2 = rsi_ma2 - 50

    m = scols(histo1, histo2, upper, lower, names=["H1", "H2", "U", "L"])
    _gb_cond = (m.H2 > threshhold2) & (m.H1 > m.U)
    _rb_cond = (m.H2 < -threshhold2) & (m.H1 < m.L)
    _sb_cond = ((histo2 > threshhold2) | (histo2 < -threshhold2)) & ~_gb_cond & ~_rb_cond
    green_bars = m[_gb_cond].H2
    red_bars = m[_rb_cond].H2
    silver_bars = m[_sb_cond].H2

    res = scols(qqe_line, green_bars, red_bars, silver_bars, names=["qqe", "green", "red", "silver"])
    res = res.assign(
        # here we code hist bars:
        #  -1 -> silver < 0, +1 -> silver > 0, +2 -> green, -2 -> red
        code=-1 * (res.silver < 0) + 1 * (res.silver > 0) + 2 * (res.green > 0) - 2 * (res.red < 0)
    )

    return res


def ssl_exits(
    data: pd.DataFrame,
    baseline_type="hma",
    baseline_period=60,
    exit_type="hma",
    exit_period=15,
    atr_type="ema",
    atr_period=14,
    multy=0.2,
) -> pd.DataFrame:
    """
    Exits generator based on momentum reversal (from SSL Hybrid in TV)
    """
    check_frame_columns(data, "high", "low", "close")
    close = data.close
    lows = data.low
    highs = data.high

    base_line = smooth(close, baseline_type, baseline_period)
    exit_hi = smooth(highs, exit_type, exit_period)
    exit_lo = smooth(lows, exit_type, exit_period)

    tr = atr(data, atr_period, atr_type)
    upperk = base_line + multy * tr
    lowerk = base_line - multy * tr

    hlv3 = pd.Series(np.nan, close.index)
    hlv3.loc[close > exit_hi] = +1
    hlv3.loc[close < exit_lo] = -1
    hlv3 = hlv3.ffill()

    ssl_exit = srows(exit_hi[hlv3 < 0], exit_lo[hlv3 > 0])
    m = scols(
        ssl_exit,
        close,
        lows,
        highs,
        base_line,
        upperk,
        lowerk,
        names=["exit", "close", "low", "high", "base_line", "upperk", "lowerk"],
    )

    exit_short = m[(m.close.shift(1) <= m.exit.shift(1)) & (m.close > m.exit)].low
    exit_long = m[(m.close.shift(1) >= m.exit.shift(1)) & (m.close < m.exit)].high

    grow_line = pd.Series(np.nan, index=m.index)
    decl_line = pd.Series(np.nan, index=m.index)

    g = m[m.close > m.upperk].base_line
    d = m[m.close < m.lowerk].base_line
    grow_line[g.index] = g
    decl_line[d.index] = d

    return scols(
        exit_long,
        exit_short,
        grow_line,
        decl_line,
        m.base_line,
        m.upperk,
        m.lowerk,
        names=["exit_long", "exit_short", "grow", "decline", "base", "upper_k", "lower_k"],
    )


def streaks(xs: pd.Series):
    """
    Count consequently rising values in input series, actually it measures the duration of the trend.
    It is the number of points in a row with value has been higher (up) or lower (down) than the previous one.

    streaks(pd.Series([1,2,1,2,3,4,5,1]))

        0    0.0
        1    1.0
        2   -1.0
        3    1.0
        4    2.0
        5    3.0
        6    4.0
        7   -1.0

    """

    def consecutive_count(b):
        cs = b.astype(int).cumsum()
        return cs.sub(cs.mask(b).ffill().fillna(0))

    prev = xs.shift(1)
    return consecutive_count(xs > prev) - consecutive_count(xs < prev)


def percentrank(xs: pd.Series, period: int) -> pd.Series:
    """
    Percent rank is the percents of how many previous values was less than or equal to the current value of given series.
    """
    r = {}
    for t, x in zip(running_view(xs.index.values, period), running_view(xs.values, period)):
        r[t[-1]] = 100 * sum(x[-1] > x[:-1]) / period
    return pd.Series(r).reindex(xs.index)


def connors_rsi(close, rsi_period=3, streaks_period=2, percent_rank_period=100) -> pd.Series:
    """
    Connors RSI indicator (https://www.quantifiedstrategies.com/connors-rsi/)

    """
    return (rsi(close, rsi_period) + rsi(streaks(close), streaks_period) + percentrank(close, percent_rank_period)) / 3


def swings(ohlc: pd.DataFrame, trend_indicator: Callable = psar, **indicator_args) -> Struct:
    """
    Swing detector based on trend indicator
    """
    check_frame_columns(ohlc, "high", "low", "close")

    def _find_reversal_pts(highs, lows, indicator, is_lows):
        pts = {}
        cdp = continuous_periods(indicator, ~np.isnan(indicator))
        for b in cdp.blocks:
            ex_t = highs[b].idxmax() if is_lows else lows[b].idxmin()
            pts[ex_t] = highs.loc[ex_t] if is_lows else lows.loc[ex_t]
        return pts

    trend_detector = trend_indicator(ohlc, **indicator_args)
    down, up = None, None
    if trend_detector is not None and isinstance(trend_detector, pd.DataFrame):
        _d = "down" if "down" in trend_detector.columns else "utl"
        _u = "up" if "up" in trend_detector.columns else "dtl"
        down = trend_detector[_d]
        up = trend_detector[_u]

    hp = pd.Series(_find_reversal_pts(ohlc.high, ohlc.low, down, True), name="H")
    lp = pd.Series(_find_reversal_pts(ohlc.high, ohlc.low, up, False), name="L")

    u_tr, d_tr = {}, {}
    prev_t, prev_pt = None, None
    swings = {}
    for t, (h, l) in scols(hp, lp).iterrows():
        if np.isnan(h):
            if prev_pt:
                length = abs(prev_pt - l)
                u_tr[prev_t] = {"start_price": prev_pt, "end_price": l, "delta": length, "end": t}
                swings[prev_t] = {"p0": prev_pt, "p1": l, "direction": -1, "duration": t - prev_t, "length": length}
            prev_pt = l
            prev_t = t

        elif np.isnan(l):
            if prev_pt:
                length = abs(prev_pt - h)
                d_tr[prev_t] = {"start_price": prev_pt, "end_price": h, "delta": length, "end": t}
                swings[prev_t] = {"p0": prev_pt, "p1": h, "direction": +1, "duration": t - prev_t, "length": length}
            prev_pt = h
            prev_t = t

    trends_splits = scols(
        pd.DataFrame.from_dict(u_tr, orient="index"),
        pd.DataFrame.from_dict(d_tr, orient="index"),
        keys=["DownTrends", "UpTrends"],
    )

    swings = pd.DataFrame.from_dict(swings, orient="index")

    return Struct(swings=swings, trends=trends_splits, tops=hp, bottoms=lp)


@njit
def norm_pdf(x):
    return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


@njit
def lognorm_pdf(x, s):
    return np.exp(-(np.log(x) ** 2) / (2 * s**2)) / (x * s * np.sqrt(2 * np.pi))


@njit
def _pwma(_x, a, beta, T):
    _mean, _std, _var = np.zeros(_x.shape), np.zeros(_x.shape), np.zeros(_x.shape)
    _mean[0] = _x[0]

    for i in range(1, len(_x)):
        i_1 = i - 1
        diff = _x[i] - _mean[i_1]
        p = norm_pdf(diff / _std[i_1]) if _std[i_1] != 0 else 0  # Prob of observing diff
        a_t = a * (1 - beta * p) if i_1 > T else 1 - 1 / i  # weight to give to this point
        incr = (1 - a_t) * diff

        # Update Mean, Var, Std
        v = a_t * (_var[i_1] + diff * incr)
        _mean[i] = _mean[i_1] + incr
        _var[i] = v
        _std[i] = np.sqrt(v)
    return _mean, _var, _std


def pwma(x: pd.Series, alpha: float, beta: float, T: int) -> pd.DataFrame:
    """
    Implementation of probabilistic exponential weighted ma (https://sci-hub.shop/10.1109/SSP.2012.6319708)
    """
    m, v, s = _pwma(x.values, alpha, beta, T)
    return pd.DataFrame({"Mean": m, "Var": v, "Std": s}, index=x.index)


@njit
def _pwma_outliers_detector(x, a, beta, T, z_th, dist):
    x0 = 0 if np.isnan(x[0]) else x[0]
    s1, s2, s1_n, std_n = x0, x0**2, x0, 0

    s1a, stda, za, probs = np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)
    uba, lba = np.zeros(x.shape), np.zeros(x.shape)
    outliers = []

    for i in range(0, len(x)):
        s1 = s1_n
        std = std_n
        xi = x[i]

        z_t = ((xi - s1) / std) if (std != 0 and not np.isnan(xi)) else 0
        ub, lb = (z_t + z_th) * std + s1, (z_t - z_th) * std + s1

        # find probability
        p = norm_pdf(z_t)
        a_t = a * (1 - beta * p) if i + 1 >= T else 1 - 1 / (i + 1)

        # Update Mean, Var, Std
        if not np.isnan(xi):
            s1 = a_t * s1 + (1 - a_t) * xi
            s2 = a_t * s2 + (1 - a_t) * xi**2
            s1_n = s1
            std_n = np.sqrt(abs(s2 - np.square(s1)))

        # detects outlier
        if abs(z_t) >= z_th:
            outliers.append(i)

        s1a[i] = s1_n
        stda[i] = std_n
        probs[i] = p
        za[i] = z_t

        # upper and lower boundaries
        ub, lb = s1_n + z_th * std_n, s1_n - z_th * std_n
        uba[i] = ub
        lba[i] = lb
        # print('[%d] %.3f  -> s1_n: %.3f  s1: %.3f Z: %.3f s: %.3f s_n: %.3f' % (i, x[i], s1_n, s1, z_t, std, std_n))
    return s1a, stda, probs, za, uba, lba, outliers


def pwma_outliers_detector(x: pd.Series, alpha: float, beta: float, T=30, threshold=0.05, dist="norm") -> Struct:
    """
    Outliers detector based on pwma
    """
    import scipy.stats

    z_thr = scipy.stats.norm.ppf(1 - threshold / 2)
    m, s, p, z, u, l, oi = _pwma_outliers_detector(x.values, alpha, beta, T, z_thr, dist)
    res = pd.DataFrame({"Mean": m, "Std": s, "Za": z, "Uba": u, "Lba": l, "Prob": p}, index=x.index)

    return Struct(
        m=res["Mean"],
        s=res["Std"],
        z=res["Za"],
        u=res["Uba"],
        l=res["Lba"],
        p=res["Prob"],
        outliers=x.iloc[oi] if len(oi) else None,
        z_bounds=(z_thr, -z_thr),
    )


@njit
def __fast_ols(x, y):
    n = len(x)
    p, _, _, _ = np.linalg.lstsq(x, y, rcond=-1)
    r2 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) ** 2 / (
        (n * np.sum(x**2) - np.sum(x) ** 2) * (n * np.sum(y**2) - np.sum(y) ** 2)
    )
    return p[0][0], p[1][0], r2


def fast_ols(x, y) -> Struct:
    b, c, r2 = __fast_ols(column_vector(x), column_vector(y))
    return Struct(const=c, beta=b, r2=r2)


@njit
def fast_alpha(x, order=1, factor=10, min_threshold=1e-10):
    """
    Returns alpha based on following calculations:

    alpha = exp(-F*(1 - R2))

    where R2 - r squared metric from regression of x data against straight line y = x
    """
    x = x[~np.isnan(x)]
    x_max, x_min = np.max(x), np.min(x)

    if x_max - x_min > min_threshold:
        yy = 2 * (x - x_min) / (x_max - x_min) - 1
        xx = np.vander(np.linspace(-1, 1, len(yy)), order + 1)
        slope, intercept, r2 = __fast_ols(xx, yy.reshape(-1, 1))
    else:
        slope, intercept, r2 = np.inf, 0, 0

    return np.exp(-factor * (1 - r2)), r2, slope, intercept


@njit
def __rolling_slope(x, period, alpha_factor):
    ri = nans((len(x), 2))

    for i in range(period, x.shape[0]):
        a, r2, s, _ = fast_alpha(x[i - period : i], factor=alpha_factor)
        ri[i, :] = [r2, s]

    return ri


def rolling_slope(x: pd.Series, period: int, alpha_factor=10) -> pd.DataFrame:
    """
    Calculates slope/R2 on rolling basis for series from x
    returns DataFrame with 2 columns: R2, Slope
    """
    return pd.DataFrame(__rolling_slope(column_vector(x), period, alpha_factor), index=x.index, columns=["R2", "Slope"])


def find_movements_hilo(
    x: pd.DataFrame,
    threshold: float = -np.inf,
    pcntg=0.75,
    t_window: List | Tuple | int = 10,
    use_prev_movement_size_for_percentage=True,
    result_as_frame=False,
    collect_log=False,
    init_direction=0,
    silent=False,
):
    """
    Finds all movements in DataFrame x (should be pandas Dataframe object with low & high columns) which have absolute magnitude >= threshold
    and lasts not more than t_window bars.

    # Example:
    # -----------------

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from pylab import *

    z = 50 + np.random.normal(0, 0.2, 1000).cumsum()
    x = pd.Series(z, index=pd.date_range('1/1/2000 16:00:00', periods=len(z), freq='30s'))

    i_drops, i_grows, _, _ = find_movements(x, threshold=1, t_window=120, pcntg=.75)

    plt.figure(figsize=(15,10))

    # plot series
    plt.plot(x)

    # plot movements
    plt.plot(x.index[i_drops].T, x[i_drops].T, 'r--', lw=1.2);
    plt.plot(x.index[i_grows].T, x[i_grows].T, 'w--', lw=1.2);

    # or new version (after 2018-08-31)
    trends = find_movements(x, threshold=1, t_window=120, pcntg=.75, result_as_indexes=False)
    u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
    plt.plot([u.index, u.end], [u.start_price, u.end_price], 'w--', lw=0.7, marker='.', markersize=5);
    plt.plot([d.index, d.end], [d.start_price, d.end_price], 'r--', lw=0.7);

    plt.draw()
    plt.show()

    # -----------------

    :param x: pandas DataFrame object
    :param threshold: movement minimal magnitude threshold
    :param pcntg: percentage of previous movement (if use_prev_movement_size_for_percentage is True) that considered as start of new movement (1 == 100%)
    :param use_prev_movement_size_for_percentage: False if use percentage from previous price extremum (otherwise it uses prev. movement) [True]
    :param t_window: movement's length filter in bars or range: 120 or (0, 100) or (100, np.inf) etc
    :param drop_out_of_market: True if need to drop movements between sessions
    :param drop_weekends_crossings: True if need to drop movemets crossing weekends (for intraday data)
    :param silent: if True it doesn't show progress bar [False by default]
    :param result_as_frame: if False (default) result returned as tuple of indexes otherwise as DataFrame
    :param collect_log: True if need to collect track of tops/bottoms at times when they appeared
    :param init_direction: initial direction, can be 0, 1, -1
    :return: tuple with indexes of (droping movements, growing movements, droping magnitudes, growing magnitudes)
    """

    # check input arguments
    check_frame_columns(x, "high", "low")

    direction = init_direction
    mi, mx = 0, 0
    i_drops, i_grows = [], []
    log_rec = OrderedDict()
    timeline = x.index

    # check filter values
    if isinstance(t_window, int):
        t_window = [0, t_window]
    elif len(t_window) != 2 or t_window[0] >= t_window[1]:
        raise ValueError("t_window must have 2 ascending elements")

    if not silent:
        print(" -[", end="")
    n_p_len = max(int(len(x) / 100), 1)

    prev_vL = 0
    prev_vH = 0
    prev_mx = 0
    prev_mi = 0
    last_drop = None
    last_grow = None

    LO = x["low"].values
    HI = x["high"].values

    xL_mi = LO[mi]
    xH_mx = HI[mx]

    # for i in range(1, len(x)):
    i = 1
    x_len = len(x)
    while i < x_len:
        vL, vH = LO[i], HI[i]

        if direction <= 0:
            if direction < 0 and vH > prev_vH and last_grow is not None:
                # extend to previous grow start
                last_grow[1] = i

                # extend to current point
                mx = i
                xH_mx = HI[mx]
                prev_mx = mx
                prev_vH = xH_mx
                last_drop = None  # already added, reset to avoid duplicates

                mi = i
                xL_mi = LO[mi]

            elif vL < xL_mi:
                mi = i
                xL_mi = LO[mi]
                direction = -1

            else:
                # floating up
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (xH_mx - xL_mi)
                else:
                    l_mv = pcntg * xL_mi

                # check condition
                if (vL - xL_mi >= threshold) or (l_mv < vL - xL_mi):
                    # case when HighLow of a one bar are extreme points, to avoid infinite loop
                    if mx == mi:
                        mi += 1
                        # xL_mi = x.low.values[mi]

                    last_drop = [mx, mi]
                    # i_drops.append([mx, mi])
                    if last_grow:
                        # check if not violate the previous drop
                        min_idx = np.argmin(LO[last_grow[0] : last_grow[1] + 1])
                        if last_grow[1] > (last_grow[0] + 1) and min_idx > 0 and len(i_drops) > 0:
                            # we have low, which is lower than start of uptrend,
                            # remove the previous drop and replace it with the new one
                            new_drop = [i_drops[-1][0], last_grow[0] + min_idx]
                            i_drops[-1] = new_drop
                            last_grow[0] = last_grow[0] + min_idx

                        i_grows.append(last_grow)
                        last_grow = None

                    prev_vL = xL_mi
                    prev_mi = mi

                    if collect_log:
                        log_rec[timeline[i]] = {"Type": "-", "Time": timeline[mi], "Price": xL_mi}

                    # need to move back to the end of last drop
                    i = mi
                    mx = i
                    direction = 1
                    xH_mx = x.high.values[mx]
                    xL_mi = x.low.values[mi]

        if direction >= 0:
            if direction > 0 and vL < prev_vL and last_drop is not None:
                # extend to previous drop start
                last_drop[1] = i

                # extend to current point
                mi = i
                xL_mi = LO[mi]
                prev_mi = mi
                prev_vL = xL_mi
                last_grow = None  # already added, reset to avoid duplicates

                mx = i
                xH_mx = HI[mx]

            elif vH > xH_mx:
                mx = i
                xH_mx = HI[mx]
                direction = +1
            else:
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (xH_mx - xL_mi)
                else:
                    l_mv = pcntg * xH_mx

                if (xH_mx - vH >= threshold) or (l_mv < xH_mx - vH):
                    # i_grows.append([mi, mx])

                    # case when HighLow of a one bar are extreme points, to avoid infinite loop
                    if mx == mi:
                        mx += 1
                        # xH_mx = x.high.values[mx]

                    last_grow = [mi, mx]
                    if last_drop:
                        # check if not violate the previous drop
                        max_idx = np.argmax(HI[last_drop[0] : last_drop[1] + 1])
                        if last_drop[1] > (last_drop[0] + 1) and max_idx > 0 and len(i_grows) > 0:
                            # more than 1 bar between points
                            # we have low, which is lower than start of uptrend,
                            # remove the previous drop and replace it with the new one
                            new_grow = [i_grows[-1][0], last_drop[0] + max_idx]
                            i_grows[-1] = new_grow
                            last_drop[0] = last_drop[0] + max_idx

                        i_drops.append(last_drop)
                        last_drop = None

                    prev_vH = xH_mx
                    prev_mx = mx

                    if collect_log:
                        log_rec[timeline[i]] = {"Type": "+", "Time": timeline[mx], "Price": xH_mx}

                    # need to move back to the end of last grow
                    i = mx
                    mi = i
                    xL_mi = LO[mi]
                    xH_mx = HI[mx]
                    direction = -1

        i += 1
        if not silent and not (i % n_p_len):
            print(":", end="")

    if last_grow:
        i_grows.append(last_grow)
        last_grow = None

    if last_drop:
        i_drops.append(last_drop)
        last_drop = None

    if not silent:
        print("]-")
    i_drops = np.array(i_drops)
    i_grows = np.array(i_grows)

    # Nothing is found
    if len(i_drops) == 0 or len(i_grows) == 0:
        if not silent:
            print("\n\t[WARNING] find_movements_hilo: No trends found for given conditions !")
        return pd.DataFrame({"UpTrends": [], "DownTrends": []}) if result_as_frame else ([], [], [], [])

    # retain only movements equal or exceed specified threshold
    if not np.isinf(threshold):
        if i_drops.size:
            i_drops = i_drops[abs(x["low"][i_drops[:, 1]].values - x["high"][i_drops[:, 0]].values) >= threshold, :]
        if i_grows.size:
            i_grows = i_grows[abs(x["high"][i_grows[:, 1]].values - x["low"][i_grows[:, 0]].values) >= threshold, :]

    # retain only movements which shorter than specified window
    __drops_len = abs(i_drops[:, 1] - i_drops[:, 0])
    __grows_len = abs(i_grows[:, 1] - i_grows[:, 0])
    if i_drops.size:
        i_drops = i_drops[(__drops_len >= t_window[0]) & (__drops_len <= t_window[1]), :]
    if i_grows.size:
        i_grows = i_grows[(__grows_len >= t_window[0]) & (__grows_len <= t_window[1]), :]

    # Removed - filter out all movements which cover period from 16:00 till 9:30 next day

    # Removed - drop crossed weekend if required (we would not want to drop them when use daily prices)

    # drops and grows magnitudes
    v_drops = []
    if i_drops.size:
        v_drops = abs(x["low"][i_drops[:, 1]].values - x["high"][i_drops[:, 0]].values)

    v_grows = []
    if i_grows.size:
        v_grows = abs(x["high"][i_grows[:, 1]].values - x["low"][i_grows[:, 0]].values)

    # - return results
    indexes = np.array(x.index)
    i_d, i_g = indexes[i_drops], indexes[i_grows]
    x_d = np.array([x.high[i_d[:, 0]].values, x.low[i_d[:, 1]].values]).transpose()
    x_g = np.array([x.low[i_g[:, 0]].values, x.high[i_g[:, 1]].values]).transpose()

    d = pd.DataFrame(
        OrderedDict({"start_price": x_d[:, 0], "end_price": x_d[:, 1], "delta": v_drops, "end": i_d[:, 1]}),
        index=i_d[:, 0],
    )

    g = pd.DataFrame(
        OrderedDict({"start_price": x_g[:, 0], "end_price": x_g[:, 1], "delta": v_grows, "end": i_g[:, 1]}),
        index=i_g[:, 0],
    )

    trends = pd.concat((g, d), axis=1, keys=["UpTrends", "DownTrends"])
    if collect_log:
        return trends, pd.DataFrame.from_dict(log_rec, orient="index")

    return trends


_SMOOTHERS = {
    "sma": sma,
    "ema": ema,
    "tema": tema,
    "dema": dema,
    "zlema": zlema,
    "kama": kama,
    "jma": jma,
    "wma": wma,
    "mcginley": mcginley,
    "hma": hma,
    "ema_time": ema_time,
}


def smooth(x: pd.Series, smoother: str | Callable[[pd.Series, Any], pd.Series], *args, **kwargs) -> pd.Series:
    """
    Smooth series using either given function or find it by name from registered smoothers
    """

    f_sm = __empty_smoother
    if isinstance(smoother, str):
        if smoother in _SMOOTHERS:
            f_sm = _SMOOTHERS.get(smoother)
        else:
            raise ValueError(
                f"Smoothing method '{smoother}' is not supported, supported methods: {list(_SMOOTHERS.keys())}"
            )

    if isinstance(smoother, types.FunctionType):
        f_sm = smoother

    # smoothing
    x_sm = f_sm(x, *args, **kwargs)

    return x_sm if isinstance(x_sm, pd.Series) else pd.Series(x_sm.flatten(), index=x.index)


def volatility(
    data: pd.DataFrame,
    period: int,
    method: str = "c",
    volume_adjusting: bool = False,
    volume_adjustment_method: str = "normal",
    percentage: bool = True,
) -> pd.Series:
    """
    Calculate historical volatility (as standard deviation) using various estimation methods.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data frame containing 'open', 'high', 'low', 'close', 'volume' columns
    period : int
        Rolling window size for volatility calculation
    method : str, default 'c'
        Volatility estimation method:
        - 't': True range volatility
        - 'te': True range with ema smoothing
        - 'c': Classical close-to-close volatility
        - 'p': Parkinson high-low volatility
        - 'rs': Rogers-Satchell volatility
        - 'gk': Garman-Klass volatility
        - 'yz': Yang-Zhang volatility (combines overnight and trading volatility)
    volume_adjusting : bool, default False
        If True, adjusts volatility by relative volume changes
    volume_adjustment_method : str, default 'normal'
        Volume adjustment method:
        - 'normal': Normal volume adjustment: f_t = v_t / v_{t-1}
        - 'entropy': Entropy volume adjustment: f_t = v_t / sma(v_t, period)
    percentage : bool, default True
        If True, returns volatility in percentage terms, otherwise in price terms

    Returns
    -------
    pd.Series
        Volatility estimates for each period
    """
    if volume_adjusting:
        check_frame_columns(data, "open", "high", "low", "close", "volume")
        xr = data[["open", "high", "low", "close", "volume"]]
        o, h, l, c, v = xr["open"], xr["high"], xr["low"], xr["close"], xr["volume"]

        match volume_adjustment_method:
            case "normal":
                vi = v / v.shift(1)
            case "entropy":
                vi = v / sma(v, period)
            case _:
                raise ValueError(f"Invalid volume adjustment method: {volume_adjustment_method}")

    else:
        check_frame_columns(data, "open", "high", "low", "close")
        xr = data[["open", "high", "low", "close"]]
        o, h, l, c = xr["open"], xr["high"], xr["low"], xr["close"]
        vi = 1

    c1 = c.shift(1)
    oi = np.log(o) - np.log(c1)
    ci = np.log(c) - np.log(o)
    di = np.log(l) - np.log(o)
    ui = np.log(h) - np.log(o)

    match method:
        case "t":
            # - true range
            vol = sma(vi * pd.concat((abs(h - l), abs(h - c1), abs(l - c1)), axis=1).max(axis=1), period)
            return (100 * vol / c) if percentage else vol

        case "te":
            # - true range with ema smoothing
            vol = ema(vi * pd.concat((abs(h - l), abs(h - c1), abs(l - c1)), axis=1).max(axis=1), period)
            return (100 * vol / c) if percentage else vol

        case "c":
            # - classical
            m = sma(oi + ci, period)
            vol = sma(vi * ((oi + ci) - m) ** 2, period)

        case "p":
            # - parkinson
            vol = sma(vi * (ui - di) ** 2, period) / (4 * np.log(2))

        case "rs":
            # - rogers-satchell
            vol = sma(vi * (ui * (ui - ci) + di * (di - ci)), period)

        case "gk":
            # - garman-klass
            vol = sma(vi * ((ui - di) ** 2 - (4 * np.log(2) - 2) * ci**2), period)

        case "yz":
            # - yang-zhang
            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            vol = sma(
                vi * (ci - sma(ci, period)) ** 2
                + k * (oi - sma(oi, period)) ** 2
                + (1 - k) * (ui * (ui - ci) + di * (di - ci)),
                period,
            )
        case _:
            raise ValueError(f"Invalid method: {method} only t, te, c, p, rs, gk, yz are supported")
    return 100 * vol**0.5 if percentage else vol**0.5 * c


def fisher_transform(x: pd.Series, period: int) -> pd.Series:
    """
    Fisher transform of a timeseries
    """
    r = x.rolling(window=period)
    xp = 2 * (r.median() - r.min()) / (r.max() - r.min()) - 1
    return 0.5 * np.log((1 + xp) / (1 - xp))
