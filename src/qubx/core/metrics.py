import base64
import os
import re
from copy import copy
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Any, Callable, List, Tuple

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from IPython.display import HTML
from scipy import stats
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.series import OHLCV
from qubx.pandaz.utils import ohlc_resample, srows
from qubx.utils.charting.lookinglass import LookingGlass
from qubx.utils.charting.mpl_helpers import sbp
from qubx.utils.misc import makedirs, version
from qubx.utils.time import infer_series_frequency

YEARLY = 1
MONTHLY = 12
WEEKLY = 52
DAILY = 252
DAILY_365 = 365
HOURLY = DAILY * 6.5
MINUTELY = HOURLY * 60
HOURLY_FX = DAILY * 24
MINUTELY_FX = HOURLY_FX * 60

_D1 = pd.Timedelta("1D")
_W1 = pd.Timedelta("1W")


def absmaxdd(data: List | Tuple | pd.Series | np.ndarray) -> Tuple[float, int, int, int, pd.Series]:
    """

    Calculates the maximum absolute drawdown of series data.

    Args:
        data: vector of doubles. Data may be presented as list,
        tuple, numpy array or pandas series object.

    Returns:
        (max_abs_dd, d_start, d_peak, d_recovered, dd_data)

    Where:
        - max_abs_dd: absolute maximal drawdown value
        - d_start: index from data array where drawdown starts
        - d_peak: index when drawdown reach it's maximal value
        - d_recovered: index when DD is fully recovered
        - dd_data: drawdown series

    Example:

    mdd, ds, dp, dr, dd_data = absmaxdd(np.random.randn(1,100).cumsum())
    """

    if not isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError("Unknown type of input series")

    datatype = type(data)

    if datatype is pd.Series:
        indexes = data.index
        data = data.values
    elif datatype is not np.ndarray:
        data = np.array(data)

    dd = np.maximum.accumulate(data) - data
    mdd = dd.max()
    d_peak = dd.argmax()

    if mdd == 0:
        return 0, 0, 0, 0, [0]

    zeros_ixs = np.where(dd == 0)[0]
    zeros_ixs = np.insert(zeros_ixs, 0, 0)
    zeros_ixs = np.append(zeros_ixs, dd.size)

    d_start = zeros_ixs[zeros_ixs < d_peak][-1]
    d_recover = zeros_ixs[zeros_ixs > d_peak][0]

    if d_recover >= data.__len__():
        d_recover = data.__len__() - 1

    if datatype is pd.Series:
        dd = pd.Series(dd, index=indexes)

    return mdd, d_start, d_peak, d_recover, dd


def max_drawdown_pct(returns):
    """
    Finds the maximum drawdown of a strategy returns in percents

    :param returns: pd.Series or np.ndarray daily returns of the strategy, noncumulative
    :return: maximum drawdown in percents
    """
    if len(returns) < 1:
        return np.nan

    if isinstance(returns, pd.Series):
        returns = returns.values

    # drop nans
    returns[np.isnan(returns) | np.isinf(returns)] = 0.0

    cumrets = 100 * (returns + 1).cumprod(axis=0)
    max_return = np.fmax.accumulate(cumrets)
    return np.nanmin((cumrets - max_return) / max_return)


def portfolio_returns(portfolio_log: pd.DataFrame, method="pct", init_cash: float = 0.0) -> pd.Series:
    """
    Calculates returns based on specified method.

    :param pfl_log: portfolio log frame
    :param method: method to calculate, there are 3 main methods:
                    - percentage on equity ('pct', 'equity', 'on equity')
                    - percentage on previous portfolio value ('gmv', 'gross')
                    - percentage on fixed deposit amount ('fixed')

    :param init_cash: must be > 0 if used method is 'depo'
    :return: returns series
    """
    if "Total_PnL" not in portfolio_log.columns:
        portfolio_log = calculate_total_pnl(portfolio_log, split_cumulative=True)

    if method in ["pct", "equity", "on equity"]:
        # 'standard' percent of changes. It also takes initial deposit
        rets = (portfolio_log["Total_PnL"] + init_cash).pct_change()
    elif method in ["gmv", "gross"]:
        # today return is pct of yesterday's portfolio value (is USD)
        rets = (
            portfolio_log["Total_PnL"].diff() / (portfolio_log.filter(regex=".*_Value").abs().sum(axis=1).shift(1))
        ).fillna(0)
    elif method in ["fixed"]:
        # return is pct of PL changes to initial deposit (for fixed BP)
        if init_cash <= 0:
            raise ValueError("You must specify exact initial cash value when using 'fixed' method")
        rets = portfolio_log["Total_PnL"].diff() / init_cash
    else:
        raise ValueError("Unknown returns calculation method '%s'" % method)

    # cleanup returns
    rets.name = "Returns"
    rets[np.isinf(abs(rets))] = 0
    rets[np.isnan(rets)] = 0

    return rets


def cagr(returns, periods=DAILY):
    """
    Calculates the Compound Annual Growth Rate (CAGR) for the portfolio, by determining the number of years
    and then creating a compound annualised rate based on the total return.

    :param returns: A pandas Series or np.ndarray representing the returns
    :param periods: Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: CAGR's value
    """
    if len(returns) < 1:
        return np.nan

    cumrets = (returns + 1).cumprod(axis=0)
    years = len(cumrets) / float(periods)
    return (cumrets.iloc[-1] ** (1.0 / years)) - 1.0


def calmar_ratio(returns, periods=DAILY):
    """
    Calculates the Calmar ratio, or drawdown ratio, of a strategy.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: Calmar ratio (drawdown ratio) as float
    """
    max_dd = max_drawdown_pct(returns)
    if max_dd < 0:
        temp = cagr(returns, periods) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def sharpe_ratio(returns, risk_free=0.0, periods=DAILY) -> float:
    """
    Calculates the Sharpe ratio.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: Sharpe ratio
    """
    if len(returns) < 2:
        return np.nan

    returns_risk_adj = returns - risk_free
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]

    if np.std(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * np.sqrt(periods)


def rolling_sharpe_ratio(returns, risk_free=0.0, periods=DAILY) -> pd.Series:
    """
    Rolling Sharpe ratio.
    :param returns: pd.Series periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param periods: rolling window length
    :return:
    """
    returns_risk_adj = returns - risk_free
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]
    rolling = returns_risk_adj.rolling(window=periods)
    return pd.Series(np.sqrt(periods) * (rolling.mean() / rolling.std()), name="RollingSharpe")


def sortino_ratio(returns: pd.Series, required_return=0, periods=DAILY, _downside_risk=None) -> float:
    """
    Calculates the Sortino ratio of a strategy.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param required_return: minimum acceptable return
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :param _downside_risk: the downside risk of the given inputs, if known. Will be calculated if not provided
    :return: annualized Sortino ratio
    """
    if len(returns) < 2:
        return np.nan

    mu = np.nanmean(returns - required_return, axis=0)
    dsr = _downside_risk if _downside_risk is not None else downside_risk(returns, required_return)
    if dsr == 0.0:
        return np.nan if mu == 0 else np.inf
    return periods * mu / dsr


def information_ratio(returns, factor_returns) -> float:
    """
    Calculates the Information ratio of a strategy (see https://en.wikipedia.org/wiki/information_ratio)
    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param factor_returns: benchmark return to compare returns against
    :return: information ratio
    """
    if len(returns) < 2:
        return np.nan

    active_return = returns - factor_returns
    tracking_error = np.nanstd(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.nanmean(active_return) / tracking_error


def downside_risk(returns, required_return=0.0, periods=DAILY):
    """
    Calculates the downside deviation below a threshold

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param required_return: minimum acceptable return
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                     Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :return: annualized downside deviation
    """
    if len(returns) < 1:
        return np.nan

    downside_diff = (returns - required_return).copy()
    downside_diff[downside_diff > 0] = 0.0
    mean_squares = np.nanmean(np.square(downside_diff), axis=0)
    ds_risk = np.sqrt(mean_squares) * np.sqrt(periods)

    if len(returns.shape) == 2 and isinstance(returns, pd.DataFrame):
        ds_risk = pd.Series(ds_risk, index=returns.columns)

    return ds_risk


def omega_ratio(returns, risk_free=0.0, required_return=0.0, periods=DAILY):
    """
    Omega ratio (see https://en.wikipedia.org/wiki/Omega_ratio for more details)

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param risk_free: constant risk-free return throughout the period
    :param required_return: Minimum acceptance return of the investor. Threshold over which to
                             consider positive vs negative returns. It will be converted to a
                             value appropriate for the period of the returns. E.g. An annual minimum
                             acceptable return of 100 will translate to a minimum acceptable
                             return of 0.018.
    :param periods: Factor used to convert the required_return into a daily
                     value. Enter 1 if no time period conversion is necessary.
    :return: Omega ratio
    """
    if len(returns) < 2:
        return np.nan

    if periods == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1.0 / periods) - 1

    returns_less_thresh = returns - risk_free - return_threshold
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    return (numer / denom) if denom > 0.0 else np.nan


def aggregate_returns(returns: pd.Series, convert_to: str) -> pd.DataFrame | pd.Series:
    """
    Aggregates returns by specified time period
    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param convert_to: 'D', 'W', 'M', 'Y' (and any supported in pandas.resample method)
    :return: aggregated returns
    """

    def cumulate_returns(x):
        return ((x + 1).cumprod(axis=0) - 1).iloc[-1] if len(x) > 0 else 0.0

    str_check = convert_to.lower()
    resample_mod = None
    if str_check in ["a", "annual", "y", "yearly"]:
        resample_mod = "YE"
    elif str_check in ["m", "monthly", "mon"]:
        resample_mod = "ME"
    elif str_check in ["w", "weekly"]:
        resample_mod = "W"
    elif str_check in ["d", "daily"]:
        resample_mod = "D"
    else:
        resample_mod = convert_to

    return returns.resample(resample_mod).apply(cumulate_returns)


def annual_volatility(returns, periods=DAILY, alpha=2.0):
    """
    Calculates annual volatility of a strategy

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                    Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :param alpha: scaling relation (Levy stability exponent).
    :return:
    """
    if len(returns) < 2:
        return np.nan

    return np.nanstd(returns, ddof=1) * (periods ** (1.0 / alpha))


def stability_of_returns(returns):
    """
    Calculates R-squared of a linear fit to the cumulative log returns.
    Computes an ordinary least squares linear fit, and returns R-squared.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :return: R-squared
    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]
    cum_log_returns = np.log1p(returns, where=returns > -1).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns).rvalue  # type: ignore
    return rhat**2


def tail_ratio(returns):
    """
    Calculates the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times as bad as profits.

    :param returns: pd.Series or np.ndarray periodic returns of the strategy, noncumulative
    :return: tail ratio
    """
    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    pc5 = np.abs(np.percentile(returns, 5))

    return (np.abs(np.percentile(returns, 95)) / pc5) if pc5 != 0 else np.nan


def split_cumulative_pnl(pfl_log: pd.DataFrame) -> pd.DataFrame:
    """
    Position.pnl tracks cumulative PnL (realized+unrealized) but if we want to operate with PnL for every bar
    we need to find diff from these cumulative series

    :param pfl_log: position manager log (portfolio log)
    :return: frame with splitted PL
    """
    # take in account commissions (now we cumsum it)
    pl = pfl_log.filter(regex=r".*_PnL|.*_Commissions")
    if pl.shape[1] == 0:
        raise ValueError("PnL columns not found. Input frame must contain at least 1 column with '_PnL' suffix")

    pl_diff = pl.diff()

    # at first row we use first value of PnL
    pl_diff.loc[pl.index[0]] = pl.iloc[0]

    # substitute new diff PL
    pfl_splitted = pfl_log.copy()
    pfl_splitted.loc[:, pfl_log.columns.isin(pl_diff.columns)] = pl_diff
    return pfl_splitted


def calculate_total_pnl(pfl_log: pd.DataFrame, split_cumulative=True) -> pd.DataFrame:
    """
    Finds summary of all P&L column (should have '_PnL' suffix) in given portfolio log dataframe.
    Attaches additional Total_PnL column with result.

    :param pfl_log: position manager log (portfolio log)
    :param split_cumulative: set true if we need to split cumulative PnL [default is True]
    :return:
    """
    n_pfl = pfl_log.copy()
    if "Total_PnL" not in n_pfl.columns:
        if split_cumulative:
            n_pfl = split_cumulative_pnl(n_pfl)

        n_pfl["Total_PnL"] = n_pfl.filter(regex=r".*_PnL").sum(axis=1)
        n_pfl["Total_Commissions"] = n_pfl.filter(regex=r".*_Commissions").sum(axis=1)

    return n_pfl


def alpha(returns, factor_returns, risk_free=0.0, period=DAILY, _beta=None):
    """
    Calculates annualized alpha of portfolio.

    :param returns: Daily returns of the strategy, noncumulative.
    :param factor_returns: Daily noncumulative returns of the factor to which beta is
           computed. Usually a benchmark such as the market
    :param risk_free: Constant risk-free return throughout the period. For example, the
                      interest rate on a three month us treasury bill
    :param periods: Defines the periodicity of the 'returns' data for purposes of annualizing.
                    Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    :param _beta: The beta for the given inputs, if already known. Will be calculated
                internally if not provided.
    :return: alpha
    """
    if len(returns) < 2:
        return np.nan

    if _beta is None:
        _beta = beta(returns, factor_returns, risk_free)

    adj_returns = returns - risk_free
    adj_factor_returns = factor_returns - risk_free
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    return np.nanmean(alpha_series) * period


def beta(returns, benchmark_returns, risk_free=0.0):
    """
    Calculates beta of portfolio.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    :param returns: pd.Series or np.ndarray. Daily returns of the strategy, noncumulative.
    :param benchmark_returns: pd.Series or np.ndarray. Daily noncumulative returns of the factor to which beta is
           computed. Usually a benchmark such as the market.
    :param risk_free: Constant risk-free return throughout the period. For example, the interest rate
                      on a three month us treasury bill.
    :return: beta
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return np.nan

    # Filter out dates with np.nan as a return value

    if len(returns) != len(benchmark_returns):
        if len(returns) > len(benchmark_returns):
            returns = returns.drop(returns.index.difference(benchmark_returns.index))
        else:
            benchmark_returns = benchmark_returns.drop(benchmark_returns.index.difference(returns.index))

    joint = np.vstack([returns - risk_free, benchmark_returns])
    joint = joint[:, ~np.isnan(joint).any(axis=0)]
    if joint.shape[1] < 2:
        return np.nan

    cov = np.cov(joint, ddof=0)

    if np.absolute(cov[1, 1]) < 1.0e-30:
        return np.nan

    return cov[0, 1] / cov[1, 1]


def var_cov_var(P_usd, mu, sigma, c=0.95):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio of value P.

    https://www.quantstart.com/articles/Value-at-Risk-VaR-for-Algorithmic-Trading-Risk-Management-Part-I

    also here:
    http://stackoverflow.com/questions/30878265/calculating-value-at-risk-or-most-probable-loss-for-a-given-distribution-of-r#30895548

    :param P_usd: portfolio value
    :param c: confidence level
    :param mu: mean of returns
    :param sigma: standard deviation of returns
    :return: value at risk
    """
    alpha = norm.ppf(1 - c, mu, sigma) if sigma != 0.0 else 0
    return P_usd - P_usd * (alpha + 1)


def qr(equity):
    """

    QR = R2 * B / S

    Where:
     B - slope (roughly in the ranges of average trade PnL, higher is better)
     R2 - r squared metric (proportion of variance explained by linear model, straight line has r2 == 1)
     S - standard error represents volatility of equity curve (lower is better)

    :param equity: equity (cumulative)
    :return: QR measure or NaN if not enough data for calculaitons
    """
    if len(equity) < 1 or all(equity == 0.0):
        return np.nan

    rgr = OLS(equity, np.vander(np.linspace(-1, 1, len(equity)), 2)).fit()
    b = rgr.params.iloc[0] if isinstance(rgr.params, pd.Series) else rgr.params[0]
    return rgr.rsquared * b / np.std(rgr.resid)


def monthly_returns(
    portfolio, init_cash, period="monthly", daily="pct", monthly="pct", weekly="pct", performace_period=DAILY
):
    """
    Calculate monthly or weekly returns table along with account balance
    """
    pft_total = calculate_total_pnl(portfolio, split_cumulative=False)
    pft_total["Total_PnL"] = pft_total["Total_PnL"].cumsum()
    returns = portfolio_returns(pft_total, init_cash=init_cash, method=daily)
    r_daily = aggregate_returns(returns, "daily")
    print("CAGR: %.2f%%" % (100 * cagr(r_daily, performace_period)))

    if period == "weekly":
        returns = portfolio_returns(pft_total, init_cash=init_cash, method=weekly)
        r_month = aggregate_returns(returns, "weekly")
        acc_balance = init_cash + pft_total.Total_PnL.groupby(pd.Grouper(freq="1W")).last()
    else:
        returns = portfolio_returns(pft_total, init_cash=init_cash, method=monthly)
        r_month = aggregate_returns(returns, "monthly")
        acc_balance = init_cash + pft_total.Total_PnL.groupby(pd.Grouper(freq="1M")).last()

    return pd.concat((100 * r_month, acc_balance), axis=1, keys=["Returns", "Balance"])


class TradingSessionResult:
    # fmt: off
    id: int
    name: str
    start: str | pd.Timestamp
    stop: str | pd.Timestamp
    exchanges: list[str]                             # exchange names
    instruments: list[Instrument]                    # instruments used at the start of the session (TODO: need to collect all traded instruments)
    capital: float | dict[str, float]
    base_currency: str
    commissions: str | dict[str, str] | None         # used commissions ("vip0_usdt" etc)
    portfolio_log: pd.DataFrame                      # portfolio log records
    executions_log: pd.DataFrame                     # executed trades
    signals_log: pd.DataFrame                        # signals generated by the strategy
    targets_log: pd.DataFrame                        # targets generated by the strategy
    strategy_class: str                              # strategy full qualified class name
    parameters: dict[str, Any]                       # strategy parameters if provided
    is_simulation: bool
    creation_time: pd.Timestamp | None = None        # when result was created
    author: str | None = None                        # who created the result
    qubx_version: str | None = None                  # Qubx version used to create the result
    _metrics: dict[str, float] | None = None         # performance metrics
    variation_name: str | None = None                # variation name if this belongs to a variated set
    emitter_data: pd.DataFrame | None = None         # metrics emitter data if available
    # fmt: on

    def __init__(
        self,
        id: int,
        name: str,
        start: str | pd.Timestamp,
        stop: str | pd.Timestamp,
        exchanges: list[str],
        instruments: list[Instrument],
        capital: float | dict[str, float],
        base_currency: str,
        commissions: str | dict[str, str] | None,
        portfolio_log: pd.DataFrame,
        executions_log: pd.DataFrame,
        signals_log: pd.DataFrame,
        targets_log: pd.DataFrame,
        strategy_class: str,
        parameters: dict[str, Any] | None = None,
        is_simulation=True,
        creation_time: str | pd.Timestamp | None = None,
        author: str | None = None,
        variation_name: str | None = None,
        emitter_data: pd.DataFrame | None = None,
    ):
        self.id = id
        self.name = name
        self.start = start
        self.stop = stop
        self.exchanges = exchanges
        self.instruments = instruments
        self.capital = capital
        self.base_currency = base_currency
        self.commissions = commissions
        self.portfolio_log = portfolio_log
        self.executions_log = executions_log
        self.signals_log = signals_log
        self.targets_log = targets_log
        self.strategy_class = strategy_class
        self.parameters = parameters if parameters else {}
        self.is_simulation = is_simulation
        self.creation_time = pd.Timestamp(creation_time) if creation_time else pd.Timestamp.now()
        self.author = author
        self.qubx_version = version()
        self.variation_name = variation_name
        self.emitter_data = emitter_data
        self._metrics = None

    def performance(self) -> dict[str, float]:
        """
        Calculate performance metrics for the trading session
        """
        if not self._metrics:
            # - caluclate short statistics
            self._metrics = portfolio_metrics(
                self.portfolio_log,
                self.executions_log,
                self.capital,
                performance_statistics_period=DAILY_365,
                account_transactions=True,
                commission_factor=1,
            )
            # - convert timestamps to isoformat
            for k, v in self._metrics.items():
                match v:
                    case pd.Timestamp():
                        self._metrics[k] = v.isoformat()
                    case np.float64():
                        self._metrics[k] = float(v)
            # fmt: off
            for k in [
                "equity", "drawdown_usd", "drawdown_pct",
                "compound_returns", "returns_daily", "returns", "monthly_returns",
                "rolling_sharpe", "long_value", "short_value",
            ]:
                self._metrics.pop(k, None)
            # fmt: on

        return self._metrics

    def get_total_capital(self) -> float:
        return sum(self.capital.values()) if isinstance(self.capital, dict) else self.capital

    @property
    def symbols(self) -> list[str]:
        """
        Extracts all traded symbols from the portfolio log
        """
        if not self.portfolio_log.empty:
            return list(set(self.portfolio_log.columns.str.split("_").str.get(0).values))
        return []

    def config(self, short=True) -> str:
        """
        Return configuration as string: "test.strategies.Strategy1(parameter1=12345)"
        TODO: probably we need to return recreated new object
        """
        _cfg = ""
        if self.strategy_class:
            _params = ", ".join([f"{k}={repr(v)}" for k, v in self.parameters.items()])
            _class = self.strategy_class.split(".")[-1] if short else self.strategy_class
            _cfg = f"{_class}({_params})"
            # _cfg = f"{{ {repr(self.name)}: {_class}({_params}) }}"
            # if instantiated: return eval(_cfg)
        return _cfg

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "start": pd.Timestamp(self.start).isoformat(),
            "stop": pd.Timestamp(self.stop).isoformat(),
            "exchanges": self.exchanges,
            "capital": self.capital,
            "base_currency": self.base_currency,
            "commissions": self.commissions,
            "strategy_class": self.strategy_class,
            "parameters": self.parameters,
            "is_simulation": self.is_simulation,
            "creation_time": pd.Timestamp(self.creation_time).isoformat(),
            "author": self.author,
            "qubx_version": self.qubx_version,
            "symbols": self.symbols,
            "performance": dict(self.performance()),
            "variation_name": self.variation_name,
        }

    def to_html(self, compound=True) -> HTML:
        table: pd.DataFrame = tearsheet(self, compound=compound, plot_equities=True, plot_leverage=True, no_title=True)  # type: ignore

        # - make it bit more readable
        table.index = table.index.map(lambda x: "/".join(x.split(",")[:3]))
        _rep = table.round(3).to_html(classes="rep_table")
        _eqty = _plt_to_base64()

        _s = "Simulation Report" if self.is_simulation else "Live"
        _name = f"{_s} for (<font color='red'>{self.name}.{self.id}</font>) generated <font color='green'>{str(self.creation_time)}</font>"
        _cap = f"{self.capital} {self.base_currency} ({self.commissions} @ {self.exchanges})"

        _tmpl = f"""
            <style>
                .report div {{ 
                    font-family: 'Maven Pro', 'Roboto', 'JetBrains mono', 'Meslo LG S', 'Pragmata Pro Mono', 'hasklig semibold' !important; 
                    font-size: 12px; background-color: #000; 
                }}
                .wrap_table th {{ text-align:center !important; font-weight: bold; font-size: 18px; color: #756732; }}
                .wrap_table td, .wrap_table tr {{ background: none !important; text-align:left !important; }}

                .rep_table table {{ width:100%;}}
                .rep_table th {{ text-align:center !important; font-weight: bold; font-size: 12px; color: #328032; }}
                .rep_table td, .rep_table tr {{ background: none !important; text-align:left !important; }}

                .flex-container {{ display: flex; align-items: flex-start; width: 100%; }}
                .table_block {{ width:100%; }}
                .wrap_table table, .wrap_table td, .wrap_table tr, .wrap_table th {{
                    border: 1px solid #55554a85; border-collapse: collapse; color: #9eb9c3d9 !important; background-color: #000; padding-left: 5px;
                }}
            </style>
            <h1>{_name}</h1>
            <div class="report">
            <table class="wrap_table" width=100%> 
                <tr><td width=15%>Strategy</td> <td>{self.config(False)}</td></tr>
                <tr><td width=15%>Period</td><td>{str(self.start)} : {str(self.stop)}</td></tr>
                <tr><td width=15%>Instruments</td> <td>{self.symbols}</td></tr>
                <tr><td width=15%>Capital</td> <td>{_cap}</td></tr>
                <tr><td width=15%>Author</td> <td>{self.author}</td></tr>
                <tr><td width=15%>Qubx version</td> <td>{self.qubx_version}</td></tr>
            </table>
                <div class="report"> 
                    <table class="wrap_table" width=100%> <th>Performance</th> </table>
                    {_rep}
                    <table class="wrap_table" width=100%> <th>Equity</th> </table>
                    <img src='{_eqty}' style="max-width:1000px; width:100%; height:450px;"/> 
                </div>
            </div>
        """
        return HTML(_tmpl)

    def to_file(
        self,
        name: str,
        description: str | None = None,
        compound=True,
        archive=True,
        suffix: str | None = None,
        attachments: list[str] | None = None,
    ):
        """
        Save the trading session results to files.

        Args:
            name (str): Base name/path for saving the files
            description (str | None, optional): Description to include in info file. Defaults to None.
            compound (bool, optional): Whether to use compound returns in report. Defaults to True.
            archive (bool, optional): Whether to zip the output files. Defaults to True.
            suffix (str | None, optional): Optional suffix to append to filename. Defaults to None.
            attachments (list[str] | None, optional): Additional files to include. Defaults to None.

        The following files are saved:
            - info.yml: Contains strategy configuration and metadata
            - portfolio.csv: Portfolio state log
            - executions.csv: Trade execution log
            - signals.csv: Strategy signals log
            - report.html: HTML performance report
            - Any provided attachment files

        If archive=True, all files are zipped into a single archive and the directory is removed.
        """
        import shutil

        if suffix is not None:
            name = f"{name}{suffix}"
        else:
            name = (name + self.creation_time.strftime("%Y%m%d%H%M%S")) if self.creation_time else name
        p = Path(makedirs(name))
        with open(p / "info.yml", "w") as f:
            info = self.info()
            if description:
                info["description"] = description
            # - set name if not specified
            if info.get("name") is None:
                info["name"] = name

            # - add numpy array representer
            yaml.SafeDumper.add_representer(np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist()))
            yaml.safe_dump(info, f, sort_keys=False, indent=4)

        # - save logs
        self.portfolio_log.to_csv(p / "portfolio.csv")
        self.executions_log.to_csv(p / "executions.csv")
        self.signals_log.to_csv(p / "signals.csv")
        self.targets_log.to_csv(p / "targets.csv")

        # - save report
        with open(p / "report.html", "w") as f:
            f.write(self.to_html(compound=compound).data)

        # - save attachments
        if attachments:
            for a in attachments:
                if (af := Path(a)).is_file():
                    shutil.copy(af, p / af.name)

        if archive:
            shutil.make_archive(name, "zip", p)  # type: ignore
            shutil.rmtree(p)  # type: ignore

    @staticmethod
    def from_file(path: str):
        import zipfile

        path = path + ".zip" if not path.endswith(".zip") else path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        with zipfile.ZipFile(path, "r") as zip_ref:
            info = yaml.safe_load(zip_ref.read("info.yml"))
            try:
                portfolio = pd.read_csv(
                    zip_ref.open("portfolio.csv"), index_col=["timestamp"], parse_dates=["timestamp"]
                )
            except:
                portfolio = pd.DataFrame()
            try:
                executions = pd.read_csv(
                    zip_ref.open("executions.csv"), index_col=["timestamp"], parse_dates=["timestamp"]
                )
            except:
                executions = pd.DataFrame()
            try:
                signals = pd.read_csv(zip_ref.open("signals.csv"), index_col=["timestamp"], parse_dates=["timestamp"])
            except:
                signals = pd.DataFrame()
            try:
                targets = pd.read_csv(zip_ref.open("targets.csv"), index_col=["timestamp"], parse_dates=["timestamp"])
            except:
                targets = pd.DataFrame()

        # load result
        _qbx_version = info.pop("qubx_version")
        _decr = info.pop("description", None)
        _perf = info.pop("performance", None)
        info["instruments"] = info.pop("symbols")
        # - fix for old versions
        _exch = info.pop("exchange") if "exchange" in info else info.pop("exchanges")
        info["exchanges"] = _exch if isinstance(_exch, list) else [_exch]
        tsr = TradingSessionResult(
            **info, portfolio_log=portfolio, executions_log=executions, signals_log=signals, targets_log=targets
        )
        tsr.qubx_version = _qbx_version
        tsr._metrics = _perf
        return tsr

    def tearsheet(
        self,
        compound: bool = True,
        account_transactions=True,
        performance_statistics_period=365,
        timeframe: str | pd.Timedelta | None = None,
        sort_by: str | None = "Sharpe",
        sort_ascending: bool = False,
        plot_equities: bool = True,
        commission_factor: float = 1,
        plot_leverage: bool = False,
        use_plotly: bool = False,
        no_title: bool = False,
    ):
        return tearsheet(
            self,
            compound,
            account_transactions,
            performance_statistics_period,
            timeframe,
            sort_by,
            sort_ascending,
            plot_equities,
            commission_factor,
            plot_leverage,
            use_plotly,
            no_title,
        )

    def __repr__(self) -> str:
        _s = "Simulation" if self.is_simulation else "Live"
        _t = f"[{self.start} - {self.stop}]" if self.is_simulation else ""
        r = f"""::: {_s} {self.id} ({self.name}) {_t}
 :   QUBX: {self.qubx_version}
 :   Capital: {self.capital} {self.base_currency} ({self.commissions} @ {self.exchanges})
 :   Instruments: [{",".join(self.symbols)}]
 :   Created: {self.creation_time} by {self.author} 
 :   Strategy: {self.config(False)}
 :   Generated: {len(self.signals_log)} signals, {len(self.executions_log)} executions
"""
        _perf = pd.DataFrame.from_dict(self.performance(), orient="index").T.to_string(index=None)
        for _i, s in enumerate(_perf.split("\n")):
            r += f"       : {s}\n" if _i > 0 else f"  `----: {s}\n"
        return r


def portfolio_symbols(src: pd.DataFrame | TradingSessionResult) -> List[str]:
    """
    Get list of symbols from portfolio log
    """
    df = src.portfolio_log if isinstance(src, TradingSessionResult) else src
    return list(df.columns[::5].str.split("_").str.get(0).values)


def pnl(
    src: pd.DataFrame | TradingSessionResult, c=1, cum=False, total=False, resample=None
) -> pd.Series | pd.DataFrame:
    """
    Extract PnL from portfolio log
    """
    x = src.portfolio_log if isinstance(src, TradingSessionResult) else src
    pl = x.filter(regex=".*_PnL").rename(lambda x: x.split("_")[0], axis=1)
    comms = x.filter(regex=".*_Commissions").rename(lambda x: x.split("_")[0], axis=1)
    r = pl - c * comms
    if resample:
        r = r.resample(resample).sum()
    r = r.cumsum() if cum else r
    return r.sum(axis=1) if total else r


def drop_symbols(src: pd.DataFrame | TradingSessionResult, *args, quoted="USDT") -> pd.DataFrame:
    """
    Drop symbols (is quoted currency) from portfolio log
    """
    s = "|".join([f"{a}{quoted}" if not a.endswith(quoted) else a for a in args])
    df = src.portfolio_log if isinstance(src, TradingSessionResult) else src
    return df.filter(filter(lambda si: not re.match(f"^{s}_.*", si), df.columns))


def pick_symbols(src: pd.DataFrame | TradingSessionResult, *args, quoted="USDT") -> pd.DataFrame:
    """
    Select symbols (is quoted currency) from portfolio log
    """
    df = src.portfolio_log if isinstance(src, TradingSessionResult) else src

    # - pick up from execution report
    if "instrument" in df.columns and "quantity" in df.columns:
        rx = "|".join([f"{a}.*" for a in args])
        return df[df["instrument"].str.match(rx)]

    # - pick up from PnL log report
    s = "|".join([f"{a}{quoted}" if not a.endswith(quoted) else a for a in args])
    return df.filter(filter(lambda si: re.match(f"^{s}_.*", si), df.columns))


def portfolio_metrics(
    portfolio_log: pd.DataFrame,
    executions_log: pd.DataFrame,
    init_cash: float | dict[str, float],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    risk_free: float = 0.0,
    rolling_sharpe_window=12,
    account_transactions=True,
    performance_statistics_period=DAILY_365,
    **kwargs,
) -> dict:
    if len(portfolio_log) == 0:
        raise ValueError("Can't calculate statistcis on empty portfolio")

    if isinstance(init_cash, dict):
        init_cash = sum(init_cash.values())

    sheet = dict()

    pft_total = calculate_total_pnl(portfolio_log, split_cumulative=False)
    pft_total["Total_PnL"] = pft_total["Total_PnL"].cumsum()
    pft_total["Total_Commissions"] = pft_total["Total_Commissions"].cumsum()

    # if it's asked to account transactions into equ
    pft_total["Total_Commissions"] *= kwargs.get("commission_factor", 1)
    if account_transactions:
        pft_total["Total_PnL"] -= pft_total["Total_Commissions"]

    # calculate returns
    returns = portfolio_returns(pft_total, init_cash=init_cash, method="pct")
    returns_on_init_bp = portfolio_returns(pft_total, init_cash=init_cash, method="fixed")

    if start:
        returns = returns[start:]
        returns_on_init_bp = returns_on_init_bp[start:]

    if end:
        returns = returns[:end]
        returns_on_init_bp = returns_on_init_bp[:end]

    # - aggregate returns to higher timeframe
    try:
        _conversion = "daily"
        match _s_freq := infer_series_frequency(returns):
            case _ if _s_freq <= _D1.to_timedelta64():
                _conversion = "daily"
            case _ if _s_freq > _D1.to_timedelta64() and _s_freq <= _W1.to_timedelta64():
                _conversion = "weekly"
            case _:
                _conversion = "monthly"

        returns_daily = aggregate_returns(returns, _conversion)
        returns_on_init_bp = aggregate_returns(returns_on_init_bp, _conversion)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to aggregate returns: {e}. Using raw returns.")
        returns_daily = returns

    # todo: add transaction_cost calculations
    equity = init_cash + pft_total["Total_PnL"]
    mdd, ddstart, ddpeak, ddrecover, dd_data = absmaxdd(equity)
    execs = len(executions_log)
    mdd_pct = 100 * dd_data / equity.cummax() if execs > 0 else pd.Series(0, index=equity.index)
    sheet["equity"] = equity
    sheet["gain"] = sheet["equity"].iloc[-1] - sheet["equity"].iloc[0]
    sheet["cagr"] = cagr(returns_daily, performance_statistics_period)
    sheet["sharpe"] = sharpe_ratio(returns_daily, risk_free, performance_statistics_period)
    sheet["qr"] = qr(equity) if execs > 0 else 0
    sheet["drawdown_usd"] = dd_data
    sheet["drawdown_pct"] = mdd_pct
    # 25-May-2019: MDE fixed Max DD pct calculations
    sheet["max_dd_pct"] = max(mdd_pct)
    # sheet["max_dd_pct_on_init"] = 100 * mdd / init_cash
    sheet["mdd_usd"] = mdd
    sheet["mdd_start"] = equity.index[ddstart]
    sheet["mdd_peak"] = equity.index[ddpeak]
    sheet["mdd_recover"] = equity.index[ddrecover]
    sheet["returns"] = returns
    sheet["returns_daily"] = returns_daily
    sheet["compound_returns"] = (returns + 1).cumprod(axis=0) - 1
    sheet["rolling_sharpe"] = rolling_sharpe_ratio(returns_daily, risk_free, periods=rolling_sharpe_window)
    sheet["sortino"] = sortino_ratio(
        returns_daily, risk_free, performance_statistics_period, _downside_risk=kwargs.pop("downside_risk", None)
    )
    sheet["calmar"] = calmar_ratio(returns_daily, performance_statistics_period)
    # sheet["ann_vol"] = annual_volatility(returns_daily)
    sheet["tail_ratio"] = tail_ratio(returns_daily)
    sheet["stability"] = stability_of_returns(returns_daily)
    sheet["monthly_returns"] = aggregate_returns(returns_daily, convert_to="mon")
    r_m = np.mean(returns_daily)
    r_s = np.std(returns_daily)
    sheet["var"] = var_cov_var(init_cash, r_m, r_s)
    sheet["avg_return"] = 100 * r_m

    # portfolio market values
    mkt_value = pft_total.filter(regex=".*_Value")
    sheet["long_value"] = mkt_value[mkt_value > 0].sum(axis=1).fillna(0)
    sheet["short_value"] = mkt_value[mkt_value < 0].sum(axis=1).fillna(0)

    # total commissions
    sheet["fees"] = pft_total["Total_Commissions"].iloc[-1]

    # funding payments (if available)
    funding_columns = pft_total.filter(regex=".*_Funding")
    if not funding_columns.empty:
        total_funding = funding_columns.sum(axis=1)
        sheet["funding_pnl"] = 100 * total_funding.iloc[-1] / init_cash  # as percentage of initial capital
    else:
        sheet["funding_pnl"] = 0.0

    # executions metrics
    sheet["execs"] = execs

    return sheet


def find_session(sessions: list[TradingSessionResult], name: str) -> TradingSessionResult:
    """
    Match the session by a regex pattern. It can also be a substring.
    """
    for s in sessions:
        if re.match(name, s.name):
            return s
        # Check for substring match
        if name in s.name:
            return s
    raise ValueError(f"Session with name {name} not found")


def find_sessions(sessions: list[TradingSessionResult], name: str) -> list[TradingSessionResult]:
    """
    Match the session by a regex pattern. It can also be a substring.
    """
    return [s for s in sessions if re.match(name, s.name) or name in s.name]


def tearsheet(
    session: TradingSessionResult | List[TradingSessionResult],
    compound: bool = True,
    account_transactions=True,
    performance_statistics_period=365,
    timeframe: str | pd.Timedelta | None = None,
    sort_by: str | None = "Sharpe",
    sort_ascending: bool = False,
    plot_equities: bool = True,
    commission_factor: float = 1,
    plot_leverage: bool = False,
    use_plotly: bool = False,
    no_title: bool = False,
):
    """
    Generate a tearsheet for one or multiple trading sessions.

    This function creates a performance report and visualization for trading session(s).
    It can handle both single and multiple sessions, providing different outputs accordingly.

    Parameters:
    -----------
    session : TradingSessionResult | List[TradingSessionResult]
        The trading session(s) to analyze. Can be a single session or a list of sessions.
    compound : bool, optional
        Whether to use compound returns for charting (default is True).
    account_transactions : bool, optional
        Whether to account for transactions in calculations (default is True).
    performance_statistics_period : int, optional
        The period for performance statistics calculations in days (default is 365).
    timeframe : str | pd.Timedelta, optional
        The timeframe for resampling data. If None, it will be estimated (default is None).
    sort_by : str, optional
        The metric to sort multiple sessions by (default is "Sharpe").
    sort_ascending : bool, optional
        Whether to sort in ascending order (default is False).
    plot_equities : bool, optional
        Whether to plot equity curves for multiple sessions (default is True).
    commission_factor : float, optional
        Factor to apply to commissions (default is 1).
    use_plotly : bool, optional
        Whether to use Plotly for visualizations instead of Matplotlib (default is plotly).

    Returns:
    --------
    For a single session:
        A Plotly or Matplotlib visualization of the session's performance.
    For multiple sessions:
        A pandas DataFrame containing performance metrics for all sessions,
        optionally accompanied by a plot of equity curves.
    """
    if timeframe is None:
        timeframe = _estimate_timeframe(session)

    if isinstance(session, list):
        if len(session) == 1:
            return _tearsheet_single(
                session[0],
                compound,
                account_transactions,
                performance_statistics_period,
                timeframe=timeframe,
                commission_factor=commission_factor,
                use_plotly=use_plotly,
                plot_leverage=plot_leverage,
                no_title=no_title,
            )
        else:
            import matplotlib.pyplot as plt

            # multiple sessions - just show table
            _rs = []
            # _eq = []
            for s in session:
                report, mtrx = _pfl_metrics_prepare(
                    s, account_transactions, performance_statistics_period, commission_factor=commission_factor
                )
                _rs.append(report)
                if plot_equities:
                    if compound:
                        # _eq.append(pd.Series(100 * mtrx["compound_returns"], name=s.trading_id))
                        compound_returns = mtrx["compound_returns"].resample(timeframe).ffill()
                        plt.plot(100 * compound_returns, label=s.name)
                    else:
                        # _eq.append(pd.Series(mtrx["equity"], name=s.trading_id))
                        equity = mtrx["equity"].resample(timeframe).ffill()
                        plt.plot(equity, label=s.name)

            if plot_equities:
                if len(session) <= 15:
                    plt.legend(ncol=max(1, len(session) // 5))
                plt.title("Comparison of Equity Curves")

            report = pd.concat(_rs, axis=1).T
            report["id"] = [s.id for s in session]
            report = report.set_index("id", append=True).swaplevel()
            if sort_by:
                report = report.sort_values(by=sort_by, ascending=sort_ascending)
            return report

    else:
        return _tearsheet_single(
            session,
            compound,
            account_transactions,
            performance_statistics_period,
            timeframe=timeframe,
            commission_factor=commission_factor,
            use_plotly=use_plotly,
            plot_leverage=plot_leverage,
            no_title=no_title,
        )


def get_cum_pnl(
    sessions: TradingSessionResult | list[TradingSessionResult],
    account_transactions: bool = True,
    timeframe: str | None = None,
) -> pd.DataFrame | pd.Series:
    if timeframe is None:
        timeframe = _estimate_timeframe(sessions)

    def _get_single_equity(session: TradingSessionResult) -> pd.Series:
        pnl = calculate_total_pnl(session.portfolio_log, split_cumulative=False)
        pnl["Total_PnL"] = pnl["Total_PnL"].cumsum()
        if account_transactions:
            pnl["Total_PnL"] -= pnl["Total_Commissions"].cumsum()
        returns = portfolio_returns(pnl, init_cash=session.capital)
        return ((returns + 1).cumprod(axis=0) - 1).resample(timeframe).ffill().rename(session.name)

    if isinstance(sessions, list):
        return pd.concat([_get_single_equity(s) for s in sessions], axis=1, names=[s.name for s in sessions])
    else:
        return _get_single_equity(sessions)


def _estimate_timeframe(
    session: TradingSessionResult | list[TradingSessionResult], start: str | None = None, stop: str | None = None
) -> str:
    session = session[0] if isinstance(session, list) else session
    start, end = pd.Timestamp(start or session.start), pd.Timestamp(stop or session.stop)
    diff = end - start
    if diff > pd.Timedelta("360d"):
        return "1d"
    elif diff > pd.Timedelta("30d"):
        return "1h"
    elif diff > pd.Timedelta("7d"):
        return "15min"
    elif diff > pd.Timedelta("1d"):
        return "5min"
    else:
        return "1min"


def _pfl_metrics_prepare(
    session: TradingSessionResult,
    account_transactions: bool,
    performance_statistics_period: int,
    commission_factor: float = 1,
) -> Tuple[pd.Series, dict]:
    mtrx = portfolio_metrics(
        session.portfolio_log,
        session.executions_log,
        session.capital,
        performance_statistics_period=performance_statistics_period,
        account_transactions=account_transactions,
        commission_factor=commission_factor,
    )
    rpt = {}
    for k, v in mtrx.items():
        if isinstance(v, (float, int, str)):
            n = (k[0].upper() + k[1:]).replace("_", " ")
            rpt[n] = v if np.isfinite(v) else 0
    return pd.Series(rpt, name=session.name), mtrx


def _tearsheet_single(
    session: TradingSessionResult,
    compound: bool = True,
    account_transactions=True,
    performance_statistics_period=365,
    timeframe: str | pd.Timedelta = "1h",
    commission_factor: float = 1,
    use_plotly: bool = True,
    plot_leverage: bool = False,
    no_title=False,
):
    report, mtrx = _pfl_metrics_prepare(
        session, account_transactions, performance_statistics_period, commission_factor=commission_factor
    )
    eqty = 100 * mtrx["compound_returns"] if compound else mtrx["equity"] - mtrx["equity"].iloc[0]
    eqty = eqty.resample(timeframe).ffill()
    _eqty = ["area", "green", eqty]
    dd = mtrx["drawdown_pct"] if compound else mtrx["drawdown_usd"]
    dd = dd.resample(timeframe).ffill()

    # - make plotly charts
    if use_plotly:
        _dd = ["area", -dd, "lim", [-dd, 0]]
        tbl = go.Table(
            columnwidth=[130, 130, 130, 130, 200],
            header=dict(
                values=report.index,
                line_color="darkslategray",
                fill_color="#303030",
                font=dict(color="white", size=11),
            ),
            cells=dict(
                values=round(report, 3).values.tolist(),
                line_color="darkslategray",
                fill_color="#101010",
                align=["center", "left"],
                font=dict(size=10),
            ),
        )
        chart = (
            LookingGlass(
                _eqty,
                {
                    "Drawdown": _dd,
                },
                study_plot_height=75,
            )
            .look(title=("Simulation: " if session.is_simulation else "") + session.name)
            .hover(h=500)
        )
        table = go.FigureWidget(tbl).update_layout(margin=dict(r=5, l=5, t=0, b=1), height=80)
        chart.show()
        table.show()
    # - make mpl charts
    else:
        _n = 51 if plot_leverage else 41
        ax = sbp(_n, 1, r=3)
        plt.plot(eqty, lw=2, c="g", label="Equity")
        plt.fill_between(eqty.index, eqty.values, 0, color="#003000", alpha=0.8)
        if not no_title:
            from textwrap import wrap

            _titl_txt = ("Simulation: " if session.is_simulation else "") + session.name
            plt.title("\n".join(wrap(_titl_txt, 60)), fontsize=18)
        plt.legend()
        ay = sbp(_n, 4)
        plt.plot(-dd, c="r", lw=1.5, label="Drawdown")
        plt.fill_between(dd.index, -dd.values, 0, color="#800000", alpha=0.8)
        if not compound:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: str(y / 1000) + " K"))
            ay.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: str(y / 1000) + " K"))
        if plot_leverage:
            init_capital = session.get_total_capital()
            lev = calculate_leverage(session.portfolio_log, init_capital, session.start)
            ay = sbp(_n, 5)
            plt.plot(lev, c="c", lw=1.5, label="Leverage")
        plt.subplots_adjust(hspace=0)
        return pd.DataFrame(report).T.round(3)


def calculate_leverage(
    portfolio: pd.DataFrame, init_capital: float, start: str | pd.Timestamp, symbol=".*"
) -> pd.Series:
    total_pnl = calculate_total_pnl(portfolio, split_cumulative=False).loc[start:]
    capital = init_capital + total_pnl["Total_PnL"].cumsum() - total_pnl["Total_Commissions"].cumsum()
    value = portfolio.filter(regex=f"{symbol}_Value").loc[start:].sum(axis=1)
    return (value.squeeze() / capital).mul(100).rename("Leverage")  # type: ignore


def chart_signals(
    result: TradingSessionResult,
    symbol: str,
    ohlc: dict | pd.DataFrame,
    timeframe: str | None = None,
    start=None,
    end=None,
    apply_commissions: bool = True,
    indicators={},
    overlay=[],
    info=True,
    show_trades: bool = True,
    show_signals: bool = False,
    show_quantity: bool = False,
    show_value: bool = False,
    show_leverage: bool = True,
    show_table: bool = False,
    show_portfolio: bool = True,
    height: int = 800,
    plugins: list[Callable[[LookingGlass, pd.DataFrame, str | pd.Timestamp, str | pd.Timestamp], LookingGlass]]
    | None = None,
):
    """
    Show trading signals on chart
    """
    indicators = indicators | {}
    if timeframe is None:
        timeframe = _estimate_timeframe(result, start, end)

    executions = result.executions_log.rename(columns={"filled_qty": "quantity", "price": "exec_price"})
    portfolio = result.portfolio_log

    if start is None:
        start = executions.index[0]

    if end is None:
        end = executions.index[-1]

    init_capital = result.get_total_capital()

    if portfolio is not None and show_portfolio:
        if show_quantity:
            pos = portfolio.filter(regex=f"{symbol}_Pos").loc[start:]
            indicators["Pos"] = ["area", "cyan", pos]
        if show_value:
            value = portfolio.filter(regex=f"{symbol}_Value").loc[start:]
            indicators["Value"] = ["area", "cyan", value]
        if show_leverage:
            leverage = calculate_leverage(portfolio, init_capital, start, symbol)
            indicators["Leverage"] = ["area", "cyan", leverage]
        # symbol_count = len(portfolio.filter(like="_PnL").columns)
        pnl = portfolio.filter(regex=f"{symbol}_PnL").cumsum() + init_capital  # / symbol_count
        pnl = pnl.loc[start:]
        if apply_commissions:
            comm = portfolio.filter(regex=f"{symbol}_Commissions").loc[start:].cumsum()
            pnl -= comm.values
        pnl = (pnl / pnl.iloc[0] - 1) * 100
        indicators["PnL"] = ["area", "green", pnl]

    if isinstance(ohlc, dict):
        bars = ohlc[symbol]
        if isinstance(bars, OHLCV):
            bars = bars.pd()
        bars = ohlc_resample(bars, timeframe) if timeframe else bars
    elif isinstance(ohlc, pd.DataFrame):
        bars = ohlc
        bars = ohlc_resample(bars, timeframe) if timeframe else bars
    elif isinstance(ohlc, OHLCV):
        bars = ohlc.pd()
        bars = ohlc_resample(bars, timeframe) if timeframe else bars
    else:
        raise ValueError(f"Invalid data type {type(ohlc)}")

    if timeframe:

        def __resample(ind):
            if isinstance(ind, list):
                return [__resample(i) for i in ind]
            elif isinstance(ind, pd.Series) or isinstance(ind, pd.DataFrame):
                return ind.resample(timeframe).ffill()
            else:
                return ind

        indicators = {k: __resample(v) for k, v in indicators.items()}

    if show_trades:
        excs = executions[executions["symbol"] == symbol][
            ["quantity", "exec_price", "commissions", "commissions_quoted", "order_id"]
        ]
        overlay = list(overlay) + [excs]

    if show_signals:
        sigs = result.signals_log[result.signals_log["symbol"] == symbol]
        overlay = list(overlay) + [sigs]

    chart = LookingGlass([bars, *overlay], indicators).look(start, end, title=symbol).hover(show_info=info, h=height)

    # - run plugins
    if plugins is not None:
        for plugin in plugins if isinstance(plugins, list) else [plugins]:
            chart = plugin(bars, start, end, figure=chart)

    if not show_table:
        return chart  # .show()

    q_pos = excs["quantity"].cumsum()[start:end]
    excs = excs[start:end]
    colors = ["red" if t == 0 else "green" for t in q_pos]

    tbl = go.Table(
        # columnorder = [1,2],
        columnwidth=[200, 150, 150, 100, 100],
        header=dict(
            values=["time"] + list(excs.columns),
            line_color="darkslategray",
            fill_color="#303030",
            font=dict(color="white", size=11),
        ),
        cells=dict(
            values=[excs.index.strftime("%Y-%m-%d %H:%M:%S")] + list(excs.T.values),
            line_color="darkslategray",
            fill_color="#101010",
            align=["center", "left"],
            font=dict(color=[colors], size=10),
        ),
    )
    table = go.FigureWidget(tbl).update_layout(margin=dict(r=5, l=5, t=5, b=5), height=200)
    return chart.show(), table.show()


def get_symbol_pnls(
    session: TradingSessionResult | List[TradingSessionResult],
) -> pd.DataFrame:
    if isinstance(session, TradingSessionResult):
        session = [session]

    pnls = []
    for s in session:
        pnls.append(s.portfolio_log.filter(like="_PnL").cumsum().iloc[-1])

    return pd.DataFrame(pnls, index=[s.name for s in session])


def combine_sessions(sessions: list[TradingSessionResult], name: str = "Portfolio") -> TradingSessionResult:
    """
    DEPRECATED: use extend_trading_results instead
    """
    session = copy(sessions[0])
    session.name = name
    session.instruments = list(set(chain.from_iterable([e.instruments for e in sessions])))
    session.capital = sessions[0].get_total_capital() * len(sessions)
    session.portfolio_log = pd.concat(
        [e.portfolio_log.loc[:, (e.portfolio_log != 0).any(axis=0)] for e in sessions], axis=1
    )
    # remove duplicated columns, keep first
    session.portfolio_log = session.portfolio_log.loc[:, ~session.portfolio_log.columns.duplicated()]
    session.executions_log = pd.concat([s.executions_log for s in sessions], axis=0).sort_index()
    session.signals_log = pd.concat([s.signals_log for s in sessions], axis=0).sort_index()
    # remove duplicated rows
    session.executions_log = (
        session.executions_log.set_index("symbol", append=True).drop_duplicates().reset_index("symbol")
    )
    session.signals_log = session.signals_log.set_index("symbol", append=True).drop_duplicates().reset_index("symbol")
    return session


def extend_trading_results(results: list[TradingSessionResult]) -> TradingSessionResult:
    """
    Combine multiple trading session results into a single result by extending the sessions.
    """
    import os

    pfls, execs, exch, names, instrs, clss = [], [], [], [], [], []
    cap = 0.0

    for b in sorted(results, key=lambda x: x.start):
        pfls.append(b.portfolio_log)
        execs.append(b.executions_log)
        exch.extend(b.exchanges)
        names.append(b.name)
        cap += b.get_total_capital()
        instrs.extend(b.instruments)
        clss.append(b.strategy_class)
    cmn = os.path.commonprefix(names)
    names = [x[len(cmn) :] for x in names]
    f_pfls: pd.DataFrame = srows(*pfls, keep="last")  # type: ignore
    f_execs: pd.DataFrame = srows(*execs, keep="last")  # type: ignore
    r = TradingSessionResult(
        0,
        cmn + "-".join(names),
        start=f_pfls.index[0],
        stop=f_pfls.index[-1],
        exchanges=list(set(exch)),
        capital=cap / len(results),  # average capital ???
        instruments=list(set(instrs)),
        base_currency=results[0].base_currency,
        commissions=results[0].commissions,  # what if different commissions ???
        portfolio_log=f_pfls,
        executions_log=f_execs,
        targets_log=pd.DataFrame(),
        signals_log=pd.DataFrame(),
        strategy_class="-".join(set(clss)),  # what if different strategy classes ???
    )
    return r


def _plt_to_base64() -> str:
    fig = plt.gcf()

    imgdata = BytesIO()
    plt.subplots_adjust(hspace=0)
    fig.savefig(imgdata, format="png", transparent=True, bbox_inches="tight")
    # fig.savefig(imgdata, format="png", transparent=True)
    imgdata.seek(0)
    uri = "data:image/png;base64," + base64.b64encode(imgdata.getvalue()).decode("utf8")
    plt.clf()
    plt.close()

    return uri
