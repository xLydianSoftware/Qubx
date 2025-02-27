import numpy as np
import pandas as pd


def percentile_rank(x: np.ndarray, v, pctls=np.arange(1, 101)):
    """
    Find percentile rank of value v
    :param x: values array
    :param v: vakue to be ranked
    :param pctls: percentiles
    :return: rank

    >>> percentile_rank(np.random.randn(1000), 1.69)
    >>> 95
    >>> percentile_rank(np.random.randn(1000), 1.69, [10,50,100])
    >>> 2
    """
    return np.argmax(np.sign(np.append(np.percentile(x, pctls), np.inf) - v))


def compare_to_norm(xs, xranges=None):
    """
    Compare distribution from xs against normal using estimated mean and std
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import seaborn as sns

    from qubx.utils.charting.mpl_helpers import sbp

    _m, _s = np.mean(xs), np.std(xs)
    fit = stats.norm.pdf(sorted(xs), _m, _s)

    sbp(12, 1)
    plt.plot(sorted(xs), fit, "r--", lw=2, label="N(%.2f, %.2f)" % (_m, _s))
    plt.legend(loc="upper right")

    sns.kdeplot(xs, color="g", label="Data", fill=True)
    if xranges is not None and len(xranges) > 1:
        plt.xlim(xranges)
    plt.legend(loc="upper right")

    sbp(12, 2)
    stats.probplot(xs, dist="norm", sparams=(_m, _s), plot=plt)


def kde(array, cut_down=True, bw_method="scott"):
    """
    Kernel dense estimation
    """
    from scipy.stats import gaussian_kde

    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        f_mean = counts.mean()
        f_above_mean = bins[counts > f_mean]
        if len(f_above_mean) > 0:
            bounds = [f_above_mean.min(), f_above_mean.max()]
            array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]

    return gaussian_kde(array, bw_method=bw_method)


def hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Calculate the Hurst exponent to determine the long-term memory of a time series.

    The Hurst exponent (H) is a measure that helps identify:
    - Random Walk (H ≈ 0.5): Each step is independent of past values
    - Trending/Persistent (H > 0.5): Positive values tend to be followed by positive values
    - Mean Reverting/Anti-persistent (H < 0.5): Positive values tend to be followed by negative values

    The calculation uses the relationship between the range of the data and the time lag,
    specifically examining how the variance of price differences scales with increasing lags.

    Parameters
    ----------
    series : np.ndarray
        Input time series data (typically price or returns)
    max_lag : int, optional
        Maximum lag to consider in calculation, by default 20

    Returns
    -------
    float
        Hurst exponent value between 0 and 1

    Notes
    -----
    - Values very close to 0 or 1 may indicate issues with the data
    - Requires sufficient data points for reliable estimation
    - Implementation uses variance scaling method
    """
    tau, lagvec = [], []

    # Step through the different lags
    for lag in range(2, max_lag):
        # Produce price different with lag
        pp = np.subtract(series[lag:], series[:-lag])

        # Write the different lags into a vector
        lagvec.append(lag)

        # Calculate the variance of the difference
        tau.append(np.sqrt(np.std(pp)))

    # Linear fit to a double-log graph to get power
    m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)

    # Calculate hurst
    return m[0] * 2


def half_life(price: pd.Series) -> int:
    """
    Half-life is the period of time it takes for the price to revert back to the mean.
    """
    import statsmodels.api as sm

    xs_lag = price.shift(1).bfill()
    xs_ret = price.diff().bfill()
    res = sm.OLS(xs_ret, sm.add_constant(xs_lag)).fit()
    return int(-np.log(2) / res.params.iloc[1])


def cointegration_test(p1: pd.Series, p2: pd.Series, alpha: float = 0.05) -> tuple[bool, float]:
    from statsmodels.tsa.stattools import coint

    p1, p2 = p1.dropna().align(p2.dropna(), join="inner")
    _, pvalue, _ = coint(p1, p2)
    return bool(pvalue < alpha), float(pvalue)
