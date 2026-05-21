from collections.abc import Callable

import numpy as np

from qubx.core.series import OHLCV, Bar, Indicator, TimeSeries
from qubx.pandaz.utils import scols
from qubx.utils.time import find_minimal_timeframe, to_timedelta, to_timestamp


def series_from_bars(bars: list[Bar], timeframe: str | None = None) -> OHLCV:
    timeframe = timeframe or find_minimal_timeframe([to_timestamp(x.time) for x in bars])
    ohlc_formed = OHLCV(f"ohlc_{timeframe}", timeframe)
    if bars:
        ohlc_formed.update_by_bars(bars)
    return ohlc_formed


def generate_ohlc_bars(
    timeframe: str, n_bars: int, base_price=30_000.0, start_t="2025-01-01", bars_only=False
) -> list[Bar]:
    _tf = to_timedelta(timeframe).value
    rng = np.random.default_rng(_tf // 100)

    start_t = to_timestamp(start_t).as_unit("ns").asm8.astype(int)
    prices = base_price * np.cumprod(1 + rng.normal(0, 0.002, n_bars))

    bars = []
    for i in range(n_bars):
        c = prices[i]
        o = prices[i - 1] if i > 0 else c
        h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
        v = rng.uniform(10, 100)
        bars.append(Bar(start_t + i * _tf, o, h, l, c, v))

    return bars


def check_indicator_intrabar(
    indicator: Callable[[OHLCV | TimeSeries], Indicator],
    n_bars: int,
    small_timeframe: str = "1Min",
    working_timeframe: str = "15Min",
    initial_data: int = 0,
    tolerance: float = 1e-4,
) -> None:
    """
    Assert that a streaming indicator produces identical values in batch and streaming modes.

    Generates `n_bars` of `small_timeframe` bars, aggregates them into `working_timeframe`
    OHLCV series, then compares two execution paths:

    - **Batch**: indicator is attached to a fully pre-formed series (all bars already present).
    - **Streaming**: series starts empty (or with `initial_data` bars), indicator is attached,
      then remaining `small_timeframe` bars are fed one by one via ``update_by_bars``.

    Both paths must agree within `tolerance` at every confirmed bar close. This guards against
    live-vs-backtest silent divergence caused by incorrect ``_store``/``_restore`` logic,
    ``_initial_data_recalculate`` divergence, or manual series (``ser0`` / ``_close``) not
    being updated in streaming mode.

    Args:
        indicator:         Factory callable — receives an ``OHLCV`` (or ``TimeSeries``) and
                           returns the attached ``Indicator``. Called twice: once for the batch
                           series and once for the streaming series.
        n_bars:            Number of ``small_timeframe`` bars to generate. Must be large enough
                           to produce at least a few ``working_timeframe`` bars (e.g. 200 1Min
                           bars → ~13 15Min bars).
        small_timeframe:   Timeframe of generated raw bars fed into the series (default ``1Min``).
        working_timeframe: Timeframe the indicator operates on; must be strictly larger than
                           ``small_timeframe`` (default ``15Min``).
        initial_data:      Number of ``small_timeframe`` bars to pre-load into the streaming
                           series *before* attaching the indicator (default 0). Useful for
                           testing warm-start behaviour.
        tolerance:         Maximum allowed absolute difference between batch and streaming
                           indicator values at any bar close (default ``1e-4``).

    Raises:
        AssertionError: If the streaming series diverges from batch, if the resulting
                        difference series is empty/all-NaN, or if ``max(|diff|) >= tolerance``.

    Examples:
        Basic usage — simple EMA on close:

        >>> from qubx.ta.indicators import ema
        >>> check_indicator_intrabar(
        ...     indicator=lambda s: ema(s.close, 20),
        ...     n_bars=500,
        ... )

        Custom quantkit indicator with non-default timeframes:

        >>> from quantkit.metals.indicators import volatility
        >>> check_indicator_intrabar(
        ...     indicator=lambda s: volatility(s.close, lookback=32, std_timeframe="1h"),
        ...     n_bars=2000,
        ...     small_timeframe="1Min",
        ...     working_timeframe="1h",
        ... )

        Testing with a warm-start (indicator attached after some bars are already present):

        >>> from quantkit.metals.indicators import risk_adjusted_reversal
        >>> check_indicator_intrabar(
        ...     indicator=lambda s: risk_adjusted_reversal(s.close, 48, 32, "1d"),
        ...     n_bars=3000,
        ...     small_timeframe="1Min",
        ...     working_timeframe="15Min",
        ...     initial_data=500,
        ... )
    """
    assert to_timedelta(small_timeframe) < to_timedelta(working_timeframe), (
        "Small timeframe must be < working timeframe"
    )
    assert initial_data < n_bars, "Init data index must be less than n_bars !"
    m1bars = generate_ohlc_bars(small_timeframe, n_bars)

    # - create formed series (batch)
    m15_batch = series_from_bars(m1bars, working_timeframe)

    # - run indicator
    i_batch = indicator(m15_batch)

    # - create emty series for working timeframe
    m15_dyn = series_from_bars([], working_timeframe)

    # - put initial data into series
    for b in m1bars[:initial_data]:
        m15_dyn.update_by_bars([b])

    # - attach indicator to series
    i_dyn = indicator(m15_dyn)

    # - update intrabar
    for b in m1bars[initial_data:]:
        m15_dyn.update_by_bars([b])

    assert np.allclose(m15_dyn.pd().dropna(), m15_batch.pd().dropna()), "Dynamic generated series differs from batch"

    s_diff = (i_batch.pd() - i_dyn.pd()).dropna()
    assert len(s_diff) > 0, "Resulting indicators difference generated all nans or empty series"
    # assert sum(abs(s_diff)) < tolerance, f"Difference is too big: {sum(s_diff)} vs tolerance {tolerance} !"
    if s_diff.abs().max() >= tolerance:
        print(scols(i_batch.pd(), i_dyn.pd()).dropna())
        assert False, f"Difference is too big: {sum(s_diff)} vs tolerance {tolerance} !"
