from qubx.core.series import Indicator, TimeSeries
from qubx.ta.indicators import Ema


class Macd(Indicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    This indicator uses exponential moving averages to identify trend changes and momentum.
    It consists of:
    - MACD line (fast EMA - slow EMA)
    - Signal line (EMA of MACD line)
    - Histogram (MACD line - Signal line)
    """

    def __init__(self, name: str, series: TimeSeries, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal

        # Initialize EMAs
        self.prices = TimeSeries("prices", series.timeframe, series.max_series_length)

        self.fast_ema = Ema("fast_ema", self.prices, fast)
        self.slow_ema = Ema("slow_ema", self.prices, slow)

        # Create series for MACD components
        self.macd_series = TimeSeries("macd", series.timeframe, series.max_series_length)
        self.signal_line = Ema("signal", self.macd_series, signal)
        self.histogram = TimeSeries("hist", series.timeframe, series.max_series_length)

        super().__init__(name, series)

    def calculate(self, time: int, value: float, new_item_started: bool) -> float | None:
        """Calculate MACD values for each new price update."""
        self.prices.update(time, value)

        # Calculate EMAs
        fast_value = self.fast_ema[0]
        slow_value = self.slow_ema[0]

        if fast_value is None or slow_value is None:
            return None

        # Calculate MACD line
        macd_value = fast_value - slow_value
        self.macd_series.update(time, macd_value)

        # Calculate signal line
        signal_value = self.signal_line[0]
        if signal_value is not None:
            # Calculate histogram
            hist_value = macd_value - signal_value
            self.histogram.update(time, hist_value)

        return signal_value


def macd(series: TimeSeries, fast: int = 12, slow: int = 26, signal: int = 9) -> Macd:
    """
    Create a new MACD indicator instance.

    Args:
        series: Input price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)

    Returns:
        MACD indicator instance
    """
    return Macd.wrap(series, fast, slow, signal)  # type: ignore
