from qubx.core.series import OHLCV, Indicator, IndicatorOHLC, TimeSeries

def sma(series: TimeSeries, period: int): ...
def ema(series: TimeSeries, period: int, init_mean: bool = True): ...
def tema(series: TimeSeries, period: int, init_mean: bool = True): ...
def dema(series: TimeSeries, period: int, init_mean: bool = True): ...
def kama(series: TimeSeries, period: int, fast_span: int = 2, slow_span: int = 30): ...
def highest(series: TimeSeries, period: int): ...
def lowest(series: TimeSeries, period: int): ...
def std(series: TimeSeries, period: int, mean=0): ...
def zscore(series: TimeSeries, period: int, smoother: str = "sma"): ...
def pewma(series: TimeSeries, alpha: float, beta: float, T: int = 30): ...
def pewma_outliers_detector(series: TimeSeries, alpha: float, beta: float, T: int = 30, threshold=0.05, **kwargs): ...
def psar(series: OHLCV, iaf: float = 0.02, maxaf: float = 0.2): ...
def smooth(series: TimeSeries, smoother: str, *args, **kwargs) -> Indicator: ...
def atr(series: OHLCV, period: int = 14, smoother="sma", percentage: bool = False): ...
def swings(series: OHLCV, trend_indicator, **indicator_args) -> Indicator: ...

class Sma(Indicator):
    def __init__(self, name: str, series: TimeSeries, period: int): ...

class Std(Indicator):
    def __init__(self, name: str, series: TimeSeries, period: int): ...

class Zscore(Indicator):
    def __init__(self, name: str, series: TimeSeries, period: int, smoother: str): ...

class Ema(Indicator):
    def __init__(self, name: str, series: TimeSeries, period: int, init_mean: bool = True): ...

class Kama(Indicator):
    def __init__(self, name: str, series: TimeSeries, period: int, fast_span: int = 2, slow_span: int = 30): ...

class Atr(IndicatorOHLC):
    def __init__(self, name: str, series: OHLCV, period: int, smoother: str, percentage: bool): ...

class Swings(IndicatorOHLC):
    tops: TimeSeries
    bottoms: TimeSeries
    middles: TimeSeries
    deltas: TimeSeries
