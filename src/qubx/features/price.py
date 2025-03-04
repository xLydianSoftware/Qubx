from qubx.core.basics import Instrument
from qubx.core.interfaces import IMarketManager
from qubx.core.series import TimeSeries
from qubx.ta.indicators import atr

from .core import FeatureProvider


class AtrFeatureProvider(FeatureProvider):
    name: str = "ATR"
    period: int = 14
    smoother: str = "sma"
    percentage: bool = True

    def outputs(self) -> list[str]:
        return [self.get_output_name(self.period, self.timeframe, self.smoother, pct=self.percentage)]

    def on_instrument_added(self, ctx: IMarketManager, instrument: Instrument) -> TimeSeries:
        _ohlc = ctx.ohlc(instrument, self.timeframe)
        return atr(_ohlc, self.period, self.smoother, self.percentage)
