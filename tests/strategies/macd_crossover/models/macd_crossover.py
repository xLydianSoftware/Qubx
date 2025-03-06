from macd_crossover.indicators.macd import Macd, macd

from qubx import logger
from qubx.core.basics import DataType, Instrument, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.trackers import StopTakePositionTracker
from qubx.trackers.sizers import FixedLeverageSizer


class MacdCrossoverStrategy(IStrategy):
    """
    MACD Crossover Strategy.
    """

    timeframe: str = "1h"
    leverage: float = 1.0

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    take_target: float = 2
    stop_risk: float = 1

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return StopTakePositionTracker(
            take_target=self.take_target, stop_risk=self.stop_risk, sizer=FixedLeverageSizer(self.leverage)
        )

    def on_init(self, ctx: IStrategyContext) -> None:
        ctx.set_base_subscription(DataType.OHLC[self.timeframe])
        self._indicators: dict[Instrument, Macd] = {}

    def on_start(self, ctx: IStrategyContext) -> None:
        for i in ctx.instruments:
            self._indicators[i] = macd(
                ctx.ohlc(i, self.timeframe).close,
                fast=self.fast_period,
                slow=self.slow_period,
                signal=self.signal_period,
            )

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        signals = []

        # Process each instrument
        for instrument in ctx.instruments:
            # Get current MACD values
            buy_signal, sell_signal = self._check_crossover(instrument.symbol, self._indicators[instrument])
            _price = ctx.ohlc(instrument, self.timeframe, 2 * self.slow_period).close[0]

            if buy_signal:
                signals.append(instrument.signal(1, comment="MACD crossed above signal line"))
                logger.info(f"<g>BUY signal for {instrument.symbol} at {_price}</g>")

            elif sell_signal:
                signals.append(instrument.signal(-1, comment="MACD crossed below signal line"))
                logger.info(f"<r>SELL signal for {instrument.symbol} at {_price}</r>")

        return signals

    def _check_crossover(self, instrument_symbol: str, macd_indicator: Macd) -> tuple[bool, bool]:
        """Check for MACD crossover signals."""
        if len(macd_indicator.signal_line) < 2:
            return False, False

        _m0, _m1 = macd_indicator.macd_series[0], macd_indicator.macd_series[1]
        _s0, _s1 = macd_indicator.signal_line[0], macd_indicator.signal_line[1]

        if _m0 is None or _m1 is None or _s0 is None or _s1 is None:
            return False, False

        buy_signal = _m0 > _s0 and _m1 <= _s1
        sell_signal = _m0 < _s0 and _m1 >= _s1

        return buy_signal, sell_signal
