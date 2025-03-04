import numpy as np

from qubx import logger
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.core.series import OrderBook, TimeSeries
from qubx.features.core import FeatureManager
from qubx.features.orderbook import OrderbookImbalance
from qubx.ta.indicators import zscore
from qubx.trackers.sizers import FixedLeverageSizer


class ObiTraderStrategy(IStrategy):
    """
    OrderBook Imbalance (OBI) Trading Strategy.

    This strategy enters long positions when orderbook imbalance is positive
    and short positions when imbalance is negative.
    """

    timeframe: str = "1s"  # Timeframe for orderbook imbalance calculation
    leverage: float = 1.0  # Fixed leverage for positions
    tick_size_pct: float = 0.01  # Tick size for orderbook imbalance calculation
    depth: int = 1000  # Depth level for orderbook imbalance calculation
    threshold: float = 0.1  # Threshold for signal generation (absolute value)
    zscore_period: int = 3600  # Period for zscore calculation

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return PositionsTracker(FixedLeverageSizer(self.leverage))

    def on_init(self, ctx: IStrategyContext) -> None:
        # Subscribe to orderbook updates
        ctx.set_base_subscription(DataType.ORDERBOOK[self.tick_size_pct, self.depth])
        # ctx.subscribe(DataType.QUOTE)
        ctx.set_event_schedule("1s")

        # Initialize the feature manager
        self._feature_manager = FeatureManager(max_series_length=100_000)
        self._obi_provider = OrderbookImbalance(timeframe=self.timeframe, depths=[self.depth])
        self._feature_manager += self._obi_provider

    def on_start(self, ctx: IStrategyContext) -> None:
        # Start the feature manager
        self._feature_manager.on_start(ctx)
        self._instrument = ctx.instruments[0]
        self._obi: TimeSeries = self._feature_manager.get_feature(
            instrument=self._instrument, feature_name=self._obi_provider.outputs()[0]
        )
        self._zscore_obi = zscore(self._obi, period=self.zscore_period, smoother="sma")

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        self._feature_manager.on_market_data(ctx, data)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        signals = []

        # Get current price from orderbook
        q = ctx.quote(self._instrument)
        assert q is not None
        current_price = q.mid_price()

        # Generate signals based on OBI value
        # Check if we have at least 2 OBI values to detect crossovers
        if len(self._zscore_obi) < 2:
            return signals

        if np.isnan(self._zscore_obi[0]) or np.isnan(self._zscore_obi[1]):
            return signals

        pos_qty = ctx.positions[self._instrument].quantity

        current_obi = self._zscore_obi[0]  # Latest OBI value
        previous_obi = self._zscore_obi[1]  # Previous OBI value

        # Detect crossover above threshold (positive)
        if pos_qty <= 0 and current_obi >= self.threshold and previous_obi < self.threshold:
            # Positive threshold crossover - go long
            signals.append(self._instrument.signal(1, comment=f"OBI threshold crossover: {current_obi:.4f}"))
            logger.info(
                f"<g>BUY signal for {self._instrument.symbol} at {current_price} (OBI crossover: {current_obi:.4f})</g>"
            )

        # Detect crossunder below negative threshold
        elif pos_qty >= 0 and current_obi <= -self.threshold and previous_obi > -self.threshold:
            # Negative threshold crossunder - go short
            signals.append(self._instrument.signal(-1, comment=f"OBI threshold crossunder: {current_obi:.4f}"))
            logger.info(
                f"<r>SELL signal for {self._instrument.symbol} at {current_price} (OBI crossunder: {current_obi:.4f})</r>"
            )

        return signals
