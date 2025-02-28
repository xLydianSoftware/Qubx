from qubx import logger
from qubx.core.basics import DataType, Instrument, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.core.series import OrderBook
from qubx.features.core import FeatureManager
from qubx.features.orderbook import OrderbookImbalance
from qubx.trackers.sizers import FixedLeverageSizer


class ObiTraderStrategy(IStrategy):
    """
    OrderBook Imbalance (OBI) Trading Strategy.

    This strategy enters long positions when orderbook imbalance is positive
    and short positions when imbalance is negative.
    """

    timeframe: str = "1s"  # Timeframe for orderbook imbalance calculation
    leverage: float = 1.0  # Fixed leverage for positions
    depth: int = 10  # Depth level for orderbook imbalance calculation
    threshold: float = 0.1  # Threshold for signal generation (absolute value)

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker:
        return PositionsTracker(FixedLeverageSizer(self.leverage))

    def on_init(self, ctx: IStrategyContext) -> None:
        # Subscribe to orderbook updates
        ctx.set_base_subscription(DataType.ORDERBOOK)

        # Initialize the feature manager
        self._feature_manager = FeatureManager()

        # Initialize the OBI feature provider and add it to the feature manager
        self._obi_provider = OrderbookImbalance(timeframe=self.timeframe)
        self._feature_manager += self._obi_provider  # Register using the + operator

        # Dictionary to store the latest OBI values for each instrument
        self._latest_obi = {}

    def on_start(self, ctx: IStrategyContext) -> None:
        # Initialize OBI values for each instrument
        for instrument in ctx.instruments:
            self._latest_obi[instrument] = 0.0

        # Start the feature manager
        self._feature_manager.on_start(ctx)

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal]:
        signals = []

        # Process orderbook events
        if hasattr(event, "data") and isinstance(event.data, OrderBook):
            orderbook = event.data
            instrument = event.instrument

            if instrument is None or orderbook is None:
                return signals

            # Process the orderbook with the feature manager
            features = self._feature_manager.process_orderbook(instrument, orderbook)

            # Check if we have OBI features
            obi_feature_name = self._obi_provider.get_output_name(self.depth)
            if features and obi_feature_name in features:
                obi_value = features[obi_feature_name]
                self._latest_obi[instrument] = obi_value

                # Get current price from orderbook
                current_price = orderbook.mid_price()

                # Generate signals based on OBI value
                if abs(obi_value) >= self.threshold:
                    if obi_value > 0:
                        # Positive imbalance - go long
                        signals.append(instrument.signal(1, comment=f"OBI long signal: {obi_value:.4f}"))
                        logger.info(
                            f"<g>BUY signal for {instrument.symbol} at {current_price} (OBI: {obi_value:.4f})</g>"
                        )
                    else:
                        # Negative imbalance - go short
                        signals.append(instrument.signal(-1, comment=f"OBI short signal: {obi_value:.4f}"))
                        logger.info(
                            f"<r>SELL signal for {instrument.symbol} at {current_price} (OBI: {obi_value:.4f})</r>"
                        )

        return signals
