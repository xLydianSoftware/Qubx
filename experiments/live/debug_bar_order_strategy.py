"""
Bare-minimum strategy that logs every bar update to debug out-of-order delivery.
"""

from datetime import datetime, timezone

from qubx import logger
from qubx.core.basics import Signal, TriggerEvent
from qubx.core.interfaces import Instrument, IStrategy, IStrategyContext, IStrategyInitializer, MarketEvent


class DebugBarOrderStrategy(IStrategy):
    """Logs every market data update and event trigger with timestamps for debugging."""

    live_base_subscription: str = "ohlc(1m)"
    live_event_schedule: str = "1min -1s"

    _last_bar_time: dict  # per instrument: last bar timestamp seen

    def on_init(self, initializer: IStrategyInitializer):
        self._last_bar_time = {}
        initializer.set_base_subscription(self.live_base_subscription)

    def on_start(self, ctx: IStrategyContext):
        logger.info(f"[DEBUG_STRATEGY] Started with {len(ctx.instruments)} instruments")

    def on_warmup_finished(self, ctx: IStrategyContext):
        logger.info("[DEBUG_STRATEGY] Warmup finished")
        if self.live_event_schedule:
            ctx.set_event_schedule(self.live_event_schedule)
        # log current OHLCV state
        for i in ctx.instruments:
            ohlcv = ctx.ohlc(i)
            if ohlcv and ohlcv.times:
                logger.info(
                    f"[DEBUG_STRATEGY] {i.symbol} OHLCV head: "
                    f"time={datetime.fromtimestamp(ohlcv.times[0] / 1e9, tz=timezone.utc).strftime('%H:%M:%S')} "
                    f"close={ohlcv.close[0]:.2f} len={len(ohlcv.times)}"
                )

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        if not hasattr(data.data, "time"):
            return None

        instrument = data.instrument
        if instrument is None:
            return None

        bar_time_ns = data.data.time
        sym = instrument.symbol
        prev = self._last_bar_time.get(sym)

        status = ""
        if prev is not None:
            if bar_time_ns < prev:
                status = "*** OUT OF ORDER ***"
                logger.warning(
                    f"[DEBUG_STRATEGY] {sym} OUT OF ORDER bar! "
                    f"bar_time={datetime.fromtimestamp(bar_time_ns / 1e9, tz=timezone.utc).strftime('%H:%M:%S.%f')[:-3]} "
                    f"prev={datetime.fromtimestamp(prev / 1e9, tz=timezone.utc).strftime('%H:%M:%S.%f')[:-3]} "
                    f"dtype={data.type}"
                )
            elif bar_time_ns > prev:
                status = ">>> NEW CANDLE"
                logger.info(
                    f"[DEBUG_STRATEGY] {sym} NEW CANDLE "
                    f"bar_time={datetime.fromtimestamp(bar_time_ns / 1e9, tz=timezone.utc).strftime('%H:%M:%S')} "
                    f"close={data.data.close:.2f} dtype={data.type}"
                )

        self._last_bar_time[sym] = bar_time_ns
        return None

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        return None
