import pandas as pd

from qubx import logger
from qubx.backtester.data import SimulatedDataProvider
from qubx.core.basics import DataType
from qubx.core.interfaces import IStrategyContext

from .utils import SimulationDataConfig


class BacktestContextRunner:
    """
    A wrapper around the StrategyContext that encapsulates the simulation logic.
    This class is responsible for running a backtest context from a start time to an end time.
    """

    def __init__(self, ctx: IStrategyContext, data_provider: SimulatedDataProvider, data_config: SimulationDataConfig):
        """
        Initialize the BacktestContextRunner with a strategy context.

        Args:
            ctx (IStrategyContext): The strategy context to run.
            data_provider (SimulatedDataProvider): The data provider to use.
            data_config (SimulationDataConfig): The data configuration to use.
        """
        self.ctx = ctx
        self.data_provider = data_provider
        self.data_config = data_config

    def run(self, start: pd.Timestamp | str, stop: pd.Timestamp | str, silent: bool = False):
        """
        Run the backtest from start to stop.

        Args:
            start (pd.Timestamp | str): The start time of the simulation.
            stop (pd.Timestamp | str): The end time of the simulation.
            silent (bool, optional): Whether to suppress progress output. Defaults to False.
        """
        start_ts = pd.Timestamp(start)
        stop_ts = pd.Timestamp(stop)

        logger.debug(f"[<y>BacktestContextRunner</y>] :: Running simulation from {start_ts} to {stop_ts}")

        # Start the context
        self.ctx.start()

        # Apply default warmup periods if strategy didn't set them
        for s in self.ctx.get_subscriptions():
            if not self.ctx.get_warmup(s) and (_d_wt := self.data_config.default_warmups.get(s)):
                logger.debug(
                    f"[<y>BacktestContextRunner</y>] :: Strategy didn't set warmup period for <c>{s}</c> so default <c>{_d_wt}</c> will be used"
                )
                self.ctx.set_warmup({s: _d_wt})

        # Subscribe to any custom data types if needed
        def _is_known_type(t: str):
            try:
                DataType(t)
                return True
            except:  # noqa: E722
                return False

        for t, r in self.data_config.data_providers.items():
            if not _is_known_type(t) or t in [
                DataType.TRADE,
                DataType.OHLC_TRADES,
                DataType.OHLC_QUOTES,
                DataType.QUOTE,
                DataType.ORDERBOOK,
            ]:
                logger.debug(f"[<y>BacktestContextRunner</y>] :: Subscribing to: {t}")
                self.ctx.subscribe(t, self.ctx.instruments)

        try:
            # Run the data provider
            self.data_provider.run(start_ts, stop_ts, silent=silent)
        except KeyboardInterrupt:
            logger.error("Simulated trading interrupted by user!")
        finally:
            # Stop the context
            self.ctx.stop()
