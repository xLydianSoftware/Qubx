from typing import Any

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import SW, DataType
from qubx.core.context import StrategyContext
from qubx.core.exceptions import SimulationConfigError, SimulationError
from qubx.core.helpers import extract_parameters_from_object, full_qualified_class_name
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IMetricEmitter, IStrategy, IStrategyContext, StrategyState
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.core.lookups import lookup
from qubx.pandaz.utils import _frame_to_str

from .account import SimulatedAccountProcessor
from .broker import SimulatedBroker
from .data import SimulatedDataProvider
from .utils import (
    SetupTypes,
    SignalsProxy,
    SimulatedCtrlChannel,
    SimulatedScheduler,
    SimulatedTimeProvider,
    SimulationDataConfig,
    SimulationSetup,
)


class SimulationRunner:
    """
    A wrapper around the StrategyContext that encapsulates the simulation logic.
    This class is responsible for running a backtest context from a start time to an end time.
    """

    setup: SimulationSetup
    data_config: SimulationDataConfig
    start: pd.Timestamp
    stop: pd.Timestamp
    account_id: str
    portfolio_log_freq: str
    ctx: IStrategyContext
    data_provider: SimulatedDataProvider
    logs_writer: InMemoryLogsWriter

    strategy_params: dict[str, Any]
    strategy_class: str

    # adjusted times
    _stop: pd.Timestamp | None = None

    def __init__(
        self,
        setup: SimulationSetup,
        data_config: SimulationDataConfig,
        start: pd.Timestamp | str,
        stop: pd.Timestamp | str,
        account_id: str = "SimulatedAccount",
        portfolio_log_freq: str = "5Min",
        emitter: IMetricEmitter | None = None,
        strategy_state: StrategyState | None = None,
        initializer: BasicStrategyInitializer | None = None,
    ):
        """
        Initialize the BacktestContextRunner with a strategy context.

        Args:
            setup (SimulationSetup): The setup to run.
            data_config (SimulationDataConfig): The data setup to use.
            start (pd.Timestamp): The start time of the simulation.
            stop (pd.Timestamp): The end time of the simulation.
            account_id (str): The account id to use.
            portfolio_log_freq (str): The portfolio log frequency to use.
            emitter (IMetricEmitter): The emitter to use.
        """
        self.setup = setup
        self.data_config = data_config
        self.start = pd.Timestamp(start)
        self.stop = pd.Timestamp(stop)
        self.account_id = account_id
        self.portfolio_log_freq = portfolio_log_freq
        self.emitter = emitter
        self.strategy_state = strategy_state if strategy_state is not None else StrategyState()
        self.initializer = initializer
        self.ctx = self._create_backtest_context()

        # - get strategy parameters BEFORE simulation start
        #   potentially strategy may change it's parameters during simulation
        self.strategy_params = {}
        self.strategy_class = ""
        if self.setup.setup_type in [SetupTypes.STRATEGY, SetupTypes.STRATEGY_AND_TRACKER]:
            self.strategy_params = extract_parameters_from_object(self.setup.generator)
            self.strategy_class = full_qualified_class_name(self.setup.generator)

    def run(self, silent: bool = False, catch_keyboard_interrupt: bool = True, close_data_readers: bool = False):
        """
        Run the backtest from start to stop.

        Args:
            start (pd.Timestamp | str): The start time of the simulation.
            stop (pd.Timestamp | str): The end time of the simulation.
            silent (bool, optional): Whether to suppress progress output. Defaults to False.
        """
        logger.debug(f"[<y>SimulationRunner</y>] :: Running simulation from {self.start} to {self.stop}")

        # Start the context
        self.ctx.start()

        # Apply default warmup periods if strategy didn't set them
        for s in self.ctx.get_subscriptions():
            if not self.ctx.get_warmup(s) and (_d_wt := self.data_config.default_warmups.get(s)):
                logger.debug(
                    f"[<y>SimulationRunner</y>] :: Strategy didn't set warmup period for <c>{s}</c> so default <c>{_d_wt}</c> will be used"
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

        stop = self._stop or self.stop

        try:
            self.data_provider.run(self.start, stop, silent=silent)
        except KeyboardInterrupt:
            logger.error("Simulated trading interrupted by user!")
            if not catch_keyboard_interrupt:
                raise
        finally:
            # Stop the context
            self.ctx.stop()
            if close_data_readers:
                assert isinstance(self.data_provider, SimulatedDataProvider)
                for reader in self.data_provider._readers.values():
                    if hasattr(reader, "close"):
                        reader.close()  # type: ignore

    def print_latency_report(self) -> None:
        _l_r = SW.latency_report()
        if _l_r is not None:
            logger.info(
                "<BLUE>   Time spent in simulation report   </BLUE>\n<r>"
                + _frame_to_str(
                    _l_r.sort_values("latency", ascending=False).reset_index(drop=True), "simulation", -1, -1, False
                )
                + "</r>"
            )

    def _create_backtest_context(self) -> IStrategyContext:
        tcc = lookup.fees.find(self.setup.exchange.lower(), self.setup.commissions)
        if tcc is None:
            raise SimulationConfigError(
                f"Can't find transaction costs calculator for '{self.setup.exchange}' for specification '{self.setup.commissions}' !"
            )

        channel = SimulatedCtrlChannel("databus", sentinel=(None, None, None, None))
        simulated_clock = SimulatedTimeProvider(np.datetime64(self.start, "ns"))

        logger.debug(
            f"[<y>simulator</y>] :: Preparing simulated trading on <g>{self.setup.exchange.upper()}</g> for {self.setup.capital} {self.setup.base_currency}..."
        )

        account = SimulatedAccountProcessor(
            account_id=self.account_id,
            channel=channel,
            base_currency=self.setup.base_currency,
            initial_capital=self.setup.capital,
            time_provider=simulated_clock,
            tcc=tcc,
            accurate_stop_orders_execution=self.setup.accurate_stop_orders_execution,
        )
        scheduler = SimulatedScheduler(channel, lambda: simulated_clock.time().item())
        broker = SimulatedBroker(channel, account, self.setup.exchange)
        data_provider = SimulatedDataProvider(
            exchange_id=self.setup.exchange,
            channel=channel,
            scheduler=scheduler,
            time_provider=simulated_clock,
            account=account,
            readers=self.data_config.data_providers,
            open_close_time_indent_secs=self.data_config.adjusted_open_close_time_indent_secs,
        )
        # - get aux data provider
        _aux_data = self.data_config.get_timeguarded_aux_reader(simulated_clock)
        # - it will store simulation results into memory
        logs_writer = InMemoryLogsWriter(self.account_id, self.setup.name, "0")

        # - it will store simulation results into memory
        strat: IStrategy | None = None

        match self.setup.setup_type:
            case SetupTypes.STRATEGY:
                strat = self.setup.generator  # type: ignore

            case SetupTypes.STRATEGY_AND_TRACKER:
                strat = self.setup.generator  # type: ignore
                strat.tracker = lambda ctx: self.setup.tracker  # type: ignore

            case SetupTypes.SIGNAL:
                strat = SignalsProxy(timeframe=self.setup.signal_timeframe)
                data_provider.set_generated_signals(self.setup.generator)  # type: ignore

                # - we don't need any unexpected triggerings
                self._stop = min(self.setup.generator.index[-1], self.stop)  # type: ignore

            case SetupTypes.SIGNAL_AND_TRACKER:
                strat = SignalsProxy(timeframe=self.setup.signal_timeframe)
                strat.tracker = lambda ctx: self.setup.tracker
                data_provider.set_generated_signals(self.setup.generator)  # type: ignore

                # - we don't need any unexpected triggerings
                self._stop = min(self.setup.generator.index[-1], self.stop)  # type: ignore

            case _:
                raise SimulationError(f"Unsupported setup type: {self.setup.setup_type} !")

        if not isinstance(strat, IStrategy):
            raise SimulationConfigError(f"Strategy should be an instance of IStrategy, but got {strat} !")

        ctx = StrategyContext(
            strategy=strat,
            broker=broker,
            data_provider=data_provider,
            account=account,
            scheduler=scheduler,
            time_provider=simulated_clock,
            instruments=self.setup.instruments,
            logging=StrategyLogging(logs_writer, portfolio_log_freq=self.portfolio_log_freq),
            aux_data_provider=_aux_data,
            emitter=self.emitter,
            strategy_state=self.strategy_state,
            initializer=self.initializer,
        )

        if self.emitter is not None:
            self.emitter.set_time_provider(simulated_clock)

        # - setup base subscription from spec
        if ctx.get_base_subscription() == DataType.NONE:
            logger.debug(
                f"[<y>simulator</y>] :: Setting up default base subscription: {self.data_config.default_base_subscription}"
            )
            ctx.set_base_subscription(self.data_config.default_base_subscription)

        # - set default on_event schedule if detected and strategy didn't set it's own schedule
        if not ctx.get_event_schedule("time") and self.data_config.default_trigger_schedule:
            logger.debug(f"[<y>simulator</y>] :: Setting default schedule: {self.data_config.default_trigger_schedule}")
            ctx.set_event_schedule(self.data_config.default_trigger_schedule)

        self.data_provider = data_provider
        self.logs_writer = logs_writer
        return ctx
