from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from qubx import logger
from qubx.backtester.simulated_data import IterableSimulationData
from qubx.core.account import CompositeAccountProcessor
from qubx.core.basics import SW, DataType, Instrument, TransactionCostsCalculator
from qubx.core.context import StrategyContext
from qubx.core.exceptions import SimulationConfigError, SimulationError
from qubx.core.helpers import extract_parameters_from_object, full_qualified_class_name
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import (
    CtrlChannel,
    IMetricEmitter,
    IStrategy,
    IStrategyContext,
    ITimeProvider,
    StrategyState,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.lookups import lookup
from qubx.loggers.inmemory import InMemoryLogsWriter
from qubx.pandaz.utils import _frame_to_str

from .account import SimulatedAccountProcessor
from .broker import SimulatedBroker
from .data import SimulatedDataProvider
from .simulated_exchange import get_simulated_exchange
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
    logs_writer: InMemoryLogsWriter

    account: CompositeAccountProcessor
    channel: CtrlChannel
    time_provider: SimulatedTimeProvider
    scheduler: SimulatedScheduler
    strategy_params: dict[str, Any]
    strategy_class: str

    # adjusted times
    _stop: pd.Timestamp | None = None

    _data_source: IterableSimulationData
    _data_providers: list[SimulatedDataProvider]
    _exchange_to_data_provider: dict[str, SimulatedDataProvider]

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
        self._pregenerated_signals = dict()
        self._to_process = {}

        # - get strategy parameters BEFORE simulation start
        #   potentially strategy may change it's parameters during simulation
        self.strategy_params = {}
        self.strategy_class = ""
        if self.setup.setup_type in [SetupTypes.STRATEGY, SetupTypes.STRATEGY_AND_TRACKER]:
            self.strategy_params = extract_parameters_from_object(self.setup.generator)
            self.strategy_class = full_qualified_class_name(self.setup.generator)

        self.ctx = self._create_backtest_context()

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
            self._run(self.start, stop, silent=silent)
        except KeyboardInterrupt:
            logger.error("Simulated trading interrupted by user!")
            if not catch_keyboard_interrupt:
                raise
        finally:
            # Stop the context
            self.ctx.stop()
            if close_data_readers:
                for dp in self._data_providers:
                    for reader in dp._readers.values():
                        if hasattr(reader, "close"):
                            reader.close()  # type: ignore

    def _set_generated_signals(self, signals: pd.Series | pd.DataFrame):
        logger.debug(
            f"[<y>{self.__class__.__name__}</y>] :: Using pre-generated signals:\n {str(signals.count()).strip('ndtype: int64')}"
        )
        # - sanity check
        signals.index = pd.DatetimeIndex(signals.index)

        if isinstance(signals, pd.Series):
            self._pregenerated_signals[str(signals.name)] = signals  # type: ignore

        elif isinstance(signals, pd.DataFrame):
            for col in signals.columns:
                self._pregenerated_signals[col] = signals[col]  # type: ignore
        else:
            raise ValueError("Invalid signals or strategy configuration")

    def _prepare_generated_signals(self, start: str | pd.Timestamp, end: str | pd.Timestamp):
        for s, v in self._pregenerated_signals.items():
            _s_inst = None

            for i in self._data_providers[0].get_subscribed_instruments():
                # - we can process series with variable id's if we can find some similar instrument
                if s == i.symbol or s == str(i) or s == f"{i.exchange}:{i.symbol}" or str(s) == str(i):
                    _start, _end = pd.Timestamp(start), pd.Timestamp(end)
                    _start_idx, _end_idx = v.index.get_indexer([_start, _end], method="ffill")
                    sel = v.iloc[max(_start_idx, 0) : _end_idx + 1]

                    # TODO: check if data has exec_price - it means we have deals
                    self._to_process[i] = list(zip(sel.index, sel.values))
                    _s_inst = i
                    break

            if _s_inst is None:
                logger.error(f"Can't find instrument for pregenerated signals with id '{s}'")
                raise SimulationError(f"Can't find instrument for pregenerated signals with id '{s}'")

    def _process_generated_signals(self, instrument: Instrument, data_type: str, data: Any, is_hist: bool) -> bool:
        cc = self.channel
        t = np.datetime64(data.time, "ns")
        _account = self.account.get_account_processor(instrument.exchange)
        _data_provider = self._exchange_to_data_provider[instrument.exchange]
        assert isinstance(_account, SimulatedAccountProcessor)
        assert isinstance(_data_provider, SimulatedDataProvider)

        if not is_hist:
            # - signals for this instrument
            sigs = self._to_process[instrument]

            while sigs and t >= (_signal_time := sigs[0][0].as_unit("ns").asm8):
                self.time_provider.set_time(_signal_time)
                cc.send((instrument, "event", {"order": sigs[0][1]}, False))
                sigs.pop(0)

            if q := _account._exchange.emulate_quote_from_data(instrument, t, data):
                _data_provider._last_quotes[instrument] = q

        self.time_provider.set_time(t)
        cc.send((instrument, data_type, data, is_hist))

        return cc.control.is_set()

    def _process_strategy(self, instrument: Instrument, data_type: str, data: Any, is_hist: bool) -> bool:
        cc = self.channel
        t = np.datetime64(data.time, "ns")
        _account = self.account.get_account_processor(instrument.exchange)
        _data_provider = self._exchange_to_data_provider[instrument.exchange]
        assert isinstance(_account, SimulatedAccountProcessor)
        assert isinstance(_data_provider, SimulatedDataProvider)

        if not is_hist:
            if t >= (_next_exp_time := self.scheduler.next_expected_event_time()):
                # - we use exact event's time
                self.time_provider.set_time(_next_exp_time)
                self.scheduler.check_and_run_tasks()

            if q := _account._exchange.emulate_quote_from_data(instrument, t, data):
                _data_provider._last_quotes[instrument] = q

        self.time_provider.set_time(t)
        cc.send((instrument, data_type, data, is_hist))

        return cc.control.is_set()

    def _run(self, start: pd.Timestamp, stop: pd.Timestamp, silent: bool = False) -> None:
        logger.info(f"{self.__class__.__name__} ::: Simulation started at {start} :::")

        if self._pregenerated_signals:
            self._prepare_generated_signals(start, stop)
            _run = self._process_generated_signals
        else:
            _run = self._process_strategy

        start, stop = pd.Timestamp(start), pd.Timestamp(stop)
        total_duration = stop - start
        update_delta = total_duration / 100
        prev_dt = pd.Timestamp(start)

        # - date iteration
        qiter = self._data_source.create_iterable(start, stop)
        if silent:
            for instrument, data_type, event, is_hist in qiter:
                if not _run(instrument, data_type, event, is_hist):
                    break
        else:
            _p = 0
            with tqdm(total=100, desc="Simulating", unit="%", leave=False) as pbar:
                for instrument, data_type, event, is_hist in qiter:
                    if not _run(instrument, data_type, event, is_hist):
                        break
                    dt = pd.Timestamp(event.time)
                    # update only if date has changed
                    if dt - prev_dt > update_delta:
                        _p += 1
                        pbar.n = _p
                        pbar.refresh()
                        prev_dt = dt
                pbar.n = 100
                pbar.refresh()

        logger.info(f"{self.__class__.__name__} ::: Simulation finished at {stop} :::")

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
        logger.debug(
            f"[<y>Simulator</y>] :: Preparing simulated trading on <g>{self.setup.exchanges}</g> "
            f"for {self.setup.capital} {self.setup.base_currency}..."
        )

        data_source = IterableSimulationData(
            self.data_config.data_providers,
            open_close_time_indent_secs=self.data_config.adjusted_open_close_time_indent_secs,
        )

        channel = SimulatedCtrlChannel("databus", sentinel=(None, None, None, None))
        simulated_clock = SimulatedTimeProvider(np.datetime64(self.start, "ns"))

        account = self._construct_account_processor(
            self.setup.exchanges, self.setup.commissions, simulated_clock, channel
        )

        scheduler = SimulatedScheduler(channel, lambda: simulated_clock.time().item())

        brokers = []
        for exchange in self.setup.exchanges:
            _exchange_account = account.get_account_processor(exchange)
            assert isinstance(_exchange_account, SimulatedAccountProcessor)
            brokers.append(SimulatedBroker(channel, _exchange_account, _exchange_account._exchange))

        data_providers = []
        for exchange in self.setup.exchanges:
            _exchange_account = account.get_account_processor(exchange)
            assert isinstance(_exchange_account, SimulatedAccountProcessor)
            data_providers.append(
                SimulatedDataProvider(
                    exchange_id=exchange,
                    channel=channel,
                    scheduler=scheduler,
                    time_provider=simulated_clock,
                    account=_exchange_account,
                    readers=self.data_config.data_providers,
                    data_source=data_source,
                    open_close_time_indent_secs=self.data_config.adjusted_open_close_time_indent_secs,
                )
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
                if len(data_providers) > 1:
                    raise SimulationConfigError("Signal setup is not supported for multiple exchanges !")

                self._set_generated_signals(self.setup.generator)  # type: ignore

                # - we don't need any unexpected triggerings
                self._stop = min(self.setup.generator.index[-1], self.stop)  # type: ignore

            case SetupTypes.SIGNAL_AND_TRACKER:
                strat = SignalsProxy(timeframe=self.setup.signal_timeframe)
                strat.tracker = lambda ctx: self.setup.tracker
                if len(data_providers) > 1:
                    raise SimulationConfigError("Signal setup is not supported for multiple exchanges !")

                self._set_generated_signals(self.setup.generator)  # type: ignore

                # - we don't need any unexpected triggerings
                self._stop = min(self.setup.generator.index[-1], self.stop)  # type: ignore

            case _:
                raise SimulationError(f"Unsupported setup type: {self.setup.setup_type} !")

        if not isinstance(strat, IStrategy):
            raise SimulationConfigError(f"Strategy should be an instance of IStrategy, but got {strat} !")

        ctx = StrategyContext(
            strategy=strat,
            brokers=brokers,
            data_providers=data_providers,
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

        self.logs_writer = logs_writer
        self.channel = channel
        self.time_provider = simulated_clock
        self.account = account
        self.scheduler = scheduler
        self._data_source = data_source
        self._data_providers = data_providers
        self._exchange_to_data_provider = {dp.exchange(): dp for dp in data_providers}
        return ctx

    def _construct_tcc(
        self, exchanges: list[str], commissions: str | dict[str, str | None] | None
    ) -> dict[str, TransactionCostsCalculator]:
        _exchange_to_tcc = {}
        if isinstance(commissions, (str, type(None))):
            commissions = {e: commissions for e in exchanges}
        for exchange in exchanges:
            _exchange_to_tcc[exchange] = lookup.find_fees(exchange.lower(), commissions.get(exchange))
        return _exchange_to_tcc

    def _construct_account_processor(
        self,
        exchanges: list[str],
        commissions: str | dict[str, str | None] | None,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
    ) -> CompositeAccountProcessor:
        _exchange_to_tcc = self._construct_tcc(exchanges, commissions)
        for tcc in _exchange_to_tcc.values():
            if tcc is None:
                raise SimulationConfigError(
                    f"Can't find transaction costs calculator for '{self.setup.exchanges}' for specification '{self.setup.commissions}' !"
                )

        _exchange_to_simulated_exchange = {}
        for exchange in self.setup.exchanges:
            # - create simulated exchange:
            #   - we can use different emulations of real exchanges features in future here: for Binance, Bybit, InteractiveBrokers, etc.
            #   - for now we use simple basic simulated exchange implementation
            _exchange_to_simulated_exchange[exchange] = get_simulated_exchange(
                exchange, time_provider, _exchange_to_tcc[exchange], self.setup.accurate_stop_orders_execution
            )

        _account_processors = {}
        for exchange in self.setup.exchanges:
            _initial_capital = self.setup.capital
            if isinstance(_initial_capital, dict):
                _initial_capital = _initial_capital[exchange]
            assert isinstance(_initial_capital, (float, int))
            _account_processors[exchange] = SimulatedAccountProcessor(
                account_id=self.account_id,
                exchange=_exchange_to_simulated_exchange[exchange],
                channel=channel,
                base_currency=self.setup.base_currency,
                initial_capital=_initial_capital,
            )

        return CompositeAccountProcessor(
            time_provider=time_provider,
            account_processors=_account_processors,
        )
