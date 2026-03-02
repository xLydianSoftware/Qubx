from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from qubx import QubxLogConfig, logger
from qubx.backtester.sentinels import NoDataContinue
from qubx.backtester.simulated_data import SimulatedDataIterator
from qubx.core.account import CompositeAccountProcessor
from qubx.core.basics import SW, DataType, Instrument, TransactionCostsCalculator
from qubx.core.context import StrategyContext
from qubx.core.exceptions import SimulationConfigError, SimulationError
from qubx.core.helpers import extract_parameters_from_object, full_qualified_class_name
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import (
    CtrlChannel,
    IDataProvider,
    IHealthMonitor,
    IMetricEmitter,
    IStrategy,
    IStrategyContext,
    IStrategyNotifier,
    ITimeProvider,
    StrategyState,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.lookups import lookup
from qubx.core.utils import time_delta_to_str
from qubx.data.cache import CachedStorage, MemoryCache
from qubx.data.guards import TimeGuardedStorage
from qubx.data.storage import IStorage
from qubx.health import DummyHealthMonitor
from qubx.loggers.inmemory import InMemoryLogsWriter
from qubx.pandaz.utils import _frame_to_str
from qubx.utils.runner.configs import PrefetchConfig
from qubx.utils.time import now_ns

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
    SimulationStatusWriter,
    _get_default_warmup_period,
    find_open_close_time_indent_secs_from_subscription,
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
    notifier: IStrategyNotifier | None

    account: CompositeAccountProcessor
    channel: CtrlChannel
    time_provider: SimulatedTimeProvider
    scheduler: SimulatedScheduler
    strategy_params: dict[str, Any]
    strategy_class: str

    # adjusted times
    _stop: pd.Timestamp | None = None

    _simulated_data_source: SimulatedDataIterator
    _data_providers: list[IDataProvider]
    _exchange_to_data_provider: dict[str, IDataProvider]
    _aux_storage: IStorage | None

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
        notifier: IStrategyNotifier | None = None,
        warmup_mode: bool = False,
        status_writer: SimulationStatusWriter | None = None,
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
        self.notifier = notifier
        self.strategy_state = strategy_state if strategy_state is not None else StrategyState()
        self.initializer = initializer
        self.warmup_mode = warmup_mode
        self.status_writer = status_writer
        self.strategy_params = {}
        self.strategy_class = ""
        self._pregenerated_signals = dict()
        self._to_process = {}
        self._aux_storage = None

        self._basic_initialization()
        self._create_backtest_context()
        self._handle_ctx_subscriptions()

    def run(self, silent: bool = False, catch_keyboard_interrupt: bool = True, close_data_readers: bool = False):
        """
        Run the backtest from start to stop.

        Args:
            silent (bool, optional): Whether to suppress progress output. Defaults to False.
            catch_keyboard_interrupt (bool, optional): Whether to catch KeyboardInterrupt. Defaults to True.
            close_data_readers (bool, optional): Whether to close IReader instances after run (releases DB connections etc). Defaults to False.
        """
        logger.debug(f"[<y>SimulationRunner</y>] :: Running simulation from {self.start} to {self.stop}")

        # - Start the context
        self.ctx.start()

        stop = self._stop or self.stop
        try:
            self._run(self.start, stop, silent=silent)
        except KeyboardInterrupt:
            logger.error("Simulated trading interrupted by user!")
            if not catch_keyboard_interrupt:
                raise
        except Exception as e:
            raise e
        finally:
            self.ctx.stop()
            if close_data_readers:
                self._simulated_data_source.close()

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
                    _start, _end = np.datetime64(start), np.datetime64(end)
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
        t = np.datetime64(int(data.time), "ns")
        _account = self.account.get_account_processor(instrument.exchange)
        _data_provider = self._get_data_provider(instrument.exchange)
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
        t = np.datetime64(int(data.time), "ns")
        _account = self.account.get_account_processor(instrument.exchange)
        _data_provider = self._get_data_provider(instrument.exchange)
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

    def _get_data_provider(self, exchange: str) -> IDataProvider:
        if exchange in self._exchange_to_data_provider:
            return self._exchange_to_data_provider[exchange]
        raise ValueError(f"Data provider for exchange {exchange} not found")

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
        prev_dt = np.datetime64(start)

        # - date iteration
        qiter = self._simulated_data_source.create_iterable(start, stop)

        if silent:
            for instrument, data_type, event, is_hist in qiter:
                # - handle NoDataContinue sentinel
                if isinstance(event, NoDataContinue):
                    if not self._handle_no_data_scenario(stop):
                        break
                    continue

                if not self._process_event(instrument, data_type, event, is_hist, _run, stop):
                    break
        else:
            _p = 0
            with tqdm(total=100, desc="Simulating", unit="%", leave=False) as pbar:
                for instrument, data_type, event, is_hist in qiter:
                    # - handle NoDataContinue sentinel
                    if isinstance(event, NoDataContinue):
                        if not self._handle_no_data_scenario(stop):
                            break
                        continue

                    if not self._process_event(instrument, data_type, event, is_hist, _run, stop):
                        break
                    dt = np.datetime64(int(event.time), "ns")
                    # - update progress bar and status writer every 1% of simulation time
                    if dt - prev_dt > update_delta:
                        _p += 1
                        pbar.n = _p
                        pbar.refresh()
                        prev_dt = dt
                        # - non-blocking: 2-tuple enqueued; record built in background thread
                        # - update every 10% to minimise queue.put() overhead in the hot loop
                        if self.status_writer is not None and _p % 10 == 0:
                            self.status_writer.update(float(_p), int(dt))
                pbar.n = 100
                pbar.refresh()

        logger.info(f"{self.__class__.__name__} ::: Simulation finished at {stop} :::")

    def _process_event(self, instrument, data_type, event, is_hist, _run, stop_time):
        """Process a single simulation event with proper time advancement and scheduler checks."""
        # During warmup, clamp future timestamps to current time
        if self.warmup_mode and hasattr(event, "time"):
            current_real_time = now_ns()
            if event.time > current_real_time:
                event.time = current_real_time

        if not _run(instrument, data_type, event, is_hist):
            return False

        return True

    def _handle_no_data_scenario(self, stop_time):
        """Handle scenario when no data is available but scheduler might have events."""
        # Check if we have pending scheduled events
        if hasattr(self.scheduler, "_next_nearest_time"):
            next_scheduled_time = self.scheduler._next_nearest_time
            current_time = self.time_provider.time()

            # Convert to int64 for numerical comparisons (avoid type issues)
            next_time_ns = next_scheduled_time.astype("int64")
            current_time_ns = current_time.astype("int64")
            stop_time_ns = stop_time.value  # Already int64

            # Check if we've reached the stop time
            if current_time_ns >= stop_time_ns:
                return False  # Stop simulation

            # If there's a scheduled event before stop time, advance to it
            if next_time_ns < np.iinfo(np.int64).max and next_time_ns < stop_time_ns:
                # Use the original datetime64 object for set_time (not the int64 conversion)
                self.time_provider.set_time(next_scheduled_time)
                self.scheduler.check_and_run_tasks()
                return True  # Continue simulation

        return False  # No scheduled events, stop simulation

    def print_latency_report(self) -> None:
        if (_l_r := SW.latency_report()) is not None:
            _llvl = QubxLogConfig.get_log_level()
            QubxLogConfig.set_log_level("INFO")
            logger.info(
                "<BLUE>   Time spent in simulation report   </BLUE>\n<r>"
                + _frame_to_str(
                    _l_r.sort_values("latency", ascending=False).reset_index(drop=True), "simulation", -1, -1, False
                )
                + "</r>"
            )
            QubxLogConfig.set_log_level(_llvl)

    def _basic_initialization(self):
        logger.debug(
            f"[<y>Simulator</y>] :: Preparing simulated trading on <g>{self.setup.exchanges}</g> "
            f"for {self.setup.capital} {self.setup.base_currency}"
        )

        # - get strategy parameters BEFORE simulation start
        #   potentially strategy may change it's parameters during simulation
        if self.setup.setup_type in [SetupTypes.STRATEGY, SetupTypes.STRATEGY_AND_TRACKER]:
            self.strategy_params = extract_parameters_from_object(self.setup.generator)
            self.strategy_class = full_qualified_class_name(self.setup.generator)

        # - main databus communication channel
        #   for simulation it just calls registered callback fn instead of processing it for speedup
        self.channel = SimulatedCtrlChannel("databus", sentinel=(None, None, None, None))
        health_monitor = DummyHealthMonitor()

        # - simulate exchange's time based on market data
        self.time_provider = SimulatedTimeProvider(np.datetime64(self.start, "ns"))

        # - simulated account
        self.account = self._construct_account_processor(
            self.setup.exchanges, self.setup.commissions, self.time_provider, self.channel, health_monitor
        )

        # - scheduler for simulated events
        self.scheduler = SimulatedScheduler(self.channel, lambda: self.time_provider.time().item())

        # - need to prefetch data
        _stor = self.data_config.data_storage
        _c_stor = self.data_config.customized_data_storages
        self._aux_storage = TimeGuardedStorage(self.data_config.aux_storage or _stor, self.time_provider)

        if self.data_config.prefetch_config is not None and self.data_config.prefetch_config.enabled:
            # - main simulation data: CachedStorage only (no TimeGuard).
            # - DataPump explicitly passes its own [start, end] window — it must NOT be
            #   clamped by TimeGuardedStorage (which would reduce stop to sim_start at t=0
            #   and produce an empty read, stalling the simulation entirely).
            _stor = self._wrap_storage(self.data_config.data_storage, self.data_config.prefetch_config)
            _c_stor = self._wrap_storage_dict(
                self.data_config.customized_data_storages, self.data_config.prefetch_config
            )
            # - aux / strategy-facing storage: TimeGuard ON TOP of cache so the strategy
            #   cannot see future data, while still benefiting from in-memory caching.
            if self.data_config.aux_storage:
                self._aux_storage = TimeGuardedStorage(
                    self._wrap_storage(self.data_config.aux_storage, self.data_config.prefetch_config),
                    self.time_provider,
                )
            else:
                self._aux_storage = TimeGuardedStorage(_stor, self.time_provider)

        # - main data iterator
        self._simulated_data_source = SimulatedDataIterator(
            _stor,
            _c_stor,
            trading_session=self.data_config.trading_sessions_time,
            default_trading_session=self.data_config.default_trading_sessions_time,
        )

        # - create time guarded aux data storage, optionally wrapped with in-memory cache.
        # - stack (outer → inner): TimeGuardedStorage → CachedStorage → inner storage
        # - TimeGuardedStorage clamps stop to current sim time (look-ahead guard).
        # - CachedStorage uses prefetch_period = full sim duration so that the FIRST read
        #   for any (dtype, symbols) fetches the entire backtest range in ONE DB query — same
        #   behaviour as the old upfront prefetch but lazy-triggered per dtype actually used.
        # - Subsequent reads (across all sim ticks) return from cache → zero DB queries.
        # - NOTE: PrefetchConfig.aux_data_names / args are no longer needed — caching is
        #   transparent and only warms dtypes the strategy actually accesses.
        # _inner_aux = self.data_config.aux_storage or self.data_config.data_storage
        # _pcfg = self.data_config.prefetch_config
        # if _pcfg is not None and _pcfg.enabled:
        #     # - use max(configured period, full sim duration) so any read grabs the full range
        #     _sim_duration = self.stop - self.start
        #     _effective_prefetch = max(_sim_duration, pd.Timedelta(_pcfg.prefetch_period))
        #     _cache_size_mb = _pcfg.cache_size_mb
        #     _inner_aux = CachedStorage(
        #         _inner_aux,
        #         prefetch_period=str(_effective_prefetch),
        #         cache_factory=lambda: MemoryCache(_cache_size_mb),
        #     )
        # self._aux_storage = TimeGuardedStorage(_inner_aux, self.time_provider)

    def _wrap_storage(self, storage: IStorage, prefetch_cfg: PrefetchConfig) -> IStorage:
        # - wrap with CachedStorage only — no TimeGuardedStorage here.
        # - TimeGuard is applied by callers that need it (aux/strategy-facing access).
        # - Main simulation data (DataPump) must NOT be time-guarded: DataPump provides
        #   its own explicit [start, end] window and needs to read the full range upfront.
        _prefetch_period = str(max(self.stop - self.start, pd.Timedelta(prefetch_cfg.prefetch_period)))
        return CachedStorage(
            storage, prefetch_period=_prefetch_period, cache_factory=lambda: MemoryCache(prefetch_cfg.cache_size_mb)
        )

    def _wrap_storage_dict(self, storages: dict[str, IStorage], prefetch_cfg: PrefetchConfig) -> dict[str, IStorage]:
        return {k: self._wrap_storage(s, prefetch_cfg) for k, s in storages.items()}

    def _create_backtest_context(self):
        # - create simulated brokers and data providers objects: exchange -> broker | provider
        self._data_providers = []
        _brokers = []
        for exchange in self.setup.exchanges:
            _exchange_account = self.account.get_account_processor(exchange)
            assert isinstance(_exchange_account, SimulatedAccountProcessor)

            _broker = SimulatedBroker(self.channel, _exchange_account, _exchange_account._exchange)
            _dprovider = SimulatedDataProvider(
                exchange_id=exchange,
                channel=self.channel,
                time_provider=self.time_provider,
                account=_exchange_account,
                data_source=self._simulated_data_source,
            )
            _brokers.append(_broker)
            self._data_providers.append(_dprovider)

        # - create mapping: exch -> IDataProvider
        self._exchange_to_data_provider = {dp.exchange(): dp for dp in self._data_providers}

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
                if len(self._data_providers) > 1:
                    raise SimulationConfigError("Signal setup is not supported for multiple exchanges !")

                self._set_generated_signals(self.setup.generator)  # type: ignore

                # - we don't need any unexpected triggerings
                self._stop = min(self.setup.generator.index[-1], self.stop)  # type: ignore

            case SetupTypes.SIGNAL_AND_TRACKER:
                strat = SignalsProxy(timeframe=self.setup.signal_timeframe)
                strat.tracker = lambda ctx: self.setup.tracker
                if len(self._data_providers) > 1:
                    raise SimulationConfigError("Signal setup is not supported for multiple exchanges !")

                self._set_generated_signals(self.setup.generator)  # type: ignore

                # - we don't need any unexpected triggerings
                self._stop = min(self.setup.generator.index[-1], self.stop)  # type: ignore

            case _:
                raise SimulationError(f"Unsupported setup type: {self.setup.setup_type} !")

        if not isinstance(strat, IStrategy):
            raise SimulationConfigError(f"Strategy should be an instance of IStrategy, but got {strat} !")

        # - it will store simulation results into memory
        self.logs_writer = InMemoryLogsWriter(self.account_id, self.setup.name, "0")

        # - _aux_storage is always set by _basic_initialization() which runs before this
        assert self._aux_storage is not None, "_basic_initialization() must run before _create_backtest_context()"

        # - create strategy context with setup
        self.ctx = StrategyContext(
            strategy=strat,
            brokers=_brokers,
            data_providers=self._data_providers,
            account=self.account,
            scheduler=self.scheduler,
            time_provider=self.time_provider,
            instruments=self.setup.instruments,
            logging=StrategyLogging(self.logs_writer, portfolio_log_freq=self.portfolio_log_freq),
            aux_data_storage=self._aux_storage,
            emitter=self.emitter,
            strategy_name=self.setup.name,
            strategy_state=self.strategy_state,
            notifier=self.notifier,
            initializer=self.initializer,
        )

        # - attach emmiter
        if self.emitter is not None:
            self.emitter.set_context(self.ctx)

    def _handle_ctx_subscriptions(self):
        # - check if strategy has base subscription
        if (_base_subscription := self.ctx.get_base_subscription()) == DataType.NONE:
            # - it's required to setup base subscription in strategy init
            raise SimulationConfigError(
                "No base subscription is set in initialization. Use initializer.set_base_subscription() in on_init(...)"
            )
        else:
            _indent, _base_tf = find_open_close_time_indent_secs_from_subscription(_base_subscription, 0)

            # - deduct correct emulation time indent from base subscription
            self._simulated_data_source.update_emulation_time_indent_seconds(_indent)

            # - if base subscription is OHLC related it should have timeframe
            if _base_subscription in [DataType.OHLC, DataType.OHLC_QUOTES, DataType.OHLC_TRADES] and _base_tf is None:
                raise SimulationConfigError(
                    "Timeframe is not provided for OHLC based subscription - unable to detect time indent for simulated data !"
                )

            # - Check warmup period for base subscription (strategy warmups take precedence)
            _merged_warmups = {
                **{
                    str(_base_subscription): time_delta_to_str(
                        _get_default_warmup_period(str(_base_subscription), _base_tf).asm8.item()
                    )
                },
                **self.ctx.initializer.get_subscription_warmup(),
            }
            if _merged_warmups:
                logger.debug(f"[<y>SimulationRunner</y>] :: Setting warmups: {_merged_warmups}")
                self.ctx.set_warmup(_merged_warmups)

        # - check default on_event schedule if detected and strategy didn't set it's own schedule
        if not self.ctx.get_event_schedule("time"):
            logger.warning(
                "[<y>simulator</y>] :: Event schedule is not specified.\n - Only on_market_data() will be triggered !\n - To enable on_event() call initializer.set_event_schedule(...) in on_init()"
            )

        if self.setup.enable_funding:
            logger.debug("[<y>simulator</y>] :: Enabling funding rate simulation")
            self.ctx.subscribe(DataType.FUNDING_PAYMENT)

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
        health_monitor: IHealthMonitor,
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
                health_monitor=health_monitor,
                base_currency=self.setup.base_currency,
                exchange_name=exchange,
                initial_capital=_initial_capital,
            )

        return CompositeAccountProcessor(
            time_provider=time_provider,
            account_processors=_account_processors,
        )
