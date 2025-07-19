import traceback
import uuid
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from types import FunctionType
from typing import Any, Callable

import pandas as pd

from qubx import logger
from qubx.core.basics import (
    DataType,
    Deal,
    FundingPayment,
    InitializingSignal,
    Instrument,
    MarketEvent,
    Order,
    Signal,
    TargetPosition,
    Timestamped,
    TriggerEvent,
    dt_64,
    td_64,
)
from qubx.core.errors import BaseErrorEvent
from qubx.core.exceptions import StrategyExceededMaxNumberOfRuntimeFailuresError
from qubx.core.helpers import BasicScheduler, CachedMarketDataHolder, process_schedule_spec
from qubx.core.interfaces import (
    IAccountProcessor,
    IHealthMonitor,
    IMarketManager,
    IPositionGathering,
    IProcessingManager,
    IStrategy,
    IStrategyContext,
    ISubscriptionManager,
    ITimeProvider,
    ITradeDataExport,
    IUniverseManager,
    PositionsTracker,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.series import Bar, OrderBook, Quote, Trade
from qubx.core.stale_data_detector import StaleDataDetector
from qubx.trackers.riskctrl import _InitializationStageTracker


class ProcessingManager(IProcessingManager):
    MAX_NUMBER_OF_STRATEGY_FAILURES: int = 10
    DATA_READY_TIMEOUT: td_64 = td_64(60, "s")

    _context: IStrategyContext
    _strategy: IStrategy
    _logging: StrategyLogging
    _market_data: IMarketManager
    _subscription_manager: ISubscriptionManager
    _time_provider: ITimeProvider
    _account: IAccountProcessor
    _position_tracker: PositionsTracker
    _position_gathering: IPositionGathering
    _cache: CachedMarketDataHolder
    _scheduler: BasicScheduler
    _universe_manager: IUniverseManager
    _exporter: ITradeDataExport | None = None
    _health_monitor: IHealthMonitor
    _stale_data_detector: StaleDataDetector

    _handlers: dict[str, Callable[["ProcessingManager", Instrument, str, Any], TriggerEvent | None]]
    _strategy_name: str

    _trigger_on_time_event: bool = False
    _fit_is_running: bool = False
    _fails_counter: int = 0
    _is_simulation: bool
    _pool: ThreadPool | None
    _trig_bar_freq_nsec: int | None = None
    _cur_sim_step: int | None = None
    _updated_instruments: set[Instrument] = set()
    _data_ready_start_time: dt_64 | None = None
    _last_data_ready_log_time: dt_64 | None = None
    _all_instruments_ready_logged: bool = False
    _emitted_signals: list[Signal] = []  # signals that were emitted
    _active_targets: dict[Instrument, TargetPosition] = {}

    # - post-warmup initialization
    _init_stage_position_tracker: PositionsTracker
    _instruments_in_init_stage: set[Instrument] = set()

    # - custom scheduled methods
    _custom_scheduled_methods: dict[str, Callable] = {}

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        logging: StrategyLogging,
        market_data: IMarketManager,
        subscription_manager: ISubscriptionManager,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        position_tracker: PositionsTracker,
        position_gathering: IPositionGathering,
        universe_manager: IUniverseManager,
        cache: CachedMarketDataHolder,
        scheduler: BasicScheduler,
        is_simulation: bool,
        health_monitor: IHealthMonitor,
        exporter: ITradeDataExport | None = None,
    ):
        self._context = context
        self._strategy = strategy
        self._logging = logging
        self._market_data = market_data
        self._subscription_manager = subscription_manager
        self._time_provider = time_provider
        self._account = account
        self._is_simulation = is_simulation
        self._position_gathering = position_gathering
        self._position_tracker = position_tracker
        self._universe_manager = universe_manager
        self._cache = cache
        self._scheduler = scheduler
        self._exporter = exporter
        self._health_monitor = health_monitor

        # Initialize stale data detector with default disabled state
        # Will be configured later based on strategy settings
        self._stale_data_detector = StaleDataDetector(
            cache=cache,
            time_provider=time_provider,
        )
        self._stale_data_detection_enabled = False

        self._pool = ThreadPool(2) if not self._is_simulation else None
        self._handlers = {
            n.split("_handle_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) is FunctionType and n.startswith("_handle_")
        }
        self._strategy_name = strategy.__class__.__name__
        self._trig_bar_freq_nsec = None
        self._updated_instruments = set()
        self._data_ready_start_time = None
        self._last_data_ready_log_time = None
        self._all_instruments_ready_logged = False
        self._emitted_signals = []

        # - special tracker for post-warmup initialization signals
        self._init_stage_position_tracker = _InitializationStageTracker()
        self._instruments_in_init_stage = set()
        self._active_targets = {}
        self._custom_scheduled_methods = {}

        # - schedule daily delisting check at 23:30 (end of day)
        self._scheduler.schedule_event("30 23 * * *", "delisting_check")

    def set_fit_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if rule.get("type") != "cron":
            raise ValueError("Only cron type is supported for fit schedule")
        self._scheduler.schedule_event(rule["schedule"], "fit")

    def set_event_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if not rule or "type" not in rule:
            raise ValueError(f"Can't recognoize schedule format: '{schedule}'")

        if rule["type"] != "cron":
            raise ValueError("Only cron type is supported for event schedule")

        self._scheduler.schedule_event(rule["schedule"], "time")
        self._trigger_on_time_event = True

    def get_event_schedule(self, event_id: str) -> str | None:
        return self._scheduler.get_schedule_for_event(event_id)

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> None:
        """
        Register a custom method to be called at specified times.

        Args:
            cron_schedule: Cron-like schedule string (e.g., "0 0 * * *" for daily at midnight)
            method: Method to call when schedule triggers
        """
        rule = process_schedule_spec(cron_schedule)
        if not rule or rule.get("type") != "cron":
            raise ValueError("Only cron type is supported for custom schedules")

        # Generate unique event ID for this custom schedule
        event_id = f"custom_schedule_{str(uuid.uuid4()).replace('-', '_')}"

        # Store the method reference
        self._custom_scheduled_methods[event_id] = method

        # Schedule the event
        self._scheduler.schedule_event(rule["schedule"], event_id)

    def configure_stale_data_detection(
        self, enabled: bool, detection_period: str | None = None, check_interval: str | None = None
    ) -> None:
        """
        Configure stale data detection settings.

        Args:
            enabled: Whether to enable stale data detection
            detection_period: Period to consider data as stale (e.g., "5Min", "1h"). If None, uses detector default.
            check_interval: Interval between stale data checks (e.g., "30s", "1Min"). If None, uses detector default.
        """
        self._stale_data_detection_enabled = enabled

        if enabled and (detection_period is not None or check_interval is not None):
            # Recreate the detector with new parameters
            kwargs = {}
            if detection_period is not None:
                kwargs["detection_period"] = detection_period
            if check_interval is not None:
                kwargs["check_interval"] = check_interval

            self._stale_data_detector = StaleDataDetector(
                cache=self._cache, time_provider=self._time_provider, **kwargs
            )

    def process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        should_stop = self.__process_data(instrument, d_type, data, is_historical)
        if not is_historical:
            self._logging.notify(self._time_provider.time())
            if self._context.emitter is not None:
                self._context.emitter.notify(self._context)
        return should_stop

    def is_fitted(self) -> bool:
        return self._context._strategy_state.is_on_fit_called

    def __process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        handler = self._handlers.get(d_type)
        with self._health_monitor("process_data"):
            if not d_type:
                event = None
            elif is_historical:
                event = self._process_hist_event(instrument, d_type, data)
            elif handler:
                event = handler(self, instrument, d_type, data)
            else:
                event = self._process_custom_event(instrument, d_type, data)

        if not self._context._strategy_state.is_on_start_called and not self._is_order_update(d_type):
            self._handle_start()

        if (
            not self._context._strategy_state.is_on_warmup_finished_called
            and not self._context._strategy_state.is_warmup_in_progress
            and not self._is_order_update(d_type)
        ):
            if self._context.get_warmup_positions() or self._context.get_warmup_orders():
                self._handle_state_resolution()
            self._handle_warmup_finished()

        # - check if it still didn't call on_fit() for first time
        if not self._context._strategy_state.is_on_fit_called and not self._fit_is_running:
            self._handle_fit(None, "fit", (None, self._time_provider.time()))
            return False

        if not event and not self._emitted_signals:
            return False

        # - if fit was not called - skip on_event call
        if not self._context._strategy_state.is_on_fit_called or (
            not self._is_simulation and not self._context._strategy_state.is_on_warmup_finished_called
        ):
            return False

        # - if strategy still fitting - skip on_event call
        if self._fit_is_running:
            # logger.debug(
            #     f"Skipping {self._strategy_name}::on_event({instrument}, {d_type}, [...], {is_historical}) fitting in progress (orders and deals processed)!"
            # )
            return False

        signals: list[Signal] = []

        # - if there are signals that were emitted before, we need to process them first
        if self._emitted_signals:
            signals.extend(self._emitted_signals)
            self._emitted_signals.clear()

        try:
            # - some small optimization
            _is_market_ev = isinstance(event, MarketEvent)
            _is_trigger_ev = isinstance(event, TriggerEvent)

            if _is_market_ev:
                with self._health_monitor("stg.market_event"):
                    signals.extend(self._as_list(self._strategy.on_market_data(self._context, event)))

            if _is_trigger_ev or (_is_market_ev and event.is_trigger):
                _trigger_event = event.to_trigger() if _is_market_ev else event

                # FIXME: (2025-06-17) we need to refactor this to avoid doing it here !!!
                # - on trigger event we need to be sure that all instruments have finalized OHLC data
                # - - - - - - IMPORTANT NOTES - - - - - -
                # This is a temporary fix to ensure that all instruments have finalized OHLC data.
                # In live mode with multiple instruments, we can have a situation where one instrument
                # is updated with new bar data while other instruments are not updated yet.
                # This leads to a situation where indicators are not calculated correctly for all instruments in the universe.
                # A simple dirty solution is to update OHLC data for all instruments in the universe with the last update value but with the actual time.
                # This leads to finalization of OHLC data, but the open price may differ slightly from the real one.
                # - - - - - - - - - - - - - - - - - - - -

                # - finalize OHLC data for all instruments
                self._cache.finalize_ohlc_for_instruments(event.time, self._context.instruments)

                with self._health_monitor("stg.trigger_event"):
                    signals.extend(self._as_list(self._strategy.on_event(self._context, _trigger_event)))

                # - we reset failures counter when we successfully process on_event
                self._fails_counter = 0

            if isinstance(event, Order):
                with self._health_monitor("stg.order_update"):
                    signals.extend(self._as_list(self._strategy.on_order_update(self._context, event)))

            self._subscription_manager.commit()  # apply pending operations

        except Exception as strat_error:
            # Record event dropped due to exception
            self._health_monitor.record_event_dropped(d_type)

            # - probably we need some cooldown interval after exception to prevent flooding
            logger.error(f"Strategy {self._strategy_name} raised an exception: {strat_error}")
            logger.opt(colors=False).error(traceback.format_exc())

            #  - we stop execution after maximal number of errors in a row
            self._fails_counter += 1
            if self._fails_counter >= self.MAX_NUMBER_OF_STRATEGY_FAILURES:
                logger.error(f"STRATEGY FAILED {self.MAX_NUMBER_OF_STRATEGY_FAILURES} TIMES IN THE ROW - STOPPING ...")
                raise StrategyExceededMaxNumberOfRuntimeFailuresError()

        # - process and execute signals if they are provided
        if signals:
            with self._health_monitor("position_processing"):
                self.__process_signals(signals)

        return False

    def _get_tracker_for(self, instrument: Instrument) -> PositionsTracker:
        return (
            self._init_stage_position_tracker
            if instrument in self._instruments_in_init_stage
            else self._position_tracker
        )

    def __preprocess_signals_and_split_by_stage(
        self, signals: list[Signal]
    ) -> tuple[list[Signal], list[Signal], set[Instrument]]:
        """
        Preprocess signals:
            - split into standard and initializing signals
            - prevent service signals to be processed
            - set strategy group name if not set
            - update reference prices for signals
            - switch tracker for initializing signals to the post-warmup one
            - return list of standard signals, list of initializing signals, and set of instruments to cancel post-warmup trackers for
        """
        _init_signals: list[Signal] = []
        _std_signals: list[Signal] = []
        _cancel_init_stage_instruments_tracker = set()

        for signal in signals:
            instr = signal.instrument

            # - set strategy group name if not set
            if not signal.group:
                signal.group = self._strategy_name

            # - update reference prices for signals
            if signal.reference_price is None:
                if q := self._market_data.quote(instr):
                    signal.reference_price = q.mid_price()

            # - prevent service signals to be processed
            if signal.is_service:
                continue

            # - if there is initializing signal, we need to switch tracker to the post-warmup one for this instrument
            if isinstance(signal, InitializingSignal):
                # - if target is already active and it receives initializing signal
                # - it means that something was sent wrong and we need to skip this signal
                if instr in self.get_active_targets():
                    logger.warning(
                        f"Skip initializing signal <y>{signal}</y> for <g>{instr}</g> because strategy has active target {self.get_active_targets()[instr]} !"
                    )
                else:
                    _init_signals.append(signal)
                    self._instruments_in_init_stage.add(instr)
                    logger.info(f"Switching tracker for <g>{instr}</g> to post-warmup initialization")
            else:
                _std_signals.append(signal)
                if instr in self._instruments_in_init_stage:
                    _cancel_init_stage_instruments_tracker.add(instr)
                    self._instruments_in_init_stage.remove(instr)
                    logger.info(f"Switching tracker for <g>{instr}</g> back to defined position tracker")

        return _std_signals, _init_signals, _cancel_init_stage_instruments_tracker

    def __process_signals(self, signals: list[Signal]):
        _targets_from_trackers: list[TargetPosition] = []

        # - preprocess signals: split into usual and initializing signals
        _std_signals, _init_signals, _cancel_init_trackers_for = self.__preprocess_signals_and_split_by_stage(signals)

        # - cancel post-warmup trackers
        if _cancel_init_trackers_for:
            for instr in _cancel_init_trackers_for:
                self._init_stage_position_tracker.cancel_tracking(self._context, instr)

        # - notify post-warmup position tracker for the new initializing signals
        if _init_signals:
            _targets_from_trackers.extend(
                self._as_list(self._init_stage_position_tracker.process_signals(self._context, _init_signals))
            )

        # - notify position tracker for the new signals
        if _std_signals:
            _targets_from_trackers.extend(
                self._as_list(self._position_tracker.process_signals(self._context, _std_signals))
            )

        # - notify position gatherer for the new target positions
        if _targets_from_trackers:
            self._position_gathering.alter_positions(
                self._context, self.__preprocess_and_log_target_positions(_targets_from_trackers)
            )

        # - log all signals and export signals if exporter is specified after processing because trackers can modify the signals
        self._logging.save_signals(signals)

        # - export signals if exporter is specified
        if self._exporter is not None and signals:
            self._exporter.export_signals(self._time_provider.time(), signals, self._account)

        # - emit signals to metric emitters if available
        if self._context.emitter is not None and signals:
            self._context.emitter.emit_signals(
                self._time_provider.time(), signals, self._account, _targets_from_trackers
            )

    def __invoke_on_fit(self) -> None:
        with self._health_monitor("ctx.on_fit"):
            try:
                logger.debug(f"[<y>{self.__class__.__name__}</y>] :: Invoking <g>{self._strategy_name}</g> on_fit")
                self._strategy.on_fit(self._context)
                self._subscription_manager.commit()  # apply pending operations
                logger.debug(f"[<y>{self.__class__.__name__}</y>] :: <g>{self._strategy_name}</g> is fitted")
            except Exception as strat_error:
                logger.error(
                    f"[{self.__class__.__name__}] :: Strategy {self._strategy_name} on_fit raised an exception: {strat_error}"
                )
                logger.opt(colors=False).error(traceback.format_exc())
            finally:
                self._fit_is_running = False
                self._context._strategy_state.is_on_fit_called = True

    def __preprocess_and_log_target_positions(self, target_positions: list[TargetPosition]) -> list[TargetPosition]:
        filtered_target_positions = []

        if target_positions:
            # - check if trading is allowed for each target position
            for t in target_positions:
                _instr = t.instrument
                if self._universe_manager.is_trading_allowed(_instr):
                    filtered_target_positions.append(t)

                    # - new position will be non-zero
                    if abs(t.target_position_size) > _instr.min_size:
                        self._active_targets[_instr] = t
                    else:
                        self._active_targets.pop(_instr, None)

            # - log target positions
            self._logging.save_targets(filtered_target_positions)

            # - export target positions if exporter is available
            if self._exporter is not None:
                self._exporter.export_target_positions(self._time_provider.time(), target_positions, self._account)

        return filtered_target_positions

    def _run_in_thread_pool(self, func: Callable, args=()):
        # For simulation we don't need to call function in thread
        if self._is_simulation:
            func(*args)
        else:
            assert self._pool
            self._pool.apply_async(func, args)

    @staticmethod
    def _as_list(xs: list[Any] | Any | None) -> list[Any]:
        if xs is None:
            return []

        if isinstance(xs, list):
            return xs

        return [xs]

    __SUBSCR_TO_DATA_MATCH_TABLE = {
        DataType.OHLC: [Bar],
        DataType.OHLC_QUOTES: [Quote, OrderBook],
        DataType.OHLC_TRADES: [Trade],
        DataType.QUOTE: [Quote],
        DataType.TRADE: [Trade],
        DataType.ORDERBOOK: [OrderBook],
    }

    def _is_base_data(self, data: Timestamped) -> tuple[bool, Timestamped]:
        _base_ss = DataType.from_str(self._subscription_manager.get_base_subscription())[0]
        _d_probe = data
        return (
            type(_d_probe) in _rule if (_rule := self.__SUBSCR_TO_DATA_MATCH_TABLE.get(_base_ss)) else False,
            _d_probe,
        )

    def _is_data_ready(self) -> bool:
        """
        Check if strategy can start based on data availability with timeout logic.

        Two-phase approach:
        - Phase 1 (0-DATA_READY_TIMEOUT): Wait for ALL instruments to have data
        - Phase 2 (after timeout): Wait for at least 1 instrument to have data

        Returns:
            bool: True if strategy can start, False if still waiting
        """
        total_instruments = len(self._context.instruments)

        # Handle edge case: no instruments
        if total_instruments == 0:
            return True

        ready_instruments = len(self._updated_instruments)
        current_time = self._time_provider.time()

        # Record start time on first call
        if self._data_ready_start_time is None:
            self._data_ready_start_time = current_time

        # Phase 1: Try to get all instruments ready within timeout
        elapsed_td = current_time - self._data_ready_start_time

        if elapsed_td <= self.DATA_READY_TIMEOUT:
            # Within timeout period - wait for ALL instruments
            if ready_instruments == total_instruments:
                if not self._all_instruments_ready_logged:
                    logger.info(f"All {total_instruments} instruments have data - strategy ready to start")
                    self._all_instruments_ready_logged = True
                return True
            else:
                # Log periodic status during Phase 1 - throttled to once per 10 seconds
                elapsed_seconds = elapsed_td / td_64(1, "s")
                should_log = self._last_data_ready_log_time is None or (
                    current_time - self._last_data_ready_log_time
                ) >= td_64(10, "s")

                if should_log and elapsed_seconds > 0:
                    missing_instruments = set(self._context.instruments) - self._updated_instruments
                    missing_symbols = [inst.symbol for inst in missing_instruments]
                    remaining_timeout = (self.DATA_READY_TIMEOUT - elapsed_td) / td_64(1, "s")
                    logger.info(
                        f"Waiting for all instruments ({ready_instruments}/{total_instruments} ready). "
                        f"Missing: {missing_symbols}. Will start with partial data in {remaining_timeout:.0f}s"
                    )
                    self._last_data_ready_log_time = current_time
                return False
        else:
            # Phase 2: After timeout - need at least 1 instrument
            if ready_instruments >= total_instruments:
                if not self._all_instruments_ready_logged:
                    logger.info(f"All {total_instruments} instruments have data - strategy ready to start")
                    self._all_instruments_ready_logged = True
                return True

            elif ready_instruments >= 1:
                missing_instruments = set(self._context.instruments) - self._updated_instruments
                missing_symbols = [inst.symbol for inst in missing_instruments]

                # Log once when entering Phase 2
                should_log = self._last_data_ready_log_time is None or (
                    current_time - self._last_data_ready_log_time
                ) >= td_64(10, "s")
                if should_log:
                    logger.info(
                        f"Timeout reached - starting with {ready_instruments}/{total_instruments} instruments ready. "
                        f"Missing: {missing_symbols}"
                    )
                    self._last_data_ready_log_time = current_time
                return True
            else:
                # Still no instruments ready - keep waiting and log periodically
                should_log = self._last_data_ready_log_time is None or (
                    current_time - self._last_data_ready_log_time
                ) >= td_64(10, "s")
                if should_log:
                    # logger.warning(
                    #     f"No instruments ready after timeout - still waiting "
                    #     f"({ready_instruments}/{total_instruments} ready)"
                    # )
                    self._last_data_ready_log_time = current_time
                return False

    def __update_base_data(
        self, instrument: Instrument, event_type: str, data: Timestamped, is_historical: bool = False
    ) -> bool:
        """
        Updates the base data cache with the provided data.

        Returns:
            bool: True if the data is base data and the strategy should be triggered, False otherwise.
        """
        is_base_data, _update = self._is_base_data(data)
        # logger.info(f"{_update} {is_base_data and not self._trigger_on_time_event}")

        # update cached ohlc is this is base subscription
        _update_ohlc = is_base_data
        self._cache.update(
            instrument,
            event_type,
            _update,
            update_ohlc=_update_ohlc,
            is_historical=is_historical,
            is_base_data=is_base_data,
        )

        # update trackers, gatherers on base data
        if not is_historical:
            if is_base_data:
                # - mark instrument as updated
                self._updated_instruments.add(instrument)

                # - update position price
                self._account.update_position_price(self._time_provider.time(), instrument, _update)

                # - update tracker
                _targets_from_tracker = self._get_tracker_for(instrument).update(self._context, instrument, _update)

                # - notify position gatherer for the new target positions
                if _targets_from_tracker:
                    # - tracker generated new targets on update, notify position gatherer
                    self._position_gathering.alter_positions(
                        self._context, self.__preprocess_and_log_target_positions(self._as_list(_targets_from_tracker))
                    )

                # - check for stale data periodically (only for base data updates)
                # This ensures we only check when we have new meaningful data
                if self._stale_data_detection_enabled and self._context._strategy_state.is_on_start_called:
                    stale_instruments = self._stale_data_detector.detect_stale_instruments(self._context.instruments)
                    if stale_instruments:
                        for instr in stale_instruments:
                            logger.info(f"Detected stale data for instrument {instr.symbol}")
                        logger.info(
                            f"Removing {len(stale_instruments)} stale instruments from universe: {[i.symbol for i in stale_instruments]}"
                        )
                        self._universe_manager.remove_instruments(stale_instruments, if_has_position_then="close")
            else:
                # - if it's not base data, we need to process it as market data
                self._account.process_market_data(self._time_provider.time(), instrument, _update)

        return is_base_data and not self._trigger_on_time_event

    ###########################################################################
    # - Handlers for different types of incoming data
    ###########################################################################

    # it's important that we call it with _process to not include in the handlers map
    def _process_custom_event(
        self, instrument: Instrument | None, event_type: str, event_data: Any
    ) -> MarketEvent | None:
        # Handle custom scheduled events
        if event_type in self._custom_scheduled_methods:
            try:
                method = self._custom_scheduled_methods[event_type]
                method(self._context)
                logger.debug(f"[ProcessingManager] :: Executed custom scheduled method for event: {event_type}")
            except Exception as e:
                logger.error(
                    f"[ProcessingManager] :: Error executing custom scheduled method for event {event_type}: {e}"
                )
                logger.opt(colors=False).error(traceback.format_exc())
            # Don't return a MarketEvent for custom scheduled events - they shouldn't trigger strategy.on_event
            return None

        if instrument is not None:
            self.__update_base_data(instrument, event_type, event_data)

        elif instrument is None and isinstance(event_data, dict):
            for _instrument, data in event_data.items():
                if isinstance(_instrument, Instrument):
                    self.__update_base_data(_instrument, event_type, data)

        return MarketEvent(self._time_provider.time(), event_type, instrument, event_data)

    def _process_hist_event(self, instrument: Instrument, event_type: str, event_data: Any) -> None:
        if not isinstance(event_data, list):
            event_data = [event_data]
        if DataType.OHLC == event_type:
            # - update ohlc using the list directly, this allows to update
            # multiple timeframes with different data (1h can have more bars than 1m)
            _, sub_params = DataType.from_str(event_type)
            timeframe = sub_params.get("timeframe", self._cache.default_timeframe)

            self._cache.update_by_bars(instrument, timeframe, event_data)
        else:
            for data in event_data:
                self.__update_base_data(instrument, event_type, data, is_historical=True)

    def _handle_event(self, instrument: Instrument, event_type: str, event_data: Any) -> TriggerEvent:
        return TriggerEvent(self._time_provider.time(), event_type, instrument, event_data)

    def _handle_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent:
        return TriggerEvent(self._time_provider.time(), event_type, instrument, data)

    def _handle_service_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent | None:
        """It is used by simulation as a dummy to trigger actual time events."""
        pass

    def _handle_start(self) -> None:
        if not self._is_data_ready():
            return
        self._strategy.on_start(self._context)
        self._context._strategy_state.is_on_start_called = True

    def _handle_state_resolution(self) -> None:
        if not self._is_data_ready():
            return

        _ctx = self._context
        resolver = _ctx.initializer.get_state_resolver()
        if resolver is None:
            logger.warning("No state resolver found, skipping state resolution")
            return

        self._log_state_mismatch()

        resolver_name = (
            getattr(resolver, "__name__", str(resolver))
            if callable(resolver) and hasattr(resolver, "__name__")
            else resolver.__class__.__name__
        )

        logger.info(f"<yellow>Resolving state mismatch with:</yellow> <g>{resolver_name}</g>")

        resolver(_ctx, _ctx.get_warmup_positions(), _ctx.get_warmup_orders(), _ctx.get_warmup_active_targets())

    def _handle_warmup_finished(self) -> None:
        if not self._is_data_ready():
            return
        self._strategy.on_warmup_finished(self._context)
        self._context._strategy_state.is_on_warmup_finished_called = True

    def _handle_fit(self, instrument: Instrument | None, event_type: str, data: tuple[dt_64 | None, dt_64]) -> None:
        """
        When scheduled fit event is happened - we need to invoke strategy on_fit method
        """
        if not self._is_data_ready():
            return
        self._fit_is_running = True
        current_time = data[1]
        self._cache.finalize_ohlc_for_instruments(current_time, self._context.instruments)
        self._run_in_thread_pool(self.__invoke_on_fit)

    def _handle_delisting_check(
        self, instrument: Instrument | None, event_type: str, data: tuple[dt_64 | None, dt_64]
    ) -> None:
        """
        Daily delisting check - close positions for instruments delisting within 1 day.
        This is a system-wide scheduled event, so instrument will be None.
        """
        if not self._is_data_ready():
            return

        logger.debug("Performing daily delisting check")

        current_time = data[1]
        current_timestamp = pd.Timestamp(current_time, unit="ns")
        one_day_ahead = current_timestamp + pd.Timedelta(days=1)

        # Find instruments delisting within 1 day
        instruments_to_close = []
        for instr in self._context.instruments:
            if instr.delist_date is not None:
                delist_timestamp = pd.Timestamp(instr.delist_date).tz_localize(None)
                if delist_timestamp <= one_day_ahead:
                    instruments_to_close.append(instr)

        if instruments_to_close:
            logger.info(
                f"Found {len(instruments_to_close)} instruments scheduled for delisting: {instruments_to_close}"
            )

            # Force close positions and remove from universe
            self._universe_manager.remove_instruments(
                instruments_to_close,
                if_has_position_then="close",  # Force close positions
            )

            logger.info("Closed positions and removed instruments scheduled for delisting")

    def _handle_ohlc(self, instrument: Instrument, event_type: str, bar: Bar) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, bar)
        return MarketEvent(self._time_provider.time(), event_type, instrument, bar, is_trigger=base_update)

    def _handle_trade(self, instrument: Instrument, event_type: str, trade: Trade) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, trade)
        return MarketEvent(self._time_provider.time(), event_type, instrument, trade, is_trigger=base_update)

    def _handle_orderbook(self, instrument: Instrument, event_type: str, orderbook: OrderBook) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, orderbook)
        return MarketEvent(self._time_provider.time(), event_type, instrument, orderbook, is_trigger=base_update)

    def _handle_quote(self, instrument: Instrument, event_type: str, quote: Quote) -> MarketEvent:
        base_update = self.__update_base_data(instrument, event_type, quote)
        return MarketEvent(self._time_provider.time(), event_type, instrument, quote, is_trigger=base_update)

    def _handle_funding_payment(
        self, instrument: Instrument, event_type: str, funding_payment: FundingPayment
    ) -> MarketEvent:
        # Apply funding payment to position
        self._account.process_funding_payment(instrument, funding_payment)

        # Continue with existing event processing
        base_update = self.__update_base_data(instrument, event_type, funding_payment)
        return MarketEvent(self._time_provider.time(), event_type, instrument, funding_payment, is_trigger=base_update)

    def _handle_error(self, instrument: Instrument | None, event_type: str, error: BaseErrorEvent) -> None:
        self._strategy.on_error(self._context, error)

    def _handle_order(self, instrument: Instrument, event_type: str, order: Order) -> Order:
        with self._health_monitor("ctx.handle_order"):
            self._account.process_order(order)
            return order

    def _handle_deals(self, instrument: Instrument | None, event_type: str, deals: list[Deal]) -> TriggerEvent | None:
        with self._health_monitor("ctx.handle_deals"):
            if instrument is None:
                logger.debug(
                    f"[<y>{self.__class__.__name__}</y>] :: Execution report for unknown instrument <r>{instrument}</r>"
                )
                return None

            # - process deals only for subscribed instruments
            self._account.process_deals(instrument, deals)
            self._logging.save_deals(instrument, deals)

            # - Process all deals first
            for d in deals:
                # - notify position gatherer and tracker
                self._position_gathering.on_execution_report(self._context, instrument, d)
                self._get_tracker_for(instrument).on_execution_report(self._context, instrument, d)

                logger.debug(
                    f"[<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: executed <r>{d.order_id}</r> | {d.amount} @ {d.price}"
                )

            if self._exporter is not None and (q := self._market_data.quote(instrument)) is not None:
                # - export position changes if exporter is available
                self._exporter.export_position_changes(
                    time=self._time_provider.time(),
                    instrument=instrument,
                    price=q.mid_price(),
                    account=self._account,
                )

            # - notify universe manager about position change
            self._universe_manager.on_alter_position(instrument)

            # - process active targets: if we got 0 position after executions remove current position from active
            if not self._context.get_position(instrument).is_open():
                self._active_targets.pop(instrument, None)

            return None

    def _is_order_update(self, d_type: str) -> bool:
        return d_type in ["order", "deals"]

    def _log_state_mismatch(self) -> None:
        logger.info("<yellow>State comparison between warmup and current state:</yellow>")

        warmup_positions, warmup_orders = self._context.get_warmup_positions(), self._context.get_warmup_orders()

        positions = self._account.get_positions()
        orders = self._account.get_orders()
        instrument_to_orders = defaultdict(list)
        for o in orders.values():
            instrument_to_orders[o.instrument].append(o)

        all_instruments = (
            set(warmup_positions.keys())
            | set(positions.keys())
            | set(warmup_orders.keys())
            | set(instrument_to_orders.keys())
        )

        for instrument in sorted(all_instruments, key=lambda x: x.symbol):
            # Get positions for this instrument
            warmup_pos = warmup_positions.get(instrument)
            current_pos = positions.get(instrument)

            # Get orders for this instrument
            warmup_ord = warmup_orders.get(instrument, [])
            current_ord = instrument_to_orders.get(instrument, [])

            # Format position information
            warmup_pos_info = f"size={warmup_pos.quantity:.6f}" if warmup_pos else "None"
            current_pos_info = f"size={current_pos.quantity:.6f}" if current_pos else "None"

            # Format order information
            warmup_ord_info = f"{len(warmup_ord)} orders" if warmup_ord else "No orders"
            current_ord_info = f"{len(current_ord)} orders" if current_ord else "No orders"

            # Determine if there's a mismatch
            pos_mismatch = (warmup_pos is None) != (current_pos is None) or (
                warmup_pos and current_pos and abs(warmup_pos.quantity - current_pos.quantity) > 1e-10
            )
            ord_mismatch = len(warmup_ord) != len(current_ord)

            # Set color based on mismatch
            color = "<r>" if pos_mismatch or ord_mismatch else "<g>"

            logger.info(
                f"{color}{instrument.symbol}</> - "
                f"Warmup: [Position: {warmup_pos_info}, {warmup_ord_info}] | "
                f"Current: [Position: {current_pos_info}, {current_ord_info}]"
            )

    def get_active_targets(self) -> dict[Instrument, TargetPosition]:
        return self._active_targets

    def emit_signal(self, signal: Signal) -> None:
        # - add signal to the queue. it will be processed in the data processing loop
        self._emitted_signals.append(signal)
