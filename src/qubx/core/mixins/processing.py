import asyncio
import concurrent.futures
import inspect
import traceback
import uuid
from collections import defaultdict
from types import FunctionType
from typing import Any, Callable

from qubx import logger
from qubx.core.account_manager import AccountManager
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.basics import (
    DataType,
    Deal,
    InitializingSignal,
    Instrument,
    MarketEvent,
    OrderChange,
    RestoredState,
    Signal,
    TargetPosition,
    Timestamped,
    TriggerEvent,
    dt_64,
    td_64,
)
from qubx.core.connector import IConnector, IMarketDataSink
from qubx.core.detectors import DelistingDetector, StaleDataDetector
from qubx.core.errors import BaseErrorEvent
from qubx.core.events import (
    AccountMessage,
    ChannelMessage,
    DealEvent,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import InvalidOrderTransition, StrategyExceededMaxNumberOfRuntimeFailuresError
from qubx.core.helpers import BasicScheduler, process_schedule_spec
from qubx.core.interfaces import (
    IHealthMonitor,
    IMarketDataCache,
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
from qubx.state.dummy import DummyStatePersistence
from qubx.trackers.riskctrl import _InitializationStageTracker
from qubx.utils.throttler import InstrumentThrottler
from qubx.utils.time import interval_to_cron


def validate_account_callback_signatures(strategy: IStrategy) -> None:
    """One-time construction guard for the unified account callbacks.

    A strategy overriding ``on_order`` / ``on_execution`` / ``on_position_change`` with a
    stale arity would die in a TypeError on every dispatch, swallowed by ``_safe_call`` —
    fail loudly here instead. Same for the removed pre-collapse callback names: they would
    silently never fire.
    """
    if not isinstance(strategy, IStrategy):
        return
    for removed in ("on_order_update", "on_deals", "on_account_update"):
        if callable(getattr(type(strategy), removed, None)):
            raise TypeError(
                f"{type(strategy).__name__} defines {removed}, which no longer exists: "
                "on_order_update/on_deals/on_account_update were replaced by "
                "on_order/on_execution/on_position_change — see docs/account-management/design.md"
            )
    for name in ("on_order", "on_execution", "on_position_change"):
        impl = getattr(type(strategy), name)
        base = getattr(IStrategy, name)
        if impl is base:
            continue
        params = list(inspect.signature(impl).parameters.values())
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            continue
        positional = [
            p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        required = [p for p in positional if p.default is inspect.Parameter.empty]
        n_expected = len(inspect.signature(base).parameters)  # incl. self
        if len(required) > n_expected or len(positional) < n_expected:
            expected = ", ".join(inspect.signature(base).parameters)
            raise TypeError(
                f"{type(strategy).__name__}.{name}{inspect.signature(impl)} has an incompatible signature: "
                f"the framework calls it as {name}({expected}) — update the override to match"
            )


class ProcessingManager(IProcessingManager):
    MAX_NUMBER_OF_STRATEGY_FAILURES: int = 10
    DATA_READY_TIMEOUT: td_64 = td_64(60, "s")
    # - after market data is ready, wait up to this long for the INITIAL account snapshot before
    #   proceeding anyway (so a missing/failed snapshot never hangs the strategy forever).
    ACCOUNT_SYNC_TIMEOUT: td_64 = td_64(15, "s")

    _context: IStrategyContext
    _strategy: IStrategy
    _logging: StrategyLogging
    _market_data: IMarketManager
    _subscription_manager: ISubscriptionManager
    _time_provider: ITimeProvider
    _account_manager: AccountManager
    _connectors: dict[str, IConnector]
    _position_tracker: PositionsTracker
    _position_gathering: IPositionGathering
    _cache: IMarketDataCache
    _scheduler: BasicScheduler
    _universe_manager: IUniverseManager
    _exporter: ITradeDataExport | None = None
    _health_monitor: IHealthMonitor
    _stale_data_detector: StaleDataDetector
    _delisting_detector: DelistingDetector
    _data_throttler: InstrumentThrottler | None

    _handlers: dict[str, Callable[["ProcessingManager", Instrument, str, Any], TriggerEvent | None]]
    _strategy_name: str

    _trigger_on_time_event: bool = False
    # Same-thread re-entrancy guards: on_fit/on_warmup_finished run inline on the processing
    # thread and may pump events that re-enter the trigger tail before is_on_fit_called /
    # is_on_warmup_finished_called flip (set only after the callback returns).
    _fit_is_running: bool = False
    _warmup_finished_is_running: bool = False
    _fails_counter: int = 0
    _is_simulation: bool
    _trig_bar_freq_nsec: int | None = None
    _cur_sim_step: int | None = None
    _updated_instruments: set[Instrument] = set()
    _data_ready_start_time: dt_64 | None = None
    _last_data_ready_log_time: dt_64 | None = None
    _all_instruments_ready_logged: bool = False
    _account_sync_deadline: dt_64 | None = None  # - startup: bound the wait for the initial snapshot
    _account_sync_timeout_logged: bool = False
    _emitted_signals: list[Signal] = []  # signals that were emitted
    _active_targets: dict[Instrument, TargetPosition] = {}
    _pending_no_quote_signals: dict[Instrument, Signal] = {}  # signals waiting for quote to arrive

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
        account_manager: AccountManager,
        connectors: dict[str, IConnector],
        position_tracker: PositionsTracker,
        position_gathering: IPositionGathering,
        universe_manager: IUniverseManager,
        scheduler: BasicScheduler,
        is_simulation: bool,
        health_monitor: IHealthMonitor,
        delisting_detector: DelistingDetector,
        exporter: ITradeDataExport | None = None,
        data_throttler: InstrumentThrottler | None = None,
    ):
        validate_account_callback_signatures(strategy)
        self._context = context
        self._strategy = strategy
        self._logging = logging
        self._market_data = market_data
        self._subscription_manager = subscription_manager
        self._time_provider = time_provider
        self._account_manager = account_manager
        self._connectors = connectors
        self._is_simulation = is_simulation
        self._position_gathering = position_gathering
        self._position_tracker = position_tracker
        self._universe_manager = universe_manager
        self._scheduler = scheduler
        self._exporter = exporter
        self._health_monitor = health_monitor
        self._delisting_detector = delisting_detector
        self._data_throttler = data_throttler
        self._cache = market_data.get_market_data_cache()

        # Initialize stale data detector with default disabled state
        # Will be configured later based on strategy settings
        self._stale_data_detector = StaleDataDetector(
            market_data_provider=market_data,
            time_provider=time_provider,
        )
        self._stale_data_detection_enabled = False

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
        self._account_sync_deadline = None
        self._account_sync_timeout_logged = False
        self._emitted_signals = []

        # - special tracker for post-warmup initialization signals
        self._init_stage_position_tracker = _InitializationStageTracker()
        self._instruments_in_init_stage = set()
        self._active_targets = {}
        self._custom_scheduled_methods = {}
        self._pending_no_quote_signals = {}

        # - schedule daily delisting check at 23:30 (end of day)
        self._scheduler.schedule_event("30 23 * * *", "delisting_check")

    def set_fit_schedule(self, schedule: str) -> None:
        logger.info(f"Setting fit schedule to {schedule}")
        rule = process_schedule_spec(schedule)
        if rule.get("type") != "cron":
            raise ValueError("Only cron type is supported for fit schedule")
        self._scheduler.unschedule_event("fit")
        self._scheduler.schedule_event(rule["schedule"], "fit")

    def set_event_schedule(self, schedule: str) -> None:
        logger.info(f"Setting event schedule to {schedule}")
        rule = process_schedule_spec(schedule)
        if not rule or "type" not in rule:
            raise ValueError(f"Can't recognize schedule format: '{schedule}'")

        if rule["type"] != "cron":
            raise ValueError("Only cron type is supported for event schedule")

        # Unschedule existing time event first to avoid conflicts
        self._scheduler.unschedule_event("time")

        self._scheduler.schedule_event(rule["schedule"], "time")
        self._trigger_on_time_event = True

    def get_event_schedule(self, event_id: str) -> str | None:
        return self._scheduler.get_schedule_for_event(event_id)

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> str:
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

        return event_id

    def unschedule(self, event_id: str) -> bool:
        return self._scheduler.unschedule_event(event_id)

    def delay(self, duration: str, method: Callable[[IStrategyContext], None]) -> str:
        """
        Schedule a method to run once after a delay.

        Args:
            duration: Delay period (e.g., "30s", "5Min", "1h")
            method: Method to call once - should accept IStrategyContext

        Returns:
            str: Event ID (can be used with unschedule() to cancel)
        """
        # Generate unique event ID
        event_id = f"delay_{str(uuid.uuid4()).replace('-', '_')}"

        # Store the method reference
        self._custom_scheduled_methods[event_id] = method

        # Schedule the delayed event
        self._scheduler.delay(duration, event_id)

        return event_id

    def trigger_fit(self) -> None:
        """Run on_fit once, on demand. Schedules a one-off event that invokes the
        fit handler on the strategy thread (via delay -> _handle_fit); uses a unique
        delay event id, so the recurring fit schedule is untouched."""
        # short delay defers the fit onto the strategy thread via the scheduler/channel
        # (any non-zero delay works); a unique delay event id leaves the recurring fit untouched.
        # Bind _handle_fit on the processing manager itself: the scheduler invokes the delayed
        # method with the context, which (composition, not mixin inheritance) has no _handle_fit.
        self.delay("1s", lambda _ctx: self._handle_fit(None, "fit", (None, self._time_provider.time())))

    def configure_stale_data_detection(
        self, enabled: bool, detection_period: str | None = None, check_interval: str | None = None
    ) -> None:
        """
        Configure stale data detection settings.

        Args:
            enabled: Whether to enable stale data detection
            detection_period: Period to consider data as stale (e.g., "5Min", "1h"). If None, uses detector default.
            check_interval: Interval between stale data checks (e.g., "30s", "1Min"). If None, uses detector default.
                           This controls how often the scheduled check runs.
        """
        self._stale_data_detection_enabled = enabled

        # Unschedule any existing stale data check
        self._scheduler.unschedule_event("stale_data_check")

        if enabled:
            if detection_period is not None:
                self._stale_data_detector = StaleDataDetector(
                    self._market_data, time_provider=self._time_provider, detection_period=detection_period
                )

            # Schedule periodic stale data check based on check_interval
            if check_interval is not None:
                cron_schedule = interval_to_cron(check_interval)
            else:
                cron_schedule = interval_to_cron("10m")  # default check interval
            self._scheduler.schedule_event(cron_schedule, "stale_data_check")

    def configure_state_snapshot(self, interval: str | None) -> None:
        """
        Configure periodic state snapshot persistence.

        When enabled, periodically saves strategy state (capital, positions, leverage, orders)
        to the configured state persistence backend (e.g., Redis) under the key "state".
        Only active in live mode when a real state persistence backend is configured.

        Args:
            interval: Snapshot interval (e.g., "5s", "30s", "1m"). None to disable.
        """
        self._scheduler.unschedule_event("state_snapshot")

        if interval is not None and not self._is_simulation:
            if not isinstance(self._context.persistence, DummyStatePersistence):
                cron_schedule = interval_to_cron(interval)
                self._scheduler.schedule_event(cron_schedule, "state_snapshot")
                logger.info(f"State snapshot enabled with interval: {interval}")

    def configure_rate_limit_metrics(self, interval: str | None) -> None:
        """
        Configure periodic rate limit metric emission.

        Emits pool utilization, gate state, hit counts etc. for all registered
        rate limiters via ctx.emitter.emit(). Only active in live mode.

        Args:
            interval: Emission interval (e.g., "60s", "1m"). None to disable.
        """
        self._scheduler.unschedule_event("rate_limit_metrics")

        if interval is not None and not self._is_simulation:
            cron_schedule = interval_to_cron(interval)
            self._scheduler.schedule_event(cron_schedule, "rate_limit_metrics")
            self._custom_scheduled_methods["rate_limit_metrics"] = self._emit_rate_limit_metrics
            logger.info(f"Rate limit metrics emission enabled with interval: {interval}")

    def _emit_rate_limit_metrics(self, ctx) -> None:
        """Emit rate limit metrics for all registered rate limiters."""
        if ctx.emitter is None or ctx.event_loop is None:
            return

        for exchange_name, rate_limiter in ctx.rate_limiters.items():
            try:
                future = asyncio.run_coroutine_threadsafe(rate_limiter.collect_metrics(), ctx.event_loop)
                metrics = future.result(timeout=5)
                self._do_emit_metrics(ctx, metrics)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Rate limit metrics collection timed out for {exchange_name}")
            except Exception as e:
                logger.opt(colors=False).error(f"Failed to collect rate limit metrics for {exchange_name}: {e}")

    @staticmethod
    def _do_emit_metrics(ctx, metrics: list[dict]) -> None:
        for m in metrics:
            ctx.emitter.emit(m["name"], m["value"], m["tags"])

    def process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        # Non-account events ride tuples through here; account events go to process_event.
        should_stop = self.__process_data(instrument, d_type, data, is_historical)
        if not is_historical:
            self._logging.notify(self._time_provider.time())
            if self._context.emitter is not None:
                self._context.emitter.notify(self._context)
        return should_stop

    def is_fitted(self) -> bool:
        return self._context._strategy_state.is_on_fit_called

    def __process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        if (
            not is_historical
            and self._data_throttler is not None
            and not self._data_throttler.should_send(d_type, instrument)
        ):
            return False

        if not d_type:
            handler = None
        else:
            handler = self._handlers.get(d_type)
            if handler is None:
                _dtype, _ = DataType.from_str(d_type)
                handler = self._handlers.get(_dtype.value)

        if not d_type:
            event = None
        elif is_historical:
            event = self._process_hist_event(instrument, d_type, data)
        elif handler:
            event = handler(self, instrument, d_type, data)
        else:
            event = self._process_custom_event(instrument, d_type, data)

        return self._run_strategy_pipeline(event)

    def _run_strategy_pipeline(self, event: Any) -> bool:
        """Warmup/fit gating, strategy callback firing and signal processing — the
        shared tail of the tuple data path (__process_data)."""
        if not self._context._strategy_state.is_on_start_called:
            self._handle_start()

        if (
            not self._context._strategy_state.is_on_warmup_finished_called
            and not self._context._strategy_state.is_warmup_in_progress
            and not self._warmup_finished_is_running
        ):
            if self._context.get_warmup_positions() or self._context.get_warmup_orders():
                self._handle_state_resolution()

            # Restore tracker and gatherer state if available
            restored_state = self._context.get_restored_state()
            if restored_state is not None:
                self._restore_tracker_and_gatherer_state(restored_state)

            self._handle_warmup_finished()

        # - check if it still didn't call on_fit() for first time
        if (
            not self._context._strategy_state.is_on_fit_called
            and not self._fit_is_running
            and not self._warmup_finished_is_running
        ):
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
        if self._fit_is_running or self._warmup_finished_is_running:
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

            # TODO: remove the is_trigger logic from market events, only trigger on_event when event schedule is provided
            # if _is_trigger_ev or (_is_market_ev and event.is_trigger):
            if _is_trigger_ev:
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

            self._subscription_manager.commit()  # apply pending operations

        except Exception as strat_error:
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
                    logger.debug(f"Switching tracker for <g>{instr}</g> to post-warmup initialization")
            else:
                _std_signals.append(signal)
                if instr in self._instruments_in_init_stage:
                    _cancel_init_stage_instruments_tracker.add(instr)
                    self._instruments_in_init_stage.remove(instr)
                    logger.debug(f"Switching tracker for <g>{instr}</g> back to defined position tracker")

        return _std_signals, _init_signals, _cancel_init_stage_instruments_tracker

    def __process_signals(self, signals: list[Signal]):
        _targets_from_trackers: list[TargetPosition] = []

        # - separate signals with/without quotes; signals without quotes are stored for retry when quote arrives
        signals_with_quote: list[Signal] = []
        for signal in signals:
            # Skip service signals - they are handled elsewhere
            if signal.is_service:
                signals_with_quote.append(signal)
                continue

            # Check if quote is available for this instrument
            if self._market_data.quote(signal.instrument) is not None:
                signals_with_quote.append(signal)
            else:
                # Store latest signal per instrument (replaces older ones)
                self._pending_no_quote_signals[signal.instrument] = signal
                logger.warning(
                    f"Signal for <g>{signal.instrument}</g> pending - no quote available yet (will retry when quote arrives)"
                )

        # Use only signals with quotes for further processing
        signals = signals_with_quote

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
            self._exporter.export_signals(self._time_provider.time(), signals, self._account_manager)

        # - emit signals to metric emitters if available
        if self._context.emitter is not None and signals:
            self._context.emitter.emit_signals(
                self._time_provider.time(), signals, self._account_manager, _targets_from_trackers
            )

    def __invoke_on_fit(self) -> None:
        with self._health_monitor("ctx.on_fit"):
            try:
                # - enforce the blacklist before fitting: refresh the cache AND force-close
                #   any held blacklisted positions (no change callbacks), so on_fit's
                #   get_universe / filter_blacklisted select over current data and no
                #   blacklisted instrument is held into the rebalance. No-op for the Null
                #   instrument service.
                self._context._instrument_service_manager.enforce_at_fit()
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
                self._context._strategy_state.is_on_fit_called = True

    def __invoke_on_warmup_finished(self) -> None:
        with self._health_monitor("ctx.on_warmup_finished"):
            try:
                logger.debug(
                    f"[<y>{self.__class__.__name__}</y>] :: Invoking <g>{self._strategy_name}</g> on_warmup_finished"
                )
                self._strategy.on_warmup_finished(self._context)
                self._subscription_manager.commit()
                logger.debug(
                    f"[<y>{self.__class__.__name__}</y>] :: <g>{self._strategy_name}</g> warmup finished completed"
                )
            except Exception as strat_error:
                logger.error(
                    f"[{self.__class__.__name__}] :: Strategy {self._strategy_name} on_warmup_finished raised an exception: {strat_error}"
                )
                logger.opt(colors=False).error(traceback.format_exc())
            finally:
                self._context._strategy_state.is_on_warmup_finished_called = True

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
                self._exporter.export_target_positions(
                    self._time_provider.time(), target_positions, self._account_manager
                )

        return filtered_target_positions

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
                    if not self._context.is_simulation:
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
                    if not self._context.is_simulation:
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
                    if not self._context.is_simulation:
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
                    if not self._context.is_simulation:
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

    def _is_ready(self) -> bool:
        """Startup readiness gate: market data ready AND the initial account snapshot applied.

        Gates on_start / on_fit / state-resolution / restore so those callbacks see REAL capital +
        venue positions instead of the pre-sync zero state (which would produce 0-capital sizing and
        transiently-flat restored positions). No account requirement in simulation (no venue snapshot).
        Falls through after ACCOUNT_SYNC_TIMEOUT so a missing/failed snapshot never hangs the strategy
        — mirrors the data-ready two-phase timeout; the reconciler heals state on the next snapshot.
        """
        if not self._is_data_ready():
            return False
        # - simulation & paper have a locally-seeded account (no venue snapshot to wait for)
        if self._context.is_simulation or self._context.is_paper_trading or self._account_manager.is_synced():
            return True
        # - data ready but the initial venue snapshot hasn't applied yet: wait, bounded.
        now = self._time_provider.time()
        if self._account_sync_deadline is None:
            self._account_sync_deadline = now + self.ACCOUNT_SYNC_TIMEOUT
            return False
        if now >= self._account_sync_deadline:
            if not self._account_sync_timeout_logged:
                logger.warning(
                    "Initial account snapshot not applied within timeout — starting without it; "
                    "capital/positions may be incomplete until the next reconcile"
                )
                self._account_sync_timeout_logged = True
            return True
        return False

    def __update_base_data(
        self, instrument: Instrument, event_type: str, data: Timestamped, is_historical: bool = False
    ) -> bool:
        """
        Updates the base data cache with the provided data.

        Returns:
            bool: True if the data is base data and the strategy should be triggered, False otherwise.
        """
        if not is_historical:
            self._health_monitor.on_data_arrival(instrument, event_type, dt_64(data.time, "ns"))
            # - paper trading: drive the simulated connector's OME with this tick so resting
            #   orders match BEFORE the strategy reacts. No-op in backtest (the runner feeds the
            #   OME directly via the connector) and in live (which executes at the venue).
            self._feed_simulated_connector(instrument, data)

        is_base_data, _update = self._is_base_data(data)

        # update cached ohlc is this is base subscription
        self._cache.update(instrument, event_type, _update, update_ohlc=is_base_data)

        # update trackers, gatherers on base data
        if not is_historical:
            if is_base_data:
                # - mark instrument as updated
                self._updated_instruments.add(instrument)

                # - check for pending signals and retry now that quote is available
                if instrument in self._pending_no_quote_signals:
                    pending_signal = self._pending_no_quote_signals.pop(instrument)
                    logger.info(f"Retrying pending signal for <g>{instrument}</g> - quote now available")
                    self._emitted_signals.append(pending_signal)

            # - mark-to-market the position from any market-data type (quote/trade/orderbook/bar)
            self._mark_to_market(instrument, _update)

            # - update tracker
            _targets_from_tracker = self._get_tracker_for(instrument).update(self._context, instrument, _update)

            # - notify position gatherer for the new target positions
            if _targets_from_tracker:
                # - tracker generated new targets on update, notify position gatherer
                self._position_gathering.alter_positions(
                    self._context, self.__preprocess_and_log_target_positions(self._as_list(_targets_from_tracker))
                )

            # - update position gatherer with market data
            self._position_gathering.update(self._context, instrument, _update)

        return is_base_data and not self._trigger_on_time_event

    def _mark_to_market(self, instrument: Instrument, data: Timestamped) -> None:
        # AM marks positions off the quote mid. Quotes mark directly; other market
        # data types reuse the latest cached quote (kept fresh by self._cache.update
        # above), which is what the simulated exchange emulates from trades/bars.
        if isinstance(data, Quote):
            self._account_manager.on_market_quote(instrument, data)
            return
        quote = self._market_data.quote(instrument)
        if quote is not None:
            self._account_manager.on_market_quote(instrument, quote)

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
                # logger.debug(f"[ProcessingManager] :: Executed custom scheduled method for event: {event_type}")
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

    def _handle_event(self, instrument: Instrument, event_type: str, event_data: Any) -> TriggerEvent:
        return TriggerEvent(self._time_provider.time(), event_type, instrument, event_data)

    def _handle_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent:
        return TriggerEvent(self._time_provider.time(), event_type, instrument, data)

    def _handle_service_time(self, instrument: Instrument, event_type: str, data: dt_64) -> TriggerEvent | None:
        """It is used by simulation as a dummy to trigger actual time events."""
        pass

    def _handle_start(self) -> None:
        if not self._is_ready():
            return
        self._strategy.on_start(self._context)
        self._context._strategy_state.is_on_start_called = True

    def _handle_state_resolution(self) -> None:
        if not self._is_ready():
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

    def _restore_tracker_and_gatherer_state(self, restored_state: RestoredState) -> None:
        """
        Restore state for position tracker and gatherer.

        Args:
            restored_state: The restored state containing signals and target positions
        """
        if not self._is_ready():
            return

        # Restore tracker state from signals
        all_signals = []
        for instrument, signals in restored_state.instrument_to_signal_positions.items():
            all_signals.extend(signals)

        if all_signals:
            logger.info(f"<yellow>Restoring tracker state from {len(all_signals)} signals</yellow>")
            self._position_tracker.restore_position_from_signals(self._context, all_signals)

        # Restore gatherer state from latest target positions only
        latest_targets = []
        for instrument, targets in restored_state.instrument_to_target_positions.items():
            if targets:  # Only if there are targets for this instrument
                # Get the latest target position (assuming they are sorted by time)
                latest_target = max(targets, key=lambda t: t.time)
                latest_targets.append(latest_target)

        if latest_targets:
            logger.info(f"<yellow>Restoring gatherer state from {len(latest_targets)} latest target positions</yellow>")
            self._position_gathering.restore_from_target_positions(self._context, latest_targets)

    def _handle_warmup_finished(self) -> None:
        if not self._is_ready():
            return
        # The flag reset lives in an outer finally so no raise (health monitor included)
        # can strand it True — a stranded flag silently disables the strategy forever.
        self._warmup_finished_is_running = True
        try:
            # Runs inline on the processing thread (single-mutator invariant): account events
            # queue in the channel until the callback returns.
            self.__invoke_on_warmup_finished()
        finally:
            self._warmup_finished_is_running = False

    def _handle_fit(self, instrument: Instrument | None, event_type: str, data: tuple[dt_64 | None, dt_64]) -> None:
        """
        When scheduled fit event is happened - we need to invoke strategy on_fit method
        """
        if not self._is_ready():
            return
        # The flag reset lives in an outer finally so no raise (finalize_ohlc or the health
        # monitor) can strand it True — a stranded flag silently disables the strategy
        # forever; on a raise is_on_fit_called stays False, so the next tick retries the fit.
        self._fit_is_running = True
        try:
            current_time = data[1]
            self._cache.finalize_ohlc_for_instruments(current_time, self._context.instruments)
            # Runs inline on the processing thread (single-mutator invariant): a long fit
            # blocks event processing until it returns.
            self.__invoke_on_fit()
        finally:
            self._fit_is_running = False

    def _handle_delisting_check(
        self, instrument: Instrument | None, event_type: str, data: tuple[dt_64 | None, dt_64]
    ) -> None:
        """
        Daily delisting check - close positions for instruments delisting within configured days.
        This is a system-wide scheduled event, so instrument will be None.
        """
        if not self._is_data_ready():
            return

        logger.debug(
            f"Performing daily delisting check (checking {self._delisting_detector.delisting_check_days} days ahead)"
        )

        # Find instruments delisting within configured days
        instruments_to_close = self._delisting_detector.detect_delistings(self._context.instruments)

        if instruments_to_close:
            logger.info(
                f"Found {len(instruments_to_close)} instruments scheduled for delisting: "
                f"{[instr.symbol for instr in instruments_to_close]}"
            )

            # Force close positions and remove from universe
            self._universe_manager.remove_instruments(
                instruments_to_close,
                if_has_position_then="close",  # Force close positions
            )

            logger.info("Closed positions and removed instruments scheduled for delisting")

    def _handle_stale_data_check(
        self, instrument: Instrument | None, event_type: str, data: tuple[dt_64 | None, dt_64]
    ) -> None:
        """
        Scheduled stale data check - detect and remove instruments with stale market data.
        """
        if not self._is_data_ready():
            return

        if not self._stale_data_detection_enabled or not self._context._strategy_state.is_on_start_called:
            return

        stale_instruments = self._stale_data_detector.detect_stale_instruments(self._context.instruments)
        if stale_instruments:
            for instr in stale_instruments:
                logger.info(f"Detected stale data for instrument {instr.symbol}")
            logger.info(
                f"Removing {len(stale_instruments)} stale instruments from universe: {[i.symbol for i in stale_instruments]}"
            )
            self._universe_manager.remove_instruments(stale_instruments, if_has_position_then="close")

    def _handle_state_snapshot(
        self, instrument: Instrument | None, event_type: str, data: tuple[dt_64 | None, dt_64]
    ) -> None:
        """
        Periodic state snapshot — persists current strategy state (capital, positions,
        leverage, orders, balances) per exchange to the configured state persistence
        backend for dashboard monitoring.
        """
        if not self._context._strategy_state.is_on_start_called:
            return

        account = self._context.account

        exchanges = self._context.exchanges

        exchanges_snapshot: dict[str, dict] = {}
        for exchange in exchanges:
            positions = account.get_positions(exchange)
            orders = account.get_orders(exchange=exchange)

            # Group orders by instrument symbol
            orders_by_symbol: dict[str, list[dict]] = defaultdict(list)
            for order in orders.values():
                orders_by_symbol[order.instrument.symbol].append(
                    {
                        "id": order.venue_order_id or order.client_order_id,
                        "type": order.type,
                        "side": order.side,
                        "quantity": order.quantity,
                        "price": order.price,
                        "status": order.status,
                    }
                )

            # Build positions snapshot
            positions_snapshot: dict[str, dict] = {}
            open_positions = 0
            for instr, pos in positions.items():
                if pos.is_open():
                    open_positions += 1
                positions_snapshot[instr.symbol] = {
                    "quantity": pos.quantity,
                    "avg_price": pos.position_avg_price,
                    "market_value": pos.market_value_funds,
                    "unrealized_pnl": pos.unrealized_pnl(),
                    "current_price": pos.last_update_price,
                    "leverage": account.get_leverage(instr),
                }

            # Add order-only instruments (no position yet) to orders snapshot
            for symbol in orders_by_symbol:
                if symbol not in positions_snapshot:
                    orders_by_symbol[symbol]  # ensure key exists in defaultdict

            # Build balances snapshot
            balances_snapshot: dict[str, dict] = {}
            for bal in account.get_balances(exchange):
                balances_snapshot[bal.currency] = {
                    "total": bal.total,
                    "free": bal.free,
                    "locked": bal.locked,
                }

            exchanges_snapshot[exchange] = {
                "capital": {
                    "total": account.get_total_capital(exchange),
                    "available": account.get_available_margin(exchange),
                },
                "balances": balances_snapshot,
                "net_leverage": account.get_net_leverage(exchange),
                "gross_leverage": account.get_gross_leverage(exchange),
                "open_positions": open_positions,
                "positions": positions_snapshot,
                "orders": dict(orders_by_symbol),
            }

        snapshot = {
            "timestamp": str(self._time_provider.time()),
            "exchanges": exchanges_snapshot,
        }

        try:
            self._context.persistence.save("state", snapshot)
        except Exception as e:
            logger.warning(f"Failed to save state snapshot: {e}")

    def _handle_error(self, instrument: Instrument | None, event_type: str, error: BaseErrorEvent) -> None:
        self._strategy.on_error(self._context, error)
        self._position_gathering.on_error(self._context, error)

    # ----------------------------------------------------------------------
    # - Typed-event dispatch (order/account lifecycle)
    # ----------------------------------------------------------------------

    def process_event(self, event: ChannelMessage) -> None:
        # Account-management events ride the typed channel and are applied to AccountState here.
        # Everything else — market data, scheduled triggers, custom/feature streams, errors —
        # rides (instrument, d_type, data, is_historical) tuples through process_data.
        # FundingPaymentEvent is an AccountMessage (it books into balances), so it dispatches
        # here like any other account event.
        if isinstance(event, AccountMessage):
            self._dispatch_account(event)
            return
        logger.warning(f"unknown event type: {type(event)}")

    def _feed_simulated_connector(self, instrument: Instrument, payload: Any) -> None:
        # Paper only. is_paper_trading is also True in the backtester (it too uses a
        # SimulatedConnector), but there the SimulationRunner already feeds the OME via the
        # connector per tick — so gate on `not _is_simulation` as well to avoid double-feeding
        # the OME in backtests. Live (not paper) executes at the venue and has no local OME.
        if self._is_simulation or not self._context.is_paper_trading:
            return
        connector = self._connectors.get(instrument.exchange)
        if connector is None:
            return
        if not isinstance(connector, IMarketDataSink):
            logger.warning(
                f"paper connector for {instrument.exchange} is not an IMarketDataSink "
                f"({type(connector).__name__}); skipping market-data feed"
            )
            return
        connector.process_market_data(instrument, payload)

    def _dispatch_account(self, event: AccountMessage) -> None:
        try:
            result = self._account_manager.apply(event)
        except InvalidOrderTransition as e:
            logger.warning(f"invalid transition: {e}; skipping")
            return
        except Exception:
            logger.exception(f"AM.apply raised on {type(event).__name__}")
            self._emit_error_metric("account_manager_apply_errors", event=type(event).__name__)
            return
        self._safe_fire_account_callback(event, result)

    def _emit_error_metric(self, name: str, value: float = 1.0, **tags: str) -> None:
        """Best-effort operational-counter emission; no-op when no emitter is configured."""
        emitter = self._context.emitter
        if emitter is not None:
            try:
                emitter.emit(name, value, tags)
            except Exception:
                logger.exception("metric emitter raised while recording error metric")

    def _safe_call(self, fn: Callable, *args) -> None:
        try:
            fn(self._context, *args)
        except Exception:
            logger.exception(f"strategy callback {getattr(fn, '__name__', fn)} raised")
            self._emit_error_metric("strategy_callback_errors", callback=getattr(fn, "__name__", "unknown"))

    def _safe_call_gatherer(self, fn: Callable, *args) -> None:
        # Same ApplyResult-driven dispatch as the strategy callbacks, but onto the position
        # gatherer (on_order / on_position_change). Default no-ops on IPositionGathering, so
        # gatherers that don't override pay nothing; error-isolated like the strategy fires.
        try:
            fn(self._context, *args)
        except Exception:
            logger.exception(f"position gatherer callback {getattr(fn, '__name__', fn)} raised")
            self._emit_error_metric(
                "strategy_callback_errors", callback=f"gatherer.{getattr(fn, '__name__', 'unknown')}"
            )

    def _safe_fire_account_callback(self, event: AccountMessage, result: ApplyResult) -> None:
        # Purely ApplyResult-driven dispatch onto the three strategy callbacks: each set
        # field fires its callback, an empty result means the AM suppressed the event
        # (late/duplicate/stale/terminal/unknown/deduped funding — already logged) and
        # nothing fires. A size-equal venue position push carries the LOCAL position in
        # result.position (the reducer never writes size off a push; drift triggers a
        # snapshot correction, firing nothing strategy-side). BalanceUpdateEvent
        # deliberately fires NO callback: balances are read via ctx.
        if isinstance(event, OrderEvent):
            # Cancel/update rejections are dangerous-but-recoverable (order still alive at the
            # venue): surface loudly here so they aren't lost if the strategy ignores them.
            if result.order_change is OrderChange.CANCEL_REJECTED and isinstance(event, OrderCancelRejectedEvent):
                logger.warning(
                    f"[{event.client_order_id}] cancel rejected by venue: {event.reason}; "
                    f"order is STILL ALIVE at the venue"
                )
            elif result.order_change is OrderChange.UPDATE_REJECTED and isinstance(event, OrderUpdateRejectedEvent):
                logger.warning(
                    f"[{event.client_order_id}] update rejected by venue: {event.reason}; "
                    f"order is STILL ALIVE with prior parameters"
                )
            if result.order is not None and result.order_change is not None:
                self._safe_call(self._strategy.on_order, result.order, result.order_change)
                self._safe_call_gatherer(self._position_gathering.on_order, result.order, result.order_change)
            if result.deal is not None:
                # A historical (RequestHistDeals) recovery deal is a reconciliation artifact, not a
                # live execution — the position was already corrected by the snapshot. Record it in
                # the ledger for the audit trail, but fire NOTHING strategy-facing: no on_execution,
                # no tracker/gatherer reaction, and no realize-only on_position_change below.
                if isinstance(event, DealEvent) and event.historical:
                    self._record_recovered_deal(event.instrument, result.deal)
                    return
                if event.instrument is not None:
                    self._safe_call(self._strategy.on_execution, event.instrument, result.deal)
                # A newly applied deal also drives framework-internal downstream consumers
                # (trackers, gatherers, logging, export) — independent of the strategy callback.
                # Keyed off result.deal, NOT event.fill: a re-delivered fill the AM deduped
                # must not reach save_deals/gatherers/trackers twice.
                self._notify_downstream_fill(event.instrument, result.deal)
        if result.position is not None:
            self._safe_call(self._strategy.on_position_change, result.position)
            self._safe_call_gatherer(self._position_gathering.on_position_change, result.position)
        # Reconciler-path snapshot corrections — one on_position_change per reconciled position.
        for position in result.positions:
            self._safe_call(self._strategy.on_position_change, position)
            self._safe_call_gatherer(self._position_gathering.on_position_change, position)

    def _record_recovered_deal(self, instrument: Instrument | None, fill: Deal) -> None:
        # Ledger-only sink for a historical recovery deal: persist it for the audit trail, but
        # never touch trackers/gatherers/exporter (those drive live trading reactions). The
        # internal position booking already happened in the reducer (realize-only).
        if instrument is None:
            return
        try:
            self._logging.save_deals(instrument, [fill])
        except Exception:
            logger.exception("deals writer raised (recovered deal)")
            self._emit_error_metric("downstream_fill_errors", consumer="save_deals")

    def _notify_downstream_fill(self, instrument: Instrument | None, fill: Deal) -> None:
        if instrument is None:
            logger.debug(f"[<y>{self.__class__.__name__}</y>] :: fill for unknown instrument; skipping downstream")
            return

        # Every consumer here is error-isolated: a persistent failure in one (e.g. an
        # I/O-backed deals writer) must not suppress the rest or the position callbacks
        # that the dispatcher fires after this method returns.
        try:
            self._logging.save_deals(instrument, [fill])
        except Exception:
            logger.exception("deals writer raised")
            self._emit_error_metric("downstream_fill_errors", consumer="save_deals")

        try:
            self._position_gathering.on_execution_report(self._context, instrument, fill)
        except Exception:
            logger.exception("position gatherer raised")
            self._emit_error_metric("downstream_fill_errors", consumer="gatherer")

        try:
            self._get_tracker_for(instrument).on_execution_report(self._context, instrument, fill)
        except Exception:
            logger.exception("position tracker raised")
            self._emit_error_metric("downstream_fill_errors", consumer="tracker")

        if self._exporter is not None and (q := self._market_data.quote(instrument)) is not None:
            try:
                _active = self._active_targets.get(instrument)
                self._exporter.export_position_changes(
                    time=self._time_provider.time(),
                    instrument=instrument,
                    price=q.mid_price(),
                    account=self._account_manager,
                    metadata=_active.options if _active else None,
                )
            except Exception:
                logger.exception("exporter raised")

        try:
            self._universe_manager.on_alter_position(instrument)
        except Exception:
            logger.exception("universe manager on_alter_position raised")
            self._emit_error_metric("downstream_fill_errors", consumer="on_alter_position")

        if self._context.emitter is not None:
            try:
                self._context.emitter.emit_deals(
                    time=self._time_provider.time(),
                    instrument=instrument,
                    deals=[fill],
                    account=self._account_manager,
                )
            except Exception:
                logger.exception("emitter raised")

        # - process active targets: if position closed after the fill, drop the active target
        pos = self._account_manager.get_position(instrument)
        if pos is None or not pos.is_open():
            self._active_targets.pop(instrument, None)

    def _log_state_mismatch(self) -> None:
        logger.info("<yellow>State comparison between warmup and current state:</yellow>")

        warmup_positions, warmup_orders = self._context.get_warmup_positions(), self._context.get_warmup_orders()

        positions = self._account_manager.get_positions()
        orders = self._account_manager.get_orders()
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

    def emit_signal(self, signal: Signal | list[Signal]) -> None:
        # - add signal to the queue. it will be processed in the data processing loop
        if isinstance(signal, list):
            self._emitted_signals.extend(signal)
        else:
            self._emitted_signals.append(signal)
