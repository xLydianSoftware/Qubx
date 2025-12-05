import atexit
import signal
import traceback
from functools import wraps
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from qubx import logger

if TYPE_CHECKING:
    from qubx.utils.throttler import InstrumentThrottler
from qubx.core.account import CompositeAccountProcessor
from qubx.core.basics import (
    AssetBalance,
    CtrlChannel,
    DataType,
    Instrument,
    ITimeProvider,
    MarketType,
    Order,
    OrderRequest,
    Position,
    RestoredState,
    Signal,
    TargetPosition,
    dt_64,
    td_64,
)
from qubx.core.detectors import DelistingDetector
from qubx.core.errors import BaseErrorEvent, ErrorLevel
from qubx.core.exceptions import StrategyExceededMaxNumberOfRuntimeFailuresError
from qubx.core.helpers import (
    BasicScheduler,
    CachedMarketDataHolder,
    set_parameters_to_object,
)
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import (
    IAccountProcessor,
    IBroker,
    IDataProvider,
    IHealthMonitor,
    IMarketManager,
    IMetricEmitter,
    IPositionGathering,
    IProcessingManager,
    IStrategy,
    IStrategyContext,
    IStrategyNotifier,
    ISubscriptionManager,
    ITradeDataExport,
    ITradingManager,
    ITransferManager,
    IUniverseManager,
    PositionsTracker,
    RemovalPolicy,
    StrategyState,
)
from qubx.core.loggers import StrategyLogging
from qubx.data.readers import DataReader
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.health import DummyHealthMonitor
from qubx.trackers.sizers import FixedSizer

from .mixins import (
    MarketManager,
    ProcessingManager,
    SubscriptionManager,
    TradingManager,
    UniverseManager,
)

DEFAULT_POSITION_TRACKER: Callable[[], PositionsTracker] = lambda: PositionsTracker(
    FixedSizer(1.0, amount_in_quote=False)
)


def check_transfer_manager(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._transfer_manager is None:
            raise RuntimeError(
                "Transfer manager not configured. "
                "For live mode, set via ctx.initializer.set_transfer_manager() in on_init(). "
                "For simulation mode, transfer manager is auto-assigned on start()."
            )

        return func(self, *args, **kwargs)

    return wrapper


class StrategyContext(IStrategyContext):
    _market_data_provider: IMarketManager
    _universe_manager: IUniverseManager
    _subscription_manager: ISubscriptionManager
    _trading_manager: ITradingManager
    _processing_manager: IProcessingManager

    _brokers: list[IBroker]  # service for exchange API: orders managemewnt
    _data_providers: list[IDataProvider]  # market data provider
    _logging: StrategyLogging  # recording all activities for the strat: execs, positions, portfolio
    _cache: CachedMarketDataHolder
    _scheduler: BasicScheduler
    _initial_instruments: list[Instrument]
    _strategy_name: str
    _delisting_detector: DelistingDetector
    _notifier: IStrategyNotifier
    _aux: DataReader | None

    _thread_data_loop: Thread | None = None  # market data loop
    _is_initialized: bool = False
    _exporter: ITradeDataExport | None = None  # Add exporter attribute
    _transfer_manager: ITransferManager | None = None  # Transfer manager for fund transfers

    _warmup_positions: dict[Instrument, Position] | None = None
    _warmup_orders: dict[Instrument, list[Order]] | None = None
    _warmup_active_targets: dict[Instrument, list[TargetPosition]] | None = None

    # Shutdown handling
    _is_stopping: bool = False
    _stop_lock: Lock
    _original_sigint_handler: Any = None
    _original_sigterm_handler: Any = None
    _atexit_registered: bool = False

    def __init__(
        self,
        strategy: IStrategy,
        brokers: list[IBroker],
        data_providers: list[IDataProvider],
        account: IAccountProcessor,
        scheduler: BasicScheduler,
        time_provider: ITimeProvider,
        instruments: list[Instrument],
        logging: StrategyLogging,
        config: dict[str, Any] | None = None,
        position_gathering: IPositionGathering | None = None,  # TODO: make position gathering part of the strategy
        aux_data_provider: DataReader | None = None,
        exporter: ITradeDataExport | None = None,
        emitter: IMetricEmitter | None = None,
        notifier: IStrategyNotifier | None = None,
        initializer: BasicStrategyInitializer | None = None,
        strategy_name: str | None = None,
        strategy_state: StrategyState | None = None,
        health_monitor: IHealthMonitor | None = None,
        restored_state: RestoredState | None = None,
        data_throttler: "InstrumentThrottler | None" = None,
    ) -> None:
        self.account = account
        self.strategy = self.__instantiate_strategy(strategy, config)
        self.emitter = emitter if emitter is not None else IMetricEmitter()
        self.initializer = (
            initializer
            if initializer is not None
            else BasicStrategyInitializer(simulation=data_providers[0].is_simulation)
        )

        # - additional sanity check that it's defined if we are in simulation or live mode
        if self.initializer.is_simulation is None:
            raise ValueError("Live or simulation mode must be defined in strategy initializer !")

        self._time_provider = time_provider
        self._brokers = brokers
        self._data_providers = data_providers
        self._logging = logging
        self._scheduler = scheduler
        self._initial_instruments = instruments

        self._cache = CachedMarketDataHolder()
        self._exporter = exporter
        self._notifier = notifier if notifier is not None else IStrategyNotifier()
        self._strategy_state = strategy_state if strategy_state is not None else StrategyState()
        self._strategy_name = strategy_name if strategy_name is not None else strategy.__class__.__name__
        self._restored_state = restored_state
        self._aux = aux_data_provider

        self._health_monitor = health_monitor or DummyHealthMonitor()
        self.health = self._health_monitor

        # Initialize shutdown handling
        self._stop_lock = Lock()
        self._is_stopping = False
        self._atexit_registered = False

        __position_tracker = self.strategy.tracker(self)
        if __position_tracker is None:
            __position_tracker = DEFAULT_POSITION_TRACKER()

        __position_gathering = self.strategy.gatherer(self)
        if __position_gathering is None:
            __position_gathering = position_gathering if position_gathering is not None else SimplePositionGatherer()

        self._subscription_manager = SubscriptionManager(
            time_provider=self._time_provider,
            data_providers=self._data_providers,
            health_monitor=self._health_monitor,
            strategy_state=self._strategy_state,
            default_base_subscription=DataType.ORDERBOOK
            if not self._data_providers[0].is_simulation
            else DataType.NONE,
        )
        self.account.set_subscription_manager(self._subscription_manager)

        self._market_data_provider = MarketManager(
            time_provider=self._time_provider,
            cache=self._cache,
            data_providers=self._data_providers,
            universe_manager=self,
            aux_data_provider=aux_data_provider,
        )

        # Create delisting detector to be shared between universe and processing managers
        self._delisting_detector = DelistingDetector(
            time_provider=self,
            delisting_check_days=self.initializer.get_delisting_check_days(),
        )

        self._universe_manager = UniverseManager(
            context=self,
            strategy=self.strategy,
            cache=self._cache,
            logging=self._logging,
            subscription_manager=self,
            trading_manager=self,
            time_provider=self,
            account=self.account,
            position_gathering=__position_gathering,
            delisting_detector=self._delisting_detector,
        )
        self._trading_manager = TradingManager(
            context=self,
            brokers=self._brokers,
            account=self.account,
            health_monitor=self._health_monitor,
            strategy_name=self._strategy_name,
        )
        self._processing_manager = ProcessingManager(
            context=self,
            strategy=self.strategy,
            logging=self._logging,
            market_data=self,
            subscription_manager=self,
            time_provider=self,
            account=self.account,
            position_tracker=__position_tracker,
            position_gathering=__position_gathering,
            universe_manager=self._universe_manager,
            cache=self._cache,
            scheduler=self._scheduler,
            is_simulation=self._data_providers[0].is_simulation,
            exporter=self._exporter,
            health_monitor=self._health_monitor,
            delisting_detector=self._delisting_detector,
            data_throttler=data_throttler,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self._strategy_state.is_on_init_called:
            self.strategy.on_init(self.initializer)
            self._strategy_state.is_on_init_called = True

        self._delisting_detector.delisting_check_days = self.initializer.get_delisting_check_days()

        if subscription_warmup := self.initializer.get_subscription_warmup():
            self.set_warmup(subscription_warmup)

        if base_sub := self.initializer.get_base_subscription():
            self.set_base_subscription(base_sub)

        if auto_sub := self.initializer.get_auto_subscribe():
            self.auto_subscribe = auto_sub

        if fit_schedule := self.initializer.get_fit_schedule():
            self.set_fit_schedule(fit_schedule)

        if event_schedule := self.initializer.get_event_schedule():
            self.set_event_schedule(event_schedule)

        if pending_global_subscriptions := self.initializer.get_pending_global_subscriptions():
            for sub_type in pending_global_subscriptions:
                self.subscribe(sub_type)

        if pending_instrument_subscriptions := self.initializer.get_pending_instrument_subscriptions():
            for sub_type, instruments in pending_instrument_subscriptions.items():
                self.subscribe(sub_type, list(instruments))

        if custom_schedules := self.initializer.get_custom_schedules():
            for schedule_id, (cron_schedule, method) in custom_schedules.items():
                self._processing_manager.schedule(cron_schedule, method)

        # Configure stale data detection based on strategy settings
        stale_data_config = self.initializer.get_stale_data_detection_config()
        self._processing_manager.configure_stale_data_detection(*stale_data_config)

        if self.is_simulation and isinstance(self.account, CompositeAccountProcessor):
            # Auto-assign simulation transfer manager
            from qubx.backtester.transfers import SimulationTransferManager

            self._transfer_manager = SimulationTransferManager(self.account, self._time_provider)
            logger.debug("[StrategyContext] :: Auto-assigned SimulationTransferManager")
        else:
            # In live mode, check if strategy set one via initializer
            self._transfer_manager = self.initializer.get_transfer_manager()
            if self._transfer_manager is not None:
                logger.info(f"[StrategyContext] :: Using transfer manager: {type(self._transfer_manager).__name__}")

        # - update cache default timeframe
        sub_type = self.get_base_subscription()
        _, params = DataType.from_str(sub_type)
        __default_timeframe = params.get("timeframe", "1sec")
        self._cache.update_default_timeframe(__default_timeframe)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signals (SIGINT, SIGTERM) for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.info(f"[StrategyContext] :: Received {sig_name} signal - initiating graceful shutdown")
        self.stop()

    @property
    def strategy_name(self) -> str:
        return self._strategy_name or self.strategy.__class__.__name__

    @property
    def aux(self) -> DataReader | None:
        return self._aux

    def start(self, blocking: bool = False):
        if self._is_initialized:
            raise ValueError("Strategy is already started !")

        # Register signal handlers for graceful shutdown
        try:
            self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError) as e:
            # Signal registration can fail in threads or non-main contexts
            logger.warning(f"[StrategyContext] :: Could not register signal handlers: {e}")

        # Register atexit handler as backup for abnormal termination
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        # - run cron scheduler
        self._scheduler.run()

        # - create incoming market data processing
        databus = self._data_providers[0].channel
        databus.register(self)

        # - start account processing
        self.account.start()

        # - start health metrics monitor
        self._health_monitor.start()

        # Update initial instruments if strategy set them after warmup
        if self.get_warmup_positions():
            self._initial_instruments = list(set(self.get_warmup_positions().keys()) | set(self._initial_instruments))

        # Add open positions to initial instruments
        open_positions = {k: p for k, p in self.get_positions().items() if p.is_open()}
        self._initial_instruments = list(set(open_positions.keys()) | set(self._initial_instruments))

        # Notify strategy start
        if self._notifier and not self.is_simulation:
            try:
                self._notifier.notify_start(
                    {
                        "Exchanges": "|".join(self.exchanges),
                        "Total Capital": f"${self.get_total_capital():,.0f}",
                        "Open Positions": len(open_positions),
                        "Instruments": len(self._initial_instruments),
                        "Mode": "Paper" if self.is_paper_trading else "Live",
                    },
                )
            except Exception as e:
                logger.error(f"[StrategyContext] :: Failed to notify strategy start: {e}")

        # - update universe with initial instruments after the strategy is initialized
        self.set_universe(self._initial_instruments, skip_callback=True)

        # - for live we run loop
        if not self.is_simulation:
            self._thread_data_loop = Thread(target=self.__process_incoming_data_loop, args=(databus,), daemon=True)
            self._thread_data_loop.start()
            logger.info("[StrategyContext] :: strategy is started in thread")
            if blocking:
                self._thread_data_loop.join()

        self._is_initialized = True

    def stop(self):
        """
        Stop the strategy context with robust shutdown handling.

        Features:
        - Double-stop prevention with lock
        - Priority-based cleanup (critical paths first)
        - Fault-tolerant (exceptions don't block other cleanup)
        - Thread cleanup with timeout
        - Signal handler and atexit cleanup
        """
        # Prevent concurrent or repeated stops
        with self._stop_lock:
            if self._is_stopping:
                logger.debug("[StrategyContext] :: Stop already in progress, skipping duplicate call")
                return
            self._is_stopping = True

        # PRIORITY 1: Critical path - always execute notifier and on_stop
        # These are the most important callbacks that must always run

        # Notify strategy stop
        if self._notifier and not self.is_simulation:
            try:
                self._notifier.notify_stop(
                    {
                        "Total Capital": f"{self.get_total_capital():,.0f}",
                        "Net Leverage": f"{self.get_net_leverage():.2%}",
                        "Positions": len([p for i, p in self.get_positions().items() if abs(p.quantity) > i.min_size]),
                        "Mode": "Paper" if self.is_paper_trading else "Live",
                    },
                )
            except Exception as e:
                logger.error(f"[StrategyContext] :: Failed to notify strategy stop: {e}")
                logger.opt(colors=False).error(traceback.format_exc())

        # Invoke strategy's stop code
        try:
            if not self.is_warmup_in_progress:
                self.strategy.on_stop(self)
                logger.debug("[StrategyContext] :: Strategy on_stop() completed")
        except Exception as strat_error:
            logger.error(
                f"[<y>StrategyContext</y>] :: Strategy {self._strategy_name} raised an exception in on_stop: {strat_error}"
            )
            logger.opt(colors=False).error(traceback.format_exc())

            # Notify strategy error
            if self._notifier:
                try:
                    self._notifier.notify_error(strat_error)
                except Exception as e:
                    logger.error(f"[StrategyContext] :: Failed to notify strategy error: {e}")

        # PRIORITY 2: Stop data providers and thread

        # Close data providers
        if self._thread_data_loop:
            try:
                for data_provider in self._data_providers:
                    try:
                        data_provider.close()
                        logger.debug(f"[StrategyContext] :: Closed data provider: {type(data_provider).__name__}")
                    except Exception as e:
                        logger.error(
                            f"[StrategyContext] :: Failed to close data provider {type(data_provider).__name__}: {e}"
                        )
            except Exception as e:
                logger.error(f"[StrategyContext] :: Error iterating data providers: {e}")

            # Stop the channel
            try:
                self._data_providers[0].channel.stop()
            except Exception as e:
                logger.error(f"[StrategyContext] :: Failed to stop data channel: {e}")

            # Join thread with timeout
            try:
                thread_timeout = 30.0  # 30 seconds timeout
                self._thread_data_loop.join(timeout=thread_timeout)
                if self._thread_data_loop.is_alive():
                    logger.warning(
                        f"[StrategyContext] :: Data loop thread did not stop within {thread_timeout}s timeout - may still be running"
                    )
                else:
                    logger.debug("[StrategyContext] :: Data loop thread stopped gracefully")
                self._thread_data_loop = None
            except Exception as e:
                logger.error(f"[StrategyContext] :: Error joining data loop thread: {e}")

        # PRIORITY 3: Stop account processing
        try:
            self.account.stop()
        except Exception as e:
            logger.error(f"[StrategyContext] :: Failed to stop account processor: {e}")
            logger.opt(colors=False).error(traceback.format_exc())

        # PRIORITY 4: Stop health metrics monitor
        try:
            self._health_monitor.stop()
        except Exception as e:
            logger.error(f"[StrategyContext] :: Failed to stop health monitor: {e}")
            logger.opt(colors=False).error(traceback.format_exc())

        # PRIORITY 5: Close logging
        try:
            self._logging.close()
        except Exception as e:
            logger.error(f"[StrategyContext] :: Failed to close logging: {e}")
            logger.opt(colors=False).error(traceback.format_exc())

        # CLEANUP: Restore signal handlers and deregister atexit
        try:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
                self._original_sigint_handler = None
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
                self._original_sigterm_handler = None
        except Exception as e:
            logger.warning(f"[StrategyContext] :: Failed to restore signal handlers: {e}")

        try:
            if self._atexit_registered:
                atexit.unregister(self.stop)
                self._atexit_registered = False
        except Exception as e:
            logger.warning(f"[StrategyContext] :: Failed to unregister atexit handler: {e}")

        logger.info("[StrategyContext] :: Strategy context stopped")

    def is_running(self):
        return self._thread_data_loop is not None and self._thread_data_loop.is_alive()

    @property
    def is_simulation(self) -> bool:
        return self._data_providers[0].is_simulation

    @property
    def is_paper_trading(self) -> bool:
        return self._brokers[0].is_simulated_trading

    @property
    def notifier(self) -> IStrategyNotifier:
        return self._notifier

    # IAccountViewer delegation

    # capital information
    def get_capital(self, exchange: str | None = None) -> float:
        return self.account.get_capital(exchange)

    def get_total_capital(self, exchange: str | None = None) -> float:
        return self.account.get_total_capital(exchange)

    def get_base_currency(self, exchange: str | None = None) -> str:
        return self.account.get_base_currency(exchange)

    # balance and position information
    def get_balances(self, exchange: str | None = None) -> list[AssetBalance]:
        return self.account.get_balances(exchange)

    def get_balance(self, currency: str, exchange: str | None = None) -> AssetBalance:
        return self.account.get_balance(currency, exchange)

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
        return self.account.get_positions(exchange)

    def get_position(self, instrument: Instrument) -> Position:
        return self.account.get_position(instrument)

    @property
    def positions(self):
        return self.account.get_positions()

    def get_orders(self, instrument: Instrument | None = None, exchange: str | None = None) -> dict[str, Order]:
        return self.account.get_orders(instrument, exchange)

    def position_report(self, exchange: str | None = None) -> dict:
        return self.account.position_report(exchange)

    # leverage information
    def get_leverage(self, instrument: Instrument) -> float:
        return self.account.get_leverage(instrument)

    def get_leverages(self, exchange: str | None = None) -> dict[Instrument, float]:
        return self.account.get_leverages(exchange)

    def get_net_leverage(self, exchange: str | None = None) -> float:
        return self.account.get_net_leverage(exchange)

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        return self.account.get_gross_leverage(exchange)

    # margin information
    def get_total_required_margin(self, exchange: str | None = None) -> float:
        return self.account.get_total_required_margin(exchange)

    def get_available_margin(self, exchange: str | None = None) -> float:
        return self.account.get_available_margin(exchange)

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        return self.account.get_margin_ratio(exchange)

    # IMarketDataProvider delegation
    def time(self) -> dt_64:
        return self._market_data_provider.time()

    def ohlc(self, instrument: Instrument, timeframe: str | td_64 | None = None, length: int | None = None):
        return self._market_data_provider.ohlc(instrument, timeframe, length)

    def ohlc_pd(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        length: int | None = None,
        consolidated: bool = True,
    ) -> pd.DataFrame:
        return self._market_data_provider.ohlc_pd(instrument, timeframe, length, consolidated)

    def quote(self, instrument: Instrument):
        return self._market_data_provider.quote(instrument)

    def get_data(self, instrument: Instrument, sub_type: str) -> list[Any]:
        return self._market_data_provider.get_data(instrument, sub_type)

    def get_aux_data(self, data_id: str, **parameters):
        return self._market_data_provider.get_aux_data(data_id, **parameters)

    def get_instruments(self):
        return self._market_data_provider.get_instruments()

    def query_instrument(self, symbol: str, exchange: str | None = None) -> Instrument | None:
        return self._market_data_provider.query_instrument(symbol, exchange)

    # ITradingManager delegation
    def trade(self, instrument: Instrument, amount: float, price: float | None = None, time_in_force="gtc", **options):
        # TODO: we need to generate target position and apply it in the processing manager
        # - one of the options is to have multiple entry levels in TargetPosition class
        return self._trading_manager.trade(instrument, amount, price, time_in_force, **options)

    def trade_async(
        self, instrument: Instrument, amount: float, price: float | None = None, time_in_force="gtc", **options
    ):
        return self._trading_manager.trade_async(instrument, amount, price, time_in_force, **options)

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[Order]:
        return self._trading_manager.submit_orders(order_requests)

    def set_target_position(
        self, instrument: Instrument, target: float, price: float | None = None, **options
    ) -> Order:
        return self._trading_manager.set_target_position(instrument, target, price, **options)

    def set_target_leverage(
        self, instrument: Instrument, leverage: float, price: float | None = None, **options
    ) -> None:
        return self._trading_manager.set_target_leverage(instrument, leverage, price, **options)

    def close_position(self, instrument: Instrument, without_signals: bool = False) -> None:
        return self._trading_manager.close_position(instrument, without_signals)

    def close_positions(self, market_type: MarketType | None = None, without_signals: bool = False) -> None:
        return self._trading_manager.close_positions(market_type, without_signals)

    def cancel_order(self, order_id: str, exchange: str | None = None) -> bool:
        """Cancel a specific order synchronously."""
        return self._trading_manager.cancel_order(order_id, exchange)

    def cancel_order_async(self, order_id: str, exchange: str | None = None) -> None:
        """Cancel a specific order asynchronously (non blocking)."""
        return self._trading_manager.cancel_order_async(order_id, exchange)

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument."""
        return self._trading_manager.cancel_orders(instrument)

    def update_order(self, order_id: str, price: float, amount: float, exchange: str | None = None) -> Order:
        """Update an existing limit order with new price and amount."""
        return self._trading_manager.update_order(order_id, price, amount, exchange)

    def get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        return self._trading_manager.get_min_size(instrument, amount)

    # IUniverseManager delegation
    def set_universe(
        self, instruments: list[Instrument], skip_callback: bool = False, if_has_position_then: RemovalPolicy = "close"
    ):
        return self._universe_manager.set_universe(instruments, skip_callback, if_has_position_then)

    def add_instruments(self, instruments: list[Instrument]):
        return self._universe_manager.add_instruments(instruments)

    def remove_instruments(self, instruments: list[Instrument], if_has_position_then: RemovalPolicy = "close"):
        return self._universe_manager.remove_instruments(instruments, if_has_position_then)

    @property
    def instruments(self):
        return self._universe_manager.instruments

    @property
    def exchanges(self) -> list[str]:
        return self._trading_manager.exchanges()

    # ISubscriptionManager delegation
    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None):
        return self._subscription_manager.subscribe(subscription_type, instruments)

    def unsubscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None):
        return self._subscription_manager.unsubscribe(subscription_type, instruments)

    def has_subscription(self, instrument: Instrument, subscription_type: str):
        return self._subscription_manager.has_subscription(instrument, subscription_type)

    def get_subscriptions(self, instrument: Instrument | None = None) -> list[str]:
        return self._subscription_manager.get_subscriptions(instrument)

    def get_base_subscription(self) -> str:
        return self._subscription_manager.get_base_subscription()

    def set_base_subscription(self, subscription_type: str):
        return self._subscription_manager.set_base_subscription(subscription_type)

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        return self._subscription_manager.get_subscribed_instruments(subscription_type)

    def get_warmup(self, subscription_type: str) -> str | None:
        return self._subscription_manager.get_warmup(subscription_type)

    def set_warmup(self, configs: dict[Any, str]):
        return self._subscription_manager.set_warmup(configs)

    def set_stale_data_detection(
        self, enabled: bool, detection_period: str | None = None, check_interval: str | None = None
    ) -> None:
        return self.initializer.set_stale_data_detection(enabled, detection_period, check_interval)

    def get_stale_data_detection_config(self) -> tuple[bool, str | None, str | None]:
        return self.initializer.get_stale_data_detection_config()

    def commit(self):
        return self._subscription_manager.commit()

    @property
    def auto_subscribe(self) -> bool:
        return self._subscription_manager.auto_subscribe

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool):
        self._subscription_manager.auto_subscribe = value

    # IProcessingManager delegation
    def process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool):
        return self._processing_manager.process_data(instrument, d_type, data, is_historical)

    def set_fit_schedule(self, schedule: str):
        return self._processing_manager.set_fit_schedule(schedule)

    def set_event_schedule(self, schedule: str):
        return self._processing_manager.set_event_schedule(schedule)

    def get_event_schedule(self, event_id: str) -> str | None:
        return self._processing_manager.get_event_schedule(event_id)

    def is_fitted(self) -> bool:
        return self._processing_manager.is_fitted()

    def get_active_targets(self) -> dict[Instrument, TargetPosition]:
        return self._processing_manager.get_active_targets()

    def emit_signal(self, signal: Signal | list[Signal]) -> None:
        return self._processing_manager.emit_signal(signal)

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> str:
        return self._processing_manager.schedule(cron_schedule, method)

    def unschedule(self, event_id: str) -> bool:
        return self._processing_manager.unschedule(event_id)

    # IWarmupStateSaver delegation
    def set_warmup_positions(self, positions: dict[Instrument, Position]) -> None:
        self._warmup_positions = positions

    def set_warmup_active_targets(self, active_targets: dict[Instrument, list[TargetPosition]]) -> None:
        self._warmup_active_targets = active_targets

    def set_warmup_orders(self, orders: dict[Instrument, list[Order]]) -> None:
        self._warmup_orders = orders

    def get_warmup_positions(self) -> dict[Instrument, Position]:
        return self._warmup_positions if self._warmup_positions is not None else {}

    def get_warmup_active_targets(self) -> dict[Instrument, list[TargetPosition]]:
        return self._warmup_active_targets if self._warmup_active_targets is not None else {}

    def get_warmup_orders(self) -> dict[Instrument, list[Order]]:
        return self._warmup_orders if self._warmup_orders is not None else {}

    def get_restored_state(self) -> RestoredState | None:
        return self._restored_state

    # ITransferManager delegation methods
    @check_transfer_manager
    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        assert self._transfer_manager is not None
        return self._transfer_manager.transfer_funds(from_exchange, to_exchange, currency, amount)

    @check_transfer_manager
    def get_transfer_status(self, transaction_id: str) -> dict[str, Any]:
        assert self._transfer_manager is not None
        return self._transfer_manager.get_transfer_status(transaction_id)

    @check_transfer_manager
    def get_transfers(self) -> dict[str, dict[str, Any]]:
        assert self._transfer_manager is not None
        return self._transfer_manager.get_transfers()

    # private methods
    def __process_incoming_data_loop(self, channel: CtrlChannel):
        logger.info("[StrategyContext] :: Start processing market data")
        while channel.control.is_set():
            try:
                # - waiting for incoming market data
                instrument, d_type, data, hist = channel.receive()

                # - notify error if error level is medium or higher
                if self._notifier and isinstance(data, BaseErrorEvent) and data.level.value >= ErrorLevel.MEDIUM.value:
                    self._notifier.notify_error(data.error or Exception("Unknown error"), {"message": str(data)})

                with self._health_monitor(d_type):
                    if self.process_data(instrument, d_type, data, hist):
                        channel.stop()
                        break

            except StrategyExceededMaxNumberOfRuntimeFailuresError:
                channel.stop()
                break
            except Exception as e:
                logger.error(f"Error processing market data: {e}")
                logger.opt(colors=False).error(traceback.format_exc())
                if self._notifier:
                    self._notifier.notify_error(e)
                # Don't stop the channel here, let it continue processing

        logger.info("[StrategyContext] :: Market data processing stopped")

    def __instantiate_strategy(self, strategy: IStrategy, config: dict[str, Any] | None) -> IStrategy:
        __strategy = strategy() if isinstance(strategy, type) else strategy
        __strategy.ctx = self
        set_parameters_to_object(__strategy, **config if config else {})
        return __strategy
