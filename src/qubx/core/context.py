import traceback
from threading import Thread
from typing import Any, Callable

from qubx import logger
from qubx.core.basics import (
    SW,
    AssetBalance,
    CtrlChannel,
    DataType,
    Instrument,
    ITimeProvider,
    MarketType,
    Order,
    OrderRequest,
    Position,
    dt_64,
)
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
    IMarketManager,
    IPositionGathering,
    IProcessingManager,
    IStrategy,
    IStrategyContext,
    ISubscriptionManager,
    ITradeDataExport,
    ITradingManager,
    IUniverseManager,
    PositionsTracker,
    RemovalPolicy,
)
from qubx.core.loggers import StrategyLogging
from qubx.data.readers import DataReader
from qubx.gathering.simplest import SimplePositionGatherer
from qubx.trackers.sizers import FixedSizer

from .mixins import (
    MarketManager,
    ProcessingManager,
    SubscriptionManager,
    TradingManager,
    UniverseManager,
)


class StrategyContext(IStrategyContext):
    DEFAULT_POSITION_TRACKER: Callable[[], PositionsTracker] = lambda: PositionsTracker(
        FixedSizer(1.0, amount_in_quote=False)
    )

    _market_data_provider: IMarketManager
    _universe_manager: IUniverseManager
    _subscription_manager: ISubscriptionManager
    _trading_manager: ITradingManager
    _processing_manager: IProcessingManager

    _broker: IBroker  # service for exchange API: orders managemewnt
    _logging: StrategyLogging  # recording all activities for the strat: execs, positions, portfolio
    _data_provider: IDataProvider  # market data provider
    _cache: CachedMarketDataHolder
    _scheduler: BasicScheduler
    _initial_instruments: list[Instrument]

    _thread_data_loop: Thread | None = None  # market data loop
    _is_initialized: bool = False
    _exporter: ITradeDataExport | None = None  # Add exporter attribute

    def __init__(
        self,
        strategy: IStrategy,
        broker: IBroker,
        data_provider: IDataProvider,
        account: IAccountProcessor,
        scheduler: BasicScheduler,
        time_provider: ITimeProvider,
        instruments: list[Instrument],
        logging: StrategyLogging,
        config: dict[str, Any] | None = None,
        position_gathering: IPositionGathering | None = None,  # TODO: make position gathering part of the strategy
        aux_data_provider: DataReader | None = None,
        exporter: ITradeDataExport | None = None,
        initializer: BasicStrategyInitializer | None = None,
    ) -> None:
        self.account = account
        self.strategy = self.__instantiate_strategy(strategy, config)
        self.initializer = initializer if initializer is not None else BasicStrategyInitializer()

        self._time_provider = time_provider
        self._broker = broker
        self._data_provider = data_provider
        self._logging = logging
        self._scheduler = scheduler
        self._initial_instruments = instruments

        self._cache = CachedMarketDataHolder()
        self._exporter = exporter  # Store the exporter

        __position_tracker = self.strategy.tracker(self)
        if __position_tracker is None:
            __position_tracker = StrategyContext.DEFAULT_POSITION_TRACKER()

        __position_gathering = position_gathering if position_gathering is not None else SimplePositionGatherer()

        self._subscription_manager = SubscriptionManager(
            data_provider=self._data_provider,
            default_base_subscription=DataType.ORDERBOOK if not self._data_provider.is_simulation else DataType.NONE,
        )
        self.account.set_subscription_manager(self._subscription_manager)

        self._market_data_provider = MarketManager(
            time_provider=self._time_provider,
            cache=self._cache,
            data_provider=self._data_provider,
            universe_manager=self,
            aux_data_provider=aux_data_provider,
        )
        self._universe_manager = UniverseManager(
            context=self,
            strategy=self.strategy,
            broker=self._data_provider,
            trading_service=self._broker,
            cache=self._cache,
            logging=self._logging,
            subscription_manager=self,
            trading_manager=self,
            time_provider=self,
            account=self.account,
            position_gathering=__position_gathering,
        )
        self._trading_manager = TradingManager(
            time_provider=self,
            broker=self._broker,
            account=self.account,
            strategy_name=self.strategy.__class__.__name__,
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
            is_simulation=self._data_provider.is_simulation,
            exporter=self._exporter,  # Pass exporter to processing manager
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        self.strategy.on_init(self.initializer)

        if base_sub := self.initializer.get_base_subscription():
            self.set_base_subscription(base_sub)

        if auto_sub := self.initializer.get_auto_subscribe():
            self.auto_subscribe = auto_sub

        if fit_schedule := self.initializer.get_fit_schedule():
            self.set_fit_schedule(fit_schedule)

        if event_schedule := self.initializer.get_event_schedule():
            self.set_event_schedule(event_schedule)

        # - update cache default timeframe
        sub_type = self.get_base_subscription()
        _, params = DataType.from_str(sub_type)
        __default_timeframe = params.get("timeframe", "1sec")
        self._cache.update_default_timeframe(__default_timeframe)

    def start(self, blocking: bool = False):
        if self._is_initialized:
            raise ValueError("Strategy is already started !")

        # - run cron scheduler
        self._scheduler.run()

        # - create incoming market data processing
        databus = self._data_provider.channel
        databus.register(self)

        # - start account processing
        self.account.start()

        # - update universe with initial instruments after the strategy is initialized
        self.set_universe(self._initial_instruments, skip_callback=True)

        # - initialize strategy (should we do that after any first market data received ?)
        if not self._is_initialized:
            try:
                self.strategy.on_start(self)
                self._is_initialized = True
            except Exception as strat_error:
                logger.error(
                    f"[StrategyContext] :: Strategy {self.strategy.__class__.__name__} raised an exception in on_start: {strat_error}"
                )
                logger.error(traceback.format_exc())
                return

        # - for live we run loop
        if not self._data_provider.is_simulation:
            self._thread_data_loop = Thread(target=self.__process_incoming_data_loop, args=(databus,), daemon=True)
            self._thread_data_loop.start()
            logger.info("[StrategyContext] :: strategy is started in thread")
            if blocking:
                self._thread_data_loop.join()

    def stop(self):
        # - invoke strategy's stop code
        try:
            self.strategy.on_stop(self)
        except Exception as strat_error:
            logger.error(
                f"[<y>StrategyContext</y>] :: Strategy {self.strategy.__class__.__name__} raised an exception in on_stop: {strat_error}"
            )
            logger.opt(colors=False).error(traceback.format_exc())

        if self._thread_data_loop:
            self._data_provider.close()
            self._data_provider.channel.stop()
            self._thread_data_loop.join()
            self._thread_data_loop = None

        # - stop account processing
        self.account.stop()

        # - close logging
        self._logging.close()

    def is_running(self):
        return self._thread_data_loop is not None and self._thread_data_loop.is_alive()

    @property
    def is_simulation(self) -> bool:
        return self._data_provider.is_simulation

    # IAccountViewer delegation

    # capital information
    def get_capital(self) -> float:
        return self.account.get_capital()

    def get_total_capital(self) -> float:
        return self.account.get_total_capital()

    # balance and position information
    def get_balances(self) -> dict[str, AssetBalance]:
        return dict(self.account.get_balances())

    def get_positions(self) -> dict[Instrument, Position]:
        return dict(self.account.get_positions())

    def get_position(self, instrument: Instrument) -> Position:
        return self.account.get_position(instrument)

    @property
    def positions(self):
        return self.account.positions

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        return self.account.get_orders(instrument)

    def position_report(self) -> dict:
        return self.account.position_report()

    # leverage information
    def get_leverage(self, instrument: Instrument) -> float:
        return self.account.get_leverage(instrument)

    def get_leverages(self) -> dict[Instrument, float]:
        return self.account.get_leverages()

    def get_net_leverage(self) -> float:
        return self.account.get_net_leverage()

    def get_gross_leverage(self) -> float:
        return self.account.get_gross_leverage()

    # margin information
    def get_total_required_margin(self) -> float:
        return self.account.get_total_required_margin()

    def get_available_margin(self) -> float:
        return self.account.get_available_margin()

    def get_margin_ratio(self) -> float:
        return self.account.get_margin_ratio()

    # IMarketDataProvider delegation
    def time(self) -> dt_64:
        return self._market_data_provider.time()

    def ohlc(self, instrument: Instrument, timeframe: str | None = None, length: int | None = None):
        return self._market_data_provider.ohlc(instrument, timeframe, length)

    def quote(self, instrument: Instrument):
        return self._market_data_provider.quote(instrument)

    def get_data(self, instrument: Instrument, sub_type: str) -> list[Any]:
        return self._market_data_provider.get_data(instrument, sub_type)

    def get_aux_data(self, data_id: str, **parameters):
        return self._market_data_provider.get_aux_data(data_id, **parameters)

    def get_instruments(self):
        return self._market_data_provider.get_instruments()

    def query_instrument(self, symbol: str, exchange: str | None = None) -> Instrument | None:
        return self._market_data_provider.query_instrument(
            symbol, exchange if exchange is not None else self.exchanges[0]
        )

    # ITradingManager delegation
    def trade(self, instrument: Instrument, amount: float, price: float | None = None, time_in_force="gtc", **options):
        return self._trading_manager.trade(instrument, amount, price, time_in_force, **options)

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[Order]:
        return self._trading_manager.submit_orders(order_requests)

    def set_target_position(
        self, instrument: Instrument, target: float, price: float | None = None, **options
    ) -> Order:
        return self._trading_manager.set_target_position(instrument, target, price, **options)

    def close_position(self, instrument: Instrument) -> None:
        return self._trading_manager.close_position(instrument)

    def close_positions(self, market_type: MarketType | None = None) -> None:
        return self._trading_manager.close_positions(market_type)

    def cancel_order(self, order_id: str) -> None:
        return self._trading_manager.cancel_order(order_id)

    def cancel_orders(self, instrument: Instrument):
        return self._trading_manager.cancel_orders(instrument)

    # IUniverseManager delegation
    def set_universe(
        self, instruments: list[Instrument], skip_callback: bool = False, if_has_position_then: RemovalPolicy = "close"
    ):
        return self._universe_manager.set_universe(instruments, skip_callback, if_has_position_then)

    def add_instruments(self, instruments: list[Instrument]):
        return self._universe_manager.add_instruments(instruments)

    def remove_instruments(self, instruments: list[Instrument]):
        return self._universe_manager.remove_instruments(instruments)

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

    # private methods
    def __process_incoming_data_loop(self, channel: CtrlChannel):
        logger.info("[StrategyContext] :: Start processing market data")
        while channel.control.is_set():
            with SW("StrategyContext._process_incoming_data"):
                # - waiting for incoming market data
                instrument, d_type, data, hist = channel.receive()
                if self.process_data(instrument, d_type, data, hist):
                    channel.stop()
                    break
        logger.info("[StrategyContext] :: Market data processing stopped")

    def __instantiate_strategy(self, strategy: IStrategy, config: dict[str, Any] | None) -> IStrategy:
        __strategy = strategy() if isinstance(strategy, type) else strategy
        __strategy.ctx = self
        set_parameters_to_object(__strategy, **config if config else {})
        return __strategy
