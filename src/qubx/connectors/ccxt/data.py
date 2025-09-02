import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# CCXT exceptions are now handled in ConnectionManager
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider, dt_64
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IDataArrivalListener, IDataProvider, IHealthMonitor
from qubx.core.series import Bar, Quote
from qubx.health import DummyHealthMonitor
from qubx.utils.misc import AsyncThreadLoop

from .connection_manager import ConnectionManager
from .exchange_manager import ExchangeManager
from .handlers import DataTypeHandlerFactory
from .handlers.ohlc import OhlcDataHandler
from .subscription_config import SubscriptionConfiguration
from .subscription_manager import SubscriptionManager
from .subscription_orchestrator import SubscriptionOrchestrator
from .warmup_service import WarmupService


class CcxtDataProvider(IDataProvider):
    time_provider: ITimeProvider
    _exchange_manager: ExchangeManager
    _scheduler: BasicScheduler | None = None

    # Core state - still needed
    _last_quotes: dict[Instrument, Optional[Quote]]
    _warmup_timeout: int

    def __init__(
        self,
        exchange_manager: ExchangeManager,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        max_ws_retries: int = 10,
        warmup_timeout: int = 120,
        health_monitor: IHealthMonitor | None = None,
    ):
        # Store the exchange manager (always ExchangeManager now)
        self._exchange_manager = exchange_manager
        
        self.time_provider = time_provider
        self.channel = channel
        self.max_ws_retries = max_ws_retries
        self._warmup_timeout = warmup_timeout
        self._health_monitor = health_monitor or DummyHealthMonitor()

        self._data_arrival_listeners: List[IDataArrivalListener] = [
            self._health_monitor,
            self._exchange_manager
        ]
        
        logger.debug(f"Registered {len(self._data_arrival_listeners)} data arrival listeners")

        # Core components - access exchange directly via exchange_manager.exchange
        self._exchange_id = str(self._exchange_manager.exchange.name)

        # Initialize composed components
        self._subscription_manager = SubscriptionManager()
        self._connection_manager = ConnectionManager(
            exchange_id=self._exchange_id,
            exchange_manager=self._exchange_manager,
            max_ws_retries=max_ws_retries,
            subscription_manager=self._subscription_manager,
        )
        self._subscription_orchestrator = SubscriptionOrchestrator(
            exchange_id=self._exchange_id,
            subscription_manager=self._subscription_manager,
            connection_manager=self._connection_manager,
            exchange_manager=self._exchange_manager,
        )

        # Data type handler factory for clean separation of data processing logic
        self._data_type_handler_factory = DataTypeHandlerFactory(
            data_provider=self,
            exchange_manager=self._exchange_manager,
            exchange_id=self._exchange_id,
        )

        # Warmup service for handling historical data warmup
        self._warmup_service = WarmupService(
            handler_factory=self._data_type_handler_factory,
            channel=channel,
            exchange_id=self._exchange_id,
            exchange_manager=self._exchange_manager,
            warmup_timeout=warmup_timeout,
        )

        # Quote caching for synthetic quote generation
        self._last_quotes = defaultdict(lambda: None)

        # Start ExchangeManager monitoring
        self._exchange_manager.start_monitoring()

        # Register recreation callback for automatic resubscription
        self._exchange_manager.register_recreation_callback(self._handle_exchange_recreation)

        logger.info(f"<yellow>{self._exchange_id}</yellow> Initialized")

    @property
    def _loop(self) -> AsyncThreadLoop:
        """Get current AsyncThreadLoop for the exchange."""
        return AsyncThreadLoop(self._exchange_manager.exchange.asyncio_loop)
    
    def notify_data_arrival(self, event_type: str, event_time: dt_64) -> None:
        """Notify all registered listeners about data arrival.
        
        Args:
            event_type: Type of data event (e.g., "ohlcv:BTC/USDT:1m")
            event_time: Timestamp of the data event
        """
        for listener in self._data_arrival_listeners:
            try:
                listener.on_data_arrival(event_type, event_time)
            except Exception as e:
                logger.error(f"Error notifying data arrival listener {type(listener).__name__}: {e}")





    @property
    def is_simulation(self) -> bool:
        return False

    def subscribe(
        self,
        subscription_type: str,
        instruments: List[Instrument],
        reset: bool = False,
    ) -> None:
        if not instruments:
            # In case of no instruments, do nothing, unsubscribe should handle this case
            return

        # Delegate to subscription manager for state management
        _updated_instruments = self._subscription_manager.add_subscription(subscription_type, instruments, reset)

        # Get handler from factory
        _sub_type, _params = DataType.from_str(subscription_type)
        handler = self._data_type_handler_factory.get_handler(_sub_type)
        if handler is None:
            raise ValueError(f"{self._exchange_id}: Subscription type {subscription_type} is not supported")

        # Delegate to orchestrator for complex subscription logic (clean facade pattern)
        self._subscription_orchestrator.execute_subscription(
            subscription_type=subscription_type,
            instruments=_updated_instruments,
            handler=handler,
            exchange=self._exchange_manager.exchange,
            channel=self.channel,
            **_params,
        )

    def unsubscribe(self, subscription_type: str, instruments: list[Instrument]) -> None:
        """Unsubscribe from instruments and handle partial/complete unsubscription."""
        # Get current instruments before removal (check both active and pending)
        current_instruments = set(self._subscription_manager.get_subscribed_instruments(subscription_type))

        # Early exit if no subscription exists
        if not current_instruments:
            logger.debug(f"No active subscription for {subscription_type}")
            return

        # Remove instruments from subscription manager
        self._subscription_manager.remove_subscription(subscription_type, instruments)

        # Get remaining instruments after removal
        remaining_instruments = set(self._subscription_manager.get_subscribed_instruments(subscription_type))

        if not remaining_instruments:
            # Complete unsubscription - no instruments left
            config = SubscriptionConfiguration(
                subscription_type=subscription_type,
                channel=self.channel,
                stream_name=f"cleanup_{subscription_type}",
            )
            self._subscription_orchestrator.execute_unsubscription(config)
        elif remaining_instruments != current_instruments:
            # Partial unsubscription - resubscribe with remaining instruments
            logger.info(
                f"Partial unsubscription for {subscription_type}, resubscribing with {len(remaining_instruments)} instruments"
            )
            _sub_type, _params = DataType.from_str(subscription_type)
            handler = self._data_type_handler_factory.get_handler(_sub_type)
            if handler:
                self._subscription_orchestrator.execute_subscription(
                    subscription_type=subscription_type,
                    instruments=remaining_instruments,
                    handler=handler,
                    exchange=self._exchange_manager.exchange,
                    channel=self.channel,
                    **_params,
                )

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        """Get list of active subscription types (delegated to subscription manager)."""
        return self._subscription_manager.get_subscriptions(instrument)

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        """Get list of subscribed instruments (delegated to subscription manager)."""
        return self._subscription_manager.get_subscribed_instruments(subscription_type)

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if instrument has active subscription (delegated to subscription manager)."""
        return self._subscription_manager.has_subscription(instrument, subscription_type)

    def has_pending_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if instrument has pending subscription (delegated to subscription manager)."""
        return self._subscription_manager.has_pending_subscription(instrument, subscription_type)

    def warmup(self, warmups: Dict[Tuple[str, Instrument], str]) -> None:
        """Execute warmup (delegated to warmup service)."""
        self._warmup_service.execute_warmup(warmups)

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> List[Bar]:
        """Get historical OHLC data (delegated to OhlcDataHandler)."""
        # Get OHLC handler from factory
        ohlc_handler = self._data_type_handler_factory.get_handler("ohlc")
        if ohlc_handler is None:
            raise ValueError(f"{self._exchange_id}: OHLC handler not available")

        # Cast to specific handler type to access get_historical_ohlc method
        if not isinstance(ohlc_handler, OhlcDataHandler):
            raise ValueError(f"{self._exchange_id}: Expected OhlcDataHandler, got {type(ohlc_handler)}")

        # Use async thread loop to call the async handler method
        async def _get_historical():
            return await ohlc_handler.get_historical_ohlc(instrument, timeframe, nbarsback)

        return self._loop.submit(_get_historical()).result(60)

    def close(self):
        """Properly close all connections and clean up resources."""
        try:
            # Stop all active subscriptions
            active_subscriptions = list(self._subscription_manager.get_subscriptions())
            for subscription_type in active_subscriptions:
                try:
                    # Create minimal config for cleanup
                    async def dummy_subscriber():
                        pass

                    config = SubscriptionConfiguration(
                        subscription_type=subscription_type,
                        channel=self.channel,
                        subscriber_func=dummy_subscriber,  # Dummy async func for cleanup
                        stream_name="cleanup",  # Dummy stream name for cleanup
                    )
                    self._subscription_orchestrator.execute_unsubscription(config)
                except Exception as e:
                    logger.error(f"Error stopping subscription {subscription_type}: {e}")

            # Stop ExchangeManager monitoring  
            self._exchange_manager.stop_monitoring()
            
            # Close exchange connection
            if hasattr(self._exchange_manager.exchange, "close"):
                future = self._loop.submit(self._exchange_manager.exchange.close())  # type: ignore
                # Wait for 5 seconds for connection to close
                future.result(5)
            else:
                del self._exchange_manager

            # Note: AsyncThreadLoop stop is handled by its own lifecycle

        except Exception as e:
            logger.error(f"Error during close: {e}")

    def _handle_exchange_recreation(self) -> None:
        """Handle exchange recreation by resubscribing to all active subscriptions."""
        logger.info(f"<yellow>{self._exchange_id}</yellow> Handling exchange recreation - resubscribing to active subscriptions")
        
        # Get snapshot of current subscriptions before cleanup
        active_subscriptions = self._subscription_manager.get_subscriptions()
        
        resubscription_data = []
        for subscription_type in active_subscriptions:
            instruments = self._subscription_manager.get_subscribed_instruments(subscription_type)
            if instruments:
                resubscription_data.append((subscription_type, instruments))
        
        logger.info(f"<yellow>{self._exchange_id}</yellow> Found {len(resubscription_data)} active subscriptions to recreate")
        
        # Track success/failure counts for reporting
        successful_resubscriptions = 0
        failed_resubscriptions = 0
        
        # Clean resubscription: unsubscribe then subscribe for each subscription type
        for subscription_type, instruments in resubscription_data:
            try:
                logger.info(f"<yellow>{self._exchange_id}</yellow> Resubscribing to {subscription_type} with {len(instruments)} instruments")
                
                self.unsubscribe(subscription_type, instruments)

                # Resubscribe with reset=True to ensure clean state
                self.subscribe(subscription_type, instruments, reset=True)
                
                successful_resubscriptions += 1
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Successfully resubscribed to {subscription_type}")
                
            except Exception as e:
                failed_resubscriptions += 1
                logger.error(f"<yellow>{self._exchange_id}</yellow> Failed to resubscribe to {subscription_type}: {e}")
                # Continue with other subscriptions even if one fails
        
        # Report final status
        total_subscriptions = len(resubscription_data)
        if failed_resubscriptions == 0:
            logger.info(f"<yellow>{self._exchange_id}</yellow> Exchange recreation resubscription completed successfully ({total_subscriptions}/{total_subscriptions})")
        else:
            logger.warning(f"<yellow>{self._exchange_id}</yellow> Exchange recreation resubscription completed with errors ({successful_resubscriptions}/{total_subscriptions} successful)")

    @property
    def subscribed_instruments(self) -> Set[Instrument]:
        """Get all subscribed instruments (delegated to subscription manager)."""
        return set(self._subscription_manager.get_all_subscribed_instruments())

    @property
    def is_read_only(self) -> bool:
        _key = self._exchange_manager.exchange.apiKey
        return _key is None or _key == ""

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int = 1) -> int:
        return (self.time_provider.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _get_exch_timeframe(self, timeframe: str) -> str:
        if timeframe is not None:
            _t = re.match(r"(\d+)(\w+)", timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange_manager.exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self._exchange_manager.exchange.name}")

        return tframe

    def _get_exch_symbol(self, instrument: Instrument) -> str:
        return f"{instrument.base}/{instrument.quote}:{instrument.settle}"

    def exchange(self) -> str:
        return self._exchange_id.upper()
