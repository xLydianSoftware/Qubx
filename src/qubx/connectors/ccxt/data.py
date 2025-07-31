import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

import ccxt.pro as cxp

# CCXT exceptions are now handled in ConnectionManager
from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IDataProvider, IHealthMonitor
from qubx.core.series import Bar, Quote
from qubx.health import DummyHealthMonitor
from qubx.utils.misc import AsyncThreadLoop

from .connection_manager import ConnectionManager
from .handlers import DataTypeHandlerFactory
from .subscription_manager import SubscriptionManager
from .subscription_orchestrator import SubscriptionOrchestrator
from .utils import instrument_to_ccxt_symbol
from .warmup_service import WarmupService


class CcxtDataProvider(IDataProvider):
    time_provider: ITimeProvider
    _exchange: Exchange
    _scheduler: BasicScheduler | None = None

    # Core state - still needed
    _last_quotes: Dict[Instrument, Optional[Quote]]
    _loop: AsyncThreadLoop
    _warmup_timeout: int

    def __init__(
        self,
        exchange: cxp.Exchange,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        max_ws_retries: int = 10,
        warmup_timeout: int = 120,
        health_monitor: IHealthMonitor | None = None,
    ):
        self._exchange_id = str(exchange.name)
        self.time_provider = time_provider
        self.channel = channel
        self.max_ws_retries = max_ws_retries
        self._warmup_timeout = warmup_timeout
        self._health_monitor = health_monitor or DummyHealthMonitor()

        # Core components
        self._exchange = exchange
        self._loop = AsyncThreadLoop(self._exchange.asyncio_loop)

        # Initialize composed components
        self._subscription_manager = SubscriptionManager()
        self._connection_manager = ConnectionManager(
            exchange_id=self._exchange_id,
            max_ws_retries=max_ws_retries,
            subscription_manager=self._subscription_manager,
        )
        self._subscription_orchestrator = SubscriptionOrchestrator(
            exchange_id=self._exchange_id,
            subscription_manager=self._subscription_manager,
            connection_manager=self._connection_manager,
        )

        # Data type handler factory for clean separation of data processing logic
        self._data_type_handler_factory = DataTypeHandlerFactory(
            data_provider=self,
            exchange=self._exchange,
            exchange_id=self._exchange_id,
        )

        # Warmup service for handling historical data warmup
        self._warmup_service = WarmupService(
            handler_factory=self._data_type_handler_factory,
            channel=channel,
            exchange_id=self._exchange_id,
            async_loop=self._loop,
            warmup_timeout=warmup_timeout,
        )

        # Quote caching for synthetic quote generation
        self._last_quotes = defaultdict(lambda: None)

        logger.info(f"<yellow>{self._exchange_id}</yellow> Initialized")

    @property
    def is_simulation(self) -> bool:
        return False

    def subscribe(
        self,
        subscription_type: str,
        instruments: List[Instrument],
        reset: bool = False,
    ) -> None:
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
            stream_name_generator=self._get_subscription_name,
            async_loop_submit=self._loop.submit,
            exchange=self._exchange,
            channel=self.channel,
            **_params,
        )

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument]) -> None:
        """Unsubscribe from instruments and stop stream if no instruments remain."""
        # Check if subscription exists before removal
        had_subscription = subscription_type in self._subscription_manager._subscriptions
        
        # Remove instruments from subscription manager
        self._subscription_manager.remove_subscription(subscription_type, instruments)
        
        # If subscription was completely removed (no instruments left), stop the stream
        subscription_removed = (
            had_subscription and 
            subscription_type not in self._subscription_manager._subscriptions
        )
        
        if subscription_removed:
            # Use async loop to call the async stop_subscription method
            async def _stop_subscription():
                await self._subscription_orchestrator.stop_subscription(subscription_type)
            
            # Submit the async operation to the event loop
            try:
                self._loop.submit(_stop_subscription()).result(timeout=5)
                logger.info(f"<yellow>{self._exchange_id}</yellow> Stopped listening to {subscription_type}")
            except Exception as e:
                logger.error(f"<yellow>{self._exchange_id}</yellow> Failed to stop stream for {subscription_type}: {e}")

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
        from .handlers.ohlc import OhlcDataHandler

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
        try:
            if hasattr(self._exchange, "close"):
                future = self._loop.submit(self._exchange.close())  # type: ignore
                # - wait for 5 seconds for connection to close
                future.result(5)
            else:
                del self._exchange
        except Exception as e:
            logger.error(e)

    @property
    def subscribed_instruments(self) -> Set[Instrument]:
        """Get all subscribed instruments (delegated to subscription manager)."""
        return set(self._subscription_manager.get_all_subscribed_instruments())

    @property
    def is_read_only(self) -> bool:
        _key = self._exchange.apiKey
        return _key is None or _key == ""

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int = 1) -> int:
        return (self.time_provider.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _get_exch_timeframe(self, timeframe: str) -> str:
        if timeframe is not None:
            _t = re.match(r"(\d+)(\w+)", timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self._exchange.name}")

        return tframe

    def _get_exch_symbol(self, instrument: Instrument) -> str:
        return f"{instrument.base}/{instrument.quote}:{instrument.settle}"

    def _get_subscription_name(
        self, subscription: str, instruments: List[Instrument] | Set[Instrument] | Instrument | None = None, **kwargs
    ) -> str:
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        _symbols = [instrument_to_ccxt_symbol(i) for i in instruments] if instruments is not None else []
        _name = f"{','.join(_symbols)} {subscription}" if _symbols else subscription
        if kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            _name += f" ({kwargs_str})"
        return _name

    def exchange(self) -> str:
        return self._exchange_id.upper()
