"""
LighterDataProvider - IDataProvider implementation for Lighter exchange.

Provides market data subscriptions via WebSocket with support for:
- Orderbook (with stateful updates via OrderBookMaintainer)
- Trades (including liquidations)
- Quotes (derived from orderbook)
- Market stats (funding rate, open interest, volume)
"""

import asyncio
from collections import defaultdict
from typing import Optional

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider, dt_64
from qubx.core.interfaces import IDataProvider
from qubx.core.series import Quote

from .client import LighterClient
from .handlers import OrderbookHandler, QuoteHandler, TradesHandler
from .instruments import LighterInstrumentLoader
from .websocket import LighterWebSocketManager


class LighterDataProvider(IDataProvider):
    """
    Data provider for Lighter exchange via WebSocket.

    Manages subscriptions and routes WebSocket messages to appropriate handlers.
    Each market gets its own set of handlers with independent state.

    Architecture:
        WebSocket → Router → Handler → CtrlChannel → Strategy
    """

    def __init__(
        self,
        client: LighterClient,
        instrument_loader: LighterInstrumentLoader,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        ws_url: str = "wss://mainnet.zklighter.elliot.ai/stream",
    ):
        """
        Initialize Lighter data provider.

        Args:
            client: LighterClient for REST API access
            instrument_loader: Loaded instruments with market_id mappings
            time_provider: Time provider for timestamps
            channel: Control channel for sending data
            ws_url: WebSocket URL (default: mainnet)
        """
        self.client = client
        self.instrument_loader = instrument_loader
        self.time_provider = time_provider
        self.channel = channel
        self.ws_url = ws_url

        # Subscription tracking
        self._subscriptions: dict[str, set[Instrument]] = defaultdict(set)
        self._handlers: dict[tuple[str, int], any] = {}  # (sub_type, market_id) -> handler

        # Quote caching (for synthetic quotes from orderbook)
        self._last_quotes: dict[Instrument, Optional[Quote]] = defaultdict(lambda: None)

        # WebSocket manager
        self._ws_manager: Optional[LighterWebSocketManager] = None
        self._ws_connected = False

        logger.info("LighterDataProvider initialized")

    @property
    def is_simulation(self) -> bool:
        """Check if provider is in simulation mode (always False for live)"""
        return False

    def subscribe(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        reset: bool = False,
    ) -> None:
        """
        Subscribe to market data for instruments.

        Args:
            subscription_type: Type of subscription (e.g., "orderbook", "trade", "quote")
            instruments: Set of instruments to subscribe to
            reset: If True, replace existing subscriptions; if False, add to existing

        Supported types:
            - "orderbook" - L2 orderbook with stateful updates
            - "trade" - Trade feed (including liquidations)
            - "quote" - Best bid/ask (derived from orderbook if not subscribed)

        Example:
            provider.subscribe("orderbook", {btc_instrument, eth_instrument})
            provider.subscribe("trade", {btc_instrument})
        """
        if not instruments:
            logger.debug(f"No instruments provided for subscription type: {subscription_type}")
            return

        # Parse subscription type
        sub_type, params = DataType.from_str(subscription_type)

        # Validate subscription type
        if sub_type not in ["orderbook", "trade", "quote"]:
            raise ValueError(f"Unsupported subscription type: {sub_type}")

        # Ensure WebSocket is connected
        if not self._ws_connected:
            self._connect_websocket()

        # Update subscription tracking
        if reset:
            self._subscriptions[subscription_type] = set(instruments)
        else:
            self._subscriptions[subscription_type].update(instruments)

        # Subscribe to each instrument
        for instrument in instruments:
            self._subscribe_instrument(sub_type, instrument, **params)

        logger.info(
            f"Subscribed to {subscription_type} for {len(instruments)} instruments: "
            f"{[i.symbol for i in instruments]}"
        )

    def _connect_websocket(self) -> None:
        """Initialize and connect WebSocket manager"""
        if self._ws_manager is None:
            self._ws_manager = LighterWebSocketManager(url=self.ws_url)

        # Connect if not already connected
        if not self._ws_manager.is_connected:
            asyncio.create_task(self._ws_manager.connect())
            # Wait briefly for connection (WebSocket manager handles async)
            self._ws_connected = True
            logger.info("WebSocket connection initiated")

    def _subscribe_instrument(self, sub_type: str, instrument: Instrument, **params) -> None:
        """
        Subscribe single instrument to a data type.

        Creates handler if needed and subscribes via WebSocket.

        Args:
            sub_type: Data type (orderbook, trade, quote)
            instrument: Instrument to subscribe
            **params: Additional parameters (e.g., depth for orderbook)
        """
        # Get market_id for this instrument
        market_id = self.instrument_loader.get_market_id(instrument.symbol)
        if market_id is None:
            raise ValueError(f"Market ID not found for {instrument.symbol}")

        handler_key = (sub_type, market_id)

        # Create handler if doesn't exist
        if handler_key not in self._handlers:
            handler = self._create_handler(sub_type, instrument, market_id, **params)
            self._handlers[handler_key] = handler

        # Subscribe via WebSocket
        if sub_type == "orderbook":
            asyncio.create_task(
                self._ws_manager.subscribe_orderbook(market_id, self._make_orderbook_callback(instrument, market_id))
            )
        elif sub_type == "trade":
            asyncio.create_task(
                self._ws_manager.subscribe_trades(market_id, self._make_trade_callback(instrument, market_id))
            )
        elif sub_type == "quote":
            # Quote is derived from orderbook, so subscribe to orderbook
            asyncio.create_task(
                self._ws_manager.subscribe_orderbook(market_id, self._make_quote_callback(instrument, market_id))
            )

    def _create_handler(self, sub_type: str, instrument: Instrument, market_id: int, **params):
        """
        Create appropriate handler for subscription type.

        Args:
            sub_type: Data type
            instrument: Instrument
            market_id: Lighter market ID
            **params: Handler-specific parameters

        Returns:
            Handler instance
        """
        if sub_type == "orderbook":
            depth = params.get("depth", 200)
            return OrderbookHandler(market_id=market_id, tick_size=instrument.tick_size, max_levels=depth)
        elif sub_type == "trade":
            return TradesHandler(market_id=market_id)
        elif sub_type == "quote":
            return QuoteHandler(market_id=market_id)
        else:
            raise ValueError(f"Unknown subscription type: {sub_type}")

    def _make_orderbook_callback(self, instrument: Instrument, market_id: int):
        """Create callback for orderbook messages"""

        async def callback(message: dict):
            handler_key = ("orderbook", market_id)
            handler = self._handlers.get(handler_key)

            if handler and handler.can_handle(message):
                orderbook = handler.handle(message)
                if orderbook:
                    # Send to channel
                    self.channel.send((instrument, "orderbook", orderbook, False))

                    # Generate synthetic quote if no quote subscription
                    if not self.has_subscription(instrument, "quote"):
                        quote = orderbook.to_quote()
                        self._last_quotes[instrument] = quote

        return callback

    def _make_trade_callback(self, instrument: Instrument, market_id: int):
        """Create callback for trade messages"""

        async def callback(message: dict):
            handler_key = ("trade", market_id)
            handler = self._handlers.get(handler_key)

            if handler and handler.can_handle(message):
                trades = handler.handle(message)
                if trades:
                    for trade in trades:
                        self.channel.send((instrument, "trade", trade, False))

        return callback

    def _make_quote_callback(self, instrument: Instrument, market_id: int):
        """Create callback for quote (from orderbook)"""

        async def callback(message: dict):
            handler_key = ("quote", market_id)
            handler = self._handlers.get(handler_key)

            if handler and handler.can_handle(message):
                quote = handler.handle(message)
                if quote:
                    self._last_quotes[instrument] = quote
                    self.channel.send((instrument, "quote", quote, False))

        return callback

    def unsubscribe(self, subscription_type: str | None, instruments: set[Instrument]) -> None:
        """
        Unsubscribe from market data.

        Args:
            subscription_type: Type of subscription to unsubscribe from (or None for all)
            instruments: Set of instruments to unsubscribe
        """
        if subscription_type is None:
            # Unsubscribe from all types
            for sub_type in list(self._subscriptions.keys()):
                self.unsubscribe(sub_type, instruments)
            return

        # Remove from tracking
        if subscription_type in self._subscriptions:
            self._subscriptions[subscription_type] -= instruments

            # If no instruments left, remove subscription type
            if not self._subscriptions[subscription_type]:
                del self._subscriptions[subscription_type]

        # Unsubscribe from WebSocket
        sub_type, _ = DataType.from_str(subscription_type)
        for instrument in instruments:
            market_id = self.instrument_loader.get_market_id(instrument.symbol)
            if market_id is not None:
                # Remove handler
                handler_key = (sub_type, market_id)
                if handler_key in self._handlers:
                    del self._handlers[handler_key]

                # Unsubscribe from WebSocket (if connected)
                if self._ws_manager:
                    if sub_type == "orderbook":
                        asyncio.create_task(self._ws_manager.unsubscribe_orderbook(market_id))
                    elif sub_type == "trade":
                        asyncio.create_task(self._ws_manager.unsubscribe_trades(market_id))

        logger.info(f"Unsubscribed from {subscription_type} for {len(instruments)} instruments")

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """
        Check if instrument has a subscription.

        Args:
            instrument: Instrument to check
            subscription_type: Subscription type

        Returns:
            True if subscribed
        """
        return instrument in self._subscriptions.get(subscription_type, set())

    def get_subscriptions(self, instrument: Instrument | None = None) -> list[str]:
        """
        Get all subscription types.

        Args:
            instrument: Filter by instrument (or None for all)

        Returns:
            List of subscription type strings
        """
        if instrument is None:
            return list(self._subscriptions.keys())

        # Return subscriptions for specific instrument
        return [sub_type for sub_type, instrs in self._subscriptions.items() if instrument in instrs]

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        """
        Get subscribed instruments.

        Args:
            subscription_type: Filter by subscription type (or None for all)

        Returns:
            List of subscribed instruments
        """
        if subscription_type is None:
            # Return all subscribed instruments across all types
            all_instruments = set()
            for instruments in self._subscriptions.values():
                all_instruments.update(instruments)
            return list(all_instruments)

        return list(self._subscriptions.get(subscription_type, set()))

    def warmup(self, configs: dict[tuple[str, Instrument], str]) -> None:
        """
        Run warmup for subscriptions using historical data.

        Args:
            configs: Dict mapping (subscription_type, instrument) to warmup period

        Example:
            warmup({
                ("trade", btc_instrument): "1h",
                ("orderbook", eth_instrument): "10min",
            })

        Note: Lighter's REST API has limited historical data support.
        Warmup primarily useful for trades (recent history).
        """
        if not configs:
            logger.debug("No warmup configs provided")
            return

        logger.info(f"Starting warmup for {len(configs)} configurations")

        for (sub_type, instrument), period in configs.items():
            try:
                self._warmup_instrument(sub_type, instrument, period)
            except Exception as e:
                logger.error(f"Warmup failed for {sub_type}/{instrument.symbol}: {e}")

        logger.info("Warmup complete")

    def _warmup_instrument(self, sub_type: str, instrument: Instrument, period: str) -> None:
        """
        Warmup single instrument.

        Args:
            sub_type: Subscription type
            instrument: Instrument to warmup
            period: Warmup period (e.g., "1h", "30min")
        """
        # Parse subscription type
        data_type, params = DataType.from_str(sub_type)

        if data_type == "trade":
            # Fetch recent trades from REST API
            market_id = self.instrument_loader.get_market_id(instrument.symbol)
            if market_id is None:
                logger.warning(f"Market ID not found for {instrument.symbol}, skipping warmup")
                return

            # TODO: Implement trade history fetching via REST API
            logger.debug(f"Trade warmup for {instrument.symbol} (period: {period})")
            # trades = self.client.get_trades(market_id, limit=100)
            # for trade in trades:
            #     self.channel.send((instrument, "trade", trade, True))  # is_warmup=True

        elif data_type == "orderbook":
            # Orderbook is realtime state, no historical warmup needed
            logger.debug(f"Orderbook warmup skipped for {instrument.symbol} (realtime data only)")

        else:
            logger.warning(f"Warmup not supported for {data_type}")

    async def close(self) -> None:
        """Close WebSocket connections and cleanup"""
        if self._ws_manager:
            await self._ws_manager.disconnect()
            logger.info("LighterDataProvider closed")
