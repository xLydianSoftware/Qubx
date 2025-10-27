"""
LighterDataProvider - IDataProvider implementation for Lighter exchange.

Provides market data subscriptions via WebSocket with support for:
- Orderbook (with stateful updates via LOB - Cython implementation)
- Trades (including liquidations)
- Quotes (derived from orderbook)
- Market stats (funding rate, open interest, volume)
"""

import asyncio
from collections import defaultdict
from typing import Any, Optional, cast

import pandas as pd

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, FundingPayment, FundingRate, Instrument, ITimeProvider, OpenInterest
from qubx.core.interfaces import IDataProvider
from qubx.core.series import Bar, OrderBook, Quote, Trade, time_as_nsec
from qubx.utils.misc import AsyncThreadLoop

from .client import LighterClient
from .handlers import MarketStatsHandler, OrderbookHandler, QuoteHandler, TradesHandler
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
        loop: asyncio.AbstractEventLoop,
        ws_manager: LighterWebSocketManager,
        ws_url: str = "wss://mainnet.zklighter.elliot.ai/stream",
    ):
        """
        Initialize Lighter data provider.

        Args:
            client: LighterClient for REST API access
            instrument_loader: Loaded instruments with market_id mappings
            time_provider: Time provider for timestamps
            channel: Control channel for sending data
            loop: Event loop for async operations (from client)
            ws_manager: WebSocket manager (shared across components)
            ws_url: WebSocket URL (default: mainnet)
        """
        self.client = client
        self.instrument_loader = instrument_loader
        self.time_provider = time_provider
        self.channel = channel
        self.ws_url = ws_url

        # Async thread loop for submitting tasks to client's event loop
        self._async_loop = AsyncThreadLoop(loop)

        # Subscription tracking
        self._subscriptions: dict[str, set[Instrument]] = defaultdict(set)
        self._handlers: dict[tuple[str, int], Any] = {}  # (sub_type, market_id) -> handler

        # Quote caching (for synthetic quotes from orderbook)
        self._last_quotes: dict[Instrument, Optional[Quote]] = defaultdict(lambda: None)

        # WebSocket manager (shared across all components)
        self._ws_manager = ws_manager
        self._ws_connected = False

        # Track if market_stats:all is subscribed (single subscription for all instruments)
        self._market_stats_subscribed: bool = False

        # Track if reconnection callback has been registered
        self._reconnection_callback_registered: bool = False

        logger.info("LighterDataProvider initialized")

    @property
    def is_simulation(self) -> bool:
        """Check if provider is in simulation mode (always False for live)"""
        return False

    @property
    def ws_manager(self) -> LighterWebSocketManager:
        """Get WebSocket manager (shared across components)"""
        return self._ws_manager

    def exchange(self) -> str:
        """Return exchange name"""
        return "LIGHTER"

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
        if sub_type not in [
            "orderbook",
            "trade",
            "quote",
            DataType.FUNDING_RATE,
            DataType.FUNDING_PAYMENT,
            DataType.OPEN_INTEREST,
        ]:
            raise ValueError(f"Unsupported subscription type: {sub_type}")

        # Ensure WebSocket is connected (wait if needed)
        try:
            self._ensure_websocket_connected(timeout=5.0)
        except (TimeoutError, ConnectionError) as e:
            logger.error(f"Cannot subscribe: {e}")
            raise

        # Update subscription tracking
        if reset:
            self._subscriptions[subscription_type] = set(instruments)
        else:
            self._subscriptions[subscription_type].update(instruments)

        # Subscribe to each instrument (connection guaranteed at this point)
        for instrument in instruments:
            self._subscribe_instrument(sub_type, instrument, **params)

        logger.info(
            f"Subscribed to {subscription_type} for {len(instruments)} instruments: {[i.symbol for i in instruments]}"
        )

    async def _on_reconnected(self) -> None:
        """
        Callback invoked after WebSocket reconnection.

        Resets all handler states to ensure clean state after reconnection.
        This is particularly important for stateful handlers like OrderbookHandler
        which maintain incremental state that becomes invalid after disconnection.
        """
        logger.info("WebSocket reconnected, resetting all handler states")

        # Reset all handlers (stateless handlers have empty reset() implementation)
        for handler in self._handlers.values():
            try:
                handler.reset()
            except Exception as e:
                logger.error(f"Error resetting handler {handler.__class__.__name__}: {e}")

        logger.debug(f"Reset {len(self._handlers)} handlers after reconnection")

    def _ensure_websocket_connected(self, timeout: float = 5.0) -> None:
        """
        Ensure WebSocket is connected, wait if necessary.

        Args:
            timeout: Maximum time to wait for connection (seconds)

        Raises:
            TimeoutError: If connection not established within timeout
            ConnectionError: If connection fails
        """
        # Already connected
        if self._ws_manager.is_connected:
            return

        # Connection in progress, wait for it
        if self._ws_connected:
            import time

            max_wait = int(timeout * 10)  # Check every 0.1s
            for _ in range(max_wait):
                if self._ws_manager.is_connected:
                    return
                time.sleep(0.1)
            raise TimeoutError(f"WebSocket connection not ready after {timeout}s")

        # Need to initiate connection
        async def _connect():
            assert self._ws_manager is not None
            await self._ws_manager.connect()
            self._ws_connected = True

            # Register reconnection callback (one-time setup)
            if not self._reconnection_callback_registered:
                self._ws_manager.on_reconnected(self._on_reconnected)
                self._reconnection_callback_registered = True
                logger.debug("Registered reconnection callback for handler state reset")

        # Submit and WAIT for connection
        future = self._async_loop.submit(_connect())
        try:
            future.result(timeout=timeout)
            logger.info("WebSocket connection established")
        except Exception as e:
            self._ws_connected = False
            raise ConnectionError(f"Failed to connect WebSocket: {e}") from e

    def _subscribe_instrument(self, sub_type: str, instrument: Instrument, **params) -> None:
        """
        Subscribe single instrument to a data type.

        Creates handler if needed and subscribes via WebSocket.

        Args:
            sub_type: Data type (orderbook, trade, quote, funding_rate, funding_payment, open_interest)
            instrument: Instrument to subscribe
            **params: Additional parameters (e.g., depth for orderbook)
        """
        # Get market_id for this instrument
        market_id = self.instrument_loader.get_market_id(instrument.symbol)
        if market_id is None:
            raise ValueError(f"Market ID not found for {instrument.symbol}")

        # Market stats subscriptions use a single shared handler and callback
        if sub_type in [DataType.OPEN_INTEREST, DataType.FUNDING_RATE, DataType.FUNDING_PAYMENT]:
            handler_key = ("market_stats", 0)  # Shared handler for all markets

            # Create handler if doesn't exist (first market stats subscription)
            if handler_key not in self._handlers:
                handler = self._create_handler(sub_type, instrument, market_id, **params)
                self._handlers[handler_key] = handler

            # Subscribe to WebSocket only ONCE (first market stats subscription)
            if not self._market_stats_subscribed:
                # Check WebSocket manager
                if self._ws_manager is None:
                    raise RuntimeError("WebSocket manager not initialized")

                # Create async subscription task (connection already guaranteed)
                async def _subscribe_market_stats():
                    assert self._ws_manager is not None
                    # Subscribe with unified callback
                    await self._ws_manager.subscribe_market_stats("all", self._make_unified_market_stats_callback())

                # Submit to event loop via AsyncThreadLoop
                self._async_loop.submit(_subscribe_market_stats())
                self._market_stats_subscribed = True
                logger.info("Subscribed to market_stats:all (shared for all instruments)")

            return  # Don't proceed with normal subscription flow

        # Normal subscription flow for non-market-stats types
        handler_key = (sub_type, market_id)

        # Create handler if doesn't exist
        if handler_key not in self._handlers:
            handler = self._create_handler(sub_type, instrument, market_id, **params)
            self._handlers[handler_key] = handler

        # Check WebSocket manager
        if self._ws_manager is None:
            raise RuntimeError("WebSocket manager not initialized")

        # Create async subscription task (connection already guaranteed)
        async def _subscribe():
            assert self._ws_manager is not None

            # Subscribe to appropriate channel
            if sub_type == "orderbook":
                await self._ws_manager.subscribe_orderbook(
                    market_id, self._make_orderbook_callback(instrument, market_id)
                )
            elif sub_type == "trade":
                await self._ws_manager.subscribe_trades(market_id, self._make_trade_callback(instrument, market_id))
            elif sub_type == "quote":
                # Quote is derived from orderbook, so subscribe to orderbook
                await self._ws_manager.subscribe_orderbook(market_id, self._make_quote_callback(instrument, market_id))

        # Submit to event loop via AsyncThreadLoop
        self._async_loop.submit(_subscribe())

    def _create_handler(self, sub_type: str, instrument: Instrument, market_id: int, **params):
        match sub_type:
            case "orderbook":
                return OrderbookHandler(
                    market_id=market_id,
                    instrument=instrument,
                    max_levels=params.get("depth", 200),
                    tick_size_pct=params.get("tick_size_pct", 0),
                )
            case "trade":
                return TradesHandler(market_id=market_id)
            case "quote":
                return QuoteHandler(market_id=market_id)
            case DataType.OPEN_INTEREST | DataType.FUNDING_RATE | DataType.FUNDING_PAYMENT:
                # All market stats subscriptions share a single handler
                return MarketStatsHandler(instrument_loader=self.instrument_loader)
            case _:
                raise ValueError(f"Unknown subscription type: {sub_type}")

    def _make_orderbook_callback(self, instrument: Instrument, market_id: int):
        """Create callback for orderbook messages"""

        async def callback(message: dict):
            handler_key = ("orderbook", market_id)
            handler = cast(OrderbookHandler, self._handlers.get(handler_key))

            if handler and handler.can_handle(message):
                orderbook = handler.handle(message)
                orderbook = cast(OrderBook, orderbook)
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
            handler = cast(TradesHandler, self._handlers.get(handler_key))

            if handler and handler.can_handle(message):
                trades = cast(list[Trade], handler.handle(message))
                if trades:
                    for trade in trades:
                        self.channel.send((instrument, "trade", trade, False))

                    # Generate synthetic quote if no quote/orderbook subscription exists
                    if len(trades) > 0 and not (
                        self.has_subscription(instrument, "quote") or self.has_subscription(instrument, "orderbook")
                    ):
                        last_trade = trades[-1]
                        _price = last_trade.price
                        _time = last_trade.time
                        _s2 = instrument.tick_size / 2.0
                        _bid, _ask = _price - _s2, _price + _s2
                        self._last_quotes[instrument] = Quote(_time, _bid, _ask, 0.0, 0.0)

        return callback

    def _make_unified_market_stats_callback(self):
        """
        Create unified callback for all market stats (processes all instruments).

        This callback processes ALL instruments returned by the handler, not just one.
        It filters to only send data for instruments that are actually subscribed.
        """

        async def callback(message: dict):
            # Use 0 as special marker for shared "all markets" handler
            handler_key = ("market_stats", 0)
            handler = cast(MarketStatsHandler, self._handlers.get(handler_key))

            if handler and handler.can_handle(message):
                results_dict = handler.handle(message)  # dict[Instrument, list[...]] | None

                if results_dict and isinstance(results_dict, dict):
                    # Process ALL instruments in the message
                    for instrument, objects in results_dict.items():
                        # Send all objects for this subscribed instrument
                        for obj in objects:
                            # Determine data type and send
                            if isinstance(obj, FundingRate):
                                d_type = DataType.FUNDING_RATE
                            elif isinstance(obj, FundingPayment):
                                d_type = DataType.FUNDING_PAYMENT
                            elif isinstance(obj, OpenInterest):
                                d_type = DataType.OPEN_INTEREST
                            else:
                                continue  # Skip unknown types

                            self.channel.send((instrument, d_type, obj, False))

        return callback

    def _make_quote_callback(self, instrument: Instrument, market_id: int):
        """Create callback for quote (from orderbook)"""

        async def callback(message: dict):
            handler_key = ("quote", market_id)
            handler = cast(QuoteHandler, self._handlers.get(handler_key))

            if handler and handler.can_handle(message):
                quote = cast(Quote, handler.handle(message))
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

        # Special handling for market stats subscriptions
        if sub_type in [DataType.OPEN_INTEREST, DataType.FUNDING_RATE, DataType.FUNDING_PAYMENT]:
            # Check if ALL market stats subscriptions are now empty
            has_market_stats_subs = (
                bool(self._subscriptions.get(DataType.FUNDING_RATE))
                or bool(self._subscriptions.get(DataType.FUNDING_PAYMENT))
                or bool(self._subscriptions.get(DataType.OPEN_INTEREST))
            )

            # If no more market stats subscriptions, unsubscribe from WebSocket
            if not has_market_stats_subs and self._market_stats_subscribed:
                if self._ws_manager is not None:
                    self._async_loop.submit(self._ws_manager.unsubscribe_market_stats("all"))

                # Remove shared handler
                handler_key = ("market_stats", 0)
                if handler_key in self._handlers:
                    del self._handlers[handler_key]

                self._market_stats_subscribed = False
                logger.info("Unsubscribed from market_stats:all (no more market stats subscriptions)")

        # Normal unsubscribe flow for non-market-stats types
        else:
            for instrument in instruments:
                market_id = self.instrument_loader.get_market_id(instrument.symbol)
                if market_id is not None:
                    # Remove handler
                    handler_key = (sub_type, market_id)
                    if handler_key in self._handlers:
                        del self._handlers[handler_key]

                    # Unsubscribe from WebSocket (if connected)
                    if self._ws_manager is not None:
                        if sub_type == "orderbook":
                            self._async_loop.submit(self._ws_manager.unsubscribe_orderbook(market_id))
                        elif sub_type == "trade":
                            self._async_loop.submit(self._ws_manager.unsubscribe_trades(market_id))
                        elif sub_type == "quote":
                            self._async_loop.submit(self._ws_manager.unsubscribe_orderbook(market_id))

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

        if data_type == "ohlc":
            # Extract timeframe from params (e.g., "ohlc[1h]" → timeframe="1h")
            timeframe = params.get("timeframe", "1m")

            # Calculate how many bars to fetch based on period
            nbarsback = int(pd.Timedelta(period) // pd.Timedelta(timeframe))  # type: ignore

            # Get market ID
            market_id = self.instrument_loader.get_market_id(instrument.symbol)
            if market_id is None:
                logger.warning(f"Market ID not found for {instrument.symbol}, skipping OHLC warmup")
                return

            logger.debug(f"OHLC warmup for {instrument.symbol}: fetching {nbarsback} {timeframe} bars")

            # Fetch historical OHLC data via REST API using AsyncThreadLoop
            async def _fetch_ohlc():
                return await self.client.get_candlesticks(
                    market_id=market_id, resolution=timeframe, count_back=nbarsback
                )

            try:
                future = self._async_loop.submit(_fetch_ohlc())
                candlesticks = future.result(timeout=30)

                # Convert candlesticks to Bar objects
                bars = []
                for candle in candlesticks:
                    # Lighter returns timestamps in milliseconds
                    ts = pd.Timestamp(candle["timestamp"], unit="ms")
                    time_ns = time_as_nsec(ts.asm8)

                    bar = Bar(
                        time=time_ns,
                        open=float(candle["open"]),
                        high=float(candle["high"]),
                        low=float(candle["low"]),
                        close=float(candle["close"]),
                        volume=float(candle.get("volume0", 0.0)),  # Base asset volume
                        volume_quote=float(candle.get("volume1", 0.0)),  # Quote asset volume
                    )
                    bars.append(bar)

                # Send bars to channel with is_warmup=True
                if bars:
                    self.channel.send((instrument, DataType.OHLC[timeframe], bars, True))
                    logger.info(f"OHLC warmup for {instrument.symbol}: loaded {len(bars)} {timeframe} bars")

                    # Generate synthetic quote from last bar
                    last_bar = bars[-1]
                    current_time = time_as_nsec(self.time_provider.time())
                    tick_size_half = instrument.tick_size / 2.0
                    quote = Quote(
                        time=current_time,
                        bid=last_bar.close - tick_size_half,
                        ask=last_bar.close + tick_size_half,
                        bid_size=0.0,
                        ask_size=0.0,
                    )
                    self.channel.send((instrument, DataType.QUOTE, quote, False))
                else:
                    logger.warning(f"No OHLC data returned for {instrument.symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch OHLC data for {instrument.symbol}: {e}")

        elif data_type == "trade":
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

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        """
        Get historical OHLC data for an instrument.

        Args:
            instrument: Instrument to get OHLC data for
            timeframe: Timeframe for bars (e.g., "1m", "5m", "1h")
            nbarsback: Number of bars to retrieve

        Returns:
            List of Bar objects, oldest first

        Note:
            This method fetches data synchronously via REST API.
            For realtime data, use OHLC subscription instead.
        """
        # Get market ID
        market_id = self.instrument_loader.get_market_id(instrument.symbol)
        if market_id is None:
            raise ValueError(f"Market ID not found for {instrument.symbol}")

        # Fetch candlesticks via REST API
        async def _fetch():
            return await self.client.get_candlesticks(market_id=market_id, resolution=timeframe, count_back=nbarsback)

        try:
            future = self._async_loop.submit(_fetch())
            candlesticks = future.result(timeout=30)

            # Convert to Bar objects
            bars = []
            for candle in candlesticks:
                ts = pd.Timestamp(candle["timestamp"], unit="ms")
                time_ns = time_as_nsec(ts.asm8)

                bar = Bar(
                    time=time_ns,
                    open=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close=float(candle["close"]),
                    volume=float(candle.get("volume0", 0.0)),
                    volume_quote=float(candle.get("volume1", 0.0)),
                )
                bars.append(bar)

            return bars

        except Exception as e:
            logger.error(f"Failed to fetch OHLC data for {instrument.symbol}: {e}")
            return []

    def get_quote(self, instrument: Instrument) -> Quote:
        """
        Get the latest quote for an instrument.

        Returns cached quote from last orderbook update or quote subscription.
        If no quote is available, raises ValueError.

        Args:
            instrument: Instrument to get quote for

        Returns:
            Quote object with bid/ask prices and sizes

        Raises:
            ValueError: If no quote is available for the instrument
        """
        quote = self._last_quotes.get(instrument)
        if quote is None:
            raise ValueError(
                f"No quote available for {instrument.symbol}. "
                "Make sure instrument is subscribed to orderbook or quote data."
            )
        return quote

    def close(self) -> None:
        """Close WebSocket connections and cleanup"""
        if self._ws_manager:
            # Submit disconnect to event loop and wait for completion
            future = self._async_loop.submit(self._ws_manager.disconnect())
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")
            logger.info("LighterDataProvider closed")
