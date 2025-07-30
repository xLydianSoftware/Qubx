import asyncio
import json
import threading
import urllib.parse
from collections import defaultdict
from typing import Any, Dict, Tuple

import aiohttp
import pandas as pd

from qubx import logger
from qubx.core.basics import (
    CtrlChannel,
    DataType,
    Instrument,
    ITimeProvider,
    dt_64,
)
from qubx.core.interfaces import IDataProvider, IHealthMonitor
from qubx.core.series import Bar, Quote, Trade
from qubx.data.tardis import TARDIS_EXCHANGE_MAPPERS
from qubx.health import DummyHealthMonitor
from qubx.utils.misc import AsyncThreadLoop, synchronized

from .utils import (
    tardis_convert_orderbook,
    tardis_convert_quote,
    tardis_convert_trade,
    tardis_extract_timeframe,
)


class TardisDataProvider(IDataProvider):
    """
    Data provider implementation for Tardis market data service.

    Uses Tardis Machine server for accessing normalized and historical market data
    via WebSocket and HTTP APIs.
    """

    def __init__(
        self,
        host: str,
        port: int,
        exchange: str,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        health_monitor: IHealthMonitor | None = None,
        asyncio_loop: asyncio.AbstractEventLoop | None = None,
    ):
        """
        Initialize the Tardis data provider.

        Args:
            host: Tardis Machine server host
            port: Tardis Machine server WebSocket port
            exchange: Exchange identifier
            time_provider: Time provider instance
            channel: Control channel for data communication
            health_monitor: Optional health monitoring interface
            asyncio_loop: Optional asyncio event loop. If not provided, a new loop will be created.
        """
        self.time_provider = time_provider
        self.channel = channel
        self._exchange_name = exchange
        self._exchange_id = TARDIS_EXCHANGE_MAPPERS.get(exchange.lower(), exchange.lower())
        self._host = host
        self._port = port
        self._health_monitor = health_monitor or DummyHealthMonitor()

        # Use provided loop or create a new one
        self._own_loop = asyncio_loop is None

        if self._own_loop:
            # Create a new loop and start it in a daemon thread
            self._event_loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_event_loop, daemon=True, name=f"tardis-{exchange.lower()}-loop"
            )
            self._thread.start()
            self._loop = AsyncThreadLoop(self._event_loop)
        else:
            # Use the provided loop
            # asyncio_loop won't be None here since we check with self._own_loop
            assert asyncio_loop is not None
            self._event_loop = asyncio_loop
            self._thread = None
            self._loop = AsyncThreadLoop(asyncio_loop)

        # Websocket base URLs
        self._ws_url_base = f"ws://{host}:{port}"
        self._http_url_base = f"http://{host}:{port - 1}"

        # Subscriptions tracking
        self._subscriptions: defaultdict[str, set[Instrument]] = defaultdict(set)
        self._last_quotes: defaultdict[Instrument, Quote | None] = defaultdict(lambda: None)

        # Store subscription parameters
        self._subscription_params: Dict[Tuple[str, Instrument], Dict[str, Any]] = {}

        # Symbol to instrument mapping for quick lookups
        self._symbol_to_instrument: dict[str, Instrument] = {}

        # WebSocket connection
        self._ws = None
        self._ws_task = None

        logger.info(f"{self.__prefix()} Initialized Tardis Data Provider")

    def __prefix(self) -> str:
        """Create a colored prefix with the exchange name for logging."""
        return f"<yellow>{self._exchange_name}</yellow>"

    def _run_event_loop(self):
        """Run the event loop in a separate thread."""
        # This is only called when we have a valid event loop
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()

    @property
    def is_simulation(self) -> bool:
        """Check if data provider is in simulation mode."""
        return False

    def subscribe(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        reset: bool = False,
    ) -> None:
        """Subscribe to market data for a list of instruments."""
        sub_type, params = DataType.from_str(subscription_type)

        if reset:
            self._subscriptions[sub_type] = set(instruments)
        else:
            self._subscriptions[sub_type].update(instruments)

        # Store subscription parameters for each instrument
        for instrument in instruments:
            self._subscription_params[(sub_type, instrument)] = params

        # Stop previous connection if it's running
        if self._ws_task is not None and not self._ws_task.done():
            try:
                self._loop.submit(self._close_websocket_connection()).result(timeout=5)
            except TimeoutError:
                logger.warning(f"{self.__prefix()} Timeout while closing previous WebSocket connection")
            except Exception as e:
                logger.error(f"{self.__prefix()} Error while closing previous WebSocket connection: {e}")
            self._ws_task = None

        # Start a new WebSocket connection with updated subscriptions
        self._ws_task = self._loop.submit(self._start_websocket_connection())

    def unsubscribe(self, subscription_type: str | None, instruments: set[Instrument]) -> None:
        """Unsubscribe from market data for a list of instruments."""
        if subscription_type is None:
            # Unsubscribe from all subscription types
            for sub_type in list(self._subscriptions.keys()):
                self._subscriptions[sub_type] = self._subscriptions[sub_type].difference(instruments)
                # Remove subscription parameters
                for instrument in instruments:
                    if (sub_type, instrument) in self._subscription_params:
                        del self._subscription_params[(sub_type, instrument)]
        else:
            sub_type, _ = DataType.from_str(subscription_type)
            if sub_type in self._subscriptions:
                self._subscriptions[sub_type] = self._subscriptions[sub_type].difference(instruments)
                # Remove subscription parameters
                for instrument in instruments:
                    if (sub_type, instrument) in self._subscription_params:
                        del self._subscription_params[(sub_type, instrument)]

        # Stop previous connection if it's running
        if self._ws_task is not None and not self._ws_task.done():
            try:
                self._loop.submit(self._close_websocket_connection()).result(timeout=5)
            except TimeoutError:
                logger.warning(f"{self.__prefix()} Timeout while closing WebSocket connection")
            except Exception as e:
                logger.error(f"{self.__prefix()} Error while closing WebSocket connection: {e}")
            self._ws_task = None

        # Start a new WebSocket connection with updated subscriptions
        self._ws_task = self._loop.submit(self._start_websocket_connection())

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if an instrument has a subscription."""
        sub_type, _ = DataType.from_str(subscription_type)
        return sub_type in self._subscriptions and instrument in self._subscriptions[sub_type]

    def get_subscriptions(self, instrument: Instrument | None = None) -> list[str]:
        """Get all subscriptions for an instrument."""
        if instrument is not None:
            return [sub for sub, instrs in self._subscriptions.items() if instrument in instrs]
        return list(self._subscriptions.keys())

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        """Get a list of instruments that are subscribed to a specific subscription type."""
        if subscription_type is None:
            # Return all subscribed instruments across all subscription types
            result = set()
            for instruments in self._subscriptions.values():
                result.update(instruments)
            return list(result)

        sub_type, _ = DataType.from_str(subscription_type)
        return list(self._subscriptions[sub_type]) if sub_type in self._subscriptions else []

    def warmup(self, configs: dict[tuple[str, Instrument], str]) -> None:
        """Run warmup for subscriptions."""
        warmup_tasks = []

        for (sub_type, instrument), period in configs.items():
            data_type, params = DataType.from_str(sub_type)

            if data_type == DataType.OHLC:
                warmup_tasks.append(self._warmup_ohlc(instrument, period, params.get("timeframe", "1m")))
            elif data_type == DataType.TRADE:
                warmup_tasks.append(self._warmup_trades(instrument, period))
            elif data_type == DataType.ORDERBOOK:
                warmup_tasks.append(
                    self._warmup_orderbook(
                        instrument, period, params.get("depth", 10), params.get("tick_size_pct", 0.01)
                    )
                )

        if warmup_tasks:
            # Run warmup tasks
            async def run_warmup_tasks():
                return await asyncio.gather(*warmup_tasks, return_exceptions=True)

            results = self._loop.submit(run_warmup_tasks()).result(timeout=120)

            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"{self.__prefix()} Warmup task {i} failed: {result}")

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        """Get historical OHLC data for an instrument."""
        if nbarsback <= 0:
            return []

        # Calculate start time based on timeframe and nbarsback
        end_time = self.time_provider.time()
        start_time = end_time - pd.Timedelta(timeframe) * nbarsback

        # Format dates for Tardis API
        start_str = pd.Timestamp(start_time).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end_time).strftime("%Y-%m-%d")

        # Get Tardis-compatible symbol
        symbol = self._get_tardis_symbol(instrument)

        # Build options for HTTP request
        options = {
            "exchange": self._exchange_id,
            "symbols": [symbol],
            "from": start_str,
            "to": end_str,
            "dataTypes": [f"trade_bar_{self._map_timeframe_to_tardis(timeframe)}"],
        }

        encoded_options = urllib.parse.quote_plus(json.dumps(options))
        url = f"{self._http_url_base}/replay-normalized?options={encoded_options}"

        async def fetch_ohlc():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Failed to fetch OHLC data: {error_text}")

                    bars = []
                    async for line in response.content:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data["type"] == "trade_bar":
                            # Convert Tardis trade_bar to Bar object
                            bar_time = pd.Timestamp(data["timestamp"]).value
                            bar = Bar(
                                bar_time,
                                data["open"],
                                data["high"],
                                data["low"],
                                data["close"],
                                volume=data["volume"],
                                bought_volume=data.get("buyVolume", 0),
                            )
                            bars.append(bar)

                    return bars[-nbarsback:] if len(bars) > nbarsback else bars

        try:
            return self._loop.submit(fetch_ohlc()).result(timeout=30)
        except Exception as e:
            logger.error(f"{self.__prefix()} Error fetching OHLC data: {e}")
            return []

    def get_quote(self, instrument: Instrument) -> Quote | None:
        """Get the latest quote for an instrument."""
        return self._last_quotes[instrument]

    def close(self):
        """Close the data provider."""
        # Cancel any ongoing WebSocket tasks
        if self._ws_task is not None and not self._ws_task.done():
            try:
                self._loop.submit(self._close_websocket_connection()).result(timeout=5)
            except TimeoutError:
                logger.warning(f"{self.__prefix()} Timeout while closing WebSocket connection")
            except Exception as e:
                logger.error(f"{self.__prefix()} Error while closing WebSocket connection: {e}")
            finally:
                self._ws_task = None
                self._ws = None

        # If we created our own loop, stop it
        if self._own_loop and self._thread is not None:
            try:
                # Stop the event loop
                # event_loop is guaranteed to be non-None if _own_loop is True
                self._event_loop.stop()

                # Wait for the thread to join
                if self._thread.is_alive():
                    self._thread.join(timeout=5)
            except Exception as e:
                logger.error(f"{self.__prefix()} Error shutting down event loop: {e}")

        logger.info(f"{self.__prefix()} Tardis data provider closed")

    def exchange(self) -> str:
        """Return the name of the exchange this provider reads data."""
        return self._exchange_name

    def _get_tardis_symbol(self, instrument: Instrument) -> str:
        """
        Get the symbol in Tardis-compatible format.

        For bitfinex-derivatives, strip the 't' prefix if it exists.
        Also stores the mapping from symbols to instruments for quick lookups.
        """
        symbol = instrument.exchange_symbol
        if self._exchange_id == "bitfinex-derivatives" and symbol.startswith("t"):
            tardis_symbol = symbol[1:]
        else:
            tardis_symbol = symbol

        # Store the mapping from Tardis symbol to instrument
        self._symbol_to_instrument[tardis_symbol] = instrument
        return tardis_symbol

    @synchronized
    async def _start_websocket_connection(self):
        """Start the WebSocket connection to Tardis Machine."""
        if not self._subscriptions:
            logger.debug(f"{self.__prefix()} No active subscriptions, not starting WebSocket")
            return

        # Build options for the WebSocket connection
        stream_options = []

        # Map all instruments by data type
        all_instruments = set()
        for instruments in self._subscriptions.values():
            all_instruments.update(instruments)

        # Use exchange_symbol instead of symbol, and handle special cases for bitfinex
        symbols = [self._get_tardis_symbol(instrument) for instrument in all_instruments]

        # Map data types to Tardis data types
        data_types = []
        for sub_type in self._subscriptions:
            tardis_type = self._map_data_type_to_tardis(sub_type)
            if tardis_type:
                data_types.append(tardis_type)

        stream_options.append(
            {
                "exchange": self._exchange_id,
                "symbols": symbols,
                "dataTypes": data_types,
            }
        )

        options = urllib.parse.quote_plus(json.dumps(stream_options))
        ws_url = f"{self._ws_url_base}/ws-stream-normalized?options={options}"

        logger.info(
            f"{self.__prefix()} Starting WebSocket connection to Tardis Machine for data types {data_types} symbols {symbols}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as websocket:
                    self._ws = websocket

                    logger.info(f"{self.__prefix()} WebSocket connected")

                    async for msg in websocket:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                await self._process_tardis_message(data)
                            except json.JSONDecodeError:
                                logger.error(f"{self.__prefix()} Invalid JSON: {msg.data}")
                            except Exception as e:
                                logger.error(f"{self.__prefix()} Error processing message: {e}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"{self.__prefix()} WebSocket error: {msg.data}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info(f"{self.__prefix()} WebSocket closed")
                            break
        except asyncio.CancelledError:
            logger.info(f"{self.__prefix()} WebSocket connection cancelled")
        except Exception as e:
            logger.error(f"{self.__prefix()} WebSocket error: {e}")
        finally:
            logger.info(f"{self.__prefix()} WebSocket connection closed")
            self._ws = None

    async def _close_websocket_connection(self):
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _process_tardis_message(self, data: dict[str, Any]):
        """Process a message from Tardis Machine."""
        if not data or "type" not in data or "exchange" not in data:
            return

        # Find matching instrument
        symbol = data.get("symbol")
        if not symbol:
            return

        # Look up the instrument directly from our symbol mapping
        instrument = self._symbol_to_instrument.get(symbol)
        if not instrument:
            # No matching instrument found
            return

        # Get message timestamp
        timestamp = pd.Timestamp(data.get("timestamp", data.get("localTimestamp")))
        msg_time = timestamp.value

        # Record data arrival for health monitoring
        tardis_type = data["type"]
        tardis_name = data["name"] if "name" in data else ""
        qubx_type = self._map_tardis_type_to_data_type(tardis_type)
        if qubx_type:
            self._health_monitor.record_data_arrival(qubx_type, dt_64(msg_time, "ns"))

        if tardis_type == "trade":
            if DataType.TRADE in self._subscriptions and instrument in self._subscriptions[DataType.TRADE]:
                trade = tardis_convert_trade(data, instrument)
                if trade:
                    self.channel.send((instrument, DataType.TRADE, trade, False))

        elif tardis_type == "book_change":
            if DataType.ORDERBOOK in self._subscriptions and instrument in self._subscriptions[DataType.ORDERBOOK]:
                # For book changes, we'd need to maintain a full order book locally
                # This is a simplified implementation
                pass

        elif tardis_type == "quote" or tardis_name == "quote":
            if DataType.QUOTE in self._subscriptions and instrument in self._subscriptions[DataType.QUOTE]:
                quote = tardis_convert_quote(data, instrument)
                if quote:
                    self._last_quotes[instrument] = quote
                    self.channel.send((instrument, DataType.QUOTE, quote, False))

        elif tardis_type == "book_snapshot":
            if DataType.ORDERBOOK in self._subscriptions and instrument in self._subscriptions[DataType.ORDERBOOK]:
                # Get orderbook parameters from subscription
                params = self._get_orderbook_params(instrument)
                levels = params.get("depth", 50)
                tick_size_pct = params.get("tick_size_pct", 0.01)

                orderbook = tardis_convert_orderbook(data, instrument, levels, tick_size_pct)
                if orderbook:
                    # Update last quote from order book
                    self._last_quotes[instrument] = orderbook.to_quote()
                    self.channel.send((instrument, DataType.ORDERBOOK, orderbook, False))

        elif tardis_type == "trade_bar":
            # Map trade_bar to OHLC data type with timeframe
            timeframe = tardis_extract_timeframe(data["name"])
            if timeframe:
                ohlc_type = DataType.OHLC[timeframe]
                sub_type, _ = DataType.from_str(ohlc_type)

                if sub_type in self._subscriptions and instrument in self._subscriptions[sub_type]:
                    bar = Bar(
                        msg_time,
                        data["open"],
                        data["high"],
                        data["low"],
                        data["close"],
                        volume=data["volume"],
                        bought_volume=data.get("buyVolume", 0),
                    )

                    self.channel.send((instrument, ohlc_type, bar, False))

    def _get_orderbook_params(self, instrument: Instrument) -> Dict[str, Any]:
        """
        Get orderbook parameters for a specific instrument.
        Returns default parameters if not found.
        """
        key = (DataType.ORDERBOOK, instrument)
        if key in self._subscription_params:
            return self._subscription_params[key]
        return {"depth": 50, "tick_size_pct": 0.01}  # Default parameters

    async def _warmup_ohlc(self, instrument: Instrument, period: str, timeframe: str):
        """Fetch historical OHLC data for warmup."""
        # Calculate start and end time
        end_time = self.time_provider.time()
        start_time = end_time - pd.Timedelta(period)

        # Format dates for Tardis API
        start_str = pd.Timestamp(start_time).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end_time).strftime("%Y-%m-%d")

        # Get Tardis-compatible symbol
        symbol = self._get_tardis_symbol(instrument)

        # Build options for HTTP request
        options = {
            "exchange": self._exchange_id,
            "symbols": [symbol],
            "from": start_str,
            "to": end_str,
            "dataTypes": [f"trade_bar_{self._map_timeframe_to_tardis(timeframe)}"],
        }

        encoded_options = urllib.parse.quote_plus(json.dumps(options))
        url = f"{self._http_url_base}/replay-normalized?options={encoded_options}"

        logger.info(f"{self.__prefix()} Warming up OHLC {timeframe} for {symbol} over {period}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to fetch warmup OHLC data: {error_text}")

                bars = []
                async for line in response.content:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data["type"] == "trade_bar" and data["symbol"] == symbol:
                        # Convert Tardis trade_bar to Bar object
                        bar_time = pd.Timestamp(data["timestamp"]).value
                        bar = Bar(
                            bar_time,
                            data["open"],
                            data["high"],
                            data["low"],
                            data["close"],
                            volume=data["volume"],
                            bought_volume=data.get("buyVolume", 0),
                        )
                        bars.append(bar)

                # Send all bars as historical data
                if bars:
                    self.channel.send((instrument, DataType.OHLC[timeframe], bars, True))
                    logger.info(f"{self.__prefix()} Loaded {len(bars)} {timeframe} bars for {symbol}")

    async def _warmup_trades(self, instrument: Instrument, period: str):
        """Fetch historical trade data for warmup."""
        # Calculate start and end time
        end_time = self.time_provider.time()
        start_time = end_time - pd.Timedelta(period)

        # Format dates for Tardis API
        start_str = pd.Timestamp(start_time).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end_time).strftime("%Y-%m-%d")

        # Get Tardis-compatible symbol
        symbol = self._get_tardis_symbol(instrument)

        # Build options for HTTP request
        options = {
            "exchange": self._exchange_id,
            "symbols": [symbol],
            "from": start_str,
            "to": end_str,
            "dataTypes": ["trade"],
        }

        encoded_options = urllib.parse.quote_plus(json.dumps(options))
        url = f"{self._http_url_base}/replay-normalized?options={encoded_options}"

        logger.info(f"{self.__prefix()} Warming up trades for {symbol} over {period}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to fetch warmup trade data: {error_text}")

                trades = []
                async for line in response.content:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data["type"] == "trade" and data["symbol"] == symbol:
                        # Convert Tardis trade to Trade object
                        trade_time = pd.Timestamp(data["timestamp"]).value
                        trade = Trade(
                            trade_time,
                            data["price"],
                            data["amount"],
                            1 if data.get("side") == "buy" else -1,
                        )
                        trades.append(trade)

                # Send all trades as historical data
                if trades:
                    self.channel.send((instrument, DataType.TRADE, trades, True))
                    logger.info(f"{self.__prefix()} Loaded {len(trades)} trades for {symbol}")

    async def _warmup_orderbook(self, instrument: Instrument, period: str, depth: int, tick_size_pct: float):
        """Fetch historical orderbook data for warmup."""
        # Calculate start and end time
        end_time = self.time_provider.time()
        start_time = end_time - pd.Timedelta(period)

        # Format dates for Tardis API
        start_str = pd.Timestamp(start_time).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end_time).strftime("%Y-%m-%d")

        # Get Tardis-compatible symbol
        symbol = self._get_tardis_symbol(instrument)

        # Build options for HTTP request
        options = {
            "exchange": self._exchange_id,
            "symbols": [symbol],
            "from": start_str,
            "to": end_str,
            "dataTypes": [f"book_snapshot_{depth}_100ms"],
        }

        encoded_options = urllib.parse.quote_plus(json.dumps(options))
        url = f"{self._http_url_base}/replay-normalized?options={encoded_options}"

        logger.info(f"{self.__prefix()} Warming up orderbook for {symbol} over {period}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Failed to fetch warmup orderbook data: {error_text}")

                orderbooks = []
                async for line in response.content:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data["type"] == "book_snapshot" and data["symbol"] == symbol:
                        # Use the utility function to convert Tardis book_snapshot to OrderBook object
                        orderbook = tardis_convert_orderbook(data, instrument, depth, tick_size_pct)
                        if orderbook:
                            orderbooks.append(orderbook)

                # Send orderbooks as historical data
                if orderbooks:
                    params = {"tick_size_pct": tick_size_pct, "depth": depth}
                    sub_type = f"{DataType.ORDERBOOK}({tick_size_pct}, {depth})"
                    self.channel.send((instrument, sub_type, orderbooks, True))
                    logger.info(f"{self.__prefix()} Loaded {len(orderbooks)} orderbooks for {symbol}")

    def _map_data_type_to_tardis(self, data_type: str) -> str:
        """Map QubX data type to Tardis data type."""
        if data_type == DataType.TRADE:
            return "trade"
        elif data_type == DataType.QUOTE:
            return "quote"
        elif data_type == DataType.ORDERBOOK:
            return "book_snapshot_2000_100ms"  # Default depth and interval
        elif data_type == DataType.OHLC:
            return "trade_bar_1m"  # Default timeframe
        else:
            return ""

    def _map_tardis_type_to_data_type(self, tardis_type: str) -> str:
        """Map Tardis data type to QubX data type."""
        if tardis_type == "trade":
            return DataType.TRADE
        elif tardis_type == "quote":
            return DataType.QUOTE
        elif tardis_type.startswith("book_snapshot"):
            return DataType.ORDERBOOK
        elif tardis_type.startswith("trade_bar"):
            return DataType.OHLC
        else:
            return ""

    def _map_timeframe_to_tardis(self, timeframe: str) -> str:
        """Map QubX timeframe to Tardis timeframe format."""
        # Map common timeframes (1m, 5m, 1h, etc.) to milliseconds format
        tf_map = {
            "1m": "60000ms",
            "5m": "300000ms",
            "15m": "900000ms",
            "30m": "1800000ms",
            "1h": "3600000ms",
            "4h": "14400000ms",
            "1d": "86400000ms",
        }

        # Try direct mapping first
        if timeframe.lower() in tf_map:
            return tf_map[timeframe.lower()]

        # Otherwise, try to parse the timeframe
        try:
            pd_timedelta = pd.Timedelta(timeframe)
            return f"{int(pd_timedelta.total_seconds() * 1000)}ms"
        except:
            # Fallback to default 1m if parsing fails
            logger.warning(f"{self.__prefix()} Could not map timeframe {timeframe}, using 1m")
            return "60000ms"
