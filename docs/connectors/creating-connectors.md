# Creating Exchange Connectors for Qubx

This guide explains how to build exchange connectors for Qubx. Connectors enable Qubx to communicate with exchanges for market data streaming and order execution.

## Overview

A complete exchange connector consists of four main components:

1. **Data Provider** (`IDataProvider`) - Streams real-time market data (quotes, trades, orderbooks, etc.)
2. **Account Processor** (`IAccountProcessor`) - Tracks positions, balances, and orders
3. **Broker** (`IBroker`) - Executes orders on the exchange
4. **Data Reader** (`DataReader`) - Fetches historical data (OHLC, funding rates, etc.)

These components are registered with Qubx's plugin system using decorators, allowing them to be loaded dynamically.

## Plugin System Architecture

Qubx uses a registry-based plugin system. Connectors are registered using decorators:

```python
from qubx.connectors.registry import data_provider, account_processor, broker
from qubx.data.registry import reader

@data_provider("myexchange")
class MyDataProvider(IDataProvider):
    ...

@account_processor("myexchange")
class MyAccountProcessor(BasicAccountProcessor):
    ...

@broker("myexchange")
class MyBroker(IBroker):
    ...

@reader("myexchange")
class MyDataReader(DataReader):
    ...
```

When users specify `connector: myexchange` in their configuration, Qubx looks up the registered classes and instantiates them.

## Project Structure

A typical connector package follows this structure:

```
my_connector/
├── pyproject.toml
├── README.md
├── src/
│   └── my_connector/
│       ├── __init__.py      # Exports + registration
│       ├── data.py          # IDataProvider implementation
│       ├── account.py       # IAccountProcessor implementation
│       ├── broker.py        # IBroker implementation
│       ├── reader.py        # DataReader implementation
│       ├── client.py        # Exchange API client
│       ├── factory.py       # Shared resource caching
│       ├── websocket.py     # WebSocket management
│       ├── parsers.py       # Message parsing utilities
│       └── handlers/        # Message handlers
│           ├── __init__.py
│           ├── orderbook.py
│           └── trades.py
└── tests/
    ├── unit/
    └── integration/
```

## pyproject.toml Configuration

```toml
[project]
name = "my-connector"
version = "0.1.0"
description = "My exchange connector for Qubx"
requires-python = ">=3.12,<4.0"
dependencies = [
    "qubx>=0.12.0",      # Peer dependency on Qubx
    # ... exchange-specific dependencies
]

[build-system]
requires = ["hatchling>=1.21.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_connector"]
```

## Component Implementation

### 1. Data Provider

The data provider streams market data from the exchange:

```python
from qubx.connectors.registry import data_provider
from qubx.core.interfaces import IDataProvider

@data_provider("myexchange")
class MyDataProvider(IDataProvider):
    SUPPORTED_SUBSCRIPTIONS = [
        DataType.ORDERBOOK,
        DataType.TRADE,
        DataType.QUOTE,
    ]

    def __init__(
        self,
        exchange_name: str,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        health_monitor: IHealthMonitor,
        account_manager: "AccountConfigurationManager",
        **kwargs,
    ):
        # Get credentials from account manager
        creds = account_manager.get_exchange_credentials(exchange_name)
        settings = account_manager.get_exchange_settings(exchange_name)

        # Initialize client and WebSocket
        self.client = get_my_client(creds.api_key, creds.secret)
        self._ws_manager = get_my_ws_manager()

    @property
    def is_simulation(self) -> bool:
        return False

    def exchange(self) -> str:
        return "MYEXCHANGE"

    def subscribe(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        reset: bool = False,
    ) -> None:
        # Subscribe to market data channels
        for instrument in instruments:
            self._subscribe_instrument(subscription_type, instrument)

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes.get(instrument)

    def close(self) -> None:
        if self._ws_manager:
            self._ws_manager.disconnect()
```

### 2. Account Processor

The account processor tracks positions, balances, and orders:

```python
from qubx.connectors.registry import account_processor
from qubx.core.account import BasicAccountProcessor

@account_processor("myexchange")
class MyAccountProcessor(BasicAccountProcessor):
    def __init__(
        self,
        exchange_name: str,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        account_manager: "AccountConfigurationManager",
        tcc: TransactionCostsCalculator,
        health_monitor: IHealthMonitor,
        **kwargs,
    ):
        creds = account_manager.get_exchange_credentials(exchange_name)
        account_id = creds.get_extra_field("account_index", exchange_name)

        super().__init__(
            account_id=str(account_id),
            time_provider=time_provider,
            base_currency=creds.base_currency,
            health_monitor=health_monitor,
            exchange="MYEXCHANGE",
            tcc=tcc,
            initial_capital=0,
        )

        # Initialize WebSocket for account updates
        self.ws_manager = get_my_ws_manager()
        self.channel = channel

    def start(self):
        # Subscribe to account updates
        self._async_loop.submit(self._subscribe_account_updates())

    def stop(self):
        # Cleanup
        pass
```

### 3. Broker

The broker executes orders on the exchange:

```python
from qubx.connectors.registry import broker
from qubx.core.interfaces import IBroker

@broker("myexchange")
class MyBroker(IBroker):
    def __init__(
        self,
        exchange_name: str,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        data_provider: IDataProvider,
        account_manager: "AccountConfigurationManager",
        health_monitor: IHealthMonitor,
        **kwargs,
    ):
        creds = account_manager.get_exchange_credentials(exchange_name)
        self.client = get_my_client(creds.api_key, creds.secret)
        self.time_provider = time_provider
        self.account = account

    @property
    def is_simulated_trading(self) -> bool:
        return False

    def exchange(self) -> str:
        return "MYEXCHANGE"

    def send_order(self, request: OrderRequest) -> Order:
        # Execute order synchronously
        return self._async_loop.submit(
            self._create_order(request)
        ).result()

    def send_order_async(self, request: OrderRequest) -> str | None:
        # Execute order asynchronously
        self._async_loop.submit(self._create_order(request))
        return request.client_id

    def cancel_order(self, order_id: str | None = None,
                     client_order_id: str | None = None) -> bool:
        # Cancel order
        ...

    def update_order(self, price: float, amount: float,
                     order_id: str | None = None,
                     client_order_id: str | None = None) -> Order:
        # Modify order
        ...
```

### 4. Data Reader

The data reader fetches historical data from the exchange REST API. This is essential for:
- Strategy warmup (loading recent OHLC data before live trading starts)
- Backtesting with recent exchange data
- Fetching funding rate history for analysis
- Research and data exploration

```python
from qubx.data.readers import DataReader, DataTransformer
from qubx.data.registry import reader

@reader("myexchange")
class MyDataReader(DataReader):
    """
    Data reader for MyExchange.

    Fetches historical OHLC data and funding payments via REST API.
    """

    SUPPORTED_DATA_TYPES = {"ohlc", "funding_payment"}

    def __init__(
        self,
        client: MyClient,
        max_bars: int = 10_000,
        max_history: str = "30d",
    ):
        """
        Initialize data reader.

        Args:
            client: Pre-configured API client instance
            max_bars: Maximum bars to fetch per request
            max_history: Maximum historical data lookback
        """
        self.client = client
        self._max_bars = max_bars
        self._max_history = pd.Timedelta(max_history)

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 0,
        timeframe: str = "1m",
        data_type: str = "ohlc",
        **kwargs,
    ) -> Iterable | list:
        """
        Read historical data for a single instrument.

        Args:
            data_id: Data identifier (e.g., "MYEXCHANGE:BTCUSDC")
            start: Start timestamp
            stop: End timestamp
            transform: Data transformer for output format
            chunksize: If > 0, return iterator of chunks
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            data_type: Type of data ("ohlc" or "funding_payment")

        Returns:
            List of data or iterator if chunksize > 0
        """
        if data_type not in self.SUPPORTED_DATA_TYPES:
            return []

        instrument = self._get_instrument(data_id)
        if instrument is None:
            return []

        # Fetch data from exchange
        data = self._fetch_data(instrument, data_type, timeframe, start, stop)

        # Apply transformation
        column_names = self._get_column_names(data_type)
        transform.start_transform(data_id, column_names, start=start, stop=stop)
        transform.process_data(data)
        return transform.collect()

    def get_candles(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLC candles for multiple symbols.

        This method is used by strategies to get historical price data
        for warmup or analysis.

        Args:
            exchange: Exchange name (must match reader name)
            symbols: List of symbols to fetch (None = all)
            start: Start timestamp
            stop: End timestamp
            timeframe: Candle timeframe

        Returns:
            DataFrame with MultiIndex (timestamp, symbol) and
            columns: open, high, low, close, volume
        """
        if exchange not in self.get_names():
            return pd.DataFrame()

        instruments = self._get_instruments_for_symbols(symbols)

        # Fetch candles for all instruments concurrently
        all_data = []
        for instrument in instruments:
            candles = self._fetch_ohlcv(instrument, timeframe, start, stop)
            for candle in candles:
                all_data.append({
                    "timestamp": candle[0],
                    "symbol": instrument.symbol,
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                })

        df = pd.DataFrame(all_data)
        df = df.sort_values("timestamp")
        df = df.set_index(["timestamp", "symbol"])
        return df

    def get_funding_payment(
        self,
        exchange: str,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        stop: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Fetch funding rate history for perpetual contracts.

        Args:
            exchange: Exchange name
            symbols: List of symbols (None = all)
            start: Start timestamp
            stop: End timestamp

        Returns:
            DataFrame with columns: funding_rate, funding_interval_hours
        """
        # Implementation similar to get_candles but for funding data
        ...

    def get_names(self, **kwargs) -> list[str]:
        """Return list of exchange names this reader supports."""
        return ["MYEXCHANGE"]

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """Return available symbols for the given data type."""
        if dtype not in self.SUPPORTED_DATA_TYPES:
            return []
        instruments = lookup.find_instruments(exchange=exchange)
        return [i.symbol for i in instruments]

    def get_time_ranges(
        self, symbol: str, dtype: str
    ) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        """Return available time range for the symbol."""
        if dtype not in self.SUPPORTED_DATA_TYPES:
            return None, None
        end_time = now_utc()
        start_time = end_time - self._max_history
        return start_time.to_datetime64(), end_time.to_datetime64()

    def close(self):
        """Clean up resources."""
        pass
```

#### Using the Reader in Strategies

The reader is configured via the `aux` section in strategy configuration:

```yaml
strategy: my_strategy.MyStrategy
parameters:
  lookback_days: 30

live:
  exchanges:
    MYEXCHANGE:
      connector: myexchange
      universe:
        - BTCUSDC
        - ETHUSDC

# Configure auxiliary data reader
aux:
  reader: myexchange
  args:
    max_history: "30d"
```

In the strategy, access historical data via the context:

```python
class MyStrategy(IStrategy):
    def on_init(self, initializer: IStrategyInitializer):
        # Request OHLC warmup data
        initializer.set_subscription_warmup({
            DataType.OHLC["1h"]: "7d"  # Warmup with 7 days of hourly data
        })

    def on_start(self, ctx: IStrategyContext):
        # Fetch funding rate history for analysis
        funding_data = ctx.get_aux_data(
            "funding_payment",
            exchange="MYEXCHANGE",
            symbols=["BTCUSDC", "ETHUSDC"],
            start="2024-01-01",
        )

        # Calculate average funding rate
        avg_funding = funding_data["funding_rate"].mean()
        logger.info(f"Average funding rate: {avg_funding:.4%}")
```

## Factory Pattern

Use a factory pattern to cache shared resources (clients, WebSocket connections):

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_my_client(api_key: str, secret: str, **kwargs) -> MyClient:
    """Get cached client instance."""
    return MyClient(api_key=api_key, secret=secret, **kwargs)

@lru_cache(maxsize=32)
def get_my_ws_manager(api_key: str, secret: str, **kwargs) -> MyWSManager:
    """Get cached WebSocket manager instance."""
    client = get_my_client(api_key, secret, **kwargs)
    return MyWSManager(client=client)

def clear_cache() -> None:
    """Clear all caches. Useful for testing."""
    get_my_client.cache_clear()
    get_my_ws_manager.cache_clear()
```

## WebSocket Message Handling

For real-time data, extend `BaseWebSocketManager`:

```python
from qubx.utils.websocket_manager import BaseWebSocketManager

class MyWSManager(BaseWebSocketManager):
    def __init__(self, client: MyClient, **kwargs):
        super().__init__(
            url="wss://api.myexchange.com/stream",
            ping_interval=None,
            app_ping_interval=20.0,
        )
        self._client = client

    async def _send_subscription_message(self, channel: str, params: dict) -> None:
        """Send exchange-specific subscription message."""
        message = {"type": "subscribe", "channel": channel}
        await self.send(message)

    async def _send_unsubscription_message(self, channel: str) -> None:
        """Send exchange-specific unsubscription message."""
        message = {"type": "unsubscribe", "channel": channel}
        await self.send(message)

    def _extract_channel(self, message: dict) -> str | None:
        """Extract channel identifier from message."""
        return message.get("channel")
```

## Message Handlers

Create handlers for different message types:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

class BaseHandler(ABC, Generic[T]):
    @abstractmethod
    def can_handle(self, message: dict) -> bool:
        """Check if handler can process this message."""
        ...

    @abstractmethod
    def handle(self, message: dict) -> T | None:
        """Process message and return Qubx data type."""
        ...

class OrderbookHandler(BaseHandler[OrderBook]):
    def __init__(self, market_id: int, instrument: Instrument):
        self.market_id = market_id
        self.instrument = instrument
        self._lob = LOB(depth=200)

    def can_handle(self, message: dict) -> bool:
        return message.get("channel") == f"orderbook:{self.market_id}"

    def handle(self, message: dict) -> OrderBook | None:
        # Parse message and update LOB state
        # Return OrderBook snapshot
        ...
```

## Configuration

Users configure your connector in their strategy YAML:

```yaml
strategy: my_strategy.MyStrategy
parameters:
  param1: value1

live:
  read_only: false
  exchanges:
    MYEXCHANGE:
      connector: myexchange
      universe:
        - BTCUSDC
        - ETHUSDC
      extra:
        # Connector-specific options
        account_index: 12345

# Optional: Configure data reader for historical data
aux:
  reader: myexchange
  args:
    max_history: "30d"
    max_bars: 10000
```

## Testing

### Unit Tests

```python
import pytest
from my_connector.handlers import OrderbookHandler

def test_orderbook_handler():
    handler = OrderbookHandler(market_id=0, instrument=mock_instrument)

    message = {
        "channel": "orderbook:0",
        "type": "snapshot",
        "bids": [{"price": "100.0", "size": "1.0"}],
        "asks": [{"price": "101.0", "size": "1.0"}],
    }

    result = handler.handle(message)
    assert result is not None
    assert result.top_bid == 100.0
    assert result.top_ask == 101.0
```

### Integration Tests

```python
import pytest

@pytest.mark.integration
async def test_websocket_connection():
    client = MyClient(api_key="test", secret="test")
    ws = MyWSManager(client=client)

    await ws.connect()
    assert ws.is_connected

    await ws.disconnect()
```

## Installation and Usage

After publishing your connector:

```bash
pip install my-connector
```

Users import the connector to register it:

```python
import my_connector  # Registers with Qubx

# Then use in configuration
```

Or specify in plugins configuration:

```yaml
plugins:
  modules:
    - my_connector
```

## Best Practices

1. **Use Caching**: Cache shared resources (clients, connections) to avoid duplication
2. **Handle Reconnection**: Implement proper WebSocket reconnection with state reset
3. **Rate Limiting**: Respect exchange rate limits using `RateLimiterRegistry`
4. **Error Handling**: Send errors through the channel for strategy notification
5. **Logging**: Use `from qubx import logger` for consistent logging
6. **Type Hints**: Use modern Python type hints (`list`, `dict`, `| None`)
7. **Testing**: Write both unit tests and integration tests

## Examples

See these connectors for reference implementations:

- **CCXT Connector**: General-purpose connector using CCXT library (`qubx.connectors.ccxt`)
- **XLighter Connector**: WebSocket-based connector for Lighter exchange (`qubx_lighter` package)
