# CCXT Exchange Integration Requirements

This document outlines the required CCXT methods that exchanges must support for full integration with the Qubx quantitative trading framework. The requirements are organized by component (Data Provider, Account Processor, Trading Broker) and based on analysis of the existing codebase.

## Overview

Qubx integrates with exchanges through the CCXT library, providing three main services:
- **Data Provider**: Real-time and historical market data
- **Account Processor**: Account balance, position, and order management  
- **Trading Broker**: Order execution and management

## Data Provider Requirements

### Market Data Methods

| Method | Purpose | Usage in Qubx |
|--------|---------|---------------|
| `fetch_ohlcv(symbol, timeframe, since, limit)` | Historical OHLCV data | `src/qubx/connectors/ccxt/reader.py:209` |
| `watch_ohlcv(symbol, timeframe)` | Real-time OHLCV streams | Data streaming handlers |
| `watch_orderbook(symbol, limit)` | Real-time order book | Order book subscriptions |
| `watch_trades(symbol)` | Real-time trade feed | Trade data subscriptions |
| `watch_ticker(symbol)` | Real-time ticker updates | Quote generation |
| `fetch_funding_rates(symbols)` | Current funding rates | Funding rate data |
| `fetch_funding_rate_history(symbol, since, limit)` | Historical funding rates | `src/qubx/connectors/ccxt/reader.py:336` |
| `watch_funding_rates(symbols)` | Real-time funding rate updates | Custom polling implementation |

### Exchange Information

| Method | Purpose | Usage in Qubx |
|--------|---------|---------------|
| `load_markets()` | Load market definitions | `src/qubx/connectors/ccxt/account.py:68` |
| `find_timeframe(timeframe)` | Validate timeframe support | `src/qubx/connectors/ccxt/data.py:219` |

### Custom Implementations

For exchanges with limited WebSocket support, Qubx provides custom implementations:

#### HyperLiquid Funding Rate Streaming
```python
# Located in: src/qubx/connectors/ccxt/exchanges/hyperliquid/hyperliquid.py
async def watch_funding_rates(self, symbols, params=None):
    # Uses PollingToWebSocketAdapter for funding rate updates
    # Converts polling fetch_funding_rates to WebSocket-like interface
```

#### Extended OHLCV Format
```python
def parse_ohlcv(self, ohlcv, market=None):
    # Returns 10-field format: [timestamp, open, high, low, close, volume, 
    # volume_quote, trade_count, bought_volume, bought_volume_quote]
```

#### Enhanced Order Parsing
```python
def parse_order(self, order, market=None):
    # Handles HyperLiquid-specific order format:
    # - Maps "limitPx" presence to limit/market order type
    # - Converts side "B"/"S" to "buy"/"sell"
    # - Extracts amount from "sz" or "origSz" fields
    # - Maps HyperLiquid status to standard CCXT status
```

#### Enhanced Trade Parsing  
```python
def parse_trade(self, trade, market=None):
    # Handles HyperLiquid-specific trade format:
    # - Maps side "B"/"S" to "buy"/"sell"
    # - Extracts price from "px" field
    # - Extracts amount from "sz" field
    # - Maps trade ID from "tid" field
```

## Account Processor Requirements

### Balance & Position Management

| Method | Purpose | Usage in Qubx |
|--------|---------|---------------|
| `fetch_balance()` | Account balances | `src/qubx/connectors/ccxt/account.py:270` |
| `fetch_positions()` | Open positions | `src/qubx/connectors/ccxt/account.py:295` |
| `fetch_tickers(symbols)` | Ticker prices for portfolio valuation | `src/qubx/connectors/ccxt/account.py:361` |

### Order Management

| Method | Purpose | Usage in Qubx |
|--------|---------|---------------|
| `fetch_orders(symbol, since, limit)` | Order history | `src/qubx/connectors/ccxt/account.py:478` |
| `fetch_open_orders(symbol)` | Active orders | `src/qubx/connectors/ccxt/account.py:478` |
| `fetch_my_trades(symbol, since)` | Trade/execution history | `src/qubx/connectors/ccxt/account.py:487` |
| `watch_orders()` | Real-time order updates | `src/qubx/connectors/ccxt/account.py:533` |
| `cancel_order(id, symbol)` | Cancel specific order | `src/qubx/connectors/ccxt/account.py:460` |

### Account Processor Integration Points

```python
# Balance updates every 30 seconds (configurable)
await self.exchange.fetch_balance()

# Position updates every 30 seconds (configurable)  
await self.exchange.fetch_positions()

# Order execution monitoring
async def _watch_executions():
    exec = await self.exchange.watch_orders()
    # Process order updates and extract deals
```

## Trading Broker Requirements

### Order Execution

| Method | Purpose | Usage in Qubx |
|--------|---------|---------------|
| `create_order(symbol, type, side, amount, price, params)` | Place orders | `src/qubx/connectors/ccxt/broker.py:260` |
| `create_order_ws(...)`* | WebSocket order placement | `src/qubx/connectors/ccxt/broker.py:258` |
| `cancel_order(id, symbol)` | Cancel orders | `src/qubx/connectors/ccxt/broker.py:362` |
| `cancel_order_ws(id, symbol)`* | WebSocket order cancellation | `src/qubx/connectors/ccxt/broker.py:360` |

*Optional WebSocket methods for faster execution

### Supported Order Types

| Order Type | CCXT Parameters | Purpose |
|------------|-----------------|---------|
| Market | `type: "market"` | Immediate execution |
| Limit | `type: "limit"` | Price-specified execution |
| Stop Market | `type: "market"`, `params.triggerPrice` | Stop-loss orders |
| Stop Limit | `type: "limit"`, `params.triggerPrice` | Stop orders with limit price |

### Time-in-Force Options

| TIF | Parameter | Behavior |
|-----|-----------|----------|
| GTC | `params.timeInForce: "GTC"` | Good till canceled |
| GTX | `params.timeInForce: "GTX"` | Good till crossing (post-only) |
| IOC | `params.timeInForce: "IOC"` | Immediate or cancel |
| FOK | `params.timeInForce: "FOK"` | Fill or kill |

### Order Parameters

```python
{
    "symbol": ccxt_symbol,
    "type": order_type.lower(),  # "market", "limit"
    "side": order_side.lower(),  # "buy", "sell"
    "amount": amount,
    "price": price,  # Required for limit orders
    "params": {
        "newClientOrderId": client_id,  # Custom order ID
        "timeInForce": "GTC",          # Time in force
        "triggerPrice": price,         # For stop orders
        "reduceOnly": False,           # Position reduction only
        "type": "swap"                 # For futures contracts
    }
}
```

### Error Handling

The broker handles various CCXT exceptions:

```python
# Account-specific errors
ccxt.InsufficientFunds() -> ErrorLevel.HIGH
ccxt.OrderNotFillable() -> ErrorLevel.LOW  
ccxt.InvalidOrder() -> ErrorLevel.LOW
ccxt.BadRequest() -> ErrorLevel.LOW

# Network/Exchange errors (with retry logic)
ccxt.NetworkError()
ccxt.ExchangeError() 
ccxt.ExchangeNotAvailable()
ccxt.OperationRejected()
```

## Implementation Guidelines

### 1. Market Data Streaming Priority
- Implement `watch_ohlcv()` for real-time price feeds
- Implement `watch_orderbook()` for order book data
- Provide `watch_funding_rates()` or use polling adapter

### 2. Account Management Priority  
- Ensure `fetch_balance()` returns complete balance information
- Implement `fetch_positions()` for futures/margin accounts
- Provide reliable `watch_orders()` for execution monitoring

### 3. Order Execution Priority
- Support standard order types (market, limit, stop)
- Implement proper error handling and retry logic
- Provide WebSocket order methods for low-latency execution

### 4. Exchange-Specific Features

For HyperLiquid specifically:
- Custom funding rate polling via `PollingToWebSocketAdapter`
- Extended OHLCV parsing with trade count data
- Proper handling of settlement currencies and futures contracts

### 5. Testing Integration

Ensure the following integration points work correctly:
- Market loading and symbol resolution
- Real-time data streaming without disconnections
- Order placement and cancellation under various market conditions
- Account state synchronization during high-frequency updates

## References

- Core interfaces: `src/qubx/core/interfaces.py`
- CCXT data provider: `src/qubx/connectors/ccxt/data.py`
- CCXT account processor: `src/qubx/connectors/ccxt/account.py`  
- CCXT broker: `src/qubx/connectors/ccxt/broker.py`
- HyperLiquid implementation: `src/qubx/connectors/ccxt/exchanges/hyperliquid/`