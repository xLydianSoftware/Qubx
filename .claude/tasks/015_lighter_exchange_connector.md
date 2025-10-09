# Task 015: Lighter Exchange Connector Implementation

**Created**: 2025-10-09
**Status**: In Progress
**Priority**: High

## Overview
Implement a comprehensive Lighter exchange connector for Qubx framework with reusable WebSocket infrastructure, following established patterns from CCXT connector.

## Objectives
1. Create reusable base WebSocket manager for future exchanges
2. Implement Lighter connector with maximum WebSocket usage
3. Support custom account configuration fields (account_index, api_key_index)
4. Register Lighter instruments in Qubx instrument lookup
5. Comprehensive unit and integration tests

## Architecture Components

### 1. Account Configuration Enhancement
**Files**: `/src/qubx/utils/runner/accounts.py`

**Changes**:
- Make `ExchangeCredentials` accept extra fields via Pydantic `model_config`
- Support Lighter-specific: `account_index`, `api_key_index`, `private_key`
- Maintain backward compatibility

**Test Config**: `/home/yuriy/accounts/xlydian1_lighter.toml`

---

### 2. Base WebSocket Manager (Reusable)
**Location**: `/src/qubx/utils/websocket_manager.py`

**Features**:
- Generic connection handling with reconnection logic
- Channel multiplexing support
- Exponential backoff retry (1s, 2s, 4s, ..., max 60s)
- Event-based subscription system
- Thread-safe operations
- Graceful shutdown

**Tests**: `/tests/qubx/utils/test_websocket_manager.py`
- Mock WebSocket server tests
- Reconnection scenarios
- Multiplexing tests

---

### 3. Lighter Connector Module
**Location**: `/src/qubx/connectors/lighter/`

#### File Structure:
```
lighter/
├── __init__.py                 # Public API
├── client.py                   # LighterClient (SDK wrapper)
├── websocket.py                # LighterWebSocketManager
├── data.py                     # LighterDataProvider (IDataProvider)
├── broker.py                   # LighterBroker (IBroker)
├── account.py                  # LighterAccountProcessor (IAccountProcessor)
├── factory.py                  # Factory for creating components
├── utils.py                    # Conversion utilities
├── constants.py                # Constants and enums
└── handlers/                   # Data handlers
    ├── __init__.py
    ├── base.py
    ├── orderbook.py
    ├── trades.py
    ├── quote.py
    ├── account.py
    └── factory.py
```

---

### 4. Component Specifications

#### **LighterClient** (`client.py`)
- Wrap `lighter.SignerClient` and `lighter.ApiClient`
- Initialize account (account_index, API keys)
- Provide REST methods (ticker metadata, candlesticks, historical data)
- Handle authentication and signing

**Tests**: Test initialization, REST API calls, error handling

#### **LighterWebSocketManager** (`websocket.py`)
- Extends `BaseWebSocketManager`
- Lighter-specific channels:
  - `order_book/{market_id}` - L2 orderbook
  - `trade/{market_id}` - Trade feed
  - `market_stats/{market_id}` or `all` - Market stats
  - `account_all/{account_id}` - Account updates
  - `user_stats/{account_id}` - User stats
  - `executed_transaction` - Global fills
- Handle subscription protocol: `{"type": "subscribe", "channel": "..."}`
- Reconnection with state restoration

**Tests**: Integration tests with Lighter mainnet (read-only channels)

#### **LighterDataProvider** (`data.py`)
- Implements `IDataProvider`
- Subscription management (orderbook, trades, quotes)
- Delegate to `LighterWebSocketManager`
- Use handlers for data transformation
- Warmup via REST API

**Tests**: Mock WebSocket, test subscriptions, warmup

#### **LighterBroker** (`broker.py`)
- Implements `IBroker`
- Order operations via WebSocket (preferred) or REST (fallback)
- Order tracking and sync
- Error handling with retry

**Tests**: Mock tests for order operations

#### **LighterAccountProcessor** (`account.py`)
- Implements `IAccountProcessor`
- Track positions, balances, orders via WS
- Process execution reports (fills)
- Update from `account_all`, `user_stats` channels
- Funding payments

**Tests**: Mock WS feed, verify state updates

#### **Data Handlers** (`handlers/*.py`)
- Transform Lighter format → Qubx format
- `OrderbookHandler`: Process L2 updates
- `TradesHandler`: Process trades
- `QuoteHandler`: Derive quotes from orderbook
- `AccountHandler`: Process account updates

**Tests**: Unit tests for each handler with sample data

---

### 5. Instrument Management

#### Create Lighter Instrument Loader
**Location**: `/src/qubx/connectors/lighter/instruments.py`

**Features**:
- Query Lighter API for all markets (`/api/v1/orderBooks`)
- Parse market metadata (symbol, precision, min_base, min_quote)
- Convert to Qubx `Instrument` objects
- Save to `~/.qubx/instruments/lighter.json`

**Format**:
```python
# Lighter: BTC-USDC (market_id: 0)
# Qubx: LIGHTER:SWAP:BTC-USDC
Instrument(
    symbol="BTC-USDC",
    asset_type=AssetType.CRYPTO,
    market_type=MarketType.SWAP,
    exchange="LIGHTER",
    base="BTC",
    quote="USDC",
    settle="USDC",
    exchange_symbol="BTC-USDC",
    tick_size=0.01,  # from supported_price_decimals
    lot_size=0.001,  # from supported_size_decimals
    min_size=0.001,  # from min_base_amount
    min_notional=5.0,  # from min_quote_amount
    ...
)
```

**Mapping Dicts**:
```python
ticker_to_market_id: dict[str, int]  # "BTC-USDC" → 0
market_id_to_ticker: dict[int, str]  # 0 → "BTC-USDC"
```

**CLI Command** (optional):
```bash
poetry run qubx refresh-instruments --exchange LIGHTER
```

**Tests**: Test fetching, parsing, saving instruments

---

### 6. WebSocket Design Patterns

#### Channel Multiplexing:
```python
MULTIPLEX_CHANNELS = {
    'order_book',    # Multiple market_ids
    'trade',         # Multiple market_ids
    'market_stats',  # Multiple market_ids or 'all'
}

DEDICATED_CHANNELS = {
    'account_all',           # One per account
    'user_stats',            # One per account
    'executed_transaction',  # Global
}
```

#### Reconnection Strategy:
- Exponential backoff: 1s → 2s → 4s → 8s → 16s → 32s → 60s (cap)
- Max retries: 10 (configurable)
- Auto resubscribe on reconnect
- Preserve subscription state

---

### 7. Configuration Example

```yaml
# config.yml
strategy: my_strategy.MyStrategy
parameters:
  foo: bar

live:
  exchanges:
    LIGHTER:
      connector: lighter
      universe:
        - BTC-USDC
        - ETH-USDC

  logging:
    logger: InMemoryLogsWriter
```

```toml
# accounts.toml
[[accounts]]
name = "xlydian1-lighter"
exchange = "LIGHTER"
api_key = "0xYOUR_ADDRESS"
secret = "0xYOUR_PRIVATE_KEY"
account_index = 225671
api_key_index = 2
base_currency = "USDC"
```

---

## Implementation Phases

### Phase 1: Foundation (Testing Ready) ✅ COMPLETED
- [x] Create task file
- [x] Update `ExchangeCredentials` to accept extra fields
  - [x] Write tests for account config (9 tests passing)
  - [x] Support for `account_index`, `api_key_index`, and custom fields
- [x] Create `BaseWebSocketManager`
  - [x] Mock WebSocket tests (14 tests passing)
  - [x] Reconnection tests with exponential backoff
  - [x] Channel multiplexing support
- [x] Create Lighter connector structure
  - [x] Constants and enums
  - [x] Utility functions (15 tests passing)
  - [x] LighterWebSocketManager
  - [x] LighterClient (SDK wrapper)
  - [x] LighterInstrumentLoader
  - [x] Fixed namespace collision with lighter SDK
- [ ] Test Lighter instrument loader with live API (ready for integration tests)

### Phase 2: Core Connector (Integration Tests)
- [ ] Implement `LighterClient`
  - [ ] Unit tests for REST methods
  - [ ] Integration test: fetch orderbooks
- [ ] Implement `LighterWebSocketManager`
  - [ ] Integration test: subscribe to orderbook
  - [ ] Integration test: subscribe to trades
  - [ ] Test reconnection behavior
- [ ] Implement data handlers
  - [ ] Unit tests with sample JSON

### Phase 3: Data Provider
- [ ] Implement `LighterDataProvider`
  - [ ] Mock WS tests
  - [ ] Test subscription lifecycle
  - [ ] Test warmup
- [ ] Integration test with live data
  - [ ] Subscribe to BTC-USDC orderbook
  - [ ] Subscribe to ETH-USDC trades
  - [ ] Verify data flow through channel

### Phase 4: Broker & Account
- [ ] Implement `LighterBroker`
  - [ ] Mock tests for orders
  - [ ] Test error handling
- [ ] Implement `LighterAccountProcessor`
  - [ ] Mock tests for position tracking
  - [ ] Test fill processing
- [ ] Integration tests (requires testnet or caution)
  - [ ] Place test order (small size)
  - [ ] Cancel order
  - [ ] Verify account updates

### Phase 5: Integration & Documentation
- [ ] Create `LighterFactory`
- [ ] End-to-end test with simple strategy
- [ ] Update documentation
- [ ] Add example strategy

---

## Testing Strategy

### Unit Tests
- Mock all external dependencies
- Test error scenarios
- Test edge cases (empty data, malformed JSON)

### Integration Tests
- Use Lighter mainnet for read-only operations
- Mark tests with `@pytest.mark.integration`
- Require environment variable: `LIGHTER_INTEGRATION_TESTS=1`
- Use small sizes for order tests (if any)

### Test Files:
```
tests/qubx/
├── utils/
│   └── test_websocket_manager.py
├── connectors/
│   └── lighter/
│       ├── test_client.py
│       ├── test_websocket.py
│       ├── test_data.py
│       ├── test_broker.py
│       ├── test_account.py
│       ├── test_instruments.py
│       └── handlers/
│           ├── test_orderbook.py
│           ├── test_trades.py
│           └── test_quote.py
└── integration/
    └── lighter/
        ├── test_websocket_live.py
        └── test_data_provider_live.py
```

---

## Dependencies

**New**:
- `websockets` (already in Qubx)
- `lighter-python` SDK (from `/home/yuriy/devs/lighter-python`)

**Existing**:
- CCXT patterns and utilities
- Qubx core interfaces

---

## Success Criteria

1. ✅ Account config supports Lighter-specific fields
2. ✅ Base WebSocket manager works with mock server
3. ✅ Lighter instruments loaded and queryable
4. ✅ Can subscribe to orderbook and trades via WebSocket
5. ✅ Data flows through handlers correctly
6. ✅ Can place and cancel orders (integration test)
7. ✅ Account state tracked correctly
8. ✅ Comprehensive test coverage (>80%)
9. ✅ Simple strategy runs successfully

---

## Notes

- SDK location: `/home/yuriy/devs/lighter-python`
- Reference code: `/home/yuriy/devs/quantpylib/quantpylib/wrappers/lighter.py`
- Account config: `/home/yuriy/accounts/xlydian1_lighter.toml`
- Test notebook: `/home/yuriy/projects/xincubator/research/live/lighter/1.0 Lighter setup.ipynb`

---

## Progress Log

### 2025-10-09 - Session 1
- Created task file
- Analyzed existing code and architecture
- Defined component structure and testing strategy

### 2025-10-09 - Session 2: Phase 1 Complete ✅
**Account Configuration:**
- Extended `ExchangeCredentials` with Pydantic `extra="allow"` for custom fields
- Added `get_extra_field()` helper method
- Created comprehensive tests (9 passing)
- Tested with actual Lighter account config

**Base WebSocket Manager:**
- Created reusable `BaseWebSocketManager` class
- Implemented reconnection with exponential backoff (1s → 60s cap)
- Channel multiplexing and subscription management
- Thread-safe operations with asyncio
- Created mock WebSocket server for testing
- 14 tests passing (connection, subscription, reconnection, multiplexing)

**Lighter Connector Structure:**
- Created `/src/qubx/connectors/lighter/` module structure
- `constants.py`: Lighter-specific constants, enums, WebSocket channels
- `utils.py`: Conversion utilities for symbols, prices, sizes, orders, quotes
  - Symbol conversion: "BTC-USDC" ↔ "BTC/USDC:USDC"
  - Order side conversion: "B"/"S" ↔ "BUY"/"SELL"
  - Price/size integer ↔ float conversion
  - OrderBook, Trade, Quote converters
  - 15 tests passing
- `websocket.py`: `LighterWebSocketManager` extending base manager
  - Lighter-specific subscription protocol
  - Helper methods for all channel types
- `client.py`: `LighterClient` wrapping Lighter SDK
  - REST API access (markets, orderbook, account)
  - Order creation/cancellation (placeholders)
  - Fixed namespace collision with `lighter` SDK using `importlib`
- `instruments.py`: `LighterInstrumentLoader` for fetching market metadata
  - Converts Lighter markets to Qubx `Instrument` objects
  - Maintains market_id ↔ symbol mappings

**Tests Created:**
- `/tests/qubx/utils/runner/test_accounts.py` (9 tests ✓)
- `/tests/qubx/utils/test_websocket_manager.py` (14 tests ✓)
- `/tests/qubx/connectors/lighter/test_utils.py` (15 tests ✓)
- `/tests/qubx/connectors/lighter/test_instruments_integration.py` (ready for live testing)

**Issues Resolved:**
- Namespace collision: `qubx.connectors.lighter` vs `lighter` SDK
  - Solution: Used `importlib` with cache clearing
  - Removed `tests/qubx/connectors/lighter/__init__.py` to prevent false package detection
- OrderSide type: Using `Literal["BUY", "SELL"]` strings (not enum)
- Quote constructor: Uses positional args `(time, bid, ask, bid_size, ask_size)`

**Total Tests Passing: 38** (9 + 14 + 15)

**Status:** Phase 1 complete and ready for Phase 2

---

## References

- Lighter API Docs: https://apidocs.lighter.xyz/
- Lighter WebSocket: wss://mainnet.zklighter.elliot.ai/stream
- CCXT Connector: `/src/qubx/connectors/ccxt/`
- Hyperliquid Connector: `/src/qubx/connectors/ccxt/exchanges/hyperliquid/`
