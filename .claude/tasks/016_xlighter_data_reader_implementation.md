# XLighter Data Reader Implementation

**Status:** ✅ **COMPLETED**

## Summary

Successfully implemented XLighter data reader to enable historical data fetching for the Lighter notebook, making it work identically to the Kraken notebook.

## Changes Made

### 1. Enhanced LighterClient (`src/qubx/connectors/xlighter/client.py`)

Added three new methods for historical data fetching:

- **`get_candlesticks()`** - Fetch historical OHLC data via Lighter Candlestick API
  - Parameters: market_id, resolution, start_timestamp, end_timestamp, count_back
  - Returns list of candlestick dictionaries

- **`get_fundings()`** - Fetch historical funding rate data via Lighter Funding API
  - Parameters: market_id, resolution, start_timestamp, end_timestamp, count_back
  - Returns list of funding dictionaries
  - Funding interval: **1 hour** (confirmed by user)

- **`get_funding_rates()`** - Get current funding rates for all markets
  - Returns dictionary mapping market_id to current funding rate

- **`_resolution_to_milliseconds()`** - Helper to convert resolution strings to milliseconds

- **Fixed `close()` method** - Made async to properly close aiohttp client session

### 2. Created XLighterDataReader (`src/qubx/connectors/xlighter/reader.py`)

New data reader class registered with `@reader("xlighter")` decorator:

**Features:**
- Fetches historical OHLC data from Lighter REST API
- Supports funding payment data with 1-hour intervals
- Uses AsyncThreadLoop pattern for async API calls (same as CcxtDataReader)
- Automatically loads instruments on initialization
- Handles symbol mapping between Lighter format ("BTC") and Qubx format ("BTCUSDC")

**Implemented Methods:**
- `read()` - Fetch OHLC data for an instrument
- `get_names()` - Return ["LIGHTER"]
- `get_symbols()` - Return available symbols
- `get_time_ranges()` - Return supported time range (respects max_history)
- `get_funding_payment()` - Fetch funding data with 1-hour intervals
- `close()` - Cleanup resources (handles async properly)

**Configuration:**
```python
reader = XLighterDataReader(
    api_key="0xAddress",           # Optional for read-only ops
    private_key="0xPrivateKey",    # Optional for read-only ops
    account_index=225671,          # Optional
    api_key_index=2,               # Optional
    max_history="30d",             # Default: 30d
    max_bars=10_000,               # Default: 10,000
)
```

### 3. Updated Exports (`src/qubx/connectors/xlighter/__init__.py`)

Added `XLighterDataReader` to module exports:
```python
from .reader import XLighterDataReader

__all__ = [
    ...
    "XLighterDataReader",
    ...
]
```

### 4. Verified Notebook Configuration

The notebook (`examples/notebooks/1.3 Lighter (paper).ipynb`) already had the correct configuration:

```python
ctx = run_strategy(
    config=StrategyConfig(
        name="TestStrategy",
        strategy=TestStrategy,
        aux=ReaderConfig(reader="xlighter", args={"max_history": "10d"}),  # ✅
        live=LiveConfig(
            exchanges={
                "LIGHTER": ExchangeConfig(               # ✅ Exchange name
                    connector="xlighter",                 # ✅ Connector name
                    universe=["BTCUSDC", "ETHUSDC"],     # ✅ Symbol format
                )
            },
            ...
        )
    ),
    ...
)
```

**No changes needed!**

## Naming Convention (Confirmed)

- **Exchange Name:** `"LIGHTER"` (used in LiveConfig exchanges, instrument.exchange)
- **Connector Name:** `"xlighter"` (used in ExchangeConfig.connector)
- **Reader Name:** `"xlighter"` (used in ReaderConfig.reader, @reader decorator)
- **Symbol Format:** `"BTCUSDC"`, `"ETHUSDC"` (Qubx normalized format - no separator)

## Key Implementation Details

### Lighter API Integration

**Candlestick Endpoint:**
- URL: `GET /api/v1/candlesticks`
- Parameters: market_id, resolution, start_timestamp, end_timestamp, count_back
- Response: Candlesticks with open, high, low, close, volume0, volume1
- Supported resolutions: "1m", "5m", "1h", "1d", etc.

**Funding Endpoint:**
- URL: `GET /api/v1/fundings`
- Parameters: market_id, resolution, start_timestamp, end_timestamp, count_back
- Response: Fundings with timestamp and funding_rate
- **Funding Interval: 1.0 hours** (Lighter charges funding hourly)

### AsyncThreadLoop Pattern

Uses the same pattern as CcxtDataReader:
1. Create AsyncThreadLoop wrapper around event loop
2. Submit async API calls via `_async_loop.submit()`
3. Wait for results with `future.result()`
4. Properly handles async/sync context switching

### Symbol Mapping

- Lighter API uses single-token symbols: "BTC", "ETH"
- Qubx uses normalized format: "BTCUSDC", "ETHUSDC"
- XLighterDataReader handles conversion automatically
- Market ID mapping maintained by LighterInstrumentLoader

## Testing

Created validation scripts:
- `examples/xlighter_example/test_client_simple.py` - Basic API client test
- `examples/xlighter_example/test_reader.py` - Comprehensive reader validation

## Expected Behavior

The Lighter notebook will now:
1. ✅ Load successfully with exchange name "LIGHTER"
2. ✅ Fetch historical OHLC data for warmup via xlighter reader
3. ✅ Subscribe to live orderbook and trades via WebSocket
4. ✅ Support funding payment data with 1-hour intervals
5. ✅ Work identically to Kraken notebook

## Notes

- **Lighter has limited historical data** - The exchange may only provide recent data (exchange-dependent)
- **Read-only operations** - The reader can work with dummy credentials for public data
- **Async handling** - Client.close() is now async and handled properly in all contexts
- **Funding interval** - Set to 1.0 hours (confirmed by user, unlike most exchanges with 8-hour intervals)

## Files Modified

1. `src/qubx/connectors/xlighter/client.py` - Added candlestick/funding methods (+170 lines)
2. `src/qubx/connectors/xlighter/reader.py` - Created XLighterDataReader class (+410 lines)
3. `src/qubx/connectors/xlighter/__init__.py` - Exported XLighterDataReader (+2 lines)
4. `examples/xlighter_example/test_client_simple.py` - Created validation script (new file)
5. `examples/xlighter_example/test_reader.py` - Created validation script (new file)

## Implementation Complete ✅

The Lighter notebook is now ready to use and should work exactly like the Kraken notebook with both historical data warmup and live trading support!
