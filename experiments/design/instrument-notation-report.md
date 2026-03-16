# Instrument Notation & Type System - Design Report

## Research Summary

### Industry Approaches

| Platform | Identifier Format | Type Discrimination |
|----------|------------------|---------------------|
| **Interactive Brokers** | `symbol + secType + exchange` | `secType`: STK, FUT, OPT, CASH, CRYPTO, CFD, CMDTY |
| **Alpaca** | `symbol + asset_class` | `asset_class`: us_equity, crypto |
| **Binance** | Separate APIs per segment | `contractType`: PERPETUAL, QUARTERLY |
| **Kraken** | Symbol prefix (`pf_`) | `tag`: perpetual, futures |
| **Hyperliquid** | Coin name / pair | Endpoint determines type |
| **Nautilus Trader** | `SYMBOL.VENUE` | Instrument subclasses (15 types) |

### Key Insight

Most platforms use a combination of:
1. **Venue/Exchange** - where you trade
2. **Product Type** - how you trade (spot, perp, future, option)
3. **Symbol** - what you trade

## Current Qubx State

```python
class Instrument:
    symbol: str
    exchange: str           # e.g., "BINANCE.UM"
    market_type: MarketType # SPOT, MARGIN, SWAP, FUTURE, OPTION
    asset_type: AssetType   # CRYPTO, STOCK, FX... (NEVER USED!)
```

**Notation:** `EXCHANGE:SYMBOL` (2-part)

**Issues:**
- `asset_type` field exists but is never used in any business logic
- No way to explicitly specify `market_type` in string notation
- Ambiguity when same symbol exists as both SWAP and FUTURE

## Proposed Changes

### 1. Extended Notation Format

```
EXCHANGE:SYMBOL              # existing (2-part) - backward compatible
EXCHANGE:MARKET_TYPE:SYMBOL  # extended (3-part) - explicit disambiguation
```

**Examples:**
```
BINANCE.UM:BTCUSDT                # works as today
BINANCE.UM:SWAP:BTCUSDT           # explicit perpetual
BINANCE.UM:FUTURE:BTCUSDT_240329  # explicit dated future
IB:AAPL                           # stock (inferred SPOT)
IB:FUTURE:ESM4                    # explicit future
IB:OPTION:AAPL230217P00155000     # explicit option
```

### 2. MarketType Enum (Updated)

```python
class MarketType(StrEnum):
    """
    Defines HOW you trade an instrument (market mechanism).
    """
    # Cash/Spot markets
    SPOT = "SPOT"           # Direct ownership (stocks, crypto, FX, bonds, commodities)
    MARGIN = "MARGIN"       # Spot with leverage

    # Derivatives
    SWAP = "SWAP"           # Perpetual futures (no expiry)
    FUTURE = "FUTURE"       # Dated futures (with expiry)
    OPTION = "OPTION"       # Options (calls/puts)
    CFD = "CFD"             # Contract for Difference (new)

    # Reference
    INDEX = "INDEX"         # Non-tradable index (new)
```

**Changes:** +2 values (CFD, INDEX) for IB compatibility.

### 3. Deprecate AssetType

| Field | Current | Proposed |
|-------|---------|----------|
| `asset_type` | CRYPTO, STOCK, FX, INDEX, CFD, BOND, CMDTY | **Remove** - never used |
| `market_type` | SPOT, MARGIN, SWAP, FUTURE, OPTION | **Keep + extend** |

**Rationale:** `asset_type` describes WHAT you trade (can be inferred from exchange/context), while `market_type` describes HOW you trade (actively used in business logic).

### 4. Exchange/Venue Mapping

| Venue | Supported MarketTypes |
|-------|----------------------|
| `BINANCE` | SPOT |
| `BINANCE.UM` | SWAP, FUTURE |
| `BINANCE.CM` | SWAP, FUTURE |
| `KRAKEN` | SPOT |
| `KRAKEN.F` | SWAP, FUTURE |
| `HYPERLIQUID.F` | SWAP |
| `IB` | SPOT, FUTURE, OPTION, CFD, INDEX |

## Migration Path

| Phase | Change | Breaking |
|-------|--------|----------|
| 1 | Add 3-part notation parsing | No |
| 2 | Add CFD, INDEX to MarketType | No |
| 3 | Mark `asset_type` as deprecated | No |
| 4 | Remove `asset_type` (major version) | Yes |

## Implementation (Issue #135)

### Files to Modify

1. **`src/qubx/backtester/utils.py`**
   - Add `parse_instrument_notation()` helper
   - Update `_process_single_symbol_or_instrument()` to handle 3-part format
   - Pass `market_type` to `lookup.find_symbol()`

2. **`src/qubx/core/basics.py`**
   - Add `CFD` and `INDEX` to `MarketType` enum

3. **`tests/qubx/backtester/test_instrument_notation.py`**
   - Unit tests for 2-part and 3-part parsing

### Backward Compatibility

- Existing `BINANCE.UM:BTCUSDT` notation continues working
- `market_type` remains optional in 2-part format (inferred by lookup)
- No changes to storage format or existing instrument files
