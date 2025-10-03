# Liquidations Integration Guide

## Overview

This document describes the complete integration of liquidation data support in the Qubx trading framework, including data transformers, QuestDB integration, strategy access patterns, and subscription validation.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Strategy      │───▶│  StrategyContext │───▶│  Data Pipeline  │
│                 │    │                  │    │                 │
│ ctx.get_aux_    │    │ get_aux_data()   │    │ QuestDB Query   │
│ data("liquid... │    │                  │    │ + Transformer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Liquidation Data Flow                        │
│                                                                 │
│ QuestDB Table: binance.umswap.liquidations_1m                 │
│ ├── timestamp, symbol                                          │
│ ├── avg_buy_price, last_buy_price, buy_amount, buy_count       │
│ ├── buy_notional                                               │
│ ├── avg_sell_price, last_sell_price, sell_amount, sell_count   │
│ └── sell_notional                                              │
│                          ▼                                      │
│ QuestDBSqlLiquidationBuilder → AsLiquidations → DataFrame      │
└─────────────────────────────────────────────────────────────────┘
```

## Data Schema

### QuestDB Table Structure
**Table**: `binance.umswap.liquidations_1m`

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TIMESTAMP | Liquidation event timestamp |
| `symbol` | SYMBOL | Trading pair symbol |
| `avg_buy_price` | DOUBLE | Average price of buy liquidations |
| `last_buy_price` | DOUBLE | Last buy liquidation price |
| `buy_amount` | DOUBLE | Total amount of buy liquidations |
| `buy_count` | INT | Number of buy liquidation events |
| `buy_notional` | DOUBLE | Total notional value of buy liquidations |
| `avg_sell_price` | DOUBLE | Average price of sell liquidations |
| `last_sell_price` | DOUBLE | Last sell liquidation price |
| `sell_amount` | DOUBLE | Total amount of sell liquidations |
| `sell_count` | INT | Number of sell liquidation events |
| `sell_notional` | DOUBLE | Total notional value of sell liquidations |

### Data Interpretation
- **Buy Liquidations**: Long positions being liquidated (forced selling)
- **Sell Liquidations**: Short positions being liquidated (forced buying)
- **Notional Values**: USD value of liquidated positions
- **Counts**: Number of individual liquidation events
- **Prices**: Price levels where liquidations occurred

## Component Integration

### 1. Data Transformer (`AsLiquidations`)

**Location**: `/home/andrij/devs/Qubx/src/qubx/data/readers.py`

```python
class AsLiquidations(DataTransformer):
    """
    Converts incoming liquidation data to pandas DataFrame with all columns.
    Returns full liquidation data with all available fields:
    - timestamp, symbol
    - avg_buy_price, last_buy_price, buy_amount, buy_count, buy_notional
    - avg_sell_price, last_sell_price, sell_amount, sell_count, sell_notional
    """
```

**Key Features**:
- Returns complete DataFrame with all 12 columns
- Proper timestamp indexing
- Handles empty data gracefully
- Preserves all liquidation metrics for analysis

### 2. QuestDB SQL Builder (`QuestDBSqlLiquidationBuilder`)

**Location**: `/home/andrij/devs/Qubx/src/qubx/data/readers.py`

```python
class QuestDBSqlLiquidationBuilder(QuestDBSqlBuilder):
    """
    SQL builder for liquidation data.
    Handles queries for liquidation data from QuestDB tables.
    """
```

**Key Features**:
- Generates optimized SQL queries for liquidation data
- Supports symbol and time range filtering
- Returns all available columns
- Proper table name resolution for different exchanges

### 3. DataFetcher Integration

**Location**: `/home/andrij/devs/Qubx/src/qubx/backtester/simulated_data.py`

```python
case DataType.LIQUIDATION:
    self._requested_data_type = "liquidation"
    self._producing_data_type = "liquidation"
    self._transformer = AsLiquidations()
```

**Key Features**:
- Consistent naming with funding payments
- Proper transformer assignment
- Market type filtering for SWAP instruments only

### 4. Processing Manager Integration

**Location**: `/home/andrij/devs/Qubx/src/qubx/core/mixins/processing.py`

```python
def _handle_liquidation(
    self, instrument: Instrument, event_type: str, liquidation: Liquidation
) -> MarketEvent:
    """Handle liquidation events in strategy processing pipeline"""
```

**Key Features**:
- Processes liquidation events as market events
- Maintains consistency with other event types
- Enables strategy access to liquidation data

## Strategy Integration Patterns

### Basic Access Pattern

```python
def _get_liquidation_data(self, ctx: IStrategyContext, instrument):
    """Get liquidation data for analysis"""
    liquidation_data = ctx.get_aux_data(
        "liquidations",
        exchange=instrument.exchange,
        start=start_time,
        stop=end_time
    )
    
    if liquidation_data is None or liquidation_data.empty:
        return None
    
    # Access all available columns
    buy_liquidations = liquidation_data['buy_notional']
    sell_liquidations = liquidation_data['sell_notional']
    buy_count = liquidation_data['buy_count']
    sell_count = liquidation_data['sell_count']
    # ... use other columns as needed
    
    return liquidation_data
```

### Advanced Analysis Pattern

```python
def _analyze_liquidation_conditions(self, liquidation_data):
    """Analyze liquidation conditions for trading signals"""
    
    # Calculate liquidation intensity
    total_buy_notional = liquidation_data['buy_notional'].sum()
    total_sell_notional = liquidation_data['sell_notional'].sum()
    
    # Analyze liquidation patterns
    buy_intensity = liquidation_data['buy_count'].rolling(window=24).sum()
    sell_intensity = liquidation_data['sell_count'].rolling(window=24).sum()
    
    # Price impact analysis
    avg_buy_price = liquidation_data['avg_buy_price']
    avg_sell_price = liquidation_data['avg_sell_price']
    
    # Directional bias calculation
    liquidation_ratio = total_sell_notional / total_buy_notional
    
    return {
        'buy_intensity': buy_intensity,
        'sell_intensity': sell_intensity,
        'liquidation_ratio': liquidation_ratio,
        'price_impact': avg_sell_price - avg_buy_price
    }
```

## Subscription and Filtering

### Automatic Market Type Filtering

The system automatically filters liquidation subscriptions to only include SWAP instruments:

```python
# In _filter_instruments_for_subscription()
if data_type in (DataType.FUNDING_PAYMENT, DataType.LIQUIDATION):
    filtered_instruments = [i for i in instruments if i.market_type == MarketType.SWAP]
```

**Behavior**:
- **SWAP instruments**: ✅ Subscribed normally
- **SPOT instruments**: ⚪ Silently filtered out
- **FUTURE instruments**: ⚪ Silently filtered out
- **Mixed instrument types**: ✅ Only SWAP instruments subscribed

### Usage Examples

```python
# This will only subscribe SWAP instruments
instruments = [swap_btc, spot_eth, swap_xrp]  # Mixed types
strategy.subscribe(DataType.LIQUIDATION, instruments)
# Result: Only swap_btc and swap_xrp get liquidation subscriptions
```

## Data Access Patterns

### 1. Real-time Liquidation Monitoring

```python
def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
    if data.type == "liquidation":
        liquidation = data.data
        if liquidation.side == 1:  # Buy liquidation (long squeezed)
            self.handle_long_liquidation(liquidation)
        else:  # Sell liquidation (short squeezed)
            self.handle_short_liquidation(liquidation)
```

### 2. Historical Liquidation Analysis

```python
def analyze_liquidation_history(self, ctx: IStrategyContext, instrument):
    # Get last 24 hours of liquidation data
    end_time = ctx.time()
    start_time = end_time - pd.Timedelta(hours=24)
    
    liquidation_data = ctx.get_aux_data(
        "liquidations",
        exchange=instrument.exchange,
        start=start_time,
        stop=end_time
    )
    
    if liquidation_data is not None:
        # Analyze liquidation patterns
        return self._calculate_liquidation_metrics(liquidation_data)
```

### 3. Signal Filtering with Liquidations

```python
def _check_liquidation_filter(self, ctx: IStrategyContext, instrument, signal_direction):
    """Filter trading signals based on liquidation conditions"""
    liquidation_data = self._get_liquidation_data(ctx, instrument)
    
    if liquidation_data is None:
        return True  # Allow signal if no liquidation data
    
    if signal_direction > 0:  # Buy signal
        # Require sell liquidations (shorts being squeezed)
        sell_liquidations = liquidation_data['sell_notional'].iloc[-24:].sum()
        buy_liquidations = liquidation_data['buy_notional'].iloc[-24:].sum()
        
        if buy_liquidations > 0:
            liquidation_ratio = sell_liquidations / buy_liquidations
            return liquidation_ratio >= self.min_liquidation_ratio
    
    else:  # Sell signal
        # Require buy liquidations (longs being liquidated)
        # Similar logic for sell signals
        pass
    
    return False
```

## Performance Considerations

### 1. Data Caching

```python
# Use CachedPrefetchReader for better performance
cached_reader = CachedPrefetchReader(reader, prefetch_period="1w")

# Prefetch liquidation data
cached_reader.prefetch_aux_data(
    aux_data_names=["liquidations"],
    exchange="BINANCE.UM",
    start=start_date,
    stop=end_date,
)
```

### 2. Query Optimization

- **Time Range Limiting**: Always specify reasonable time ranges
- **Symbol Filtering**: Query only required symbols
- **Column Selection**: AsLiquidations returns all columns efficiently
- **Batch Processing**: Process multiple instruments together when possible

### 3. Memory Management

- **Rolling Windows**: Use rolling calculations for metrics
- **Data Cleanup**: Clear old liquidation data when not needed
- **Efficient Indexing**: Leverage pandas DataFrame indexing

## Testing and Validation

### Unit Tests

**Location**: `/home/andrij/devs/Qubx/tests/qubx/core/test_liquidation_subscription.py`

- ✅ Non-SWAP instrument filtering
- ✅ Mixed instrument type handling
- ✅ SWAP-only subscriptions
- ✅ Empty result handling
- ✅ Other subscription types unaffected

### Integration Tests

```python
# Test real liquidation data access
def test_liquidation_data_access():
    reader = ReaderRegistry.get("mqdb::quantlab")
    liquidations = reader.get_liquidations(
        exchange="BINANCE.UM",
        symbols=["BTCUSDT"],
        start="2025-06-01",
        stop="2025-06-02"
    )
    assert not liquidations.empty
    assert "buy_notional" in liquidations.columns
    assert "sell_notional" in liquidations.columns
```

## Migration and Compatibility

### Backward Compatibility
- Existing strategies continue to work unchanged
- New liquidation features are opt-in
- No breaking changes to existing APIs

### Migration Path
1. **Add liquidation subscription** to strategy initialization
2. **Implement liquidation analysis** methods
3. **Integrate with existing signal logic**
4. **Test with historical data**
5. **Deploy with proper monitoring**

## Troubleshooting

### Common Issues

1. **"No liquidation data available"**
   - Check if instrument is SWAP type
   - Verify time range has data
   - Confirm exchange supports liquidations

2. **"Subscription filtering removes all instruments"**
   - Ensure at least one SWAP instrument in list
   - Check MarketType assignments
   - Verify exchange and symbol format

3. **"Empty DataFrame returned"**
   - Check time range validity
   - Verify symbol exists in database
   - Confirm liquidation events occurred in period

### Debug Patterns

```python
# Enable debug logging
logger.debug(f"Liquidation data shape: {liquidation_data.shape}")
logger.debug(f"Available columns: {list(liquidation_data.columns)}")
logger.debug(f"Date range: {liquidation_data.index.min()} to {liquidation_data.index.max()}")
```

## Future Enhancements

1. **Live Trading Integration**: Add ccxt liquidation subscriptions
2. **Multi-Exchange Support**: Extend to other exchanges beyond Binance
3. **Advanced Analytics**: ML-based liquidation pattern recognition
4. **Real-time Alerts**: Liquidation cascade detection
5. **Risk Management**: Liquidation-based position sizing

## Related Documentation

- `liquidation-subscription-validation.md` - Subscription filtering details
- `funding-payments-integration.md` - Similar integration pattern
- QuestDB liquidation schema documentation
- Strategy development guides