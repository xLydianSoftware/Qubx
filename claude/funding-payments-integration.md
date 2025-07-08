# Funding Payments Integration for Qubx

## Overview
Integrate funding payments from QuestDB table `binance.umswap.funding_payment` into the Qubx system to properly account for funding costs/earnings in perpetual swap positions. The funding payments will be processed event-driven when funding payment events arrive through the subscription system.

## Key Design Decisions

### Event-Driven Processing
- **Funding Payment Subscription**: Subscribe to funding payment events (not periodic polling)
- **Handler-Based Processing**: Define funding payment handler in processing manager
- **Variable Intervals**: Support different funding payment intervals (8h, 1h, etc.) per exchange/symbol

### Test-Driven Development
- Write comprehensive tests before implementing each phase
- Tests located under `tests/qubx/` following existing structure
- Each phase includes unit tests, integration tests, and validation tests

### Phased Implementation
- Implement in small, verifiable phases
- Each phase must be fully tested and validated before proceeding
- Maintain backward compatibility throughout

## Implementation Phases

### Phase 1: QuestDB Reader Extension (Foundation)
**Goal**: Enable reading funding payment data from QuestDB with comprehensive testing

#### 1.1 Test Implementation
**Location**: `tests/qubx/data/test_funding_payment_reader.py`

**Test Coverage**:
- `test_read_funding_payments_basic()` - Basic reading functionality
- `test_read_funding_payments_with_filters()` - Symbol and time filtering
- `test_read_funding_payments_pagination()` - Large dataset handling
- `test_read_funding_payments_empty_result()` - Empty result handling
- `test_read_funding_payments_invalid_table()` - Error handling
- `test_read_funding_payments_connection_error()` - Connection error handling
- `test_funding_payment_data_structure()` - Data structure validation

**Test Data Setup**:
- Mock QuestDB connection with sample funding payment data
- Test data covering different symbols, time ranges, and funding rates
- Edge cases: missing data, invalid formats, connection failures

#### 1.2 Data Structure Definition
**Location**: `src/qubx/core/basics.py`

```python
@dataclass
class FundingPayment:
    time: dt_64
    symbol: str
    funding_rate: float
    payment_amount: float
    position_size: float
    mark_price: float | None = None
    exchange: str = "binance"
    
    def __post_init__(self):
        # Validation logic
        if self.position_size == 0:
            raise ValueError("Position size cannot be zero")
        if abs(self.funding_rate) > 1.0:
            raise ValueError("Funding rate seems invalid")
```

#### 1.3 QuestDB Reader Extension
**Location**: `src/qubx/data/readers.py`

**New Class**: `FundingPaymentReader(DataReader)`

**Methods**:
- `read_funding_payments(symbols: list[str], start: dt_64, end: dt_64) -> list[FundingPayment]`
- `get_latest_funding_payment(symbol: str) -> FundingPayment | None`
- `get_funding_payment_history(symbol: str, limit: int = 100) -> list[FundingPayment]`

**Features**:
- Support for `binance.umswap.funding_payment` table
- Efficient querying with proper WHERE clauses
- Connection pooling and error handling
- Data validation and conversion

#### 1.4 Registry Updates
**Location**: `src/qubx/data/registry.py`

- Add `DataType.FUNDING_PAYMENT` enum value
- Register `FundingPaymentReader` with registry
- Update `Timestamped` type alias to include `FundingPayment`

#### 1.5 Acceptance Criteria
- [ ] All tests pass
- [ ] Can read funding payments from QuestDB
- [ ] Proper error handling for edge cases
- [ ] Performance acceptable for large datasets
- [ ] Data validation works correctly

### Phase 2: Funding Payment Subscription (Event Processing)
**Goal**: Enable subscription to funding payment events with handler-based processing

#### 2.1 Test Implementation
**Location**: `tests/qubx/core/test_funding_payment_subscription.py`

**Test Coverage**:
- `test_funding_payment_subscription_setup()` - Subscription initialization
- `test_funding_payment_event_processing()` - Event processing flow
- `test_funding_payment_handler_registration()` - Handler registration
- `test_funding_payment_multiple_symbols()` - Multi-symbol handling
- `test_funding_payment_error_handling()` - Error scenarios
- `test_funding_payment_unsubscribe()` - Unsubscription handling

#### 2.2 Data Type Extension
**Location**: `src/qubx/core/basics.py`

- Add `DataType.FUNDING_PAYMENT` to enum
- Update data type mappings and handlers

#### 2.3 Subscription Handler
**Location**: `src/qubx/core/interfaces.py`

**New Interface**: `IFundingPaymentHandler`

```python
class IFundingPaymentHandler:
    def process_funding_payment(self, payment: FundingPayment) -> None:
        """Process a funding payment event"""
        ...
```

#### 2.4 Processing Manager Integration
**Location**: `src/qubx/core/context.py` (or appropriate processing manager)

**New Handler**: `FundingPaymentProcessor`

```python
class FundingPaymentProcessor(IFundingPaymentHandler):
    def __init__(self, account_processor: IAccountProcessor):
        self.account_processor = account_processor
    
    def process_funding_payment(self, payment: FundingPayment) -> None:
        # Process funding payment for positions
        # Update account balances
        # Log funding payment
        pass
```

#### 2.5 Subscription Setup
**Location**: `src/qubx/connectors/ccxt/data.py`

**New Method**: `_subscribe_funding_payment`

```python
async def _subscribe_funding_payment(self, name: str, sub_type: str, channel: CtrlChannel):
    # Subscribe to funding payment events
    # Convert to FundingPayment objects
    # Send through channel for processing
    pass
```

#### 2.6 Acceptance Criteria
- [ ] All tests pass
- [ ] Can subscribe to funding payment events
- [ ] Handler processes events correctly
- [ ] Multiple symbols supported
- [ ] Error handling robust

### Phase 3: Position Enhancement (Core Logic)
**Goal**: Enhance Position class to handle funding payments

#### 3.1 Test Implementation
**Location**: `tests/qubx/core/test_position_funding.py`

**Test Coverage**:
- `test_position_apply_funding_payment()` - Basic funding application
- `test_position_funding_pnl_calculation()` - PnL calculation with funding
- `test_position_funding_payment_history()` - Payment history tracking
- `test_position_funding_edge_cases()` - Edge cases (zero position, etc.)
- `test_position_funding_serialization()` - State persistence

#### 3.2 Position Class Enhancement
**Location**: `src/qubx/core/basics.py`

**New Fields**:
```python
class Position:
    # ... existing fields ...
    funding_pnl: float = 0.0  # cumulative funding payments
    last_funding_time: dt_64 = np.datetime64('NaT')  # last funding payment time
    funding_payments: list[FundingPayment] = field(default_factory=list)
```

**New Methods**:
```python
def apply_funding_payment(self, payment: FundingPayment) -> None:
    """Apply funding payment to position"""
    
def get_funding_pnl(self) -> float:
    """Get cumulative funding PnL"""
    
def get_total_pnl_with_funding(self) -> float:
    """Get total PnL including funding"""
```

#### 3.3 Acceptance Criteria
- [ ] All tests pass
- [ ] Funding payments properly applied to positions
- [ ] PnL calculations include funding
- [ ] Position state maintained correctly

### Phase 4: Account Processor Integration (Business Logic)
**Goal**: Integrate funding payments into account processing

#### 4.1 Test Implementation
**Location**: `tests/qubx/core/test_account_funding.py`

**Test Coverage**:
- `test_account_process_funding_payments()` - Account-level processing
- `test_account_funding_balance_updates()` - Balance updates
- `test_account_funding_multi_position()` - Multiple positions
- `test_account_funding_error_handling()` - Error scenarios

#### 4.2 Account Processor Enhancement
**Location**: `src/qubx/core/account.py`

**New Method**:
```python
def process_funding_payment(self, payment: FundingPayment) -> None:
    """Process funding payment for account"""
    # Find relevant position
    # Apply funding payment
    # Update balances
    # Log transaction
```

#### 4.3 Acceptance Criteria
- [ ] All tests pass
- [ ] Account balances updated correctly
- [ ] Multiple positions handled
- [ ] Error handling robust

### Phase 5: Strategy Context Integration (API)
**Goal**: Enable strategies to access funding payment data

#### 5.1 Test Implementation
**Location**: `tests/qubx/core/test_strategy_funding.py`

**Test Coverage**:
- `test_strategy_get_funding_payments()` - Data access
- `test_strategy_funding_notifications()` - Event notifications
- `test_strategy_funding_history()` - Historical data access

#### 5.2 Strategy Context Enhancement
**Location**: `src/qubx/core/context.py`

**New Methods**:
```python
def get_funding_payments(self, instrument: Instrument, limit: int = 100) -> list[FundingPayment]:
    """Get funding payment history for instrument"""
    
def get_latest_funding_payment(self, instrument: Instrument) -> FundingPayment | None:
    """Get latest funding payment for instrument"""
```

#### 5.3 Acceptance Criteria
- [ ] All tests pass
- [ ] Strategies can access funding data
- [ ] Notifications work correctly
- [ ] Historical data accessible

### Phase 6: Logging and Monitoring (Observability)
**Goal**: Proper logging and monitoring of funding payments

#### 6.1 Test Implementation
**Location**: `tests/qubx/loggers/test_funding_payment_logger.py`

**Test Coverage**:
- `test_funding_payment_logging()` - Basic logging
- `test_funding_payment_csv_output()` - CSV format
- `test_funding_payment_metrics()` - Metrics collection

#### 6.2 Logging Enhancement
**Location**: `src/qubx/loggers/csv.py`

**New Logger**: `FundingPaymentLogger`

#### 6.3 Metrics Integration
**Location**: `src/qubx/emitters/questdb.py`

**New Metrics**:
- Funding payment amounts
- Funding payment timing
- Funding payment processing latency

#### 6.4 Acceptance Criteria
- [ ] All tests pass
- [ ] Funding payments logged correctly
- [ ] Metrics collected and emitted
- [ ] CSV output format correct

## Current Status

### Phase 1: QuestDB Reader Extension
- **Status**: COMPLETED ✅
- **Implementation Details**:
  1. ✅ Created `FundingPayment` dataclass in `src/qubx/core/basics.py`
  2. ✅ Added `DataType.FUNDING_PAYMENT` enum value
  3. ✅ Integrated funding payment support into `MultiQdbConnector`
  4. ✅ Created `QuestDBSqlFundingBuilder` for SQL query generation
  5. ✅ Added type mappings: `funding`, `funding_payment`, `funding_payments`
  6. ✅ Created comprehensive test suite in `tests/qubx/data/test_funding_payment_integration.py`

**Key Implementation Changes**:
- Used MultiQdbConnector integration instead of standalone reader (better design)
- FundingPayment dataclass matches QuestDB schema: `timestamp`, `symbol`, `funding_rate`, `funding_interval_hours`
- Added validation for funding rates (-1.0 to 1.0) and positive funding intervals
- SQL builder supports symbol filtering, time ranges, and proper table naming
- Expected table format: `binance.umswap.funding_payment`

### Usage Examples
```python
# Using MultiQdbConnector for funding payments
connector = MultiQdbConnector(host="nebula", port=8812)

# Read funding payments (various aliases supported)
funding_data = connector.read(
    "BINANCE.UM:BTCUSDT",
    start="2025-01-08T00:00:00",
    end="2025-01-08T23:59:59",
    data_type="funding_payment"  # or "funding" or "funding_payments"
)

# With custom transformer
funding_data = connector.read(
    "BINANCE.UM:BTCUSDT",
    data_type="funding",
    transform=AsPandasFrame()
)
```

### Phase 2: Funding Payment Subscription (Event Processing)
- **Status**: COMPLETED ✅
- **Implementation Details**:
  1. ✅ Created `AsFundingPayments` data transformer in `src/qubx/data/readers.py`
  2. ✅ Added `DataType.FUNDING_PAYMENT` case to `DataFetcher` match-case in `src/qubx/backtester/simulated_data.py`
  3. ✅ Added `_handle_funding_payment` method to `ProcessingManager` in `src/qubx/core/mixins/processing.py`
  4. ✅ Created comprehensive test suite in `tests/qubx/core/test_funding_payment_subscription.py`
  5. ✅ Updated imports and dependencies across all modified files

**Key Implementation Changes**:
- **AsFundingPayments Transformer**: Converts raw QuestDB data to `FundingPayment` objects following established patterns
- **DataFetcher Integration**: Added `FUNDING_PAYMENT` case that maps to `AsFundingPayments` transformer
- **ProcessingManager Handler**: Added `_handle_funding_payment` that creates `MarketEvent` objects and updates base data
- **Comprehensive Testing**: 13 tests covering transformer functionality, subscription management, and event processing

**Testing Results**: ✅ All 13 tests passing
- Data transformation working correctly (basic, column mapping, empty data, accumulation)
- DataFetcher correctly configures funding payment subscriptions
- ProcessingManager handler creates proper MarketEvent objects
- IterableSimulationData supports funding payment subscriptions
- Multi-instrument support working
- Subscription removal and error handling functional

**Phase 2 Status**: FULLY COMPLETE AND VALIDATED ✅

### Progress Log

*Date: 2025-01-08*
- Created comprehensive plan for funding payments integration
- Defined test-driven development approach
- Outlined phased implementation strategy

*Date: 2025-01-08 - Phase 1 Implementation*
- ✅ Implemented FundingPayment dataclass with validation
- ✅ Added DataType.FUNDING_PAYMENT enum
- ✅ Created QuestDBSqlFundingBuilder following existing patterns
- ✅ Integrated funding payment support into MultiQdbConnector
- ✅ Added type mappings for funding payment aliases
- ✅ Updated MultiQdbConnector docstring and type registrations
- ✅ Created comprehensive test suite with mocked QuestDB responses
- ✅ Tests cover: SQL generation, validation, type mappings, data reading
- **Design Decision**: Used MultiQdbConnector integration instead of standalone reader for better consistency

*Date: 2025-01-08 - Phase 2 Implementation*
- ✅ Implemented `AsFundingPayments` data transformer following established patterns
- ✅ Added `DataType.FUNDING_PAYMENT` case to `DataFetcher` for simulation support
- ✅ Implemented `_handle_funding_payment` method in `ProcessingManager` for event processing
- ✅ Created comprehensive test suite with 13 tests covering all subscription scenarios
- ✅ Updated imports and dependencies in `readers.py`, `simulated_data.py`, and `processing.py`
- **Design Decision**: Used existing transformer and handler patterns for consistency with framework architecture

**Next Steps**: Ready for Phase 3 (Position Enhancement - Core Logic)

## Configuration

### QuestDB Connection
```yaml
funding_payments:
  enabled: true
  questdb:
    host: "nebula"
    port: 8812
    table: "binance.umswap.funding_payment"
  update_interval: "1m"  # Check for new funding payments
```

### Strategy Configuration
```yaml
strategy:
  funding_payments:
    enabled: true
    track_history: true
    max_history: 1000
```

## Technical Notes

### QuestDB Table Schema
Expected schema for `binance.umswap.funding_payment`:
```sql
CREATE TABLE 'binance.umswap.funding_payment' (
    time TIMESTAMP,
    symbol SYMBOL,
    funding_rate DOUBLE,
    payment_amount DOUBLE,
    position_size DOUBLE,
    mark_price DOUBLE
) timestamp(time);
```

### Performance Considerations
- Use proper indexing on symbol and time columns
- Implement query caching for frequently accessed data
- Batch process funding payments when possible
- Monitor memory usage for large datasets

### Error Handling Strategy
- Graceful degradation when funding data unavailable
- Retry logic for transient database errors
- Validation of funding payment data integrity
- Logging of all error scenarios

## Future Enhancements

### Multi-Exchange Support
- Extend to other exchanges (Kraken, Bybit, etc.)
- Handle different funding payment intervals
- Exchange-specific funding calculation methods

### Advanced Features
- Funding payment forecasting
- Funding rate analysis tools
- Position sizing based on funding costs
- Funding payment alerts and notifications

### Performance Optimizations
- Streaming funding payment processing
- Compressed data storage
- Parallel processing for multiple symbols
- Memory-efficient data structures

## Risk Mitigation

### Data Integrity
- Validate funding payment amounts against position sizes
- Cross-reference with exchange funding rate data
- Implement data reconciliation procedures

### Performance Impact
- Monitor system performance during rollout
- Implement circuit breakers for resource protection
- Gradual rollout with monitoring

### Backward Compatibility
- Maintain existing functionality during integration
- Feature flags for enabling/disabling funding payments
- Rollback procedures if issues arise

## Success Metrics

1. **Functionality**: All tests pass, funding payments processed correctly
2. **Performance**: Minimal impact on system latency and throughput
3. **Reliability**: Robust error handling and data integrity
4. **Usability**: Simple configuration and monitoring
5. **Accuracy**: Funding payments match exchange records

---

**Next Action**: Begin Phase 1 by creating test file for funding payment reader.