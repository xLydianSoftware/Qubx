# Backtesting Framework

<!-- 
This page should include:
- Architecture overview
- Simulation engine details
- Data handling and processing
- Performance considerations
- Advanced configuration options
-->

## Architecture Overview

<!-- The Qubx backtesting framework is designed with a modular architecture that separates concerns and allows for flexible configuration. The main components include:

- **Data Module**: Handles loading, preprocessing, and managing market data
- **Strategy Module**: Contains the trading logic and signal generation
- **Execution Module**: Simulates order execution and portfolio management
- **Analysis Module**: Calculates performance metrics and generates reports

This separation allows you to focus on developing your strategy while the framework handles the complexities of backtesting.

## Simulation Engine

The core of Qubx is its simulation engine, which processes historical data and executes trading strategies in a realistic manner.

### Event-Driven Architecture

Qubx uses an event-driven architecture where the simulation progresses through a series of events:

1. **Market Data Events**: New price data becomes available
2. **Signal Events**: Strategy generates buy/sell signals
3. **Order Events**: Orders are created based on signals
4. **Fill Events**: Orders are filled (or rejected)
5. **Portfolio Update Events**: Portfolio is updated based on fills

### Simulation Modes

Qubx supports different simulation modes:

- **Vector Mode**: Fast backtesting using vectorized operations (pandas)
- **Event Mode**: More realistic simulation with event-by-event processing
- **Tick Mode**: Highest fidelity simulation using tick-by-tick data

## Data Handling

### Supported Data Formats

Qubx can work with various data formats:

- OHLCV (Open, High, Low, Close, Volume) candle data
- Tick data for high-frequency strategies
- Order book data for market microstructure analysis
- Custom data formats through data adapters

### Data Preprocessing

Before backtesting, Qubx preprocesses the data to ensure quality:

- Handling missing values
- Adjusting for splits and dividends
- Normalizing data from different sources
- Calculating derived features

## Performance Considerations

### Optimization Techniques

Qubx employs several techniques to optimize backtesting performance:

- Vectorized operations for fast computation
- Lazy loading of large datasets
- Caching of intermediate results
- Parallel processing for parameter sweeps

### Memory Management

For large datasets, Qubx provides memory-efficient options:

- Chunked processing of historical data
- On-demand loading of data segments
- Efficient storage of backtest results

## Advanced Configuration

### Multiple Assets

Qubx supports backtesting strategies across multiple assets:

```yaml
data:
  assets:
    - symbol: "BTCUSDT"
      timeframe: "1h"
    - symbol: "ETHUSDT"
      timeframe: "1h"
```

### Custom Execution Models

You can define custom execution models to simulate different market conditions:

```yaml
execution:
  model: "realistic"
  slippage: 0.001
  latency: 500  # milliseconds
  partial_fills: true
```

### Risk Management Rules

Implement risk management rules to protect your strategy:

```yaml
risk_management:
  max_drawdown: 0.2  # 20% maximum drawdown
  max_position_size: 0.25  # 25% of portfolio in one position
  daily_loss_limit: 0.05  # 5% daily loss limit
```

## Next Steps

- Learn about [Strategies](strategies.md)
- Understand [Data Management](data-management.md)
- Explore [Simulations](../backtesting/simulations.md)  -->