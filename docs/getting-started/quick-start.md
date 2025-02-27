# Quick Start Guide

<!-- 
This page should include:
- Basic example of a backtest
- Explanation of core concepts
- Sample strategy implementation
- Running your first backtest
- Analyzing the results
- Next steps
-->

## Your First Backtest

This guide will walk you through creating and running a simple backtest with Qubx.

### Basic Concepts

Before diving in, let's understand some key concepts:

- **Strategy**: A set of rules that determine when to enter and exit trades
- **Backtest**: The process of testing a strategy on historical data
- **Data**: Historical price and volume information used for backtesting
- **Performance Metrics**: Measurements of how well a strategy performs

### Sample Strategy

<!-- Let's create a simple moving average crossover strategy:

```python
import qubx
import pandas as pd

class MovingAverageCrossover:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, data):
        # Calculate short and long moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # Buy signal
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # Sell signal
        
        return data
```

### Running the Backtest

Now let's run a backtest with this strategy:

```python
from qubx.backtester import Backtester

# Load historical data
data = qubx.data.load_data('BTCUSDT', '1d', '2020-01-01', '2021-01-01')

# Create strategy instance
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Initialize backtester
backtester = Backtester(data=data, strategy=strategy)

# Run backtest
results = backtester.run()

# Display results
print(results.summary())
```

### Analyzing the Results

After running the backtest, you'll get a results object with performance metrics:

```python
# Plot equity curve
results.plot_equity_curve()

# Show key metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

### Next Steps

Now that you've run your first backtest, you can:

1. [Explore more complex strategies](../core-concepts/strategies.md)
2. [Learn about data management](../core-concepts/data-management.md)
3. [Understand performance metrics](../backtesting/performance-metrics.md)
4. [Try live trading](../trading/live-trading.md)

For a more detailed example, check out our [Basic Tutorials](../tutorials/basic.md).  -->