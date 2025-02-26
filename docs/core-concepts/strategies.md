# Strategies

<!-- 
This page should include:
- Strategy development guide
- Strategy interface and required methods
- Strategy parameters
- Example strategies
- Strategy optimization
- Best practices
-->

## Strategy Development

In Qubx, a strategy is a set of rules that determine when to enter and exit trades. Strategies can range from simple moving average crossovers to complex machine learning models.

### Strategy Interface

All strategies in Qubx implement a common interface:

<!-- ```python
class Strategy:
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        pass
        
    def generate_signals(self, data):
        """Generate trading signals based on market data."""
        pass
        
    def set_indicators(self, data):
        """Calculate technical indicators used by the strategy."""
        pass
```

### Required Methods

#### `__init__(**params)`

The constructor initializes the strategy with parameters:

```python
def __init__(self, short_window=20, long_window=50):
    self.short_window = short_window
    self.long_window = long_window
```

#### `generate_signals(data)`

This method analyzes market data and generates trading signals:

```python
def generate_signals(self, data):
    # Generate buy/sell signals
    data['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
    
    # Buy signal: short MA crosses above long MA
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    
    # Sell signal: short MA crosses below long MA
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    
    return data
```

#### `set_indicators(data)`

This method calculates technical indicators used by the strategy:

```python
def set_indicators(self, data):
    # Calculate moving averages
    data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
    data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
    
    return data
```

## Example Strategies

### Moving Average Crossover

```python
class MovingAverageCrossover:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        
    def set_indicators(self, data):
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        return data
        
    def generate_signals(self, data):
        # Ensure indicators are calculated
        data = self.set_indicators(data)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # Buy signal
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # Sell signal
        
        return data
```

### RSI Strategy

```python
class RSIStrategy:
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def set_indicators(self, data):
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
        
    def generate_signals(self, data):
        # Ensure indicators are calculated
        data = self.set_indicators(data)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['rsi'] < self.oversold, 'signal'] = 1  # Buy when oversold
        data.loc[data['rsi'] > self.overbought, 'signal'] = -1  # Sell when overbought
        
        return data
```

## Strategy Parameters

Strategies can be parameterized to allow for optimization:

```python
# Create strategy with custom parameters
strategy = MovingAverageCrossover(short_window=10, long_window=30)
```

In configuration files:

```yaml
strategy:
  name: "MovingAverageCrossover"
  params:
    short_window: 10
    long_window: 30
```

## Strategy Optimization

Qubx provides tools for optimizing strategy parameters:

```python
from qubx.optimization import GridSearch

# Define parameter grid
param_grid = {
    'short_window': range(5, 30, 5),
    'long_window': range(30, 100, 10)
}

# Create optimizer
optimizer = GridSearch(
    strategy_class=MovingAverageCrossover,
    param_grid=param_grid,
    data=data,
    metric='sharpe_ratio'  # Optimize for Sharpe ratio
)

# Run optimization
results = optimizer.run()

# Get best parameters
best_params = results.best_params
print(f"Best parameters: {best_params}")
```

## Best Practices

### 1. Keep Strategies Simple

Start with simple strategies and gradually add complexity. Simple strategies are easier to understand, debug, and often perform better out-of-sample.

### 2. Avoid Overfitting

Be cautious of overfitting to historical data. Use out-of-sample testing and cross-validation to ensure your strategy generalizes well.

### 3. Consider Transaction Costs

Always account for transaction costs (commissions, slippage) in your strategy evaluation.

### 4. Use Proper Risk Management

Implement risk management rules to protect your capital during drawdowns.

### 5. Document Your Strategy

Maintain clear documentation of your strategy's logic, parameters, and expected behavior.

## Next Steps

- Learn about [Data Management](data-management.md)
- Explore [Performance Metrics](../backtesting/performance-metrics.md)
- Try [Strategy Optimization](../tutorials/advanced.md#strategy-optimization)  -->