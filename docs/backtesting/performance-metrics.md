# Performance Metrics

<!-- 
This page should include:
- Standard performance indicators
- Custom metrics
- Benchmark comparison
- Statistical analysis
- Risk-adjusted metrics
- Drawdown analysis
-->

## Understanding Performance Metrics

Performance metrics help evaluate the effectiveness of trading strategies. Qubx provides a comprehensive set of metrics to analyze strategy performance from different angles.

## Standard Performance Indicators

### Return Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Total percentage return over the backtest period |
| Annualized Return | Return normalized to a yearly basis |
| Daily/Monthly Returns | Returns broken down by time periods |
| Compound Annual Growth Rate (CAGR) | Smoothed annualized return |

<!-- ### Risk Metrics

| Metric | Description |
|--------|-------------|
| Volatility | Standard deviation of returns |
| Maximum Drawdown | Largest peak-to-trough decline |
| Drawdown Duration | Time spent in drawdowns |
| Value at Risk (VaR) | Potential loss at a given confidence level |

### Trade Metrics

| Metric | Description |
|--------|-------------|
| Win Rate | Percentage of winning trades |
| Profit Factor | Gross profit divided by gross loss |
| Average Win/Loss | Average profit/loss per trade |
| Trade Duration | Average time in trades |

## Risk-Adjusted Metrics

### Sharpe Ratio

The Sharpe ratio measures excess return per unit of risk:

```
Sharpe Ratio = (Strategy Return - Risk-Free Rate) / Strategy Volatility
```

Example in Qubx:

```python
results.sharpe_ratio  # Access Sharpe ratio from results
```

### Sortino Ratio

The Sortino ratio focuses on downside risk:

```
Sortino Ratio = (Strategy Return - Risk-Free Rate) / Downside Deviation
```

Example in Qubx:

```python
results.sortino_ratio  # Access Sortino ratio from results
```

### Calmar Ratio

The Calmar ratio compares return to maximum drawdown:

```
Calmar Ratio = Annualized Return / Maximum Drawdown
```

Example in Qubx:

```python
results.calmar_ratio  # Access Calmar ratio from results
```

## Drawdown Analysis

Drawdowns represent periods of decline from a peak to a trough:

```python
# Get drawdown information
drawdowns = results.drawdowns

# Plot drawdowns
results.plot_drawdowns()

# Analyze worst drawdowns
worst_drawdowns = results.worst_drawdowns(n=5)
```

## Benchmark Comparison

Compare your strategy against market benchmarks:

```python
# Compare to a benchmark
benchmark_data = qubx.data.load_yahoo("SPY", "1d", start_date, end_date)
comparison = qubx.analysis.compare_to_benchmark(results, benchmark_data)

# Plot comparison
comparison.plot()

# Get alpha and beta
alpha = comparison.alpha
beta = comparison.beta
```

## Statistical Analysis

### Distribution of Returns

Analyze the distribution of returns:

```python
# Plot return distribution
results.plot_return_distribution()

# Get distribution statistics
stats = results.return_statistics()
print(f"Skewness: {stats['skewness']}")
print(f"Kurtosis: {stats['kurtosis']}")
```

### Autocorrelation

Check for patterns in returns:

```python
# Calculate autocorrelation
autocorr = results.autocorrelation(lags=20)

# Plot autocorrelation
results.plot_autocorrelation()
```

## Custom Metrics

Create custom performance metrics:

```python
def gain_to_pain_ratio(results):
    """Calculate the gain-to-pain ratio (sum of returns / sum of absolute losses)."""
    returns = results.returns
    return returns.sum() / abs(returns[returns < 0]).sum()

# Apply custom metric
custom_metric = gain_to_pain_ratio(results)
```

## Visualizing Performance

Qubx provides various visualization tools:

```python
# Equity curve
results.plot_equity_curve()

# Drawdowns
results.plot_drawdowns()

# Monthly returns heatmap
results.plot_monthly_returns()

# Performance summary
results.plot_summary()
```

## Exporting Results

Export performance metrics for further analysis:

```python
# Export to CSV
results.to_csv("strategy_results.csv")

# Export to Excel
results.to_excel("strategy_results.xlsx")

# Generate HTML report
results.to_html("strategy_report.html")
```

## Next Steps

- Learn about [Visualization](../analysis/visualization.md)
- Explore [Reporting](../analysis/reporting.md)
- Try [Strategy Optimization](../tutorials/advanced.md#strategy-optimization)  -->