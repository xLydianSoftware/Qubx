# MACD Crossover Strategy

This example demonstrates a simple MACD (Moving Average Convergence Divergence) crossover strategy implementation using the Qubx framework.

## Strategy Overview

The MACD Crossover strategy is a trend-following momentum indicator that shows the relationship between two moving averages of an instrument's price. The strategy generates trading signals when the MACD line crosses above or below the signal line.

### Key Components

- **Fast Period**: 12 periods (default)
- **Slow Period**: 26 periods (default)
- **Signal Period**: 9 periods (default)
- **Timeframe**: 1 hour (default)
- **Leverage**: 1.0 (default)

## Configuration

The strategy is configured in the `config.yml` file with the following parameters:
