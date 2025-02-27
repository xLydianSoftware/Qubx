# Data Management

<!-- 
This page should include:
- Supported data formats
- Data sources and providers
- Data preprocessing
- Custom data integration
- Data quality and validation
- Handling missing data
-->

## Data in Qubx

Effective backtesting requires high-quality market data. Qubx provides comprehensive tools for loading, managing, and preprocessing financial data from various sources.

<!-- ## Supported Data Formats

Qubx supports multiple data formats to accommodate different use cases:

### OHLCV Data

The most common format is OHLCV (Open, High, Low, Close, Volume) data, typically stored as:

- CSV files
- Pandas DataFrames
- HDF5 files
- SQL databases

Example OHLCV data structure:

```
timestamp,open,high,low,close,volume
2021-01-01 00:00:00,29000.0,29100.0,28900.0,29050.0,1250.5
2021-01-01 01:00:00,29050.0,29200.0,29000.0,29150.0,1300.2
...
```

### Tick Data

For high-frequency strategies, Qubx supports tick-by-tick data:

```
timestamp,price,volume,side
2021-01-01 00:00:00.123,29000.0,0.5,buy
2021-01-01 00:00:00.456,29001.0,0.3,sell
...
```

### Order Book Data

For market microstructure analysis, Qubx can work with order book snapshots:

```
timestamp,price,quantity,side
2021-01-01 00:00:00.123,29000.0,1.5,bid
2021-01-01 00:00:00.123,29001.0,2.0,bid
2021-01-01 00:00:00.123,29002.0,0.5,ask
...
```

## Data Sources

### Built-in Data Connectors

Qubx includes connectors for popular data sources:

- **Exchange APIs**: Binance, Coinbase, etc.
- **Data Providers**: Alpha Vantage, Yahoo Finance, etc.
- **Local Files**: CSV, HDF5, Parquet, etc.

Example of loading data from different sources:

```python
# Load data from CSV
data = qubx.data.load_csv("path/to/data.csv")

# Load data from Binance
data = qubx.data.load_exchange("binance", "BTCUSDT", "1h", "2021-01-01", "2022-01-01")

# Load data from Yahoo Finance
data = qubx.data.load_yahoo("AAPL", "1d", "2021-01-01", "2022-01-01")
```

### Custom Data Sources

You can implement custom data sources by creating a data loader class:

```python
from qubx.data import DataLoader

class MyCustomDataLoader(DataLoader):
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def load_data(self, symbol, timeframe, start_date, end_date):
        # Custom logic to load data
        # ...
        return data
```

## Data Preprocessing

### Cleaning and Validation

Qubx provides tools for cleaning and validating data:

```python
# Check for missing values
missing = qubx.data.check_missing(data)

# Fill missing values
data = qubx.data.fill_missing(data, method='ffill')

# Remove outliers
data = qubx.data.remove_outliers(data, method='zscore', threshold=3)
```

### Feature Engineering

Create derived features for your strategies:

```python
# Add technical indicators
data = qubx.data.add_indicator(data, 'sma', window=20)
data = qubx.data.add_indicator(data, 'rsi', window=14)

# Add custom features
def my_feature(data):
    return data['high'] - data['low']

data['range'] = my_feature(data)
```

### Resampling

Change the timeframe of your data:

```python
# Resample to a different timeframe
hourly_data = qubx.data.resample(data, '1h')
daily_data = qubx.data.resample(data, '1d')
```

## Custom Data Integration

### Data Adapters

For non-standard data formats, create a data adapter:

```python
from qubx.data import DataAdapter

class MyDataAdapter(DataAdapter):
    def __init__(self, data_format):
        self.data_format = data_format
        
    def convert(self, data):
        # Convert from custom format to Qubx format
        # ...
        return converted_data
```

### Data Pipelines

Create data processing pipelines for complex transformations:

```python
from qubx.data import Pipeline

# Create a data pipeline
pipeline = Pipeline([
    ('load', qubx.data.load_csv("data.csv")),
    ('clean', qubx.data.fill_missing(method='ffill')),
    ('indicators', qubx.data.add_indicator('sma', window=20)),
    ('normalize', qubx.data.normalize())
])

# Process data through the pipeline
processed_data = pipeline.process()
```

## Data Quality and Validation

### Quality Checks

Qubx provides tools for assessing data quality:

```python
# Check for gaps in time series
gaps = qubx.data.check_gaps(data)

# Check for price anomalies
anomalies = qubx.data.check_anomalies(data)

# Validate data integrity
is_valid = qubx.data.validate(data)
```

### Handling Missing Data

Options for handling missing data:

```python
# Forward fill
data = qubx.data.fill_missing(data, method='ffill')

# Backward fill
data = qubx.data.fill_missing(data, method='bfill')

# Interpolation
data = qubx.data.fill_missing(data, method='interpolate')

# Custom fill function
def my_fill(series):
    # Custom logic
    return filled_series

data = qubx.data.fill_missing(data, method=my_fill)
```

## Next Steps

- Learn about [Backtesting Framework](backtesting-framework.md)
- Explore [Strategies](strategies.md)
- See [Data Visualization](../analysis/visualization.md)  -->