# Data Preparation Tutorial

<!-- 
This tutorial should cover:
- Data sources supported by Qubx
- Loading and preprocessing data
- Data quality checks
- Converting data formats
- Handling missing data
- Storing data efficiently
-->

## Introduction to Data Management in Qubx

In quantitative trading, high-quality data is the foundation of successful strategies. This tutorial will guide you through the process of preparing data for use with Qubx.

## Supported Data Sources

<!-- Qubx supports various data sources through its flexible data connectors:

```python
from qubx.data import load_data

# Load data from built-in connectors
data = load_data(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2023-12-31",
    source="binance"  # Exchange source
)

# Load data from CSV files
data = load_data(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2023-12-31",
    source="csv",
    csv_path="path/to/data.csv"
)
```

## Working with the DataReader Interface

Qubx provides a flexible `DataReader` interface for working with different data sources:

```python
from qubx.data.readers import CsvStorageDataReader

# Initialize a CSV data reader
reader = CsvStorageDataReader(base_path="path/to/data")

# Load OHLCV data
ohlcv = reader.read_ohlcv(
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1h",
    start="2023-01-01",
    end="2023-12-31"
)
```

## Connecting to QuestDB

For high-performance time-series data storage, Qubx includes a QuestDB connector:

```python
from qubx.data.readers import QuestDBConnector

# Connect to QuestDB
db = QuestDBConnector(
    host="localhost",  # QuestDB server host
    port=8812,         # Default QuestDB PostgreSQL protocol port
    username="admin",
    password="quest"
)

# Load OHLCV data from QuestDB
data = db.read_ohlcv(
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1h",
    start="2023-01-01",
    end="2023-12-31"
)
```

## Data Quality Checks

Before using data in backtests, it's important to validate its quality:

```python
from qubx.data.helpers import check_data_quality

# Check for data quality issues
quality_report = check_data_quality(data)

# Print quality issues
print(f"Missing values: {quality_report['missing_values']}")
print(f"Outliers: {quality_report['outliers']}")
print(f"Gaps: {quality_report['gaps']}")
```

## Handling Missing Data

Qubx provides utilities for handling missing data:

```python
from qubx.data.helpers import fill_missing_values

# Fill missing values using forward fill method
clean_data = fill_missing_values(data, method="ffill")

# Fill missing values using interpolation
clean_data = fill_missing_values(data, method="interpolate")

# Fill missing values using custom function
def custom_fill(series):
    # Custom logic to fill values
    return series

clean_data = fill_missing_values(data, method=custom_fill)
```

## Data Transformations

Transform your data for strategy development:

```python
from qubx.data.helpers import transform_data

# Normalize price data
normalized_data = transform_data(data, method="normalize")

# Calculate returns
returns_data = transform_data(data, method="returns")

# Calculate log returns
log_returns = transform_data(data, method="log_returns")
```

## Working with Multi-Asset Data

Prepare data for multi-asset strategies:

```python
from qubx.data import load_multi_asset_data

# Load data for multiple assets
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
multi_data = load_multi_asset_data(
    symbols=symbols,
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2023-12-31",
    source="binance"
)

# Access individual asset data
btc_data = multi_data["BTCUSDT"]
eth_data = multi_data["ETHUSDT"]
```

## Resampling Data

Change the timeframe of your data:

```python
from qubx.data.helpers import resample_data

# Resample hourly data to daily
daily_data = resample_data(hourly_data, timeframe="1d")

# Resample minute data to hourly
hourly_data = resample_data(minute_data, timeframe="1h")
```

## Storing Processed Data

Save processed data for future use:

```python
from qubx.data.helpers import save_data

# Save to CSV
save_data(data, format="csv", path="processed_data.csv")

# Save to HDF5
save_data(data, format="h5", path="processed_data.h5")

# Save to QuestDB
from qubx.data.readers import QuestDBConnector
db = QuestDBConnector()
db.write_data(data, table_name="processed_btcusdt_1h")
```

## Data Pipeline Example

Create a complete data preparation pipeline:

```python
from qubx.data import load_data
from qubx.data.helpers import (
    check_data_quality,
    fill_missing_values,
    resample_data,
    transform_data,
    save_data
)

# 1. Load raw data
raw_data = load_data(
    symbol="BTCUSDT",
    timeframe="1m",
    start_date="2023-01-01",
    end_date="2023-12-31",
    source="binance"
)

# 2. Check quality
quality_report = check_data_quality(raw_data)
print(f"Data quality report: {quality_report}")

# 3. Clean data
clean_data = fill_missing_values(raw_data, method="ffill")

# 4. Resample to hourly timeframe
hourly_data = resample_data(clean_data, timeframe="1h")

# 5. Calculate additional features
hourly_data["returns"] = transform_data(hourly_data["close"], method="returns")
hourly_data["volatility"] = hourly_data["returns"].rolling(window=24).std()

# 6. Save processed data
save_data(hourly_data, format="csv", path="btcusdt_1h_processed.csv")
```

## Next Steps

Now that you've learned how to prepare data with Qubx, you can:

1. Learn how to [set up QuestDB](questdb-setup.md) for efficient data storage
2. Explore [working with order book data](orderbook-analysis.md)
3. Develop strategies using high-quality data
4. Automate your data preparation workflow  -->