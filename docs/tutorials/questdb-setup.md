# QuestDB Setup Tutorial

<!-- 
This tutorial should cover:
- Introduction to QuestDB for time-series data
- Installation and configuration
- Setting up data structures
- Importing historical data
- Connecting Qubx to QuestDB
- Querying data
- Optimization tips
-->

## Introduction to QuestDB

[QuestDB](https://questdb.io/) is a high-performance, open-source time-series database optimized for financial data. Qubx integrates with QuestDB to provide efficient storage and retrieval of market data, making it an excellent choice for backtesting and live trading systems.

Key benefits of using QuestDB with Qubx include:

- High-performance ingestion and querying
- SQL interface for data analysis
- Efficient storage of time-series data
- Support for real-time and historical data
- Low latency for time-critical applications

## Installing QuestDB

### Docker Installation (Recommended)

The easiest way to set up QuestDB is using Docker:

```bash
# Pull the latest QuestDB image
docker pull questdb/questdb:latest

# Run QuestDB with persistent storage
docker run -p 8812:8812 -p 9000:9000 -p 9009:9009 -p 8081:8081 \
  -v $(pwd)/questdb-data:/root/.questdb/db \
  questdb/questdb:latest
```

This will start QuestDB with the following ports:
- 8812: PostgreSQL wire protocol
- 9000: REST API
- 9009: InfluxDB line protocol
- 8081: Web Console

### Standalone Installation

Alternatively, you can install QuestDB as a standalone application:

1. Download the latest release from [QuestDB Releases](https://github.com/questdb/questdb/releases)
2. Extract the archive to a directory of your choice
3. Run QuestDB using the provided scripts:

```bash
# On Linux/macOS
./questdb.sh start

# On Windows
questdb.bat start
```
