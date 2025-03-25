from pprint import pprint

import pandas as pd
import pytest

from qubx.data.readers import AsOhlcvSeries, AsOrderBook, AsPandasFrame, AsTrades
from qubx.data.tardis import TardisMachineReader


@pytest.fixture
def reader() -> TardisMachineReader:
    """Create a TardisMachineReader instance for testing"""
    return TardisMachineReader()


@pytest.mark.integration
class TestTardisMachineReader:
    """
    Integration tests for the TardisMachineReader class.

    These tests require access to the Tardis Machine server.
    Run with: pytest -m integration tests/qubx/data/tardis_test.py
    """

    def test_get_exchanges(self, reader: TardisMachineReader):
        """Test retrieving the list of available exchanges"""
        exchanges = reader.get_exchanges()
        pprint(exchanges)
        assert len(exchanges) > 0
        assert "bitmex" in exchanges
        assert "binance-futures" in exchanges

    def test_get_exchange_info(self, reader: TardisMachineReader):
        """Test retrieving information about a specific exchange"""
        exchange_info = reader.get_exchange_info("bitmex")
        assert exchange_info is not None
        assert "availableSymbols" in exchange_info
        assert "XBTUSD" in [symbol["id"] for symbol in exchange_info["availableSymbols"]]

        # Check that we have some symbols
        assert len(exchange_info["availableSymbols"]) > 0

        pprint(exchange_info["availableSymbols"][:10])

    def test_get_bitfinex_derivatives_info(self, reader: TardisMachineReader):
        """Test retrieving information about Bitfinex Derivatives exchange"""
        exchange_info = reader.get_exchange_info("bitfinex-derivatives")
        assert exchange_info is not None
        assert "availableSymbols" in exchange_info

        # Check that we have some symbols
        assert len(exchange_info["availableSymbols"]) > 0

        # Verify we have some common perpetual futures symbols
        symbol_ids = [symbol["id"] for symbol in exchange_info["availableSymbols"]]
        assert any(symbol.startswith("BTCF0:USTF0") for symbol in symbol_ids), "Should have BTC perpetual futures"

        print("\nBitfinex Derivatives available symbols (first 10):")
        pprint(exchange_info["availableSymbols"][:10])

    def test_read_trade_data(self, reader: TardisMachineReader):
        """Test reading trade data from the Tardis Machine server"""
        # Use a known date range with data
        start_date = "2025-03-13 00:00:00"
        end_date = "2025-03-13 00:05:00"

        # Read trade data for BTC/USD on BitMEX
        # data = reader.read(
        #     "bitmex:XBTUSD", start=start_date, stop=end_date, transform=AsPandasFrame(), data_type="trade"
        # )
        data = reader.read(
            "bitfinex-derivatives:BTCF0:USTF0",
            start=start_date,
            stop=end_date,
            transform=AsPandasFrame(),
            data_type="trade",
        )

        # Verify that we got data
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # Check that we have the expected columns
        expected_columns = ["amount", "exchange", "localTimestamp", "price", "side", "symbol", "type"]
        assert sorted(data.columns.tolist()) == sorted(expected_columns)

        # Check that the data is for the correct symbol
        assert all(data["symbol"] == "BTCF0:USTF0")
        assert all(data["exchange"] == "bitfinex-derivatives")
        assert all(data["type"] == "trade")

        # Check that the timestamps are within the expected range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Convert index to datetime without timezone
        timestamps = pd.to_datetime(data.index)
        assert timestamps.min() >= start_ts
        assert timestamps.max() <= end_ts

        print(f"Data length: {len(data)}, columns: {data.columns}")
        print(data[["localTimestamp", "price", "amount", "side"]].head())

    def test_read_book_snapshot_with_custom_levels(self, reader: TardisMachineReader):
        """Test reading book snapshot data with custom levels and interval"""
        # Use a known date range with data
        start_date = "2025-03-17 00:00:00"
        end_date = "2025-03-17 00:05:00"

        # Read orderbook snapshot data with 25 levels and 1000ms interval
        data = reader.read(
            "bitmex:XBTUSD",
            start=start_date,
            stop=end_date,
            transform=AsOrderBook(),
            data_type="book_snapshot_25_1000ms",
        )

        # Verify that we got data
        assert data is not None
        assert isinstance(data, list)
        assert len(data) > 0

    def test_read_book_snapshot_with_custom_levels2(self, reader: TardisMachineReader):
        """Test reading book snapshot data with custom levels and interval"""
        # Use a known date range with data
        start_date = "2025-03-17 00:00:00"
        end_date = "2025-03-17 00:05:00"

        # Read orderbook snapshot data with 25 levels and 1000ms interval
        data = reader.read(
            # "binance-futures:BTCUSDT",
            "bitfinex-derivatives:BTCF0:USTF0",
            start=start_date,
            stop=end_date,
            transform=AsOrderBook(),
            data_type="book_snapshot_100_1s",
            tick_size_pct=0.01,
            depth=25,
        )

        # Verify that we got data
        assert data is not None
        assert isinstance(data, list)
        assert len(data) > 0

    def test_read_trade_data_streaming(self, reader: TardisMachineReader):
        """Test reading trade data with streaming functionality"""
        # Use a small time window to test streaming
        start_date = "2025-03-13 00:00:00"
        end_date = "2025-03-13 00:05:00"

        # Read trade data with a small chunk size to test streaming
        chunks = list(
            reader.read(
                "bitfinex-derivatives:BTCF0:USTF0",
                start=start_date,
                stop=end_date,
                transform=AsPandasFrame(),
                data_type="trade",
                chunksize=100,  # Process 100 records at a time
            )
        )

        # Verify that we got chunks of data
        assert len(chunks) > 0
        assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)

        # Combine all chunks and verify the total data
        combined_data = pd.concat(chunks)
        assert len(combined_data) > 0

        # Check that we have the expected columns
        expected_columns = ["amount", "exchange", "localTimestamp", "price", "side", "symbol", "type"]
        assert sorted(combined_data.columns.tolist()) == sorted(expected_columns)

        # Check that the data is for the correct symbol
        assert all(combined_data["symbol"] == "BTCF0:USTF0")
        assert all(combined_data["exchange"] == "bitfinex-derivatives")
        assert all(combined_data["type"] == "trade")

        # Check that the timestamps are within the expected range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Convert index to datetime without timezone
        timestamps = pd.to_datetime(combined_data.index)
        assert timestamps.min() >= start_ts
        assert timestamps.max() <= end_ts

        # Verify that chunks are properly sized
        chunk_sizes = [len(chunk) for chunk in chunks]
        assert all(size <= 100 for size in chunk_sizes[:-1])  # All chunks except possibly the last one should be <= 100
        assert sum(chunk_sizes) == len(combined_data)  # Sum of chunk sizes should equal total data size

        print(f"Number of chunks: {len(chunks)}")
        print(f"Total records: {len(combined_data)}")
        print(f"Chunk sizes: {chunk_sizes}")
        print("\nFirst few records from first chunk:")
        print(chunks[0][["localTimestamp", "price", "amount", "side"]].head())

    def test_read_book_snapshot_chunked(self, reader: TardisMachineReader):
        """Test reading book snapshot data with custom levels and interval"""
        # Use a known date range with data
        start_date = "2025-03-17 00:00:00"
        end_date = "2025-03-17 00:02:00"

        # Read orderbook snapshot data with 25 levels and 1000ms interval
        it = reader.read(
            "binance-futures:BTCUSDT",
            # "bitfinex-derivatives:BTCF0:USTF0",
            start=start_date,
            stop=end_date,
            transform=AsOrderBook(),
            data_type="book_snapshot_1000_1s",
            tick_size_pct=0.01,
            depth=25,
            chunksize=100,
        )

        for chunk in it:
            assert chunk is not None
            assert isinstance(chunk, list)
            assert len(chunk) > 0

    def test_read_trades_with_as_trades_transform(self, reader: TardisMachineReader):
        """Test reading trade data with AsTrades transform"""
        # Use a known date range with data
        start_date = "2025-03-13 00:00:00"
        end_date = "2025-03-13 00:05:00"

        # Read trade data with AsTrades transform
        data = reader.read(
            "bitfinex-derivatives:BTCF0:USTF0",
            start=start_date,
            stop=end_date,
            transform=AsTrades(),
            data_type="trade",
        )

        # Verify that we got data
        assert data is not None
        assert isinstance(data, list)
        assert len(data) > 0
