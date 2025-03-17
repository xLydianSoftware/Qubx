from pprint import pprint

import pandas as pd
import pytest

from qubx.data.readers import AsOhlcvSeries, AsOrderBook, AsPandasFrame
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
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        # Convert index to datetime with UTC timezone if it's not already
        timestamps = pd.to_datetime(data.index)
        assert timestamps.min() >= start_ts
        assert timestamps.max() <= end_ts

        print(f"Data length: {len(data)}, columns: {data.columns}")
        print(data[["localTimestamp", "price", "amount", "side"]].head())

    # book change is not implemented yet
    # def test_read_orderbook_data(self, reader: TardisMachineReader):
    #     """Test reading orderbook data from the Tardis Machine server"""
    #     # Use a known date range with data
    #     start_date = "2019-07-01"
    #     end_date = "2019-07-01 00:01:00"

    #     # Read orderbook data for BTC/USD on BitMEX
    #     data = reader.read(
    #         "bitmex:XBTUSD", start=start_date, stop=end_date, transform=AsPandasFrame(), data_type="book"
    #     )

    #     # Verify that we got data
    #     assert data is not None
    #     assert isinstance(data, pd.DataFrame)
    #     assert len(data) > 0

    #     # Check that the data is for the correct symbol
    #     assert all(data["symbol"] == "XBTUSD")
    #     assert all(data["exchange"] == "bitmex")
    #     assert all(data["type"] == "book_snapshot")  # Now using book_snapshot instead of book_change

    #     # Check that the timestamps are within the expected range
    #     start_ts = pd.Timestamp(start_date, tz="UTC")
    #     end_ts = pd.Timestamp(end_date, tz="UTC")

    #     # Convert index to datetime with UTC timezone if it's not already
    #     timestamps = pd.to_datetime(data.index)
    #     assert timestamps.min() >= start_ts
    #     assert timestamps.max() <= end_ts

    #     print(f"Data length: {len(data)}, columns: {data.columns}")
    #     # Check that we have bids and asks columns for book snapshots
    #     assert "bids" in data.columns
    #     assert "asks" in data.columns
    #     print(data[["localTimestamp", "depth", "interval"]].head())

    def test_different_exchange(self, reader: TardisMachineReader):
        """Test reading data from a different exchange"""
        # Use a known date range with data
        start_date = "2025-03-13 00:00:00"
        end_date = "2025-03-13 00:05:00"

        # Read trade data for BTC/USDT on Binance Futures
        data = reader.read(
            "binance-futures:btcusdt", start=start_date, stop=end_date, transform=AsPandasFrame(), data_type="trade"
        )

        # Verify that we got data
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # Check that the data is for the correct symbol
        assert all(data["symbol"] == "BTCUSDT")
        assert all(data["exchange"] == "binance-futures")
        assert all(data["type"] == "trade")

        # Check that the timestamps are within the expected range
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        # Convert index to datetime with UTC timezone if it's not already
        timestamps = pd.to_datetime(data.index)
        assert timestamps.min() >= start_ts
        assert timestamps.max() <= end_ts

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
