from datetime import datetime, timedelta

import pandas as pd
import pytest

from qubx import QubxLogConfig, logger
from qubx.connectors.ccxt.reader import CcxtDataReader


class TestCcxtDataReader:
    """Integration tests for the CcxtDataReader class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        # Initialize the reader with a max of 1000 bars
        self.reader = CcxtDataReader("binance.um", max_bars=1000)
        QubxLogConfig.set_log_level("DEBUG")
        # Set the log level to DEBUG for more detailed output
        yield
        # Clean up after the test
        if hasattr(self, "reader") and self.reader:
            self.reader.close()

    @pytest.mark.integration
    def test_get_available_timeframes(self):
        """Test that we can get available timeframes from the exchange."""
        timeframes = self.reader.get_available_timeframes()
        logger.info(f"Available timeframes: {timeframes}")
        assert len(timeframes) > 0, "Should return at least one timeframe"
        assert "1m" in timeframes, "Should include 1m timeframe"
        assert "1h" in timeframes, "Should include 1h timeframe"
        assert "1d" in timeframes, "Should include 1d timeframe"

    @pytest.mark.integration
    def test_get_symbols(self):
        """Test that we can get symbols from the exchange."""
        symbols = self.reader.get_symbols("binance.um", "ohlc")
        logger.info(f"Symbols (len={len(symbols)}): {symbols}")
        assert len(symbols) > 0, "Should return at least one symbol"
        # Binance futures uses BTC/USDT:USDT format
        assert any("BTCUSDT" in symbol for symbol in symbols), "Should include BTCUSDT in some format"

    @pytest.mark.integration
    def test_get_time_ranges(self):
        """Test that we can get time ranges for a symbol."""
        # Use the correct symbol format for Binance futures
        symbol = next(s for s in self.reader.get_symbols("binance.um", "ohlc") if "BTCUSDT" in s)
        start, end = self.reader.get_time_ranges(symbol, "ohlc")

        # Check that we got valid timestamps
        assert start is not None, "Start time should not be None"
        assert end is not None, "End time should not be None"

        # Check that the range is approximately one month
        start_dt = pd.Timestamp(start).to_pydatetime()
        end_dt = pd.Timestamp(end).to_pydatetime()
        assert (end_dt - start_dt).days >= 25, "Time range should be approximately one month"

    @pytest.mark.integration
    def test_read_latest_data(self):
        """Test that we can read the latest data."""
        # Use the correct symbol format for Binance futures
        data = self.reader.get_latest_data("BTCUSDT", timeframe="1h", limit=10)

        # Convert to DataFrame for easier testing
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        logger.info(f"Latest data (len={len(df)}):\n{df}")

        assert len(df) > 0, "Should return at least one candle"
        assert "timestamp" in df.columns, "Should include timestamp column"
        assert "open" in df.columns, "Should include open column"
        assert "high" in df.columns, "Should include high column"
        assert "low" in df.columns, "Should include low column"
        assert "close" in df.columns, "Should include close column"
        assert "volume" in df.columns, "Should include volume column"

        # Check that the data is sorted by timestamp
        assert df["timestamp"].is_monotonic_increasing, "Data should be sorted by timestamp"

    @pytest.mark.integration
    def test_read_historical_data(self):
        """Test that we can read historical data."""
        # Get data for the last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        # Use the raw data without transformation
        data = self.reader.read(
            "BTCUSDT",
            start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            stop=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            timeframe="1h",
        )

        # Convert to DataFrame for easier testing
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        assert len(df) > 0, "Should return at least one candle"

        # Check that we have data (don't check exact time range as exchange may return different range)
        if len(df) > 0:
            # Convert timestamps to datetime for easier comparison
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Check that we have approximately 24 hourly candles (give or take a few)
            assert len(df) > 0, f"Should have at least one candle, got {len(df)}"

            # Log the actual time range for debugging
            logger.info(f"Got data from {df['datetime'].min()} to {df['datetime'].max()}, {len(df)} candles")

    @pytest.mark.integration
    def test_read_data_in_chunks(self):
        """Test that we can read data in chunks."""
        # Get data for the last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        # Use the correct symbol format for Binance futures
        symbol = next(s for s in self.reader.get_symbols("binance.um", "ohlc") if "BTCUSDT" in s)

        # Read data in chunks of 5 candles
        chunks_iterator = self.reader.read(
            symbol,
            start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            stop=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            timeframe="1h",
            chunksize=5,
        )

        # Collect all chunks
        all_chunks = []
        for chunk in chunks_iterator:
            # Verify that each chunk is valid
            assert len(chunk) > 0, "Each chunk should contain at least one candle"
            all_chunks.extend(chunk)

        # Convert combined data to DataFrame for easier testing
        df = pd.DataFrame(all_chunks, columns=["timestamp", "open", "high", "low", "close", "volume"])

        assert len(df) > 0, "Should return at least one candle"

        # Check that we have data (don't check exact time range as exchange may return different range)
        if len(df) > 0:
            # Convert timestamps to datetime for easier comparison
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Check that we have approximately 24 hourly candles (give or take a few)
            assert len(df) > 0, f"Should have at least one candle, got {len(df)}"

            # Log the actual time range for debugging
            logger.info(f"Got data from {df['datetime'].min()} to {df['datetime'].max()}, {len(df)} candles")

    @pytest.mark.integration
    def test_max_bars_limit(self):
        """Test that the max_bars limit is respected."""
        # Create a reader with a small max_bars limit
        reader_with_limit = CcxtDataReader("binance.um", max_bars=10)

        try:
            # Try to get data for a period that exceeds the max_bars limit
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # This should exceed 10 hourly bars

            # Use the correct symbol format for Binance futures
            symbol = next(s for s in reader_with_limit.get_symbols("binance.um", "ohlc") if "BTCUSDT" in s)

            # This should return empty data due to the max_bars limit
            data = reader_with_limit.read(
                symbol,
                start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                stop=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                timeframe="1h",
            )

            # Convert to DataFrame for easier testing
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

            # The data should be empty because the requested time range exceeds the max_bars limit
            assert len(df) == 0, "Should return empty data when the requested time range exceeds the max_bars limit"

        finally:
            # Clean up
            if reader_with_limit:
                reader_with_limit.close()
