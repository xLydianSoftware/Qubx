from datetime import datetime, timedelta

import pandas as pd
import pytest

from qubx import QubxLogConfig, logger
from qubx.connectors.ccxt.reader import CcxtDataReader
from qubx.data.readers import AsPandasFrame


@pytest.mark.integration
class TestCcxtDataReader:
    """Integration tests for the CcxtDataReader class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        # Initialize the reader with a max of 1000 bars
        self.reader = CcxtDataReader(exchanges=["BINANCE.UM"])
        QubxLogConfig.set_log_level("DEBUG")
        # Set the log level to DEBUG for more detailed output
        yield
        # Clean up after the test
        if hasattr(self, "reader") and self.reader:
            self.reader.close()

    def test_get_symbols(self):
        """Test that we can get symbols from the exchange."""
        symbols = self.reader.get_symbols("BINANCE.UM", "ohlc")
        logger.info(f"Symbols (len={len(symbols)}): {symbols}")
        assert len(symbols) > 0, "Should return at least one symbol"
        # Binance futures uses BTC/USDT:USDT format
        assert any("BTCUSDT" in symbol for symbol in symbols), "Should include BTCUSDT in some format"

    def test_get_time_ranges(self):
        """Test that we can get time ranges for a symbol."""
        # Use the correct symbol format for Binance futures
        symbol = next(s for s in self.reader.get_symbols("BINANCE.UM", "ohlc") if "BINANCE.UM:BTCUSDT" in s)
        start, end = self.reader.get_time_ranges(symbol, "ohlc")

        # Check that we got valid timestamps
        assert start is not None, "Start time should not be None"
        assert end is not None, "End time should not be None"

        # Check that the range is approximately one month
        start_dt = pd.Timestamp(start).to_pydatetime()
        end_dt = pd.Timestamp(end).to_pydatetime()
        assert (end_dt - start_dt).days >= 25, "Time range should be approximately one month"

    def test_read_latest_data(self):
        """Test that we can read the latest data."""
        # Use the correct symbol format for Binance futures
        df = self.reader.read("BINANCE.UM:BTCUSDT", "-2d", timeframe="1h", transform=AsPandasFrame())

        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        logger.info(f"Latest data (len={len(df)}):\n{df}")

        assert len(df) > 0, "Should return at least one candle"
        assert "open" in df.columns, "Should include open column"
        assert "high" in df.columns, "Should include high column"
        assert "low" in df.columns, "Should include low column"
        assert "close" in df.columns, "Should include close column"
        assert "volume" in df.columns, "Should include volume column"

        # Check that the data is sorted by timestamp
        assert df.index.is_monotonic_increasing, "Data should be sorted by timestamp"

    def test_read_data_in_chunks(self):
        """Test that we can read data in chunks."""
        # Get data for the last 24 hours
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(hours=24)

        # Use the correct symbol format for Binance futures
        symbol = next(s for s in self.reader.get_symbols("BINANCE.UM", "ohlc") if "BINANCE.UM:BTCUSDT" in s)

        # Read data in chunks of 5 candles
        chunks_iterator = self.reader.read(
            symbol,
            start=start_time.isoformat(),
            stop=end_time.isoformat(),
            timeframe="1h",
            chunksize=5,
            transform=AsPandasFrame(),
        )

        # Collect all chunks
        all_chunks = []
        for chunk in chunks_iterator:
            all_chunks.append(chunk)

        df = pd.concat(all_chunks)
        assert len(df) > 0, "Should return at least one candle"
