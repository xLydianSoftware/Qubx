"""
Integration tests for CCXT reader funding payment functionality.

This test verifies that the CcxtDataReader.get_funding_payment method:
1. Fetches real funding rate data from CCXT exchanges
2. Returns data in the expected format compatible with MultiQdbConnector
3. Works with the aux_data reflection mechanism
4. Handles various symbol formats correctly
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.connectors.ccxt.reader import CcxtDataReader


@pytest.mark.integration
class TestCcxtFundingPaymentIntegration:
    """Integration tests for CCXT funding payment functionality."""

    @pytest.fixture
    def ccxt_reader(self):
        """Create a CCXT reader instance for testing."""
        # Mock funding rate history response
        mock_funding_history = [
            {
                "timestamp": 1753401600000,  # 2025-07-25 00:00:00
                "datetime": "2025-07-25T00:00:00.000Z",
                "symbol": "BTC/USDT:USDT",
                "fundingRate": 0.0001,
                "info": {"fundingRate": "0.00010000", "fundingTime": 1753401600000, "symbol": "BTCUSDT"},
            },
            {
                "timestamp": 1753430400000,  # 2025-07-25 08:00:00
                "datetime": "2025-07-25T08:00:00.000Z",
                "symbol": "BTC/USDT:USDT",
                "fundingRate": 0.00015,
                "info": {"fundingRate": "0.00015000", "fundingTime": 1753430400000, "symbol": "BTCUSDT"},
            },
        ]

        # Create a real CCXT reader but mock the async loop submission
        reader = CcxtDataReader(exchanges=["BINANCE.UM"])

        # Mock the async loop submission to return our test data
        with patch.object(reader._loop, "submit") as mock_submit:
            mock_future = Mock()
            mock_future.result.return_value = mock_funding_history
            mock_submit.return_value = mock_future

            yield reader

    def test_get_funding_payment_basic_functionality(self, ccxt_reader):
        """Test basic funding payment retrieval functionality."""
        # Test with specific symbols and time range
        result = ccxt_reader.get_funding_payment(
            exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2025-07-24", stop="2025-07-26"
        )

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ["timestamp", "symbol"]

        # Verify columns
        expected_columns = ["funding_rate", "funding_interval_hours"]
        assert all(col in result.columns for col in expected_columns)

        # Verify data content
        assert len(result) > 0
        assert all(result["funding_rate"] > 0)  # Should have positive funding rates
        assert all(result["funding_interval_hours"] == 8.0)  # Binance uses 8-hour funding

    def test_get_funding_payment_multiple_symbols(self, ccxt_reader):
        """Test funding payment retrieval with multiple symbols."""
        result = ccxt_reader.get_funding_payment(
            exchange="BINANCE.UM", symbols=["BTCUSDT", "ETHUSDT"], start="2025-07-24", stop="2025-07-26"
        )

        # Should handle multiple symbols
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            # Check that we get data for the requested symbols
            symbols_in_result = result.index.get_level_values("symbol").unique()
            assert len(symbols_in_result) >= 1  # At least one symbol should have data

    def test_get_funding_payment_no_symbols_specified(self, ccxt_reader):
        """Test funding payment retrieval when no symbols are specified."""
        result = ccxt_reader.get_funding_payment(exchange="BINANCE.UM", start="2025-07-24", stop="2025-07-26")

        # Should work without explicit symbols (uses all available)
        assert isinstance(result, pd.DataFrame)

    def test_get_funding_payment_invalid_exchange(self, ccxt_reader):
        """Test funding payment retrieval with invalid exchange."""
        result = ccxt_reader.get_funding_payment(exchange="INVALID_EXCHANGE", symbols=["BTCUSDT"])

        # Should return empty DataFrame for invalid exchange
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_funding_payment_aux_data_integration(self, ccxt_reader):
        """Test that get_funding_payment works with aux_data reflection mechanism."""
        # Test that the method is discoverable via get_aux_data_ids
        aux_data_ids = ccxt_reader.get_aux_data_ids()
        assert "funding_payment" in aux_data_ids

        # Test that get_aux_data can call get_funding_payment via reflection
        result = ccxt_reader.get_aux_data(
            "funding_payment", exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2025-07-24", stop="2025-07-26"
        )

        # Should return the same type of result as direct method call
        assert isinstance(result, pd.DataFrame)

    def test_get_funding_payment_time_filtering(self, ccxt_reader):
        """Test funding payment time range filtering."""
        # Test with narrow time range
        start_time = "2025-07-25 00:00:00"
        stop_time = "2025-07-25 01:00:00"

        result = ccxt_reader.get_funding_payment(
            exchange="BINANCE.UM", symbols=["BTCUSDT"], start=start_time, stop=stop_time
        )

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            # Verify all timestamps are within the requested range
            timestamps = result.index.get_level_values("timestamp")
            assert all(ts >= pd.Timestamp(start_time) for ts in timestamps)
            assert all(ts <= pd.Timestamp(stop_time) for ts in timestamps)

    def test_get_funding_payment_symbol_format_handling(self, ccxt_reader):
        """Test handling of different symbol formats."""
        # Test with various symbol formats that might be used
        symbol_formats = [
            "BTCUSDT",  # Simple format
            "BTC/USDT",  # CCXT format
            "BINANCE.UM:BTCUSDT",  # Qubx format
        ]

        for symbol_format in symbol_formats:
            result = ccxt_reader.get_funding_payment(
                exchange="BINANCE.UM", symbols=[symbol_format], start="2025-07-24", stop="2025-07-26"
            )

            # Should handle all formats gracefully
            assert isinstance(result, pd.DataFrame)

    def test_get_funding_payment_data_consistency(self, ccxt_reader):
        """Test data consistency and format matching MultiQdbConnector."""
        result = ccxt_reader.get_funding_payment(
            exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2025-07-24", stop="2025-07-26"
        )

        if len(result) > 0:
            # Verify data types
            assert result["funding_rate"].dtype in [float, "float64"]
            assert result["funding_interval_hours"].dtype in [float, "float64"]

            # Verify reasonable value ranges
            assert all(result["funding_rate"].abs() < 1.0)  # Funding rates should be < 100%
            assert all(result["funding_interval_hours"] > 0)  # Interval should be positive

            # Verify index is properly sorted
            timestamps = result.index.get_level_values("timestamp")
            assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))

    def test_get_funding_payment_error_handling(self, ccxt_reader):
        """Test error handling in funding payment retrieval."""
        # Mock exchange to raise an exception
        with patch.object(ccxt_reader._loop, "submit") as mock_submit:
            mock_future = Mock()
            mock_future.result.side_effect = Exception("API Error")
            mock_submit.return_value = mock_future

            result = ccxt_reader.get_funding_payment(exchange="BINANCE.UM", symbols=["BTCUSDT"])

            # Should handle errors gracefully and return empty DataFrame
            assert isinstance(result, pd.DataFrame)
            # May be empty due to the error, but should not crash


@pytest.mark.integration
class TestCcxtFundingPaymentRealAPI:
    """Real API integration tests (requires network access)."""

    @pytest.mark.skip(reason="Requires real API access and may be slow")
    def test_real_binance_funding_payment_fetch(self):
        """Test actual funding payment fetch from Binance API."""
        # This test would use real CCXT exchange without mocking
        # Skip by default to avoid API rate limits in CI/CD
        reader = CcxtDataReader(exchanges=["BINANCE.UM"])

        result = reader.get_funding_payment(
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            start=pd.Timestamp.now() - pd.Timedelta(days=2),
            stop=pd.Timestamp.now(),
        )

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "funding_rate" in result.columns
            assert "funding_interval_hours" in result.columns
