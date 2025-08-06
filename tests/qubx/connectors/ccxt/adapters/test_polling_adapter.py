"""
Unit tests for SimplifiedPollingAdapter.

Tests the simplified polling adapter that replaces the complex background polling system.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock
from typing import Dict, List

from qubx.connectors.ccxt.adapters.polling_adapter import PollingToWebSocketAdapter, PollingConfig


class TestPollingAdapter:
    """Test suite for PollingToWebSocketAdapter."""

    @pytest.fixture
    def mock_fetch_method(self):
        """Create a mock fetch method that returns predictable data."""
        async def fetch_funding_rates(symbols, **params):
            # Return different data each time to verify fresh polling
            return {
                symbol: {
                    "symbol": symbol,
                    "fundingRate": 0.0001 * (hash(symbol) % 100),
                    "timestamp": int(time.time() * 1000),
                    "markPrice": 50000.0 + (hash(symbol) % 1000),
                    "interval": "1h"
                }
                for symbol in symbols
            }
        
        return AsyncMock(side_effect=fetch_funding_rates)

    @pytest.fixture
    def adapter(self, mock_fetch_method):
        """Create a basic adapter instance for testing."""
        return PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            symbols=["BTCUSDT", "ETHUSDT"],
            config=PollingConfig(poll_interval_seconds=1)  # 1 second for fast testing
        )

    def test_initialization(self, mock_fetch_method):
        """Test adapter initialization."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            symbols=["BTCUSDT", "ETHUSDT"], 
            config=PollingConfig(poll_interval_seconds=300)
        )

        assert adapter.fetch_method == mock_fetch_method
        assert adapter.config.poll_interval_seconds == 300
        assert adapter.adapter_id.startswith("polling_adapter_")
        assert "BTCUSDT" in adapter._symbols
        assert "ETHUSDT" in adapter._symbols
        assert adapter._last_poll_time is None
        assert not adapter._symbols_changed

    def test_symbol_management_sync(self, adapter):
        """Test synchronous symbol management methods."""
        # Test is_watching (for symbols)
        assert adapter.is_watching()  # Should have symbols
        assert adapter.is_watching("BTCUSDT")
        assert not adapter.is_watching("ADAUSDT")

        # Test statistics
        stats = adapter.get_statistics()
        assert stats["adapter_id"].startswith("polling_adapter_")
        assert stats["symbol_count"] == 2
        assert stats["poll_count"] == 0

    @pytest.mark.asyncio
    async def test_symbol_management_async(self, adapter):
        """Test asynchronous symbol management operations."""
        # Test adding symbols
        await adapter.add_symbols(["ADAUSDT", "SOLUSDT"])
        assert len(adapter._symbols) == 4
        assert adapter.is_watching("ADAUSDT")
        assert adapter.is_watching("SOLUSDT")
        assert adapter._symbols_changed  # Should trigger immediate poll

        # Reset symbols changed flag
        adapter._symbols_changed = False

        # Test adding duplicate symbols (should not set changed flag)
        await adapter.add_symbols(["BTCUSDT"])
        assert len(adapter._symbols) == 4  # No change
        assert not adapter._symbols_changed

        # Test removing symbols
        await adapter.remove_symbols(["ETHUSDT", "SOLUSDT"])
        assert len(adapter._symbols) == 2
        assert not adapter.is_watching("ETHUSDT")
        assert not adapter.is_watching("SOLUSDT")
        assert adapter.is_watching("BTCUSDT")  # Should still be there
        assert adapter._symbols_changed  # Should trigger immediate poll

        # Reset and test updating entire symbol list
        adapter._symbols_changed = False
        await adapter.update_symbols(["BTCUSDT", "LINKUSDT"])
        assert len(adapter._symbols) == 2
        assert adapter.is_watching("BTCUSDT")
        assert adapter.is_watching("LINKUSDT")
        assert not adapter.is_watching("ADAUSDT")  # Should be gone
        assert adapter._symbols_changed  # Should trigger immediate poll

    @pytest.mark.asyncio
    async def test_first_poll_immediate(self, adapter, mock_fetch_method):
        """Test that the first poll is immediate."""
        start_time = time.time()
        result = await adapter.get_next_data()
        elapsed = time.time() - start_time

        # First poll should be immediate (< 0.5 seconds)
        assert elapsed < 0.5
        assert mock_fetch_method.call_count == 1
        assert adapter._poll_count == 1
        assert adapter._last_poll_time is not None
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.asyncio 
    async def test_subsequent_poll_waits(self, adapter, mock_fetch_method):
        """Test that subsequent polls wait for the interval."""
        # First poll (immediate)
        await adapter.get_next_data()
        assert mock_fetch_method.call_count == 1

        # Second poll should wait for interval (1 second in our test config)
        start_time = time.time()
        result = await adapter.get_next_data()
        elapsed = time.time() - start_time

        # Should have waited approximately 1 second
        assert 0.8 <= elapsed <= 1.5  # Allow some tolerance
        assert mock_fetch_method.call_count == 2
        assert adapter._poll_count == 2

    @pytest.mark.asyncio
    async def test_symbol_change_immediate_poll(self, adapter, mock_fetch_method):
        """Test that symbol changes trigger immediate polls."""
        # First poll
        await adapter.get_next_data()
        assert mock_fetch_method.call_count == 1

        # Change symbols - should poll immediately on next call
        await adapter.update_symbols(["NEWUSDT"])

        start_time = time.time()
        result = await adapter.get_next_data()
        elapsed = time.time() - start_time

        # Should be immediate (< 0.5 seconds)
        assert elapsed < 0.5
        assert mock_fetch_method.call_count == 2
        assert adapter._poll_count == 2

    @pytest.mark.asyncio
    async def test_no_symbols_error(self, mock_fetch_method):
        """Test that adapter raises error when no symbols are configured."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            symbols=[],  # No symbols
            config=PollingConfig(poll_interval_seconds=1)
        )

        with pytest.raises(ValueError, match="No symbols configured"):
            await adapter.get_next_data()

    @pytest.mark.asyncio
    async def test_fetch_method_error_handling(self, adapter):
        """Test adapter handles fetch method errors gracefully.""" 
        # Replace fetch method with one that raises errors
        adapter.fetch_method = AsyncMock(side_effect=Exception("Simulated fetch error"))

        with pytest.raises(Exception, match="Simulated fetch error"):
            await adapter.get_next_data()

        # Error count should be incremented
        assert adapter._error_count == 1

    def test_config_validation(self):
        """Test that PollingConfig validates parameters correctly."""
        # Valid config
        config = PollingConfig(poll_interval_seconds=300)
        assert config.poll_interval_seconds == 300

        # Invalid interval (too small)
        with pytest.raises(ValueError, match="poll_interval_seconds must be between"):
            PollingConfig(poll_interval_seconds=0.5)

        # Invalid interval (too large)  
        with pytest.raises(ValueError, match="poll_interval_seconds must be between"):
            PollingConfig(poll_interval_seconds=4000)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, adapter, mock_fetch_method):
        """Test that adapter correctly tracks statistics."""
        initial_stats = adapter.get_statistics()
        assert initial_stats["poll_count"] == 0
        assert initial_stats["error_count"] == 0
        assert initial_stats["last_poll_time"] is None

        # Perform a poll
        await adapter.get_next_data()

        updated_stats = adapter.get_statistics()
        assert updated_stats["poll_count"] == 1
        assert updated_stats["error_count"] == 0
        assert updated_stats["last_poll_time"] is not None
        assert updated_stats["symbol_count"] == 2
        assert updated_stats["poll_interval_seconds"] == 1

    def test_time_calculation_logic(self, mock_fetch_method):
        """Test the wait time calculation logic."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            symbols=["BTCUSDT"],
            config=PollingConfig(poll_interval_seconds=60)  # 1 minute
        )

        # First poll should return 0 wait time
        wait_time = adapter._calculate_wait_time()
        assert wait_time == 0

        # Set last poll time to simulate a previous poll
        adapter._last_poll_time = time.time() - 30  # 30 seconds ago

        # For 60-second intervals, should wait until next minute boundary (time-aligned)
        wait_time = adapter._calculate_wait_time()
        assert 0 <= wait_time <= 60  # Should wait up to 60 seconds for next boundary

        # Test with sub-minute interval (simple interval-based)
        adapter.config.poll_interval_seconds = 30  # 30 seconds
        adapter._last_poll_time = time.time() - 15  # 15 seconds ago
        
        wait_time = adapter._calculate_wait_time()
        assert 10 <= wait_time <= 20  # Should wait ~15 more seconds