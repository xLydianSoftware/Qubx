"""
Unit tests for PollingToWebSocketAdapter.

Tests the generic polling adapter that converts fetch_* methods to watch_* behavior.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, List

from src.qubx.connectors.ccxt.adapters.polling_adapter import PollingToWebSocketAdapter


class TestPollingToWebSocketAdapter:
    """Test suite for PollingToWebSocketAdapter."""
    
    @pytest.fixture
    def mock_fetch_method(self):
        """Create a mock fetch method that returns predictable data."""
        async def fetch_funding_rates(symbols, **params):
            # Simulate CCXT fetch_funding_rates return format
            return {
                symbol: {
                    "symbol": symbol,
                    "fundingRate": 0.0001 * hash(symbol) % 100 / 10000,  # Deterministic but varied
                    "timestamp": 1640995200000,  # Fixed timestamp for testing
                    "nextFundingTime": 1640998800000,  # 1 hour later
                    "markPrice": 50000.0 + hash(symbol) % 1000,
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
            poll_interval_seconds=1,  # Fast polling for tests
            symbols=["BTCUSDT", "ETHUSDT"],
            adapter_id="test_adapter"
        )
    
    def test_initialization(self, mock_fetch_method):
        """Test adapter initialization with various parameters."""
        # Basic initialization
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=300,
            symbols=["BTCUSDT", "ETHUSDT"],
            adapter_id="test_adapter"
        )
        
        assert adapter.fetch_method == mock_fetch_method
        assert adapter.poll_interval_seconds == 300
        assert adapter.adapter_id == "test_adapter"
        assert "BTCUSDT" in adapter._symbols
        assert "ETHUSDT" in adapter._symbols
        assert not adapter._is_running
        
        # Initialization with no symbols
        adapter_empty = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=60
        )
        
        assert len(adapter_empty._symbols) == 0
        assert adapter_empty.adapter_id.startswith("adapter_")
    
    def test_symbol_management_sync(self, adapter):
        """Test synchronous symbol management methods."""
        # Test is_watching (for symbols)
        assert adapter.is_watching()  # Should have symbols
        assert adapter.is_watching("BTCUSDT")
        assert not adapter.is_watching("ADAUSDT")
        
        # Test is_running (for polling state)
        assert not adapter.is_running()  # Should not be running initially
        
        # Test statistics
        stats = adapter.get_statistics()
        assert stats["adapter_id"] == "test_adapter"
        assert stats["symbol_count"] == 2
        assert stats["poll_count"] == 0
        assert not stats["is_running"]
    
    @pytest.mark.asyncio
    async def test_symbol_management_async(self, adapter):
        """Test asynchronous symbol management operations."""
        # Test adding symbols
        await adapter.add_symbols(["ADAUSDT", "SOLUSDT"])
        assert len(adapter._symbols) == 4
        assert adapter.is_watching("ADAUSDT")
        assert adapter.is_watching("SOLUSDT")
        
        # Test adding duplicate symbols (should not increase count)
        await adapter.add_symbols(["BTCUSDT", "NEWUSDT"])
        assert len(adapter._symbols) == 5  # Only NEWUSDT added
        assert adapter.is_watching("NEWUSDT")
        
        # Test removing symbols
        await adapter.remove_symbols(["ETHUSDT", "SOLUSDT"])
        assert len(adapter._symbols) == 3
        assert not adapter.is_watching("ETHUSDT")
        assert not adapter.is_watching("SOLUSDT")
        assert adapter.is_watching("BTCUSDT")  # Should still be there
        
        # Test removing non-existent symbols (should not error)
        await adapter.remove_symbols(["NONEXISTENT"])
        assert len(adapter._symbols) == 3  # No change
        
        # Test updating entire symbol list
        await adapter.update_symbols(["BTCUSDT", "ETHUSDT", "LINKUSDT"])
        assert len(adapter._symbols) == 3
        assert adapter.is_watching("BTCUSDT")
        assert adapter.is_watching("ETHUSDT") 
        assert adapter.is_watching("LINKUSDT")
        assert not adapter.is_watching("ADAUSDT")  # Should be gone
        
        # Test updating to empty list
        await adapter.update_symbols([])
        assert len(adapter._symbols) == 0
        assert not adapter.is_watching()
    
    @pytest.mark.asyncio
    async def test_adapter_lifecycle(self, adapter, mock_fetch_method):
        """Test adapter start/stop lifecycle."""
        # Initially not running
        assert not adapter._is_running
        assert adapter._polling_task is None
        
        # Start the adapter
        await adapter.start_watching()
        assert adapter._is_running
        assert adapter._polling_task is not None
        
        # Give it a moment to do at least one poll
        await asyncio.sleep(0.1)
        
        # Check that fetch method was called
        assert mock_fetch_method.called
        assert adapter._poll_count > 0
        
        # Stop the adapter
        await adapter.stop()
        assert not adapter._is_running
        assert adapter._polling_task is None or adapter._polling_task.done()
        
        # Statistics should reflect activity
        stats = adapter.get_statistics()
        assert stats["poll_count"] > 0
        assert stats["last_poll_time"] is not None
    
    @pytest.mark.asyncio
    async def test_double_start_stop(self, adapter):
        """Test that double start/stop doesn't cause issues."""
        # Double start should not error
        await adapter.start_watching()
        await adapter.start_watching()  # Should log warning but not error
        assert adapter._is_running
        
        # Double stop should not error
        await adapter.stop()
        await adapter.stop()  # Should handle gracefully
        assert not adapter._is_running
    
    @pytest.mark.asyncio
    async def test_polling_with_no_symbols(self, mock_fetch_method):
        """Test adapter behavior when no symbols are configured."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=0.1,  # Very fast for testing
            symbols=[],  # No symbols
            adapter_id="empty_adapter"
        )
        
        await adapter.start_watching()
        
        # Give it a moment to run
        await asyncio.sleep(0.2)
        
        # Should not have called fetch method
        assert not mock_fetch_method.called
        assert adapter._poll_count == 0
        
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_fetch_method_error_handling(self, adapter):
        """Test adapter handles fetch method errors gracefully."""
        # Replace fetch method with one that raises errors
        error_fetch = AsyncMock(side_effect=Exception("Simulated fetch error"))
        adapter.fetch_method = error_fetch
        
        await adapter.start_watching()
        
        # Give it time to try polling and handle errors
        await asyncio.sleep(0.2)
        
        # Should have tried to fetch and handled errors
        assert error_fetch.called
        assert adapter._error_count > 0
        assert adapter._is_running  # Should still be running despite errors
        
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_dynamic_symbol_changes_during_polling(self, adapter, mock_fetch_method):
        """Test changing symbols while adapter is actively polling."""
        await adapter.start_watching()
        
        # Let it do initial polls
        await asyncio.sleep(0.1)
        initial_call_count = mock_fetch_method.call_count
        
        # Add symbols during polling
        await adapter.add_symbols(["ADAUSDT", "SOLUSDT"])
        
        # Wait for more polls
        await asyncio.sleep(0.2)
        
        # Should have called fetch with new symbols
        assert mock_fetch_method.call_count > initial_call_count
        
        # Check that latest calls included new symbols
        latest_call = mock_fetch_method.call_args_list[-1]
        called_symbols = latest_call[0][0]  # First positional argument
        assert "ADAUSDT" in called_symbols
        assert "SOLUSDT" in called_symbols
        
        # Remove symbols during polling
        await adapter.remove_symbols(["BTCUSDT"])
        
        await asyncio.sleep(0.2)
        
        # Latest call should not include removed symbol
        latest_call = mock_fetch_method.call_args_list[-1]
        called_symbols = latest_call[0][0]
        assert "BTCUSDT" not in called_symbols
        assert "ETHUSDT" in called_symbols  # Should still be there
        
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_polling_interval_respected(self, mock_fetch_method):
        """Test that polling interval is approximately respected."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=0.2,  # 200ms interval
            symbols=["BTCUSDT"],
            adapter_id="interval_test"
        )
        
        await adapter.start_watching()
        
        # Record initial state
        await asyncio.sleep(0.05)  # Wait for first poll
        first_poll_count = adapter._poll_count
        
        # Wait for one interval
        await asyncio.sleep(0.3)
        
        # Should have done at least one more poll
        assert adapter._poll_count > first_poll_count
        
        # Wait for another interval
        second_poll_count = adapter._poll_count
        await asyncio.sleep(0.3)
        
        # Should have done more polls
        assert adapter._poll_count > second_poll_count
        
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_cancellation_during_sleep(self, adapter):
        """Test that adapter can be stopped even during sleep periods."""
        # Use longer polling interval to test cancellation during sleep
        adapter.poll_interval_seconds = 10  # 10 seconds
        
        await adapter.start_watching()
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Stop should complete quickly even though poll interval is long
        import time
        start_time = time.time()
        await adapter.stop()
        stop_time = time.time()
        
        # Should stop in much less than the 10-second poll interval
        assert (stop_time - start_time) < 2.0
        assert not adapter._is_running
    
    def test_thread_safety_assumptions(self, adapter):
        """Test that symbol operations are thread-safe (basic checks)."""
        # This test verifies our use of asyncio.Lock for symbol operations
        assert hasattr(adapter, '_symbols_lock')
        assert isinstance(adapter._symbols_lock, asyncio.Lock)
        
        # Symbol set should be a proper set for O(1) operations
        assert isinstance(adapter._symbols, set)
    
    @pytest.mark.asyncio
    async def test_edge_case_empty_symbol_operations(self, adapter):
        """Test edge cases with empty symbol lists."""
        # Adding empty list should not error
        await adapter.add_symbols([])
        assert len(adapter._symbols) == 2  # Should still have original symbols
        
        # Removing empty list should not error
        await adapter.remove_symbols([])
        assert len(adapter._symbols) == 2  # Should still have original symbols
        
        # Updating to None should work
        await adapter.update_symbols(None)
        assert len(adapter._symbols) == 0
        
        # Updating to empty list should work
        await adapter.update_symbols([])
        assert len(adapter._symbols) == 0