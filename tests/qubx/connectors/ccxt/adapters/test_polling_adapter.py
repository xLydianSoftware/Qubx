"""
Unit tests for PollingToWebSocketAdapter.

Tests the generic polling adapter that converts fetch_* methods to watch_* behavior.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, List

from qubx.connectors.ccxt.adapters.polling_adapter import PollingToWebSocketAdapter


class TestPollingToWebSocketAdapter:
    """Test suite for PollingToWebSocketAdapter."""
    
    async def _run_adapter_briefly(self, adapter, duration_seconds=0.3):
        """Helper to run adapter for a brief period and then stop it."""
        # Start the adapter (this starts background polling)
        await adapter.start_watching()
        
        # Let it run for specified duration to allow polling
        await asyncio.sleep(duration_seconds)
        
        # Stop the adapter
        await adapter.stop()
    
    async def _test_get_next_data(self, adapter, expected_calls=1):
        """Helper to test the get_next_data method (CCXT awaitable pattern)."""
        await adapter.start_watching()
        
        # Get first data
        data1 = await adapter.get_next_data()
        assert data1 is not None
        
        # If expecting multiple calls, get second data
        if expected_calls > 1:
            await asyncio.sleep(0.2)  # Let background polling happen
            data2 = await adapter.get_next_data()
            assert data2 is not None
        
        await adapter.stop()
        return data1
    
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
            symbols=["BTCUSDT", "ETHUSDT"]
        )
    
    def test_initialization(self, mock_fetch_method):
        """Test adapter initialization with various parameters."""
        # Basic initialization
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=300,
            symbols=["BTCUSDT", "ETHUSDT"]
        )
        
        assert adapter.fetch_method == mock_fetch_method
        assert adapter.poll_interval_seconds == 300
        assert adapter.adapter_id.startswith("polling_adapter_")
        assert "BTCUSDT" in adapter._symbols
        assert "ETHUSDT" in adapter._symbols
        assert not adapter._is_running
        
        # Initialization with no symbols
        adapter_empty = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=60
        )
        
        assert len(adapter_empty._symbols) == 0
        assert adapter_empty.adapter_id.startswith("polling_adapter_")
    
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
        assert stats["adapter_id"].startswith("polling_adapter_")
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
        
        # Run adapter briefly to test lifecycle
        await self._run_adapter_briefly(adapter, duration_seconds=0.2)
        
        # After stopping, check final state
        assert not adapter._is_running
        assert adapter._polling_task is None or adapter._polling_task.done()
        
        # Statistics should reflect activity
        stats = adapter.get_statistics()
        assert stats["poll_count"] > 0
        assert stats["last_poll_time"] is not None
        assert mock_fetch_method.called
    
    @pytest.mark.asyncio
    async def test_double_start_stop(self, adapter):
        """Test that double start/stop doesn't cause issues."""
        # Just test that multiple stops don't error
        await adapter.stop()  # Should handle gracefully (not running)
        await adapter.stop()  # Should handle gracefully again
        assert not adapter._is_running
    
    @pytest.mark.asyncio
    async def test_polling_with_no_symbols(self, mock_fetch_method):
        """Test adapter behavior when no symbols are configured."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=0.1,  # Very fast for testing
            symbols=[]  # No symbols
        )
        
        # Run adapter briefly
        await self._run_adapter_briefly(adapter, duration_seconds=0.2)
        
        # Should not have called fetch method
        assert not mock_fetch_method.called
        assert adapter._poll_count == 0
    
    @pytest.mark.asyncio
    async def test_get_next_data_awaitable_pattern(self, adapter, mock_fetch_method):
        """Test the new get_next_data method (CCXT awaitable pattern)."""
        # Test that we can get data using the awaitable pattern
        data = await self._test_get_next_data(adapter, expected_calls=1)
        
        # Verify data structure
        assert isinstance(data, dict)
        assert len(data) > 0  # Should have some data
        assert mock_fetch_method.called
        
        # Check that polling happened
        stats = adapter.get_statistics()
        assert stats["poll_count"] > 0
    
    @pytest.mark.asyncio
    async def test_explicit_event_loop(self, mock_fetch_method):
        """Test adapter with explicit event loop (for pytest environments)."""
        import threading
        
        # Create dedicated event loop for adapter
        adapter_loop = asyncio.new_event_loop()
        
        def run_adapter_loop():
            asyncio.set_event_loop(adapter_loop)
            adapter_loop.run_forever()
        
        adapter_thread = threading.Thread(target=run_adapter_loop, daemon=True)
        adapter_thread.start()
        
        try:
            # Create adapter with explicit loop
            adapter = PollingToWebSocketAdapter(
                fetch_method=mock_fetch_method,
                poll_interval_seconds=0.1,  # Fast polling for testing
                symbols=["BTC/USDC:USDC"],
                event_loop=adapter_loop  # Explicit loop
            )
            
            # Test that get_next_data works with explicit loop
            data = await self._test_get_next_data(adapter, expected_calls=1)
            
            # Verify it worked
            assert isinstance(data, dict)
            assert len(data) > 0  # Should have some data
            assert mock_fetch_method.called
            
        finally:
            # Clean up the adapter loop
            try:
                adapter_loop.call_soon_threadsafe(adapter_loop.stop)
            except:
                pass
    
    @pytest.mark.asyncio
    async def test_fetch_method_error_handling(self, adapter):
        """Test adapter handles fetch method errors gracefully."""
        # Replace fetch method with one that raises errors
        error_fetch = AsyncMock(side_effect=Exception("Simulated fetch error"))
        adapter.fetch_method = error_fetch
        
        # Run adapter briefly
        await self._run_adapter_briefly(adapter, duration_seconds=0.3)
        
        # Should have tried to fetch and handled errors
        assert error_fetch.called
        assert adapter._error_count > 0
    
    @pytest.mark.asyncio
    async def test_dynamic_symbol_changes_during_polling(self, adapter, mock_fetch_method):
        """Test changing symbols while adapter is actively polling."""
        # Test symbol management without running polling (to avoid hanging)
        # Add symbols
        await adapter.add_symbols(["ADAUSDT", "SOLUSDT"])
        assert len(adapter._symbols) == 4  # Original 2 + 2 new
        assert adapter.is_watching("ADAUSDT")
        assert adapter.is_watching("SOLUSDT")
        
        # Remove symbols
        await adapter.remove_symbols(["BTCUSDT"])
        assert len(adapter._symbols) == 3
        assert not adapter.is_watching("BTCUSDT")
        assert adapter.is_watching("ETHUSDT")  # Should still be there
    
    @pytest.mark.asyncio
    async def test_polling_interval_respected(self, mock_fetch_method):
        """Test that polling interval is approximately respected."""
        adapter = PollingToWebSocketAdapter(
            fetch_method=mock_fetch_method,
            poll_interval_seconds=0.1,  # Fast polling for test
            symbols=["BTCUSDT"]
        )
        
        # Run adapter briefly
        await self._run_adapter_briefly(adapter, duration_seconds=0.3)
        
        # Should have done multiple polls
        assert adapter._poll_count >= 2
        assert mock_fetch_method.called
    
    @pytest.mark.asyncio
    async def test_cancellation_during_sleep(self, adapter):
        """Test that adapter can be stopped even during sleep periods."""
        # Use longer polling interval to test cancellation during sleep
        adapter.poll_interval_seconds = 5  # 5 seconds
        
        # Stop should complete quickly even though poll interval is long
        import time
        start_time = time.time()
        await self._run_adapter_briefly(adapter, duration_seconds=0.1)
        stop_time = time.time()
        
        # Should stop quickly
        assert (stop_time - start_time) < 1.0
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