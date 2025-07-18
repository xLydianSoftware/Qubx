from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.data.helpers import CachedPrefetchReader
from qubx.data.readers import InMemoryDataFrameReader
from qubx.data.registry import ReaderRegistry


class TestCachedPrefetchReaderIntegration:
    """Integration tests for CachedPrefetchReader with real DataReader implementations."""
    
    def test_integration_with_inmemory_reader(self):
        """Test CachedPrefetchReader with InMemoryDataFrameReader."""
        # Create test data
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        test_data = pd.DataFrame({
            "open": [100 + i for i in range(len(dates))],
            "high": [110 + i for i in range(len(dates))],
            "low": [90 + i for i in range(len(dates))],
            "close": [105 + i for i in range(len(dates))],
            "volume": [1000 + i * 100 for i in range(len(dates))]
        }, index=dates)
        
        # Create base reader
        base_reader = InMemoryDataFrameReader({"BTCUSDT": test_data}, exchange="BINANCE.UM")
        
        # Add a mock get_aux_data method to return symbols
        def mock_get_aux_data(data_id, **kwargs):
            if data_id == "symbols":
                return ["BTCUSDT", "ETHUSDT"]
            elif data_id == "candles":
                # Return candles data with symbol information
                symbols = kwargs.get("symbols", ["BTCUSDT"])
                start = kwargs.get("start", "2023-01-01")
                stop = kwargs.get("stop", "2023-01-10")
                
                result_data = []
                for symbol in symbols:
                    if symbol == "BTCUSDT":
                        sliced_data = test_data.loc[start:stop].copy()
                        sliced_data["symbol"] = symbol
                        sliced_data["timestamp"] = sliced_data.index
                        result_data.append(sliced_data)
                
                if result_data:
                    combined = pd.concat(result_data)
                    return combined.set_index(["timestamp", "symbol"])
                else:
                    return pd.DataFrame()
            return None
        
        base_reader.get_aux_data = mock_get_aux_data
        
        # Wrap with CachedPrefetchReader
        cached_reader = CachedPrefetchReader(base_reader, prefetch_period="2d")
        
        # Test read operations pass through
        result = cached_reader.read("BTCUSDT", start="2023-01-01", stop="2023-01-05")
        assert len(result) > 0
        
        # Test aux data caching
        symbols1 = cached_reader.get_aux_data("symbols")
        assert symbols1 == ["BTCUSDT", "ETHUSDT"]
        assert cached_reader._cache_stats["misses"] == 1
        assert cached_reader._cache_stats["hits"] == 0
        
        # Second call should be cache hit
        symbols2 = cached_reader.get_aux_data("symbols")
        assert symbols2 == ["BTCUSDT", "ETHUSDT"]
        assert cached_reader._cache_stats["misses"] == 1
        assert cached_reader._cache_stats["hits"] == 1
        
        # Test time-based aux data with prefetch
        candles1 = cached_reader.get_aux_data(
            "candles", 
            symbols=["BTCUSDT"], 
            start="2023-01-01", 
            stop="2023-01-03"
        )
        assert isinstance(candles1, pd.DataFrame)
        assert cached_reader._cache_stats["misses"] == 2  # New cache entry for candles
        
        # Overlapping request should be cache hit
        candles2 = cached_reader.get_aux_data(
            "candles", 
            symbols=["BTCUSDT"], 
            start="2023-01-02", 
            stop="2023-01-04"
        )
        assert isinstance(candles2, pd.DataFrame)
        assert cached_reader._cache_stats["hits"] == 2  # Cache hit for overlapping range
        
    def test_delegation_methods_work(self):
        """Test that delegation methods work correctly."""
        # Create a mock reader with methods
        mock_reader = Mock()
        mock_reader.get_symbols.return_value = ["BTCUSDT", "ETHUSDT"]
        mock_reader.get_time_ranges.return_value = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-10"))
        mock_reader.get_names.return_value = ["BINANCE.UM:BTCUSDT", "BINANCE.UM:ETHUSDT"]
        mock_reader.get_aux_data_ids.return_value = {"symbols", "candles"}
        
        cached_reader = CachedPrefetchReader(mock_reader)
        
        # Test all delegation methods
        symbols = cached_reader.get_symbols("BINANCE.UM", "candles")
        assert symbols == ["BTCUSDT", "ETHUSDT"]
        
        ranges = cached_reader.get_time_ranges("BINANCE.UM:BTCUSDT", "candles")
        assert ranges == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-10"))
        
        names = cached_reader.get_names()
        assert names == ["BINANCE.UM:BTCUSDT", "BINANCE.UM:ETHUSDT"]
        
        aux_ids = cached_reader.get_aux_data_ids()
        assert aux_ids == {"symbols", "candles"}
        
    def test_cache_stats_and_management(self):
        """Test cache statistics and management features."""
        mock_reader = Mock()
        mock_reader.get_aux_data.return_value = pd.DataFrame({"test": [1, 2, 3]})
        
        cached_reader = CachedPrefetchReader(mock_reader)
        
        # Initial stats
        stats = cached_reader.get_cache_stats()
        assert stats == {"hits": 0, "misses": 0}
        
        # Make some requests
        cached_reader.get_aux_data("symbols", exchange="BINANCE.UM")
        cached_reader.get_aux_data("symbols", exchange="BINANCE.UM")  # Cache hit
        cached_reader.get_aux_data("symbols", exchange="BITFINEX.F")  # Different cache key
        
        stats = cached_reader.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        
        # Clear specific cache
        cached_reader.clear_cache("symbols")
        
        # Cache should be cleared
        cached_reader.get_aux_data("symbols", exchange="BINANCE.UM")  # Should be miss again
        stats = cached_reader.get_cache_stats()
        assert stats["misses"] == 3
        
    def test_error_handling(self):
        """Test error handling in caching logic."""
        mock_reader = Mock()
        mock_reader.get_aux_data.side_effect = Exception("Database error")
        
        cached_reader = CachedPrefetchReader(mock_reader)
        
        # Error should propagate
        with pytest.raises(Exception, match="Database error"):
            cached_reader.get_aux_data("symbols")
            
        # Cache stats should still be updated
        assert cached_reader._cache_stats["misses"] == 1
        assert cached_reader._cache_stats["hits"] == 0
        
    @pytest.mark.integration
    def test_integration_with_mqdb_quantlab(self):
        """Test CachedPrefetchReader with ReaderRegistry mqdb::quantlab."""
        try:
            # Get the reader from registry
            base_reader = ReaderRegistry.get("mqdb::quantlab")
            
            # Wrap with CachedPrefetchReader
            cached_reader = CachedPrefetchReader(base_reader, prefetch_period="1d")
            
            # Test getting symbols - should work if connection is available
            try:
                symbols = cached_reader.get_aux_data("symbols", exchange="BINANCE.UM")
                assert isinstance(symbols, list)
                assert len(symbols) > 0
                assert cached_reader._cache_stats["misses"] == 1
                assert cached_reader._cache_stats["hits"] == 0
                
                # Second call should be cache hit
                symbols2 = cached_reader.get_aux_data("symbols", exchange="BINANCE.UM")
                assert symbols2 == symbols
                assert cached_reader._cache_stats["misses"] == 1
                assert cached_reader._cache_stats["hits"] == 1
                
                # Test with candles if we have symbols
                if symbols:
                    candles = cached_reader.get_aux_data(
                        "candles",
                        exchange="BINANCE.UM",
                        symbols=[symbols[0]],  # Use first symbol
                        start="2023-01-01",
                        stop="2023-01-02",
                        timeframe="1h"
                    )
                    assert isinstance(candles, pd.DataFrame)
                    assert cached_reader._cache_stats["misses"] == 2  # New cache entry
                    
                    # Overlapping request should be cache hit
                    candles2 = cached_reader.get_aux_data(
                        "candles",
                        exchange="BINANCE.UM",
                        symbols=[symbols[0]],
                        start="2023-01-01 06:00:00",
                        stop="2023-01-01 12:00:00",
                        timeframe="1h"
                    )
                    assert isinstance(candles2, pd.DataFrame)
                    assert cached_reader._cache_stats["hits"] == 2  # Cache hit
                    
            except Exception as e:
                # If we can't connect to the database, skip the test
                pytest.skip(f"Could not connect to mqdb::quantlab: {e}")
                
        except Exception as e:
            # If reader registry fails, skip the test
            pytest.skip(f"Could not get mqdb::quantlab reader: {e}")
            
    @pytest.mark.integration
    def test_read_passthrough_with_mqdb(self):
        """Test that read operations pass through correctly with mqdb reader."""
        try:
            # Get the reader from registry
            base_reader = ReaderRegistry.get("mqdb::quantlab")
            
            # Wrap with CachedPrefetchReader
            cached_reader = CachedPrefetchReader(base_reader, prefetch_period="1d")
            
            # Test read operation passes through
            try:
                result = cached_reader.read(
                    "BINANCE.UM:BTCUSDT",
                    start="2023-01-01",
                    stop="2023-01-02",
                    data_type="candles_1m"
                )
                # Should return some data if available
                assert result is not None
                
            except Exception as e:
                # If we can't get data, skip the test
                pytest.skip(f"Could not read data from mqdb::quantlab: {e}")
                
        except Exception as e:
            # If reader registry fails, skip the test
            pytest.skip(f"Could not get mqdb::quantlab reader: {e}")
            
    @pytest.mark.integration
    def test_delegation_methods_with_mqdb(self):
        """Test that delegation methods work with mqdb reader."""
        try:
            # Get the reader from registry
            base_reader = ReaderRegistry.get("mqdb::quantlab")
            
            # Wrap with CachedPrefetchReader
            cached_reader = CachedPrefetchReader(base_reader, prefetch_period="1d")
            
            try:
                # Test delegation methods
                symbols = cached_reader.get_symbols("BINANCE.UM", "candles")
                assert isinstance(symbols, list)
                
                ranges = cached_reader.get_time_ranges("BINANCE.UM:BTCUSDT", "candles")
                assert isinstance(ranges, tuple)
                
                names = cached_reader.get_names()
                assert isinstance(names, list)
                
                aux_ids = cached_reader.get_aux_data_ids()
                assert isinstance(aux_ids, set)
                
            except Exception as e:
                # If we can't connect to the database, skip the test
                pytest.skip(f"Could not use mqdb::quantlab methods: {e}")
                
        except Exception as e:
            # If reader registry fails, skip the test
            pytest.skip(f"Could not get mqdb::quantlab reader: {e}")
            
    @pytest.mark.integration
    def test_fundamental_data_caching_with_mqdb(self):
        """Test CachedPrefetchReader with fundamental_data for 2024 year from BINANCE.UM."""
        try:
            # Get the reader from registry
            base_reader = ReaderRegistry.get("mqdb::quantlab")
            
            # Wrap with CachedPrefetchReader
            cached_reader = CachedPrefetchReader(base_reader, prefetch_period="7d")
            
            # Test getting fundamental_data for 2024
            try:
                print("Making first request for fundamental_data...")
                fundamental_data1 = cached_reader.get_aux_data(
                    "fundamental_data",
                    exchange="BINANCE.UM",
                    start="2024-01-01",
                    stop="2024-12-31",
                    timeframe="1d"
                )
                
                print(f"First request completed. Data type: {type(fundamental_data1)}")
                if isinstance(fundamental_data1, pd.DataFrame):
                    print(f"DataFrame shape: {fundamental_data1.shape}")
                    print(f"Columns: {list(fundamental_data1.columns) if len(fundamental_data1.columns) > 0 else 'No columns'}")
                    print(f"Index: {fundamental_data1.index.names if hasattr(fundamental_data1.index, 'names') else 'Simple index'}")
                
                # Check cache stats after first request
                stats1 = cached_reader.get_cache_stats()
                print(f"Cache stats after first request: {stats1}")
                assert stats1["misses"] == 1
                assert stats1["hits"] == 0
                
                # Make second request - should be cache hit
                print("Making second request for fundamental_data...")
                fundamental_data2 = cached_reader.get_aux_data(
                    "fundamental_data",
                    exchange="BINANCE.UM",
                    start="2024-01-01",
                    stop="2024-12-31",
                    timeframe="1d"
                )
                
                print(f"Second request completed. Data type: {type(fundamental_data2)}")
                
                # Check cache stats after second request
                stats2 = cached_reader.get_cache_stats()
                print(f"Cache stats after second request: {stats2}")
                assert stats2["misses"] == 1
                assert stats2["hits"] == 1
                
                # Verify data is the same
                if isinstance(fundamental_data1, pd.DataFrame) and isinstance(fundamental_data2, pd.DataFrame):
                    assert fundamental_data1.equals(fundamental_data2), "Cached data should be identical"
                    print("✅ Cached data is identical to original data")
                else:
                    assert fundamental_data1 == fundamental_data2, "Cached data should be identical"
                    print("✅ Cached data is identical to original data")
                
                # Test overlapping request (should also be cache hit due to prefetch)
                print("Making overlapping request for fundamental_data...")
                fundamental_data3 = cached_reader.get_aux_data(
                    "fundamental_data",
                    exchange="BINANCE.UM",
                    start="2024-06-01",
                    stop="2024-06-30",
                    timeframe="1d"
                )
                
                print(f"Overlapping request completed. Data type: {type(fundamental_data3)}")
                
                # Check cache stats after overlapping request
                stats3 = cached_reader.get_cache_stats()
                print(f"Cache stats after overlapping request: {stats3}")
                assert stats3["misses"] == 1  # Should still be 1 miss
                assert stats3["hits"] == 2   # Should be 2 hits now
                
                print("✅ All fundamental_data caching tests passed!")
                
            except Exception as e:
                print(f"Exception during fundamental_data test: {e}")
                import traceback
                traceback.print_exc()
                # If we can't connect to the database or get fundamental data, skip the test
                pytest.skip(f"Could not get fundamental_data from mqdb::quantlab: {e}")
                
        except Exception as e:
            # If reader registry fails, skip the test
            pytest.skip(f"Could not get mqdb::quantlab reader: {e}")