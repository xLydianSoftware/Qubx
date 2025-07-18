from unittest.mock import Mock

import pandas as pd

from qubx.data.helpers import CachedPrefetchReader
from qubx.data.readers import DataReader, DataTransformer


class TestCachedPrefetchReader:
    """Test suite for CachedPrefetchReader."""

    def test_init(self):
        """Test basic initialization."""
        mock_reader = Mock(spec=DataReader)

        # Test with default parameters
        reader = CachedPrefetchReader(mock_reader)
        assert reader._reader == mock_reader
        assert reader._prefetch_period == pd.Timedelta("1w")
        assert reader._cache_size_mb == 1000
        assert reader._aux_cache == {}
        assert reader._cache_stats == {"hits": 0, "misses": 0}

        # Test with custom parameters
        reader = CachedPrefetchReader(mock_reader, prefetch_period="2d", cache_size_mb=500)
        assert reader._prefetch_period == pd.Timedelta("2d")
        assert reader._cache_size_mb == 500

    def test_read_passthrough(self):
        """Test that read operations pass through to underlying reader."""
        mock_reader = Mock(spec=DataReader)
        mock_reader.read.return_value = [1, 2, 3]

        reader = CachedPrefetchReader(mock_reader)

        # Test read call
        result = reader.read(
            "BINANCE.UM:BTCUSDT", start="2023-01-01", stop="2023-01-02", data_type="candles", chunksize=0
        )

        assert result == [1, 2, 3]
        # Check that the mock was called once
        mock_reader.read.assert_called_once()

        # Check the call arguments
        call_args = mock_reader.read.call_args
        assert call_args[0][0] == "BINANCE.UM:BTCUSDT"  # data_id
        assert call_args[0][1] == "2023-01-01"  # start
        assert call_args[0][2] == "2023-01-02"  # stop
        assert isinstance(call_args[0][3], DataTransformer)  # transform
        assert call_args[0][4] == 0  # chunksize
        assert call_args[1]["data_type"] == "candles"  # data_type

    def test_get_aux_data_passthrough(self):
        """Test that get_aux_data currently passes through to underlying reader."""
        mock_reader = Mock(spec=DataReader)
        mock_reader.get_aux_data.return_value = pd.DataFrame({"test": [1, 2, 3]})

        reader = CachedPrefetchReader(mock_reader)

        result = reader.get_aux_data("candles", exchange="BINANCE.UM", symbols=["BTCUSDT"])

        assert isinstance(result, pd.DataFrame)
        mock_reader.get_aux_data.assert_called_once_with("candles", exchange="BINANCE.UM", symbols=["BTCUSDT"])

    def test_delegation_methods(self):
        """Test that other DataReader methods are properly delegated."""
        mock_reader = Mock(spec=DataReader)
        mock_reader.get_symbols.return_value = ["BTCUSDT", "ETHUSDT"]
        mock_reader.get_time_ranges.return_value = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"))
        mock_reader.get_names.return_value = ["name1", "name2"]
        mock_reader.get_aux_data_ids.return_value = {"candles", "funding"}

        reader = CachedPrefetchReader(mock_reader)

        # Test get_symbols
        symbols = reader.get_symbols("BINANCE.UM", "candles")
        assert symbols == ["BTCUSDT", "ETHUSDT"]
        mock_reader.get_symbols.assert_called_once_with("BINANCE.UM", "candles")

        # Test get_time_ranges
        ranges = reader.get_time_ranges("BINANCE.UM:BTCUSDT", "candles")
        assert ranges == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"))
        mock_reader.get_time_ranges.assert_called_once_with("BINANCE.UM:BTCUSDT", "candles")

        # Test get_names
        names = reader.get_names()
        assert names == ["name1", "name2"]
        mock_reader.get_names.assert_called_once_with()

        # Test get_aux_data_ids
        ids = reader.get_aux_data_ids()
        assert ids == {"candles", "funding"}
        mock_reader.get_aux_data_ids.assert_called_once()

    def test_cache_stats(self):
        """Test cache statistics functionality."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader)

        stats = reader.get_cache_stats()
        assert stats == {"hits": 0, "misses": 0}

        # Ensure it's a copy
        stats["hits"] = 10
        assert reader._cache_stats["hits"] == 0

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader)

        # Add some dummy cache entries
        reader._aux_cache["key1"] = "value1"
        reader._aux_cache["key2"] = "value2"
        reader._aux_cache["other_key"] = "other_value"

        # Clear specific data_id
        reader.clear_cache("key")
        assert "key1" not in reader._aux_cache
        assert "key2" not in reader._aux_cache
        assert "other_key" in reader._aux_cache

        # Clear entire cache
        reader.clear_cache()
        assert len(reader._aux_cache) == 0

    def test_str_representation(self):
        """Test string representation."""
        mock_reader = Mock(spec=DataReader)
        mock_reader.__str__ = Mock(return_value="MockReader")

        reader = CachedPrefetchReader(mock_reader, prefetch_period="2d")

        str_repr = str(reader)
        assert "CachedPrefetchReader" in str_repr
        assert "2 days" in str_repr

    def test_generate_aux_cache_key(self):
        """Test cache key generation for aux data."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader)

        # Test with simple parameters
        key1 = reader._generate_aux_cache_key("candles", exchange="BINANCE.UM", symbols=["BTCUSDT"])
        expected1 = "candles|exchange|BINANCE.UM|symbols|BTCUSDT"
        assert key1 == expected1

        # Test with different order of parameters (should produce same key)
        key2 = reader._generate_aux_cache_key("candles", symbols=["BTCUSDT"], exchange="BINANCE.UM")
        assert key1 == key2

        # Test with time range parameters - time ranges should be excluded from cache key
        key3 = reader._generate_aux_cache_key(
            "candles", exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2023-01-01", stop="2023-01-02"
        )
        assert "start" not in key3
        assert "stop" not in key3
        assert key3 == expected1  # Should be same as key1

        # Test with multiple symbols
        key4 = reader._generate_aux_cache_key("candles", symbols=["BTCUSDT", "ETHUSDT"])
        assert "symbols|BTCUSDT,ETHUSDT" in key4

    def test_aux_data_caching_without_time_range(self):
        """Test aux data caching for requests without time ranges."""
        mock_reader = Mock(spec=DataReader)
        test_data = pd.DataFrame({"price": [100, 200, 300]})
        mock_reader.get_aux_data.return_value = test_data

        reader = CachedPrefetchReader(mock_reader)

        # First call should be a cache miss
        result1 = reader.get_aux_data("symbols", exchange="BINANCE.UM")
        assert result1.equals(test_data)
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        mock_reader.get_aux_data.assert_called_once_with("symbols", exchange="BINANCE.UM")

        # Second call should be a cache hit
        result2 = reader.get_aux_data("symbols", exchange="BINANCE.UM")
        assert result2.equals(test_data)
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 1
        # Mock should still be called only once
        mock_reader.get_aux_data.assert_called_once()

    def test_aux_data_caching_with_time_range(self):
        """Test aux data caching with time ranges and prefetch."""
        mock_reader = Mock(spec=DataReader)

        # Create test data with time-based index
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        test_data = pd.DataFrame({"price": range(len(dates))}, index=dates)
        mock_reader.get_aux_data.return_value = test_data

        reader = CachedPrefetchReader(mock_reader, prefetch_period="2d")

        # Request data for first 5 days
        reader.get_aux_data(
            "candles", exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2023-01-01", stop="2023-01-05"
        )

        # Should have prefetched until 2023-01-07 (5 + 2 days)
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0

        # Check that mock was called with extended range
        mock_reader.get_aux_data.assert_called_once()
        call_args = mock_reader.get_aux_data.call_args
        assert call_args[1]["start"] == "2023-01-01"
        assert call_args[1]["stop"] == "2023-01-07 00:00:00"  # Extended by 2 days

        # Request overlapping data - should be a cache hit
        reader.get_aux_data(
            "candles", exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2023-01-02", stop="2023-01-06"
        )

        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 1
        # Mock should still be called only once
        mock_reader.get_aux_data.assert_called_once()

    def test_aux_data_different_parameters_different_cache(self):
        """Test that different parameters result in different cache entries."""
        mock_reader = Mock(spec=DataReader)
        mock_reader.get_aux_data.return_value = pd.DataFrame({"test": [1, 2, 3]})

        reader = CachedPrefetchReader(mock_reader)

        # Call with different exchanges
        reader.get_aux_data("symbols", exchange="BINANCE.UM")
        reader.get_aux_data("symbols", exchange="BITFINEX.F")

        # Should be 2 cache misses (different cache keys)
        assert reader._cache_stats["misses"] == 2
        assert reader._cache_stats["hits"] == 0
        assert mock_reader.get_aux_data.call_count == 2

        # Same call again should be cache hit
        reader.get_aux_data("symbols", exchange="BINANCE.UM")
        assert reader._cache_stats["hits"] == 1

    def test_aux_data_prefetch_failure_fallback(self):
        """Test fallback behavior when prefetch fails."""
        mock_reader = Mock(spec=DataReader)

        # Mock to raise exception on extended range, succeed on exact range
        def side_effect(*args, **kwargs):
            if kwargs.get("stop") == "2023-01-07 00:00:00":
                raise Exception("Prefetch failed")
            return pd.DataFrame({"price": [100, 200]})

        mock_reader.get_aux_data.side_effect = side_effect

        reader = CachedPrefetchReader(mock_reader, prefetch_period="2d")

        # Request with time range - should fallback to exact range
        result = reader.get_aux_data(
            "candles", exchange="BINANCE.UM", symbols=["BTCUSDT"], start="2023-01-01", stop="2023-01-05"
        )

        assert isinstance(result, pd.DataFrame)
        assert reader._cache_stats["misses"] == 1
        # Should be called twice: once for prefetch (fails), once for exact range
        assert mock_reader.get_aux_data.call_count == 2

    def test_aux_data_multiindex_timestamp_filtering(self):
        """Test aux data filtering with MultiIndex having 'timestamp' level."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data with MultiIndex (timestamp, symbol)
        timestamps = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Create MultiIndex
        multi_index = pd.MultiIndex.from_product(
            [timestamps, symbols], 
            names=["timestamp", "symbol"]
        )
        
        # Create DataFrame with MultiIndex
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="2d")
        
        # Test filtering with both start and stop
        result = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"],
            start="2023-01-03",
            stop="2023-01-06"
        )
        
        # Verify the result is properly filtered
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert "timestamp" in result.index.names
        assert "symbol" in result.index.names
        
        # Check that only the requested date range is returned
        result_timestamps = result.index.get_level_values("timestamp").unique()
        expected_timestamps = pd.date_range("2023-01-03", "2023-01-06", freq="D")
        
        # Convert to same timezone if needed and compare values
        result_timestamps_values = result_timestamps.sort_values()
        expected_timestamps_values = expected_timestamps.sort_values()
        
        # Compare values directly, ignoring index names
        assert len(result_timestamps_values) == len(expected_timestamps_values)
        for i in range(len(result_timestamps_values)):
            assert result_timestamps_values[i] == expected_timestamps_values[i]
        
        # Test with only start parameter
        result_start_only = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM", 
            symbols=["BTCUSDT", "ETHUSDT"],
            start="2023-01-07"
        )
        
        result_start_timestamps = result_start_only.index.get_level_values("timestamp").unique()
        expected_start_timestamps = pd.date_range("2023-01-07", "2023-01-10", freq="D")
        
        # Compare values directly, ignoring index names
        result_start_values = result_start_timestamps.sort_values()
        expected_start_values = expected_start_timestamps.sort_values()
        assert len(result_start_values) == len(expected_start_values)
        for i in range(len(result_start_values)):
            assert result_start_values[i] == expected_start_values[i]
        
        # Test with only stop parameter
        result_stop_only = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"], 
            stop="2023-01-04"
        )
        
        result_stop_timestamps = result_stop_only.index.get_level_values("timestamp").unique()
        expected_stop_timestamps = pd.date_range("2023-01-01", "2023-01-04", freq="D")
        
        # Compare values directly, ignoring index names
        result_stop_values = result_stop_timestamps.sort_values()
        expected_stop_values = expected_stop_timestamps.sort_values()
        assert len(result_stop_values) == len(expected_stop_values)
        for i in range(len(result_stop_values)):
            assert result_stop_values[i] == expected_stop_values[i]
        
        # Verify cache stats - should have 1 miss (first call) and 2 hits (subsequent calls)
        stats = reader.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 2

    def test_aux_data_multiindex_without_timestamp_level(self):
        """Test aux data filtering with MultiIndex that doesn't have 'timestamp' level."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data with MultiIndex (date, symbol) - no 'timestamp' level
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Create MultiIndex without 'timestamp' name
        multi_index = pd.MultiIndex.from_product(
            [dates, symbols], 
            names=["date", "symbol"]
        )
        
        # Create DataFrame with MultiIndex
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="2d")
        
        # Test filtering - should return data as-is since no 'timestamp' level
        result = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"],
            start="2023-01-03",
            stop="2023-01-06"
        )
        
        # Verify the result is returned as-is (no filtering applied)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert "date" in result.index.names
        assert "symbol" in result.index.names
        
        # Should have all original data since no timestamp level exists
        assert len(result) == len(test_data)
        pd.testing.assert_frame_equal(result, test_data)
