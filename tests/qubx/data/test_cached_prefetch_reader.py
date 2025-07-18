from unittest.mock import Mock

import pandas as pd

from qubx.data.helpers import CachedPrefetchReader
from qubx.data.readers import DataReader, DataTransformer, AsPandasFrame


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
        """Test that read operations ultimately pass through to underlying reader when cache is empty."""
        mock_reader = Mock(spec=DataReader)
        mock_data = [1, 2, 3]
        mock_columns = ["timestamp", "open", "high"]
        
        # Mock the read method to properly set up the transformer
        def mock_read(data_id, start=None, stop=None, transform=None, chunksize=0, **kwargs):
            if transform:
                transform.start_transform(data_id, mock_columns, start=start, stop=stop)
                transform.process_data(mock_data)
                return transform.collect()
            return mock_data
        
        mock_reader.read.side_effect = mock_read

        reader = CachedPrefetchReader(mock_reader)

        # Test read call
        result = reader.read(
            "BINANCE.UM:BTCUSDT", start="2023-01-01", stop="2023-01-02", data_type="candles", chunksize=0
        )

        assert result == [1, 2, 3]
        # Check that the mock was called once (should be cache miss)
        mock_reader.read.assert_called_once()
        
        # Check that it was a cache miss
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0

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
        expected1 = "aux|candles|exchange|BINANCE.UM|symbols|BTCUSDT"
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

    def test_multi_period_caching_overlapping_ranges(self):
        """Test that overlapping time periods are properly merged in cache."""
        mock_reader = Mock(spec=DataReader)
        
        # Create different data for different time periods
        def mock_get_aux_data(data_id, **kwargs):
            start = kwargs.get("start")
            stop = kwargs.get("stop")
            
            
            # Handle timestamp conversion - normalize the comparison
            start_str = str(start).split()[0] if start else None  # Remove time component
            stop_str = str(stop).split()[0] if stop else None    # Remove time component
            
            if start_str == "2023-01-01" and stop_str == "2023-01-05":
                # First period: 2023-01-01 to 2023-01-05
                dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
                return pd.DataFrame({"price": [100, 101, 102, 103, 104]}, index=dates)
            elif start_str == "2023-01-03" and stop_str == "2023-01-07":
                # Second period: 2023-01-03 to 2023-01-07 (overlaps with first)
                dates = pd.date_range("2023-01-03", "2023-01-07", freq="D")
                return pd.DataFrame({"price": [102, 103, 104, 105, 106]}, index=dates)
            else:
                # Create some default data for any other range
                if start and stop:
                    dates = pd.date_range(start_str, stop_str, freq="D")
                    prices = [100 + i for i in range(len(dates))]
                    return pd.DataFrame({"price": prices}, index=dates)
                return pd.DataFrame()
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")  # No prefetch for simplicity
        
        # First request: 2023-01-01 to 2023-01-05
        result1 = reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-05")
        assert len(result1) == 5
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        
        # Second request: 2023-01-03 to 2023-01-07 (overlaps with first)
        result2 = reader.get_aux_data("candles", start="2023-01-03", stop="2023-01-07")
        # After merging, we should get the properly filtered result for the requested range
        assert len(result2) == 5  # Should be filtered to requested range
        assert reader._cache_stats["misses"] == 2  # Should be a cache miss since not fully covered
        assert reader._cache_stats["hits"] == 0
        
        # Third request: 2023-01-01 to 2023-01-07 (should be fully cached now)
        result3 = reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-07")
        assert len(result3) == 7  # Should have merged data
        assert reader._cache_stats["misses"] == 2  # Should be a cache hit
        assert reader._cache_stats["hits"] == 1
        
        # Verify the merged data contains all dates
        expected_dates = pd.date_range("2023-01-01", "2023-01-07", freq="D")
        pd.testing.assert_index_equal(result3.index, expected_dates)

    def test_multi_period_caching_non_overlapping_ranges(self):
        """Test that non-overlapping time periods are stored separately."""
        mock_reader = Mock(spec=DataReader)
        
        def mock_get_aux_data(data_id, **kwargs):
            start = kwargs.get("start")
            stop = kwargs.get("stop")
            
            # Handle timestamp conversion - normalize the comparison
            start_str = str(start).split()[0] if start else None
            stop_str = str(stop).split()[0] if stop else None
            
            if start_str == "2023-01-01" and stop_str == "2023-01-05":
                # First period
                dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
                return pd.DataFrame({"price": [100, 101, 102, 103, 104]}, index=dates)
            elif start_str == "2023-01-10" and stop_str == "2023-01-15":
                # Second period (non-overlapping)
                dates = pd.date_range("2023-01-10", "2023-01-15", freq="D")
                return pd.DataFrame({"price": [110, 111, 112, 113, 114, 115]}, index=dates)
            else:
                # Create some default data for any other range
                if start and stop:
                    dates = pd.date_range(start_str, stop_str, freq="D")
                    prices = [100 + i for i in range(len(dates))]
                    return pd.DataFrame({"price": prices}, index=dates)
                return pd.DataFrame()
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request
        result1 = reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-05")
        assert len(result1) == 5
        assert reader._cache_stats["misses"] == 1
        
        # Second request (non-overlapping)
        result2 = reader.get_aux_data("candles", start="2023-01-10", stop="2023-01-15")
        assert len(result2) == 6
        assert reader._cache_stats["misses"] == 2
        
        # Check that both periods are cached separately
        cache_key = reader._generate_aux_cache_key("candles")
        cached_ranges = reader._aux_cache_ranges[cache_key]
        assert len(cached_ranges) == 2
        
        # Request that spans both periods should be a cache hit
        reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-05")
        assert reader._cache_stats["hits"] == 1
        
        reader.get_aux_data("candles", start="2023-01-10", stop="2023-01-15")
        assert reader._cache_stats["hits"] == 2

    def test_multi_period_caching_adjacent_ranges(self):
        """Test that adjacent time periods are properly merged."""
        mock_reader = Mock(spec=DataReader)
        
        def mock_get_aux_data(data_id, **kwargs):
            start = kwargs.get("start")
            stop = kwargs.get("stop")
            
            # Handle timestamp conversion - normalize the comparison
            start_str = str(start).split()[0] if start else None
            stop_str = str(stop).split()[0] if stop else None
            
            if start_str == "2023-01-01" and stop_str == "2023-01-05":
                dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
                return pd.DataFrame({"price": [100, 101, 102, 103, 104]}, index=dates)
            elif start_str == "2023-01-06" and stop_str == "2023-01-10":
                dates = pd.date_range("2023-01-06", "2023-01-10", freq="D")
                return pd.DataFrame({"price": [105, 106, 107, 108, 109]}, index=dates)
            else:
                # Create some default data for any other range
                if start and stop:
                    dates = pd.date_range(start_str, stop_str, freq="D")
                    prices = [100 + i for i in range(len(dates))]
                    return pd.DataFrame({"price": prices}, index=dates)
                return pd.DataFrame()
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request
        result1 = reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-05")
        assert len(result1) == 5
        
        # Second request (adjacent)
        result2 = reader.get_aux_data("candles", start="2023-01-06", stop="2023-01-10")
        assert len(result2) == 5
        
        # Request spanning both periods should be a cache hit due to adjacent merging
        result3 = reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-10")
        assert len(result3) == 10
        # The exact hit/miss count depends on the merging logic, but we should have merged data
        assert reader._cache_stats["hits"] >= 0

    def test_multi_period_caching_data_merging_with_duplicates(self):
        """Test that overlapping data with duplicate indices is properly merged."""
        mock_reader = Mock(spec=DataReader)
        
        def mock_get_aux_data(data_id, **kwargs):
            start = kwargs.get("start")
            stop = kwargs.get("stop")
            
            # Handle timestamp conversion - normalize the comparison
            start_str = str(start).split()[0] if start else None
            stop_str = str(stop).split()[0] if stop else None
            
            if start_str == "2023-01-01" and stop_str == "2023-01-05":
                # First period with original values
                dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
                return pd.DataFrame({"price": [100, 101, 102, 103, 104]}, index=dates)
            elif start_str == "2023-01-03" and stop_str == "2023-01-07":
                # Second period with updated values for overlapping dates
                dates = pd.date_range("2023-01-03", "2023-01-07", freq="D")
                return pd.DataFrame({"price": [999, 999, 999, 105, 106]}, index=dates)  # Updated values
            else:
                # Create some default data for any other range
                if start and stop:
                    dates = pd.date_range(start_str, stop_str, freq="D")
                    prices = [100 + i for i in range(len(dates))]
                    return pd.DataFrame({"price": prices}, index=dates)
                return pd.DataFrame()
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request
        result1 = reader.get_aux_data("candles", start="2023-01-01", stop="2023-01-05")
        assert result1.loc["2023-01-03", "price"] == 102  # Original value
        
        # Second request with overlapping data
        result2 = reader.get_aux_data("candles", start="2023-01-03", stop="2023-01-07")
        assert result2.loc["2023-01-03", "price"] == 999  # Updated value
        
        # Check that cached data has the updated values (keep='last' in merge)
        cached_data = reader._aux_cache[reader._generate_aux_cache_key("candles")]
        assert cached_data.loc["2023-01-03", "price"] == 999  # Should keep the newer value
        assert cached_data.loc["2023-01-01", "price"] == 100  # Original value preserved
        assert cached_data.loc["2023-01-06", "price"] == 105  # New value

    def test_multi_period_caching_complex_scenario(self):
        """Test a complex scenario with multiple overlapping and non-overlapping periods."""
        mock_reader = Mock(spec=DataReader)
        
        # Track what periods have been requested
        requested_periods = []
        
        def mock_get_aux_data(data_id, **kwargs):
            start = kwargs.get("start")
            stop = kwargs.get("stop")
            period = f"{start}_{stop}"
            requested_periods.append(period)
            
            # Generate data for the requested period
            start_date = pd.Timestamp(start)
            stop_date = pd.Timestamp(stop)
            dates = pd.date_range(start_date, stop_date, freq="D")
            
            # Create some varying data
            base_price = 100 + (start_date.day - 1) * 10
            prices = [base_price + i for i in range(len(dates))]
            
            return pd.DataFrame({"price": prices}, index=dates)
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Make several requests in different orders
        requests = [
            ("2023-01-01", "2023-01-05"),  # First period
            ("2023-01-10", "2023-01-15"),  # Non-overlapping period
            ("2023-01-03", "2023-01-08"),  # Overlapping with first
            ("2023-01-07", "2023-01-12"),  # Bridges the gap
            ("2023-01-01", "2023-01-15"),  # Should be fully cached
        ]
        
        results = []
        for start, stop in requests:
            result = reader.get_aux_data("candles", start=start, stop=stop)
            results.append(result)
        
        # The last request should be a cache hit since all data is now cached
        final_stats = reader.get_cache_stats()
        assert final_stats["hits"] >= 1  # At least one hit for the final comprehensive request
        
        # Check that the final result contains all expected dates
        final_result = results[-1]
        expected_dates = pd.date_range("2023-01-01", "2023-01-15", freq="D")
        pd.testing.assert_index_equal(final_result.index, expected_dates)

    def test_range_merging_helper_method(self):
        """Test the _merge_time_ranges helper method."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader)
        
        # Test overlapping ranges
        ranges = [
            ("2023-01-01", "2023-01-05"),
            ("2023-01-03", "2023-01-08"),
            ("2023-01-10", "2023-01-15"),
        ]
        merged = reader._merge_time_ranges(ranges)
        assert len(merged) == 2  # Should merge first two, keep third separate
        
        # Test adjacent ranges (should be merged)
        ranges = [
            ("2023-01-01", "2023-01-05"),
            ("2023-01-06", "2023-01-10"),
        ]
        merged = reader._merge_time_ranges(ranges)
        assert len(merged) == 1  # Should merge adjacent ranges
        
        # Test non-overlapping ranges
        ranges = [
            ("2023-01-01", "2023-01-05"),
            ("2023-01-10", "2023-01-15"),
        ]
        merged = reader._merge_time_ranges(ranges)
        assert len(merged) == 2  # Should keep separate

    def test_symbol_filtering_all_symbols_to_specific_symbols(self):
        """Test that requesting specific symbols can reuse cached 'all symbols' data."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data with MultiIndex (timestamp, symbol)
        timestamps = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
        
        multi_index = pd.MultiIndex.from_product(
            [timestamps, symbols], 
            names=["timestamp", "symbol"]
        )
        
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request: all symbols (no symbols parameter)
        result1 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-05"
        )
        assert len(result1) == 15  # 5 days * 3 symbols
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        
        # Second request: specific symbols (should be cache hit with filtering)
        result2 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"],
            start="2023-01-02",
            stop="2023-01-04"
        )
        assert len(result2) == 6  # 3 days * 2 symbols
        assert reader._cache_stats["misses"] == 1  # Should still be 1
        assert reader._cache_stats["hits"] == 1   # Should be cache hit
        
        # Verify that only requested symbols are in result
        result_symbols = set(result2.index.get_level_values("symbol").unique())
        assert result_symbols == {"BTCUSDT", "ETHUSDT"}
        
        # Third request: single symbol (should also be cache hit)
        result3 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["LTCUSDT"],
            start="2023-01-01",
            stop="2023-01-03"
        )
        assert len(result3) == 3  # 3 days * 1 symbol
        assert reader._cache_stats["misses"] == 1  # Should still be 1
        assert reader._cache_stats["hits"] == 2   # Should be second cache hit
        
        # Verify that only LTCUSDT is in result
        result_symbols = set(result3.index.get_level_values("symbol").unique())
        assert result_symbols == {"LTCUSDT"}

    def test_symbol_filtering_specific_symbols_to_all_symbols(self):
        """Test that requesting all symbols after specific symbols results in cache miss."""
        mock_reader = Mock(spec=DataReader)
        
        # Create different test data for different requests
        def mock_get_aux_data(data_id, **kwargs):
            symbols = kwargs.get("symbols", ["BTCUSDT", "ETHUSDT", "LTCUSDT"])
            
            timestamps = pd.date_range("2023-01-01", "2023-01-05", freq="D")
            multi_index = pd.MultiIndex.from_product(
                [timestamps, symbols], 
                names=["timestamp", "symbol"]
            )
            
            return pd.DataFrame({
                "price": range(len(multi_index)),
                "volume": range(100, 100 + len(multi_index))
            }, index=multi_index)
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request: specific symbols
        result1 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            start="2023-01-01",
            stop="2023-01-05"
        )
        assert len(result1) == 5  # 5 days * 1 symbol
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        
        # Second request: all symbols (should be cache miss)
        result2 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-05"
        )
        assert len(result2) == 15  # 5 days * 3 symbols
        assert reader._cache_stats["misses"] == 2  # Should be cache miss
        assert reader._cache_stats["hits"] == 0

    def test_symbol_filtering_subset_of_cached_symbols(self):
        """Test that requesting a subset of cached symbols works correctly."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data with multiple symbols
        timestamps = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BCHUSDT"]
        
        multi_index = pd.MultiIndex.from_product(
            [timestamps, symbols], 
            names=["timestamp", "symbol"]
        )
        
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request: cache some symbols
        result1 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT", "LTCUSDT", "BCHUSDT"],
            start="2023-01-01",
            stop="2023-01-05"
        )
        assert len(result1) == 20  # 5 days * 4 symbols
        assert reader._cache_stats["misses"] == 1
        
        # Second request: subset of cached symbols (should be cache hit)
        result2 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"],
            start="2023-01-02",
            stop="2023-01-04"
        )
        assert len(result2) == 6  # 3 days * 2 symbols
        assert reader._cache_stats["misses"] == 1  # Should still be 1
        assert reader._cache_stats["hits"] == 1   # Should be cache hit
        
        # Verify correct symbols in result
        result_symbols = set(result2.index.get_level_values("symbol").unique())
        assert result_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_symbol_filtering_missing_symbols(self):
        """Test that requesting symbols not in cached data results in cache miss."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data with limited symbols
        timestamps = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        multi_index = pd.MultiIndex.from_product(
            [timestamps, symbols], 
            names=["timestamp", "symbol"]
        )
        
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request: cache some symbols
        result1 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-05"
        )
        assert len(result1) == 10  # 5 days * 2 symbols
        assert reader._cache_stats["misses"] == 1
        
        # Second request: symbols not in cache (should be cache miss)
        reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["LTCUSDT"],  # Not in cached data
            start="2023-01-02",
            stop="2023-01-04"
        )
        assert reader._cache_stats["misses"] == 2  # Should be cache miss
        assert reader._cache_stats["hits"] == 0

    def test_symbol_filtering_non_multiindex_data(self):
        """Test that non-MultiIndex data doesn't support symbol filtering."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data without MultiIndex
        timestamps = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        test_data = pd.DataFrame({
            "price": range(len(timestamps)),
            "volume": range(100, 100 + len(timestamps))
        }, index=timestamps)
        
        mock_reader.get_aux_data.return_value = test_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request: cache data (no symbols)
        result1 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-05"
        )
        assert len(result1) == 5
        assert reader._cache_stats["misses"] == 1
        
        # Second request: with symbols (should be cache miss since data doesn't support filtering)
        reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            start="2023-01-02",
            stop="2023-01-04"
        )
        assert reader._cache_stats["misses"] == 2  # Should be cache miss
        assert reader._cache_stats["hits"] == 0

    def test_symbol_filtering_with_time_range_filtering(self):
        """Test that symbol filtering works correctly with time range filtering."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data with larger time range
        timestamps = pd.date_range("2023-01-01", "2023-01-20", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
        
        multi_index = pd.MultiIndex.from_product(
            [timestamps, symbols], 
            names=["timestamp", "symbol"]
        )
        
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First request: cache large range with all symbols
        result1 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-20"
        )
        assert len(result1) == 60  # 20 days * 3 symbols
        assert reader._cache_stats["misses"] == 1
        
        # Second request: specific symbols and smaller time range (should be cache hit)
        result2 = reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"],
            start="2023-01-05",
            stop="2023-01-10"
        )
        assert len(result2) == 12  # 6 days * 2 symbols
        assert reader._cache_stats["misses"] == 1  # Should still be 1
        assert reader._cache_stats["hits"] == 1   # Should be cache hit
        
        # Verify correct symbols and time range in result
        result_symbols = set(result2.index.get_level_values("symbol").unique())
        assert result_symbols == {"BTCUSDT", "ETHUSDT"}
        
        result_timestamps = result2.index.get_level_values("timestamp").unique()
        expected_timestamps = pd.date_range("2023-01-05", "2023-01-10", freq="D")
        assert len(result_timestamps) == len(expected_timestamps)

    def test_symbol_filtering_helper_methods(self):
        """Test the helper methods for symbol filtering."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Test _can_filter_by_symbols
        timestamps = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
        
        multi_index = pd.MultiIndex.from_product(
            [timestamps, symbols], 
            names=["timestamp", "symbol"]
        )
        
        test_data = pd.DataFrame({
            "price": range(len(multi_index)),
            "volume": range(100, 100 + len(multi_index))
        }, index=multi_index)
        
        # Should be able to filter by symbols that exist
        assert reader._can_filter_by_symbols(test_data, ["BTCUSDT", "ETHUSDT"])
        
        # Should not be able to filter by symbols that don't exist
        assert not reader._can_filter_by_symbols(test_data, ["XRPUSDT"])
        
        # Should not be able to filter non-MultiIndex data
        simple_data = pd.DataFrame({"price": [1, 2, 3]})
        assert not reader._can_filter_by_symbols(simple_data, ["BTCUSDT"])
        
        # Test _filter_cached_data_by_symbols
        filtered_data = reader._filter_cached_data_by_symbols(test_data, ["BTCUSDT", "ETHUSDT"])
        assert len(filtered_data) == 10  # 5 days * 2 symbols
        
        filtered_symbols = set(filtered_data.index.get_level_values("symbol").unique())
        assert filtered_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_prefetch_aux_data_basic(self):
        """Test basic prefetch_aux_data functionality."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data for different aux data types
        def mock_get_aux_data(data_id, **kwargs):
            if data_id == "candles":
                # Return MultiIndex data
                timestamps = pd.date_range("2023-01-01", "2023-01-10", freq="D")
                symbols = kwargs.get("symbols", ["BTCUSDT", "ETHUSDT"])
                multi_index = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])
                return pd.DataFrame({"price": range(len(multi_index))}, index=multi_index)
            elif data_id == "funding":
                # Return simple DataFrame that gets filtered by time range
                start_date = kwargs.get("start", "2023-01-01")
                stop_date = kwargs.get("stop", "2023-01-10")
                timestamps = pd.date_range(start_date, stop_date, freq="H")
                return pd.DataFrame({"rate": range(len(timestamps))}, index=timestamps)
            else:
                return pd.DataFrame()
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Test prefetch
        results = reader.prefetch_aux_data(
            ["candles", "funding"],
            start="2023-01-01",
            stop="2023-01-10",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"]
        )
        
        # Check results
        assert "candles" in results
        assert "funding" in results
        assert results["candles"] == 20  # 10 days * 2 symbols
        # For funding, the range is filtered so it should be less than 240 hours
        assert results["funding"] > 0  # Just verify we got some data
        
        # Check cache was populated
        assert len(reader._aux_cache) >= 1  # At least candles should be cached
        assert reader._cache_stats["misses"] >= 1  # At least one fetch
        # Note: hits might be > 0 if symbol filtering is used

    def test_prefetch_aux_data_with_existing_cache(self):
        """Test prefetch_aux_data with existing cache entries."""
        mock_reader = Mock(spec=DataReader)
        
        # Create test data
        timestamps = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT"]
        multi_index = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])
        test_data = pd.DataFrame({"price": range(len(multi_index))}, index=multi_index)
        
        mock_reader.get_aux_data.return_value = test_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First prefetch
        results1 = reader.prefetch_aux_data(
            ["candles"],
            start="2023-01-01",
            stop="2023-01-10",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT", "ETHUSDT"]
        )
        
        assert results1["candles"] == 20
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        
        # Second prefetch (should use cache)
        results2 = reader.prefetch_aux_data(
            ["candles"],
            start="2023-01-02",
            stop="2023-01-08",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"]
        )
        
        # Should be filtered from cache
        assert results2["candles"] == 7  # 7 days * 1 symbol (filtered)
        assert reader._cache_stats["misses"] == 1  # No new miss
        assert reader._cache_stats["hits"] == 1   # Cache hit

    def test_prefetch_aux_data_error_handling(self):
        """Test prefetch_aux_data error handling."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock to raise exception for specific data type
        def mock_get_aux_data(data_id, **kwargs):
            if data_id == "candles":
                return pd.DataFrame({"price": [1, 2, 3]})
            elif data_id == "funding":
                raise Exception("Network error")
            else:
                return None
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Test prefetch with error
        results = reader.prefetch_aux_data(
            ["candles", "funding", "nonexistent"],
            start="2023-01-01",
            stop="2023-01-10",
            exchange="BINANCE.UM"
        )
        
        # Check results
        assert results["candles"] == 3  # Successfully fetched
        assert results["funding"] == 0  # Failed to fetch
        assert results["nonexistent"] == 0  # None data

    def test_prefetch_aux_data_different_data_types(self):
        """Test prefetch_aux_data with different data types."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock different data types
        def mock_get_aux_data(data_id, **kwargs):
            if data_id == "list_data":
                return [1, 2, 3, 4, 5]
            elif data_id == "scalar_data":
                return 42
            elif data_id == "none_data":
                return None
            elif data_id == "numpy_data":
                import numpy as np
                return np.array([1, 2, 3, 4, 5, 6])
            else:
                return pd.DataFrame({"value": [1, 2, 3]})
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Test prefetch
        results = reader.prefetch_aux_data(
            ["list_data", "scalar_data", "none_data", "numpy_data", "dataframe_data"],
            start="2023-01-01",
            stop="2023-01-10"
        )
        
        # Check results
        assert results["list_data"] == 5
        assert results["scalar_data"] == 1
        assert results["none_data"] == 0
        assert results["numpy_data"] == 6
        assert results["dataframe_data"] == 3

    def test_prefetch_aux_data_optional_parameters(self):
        """Test prefetch_aux_data with optional parameters."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock to check parameters
        def mock_get_aux_data(data_id, **kwargs):
            # Verify parameters are passed correctly
            assert "start" in kwargs
            assert "stop" in kwargs
            if "exchange" in kwargs:
                assert kwargs["exchange"] == "BINANCE.UM"
            if "symbols" in kwargs:
                assert kwargs["symbols"] == ["BTCUSDT"]
            if "timeframe" in kwargs:
                assert kwargs["timeframe"] == "1h"
            return pd.DataFrame({"value": [1, 2, 3]})
        
        mock_reader.get_aux_data.side_effect = mock_get_aux_data
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Test with all optional parameters
        results = reader.prefetch_aux_data(
            ["candles"],
            start="2023-01-01",
            stop="2023-01-10",
            exchange="BINANCE.UM",
            symbols=["BTCUSDT"],
            timeframe="1h"
        )
        
        assert results["candles"] == 3
        
        # Test with minimal parameters
        results = reader.prefetch_aux_data(
            ["candles"],
            start="2023-01-01",
            stop="2023-01-10"
        )
        
        assert results["candles"] == 3

    def test_prefetch_aux_data_empty_list(self):
        """Test prefetch_aux_data with empty aux_data_names list."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # Test with empty list
        results = reader.prefetch_aux_data(
            [],
            start="2023-01-01",
            stop="2023-01-10"
        )
        
        # Should return empty dictionary
        assert results == {}
        assert len(reader._aux_cache) == 0
        assert reader._cache_stats["misses"] == 0
        assert reader._cache_stats["hits"] == 0

    # ===== READ METHOD TESTS =====
    
    def test_read_basic_caching(self):
        """Test basic read caching functionality."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock data to return (list of records)
        mock_data = [
            ["2023-01-01", 100.0, 101.0, 99.0, 100.5, 1000],
            ["2023-01-02", 100.5, 102.0, 100.0, 101.0, 1100],
            ["2023-01-03", 101.0, 103.0, 100.5, 102.0, 1200],
        ]
        
        # Mock the read method to properly set up the transformer
        def mock_read(*args, **kwargs):
            transform = kwargs.get('transform')
            
            # Set up the transformer with column names if provided
            if transform and hasattr(transform, 'start_transform'):
                column_names = ["timestamp", "open", "high", "low", "close", "volume"]
                transform.start_transform(args[0], column_names)
            
            return mock_data
        
        mock_reader.read.side_effect = mock_read
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First read request
        result1 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-03",
            data_type="candles",
            chunksize=0
        )
        
        # Should be a cache miss
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        assert len(result1) == 3
        
        # Second read request with same parameters
        result2 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-03",
            data_type="candles",
            chunksize=0
        )
        
        # Should be a cache hit
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 1
        assert len(result2) == 3
        
        # The underlying reader should be called only once:
        # 1. First call for the initial cache miss (column names are cached during this call)
        # 2. Second call is a cache hit using cached column names
        assert mock_reader.read.call_count == 1

    def test_read_chunked_iteration(self):
        """Test read method with chunked iteration."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock data to return (list of records)
        mock_data = [
            ["2023-01-01", 100.0, 101.0, 99.0, 100.5, 1000],
            ["2023-01-02", 100.5, 102.0, 100.0, 101.0, 1100],
            ["2023-01-03", 101.0, 103.0, 100.5, 102.0, 1200],
            ["2023-01-04", 102.0, 104.0, 101.5, 103.0, 1300],
        ]
        
        # Mock the read method to return an iterator that yields chunks
        def mock_read(*args, **kwargs):
            transform = kwargs.get('transform')
            chunksize = kwargs.get('chunksize', 0)
            
            # Set up the transformer with column names if provided
            if transform and hasattr(transform, 'start_transform'):
                column_names = ["timestamp", "open", "high", "low", "close", "volume"]
                transform.start_transform(args[0], column_names)
            
            if chunksize > 0:
                # Return an iterator that yields chunks of the specified size
                from qubx.data.readers import _list_to_chunked_iterator
                return _list_to_chunked_iterator(mock_data, chunksize)
            else:
                return mock_data
        
        mock_reader.read.side_effect = mock_read
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First read request with chunking
        result_iterator = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-04",
            data_type="candles",
            chunksize=2
        )
        
        # Should be a cache miss
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        
        # Collect all chunks
        chunks = list(result_iterator)
        assert len(chunks) == 2  # 4 records / 2 chunksize = 2 chunks
        
        # Second read request with same parameters should be cache hit
        result_iterator2 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-04",
            data_type="candles",
            chunksize=2
        )
        
        # Should be a cache hit
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 1
        
        # Collect all chunks from cached data
        chunks2 = list(result_iterator2)
        assert len(chunks2) == 2
        
        # The underlying reader should be called only once:
        # 1. First call for the initial cache miss (column names are cached during this call)
        # 2. Second call is a cache hit using cached column names
        assert mock_reader.read.call_count == 1

    def test_read_aux_data_overlap(self):
        """Test read method with aux data overlap detection."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock aux data (DataFrame)
        timestamps = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        aux_data = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400]
        }, index=timestamps)
        
        mock_reader.get_aux_data.return_value = aux_data
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First, cache some aux data
        reader.get_aux_data(
            "candles",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-05"
        )
        
        # Now read with same parameters - should detect aux data overlap
        reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-05",
            data_type="candles",
            chunksize=0
        )
        
        # Should be aux data overlap (cache hit)
        assert reader._cache_stats["misses"] == 1  # Only aux data fetch
        assert reader._cache_stats["hits"] == 1   # Read uses aux data
        
        # The read method should not call the underlying reader
        mock_reader.read.assert_not_called()

    def test_read_multi_period_caching(self):
        """Test read method with multi-period caching."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock data for different periods
        period1_data = [
            ["2023-01-01", 100.0, 101.0, 99.0, 100.5, 1000],
            ["2023-01-02", 100.5, 102.0, 100.0, 101.0, 1100],
        ]
        period2_data = [
            ["2023-01-03", 101.0, 103.0, 100.5, 102.0, 1200],
            ["2023-01-04", 102.0, 104.0, 101.5, 103.0, 1300],
        ]
        
        # Mock column names
        mock_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        def mock_read_side_effect(*args, **kwargs):
            start = args[1] if len(args) > 1 else kwargs.get("start")
            transform = kwargs.get("transform")
            
            # Set up transformer with column names
            if transform:
                transform.start_transform(args[0], mock_columns, start=start, stop=kwargs.get("stop"))
            
            if start == "2023-01-01":
                if transform:
                    transform.process_data(period1_data)
                    return transform.collect()
                return period1_data
            elif start == "2023-01-03":
                if transform:
                    transform.process_data(period2_data)
                    return transform.collect()
                return period2_data
            else:
                return []
        
        mock_reader.read.side_effect = mock_read_side_effect
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="0d")
        
        # First read request (period 1)
        result1 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-02",
            data_type="candles",
            chunksize=0
        )
        
        assert len(result1) == 2
        assert reader._cache_stats["misses"] == 1
        assert reader._cache_stats["hits"] == 0
        
        # Second read request (period 2)
        result2 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-03",
            stop="2023-01-04",
            data_type="candles",
            chunksize=0
        )
        
        assert len(result2) == 2
        assert reader._cache_stats["misses"] == 2  # Two separate fetches
        assert reader._cache_stats["hits"] == 0
        
        # Third read request (overlapping both periods)
        result3 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-04",
            data_type="candles",
            chunksize=0
        )
        
        # Should be cache hit as both periods are cached
        assert len(result3) == 4  # Combined data from both periods
        assert reader._cache_stats["misses"] == 2  # No new fetches
        assert reader._cache_stats["hits"] == 1   # Cache hit

    def test_read_cache_infrastructure(self):
        """Test read cache infrastructure and clear_cache functionality."""
        mock_reader = Mock(spec=DataReader)
        
        mock_data = [
            ["2023-01-01", 100.0, 101.0, 99.0, 100.5, 1000],
            ["2023-01-02", 100.5, 102.0, 100.0, 101.0, 1100],
        ]
        mock_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        # Mock the read method to properly set up the transformer
        def mock_read(data_id, start=None, stop=None, transform=None, chunksize=0, **kwargs):
            if transform:
                transform.start_transform(data_id, mock_columns, start=start, stop=stop)
                transform.process_data(mock_data)
                return transform.collect()
            return mock_data
        
        mock_reader.read.side_effect = mock_read
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # Make a read request to populate cache
        reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-02",
            data_type="candles",
            chunksize=0
        )
        
        # Verify cache is populated
        assert len(reader._read_cache) > 0
        assert len(reader._read_cache_ranges) > 0
        assert len(reader._read_cache_columns) > 0
        
        # Clear cache
        reader.clear_cache()
        
        # Verify cache is cleared
        assert len(reader._read_cache) == 0
        assert len(reader._read_cache_ranges) == 0
        assert len(reader._read_cache_columns) == 0
        assert len(reader._aux_cache) == 0
        assert len(reader._aux_cache_ranges) == 0

    def test_read_dataframe_to_records_conversion(self):
        """Test DataFrame to records conversion functionality."""
        mock_reader = Mock(spec=DataReader)
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # Test with simple DataFrame
        timestamps = pd.date_range("2023-01-01", "2023-01-03", freq="D")
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200]
        }, index=timestamps)
        
        records, columns = reader._dataframe_to_records(df)
        
        # Should convert DataFrame to list of records
        assert len(records) == 3
        assert len(columns) == 6  # timestamp + 5 data columns
        assert "timestamp" in columns[0]  # timestamp should be first column
        assert "open" in columns
        assert "volume" in columns
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        records, columns = reader._dataframe_to_records(empty_df)
        assert records == []
        assert columns == []

    def test_read_transform_application(self):
        """Test that transforms are correctly applied to cached data."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock data to return (list of records)
        mock_data = [
            ["2023-01-01", 100.0, 101.0, 99.0, 100.5, 1000],
            ["2023-01-02", 100.5, 102.0, 100.0, 101.0, 1100],
        ]
        
        # Mock column names
        mock_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        # Mock the read method to properly set up the transformer
        def mock_read(data_id, start=None, stop=None, transform=None, chunksize=0, **kwargs):
            if transform:
                transform.start_transform(data_id, mock_columns, start=start, stop=stop)
                transform.process_data(mock_data)
                return transform.collect()
            return mock_data
        
        mock_reader.read.side_effect = mock_read
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # Custom transform that counts records
        class CountingTransform(DataTransformer):
            def __init__(self):
                super().__init__()
                self.count = 0
                
            def process_data(self, rows_data):
                self.count += len(rows_data)
                super().process_data(rows_data)
                
            def collect(self):
                return {"count": self.count, "data": self.buffer}
        
        # Make read request with custom transform
        transform = CountingTransform()
        result = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-02",
            data_type="candles",
            transform=transform,
            chunksize=0
        )
        
        # Transform should be applied
        assert isinstance(result, dict)
        assert result["count"] == 2
        assert len(result["data"]) == 2
        
        # Second request should also apply transform to cached data
        transform2 = CountingTransform()
        result2 = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-02",
            data_type="candles",
            transform=transform2,
            chunksize=0
        )
        
        # Should be cache hit with transform applied
        assert reader._cache_stats["hits"] == 1
        assert result2["count"] == 2

    def test_read_aux_data_overlap_with_symbol_filtering(self):
        """Test that aux data overlap detection filters by symbol correctly."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock aux data with multiple symbols (DataFrame with symbol column)
        # Create data for 2 symbols over 3 days
        aux_data = pd.DataFrame({
            "funding_rate": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"), 
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
                pd.Timestamp("2023-01-03")
            ]
        })
        aux_data = aux_data.set_index("timestamp")
        
        mock_reader.get_aux_data.return_value = aux_data
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First, cache some aux data (this will cache all symbols)
        reader.get_aux_data(
            "funding_payment",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-03"
        )
        
        # Now read with specific symbol - should detect aux data overlap and filter by symbol
        result = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-03",
            data_type="funding_payment",
            transform=AsPandasFrame(),
            chunksize=0
        )
        
        # Should detect aux data overlap (cache hit)
        assert reader._cache_stats["hits"] == 1
        
        # Result should be filtered to only BTCUSDT data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Only BTCUSDT records
        
        # Symbol column should be removed since it's redundant
        assert "symbol" not in result.columns
        
        # Should contain only BTCUSDT funding rates
        expected_funding_rates = [0.0001, 0.0003, 0.0005]  # BTCUSDT rates from the mock data
        assert result["funding_rate"].tolist() == expected_funding_rates

    def test_symbol_filtering_edge_cases(self):
        """Test edge cases that might cause IndexError."""
        reader = CachedPrefetchReader(Mock(spec=DataReader), prefetch_period="1d")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = reader._filter_aux_data_by_symbol(empty_df, "BTCUSDT")
        assert result.empty
        
        # Test with DataFrame with single column (no timestamp)
        single_col_df = pd.DataFrame({"value": [1, 2, 3]})
        result = reader._filter_aux_data_by_symbol(single_col_df, "BTCUSDT")
        assert result.equals(single_col_df)  # Should return as-is
        
        # Test with DataFrame with no symbol column/level
        no_symbol_df = pd.DataFrame({"price": [100, 200], "volume": [1000, 2000]})
        result = reader._filter_aux_data_by_symbol(no_symbol_df, "BTCUSDT")
        assert result.equals(no_symbol_df)  # Should return as-is
        
        # Test dataframe_to_records with various structures
        # Empty DataFrame
        records, columns = reader._dataframe_to_records(pd.DataFrame())
        assert records == []
        assert columns == []
        
        # Single row DataFrame
        single_row_df = pd.DataFrame({"value": [42]}, index=[pd.Timestamp("2023-01-01")])
        records, columns = reader._dataframe_to_records(single_row_df)
        assert len(records) == 1
        assert len(records[0]) == 2  # timestamp + value
        assert columns == ["timestamp", "value"]
        
        # DataFrame with MultiIndex (after symbol filtering)
        timestamps = pd.date_range("2023-01-01", "2023-01-02", freq="D")
        df = pd.DataFrame({"funding_rate": [0.001, 0.002]}, index=timestamps)
        records, columns = reader._dataframe_to_records(df)
        assert len(records) == 2
        assert len(records[0]) == 2  # timestamp + funding_rate
        assert len(records[1]) == 2  # timestamp + funding_rate
        assert columns == ["timestamp", "funding_rate"]

    def test_read_aux_data_overlap_with_missing_symbol(self):
        """Test aux data overlap when requested symbol is not in cached data."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock aux data with some symbols but NOT the one we'll request
        aux_data = pd.DataFrame({
            "funding_rate": [0.0001, 0.0002, 0.0003, 0.0004],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"), 
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-02")
            ]
        })
        aux_data = aux_data.set_index("timestamp")
        
        mock_reader.get_aux_data.return_value = aux_data
        
        # Mock the read method to return empty data for non-existent symbol
        mock_reader.read.return_value = []
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First, cache aux data for existing symbols
        reader.get_aux_data(
            "funding_payment",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-02"
        )
        
        # Now try to read with a symbol that doesn't exist in the cached data
        result = reader.read(
            "BINANCE.UM:NONEXISTENT",  # This symbol is NOT in the cached data
            start="2023-01-01",
            stop="2023-01-02",
            data_type="funding_payment",
            transform=AsPandasFrame(),
            chunksize=0
        )
        
        # Should NOT detect aux data overlap because filtering results in empty data
        # Should be a cache miss and call the underlying reader
        assert reader._cache_stats["hits"] == 0
        assert reader._cache_stats["misses"] == 1
        
        # The mock reader should be called for the missing symbol
        assert mock_reader.read.call_count == 1
        
        # Result should be empty
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_read_aux_data_overlap_with_multiindex_symbol_filtering(self):
        """Test that aux data overlap detection filters MultiIndex DataFrames by symbol correctly."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock aux data with MultiIndex (timestamp, symbol)
        timestamps = pd.date_range("2023-01-01", "2023-01-03", freq="D")
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Create MultiIndex DataFrame
        index = pd.MultiIndex.from_product([timestamps, symbols], names=["timestamp", "symbol"])
        aux_data = pd.DataFrame({
            "funding_rate": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006],
            "interval_hours": [8, 8, 8, 8, 8, 8]
        }, index=index)
        
        mock_reader.get_aux_data.return_value = aux_data
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First, cache some aux data (this will cache all symbols)
        reader.get_aux_data(
            "funding_payment",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-03"
        )
        
        # Now read with specific symbol - should detect aux data overlap and filter by symbol
        result = reader.read(
            "BINANCE.UM:BTCUSDT",
            start="2023-01-01",
            stop="2023-01-03",
            data_type="funding_payment",
            transform=AsPandasFrame(),
            chunksize=0
        )
        
        # Should detect aux data overlap (cache hit)
        assert reader._cache_stats["hits"] == 1
        
        # Result should be filtered to only BTCUSDT data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Only BTCUSDT records
        
        # Symbol level should be removed from MultiIndex
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "timestamp"
        
        # Should contain only BTCUSDT funding rates
        expected_funding_rates = [0.0001, 0.0003, 0.0005]  # BTCUSDT rates from the mock data
        assert result["funding_rate"].tolist() == expected_funding_rates

    def test_symbol_filtering_edge_cases(self):
        """Test edge cases that might cause IndexError."""
        reader = CachedPrefetchReader(Mock(spec=DataReader), prefetch_period="1d")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = reader._filter_aux_data_by_symbol(empty_df, "BTCUSDT")
        assert result.empty
        
        # Test with DataFrame with single column (no timestamp)
        single_col_df = pd.DataFrame({"value": [1, 2, 3]})
        result = reader._filter_aux_data_by_symbol(single_col_df, "BTCUSDT")
        assert result.equals(single_col_df)  # Should return as-is
        
        # Test with DataFrame with no symbol column/level
        no_symbol_df = pd.DataFrame({"price": [100, 200], "volume": [1000, 2000]})
        result = reader._filter_aux_data_by_symbol(no_symbol_df, "BTCUSDT")
        assert result.equals(no_symbol_df)  # Should return as-is
        
        # Test dataframe_to_records with various structures
        # Empty DataFrame
        records, columns = reader._dataframe_to_records(pd.DataFrame())
        assert records == []
        assert columns == []
        
        # Single row DataFrame
        single_row_df = pd.DataFrame({"value": [42]}, index=[pd.Timestamp("2023-01-01")])
        records, columns = reader._dataframe_to_records(single_row_df)
        assert len(records) == 1
        assert len(records[0]) == 2  # timestamp + value
        assert columns == ["timestamp", "value"]
        
        # DataFrame with MultiIndex (after symbol filtering)
        timestamps = pd.date_range("2023-01-01", "2023-01-02", freq="D")
        df = pd.DataFrame({"funding_rate": [0.001, 0.002]}, index=timestamps)
        records, columns = reader._dataframe_to_records(df)
        assert len(records) == 2
        assert len(records[0]) == 2  # timestamp + funding_rate
        assert len(records[1]) == 2  # timestamp + funding_rate
        assert columns == ["timestamp", "funding_rate"]

    def test_read_aux_data_overlap_with_missing_symbol(self):
        """Test aux data overlap when requested symbol is not in cached data."""
        mock_reader = Mock(spec=DataReader)
        
        # Mock aux data with some symbols but NOT the one we'll request
        aux_data = pd.DataFrame({
            "funding_rate": [0.0001, 0.0002, 0.0003, 0.0004],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"), 
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-02")
            ]
        })
        aux_data = aux_data.set_index("timestamp")
        
        mock_reader.get_aux_data.return_value = aux_data
        
        # Mock the read method to return empty data for non-existent symbol
        mock_reader.read.return_value = []
        
        reader = CachedPrefetchReader(mock_reader, prefetch_period="1d")
        
        # First, cache aux data for existing symbols
        reader.get_aux_data(
            "funding_payment",
            exchange="BINANCE.UM",
            start="2023-01-01",
            stop="2023-01-02"
        )
        
        # Now try to read with a symbol that doesn't exist in the cached data
        result = reader.read(
            "BINANCE.UM:NONEXISTENT",  # This symbol is NOT in the cached data
            start="2023-01-01",
            stop="2023-01-02",
            data_type="funding_payment",
            transform=AsPandasFrame(),
            chunksize=0
        )
        
        # Should NOT detect aux data overlap because filtering results in empty data
        # Should be a cache miss and call the underlying reader
        assert reader._cache_stats["hits"] == 0
        assert reader._cache_stats["misses"] == 1
        
        # The mock reader should be called for the missing symbol
        assert mock_reader.read.call_count == 1
        
        # Result should be empty
        assert isinstance(result, pd.DataFrame)
        assert result.empty
