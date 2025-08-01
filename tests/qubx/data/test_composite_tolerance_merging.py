"""Tests for CompositeReader tolerance-based deduplication functionality."""

from unittest.mock import Mock

import pandas as pd
import pytest

from qubx.data.composite import CompositeReader
from qubx.data.readers import DataReader


class TestCompositeReaderToleranceMerging:
    """Test suite for CompositeReader tolerance-based deduplication."""

    def test_tolerance_deduplication_multiindex_keep_last(self):
        """Test tolerance-based deduplication with MultiIndex data, keeping last."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        # Create data with near-duplicate timestamps (30 seconds apart)
        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        # Reader 1 data (earlier timestamps)
        data1 = pd.DataFrame({
            'funding_rate': [0.0001, 0.0002],
            'source': ['reader1', 'reader1']
        })
        data1.index = pd.MultiIndex.from_tuples([
            (base_time, 'BTCUSDT'),
            (base_time + pd.Timedelta(hours=8), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        # Reader 2 data (30 seconds later - should be kept with default keep='last')
        data2 = pd.DataFrame({
            'funding_rate': [0.00011, 0.00021],  # Slightly different values
            'source': ['reader2', 'reader2']
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(seconds=30), 'BTCUSDT'),
            (base_time + pd.Timedelta(hours=8, seconds=30), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Test with default tolerance (1min) and keep='last'
        result = composite_reader.get_aux_data('funding_payment')

        # Should keep 2 records (one per timestamp group), with later timestamps
        assert len(result) == 2
        result_timestamps = result.index.get_level_values('timestamp')
        
        # Should keep the later timestamps (from reader2)
        expected_timestamps = [
            base_time + pd.Timedelta(seconds=30),
            base_time + pd.Timedelta(hours=8, seconds=30)
        ]
        assert all(ts in expected_timestamps for ts in result_timestamps)
        
        # Should have reader2 data
        assert all(result['source'] == 'reader2')

    def test_tolerance_deduplication_multiindex_keep_first(self):
        """Test tolerance-based deduplication with MultiIndex data, keeping first."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        # Reader 1 data (earlier timestamps)
        data1 = pd.DataFrame({
            'funding_rate': [0.0001, 0.0002],
            'source': ['reader1', 'reader1']
        })
        data1.index = pd.MultiIndex.from_tuples([
            (base_time, 'BTCUSDT'),
            (base_time + pd.Timedelta(hours=8), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        # Reader 2 data (30 seconds later)
        data2 = pd.DataFrame({
            'funding_rate': [0.00011, 0.00021],
            'source': ['reader2', 'reader2']
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(seconds=30), 'BTCUSDT'),
            (base_time + pd.Timedelta(hours=8, seconds=30), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Test with keep='first'
        result = composite_reader.get_aux_data('funding_payment', keep='first')

        # Should keep 2 records with earlier timestamps
        assert len(result) == 2
        result_timestamps = result.index.get_level_values('timestamp')
        
        # Should keep the earlier timestamps (from reader1)
        expected_timestamps = [base_time, base_time + pd.Timedelta(hours=8)]
        assert all(ts in expected_timestamps for ts in result_timestamps)
        
        # Should have reader1 data
        assert all(result['source'] == 'reader1')

    def test_tolerance_deduplication_per_symbol(self):
        """Test that deduplication is done per symbol independently."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        # Reader 1: BTC and ETH data
        data1 = pd.DataFrame({
            'funding_rate': [0.0001, 0.0003],
            'source': ['reader1', 'reader1']
        })
        data1.index = pd.MultiIndex.from_tuples([
            (base_time, 'BTCUSDT'),
            (base_time, 'ETHUSDT'),
        ], names=['timestamp', 'symbol'])

        # Reader 2: BTC and ETH data (30 seconds later)
        data2 = pd.DataFrame({
            'funding_rate': [0.00011, 0.00031],
            'source': ['reader2', 'reader2']
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(seconds=30), 'BTCUSDT'),
            (base_time + pd.Timedelta(seconds=30), 'ETHUSDT'),
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data('funding_payment')

        # Should have 2 records (one per symbol), both from reader2 (keep='last')
        assert len(result) == 2
        symbols = result.index.get_level_values('symbol').unique()
        assert set(symbols) == {'BTCUSDT', 'ETHUSDT'}
        assert all(result['source'] == 'reader2')

    def test_strict_tolerance_no_deduplication(self):
        """Test that strict tolerance prevents deduplication."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        data1 = pd.DataFrame({
            'funding_rate': [0.0001],
            'source': ['reader1']
        })
        data1.index = pd.MultiIndex.from_tuples([
            (base_time, 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        data2 = pd.DataFrame({
            'funding_rate': [0.00011],
            'source': ['reader2']
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(seconds=30), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Use strict 10-second tolerance - should not deduplicate 30-second difference
        result = composite_reader.get_aux_data('funding_payment', tolerance='10s')

        # Should keep both records
        assert len(result) == 2
        sources = result['source'].tolist()
        assert 'reader1' in sources
        assert 'reader2' in sources

    def test_zero_tolerance_no_deduplication(self):
        """Test that zero tolerance disables deduplication entirely."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        data1 = pd.DataFrame({
            'funding_rate': [0.0001],
            'source': ['reader1']
        })
        data1.index = pd.MultiIndex.from_tuples([
            (base_time, 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        data2 = pd.DataFrame({
            'funding_rate': [0.00011],
            'source': ['reader2']
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(milliseconds=1), 'BTCUSDT'),  # Just 1ms difference
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Use zero tolerance - should not deduplicate even 1ms difference
        result = composite_reader.get_aux_data('funding_payment', tolerance='0s')

        # Should keep both records
        assert len(result) == 2

    def test_tolerance_deduplication_single_index(self):
        """Test tolerance-based deduplication with single timestamp index."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        # Single index DataFrames (just timestamp)
        data1 = pd.DataFrame({
            'value': [100, 200],
            'source': ['reader1', 'reader1']
        }, index=[base_time, base_time + pd.Timedelta(hours=1)])

        data2 = pd.DataFrame({
            'value': [101, 201],  # Slightly different values
            'source': ['reader2', 'reader2']
        }, index=[
            base_time + pd.Timedelta(seconds=30),  # 30s later
            base_time + pd.Timedelta(hours=1, seconds=30)  # 30s later
        ])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data('test_data')

        # Should keep 2 records (one per time group), with later timestamps
        assert len(result) == 2
        assert all(result['source'] == 'reader2')  # keep='last' default

    def test_tolerance_deduplication_series(self):
        """Test tolerance-based deduplication with Series data."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        data1 = pd.Series([100, 200], index=[
            base_time, 
            base_time + pd.Timedelta(hours=1)
        ])

        data2 = pd.Series([101, 201], index=[
            base_time + pd.Timedelta(seconds=30),
            base_time + pd.Timedelta(hours=1, seconds=30)
        ])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data('test_data')

        # Should keep 2 values (one per time group)
        assert len(result) == 2
        assert result.iloc[0] == 101  # Later value from reader2
        assert result.iloc[1] == 201  # Later value from reader2

    def test_complex_timestamp_clustering(self):
        """Test complex scenarios with multiple timestamp clusters."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        # Reader 1: 6 timestamps in 3 clusters
        data1 = pd.DataFrame({
            'funding_rate': [0.001, 0.001, 0.002, 0.002, 0.003, 0.003],
            'source': ['r1'] * 6
        })
        data1.index = pd.MultiIndex.from_tuples([
            # Cluster 1: 08:00 area
            (base_time, 'BTCUSDT'),
            (base_time + pd.Timedelta(seconds=15), 'BTCUSDT'),
            # Cluster 2: 08:10 area  
            (base_time + pd.Timedelta(minutes=10), 'BTCUSDT'),
            (base_time + pd.Timedelta(minutes=10, seconds=20), 'BTCUSDT'),
            # Cluster 3: 08:20 area
            (base_time + pd.Timedelta(minutes=20), 'BTCUSDT'),
            (base_time + pd.Timedelta(minutes=20, seconds=10), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        # Reader 2: Adds one more timestamp to each cluster
        data2 = pd.DataFrame({
            'funding_rate': [0.0011, 0.0021, 0.0031],
            'source': ['r2'] * 3
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(seconds=45), 'BTCUSDT'),  # Cluster 1
            (base_time + pd.Timedelta(minutes=10, seconds=50), 'BTCUSDT'),  # Cluster 2
            (base_time + pd.Timedelta(minutes=20, seconds=30), 'BTCUSDT'),  # Cluster 3
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data('funding_payment')

        # Should have 3 records (one from each cluster), all from reader2 (latest in each cluster)
        assert len(result) == 3
        assert all(result['source'] == 'r2')
        
        # Check that we got the latest timestamp from each cluster
        result_timestamps = result.index.get_level_values('timestamp')
        expected_latest = [
            base_time + pd.Timedelta(seconds=45),  # Latest from cluster 1
            base_time + pd.Timedelta(minutes=10, seconds=50),  # Latest from cluster 2
            base_time + pd.Timedelta(minutes=20, seconds=30),  # Latest from cluster 3
        ]
        assert set(result_timestamps) == set(expected_latest)

    def test_custom_tolerance_parameter(self):
        """Test custom tolerance parameter values."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        base_time = pd.Timestamp('2025-07-31 08:00:00')
        
        data1 = pd.DataFrame({
            'funding_rate': [0.0001],
            'source': ['reader1']
        })
        data1.index = pd.MultiIndex.from_tuples([
            (base_time, 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])

        data2 = pd.DataFrame({
            'funding_rate': [0.00011],
            'source': ['reader2']
        })
        data2.index = pd.MultiIndex.from_tuples([
            (base_time + pd.Timedelta(minutes=2), 'BTCUSDT'),  # 2 minutes later
        ], names=['timestamp', 'symbol'])

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Test with 3-minute tolerance - should deduplicate
        result_3min = composite_reader.get_aux_data('funding_payment', tolerance='3min')
        assert len(result_3min) == 1
        assert result_3min['source'].iloc[0] == 'reader2'  # keep='last'

        # Test with 1-minute tolerance - should not deduplicate  
        result_1min = composite_reader.get_aux_data('funding_payment', tolerance='1min')
        assert len(result_1min) == 2

    def test_kwargs_passed_through_with_tolerance(self):
        """Test that additional kwargs are passed through correctly with tolerance merging."""
        mock_reader = Mock(spec=DataReader)
        
        test_data = pd.DataFrame({
            'funding_rate': [0.0001],
        })
        test_data.index = pd.MultiIndex.from_tuples([
            (pd.Timestamp('2025-07-31 08:00:00'), 'BTCUSDT'),
        ], names=['timestamp', 'symbol'])
        
        mock_reader.get_aux_data.return_value = test_data

        composite_reader = CompositeReader([mock_reader])

        result = composite_reader.get_aux_data(
            'funding_payment', 
            tolerance='30s',
            keep='first',
            exchange='BINANCE.UM',
            symbols=['BTCUSDT']
        )

        # Check that additional kwargs were passed through (excluding tolerance and keep)
        mock_reader.get_aux_data.assert_called_once_with(
            'funding_payment',
            exchange='BINANCE.UM', 
            symbols=['BTCUSDT']
        )
        
        assert result.equals(test_data)

    def test_backwards_compatibility(self):
        """Test that existing code without tolerance parameters still works."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader2 = Mock(spec=DataReader)

        # Simple non-overlapping data
        data1 = pd.DataFrame({'col1': [1, 2]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        data2 = pd.DataFrame({'col1': [3, 4]}, index=pd.to_datetime(['2023-01-03', '2023-01-04']))

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Call without any tolerance parameters - should use defaults
        result = composite_reader.get_aux_data('test_data')

        # Should concatenate normally since no overlapping timestamps
        expected = pd.concat([data1, data2]).sort_index()
        pd.testing.assert_frame_equal(result, expected)