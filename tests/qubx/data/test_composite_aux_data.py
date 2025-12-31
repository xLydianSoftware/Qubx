"""Tests for CompositeReader aux data merging functionality."""

from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from qubx.data.composite import CompositeReader
from qubx.data.readers import DataReader


class TestCompositeReaderAuxData:
    """Test suite for CompositeReader aux data functionality."""

    def test_get_aux_data_single_reader(self):
        """Test aux data retrieval with single reader."""
        # Create mock reader with aux data
        mock_reader = Mock(spec=DataReader)
        mock_reader.get_names.return_value = []
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        mock_reader.get_aux_data.return_value = test_data

        composite_reader = CompositeReader([mock_reader])

        result = composite_reader.get_aux_data("test_data")

        assert result.equals(test_data)
        mock_reader.get_aux_data.assert_called_once_with("test_data")

    def test_get_aux_data_no_readers_have_data(self):
        """Test that ValueError is raised when no readers have the requested data."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        mock_reader1.get_aux_data.side_effect = ValueError("No such data")
        mock_reader2.get_aux_data.side_effect = ValueError("No such data")

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        with pytest.raises(ValueError, match="No reader has auxiliary data for 'missing_data'"):
            composite_reader.get_aux_data("missing_data")

    def test_get_aux_data_first_strategy(self):
        """Test 'first' merge strategy returns data from first available reader."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        data1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        data2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="first")

        assert result.equals(data1)
        mock_reader1.get_aux_data.assert_called_once()
        mock_reader2.get_aux_data.assert_called_once()

    def test_get_aux_data_concat_strategy_dataframes(self):
        """Test 'concat' merge strategy with DataFrames."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Create DataFrames with different indices (like time series)
        data1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        data2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]}, index=pd.to_datetime(["2023-01-03", "2023-01-04"]))

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="concat")

        expected = pd.concat([data1, data2]).sort_index()
        pd.testing.assert_frame_equal(result, expected)

    def test_get_aux_data_concat_strategy_with_duplicates(self):
        """Test 'concat' merge strategy handles duplicate indices."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Create DataFrames with overlapping indices
        data1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        data2 = pd.DataFrame(
            {"col1": [5, 6], "col2": [7, 8]}, index=pd.to_datetime(["2023-01-02", "2023-01-03"])
        )  # Overlap on 2023-01-02

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="concat")

        # Should keep last occurrence of duplicate index (new tolerance-based behavior)
        assert len(result) == 3  # 3 unique timestamps
        assert result.loc["2023-01-02", "col1"] == 5  # From second reader (keep='last' default)

    def test_get_aux_data_concat_strategy_series(self):
        """Test 'concat' merge strategy with Series."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        data1 = pd.Series([1, 2], index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        data2 = pd.Series([3, 4], index=pd.to_datetime(["2023-01-03", "2023-01-04"]))

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="concat")

        expected = pd.concat([data1, data2]).sort_index()
        pd.testing.assert_series_equal(result, expected)

    def test_get_aux_data_default_concat_strategy(self):
        """Test default 'concat' merge strategy with DataFrames."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Create DataFrames with different indices (like time series)
        data1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        data2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]}, index=pd.to_datetime(["2023-01-03", "2023-01-04"]))

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Call without specifying merge_strategy - should use default "concat"
        result = composite_reader.get_aux_data("test_data")

        expected = pd.concat([data1, data2]).sort_index()
        pd.testing.assert_frame_equal(result, expected)

    def test_get_aux_data_outer_join_strategy_dataframes(self):
        """Test 'outer' join merge strategy with DataFrames."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Create DataFrames with different columns
        data1 = pd.DataFrame({"col1": [1, 2]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        data2 = pd.DataFrame({"col2": [3, 4]}, index=pd.to_datetime(["2023-01-01", "2023-01-03"]))

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="outer")

        # Should have outer join with all indices and columns
        assert len(result) == 3  # 3 unique timestamps
        assert "col1" in result.columns
        assert "col2_DataReader_1" in result.columns  # Suffixed to avoid conflicts

    def test_get_aux_data_inner_join_strategy_dataframes(self):
        """Test 'inner' join merge strategy with DataFrames."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Create DataFrames with some overlapping indices
        data1 = pd.DataFrame({"col1": [1, 2, 3]}, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
        data2 = pd.DataFrame({"col2": [4, 5]}, index=pd.to_datetime(["2023-01-02", "2023-01-03"]))

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="inner")

        # Should only include overlapping indices
        assert len(result) == 2  # Only 2023-01-02 and 2023-01-03
        assert "col1" in result.columns
        assert "col2_DataReader_1" in result.columns

    def test_get_aux_data_outer_join_strategy_series(self):
        """Test 'outer' join merge strategy with Series."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        data1 = pd.Series([1, 2], index=["a", "b"], name="series1")
        data2 = pd.Series([3, 4], index=["b", "c"], name="series2")

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="outer")

        # Should create DataFrame with series as columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # indices: a, b, c
        assert "DataReader_0" in result.columns
        assert "DataReader_1" in result.columns

    def test_get_aux_data_non_pandas_fallback(self):
        """Test fallback to 'first' strategy for non-pandas data."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Return non-pandas data
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="outer")

        # Should fallback to first strategy
        assert result == data1

    def test_get_aux_data_unknown_strategy_fallback(self):
        """Test fallback to 'first' strategy for unknown merge strategies."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        data1 = pd.DataFrame({"col1": [1, 2]})
        data2 = pd.DataFrame({"col2": [3, 4]})

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        result = composite_reader.get_aux_data("test_data", merge_strategy="unknown_strategy")

        # Should fallback to first data
        assert result.equals(data1)

    def test_get_aux_data_error_handling(self):
        """Test error handling during merge operations."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []

        # Create data that will cause a merge error (e.g., incompatible types)
        data1 = pd.DataFrame({"col1": [1, 2]})
        data2 = "incompatible_string_data"

        mock_reader1.get_aux_data.return_value = data1
        mock_reader2.get_aux_data.return_value = data2

        composite_reader = CompositeReader([mock_reader1, mock_reader2])

        # Should handle error and fallback to first strategy
        result = composite_reader.get_aux_data("test_data", merge_strategy="outer")
        assert result.equals(data1)

    def test_get_aux_data_reader_exceptions(self):
        """Test handling of exceptions from individual readers."""
        mock_reader1 = Mock(spec=DataReader)
        mock_reader1.get_names.return_value = []
        mock_reader2 = Mock(spec=DataReader)
        mock_reader2.get_names.return_value = []
        mock_reader3 = Mock(spec=DataReader)
        mock_reader3.get_names.return_value = []

        # First reader raises exception, second has ValueError, third succeeds
        mock_reader1.get_aux_data.side_effect = Exception("Reader error")
        mock_reader2.get_aux_data.side_effect = ValueError("No data")
        test_data = pd.DataFrame({"col1": [1, 2]})
        mock_reader3.get_aux_data.return_value = test_data

        composite_reader = CompositeReader([mock_reader1, mock_reader2, mock_reader3])

        result = composite_reader.get_aux_data("test_data")

        # Should skip first two readers and return data from third
        assert result.equals(test_data)
        mock_reader3.get_aux_data.assert_called_once_with("test_data")

    def test_get_aux_data_kwargs_passed_through(self):
        """Test that kwargs are passed through to underlying readers."""
        mock_reader = Mock(spec=DataReader)
        mock_reader.get_names.return_value = []
        test_data = pd.DataFrame({"col1": [1, 2]})
        mock_reader.get_aux_data.return_value = test_data

        composite_reader = CompositeReader([mock_reader])

        result = composite_reader.get_aux_data("test_data", start_date="2023-01-01", end_date="2023-01-31")

        mock_reader.get_aux_data.assert_called_once_with("test_data", start_date="2023-01-01", end_date="2023-01-31")
        assert result.equals(test_data)
