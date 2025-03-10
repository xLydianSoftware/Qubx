from unittest.mock import MagicMock, patch

import pytest

from qubx.utils.runner.configs import ReaderConfig, TypedReaderConfig, WarmupConfig
from qubx.utils.runner.runner import _create_data_type_readers


class TestCreateDataTypeReaders:
    """Tests for the _create_data_type_readers function."""

    @pytest.fixture
    def mock_reader(self):
        """Create a mock reader for testing."""
        return MagicMock()

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_empty_warmup_config(self, mock_construct_reader):
        """Test with empty warmup config."""
        # Create an empty warmup config
        warmup = WarmupConfig()

        # Call the function
        result = _create_data_type_readers(warmup)

        # Check that the result is an empty dictionary
        assert result == {}
        # Check that _construct_reader was not called
        mock_construct_reader.assert_not_called()

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_none_warmup_config(self, mock_construct_reader):
        """Test with None warmup config."""
        # Call the function with None
        result = _create_data_type_readers(None)

        # Check that the result is an empty dictionary
        assert result == {}
        # Check that _construct_reader was not called
        mock_construct_reader.assert_not_called()

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_single_data_type_single_reader(self, mock_construct_reader, mock_reader):
        """Test with a single data type and a single reader."""
        # Set up the mock to return our mock reader
        mock_construct_reader.return_value = mock_reader

        # Create a warmup config with a single data type and reader
        warmup = WarmupConfig(
            readers=[
                TypedReaderConfig(
                    data_type="ohlc", readers=[ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"})]
                )
            ]
        )

        # Call the function
        result = _create_data_type_readers(warmup)

        # Check that the result has one data type
        assert len(result) == 1
        assert "ohlc" in result
        assert result["ohlc"] == mock_reader

        # Check that _construct_reader was called once with the correct arguments
        mock_construct_reader.assert_called_once()
        args, _ = mock_construct_reader.call_args
        assert args[0].reader == "mqdb::nebula"
        assert args[0].args == {"param1": "value1"}

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_single_data_type_multiple_readers(self, mock_construct_reader, mock_reader):
        """Test with a single data type and multiple readers."""
        # Create two different mock readers
        mock_reader1 = MagicMock()
        mock_reader2 = MagicMock()

        # Set up the mock to return different readers for different calls
        mock_construct_reader.side_effect = [mock_reader1, mock_reader2]

        # Create a warmup config with a single data type and multiple readers
        warmup = WarmupConfig(
            readers=[
                TypedReaderConfig(
                    data_type="ohlc",
                    readers=[
                        ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"}),
                        ReaderConfig(reader="csv::/data/", args={"param2": "value2"}),
                    ],
                )
            ]
        )

        # Call the function
        result = _create_data_type_readers(warmup)

        # Check that the result has one data type
        assert len(result) == 1
        assert "ohlc" in result

        # Check that _construct_reader was called twice with the correct arguments
        assert mock_construct_reader.call_count == 2

        # Check that a CompositeReader was created
        from qubx.data.composite import CompositeReader

        assert isinstance(result["ohlc"], CompositeReader)

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_multiple_data_types(self, mock_construct_reader, mock_reader):
        """Test with multiple data types."""
        # Create different mock readers for different data types
        mock_reader1 = MagicMock()
        mock_reader2 = MagicMock()

        # Set up the mock to return different readers for different calls
        mock_construct_reader.side_effect = [mock_reader1, mock_reader2]

        # Create a warmup config with multiple data types
        warmup = WarmupConfig(
            readers=[
                TypedReaderConfig(
                    data_type="ohlc", readers=[ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"})]
                ),
                TypedReaderConfig(
                    data_type="trades", readers=[ReaderConfig(reader="csv::/data/", args={"param2": "value2"})]
                ),
            ]
        )

        # Call the function
        result = _create_data_type_readers(warmup)

        # Check that the result has two data types
        assert len(result) == 2
        assert "ohlc" in result
        assert "trades" in result
        assert result["ohlc"] == mock_reader1
        assert result["trades"] == mock_reader2

        # Check that _construct_reader was called twice with the correct arguments
        assert mock_construct_reader.call_count == 2

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_reader_reuse(self, mock_construct_reader, mock_reader):
        """Test that identical reader configurations are only instantiated once."""
        # Set up the mock to return our mock reader
        mock_construct_reader.return_value = mock_reader

        # Create a warmup config with multiple data types using the same reader configuration
        warmup = WarmupConfig(
            readers=[
                TypedReaderConfig(
                    data_type="ohlc", readers=[ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"})]
                ),
                TypedReaderConfig(
                    data_type="trades", readers=[ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"})]
                ),
            ]
        )

        # Call the function
        result = _create_data_type_readers(warmup)

        # Check that the result has two data types
        assert len(result) == 2
        assert "ohlc" in result
        assert "trades" in result
        assert result["ohlc"] == mock_reader
        assert result["trades"] == mock_reader

        # Check that _construct_reader was called only once
        mock_construct_reader.assert_called_once()

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_reader_creation_error(self, mock_construct_reader):
        """Test error handling when reader creation fails."""
        # Set up the mock to raise an exception
        mock_construct_reader.side_effect = ValueError("Reader creation failed")

        # Create a warmup config
        warmup = WarmupConfig(
            readers=[
                TypedReaderConfig(
                    data_type="ohlc", readers=[ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"})]
                )
            ]
        )

        # Call the function and check that it raises the expected exception
        with pytest.raises(ValueError, match="Reader creation failed"):
            _create_data_type_readers(warmup)

    @patch("qubx.utils.runner.runner._construct_reader")
    def test_none_reader(self, mock_construct_reader):
        """Test error handling when _construct_reader returns None."""
        # Set up the mock to return None
        mock_construct_reader.return_value = None

        # Create a warmup config
        warmup = WarmupConfig(
            readers=[
                TypedReaderConfig(
                    data_type="ohlc", readers=[ReaderConfig(reader="mqdb::nebula", args={"param1": "value1"})]
                )
            ]
        )

        # Call the function and check that it raises the expected exception
        with pytest.raises(ValueError, match="Reader mqdb::nebula could not be created"):
            _create_data_type_readers(warmup)
