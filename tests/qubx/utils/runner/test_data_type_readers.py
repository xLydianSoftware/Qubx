from unittest.mock import MagicMock, patch

import pytest

from qubx.utils.runner.configs import StorageConfig, TypedStorageConfig
from qubx.utils.runner.factory import create_data_type_storages


class TestCreateDataTypeStorages:
    """Tests for the create_data_type_storages function."""

    @pytest.fixture
    def mock_reader(self):
        """Create a mock reader for testing."""
        return MagicMock()

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_empty_list(self, mock_construct_storage):
        """Test with empty custom_data list."""
        result = create_data_type_storages([])

        assert result == {}
        mock_construct_storage.assert_not_called()

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_none_input(self, mock_construct_storage):
        """Test with None input."""
        result = create_data_type_storages(None)

        assert result == {}
        mock_construct_storage.assert_not_called()

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_single_data_type_single_storage(self, mock_construct_storage, mock_reader):
        """Test with a single data type and a single storage."""
        mock_construct_storage.return_value = mock_reader

        custom_data = [
            TypedStorageConfig(
                data_type=["ohlc"], storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})]
            )
        ]

        result = create_data_type_storages(custom_data)

        assert len(result) == 1
        assert "ohlc" in result
        assert result["ohlc"] == mock_reader

        mock_construct_storage.assert_called_once()
        args, _ = mock_construct_storage.call_args
        assert args[0].storage == "mqdb::nebula"
        assert args[0].args == {"param1": "value1"}

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_single_data_type_multiple_storages(self, mock_construct_storage, mock_reader):
        """Test with a single data type and multiple storages — should produce MultiStorage."""
        mock_reader1 = MagicMock()
        mock_reader2 = MagicMock()
        mock_construct_storage.side_effect = [mock_reader1, mock_reader2]

        custom_data = [
            TypedStorageConfig(
                data_type=["ohlc"],
                storages=[
                    StorageConfig(storage="mqdb::nebula", args={"param1": "value1"}),
                    StorageConfig(storage="csv::/data/", args={"param2": "value2"}),
                ],
            )
        ]

        result = create_data_type_storages(custom_data)

        assert len(result) == 1
        assert "ohlc" in result
        assert mock_construct_storage.call_count == 2

        from qubx.data.storages.multi import MultiStorage

        assert isinstance(result["ohlc"], MultiStorage)

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_multiple_data_types(self, mock_construct_storage, mock_reader):
        """Test with multiple data types."""
        mock_reader1 = MagicMock()
        mock_reader2 = MagicMock()
        mock_construct_storage.side_effect = [mock_reader1, mock_reader2]

        custom_data = [
            TypedStorageConfig(
                data_type=["ohlc"], storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})]
            ),
            TypedStorageConfig(
                data_type=["trades"], storages=[StorageConfig(storage="csv::/data/", args={"param2": "value2"})]
            ),
        ]

        result = create_data_type_storages(custom_data)

        assert len(result) == 2
        assert result["ohlc"] == mock_reader1
        assert result["trades"] == mock_reader2
        assert mock_construct_storage.call_count == 2

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_storage_reuse_for_same_config(self, mock_construct_storage, mock_reader):
        """Test that identical storage configs are only instantiated once."""
        mock_construct_storage.return_value = mock_reader

        custom_data = [
            TypedStorageConfig(
                data_type="ohlc", storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})]
            ),
            TypedStorageConfig(
                data_type="trades", storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})]
            ),
        ]

        result = create_data_type_storages(custom_data)

        assert len(result) == 2
        assert result["ohlc"] == mock_reader
        assert result["trades"] == mock_reader
        # - same config → constructed only once
        mock_construct_storage.assert_called_once()

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_storage_creation_error(self, mock_construct_storage):
        """Test error propagation when storage creation fails."""
        mock_construct_storage.side_effect = ValueError("Storage creation failed")

        custom_data = [
            TypedStorageConfig(
                data_type=["ohlc"], storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})]
            )
        ]

        with pytest.raises(ValueError, match="Storage creation failed"):
            create_data_type_storages(custom_data)

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_none_storage_raises(self, mock_construct_storage):
        """Test that None returned from construct_storage raises ValueError."""
        mock_construct_storage.return_value = None

        custom_data = [
            TypedStorageConfig(
                data_type=["ohlc"], storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})]
            )
        ]

        with pytest.raises(ValueError, match="Reader mqdb::nebula could not be created"):
            create_data_type_storages(custom_data)

    @patch("qubx.utils.runner.factory.construct_storage")
    def test_multiple_data_types_in_one_config(self, mock_construct_storage, mock_reader):
        """Test that a single TypedStorageConfig with multiple data_types maps all of them."""
        mock_construct_storage.return_value = mock_reader

        custom_data = [
            TypedStorageConfig(
                data_type=["ohlc", "trades", "depth"],
                storages=[StorageConfig(storage="mqdb::nebula", args={"param1": "value1"})],
            )
        ]

        result = create_data_type_storages(custom_data)

        assert len(result) == 3
        assert result["ohlc"] == mock_reader
        assert result["trades"] == mock_reader
        assert result["depth"] == mock_reader

        # - only one unique config → constructed once
        mock_construct_storage.assert_called_once()
        args, _ = mock_construct_storage.call_args
        assert args[0].storage == "mqdb::nebula"
        assert args[0].args == {"param1": "value1"}
