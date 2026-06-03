"""
Test plugin that registers a custom data provider.
"""

from unittest.mock import MagicMock

from qubx.connectors.registry import data_provider


@data_provider("test_connector")
def create_test_data_provider(**kwargs):
    """Create a test data provider."""
    mock = MagicMock()
    mock.kwargs = kwargs
    mock.connector_type = "test_connector_data_provider"
    return mock
