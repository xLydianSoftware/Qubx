"""
Test plugin that registers a custom connector.
"""

from unittest.mock import MagicMock

from qubx.connectors.registry import account_processor, broker, data_provider


@data_provider("test_connector")
def create_test_data_provider(**kwargs):
    """Create a test data provider."""
    mock = MagicMock()
    mock.kwargs = kwargs
    mock.connector_type = "test_connector_data_provider"
    return mock


@account_processor("test_connector")
def create_test_account_processor(**kwargs):
    """Create a test account processor."""
    mock = MagicMock()
    mock.kwargs = kwargs
    mock.connector_type = "test_connector_account"
    return mock


@broker("test_connector")
def create_test_broker(**kwargs):
    """Create a test broker."""
    mock = MagicMock()
    mock.kwargs = kwargs
    mock.connector_type = "test_connector_broker"
    return mock
