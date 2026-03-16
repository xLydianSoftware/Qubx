"""
Tests for the plugin loader.
"""

from pathlib import Path

import pytest

from qubx.connectors.registry import ConnectorRegistry
from qubx.data.registry import StorageRegistry
from qubx.plugins.loader import load_plugins, reset_loaded_plugins
from qubx.utils.runner.configs import PluginsConfig


@pytest.fixture(autouse=True)
def reset_plugins():
    """Reset loaded plugins before each test."""
    reset_loaded_plugins()
    yield
    reset_loaded_plugins()


class TestPluginLoader:
    """Tests for plugin loading functionality."""

    def test_load_plugins_with_none_config(self):
        """Test that load_plugins does nothing with None config."""
        # Should not raise any errors
        load_plugins(None)

    def test_load_plugins_with_empty_config(self):
        """Test that load_plugins does nothing with empty config."""
        config = PluginsConfig()
        load_plugins(config)

    def test_load_plugins_from_path(self):
        """Test loading plugins from a filesystem path."""
        plugins_path = Path(__file__).parent.parent.parent / "fixtures" / "plugins"

        # Remove previously registered readers for clean test
        if "test_plugin_storage" in StorageRegistry._storages:
            del StorageRegistry._storages["test_plugin_storage"]

        config = PluginsConfig(paths=[str(plugins_path)])
        load_plugins(config)

        # Verify the test reader was registered
        assert StorageRegistry.is_registered("test_plugin_storage")

        # Verify we can get the reader
        storage = StorageRegistry.get("test_plugin_storage", host="testhost", port=9999)
        assert storage.host == "testhost"
        assert storage.port == 9999

    def test_load_plugins_from_path_with_tilde(self):
        """Test loading plugins from a path with tilde expansion."""
        plugins_path = Path(__file__).parent.parent.parent / "fixtures" / "plugins"

        # Create a config with a path using tilde (won't actually work unless home dir matches)
        # We're just testing that the tilde expansion doesn't crash
        config = PluginsConfig(paths=["~/nonexistent_plugin_path"])
        load_plugins(config)  # Should log a warning but not raise

    def test_load_plugins_from_nonexistent_path(self):
        """Test that loading from nonexistent path logs warning but doesn't crash."""
        config = PluginsConfig(paths=["/nonexistent/path/to/plugins"])
        load_plugins(config)  # Should log a warning but not raise

    def test_load_plugins_from_module(self):
        """Test loading plugins from an installed module."""
        import sys
        from unittest.mock import patch

        # Add fixtures path to allow module import
        fixtures_path = str(Path(__file__).parent.parent.parent / "fixtures" / "plugins")
        if fixtures_path not in sys.path:
            sys.path.insert(0, fixtures_path)

        # Remove module from cache to ensure fresh import
        if "test_connector_plugin" in sys.modules:
            del sys.modules["test_connector_plugin"]

        # Mock the registry to track registrations without affecting global state
        mock_data_providers = {}
        mock_account_processors = {}
        mock_brokers = {}

        with (
            patch.object(ConnectorRegistry, "_data_providers", mock_data_providers),
            patch.object(ConnectorRegistry, "_account_processors", mock_account_processors),
            patch.object(ConnectorRegistry, "_brokers", mock_brokers),
        ):
            config = PluginsConfig(modules=["test_connector_plugin"])
            load_plugins(config)

            # Verify the connector was registered in our mocked registries
            assert "test_connector" in mock_data_providers
            assert "test_connector" in mock_account_processors
            assert "test_connector" in mock_brokers

    def test_load_plugins_nonexistent_module(self):
        """Test that loading nonexistent module logs warning but doesn't crash."""
        config = PluginsConfig(modules=["nonexistent_module_xyz"])
        load_plugins(config)  # Should log a warning but not raise

    def test_load_plugins_idempotent(self):
        """Test that loading the same plugin twice doesn't duplicate registrations."""
        plugins_path = Path(__file__).parent.parent.parent / "fixtures" / "plugins"

        config = PluginsConfig(paths=[str(plugins_path)])

        # Load twice
        load_plugins(config)
        load_plugins(config)

        # Should still work fine
        assert StorageRegistry.is_registered("test_plugin_storage")


class TestConnectorRegistry:
    """Tests for the connector registry."""

    def test_register_data_provider(self):
        """Test registering a data provider class."""
        from qubx.connectors.registry import data_provider

        @data_provider("test_dp")
        class TestDataProvider:
            def __init__(self, **kwargs):
                self.type = "data_provider"
                self.kwargs = kwargs

        assert ConnectorRegistry.is_data_provider_registered("test_dp")
        instance = ConnectorRegistry.get_data_provider("test_dp", param1="value1")
        assert instance.type == "data_provider"
        assert instance.kwargs["param1"] == "value1"

        # Cleanup
        del ConnectorRegistry._data_providers["test_dp"]

    def test_register_account_processor(self):
        """Test registering an account processor class."""
        from qubx.connectors.registry import account_processor

        @account_processor("test_ap")
        class TestAccountProcessor:
            def __init__(self, **kwargs):
                self.type = "account_processor"
                self.kwargs = kwargs

        assert ConnectorRegistry.is_account_processor_registered("test_ap")
        instance = ConnectorRegistry.get_account_processor("test_ap", param1="value1")
        assert instance.type == "account_processor"

        # Cleanup
        del ConnectorRegistry._account_processors["test_ap"]

    def test_register_broker(self):
        """Test registering a broker class."""
        from qubx.connectors.registry import broker

        @broker("test_br")
        class TestBroker:
            def __init__(self, **kwargs):
                self.type = "broker"
                self.kwargs = kwargs

        assert ConnectorRegistry.is_broker_registered("test_br")
        instance = ConnectorRegistry.get_broker("test_br", param1="value1")
        assert instance.type == "broker"

        # Cleanup
        del ConnectorRegistry._brokers["test_br"]

    def test_case_insensitive_lookup(self):
        """Test that connector names are case-insensitive."""
        from qubx.connectors.registry import data_provider

        @data_provider("TestCase")
        class TestCaseProvider:
            def __init__(self, **kwargs):
                pass

        # Should find regardless of case
        assert ConnectorRegistry.is_data_provider_registered("testcase")
        assert ConnectorRegistry.is_data_provider_registered("TESTCASE")
        assert ConnectorRegistry.is_data_provider_registered("TestCase")

        # Cleanup
        del ConnectorRegistry._data_providers["testcase"]

    def test_get_all_registrations(self):
        """Test getting all registered connectors."""
        from qubx.connectors.registry import data_provider

        @data_provider("test_all")
        class TestAllProvider:
            def __init__(self, **kwargs):
                pass

        all_providers = ConnectorRegistry.get_all_data_providers()
        assert "test_all" in all_providers

        # Cleanup
        del ConnectorRegistry._data_providers["test_all"]


class TestBuiltinConnectors:
    """Tests for built-in connector registration."""

    def test_builtin_connectors_registered(self):
        """Test that built-in connectors are registered after import."""
        import qubx.connectors  # noqa: F401

        # Check that built-in connectors are registered
        assert ConnectorRegistry.is_data_provider_registered("ccxt")
        assert ConnectorRegistry.is_data_provider_registered("tardis")

        assert ConnectorRegistry.is_account_processor_registered("ccxt")

        assert ConnectorRegistry.is_broker_registered("ccxt")

        # Note: Paper trading connectors (SimulatedAccountProcessor, SimulatedBroker)
        # are NOT registered with the registry - they are created directly by the
        # backtester and paper trading runner
        assert not ConnectorRegistry.is_account_processor_registered("paper")
        assert not ConnectorRegistry.is_broker_registered("paper")
