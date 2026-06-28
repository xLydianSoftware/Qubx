"""
Tests for the general plugin loader (``@storage`` / ``@reader``, paths + modules).

Connector/data-provider/rate-limit plugins are now ``ExchangePlugin`` objects discovered via
entry points — see tests/qubx/connectors/test_registry.py and test_builtin_discovery.py.
"""

from pathlib import Path

import pytest

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
        """load_plugins does nothing with None config."""
        load_plugins(None)

    def test_load_plugins_with_empty_config(self):
        """load_plugins does nothing with empty config."""
        load_plugins(PluginsConfig())

    def test_load_plugins_from_path(self):
        """Load a ``@storage`` plugin from a filesystem path."""
        plugins_path = Path(__file__).parent.parent.parent / "fixtures" / "plugins"

        if "test_plugin_storage" in StorageRegistry._storages:
            del StorageRegistry._storages["test_plugin_storage"]

        load_plugins(PluginsConfig(paths=[str(plugins_path)]))

        assert StorageRegistry.is_registered("test_plugin_storage")
        storage = StorageRegistry.get("test_plugin_storage", host="testhost", port=9999)
        assert storage.host == "testhost"
        assert storage.port == 9999

    def test_load_plugins_from_path_with_tilde(self):
        """Tilde expansion doesn't crash on a nonexistent path."""
        load_plugins(PluginsConfig(paths=["~/nonexistent_plugin_path"]))

    def test_load_plugins_from_nonexistent_path(self):
        """A nonexistent path logs a warning but doesn't crash."""
        load_plugins(PluginsConfig(paths=["/nonexistent/path/to/plugins"]))

    def test_load_plugins_nonexistent_module(self):
        """A nonexistent module logs a warning but doesn't crash."""
        load_plugins(PluginsConfig(modules=["nonexistent_module_xyz"]))

    def test_load_plugins_idempotent(self):
        """Loading the same plugin path twice doesn't duplicate registrations."""
        plugins_path = Path(__file__).parent.parent.parent / "fixtures" / "plugins"

        config = PluginsConfig(paths=[str(plugins_path)])
        load_plugins(config)
        load_plugins(config)

        assert StorageRegistry.is_registered("test_plugin_storage")
