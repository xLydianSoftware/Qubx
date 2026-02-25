"""
Test plugin that registers a custom storage.
"""

from qubx.data.registry import storage
from qubx.data.storage import IStorage


@storage("test_plugin_storage")
class TestPluginStorage(IStorage):
    """A test storage for plugin testing."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port

    def get_exchanges(self):
        return ["TEST.EXCHANGE"]
