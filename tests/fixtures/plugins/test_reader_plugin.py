"""
Test plugin that registers a custom reader.
"""

from qubx.data.readers import DataReader
from qubx.data.registry import reader


@reader("test_plugin_reader")
class TestPluginReader(DataReader):
    """A test reader for plugin testing."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port

    def read(self, **kwargs):
        return {"data": f"from {self.host}:{self.port}"}
