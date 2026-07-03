from qubx.plugins.loader import PluginLoader


def test_builtin_plugins_discoverable():
    assert {"ccxt", "tardis"} <= PluginLoader.available()


def test_ccxt_loads_to_plugin():
    p = PluginLoader.load("ccxt")
    assert p is not None and p.name == "ccxt"


def test_tardis_loads_to_plugin():
    p = PluginLoader.load("tardis")
    assert p is not None and p.name == "tardis"
