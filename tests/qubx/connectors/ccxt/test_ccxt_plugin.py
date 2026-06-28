from qubx.connectors.ccxt.plugin import PLUGIN, CcxtPlugin


def test_plugin_name():
    assert PLUGIN.name == "ccxt"
    assert isinstance(PLUGIN, CcxtPlugin)


def test_rate_limits_delegates(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("qubx.connectors.ccxt.plugin.create_ccxt_rate_limit_config", lambda e: sentinel)
    assert PLUGIN.rate_limits("BINANCE.UM") is sentinel
