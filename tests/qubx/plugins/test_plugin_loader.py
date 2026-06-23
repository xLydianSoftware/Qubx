import pytest

from qubx.connectors.plugin import ExchangePlugin
from qubx.plugins.loader import PluginLoader


class _FakePlugin(ExchangePlugin):
    name = "fake"


class _FakeEP:
    name = "fake"

    def load(self):
        return _FakePlugin()


def test_available_lists_names_without_loading(monkeypatch):
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_FakeEP()])
    assert "fake" in PluginLoader.available()


def test_load_returns_plugin(monkeypatch):
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_FakeEP()])
    assert isinstance(PluginLoader.load("fake"), _FakePlugin)


def test_load_unknown_returns_none(monkeypatch):
    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_FakeEP()])
    assert PluginLoader.load("nope") is None


def test_load_rejects_non_plugin(monkeypatch):
    class _BadEP:
        name = "bad"

        def load(self):
            return object()

    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_BadEP()])
    with pytest.raises(TypeError):
        PluginLoader.load("bad")


def test_load_asserts_name_matches(monkeypatch):
    class _MismatchPlugin(ExchangePlugin):
        name = "other"

    class _EP:
        name = "fake"

        def load(self):
            return _MismatchPlugin()

    monkeypatch.setattr("qubx.plugins.loader.entry_points", lambda group: [_EP()])
    with pytest.raises(AssertionError):
        PluginLoader.load("fake")
