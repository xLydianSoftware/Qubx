import pytest

import qubx.connectors.registry as registry
from qubx.connectors.plugin import ExchangePlugin
from qubx.connectors.registry import ConnectorRegistry


@pytest.mark.parametrize(
    "name",
    [
        "broker",
        "account_processor",
        "register_broker",
        "register_account_processor",
        "connector",
        "data_provider",
        "rate_limit_config",
        "register_connector",
        "register_data_provider",
        "register_rate_limit_config",
    ],
)
def test_removed_registry_names_raise_migration_pointer(name: str):
    with pytest.raises(ImportError, match="ExchangePlugin"):
        getattr(registry, name)


def test_unknown_registry_name_raises_attribute_error():
    with pytest.raises(AttributeError):
        registry.no_such_thing


class _DataOnly(ExchangePlugin):
    name = "dataonly"

    def create_data_provider(self, ctx):
        return "DP"


def test_register_and_get_plugin():
    ConnectorRegistry._plugins.clear()
    p = _DataOnly()
    ConnectorRegistry.register(p)
    assert ConnectorRegistry.get_plugin("dataonly") is p


def test_get_data_provider_returns_the_built_provider():
    ConnectorRegistry._plugins.clear()
    ConnectorRegistry.register(_DataOnly())
    assert ConnectorRegistry.get_data_provider("dataonly", object()) == "DP"


def test_get_connector_missing_capability_raises():
    ConnectorRegistry._plugins.clear()
    ConnectorRegistry.register(_DataOnly())
    with pytest.raises(ValueError, match="no execution connector"):
        ConnectorRegistry.get_connector("dataonly", object())


def test_get_plugin_unknown_raises():
    ConnectorRegistry._plugins.clear()
    with pytest.raises(ValueError, match="No connector plugin"):
        ConnectorRegistry.get_plugin("ghost_xyz_not_a_plugin")
