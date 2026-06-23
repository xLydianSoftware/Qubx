import pytest

import qubx.connectors.registry as registry


@pytest.mark.parametrize("name", ["broker", "account_processor", "register_broker", "register_account_processor"])
def test_removed_registry_names_raise_migration_pointer(name: str):
    with pytest.raises(ImportError, match="IConnector"):
        getattr(registry, name)


def test_unknown_registry_name_raises_attribute_error():
    with pytest.raises(AttributeError):
        registry.no_such_thing
