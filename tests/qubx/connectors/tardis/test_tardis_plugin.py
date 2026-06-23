from qubx.connectors.tardis.plugin import PLUGIN, TardisPlugin


def test_tardis_is_data_only():
    assert PLUGIN.name == "tardis"
    assert isinstance(PLUGIN, TardisPlugin)
    # data-only: the default create_connector returns None
    assert PLUGIN.create_connector(object()) is None
