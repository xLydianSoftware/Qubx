from qubx.connectors.plugin import BuildContext, ConnectorBuildContext, ExchangePlugin


def test_connector_ctx_is_a_build_ctx():
    assert issubclass(ConnectorBuildContext, BuildContext)


def test_plugin_defaults_return_none():
    class P(ExchangePlugin):
        name = "p"

    p = P()
    assert p.create_connector(object()) is None
    assert p.create_data_provider(object()) is None
    assert p.rate_limits("X") is None


def test_partial_plugin_overrides_only_data_provider():
    sentinel = object()

    class DataOnly(ExchangePlugin):
        name = "dataonly"

        def create_data_provider(self, ctx):
            return sentinel

    p = DataOnly()
    assert p.create_data_provider(object()) is sentinel
    assert p.create_connector(object()) is None
