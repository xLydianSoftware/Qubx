"""The built-in ``tardis`` exchange plugin (data-only)."""

from qubx.connectors.plugin import BuildContext, ExchangePlugin
from qubx.connectors.tardis.data import TardisDataProvider
from qubx.core.interfaces import IDataProvider


class TardisPlugin(ExchangePlugin):
    """tardis-machine market-data replay: a data provider only (no execution connector)."""

    name = "tardis"

    def create_data_provider(self, ctx: BuildContext) -> IDataProvider:
        return TardisDataProvider(ctx, **ctx.params)


PLUGIN = TardisPlugin()
