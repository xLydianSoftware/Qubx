"""The built-in ``tardis`` exchange plugin (data-only)."""

from qubx.connectors.plugin import BuildContext, ExchangePlugin
from qubx.connectors.tardis.data import TardisDataProvider
from qubx.core.interfaces import IDataProvider


class TardisPlugin(ExchangePlugin):
    """tardis-machine market-data replay: a data provider only (no execution connector)."""

    name = "tardis"

    def create_data_provider(self, ctx: BuildContext) -> IDataProvider:
        return TardisDataProvider(
            exchange_name=ctx.exchange_name,
            time_provider=ctx.time_provider,
            channel=ctx.channel,
            health_monitor=ctx.health_monitor,
            credentials=ctx.credentials,
            loop=ctx.loop,
            **ctx.params,
        )


PLUGIN = TardisPlugin()
