"""The built-in ``ccxt`` exchange plugin."""

from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import create_ccxt_connector
from qubx.connectors.ccxt.rate_limits import create_ccxt_rate_limit_config
from qubx.connectors.plugin import BuildContext, ConnectorBuildContext, ExchangePlugin
from qubx.core.connector import IConnector
from qubx.core.interfaces import IDataProvider
from qubx.rate_limiting import ExchangeRateLimitConfig


class CcxtPlugin(ExchangePlugin):
    """ccxt-backed venues: market data + execution + rate-limit declaration.

    The same ``ctx.rate_limiter`` is attached to both the data provider's and the connector's
    exchange managers, so both sides of one venue share a single per-exchange budget.
    """

    name = "ccxt"

    def create_data_provider(self, ctx: BuildContext) -> IDataProvider:
        return CcxtDataProvider(
            exchange_name=ctx.exchange_name,
            time_provider=ctx.time_provider,
            channel=ctx.channel,
            health_monitor=ctx.health_monitor,
            credentials=ctx.credentials,
            loop=ctx.loop,
            rate_limiter=ctx.rate_limiter,
            **ctx.params,
        )

    def create_connector(self, ctx: ConnectorBuildContext) -> IConnector:
        return create_ccxt_connector(
            exchange_name=ctx.exchange_name,
            time_provider=ctx.time_provider,
            channel=ctx.channel,
            credentials=ctx.credentials,
            data_provider=ctx.data_provider,
            health_monitor=ctx.health_monitor,
            loop=ctx.loop,
            rate_limiter=ctx.rate_limiter,
        )

    def rate_limits(self, exchange_name: str) -> ExchangeRateLimitConfig | None:
        return create_ccxt_rate_limit_config(exchange_name)


PLUGIN = CcxtPlugin()
