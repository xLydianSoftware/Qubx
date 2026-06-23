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
        return CcxtDataProvider(ctx, **ctx.params)

    def create_connector(self, ctx: ConnectorBuildContext) -> IConnector:
        return create_ccxt_connector(ctx)

    def rate_limits(self, exchange_name: str) -> ExchangeRateLimitConfig | None:
        return create_ccxt_rate_limit_config(exchange_name)


PLUGIN = CcxtPlugin()
