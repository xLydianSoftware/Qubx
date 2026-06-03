"""
Qubx connectors package.

This module provides the connector registry and ensures all built-in connectors are registered.

Note: The paper-trading connector (SimulatedConnector) is NOT registered with the registry —
it is created directly by the backtester and paper trading runner.
"""

# Re-export the registry for convenience, and import connector modules to trigger
# decorator registration.
from qubx.connectors.ccxt.data import CcxtDataProvider  # noqa: F401
from qubx.connectors.ccxt.rate_limits import create_ccxt_rate_limit_config  # noqa: F401
from qubx.connectors.registry import (  # noqa: F401
    ConnectorRegistry,
    data_provider,
)
from qubx.connectors.tardis.data import TardisDataProvider  # noqa: F401
