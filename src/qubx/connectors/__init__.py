"""
Qubx connectors package.

This module provides the connector registry and ensures all built-in connectors are registered.

Note: Paper trading connectors (SimulatedAccountProcessor, SimulatedBroker) are NOT registered
with the registry - they are created directly by the backtester and paper trading runner.
"""

# Re-export the registry for convenience
from qubx.connectors.registry import (  # noqa: F401
    ConnectorRegistry,
    account_processor,
    broker,
    data_provider,
)

# Import connector modules to trigger decorator registration
from qubx.connectors.ccxt.account import CcxtAccountProcessor  # noqa: F401
from qubx.connectors.ccxt.broker import CcxtBroker  # noqa: F401
from qubx.connectors.ccxt.data import CcxtDataProvider  # noqa: F401
from qubx.connectors.ccxt.rate_limits import create_ccxt_rate_limit_config  # noqa: F401
from qubx.connectors.tardis.data import TardisDataProvider  # noqa: F401

# Note: Paper trading connectors (SimulatedAccountProcessor, SimulatedBroker) are
# NOT registered with the registry - they are created directly by the backtester
# and paper trading runner.
