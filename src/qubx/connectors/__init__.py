"""
Qubx connectors package.

A venue is one :class:`~qubx.connectors.plugin.ExchangePlugin` (connector + data provider +
rate-limit declaration), discovered by entry point (group ``qubx.exchange_plugins``) and resolved
via :class:`~qubx.connectors.registry.ConnectorRegistry`. The paper-trading connector
(``SimulatedConnector``) is NOT a plugin — it is created directly by the backtester / paper runner.
"""

from qubx.connectors.registry import ConnectorRegistry  # noqa: F401
