"""Connector plugin contract.

A venue is one :class:`ExchangePlugin` — it builds the venue's data provider and execution
connector from typed contexts, and declares its rate-limit topology. The framework discovers
plugins by entry point (group ``qubx.exchange_plugins``) and builds them through the registry.

The two contexts model the construction order: the data provider is built from ``BuildContext``;
the connector from ``ConnectorBuildContext``, which adds the already-built ``data_provider``. They
are intentionally NOT ``slots=True`` so the runner can extend one into the other via
``ConnectorBuildContext(**vars(base), data_provider=dp)``.
"""

import asyncio
from abc import ABC
from dataclasses import dataclass, field

from qubx.connectors.registry import CredentialsProvider
from qubx.core.basics import CtrlChannel, ITimeProvider
from qubx.core.connector import IConnector
from qubx.core.interfaces import IDataProvider, IHealthMonitor
from qubx.rate_limiting import ExchangeRateLimitConfig, ExchangeRateLimiter


@dataclass(frozen=True, kw_only=True)
class BuildContext:
    """Inputs shared by the data provider and the connector."""

    exchange_name: str
    time_provider: ITimeProvider
    channel: CtrlChannel
    credentials: CredentialsProvider
    health_monitor: IHealthMonitor
    loop: asyncio.AbstractEventLoop
    # Built centrally by RateLimitManager from plugin.rate_limits(); the same instance is
    # shared into the data provider and the connector. None when rate limiting is disabled.
    rate_limiter: ExchangeRateLimiter | None = None
    # Per-exchange config options (ExchangeConfig.params) — venue-specific knobs a plugin
    # forwards to the data provider (e.g. tardis host/port, ccxt orderbook_limit).
    params: dict = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ConnectorBuildContext(BuildContext):
    """Adds the already-built data provider the connector depends on."""

    data_provider: IDataProvider


class ExchangePlugin(ABC):
    """One venue's connector + data provider + rate-limit declaration.

    Override only what the venue provides — the defaults return ``None`` (e.g. a data-only
    venue overrides just :meth:`create_data_provider`). The registry turns a ``None`` from a
    requested capability into a clear error.
    """

    name: str

    def create_data_provider(self, ctx: BuildContext) -> IDataProvider | None:
        return None

    def create_connector(self, ctx: ConnectorBuildContext) -> IConnector | None:
        return None

    def rate_limits(self, exchange_name: str) -> ExchangeRateLimitConfig | None:
        return None
