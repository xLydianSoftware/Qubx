"""
Registry for exchange plugins (connector + data provider + rate-limit declaration).

A venue is one :class:`~qubx.connectors.plugin.ExchangePlugin`, discovered by entry point
(group ``qubx.exchange_plugins``) via :class:`~qubx.plugins.loader.PluginLoader` and resolved
here by name (the config's ``connector`` / ``data_provider`` field). ``register`` is the direct
path for tests / programmatic use. The paper/backtest ``SimulatedConnector`` is NOT a plugin —
it is the framework's built-in simulator, constructed directly by the runner.
"""

from typing import TYPE_CHECKING, Any, Protocol

from qubx.core.connector import IConnector
from qubx.core.interfaces import IDataProvider

if TYPE_CHECKING:
    from qubx.connectors.plugin import BuildContext, ConnectorBuildContext, ExchangePlugin


class ExchangeSettingsLike(Protocol):
    """The slice of per-exchange settings that plugins actually read."""

    testnet: bool


class ExchangeCredentialsLike(Protocol):
    """The slice of per-exchange credentials that plugins actually read."""

    testnet: bool
    api_key: str
    secret: str

    @property
    def model_extra(self) -> dict[str, Any] | None: ...


class CredentialsProvider(Protocol):
    """Structural view of the runner's ``AccountConfigurationManager``.

    Plugins receive it as the standardized ``credentials`` field of the build context; typing it
    structurally keeps connector code free of a connectors->runner import back-edge.
    """

    def get_exchange_settings(self, exchange: str) -> ExchangeSettingsLike: ...

    def get_exchange_credentials(self, exchange: str) -> ExchangeCredentialsLike: ...


class ConnectorRegistry:
    """Resolves :class:`ExchangePlugin` instances by name and builds their components.

    Plugins are discovered lazily via entry points (``get_plugin`` → ``PluginLoader.load``);
    ``register`` is the direct path for tests. ``get_data_provider`` / ``get_connector`` call the
    plugin's factory methods and raise a clear error when the requested capability is absent.
    """

    _plugins: dict[str, "ExchangePlugin"] = {}

    @classmethod
    def register(cls, plugin: "ExchangePlugin") -> None:
        cls._plugins[plugin.name.lower()] = plugin

    @classmethod
    def get_plugin(cls, name: str) -> "ExchangePlugin":
        key = name.lower()
        if key not in cls._plugins:
            from qubx.plugins.loader import PluginLoader

            plugin = PluginLoader.load(key)
            if plugin is None:
                raise ValueError(f"No connector plugin '{name}'. Available: {sorted(PluginLoader.available())}")
            cls._plugins[key] = plugin
        return cls._plugins[key]

    @classmethod
    def get_data_provider(cls, name: str, ctx: "BuildContext") -> IDataProvider:
        dp = cls.get_plugin(name).create_data_provider(ctx)
        if dp is None:
            raise ValueError(f"Venue plugin '{name}' provides no data provider")
        return dp

    @classmethod
    def get_connector(cls, name: str, ctx: "ConnectorBuildContext") -> IConnector:
        conn = cls.get_plugin(name).create_connector(ctx)
        if conn is None:
            raise ValueError(f"Venue plugin '{name}' provides no execution connector")
        return conn

    @classmethod
    def get_rate_limit_config(cls, name: str, exchange_name: str):
        """Rate-limit config for a venue (None when the plugin declares none)."""
        return cls.get_plugin(name).rate_limits(exchange_name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        from qubx.plugins.loader import PluginLoader

        return name.lower() in cls._plugins or name.lower() in PluginLoader.available()


# Tombstone for the pre-ExchangePlugin registry API so stale plugins fail with a pointer to the migration.
_REMOVED_NAMES = (
    "broker",
    "account_processor",
    "register_broker",
    "register_account_processor",
    "connector",
    "data_provider",
    "rate_limit_config",
    "register_connector",
    "register_data_provider",
    "register_rate_limit_config",
)


def __getattr__(name: str) -> Any:
    if name in _REMOVED_NAMES:
        raise ImportError(
            f"'{name}' was removed: a venue is now one ExchangePlugin (connector + data provider + "
            "rate_limits), discovered by entry point (group 'qubx.exchange_plugins'). See "
            "docs/superpowers/specs/2026-06-23-connector-registry-redesign-design.md."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
