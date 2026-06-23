"""
Registry for data providers and execution connectors.

This module provides a registry pattern for connectors, allowing plugins
to register custom data providers and ``IConnector`` execution connectors using
decorators (like readers and storages).
"""

from typing import Any, Callable, Protocol, TypeVar

from qubx import logger
from qubx.core.connector import IConnector
from qubx.core.interfaces import IDataProvider

T = TypeVar("T")


class ExchangeSettingsLike(Protocol):
    """The slice of per-exchange settings that registered factories actually read."""

    testnet: bool


class ExchangeCredentialsLike(Protocol):
    """The slice of per-exchange credentials that registered factories actually read."""

    testnet: bool
    api_key: str
    secret: str

    @property
    def model_extra(self) -> dict[str, Any] | None: ...


class CredentialsProvider(Protocol):
    """Structural view of the runner's ``AccountConfigurationManager``.

    Registered factories receive it as the standardized ``credentials`` argument; typing it
    structurally keeps connector code free of a connectors->runner import back-edge.
    """

    def get_exchange_settings(self, exchange: str) -> ExchangeSettingsLike: ...

    def get_exchange_credentials(self, exchange: str) -> ExchangeCredentialsLike: ...


class ConnectorRegistry:
    """
    Registry for data-provider classes and ``IConnector`` execution-connector factories.

    Plugins register their own market-data providers (``@data_provider``) and live
    execution connectors (``@connector``) by name, so the runner resolves both from the
    config's ``connector`` field rather than hardcoding a venue. The paper/backtest
    ``SimulatedConnector`` is NOT registered — it is the framework's built-in simulator,
    constructed directly by the runner.

    Data providers register the class (instantiated with standardized constructor args);
    connectors register a factory callable (so a venue can build its own auth/exchange
    plumbing and resolve per-exchange subclasses behind a uniform signature).
    """

    _data_providers: dict[str, type[IDataProvider]] = {}
    _connectors: dict[str, Callable[..., IConnector]] = {}
    _rate_limit_configs: dict[str, Callable] = {}

    @classmethod
    def register_data_provider(cls, name: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a data provider class.

        Args:
            name: The name to register the data provider under (e.g., "ccxt", "tardis")

        Returns:
            A decorator function that registers the class
        """

        def decorator(provider_cls: type[T]) -> type[T]:
            cls._data_providers[name.lower()] = provider_cls  # type: ignore
            logger.debug(f"Registered data provider: {name}")
            return provider_cls

        return decorator

    @classmethod
    def get_data_provider(cls, name: str, **kwargs: Any) -> IDataProvider:
        """
        Get a data provider instance by name.

        Args:
            name: The name of the data provider
            **kwargs: Arguments to pass to the constructor

        Returns:
            An instance of the data provider

        Raises:
            ValueError: If the data provider is not found
        """
        provider_cls = cls._data_providers.get(name.lower())
        if provider_cls is None:
            raise ValueError(f"Data provider '{name}' is not registered. Available: {list(cls._data_providers.keys())}")
        return provider_cls(**kwargs)

    @classmethod
    def register_connector(cls, name: str) -> Callable[[T], T]:
        """
        Decorator to register an ``IConnector`` factory under a connector name.

        The registered object is a callable (factory) — not necessarily a class — so a
        venue can build its own auth/exchange plumbing and pick a per-exchange subclass
        behind the uniform ``get_connector(name, **kwargs)`` signature.

        Args:
            name: The connector name to register under (e.g., "ccxt").
        """

        def decorator(factory: T) -> T:
            cls._connectors[name.lower()] = factory  # type: ignore[assignment]
            logger.debug(f"Registered connector: {name}")
            return factory

        return decorator

    @classmethod
    def get_connector(cls, name: str, **kwargs: Any) -> IConnector:
        """
        Build an ``IConnector`` instance by connector name.

        Args:
            name: The connector name (the config's ``connector`` field).
            **kwargs: Arguments forwarded to the registered factory.

        Raises:
            ValueError: If the connector is not registered.
        """
        factory = cls._connectors.get(name.lower())
        if factory is None:
            raise ValueError(f"Connector '{name}' is not registered. Available: {list(cls._connectors.keys())}")
        return factory(**kwargs)

    @classmethod
    def is_connector_registered(cls, name: str) -> bool:
        """Check if an execution connector is registered."""
        return name.lower() in cls._connectors

    @classmethod
    def register_rate_limit_config(cls, name: str) -> Callable:
        """Decorator to register a rate limit config factory for a connector.

        The factory receives (exchange_name: str) and returns ExchangeRateLimitConfig or None.
        """

        def decorator(func: Callable) -> Callable:
            cls._rate_limit_configs[name.lower()] = func
            logger.debug(f"Registered rate limit config: {name}")
            return func

        return decorator

    @classmethod
    def get_rate_limit_config(cls, name: str, exchange_name: str):
        """Get rate limit config for a connector/exchange pair.

        Returns:
            ExchangeRateLimitConfig or None if connector has no rate limiting registered
        """
        factory = cls._rate_limit_configs.get(name.lower())
        if factory is None:
            return None
        return factory(exchange_name)

    @classmethod
    def is_data_provider_registered(cls, name: str) -> bool:
        """Check if a data provider is registered."""
        return name.lower() in cls._data_providers

    @classmethod
    def get_all_data_providers(cls) -> dict[str, type[IDataProvider]]:
        """Get all registered data provider classes."""
        return cls._data_providers.copy()


# Tombstone for the pre-IConnector registry API so stale plugins fail with a pointer to the migration.
_REMOVED_NAMES = ("broker", "account_processor", "register_broker", "register_account_processor")


def __getattr__(name: str) -> Any:
    if name in _REMOVED_NAMES:
        raise ImportError(
            f"'{name}' was removed: plugins now implement a single IConnector and register a factory with "
            "@connector(name) — see docs/account-management/design.md, section 'Connectors (IConnector)'."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Convenience decorators
def data_provider(name: str) -> Callable[[type[T]], type[T]]:
    """
    Decorator for registering a data provider class.

    Usage:
        @data_provider("my_exchange")
        class MyExchangeDataProvider(IDataProvider):
            def __init__(self, exchange_name, time_provider, channel, ...):
                ...
    """
    return ConnectorRegistry.register_data_provider(name)


def connector(name: str) -> Callable[[T], T]:
    """
    Decorator for registering an IConnector factory.

    Usage:
        @connector("my_exchange")
        def create_my_exchange_connector(exchange_name, time_provider, channel, ...) -> IConnector:
            ...
    """
    return ConnectorRegistry.register_connector(name)


def rate_limit_config(name: str) -> Callable:
    """
    Decorator for registering a rate limit config factory.

    Usage:
        @rate_limit_config("my_exchange")
        def create_my_exchange_rate_limits(exchange_name: str) -> ExchangeRateLimitConfig:
            return ExchangeRateLimitConfig(pools={...}, ...)
    """
    return ConnectorRegistry.register_rate_limit_config(name)
