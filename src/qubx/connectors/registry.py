"""
Registry for data providers.

This module provides a registry pattern for connectors, allowing plugins
to register custom data providers using decorators (like readers and storages).
"""

from typing import Any, Callable, Type, TypeVar

from qubx import logger
from qubx.core.interfaces import IDataProvider

T = TypeVar("T")


class ConnectorRegistry:
    """
    Registry for data-provider classes.

    This registry allows plugins to register their own data-provider implementations
    using decorators, making them available for use in strategy configurations.
    Order execution and account state are handled by ``IConnector`` (see
    ``qubx.core.connector``), which is constructed directly by the runner rather than
    looked up here.

    Classes are registered and instantiated with standardized constructor arguments.
    """

    _data_providers: dict[str, Type[IDataProvider]] = {}
    _rate_limit_configs: dict[str, Callable] = {}

    @classmethod
    def register_data_provider(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a data provider class.

        Args:
            name: The name to register the data provider under (e.g., "ccxt", "tardis")

        Returns:
            A decorator function that registers the class
        """

        def decorator(provider_cls: Type[T]) -> Type[T]:
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
            raise ValueError(
                f"Data provider '{name}' is not registered. "
                f"Available: {list(cls._data_providers.keys())}"
            )
        return provider_cls(**kwargs)

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
    def get_all_data_providers(cls) -> dict[str, Type[IDataProvider]]:
        """Get all registered data provider classes."""
        return cls._data_providers.copy()


# Convenience decorators
def data_provider(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for registering a data provider class.

    Usage:
        @data_provider("my_exchange")
        class MyExchangeDataProvider(IDataProvider):
            def __init__(self, exchange_name, time_provider, channel, ...):
                ...
    """
    return ConnectorRegistry.register_data_provider(name)


def rate_limit_config(name: str) -> Callable:
    """
    Decorator for registering a rate limit config factory.

    Usage:
        @rate_limit_config("my_exchange")
        def create_my_exchange_rate_limits(exchange_name: str) -> ExchangeRateLimitConfig:
            return ExchangeRateLimitConfig(pools={...}, ...)
    """
    return ConnectorRegistry.register_rate_limit_config(name)
