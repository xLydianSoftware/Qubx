"""
Registry for data providers, account processors, and brokers.

This module provides a registry pattern for connectors, allowing plugins
to register custom connectors using decorators (like readers and storages).
"""

from typing import Any, Callable, Type, TypeVar

from qubx import logger
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider

T = TypeVar("T")


class ConnectorRegistry:
    """
    Registry for connector classes (data providers, account processors, brokers).

    This registry allows plugins to register their own connector implementations
    using decorators, making them available for use in strategy configurations.

    Classes are registered and instantiated with standardized constructor arguments.
    """

    _data_providers: dict[str, Type[IDataProvider]] = {}
    _account_processors: dict[str, Type[IAccountProcessor]] = {}
    _brokers: dict[str, Type[IBroker]] = {}

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
    def register_account_processor(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register an account processor class.

        Args:
            name: The name to register the account processor under

        Returns:
            A decorator function that registers the class
        """

        def decorator(processor_cls: Type[T]) -> Type[T]:
            cls._account_processors[name.lower()] = processor_cls  # type: ignore
            logger.debug(f"Registered account processor: {name}")
            return processor_cls

        return decorator

    @classmethod
    def register_broker(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a broker class.

        Args:
            name: The name to register the broker under

        Returns:
            A decorator function that registers the class
        """

        def decorator(broker_cls: Type[T]) -> Type[T]:
            cls._brokers[name.lower()] = broker_cls  # type: ignore
            logger.debug(f"Registered broker: {name}")
            return broker_cls

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
    def get_account_processor(cls, name: str, **kwargs: Any) -> IAccountProcessor:
        """
        Get an account processor instance by name.

        Args:
            name: The name of the account processor
            **kwargs: Arguments to pass to the constructor

        Returns:
            An instance of the account processor

        Raises:
            ValueError: If the account processor is not found
        """
        processor_cls = cls._account_processors.get(name.lower())
        if processor_cls is None:
            raise ValueError(
                f"Account processor '{name}' is not registered. "
                f"Available: {list(cls._account_processors.keys())}"
            )
        return processor_cls(**kwargs)

    @classmethod
    def get_broker(cls, name: str, **kwargs: Any) -> IBroker:
        """
        Get a broker instance by name.

        Args:
            name: The name of the broker
            **kwargs: Arguments to pass to the constructor

        Returns:
            An instance of the broker

        Raises:
            ValueError: If the broker is not found
        """
        broker_cls = cls._brokers.get(name.lower())
        if broker_cls is None:
            raise ValueError(
                f"Broker '{name}' is not registered. "
                f"Available: {list(cls._brokers.keys())}"
            )
        return broker_cls(**kwargs)

    @classmethod
    def is_data_provider_registered(cls, name: str) -> bool:
        """Check if a data provider is registered."""
        return name.lower() in cls._data_providers

    @classmethod
    def is_account_processor_registered(cls, name: str) -> bool:
        """Check if an account processor is registered."""
        return name.lower() in cls._account_processors

    @classmethod
    def is_broker_registered(cls, name: str) -> bool:
        """Check if a broker is registered."""
        return name.lower() in cls._brokers

    @classmethod
    def get_all_data_providers(cls) -> dict[str, Type[IDataProvider]]:
        """Get all registered data provider classes."""
        return cls._data_providers.copy()

    @classmethod
    def get_all_account_processors(cls) -> dict[str, Type[IAccountProcessor]]:
        """Get all registered account processor classes."""
        return cls._account_processors.copy()

    @classmethod
    def get_all_brokers(cls) -> dict[str, Type[IBroker]]:
        """Get all registered broker classes."""
        return cls._brokers.copy()


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


def account_processor(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for registering an account processor class.

    Usage:
        @account_processor("my_exchange")
        class MyExchangeAccountProcessor(IAccountProcessor):
            def __init__(self, exchange_name, channel, time_provider, ...):
                ...
    """
    return ConnectorRegistry.register_account_processor(name)


def broker(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for registering a broker class.

    Usage:
        @broker("my_exchange")
        class MyExchangeBroker(IBroker):
            def __init__(self, exchange_name, channel, time_provider, account, ...):
                ...
    """
    return ConnectorRegistry.register_broker(name)
