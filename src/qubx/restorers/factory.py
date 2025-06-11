"""
Factory functions for creating restorers.

This module provides factory functions for creating restorers
based on configuration.
"""

from typing import Type

from qubx.core.lookups import LookupsManager
from qubx.restorers.balance import CsvBalanceRestorer, MongoDBBalanceRestorer
from qubx.restorers.interfaces import IBalanceRestorer, IPositionRestorer, ISignalRestorer, IStateRestorer
from qubx.restorers.position import CsvPositionRestorer, MongoDBPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer, MongoDBSignalRestorer
from qubx.restorers.state import CsvStateRestorer, MongoDBStateRestorer

# Registry of position restorer types
POSITION_RESTORER_REGISTRY: dict[str, Type[IPositionRestorer]] = {
    "CsvPositionRestorer": CsvPositionRestorer,
    "MongoDBPositionRestorer": MongoDBPositionRestorer,
}


# Registry of signal restorer types
SIGNAL_RESTORER_REGISTRY: dict[str, Type[ISignalRestorer]] = {
    "CsvSignalRestorer": CsvSignalRestorer,
    "MongoDBSignalRestorer": MongoDBSignalRestorer,
}


# Registry of balance restorer types
BALANCE_RESTORER_REGISTRY: dict[str, Type[IBalanceRestorer]] = {
    "CsvBalanceRestorer": CsvBalanceRestorer,
    "MongoDBBalanceRestorer": MongoDBBalanceRestorer,
}


# Registry of state restorer types
STATE_RESTORER_REGISTRY: dict[str, Type[IStateRestorer]] = {
    "CsvStateRestorer": CsvStateRestorer,
    "MongoDBStateRestorer": MongoDBStateRestorer,
}


def create_position_restorer(
    restorer_type: str, parameters: dict | None = None, lookup: LookupsManager | None = None
) -> IPositionRestorer:
    """
    Create a position restorer based on configuration.

    Args:
        restorer_type: The type of position restorer to create.
        parameters: Parameters to pass to the restorer constructor.
        lookup: Optional GlobalLookup instance for finding instruments.
            If None, instruments will be created directly.

    Returns:
        An instance of the specified position restorer.

    Raises:
        ValueError: If the specified restorer type is not registered.
    """
    if restorer_type not in POSITION_RESTORER_REGISTRY:
        raise ValueError(
            f"Unknown position restorer type: {restorer_type}. "
            f"Available types: {', '.join(POSITION_RESTORER_REGISTRY.keys())}"
        )

    restorer_class = POSITION_RESTORER_REGISTRY[restorer_type]
    params = parameters.copy() if parameters else {}

    # Add lookup to parameters if provided
    if lookup is not None:
        params["lookup"] = lookup

    return restorer_class(**params)


def create_signal_restorer(
    restorer_type: str, parameters: dict | None = None, lookup: LookupsManager | None = None
) -> ISignalRestorer:
    """
    Create a signal restorer based on configuration.

    Args:
        restorer_type: The type of signal restorer to create.
        parameters: Parameters to pass to the restorer constructor.
        lookup: Optional GlobalLookup instance for finding instruments.
            If None, instruments will be created directly.

    Returns:
        An instance of the specified signal restorer.

    Raises:
        ValueError: If the specified restorer type is not registered.
    """
    if restorer_type not in SIGNAL_RESTORER_REGISTRY:
        raise ValueError(
            f"Unknown signal restorer type: {restorer_type}. "
            f"Available types: {', '.join(SIGNAL_RESTORER_REGISTRY.keys())}"
        )

    restorer_class = SIGNAL_RESTORER_REGISTRY[restorer_type]
    params = parameters.copy() if parameters else {}

    # Add lookup to parameters if provided
    if lookup is not None:
        params["lookup"] = lookup

    return restorer_class(**params)


def create_balance_restorer(restorer_type: str, parameters: dict | None = None) -> IBalanceRestorer:
    """
    Create a balance restorer based on configuration.

    Args:
        restorer_type: The type of balance restorer to create.
        parameters: Parameters to pass to the restorer constructor.

    Returns:
        An instance of the specified balance restorer.

    Raises:
        ValueError: If the specified restorer type is not registered.
    """
    if restorer_type not in BALANCE_RESTORER_REGISTRY:
        raise ValueError(
            f"Unknown balance restorer type: {restorer_type}. "
            f"Available types: {', '.join(BALANCE_RESTORER_REGISTRY.keys())}"
        )

    restorer_class = BALANCE_RESTORER_REGISTRY[restorer_type]
    params = parameters.copy() if parameters else {}

    return restorer_class(**params)


def create_state_restorer(restorer_type: str, parameters: dict | None = None) -> IStateRestorer:
    """
    Create a state restorer based on configuration.

    Args:
        restorer_type: The type of state restorer to create.
        parameters: Parameters to pass to the restorer constructor.

    Returns:
        An instance of the specified state restorer.

    Raises:
        ValueError: If the specified restorer type is not registered.
    """
    if restorer_type not in STATE_RESTORER_REGISTRY:
        raise ValueError(
            f"Unknown state restorer type: {restorer_type}. "
            f"Available types: {', '.join(STATE_RESTORER_REGISTRY.keys())}"
        )

    restorer_class = STATE_RESTORER_REGISTRY[restorer_type]
    params = parameters.copy() if parameters else {}
    return restorer_class(**params)


def register_position_restorer(restorer_type: str, restorer_class: Type[IPositionRestorer]) -> None:
    """
    Register a new position restorer type.

    Args:
        restorer_type: The name of the restorer type.
        restorer_class: The restorer class to register.
    """
    POSITION_RESTORER_REGISTRY[restorer_type] = restorer_class


def register_signal_restorer(restorer_type: str, restorer_class: Type[ISignalRestorer]) -> None:
    """
    Register a new signal restorer type.

    Args:
        restorer_type: The name of the restorer type.
        restorer_class: The restorer class to register.
    """
    SIGNAL_RESTORER_REGISTRY[restorer_type] = restorer_class


def register_balance_restorer(restorer_type: str, restorer_class: Type[IBalanceRestorer]) -> None:
    """
    Register a new balance restorer type.

    Args:
        restorer_type: The name of the restorer type.
        restorer_class: The restorer class to register.
    """
    BALANCE_RESTORER_REGISTRY[restorer_type] = restorer_class


def register_state_restorer(restorer_type: str, restorer_class: Type[IStateRestorer]) -> None:
    """
    Register a new state restorer type.

    Args:
        restorer_type: The name of the restorer type.
        restorer_class: The restorer class to register.
    """
    STATE_RESTORER_REGISTRY[restorer_type] = restorer_class
