"""
Factory functions for creating restorers.

This module provides factory functions for creating position and signal restorers
based on configuration.
"""

from typing import Type

from qubx.restorers.interfaces import IPositionRestorer, ISignalRestorer
from qubx.restorers.position import CsvPositionRestorer
from qubx.restorers.signal import CsvSignalRestorer

# Registry of position restorer types
POSITION_RESTORER_REGISTRY: dict[str, Type[IPositionRestorer]] = {
    "CsvPositionRestorer": CsvPositionRestorer,
}


# Registry of signal restorer types
SIGNAL_RESTORER_REGISTRY: dict[str, Type[ISignalRestorer]] = {
    "CsvSignalRestorer": CsvSignalRestorer,
}


def create_position_restorer(restorer_type: str, parameters: dict | None = None) -> IPositionRestorer:
    """
    Create a position restorer based on configuration.

    Args:
        restorer_type: The type of position restorer to create.
        parameters: Parameters to pass to the restorer constructor.

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
    return restorer_class(**(parameters or {}))


def create_signal_restorer(restorer_type: str, parameters: dict | None = None) -> ISignalRestorer:
    """
    Create a signal restorer based on configuration.

    Args:
        restorer_type: The type of signal restorer to create.
        parameters: Parameters to pass to the restorer constructor.

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
    return restorer_class(**(parameters or {}))


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
