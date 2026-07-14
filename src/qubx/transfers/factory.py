"""
Factory functions for creating transfer managers.
"""

import inspect
from collections.abc import Callable

from qubx.core.interfaces import ITransferManager
from qubx.transfers.xchanges import XChangesTransferService

# Registry of transfer manager types (lowercase keys); extend via register_transfer_manager
TRANSFER_MANAGER_REGISTRY: dict[str, type[ITransferManager]] = {
    "xchanges": XChangesTransferService,
}


def resolve_transfer_manager_class(transfer_manager_type: str) -> type[ITransferManager]:
    """
    Resolve a registered transfer manager class (case-insensitive).

    Raises:
        ValueError: If the specified transfer manager type is not registered.
    """
    cls = TRANSFER_MANAGER_REGISTRY.get(transfer_manager_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown transfer manager type: {transfer_manager_type}. "
            f"Available types: {', '.join(TRANSFER_MANAGER_REGISTRY)}"
        )
    return cls


def create_transfer_manager(
    transfer_manager_type: str,
    parameters: dict | None = None,
    is_simulation: Callable[[], bool] | None = None,
) -> ITransferManager:
    """
    Create a transfer manager based on configuration.

    Args:
        transfer_manager_type: The type of transfer manager to create (case-insensitive registry key).
        parameters: Parameters to pass to the transfer manager constructor.
        is_simulation: Framework guard, injected only when the constructor accepts it.

    Returns:
        An instance of the specified transfer manager.

    Raises:
        ValueError: If the specified transfer manager type is not registered.
        TypeError: If parameters contain a name the constructor does not accept.
    """
    cls = resolve_transfer_manager_class(transfer_manager_type)
    # no param filtering: a mistyped kwarg (e.g. max_amount) must raise TypeError, not silently drop a safety cap
    params = dict(parameters) if parameters else {}
    if is_simulation is not None and "is_simulation" in inspect.signature(cls).parameters:
        params["is_simulation"] = is_simulation
    return cls(**params)


def register_transfer_manager(type_name: str, cls: type[ITransferManager]) -> None:
    """
    Register a new transfer manager type (key lowercased).

    Args:
        type_name: The name of the transfer manager type.
        cls: The transfer manager class to register.
    """
    TRANSFER_MANAGER_REGISTRY[type_name.lower()] = cls
