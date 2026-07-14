"""
Transfers module for qubx.

This module provides transfer manager implementations for moving funds
between exchanges in live trading.
"""

from qubx.transfers.factory import (
    TRANSFER_MANAGER_REGISTRY,
    create_transfer_manager,
    register_transfer_manager,
)
from qubx.transfers.xchanges import DEFAULT_WALLET_MAPPING, TransferServiceError, XChangesTransferService

__all__ = [
    "DEFAULT_WALLET_MAPPING",
    "TRANSFER_MANAGER_REGISTRY",
    "TransferServiceError",
    "XChangesTransferService",
    "create_transfer_manager",
    "register_transfer_manager",
]
