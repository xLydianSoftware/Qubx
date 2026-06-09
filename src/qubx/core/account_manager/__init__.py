"""Central account-management subsystem.

- ``manager``       — AccountManager (state owner + state machine + reconcile) and the
                      SimulatedAccountManager backtest variant.
- ``state``         — AccountState, the pure-data container the manager mutates.
- ``state_machine`` — pure order-lifecycle transition rules (the bug-magnet, isolated).
- ``reducer``       — ApplyResult, the apply() contract the ProcessingManager fires
                      callbacks from (handler extraction lands here later).
- ``config``        — AccountManagerConfig + the tick cron helper.

Public symbols are re-exported here so callers import ``from qubx.core.account_manager
import AccountManager`` without reaching into submodules.
"""

from qubx.core.account_manager.config import AccountManagerConfig
from qubx.core.account_manager.manager import AccountManager, SimulatedAccountManager
from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.account_manager.state_machine import (
    LEGAL_TRANSITIONS,
    can_transition,
    validate_transition,
)

__all__ = [
    "AccountManager",
    "SimulatedAccountManager",
    "AccountManagerConfig",
    "AccountState",
    "VenueAccountFigures",
    "LEGAL_TRANSITIONS",
    "can_transition",
    "validate_transition",
]
