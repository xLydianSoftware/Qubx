"""Central account-management subsystem.

- ``manager``       — AccountManager (routing + read facade + ticks) and the
                      SimulatedAccountManager backtest variant.
- ``state``         — AccountState, the pure-data container the reducer mutates.
- ``state_machine`` — pure order-lifecycle transition rules (the bug-magnet, isolated).
- ``reducer``       — pure free-function event handlers + ApplyResult, the apply()
                      contract the ProcessingManager fires callbacks from.
- ``reconciler``    — FSM task engine owning snapshot reconcile + order/position tasks.
- ``reconcile``     — order-transition chokepoint + shared leaf helpers.
- ``config``        — AccountManagerConfig (tick intervals/thresholds/retention).

The account events themselves live in ``qubx.core.events`` (shared with connectors and
the backtester); they are re-exported here so the package presents the same public
surface as the event-typed apply path it implements.
"""

from qubx.core.account_manager.config import AccountManagerConfig
from qubx.core.account_manager.manager import AccountManager, SimulatedAccountManager
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.account_manager.state_machine import (
    TRANSITIONS,
    can_transition,
    validate_transition,
)
from qubx.core.events import (
    AccountMessage,
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    DealEvent,
    FundingPaymentEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)

__all__ = [
    "AccountState",
    "AccountMessage",
    "AccountSnapshot",
    "AccountSnapshotEvent",
    "BalanceUpdateEvent",
    "DealEvent",
    "FundingPaymentEvent",
    "OrderAcceptedEvent",
    "OrderCanceledEvent",
    "OrderCancelRejectedEvent",
    "OrderEvent",
    "OrderExpiredEvent",
    "OrderFilledEvent",
    "OrderPartiallyFilledEvent",
    "OrderRejectedEvent",
    "OrderUpdatedEvent",
    "OrderUpdateRejectedEvent",
    "AccountManager",
    "SimulatedAccountManager",
    "AccountManagerConfig",
    "ApplyResult",
    "VenueAccountFigures",
    "TRANSITIONS",
    "can_transition",
    "validate_transition",
]
