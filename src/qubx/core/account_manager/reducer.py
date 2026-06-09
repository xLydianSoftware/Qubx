"""The apply contract for typed AccountMessage application.

Pure state mutation: applying an event never fires callbacks or touches connectors.
AccountManager.apply mutates one AccountState and returns an ApplyResult; the
ProcessingManager fires strategy callbacks from the returned result, error-isolated.
None fields are the suppress signal — deduped duplicate fills, late events on terminal
orders, and rejects/lifecycle events for unknown orders all return empty results, so no
callback fires. (The handlers themselves still live on AccountManager; their extraction
into this module as pure free functions is a later step.)
"""

from dataclasses import dataclass

from qubx.core.basics import Deal, Order, OrderChange, Position


@dataclass
class ApplyResult:
    order: Order | None = None  # status changed -> on_order(order, order_change)
    order_change: OrderChange | None = None  # paired with order
    deal: Deal | None = None  # new deal applied -> on_execution
    position: Position | None = None  # position changed -> on_position_change
