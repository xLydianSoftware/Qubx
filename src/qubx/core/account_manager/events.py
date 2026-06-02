"""
Typed event hierarchy for the account-management redesign.

Everything that flows on the ``Channel`` is a ``ChannelMessage``. The redesign splits
these into two families:

  - ``AccountMessage`` — account / order-lifecycle events emitted by connectors
    (``IConnector.send(...)``) and consumed by ``AccountManager.apply(event)``.
  - ``MarketDataMessage`` — market data (quotes, trades, ohlc, ...).

Q1 scope is the account side: this module defines the order-lifecycle leaves and the
account snapshot. ``MarketDataMessage`` is declared as the sibling category base only —
market-data leaves continue to travel the existing qubx market-data path for now and are
not part of this PR. Field shapes are derived from the ``account-management`` excalidraw
order flows (no design doc was available; confirm names against it).
"""

from dataclasses import dataclass, field

from qubx.core.account_manager.state import ManagedOrder
from qubx.core.basics import AssetBalance, Deal, Position, dt_64


@dataclass(frozen=True)
class ChannelMessage:
    """Base for everything that travels on the Channel."""


@dataclass(frozen=True)
class MarketDataMessage(ChannelMessage):
    """Sibling category base for market data (leaves out of Q1 scope)."""


@dataclass(frozen=True)
class AccountMessage(ChannelMessage):
    """Base for account / order-lifecycle events."""


@dataclass(frozen=True)
class OrderEvent(AccountMessage):
    """Base for events about a single order, addressed by its ``client_id`` (cid)."""

    client_id: str
    timestamp: dt_64 | None = None


@dataclass(frozen=True)
class OrderAcceptedEvent(OrderEvent):
    """Venue acknowledged the order; ``venue_id`` is now known."""

    venue_id: str | None = None


@dataclass(frozen=True)
class OrderRejectedEvent(OrderEvent):
    """Venue or framework rejected the order (or it never appeared after retries)."""

    reason: str | None = None


@dataclass(frozen=True)
class OrderFilledEvent(OrderEvent):
    """A (partial or full) fill arrived; ``fill`` carries the trade (``fill.id`` = trade id)."""

    fill: Deal | None = None


@dataclass(frozen=True)
class OrderCanceledEvent(OrderEvent):
    """Venue acknowledged the cancel."""


@dataclass(frozen=True)
class OrderCancelRejectedEvent(OrderEvent):
    """Venue rejected the cancel (e.g. order not found / already terminal)."""

    reason: str | None = None


@dataclass(frozen=True)
class OrderUpdatedEvent(OrderEvent):
    """Venue confirmed a modify. For replace-style venues ``venue_id`` may change."""

    venue_id: str | None = None
    price: float | None = None
    quantity: float | None = None


@dataclass(frozen=True)
class OrderUpdateRejectedEvent(OrderEvent):
    """Venue rejected a modify; the underlying order stays alive with prior params."""

    reason: str | None = None


@dataclass(frozen=True)
class AccountSnapshotEvent(AccountMessage):
    """
    Full account snapshot emitted by a connector at ``connect()`` (and on demand via
    ``request_snapshot``). Used by ``AccountManager`` to reconcile state.
    """

    orders: list[ManagedOrder] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    balances: list[AssetBalance] = field(default_factory=list)
    timestamp: dt_64 | None = None
