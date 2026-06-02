"""
``IConnector`` — the per-exchange connector contract for the redesign.

A connector owns the venue session (REST + WS), turns ``ctx.trade/cancel/update`` calls
into venue requests, and emits typed ``AccountMessage`` events onto the Channel. It never
mutates ``AccountState`` directly — it is an event *producer*; the dispatch thread is the
sole consumer/writer (see ``ACCOUNT_MANAGEMENT_PLAN.md``).

All command methods are fire-and-forget: they dispatch REST/WS work on the connector's own
loop and return immediately. ``submit_order`` may raise *synchronously* on a framework-side
rejection (invalid params, not connected, duplicate cid, open-order cap); venue-side
rejections instead come back later as ``OrderRejectedEvent``.

Derived from the ``account-management`` excalidraw + the connector-contract spec referenced
by the rollout plan (no design doc was available; confirm signatures against it).
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from qubx.core.basics import OrderRequest

ModifyPattern = Literal["in_place", "replace"]


@dataclass
class ConnectorCapabilities:
    """
    Per-venue capability flags consumed by the conformance suite and AccountManager.

    - ``modify_pattern``: ``in_place`` (Binance/OKX/Gate — venue_id unchanged) vs
      ``replace`` (HPL — modify is cancel-old + new; venue_id changes, connector collapses
      the pair into a single ``OrderUpdatedEvent``).
    - ``cancel_by_cloid_supported``: cancel-by-client-order-id (vs only by venue id).
    - ``native_snapshot_push``: venue pushes account snapshots without an explicit request.
    - ``synthesized_trade_id_supported``: connector can synthesize stable trade ids for
      fill dedup when the venue omits them.
    """

    modify_pattern: ModifyPattern = "in_place"
    cancel_by_cloid_supported: bool = False
    native_snapshot_push: bool = False
    synthesized_trade_id_supported: bool = False


@runtime_checkable
class IConnector(Protocol):
    """Venue connector contract. Implementations live in the exchanges repo."""

    exchange: str
    capabilities: ConnectorCapabilities

    def connect(self) -> None:
        """Open the venue session, subscribe to order/deal WS, emit initial snapshot."""
        ...

    def submit_order(self, request: OrderRequest) -> None:
        """Fire the submit in the background; raise synchronously on framework rejection."""
        ...

    def cancel_order(self, client_order_id: str, venue_order_id: str | None = None) -> None:
        """Request a cancel (by venue id when present, else by cloid). Fire-and-forget."""
        ...

    def update_order(
        self,
        client_order_id: str,
        venue_order_id: str | None = None,
        price: float | None = None,
        quantity: float | None = None,
    ) -> None:
        """Request a modify. Fire-and-forget."""
        ...

    def request_order_status(self, client_order_id: str, venue_order_id: str | None = None) -> None:
        """Schedule an async venue status query; result comes back as a typed event."""
        ...

    def request_snapshot(self) -> None:
        """Schedule an async account snapshot; result comes back as ``AccountSnapshotEvent``."""
        ...
