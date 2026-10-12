"""Amend emulated as cancel + recreate, for an order the venue cannot modify in place (Binance PM
margin orders: papi has no margin amend).

The framework keeps one order identity (the client order id) across the venue orders that carry
it: A, cancelled, then B, created with the same client id. The venue only refuses a client id
already in use by an OPEN order, so the reuse is legal once A's cancel is confirmed.

The connector runs the cancel and the create; ``Replacement`` holds what it needs to keep the
venue reports of both orders consistent with that single identity:

- A's own CANCELED is the replace's, not the order's end: dropped. So is any later ACCEPTED of A,
  which would point the order back at A's venue id.
- A's fills are real and are booked; once B exists they are re-addressed to B.
- B's reports count only B's fills and size: ``filled_before`` (what A filled) is added back, so
  the cumulative figures match the order's lifetime ones the AccountManager keeps.
"""

from dataclasses import dataclass, field

from qubx.core.basics import Instrument, Order, OrderStatus

# - statuses that only describe the venue order's own life; for a superseded venue order they
#   say nothing about the framework order
_LIFECYCLE_ONLY = (OrderStatus.ACCEPTED, OrderStatus.CANCELED, OrderStatus.EXPIRED)


@dataclass
class Replacement:
    instrument: Instrument
    # - venue order being cancelled by the replace in flight (None when none is in flight)
    cancelling_vid: str | None
    # - the live venue order of this client id (None until the recreate succeeds)
    current_vid: str | None = None
    # - every venue id this client id carried before the current one
    superseded: set[str] = field(default_factory=set)
    filled_before: float = 0.0
    notional_before: float = 0.0
    cancel_requested: bool = False
    # - the old order is cancelled and the recreate request is out
    creating: bool = False

    @property
    def in_flight(self) -> bool:
        return self.cancelling_vid is not None or self.creating

    def begin(self, vid: str) -> None:
        """A replace of venue order ``vid`` starts (it may be a previous replace's B)."""
        self.cancelling_vid = vid
        self.current_vid = vid
        self.creating = False
        self.cancel_requested = False

    def cancelled(self, vid: str, filled: float, cost: float) -> None:
        """``vid`` is confirmed cancelled with ``filled`` executed on it for ``cost`` quote (the
        cancel response's figures)."""
        self.superseded.add(vid)
        self.cancelling_vid = None
        self.creating = True
        if filled > 0:
            self.filled_before += filled
            self.notional_before += cost
        if self.current_vid == vid:
            self.current_vid = None

    def recreated(self, vid: str) -> None:
        self.current_vid = vid
        self.creating = False

    def abandoned(self) -> None:
        """The replace ended without a live venue order (refused or unknown recreate)."""
        self.cancelling_vid = None
        self.creating = False

    def view(self, order: Order) -> Order | None:
        """The report ``order`` (freshly converted, owned by the caller) as the framework order
        sees it, or None when it must not reach the AccountManager."""
        vid = order.venue_order_id
        if vid is not None and vid == self.cancelling_vid:
            # - the order the replace is cancelling is still the live one: its fills (even a
            #   FILLED that beats the cancel) are the order's; only its CANCELED is ours
            if order.status == OrderStatus.CANCELED:
                return None
            return self._lifetime(order)
        if vid is not None and vid in self.superseded:
            if order.status in _LIFECYCLE_ONLY:
                return None
            # - a late fill of a superseded venue order: booked, addressed to the live one
            order.venue_order_id = self.current_vid
            order.status = OrderStatus.PARTIALLY_FILLED
            return order
        if vid is not None and vid == self.current_vid:
            return self._lifetime(order)
        return order

    def _lifetime(self, order: Order) -> Order:
        if self.filled_before <= 0:
            return order
        own = order.filled_quantity or 0.0
        own_notional = own * (order.avg_fill_price or 0.0)
        order.quantity += self.filled_before
        order.filled_quantity = own + self.filled_before
        order.avg_fill_price = (self.notional_before + own_notional) / order.filled_quantity
        return order

    def is_done(self, order: Order) -> bool:
        """True when ``order`` (a viewed report) ends the framework order: nothing left to track."""
        return (
            not self.in_flight
            and order.venue_order_id == self.current_vid
            and order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED)
        )
