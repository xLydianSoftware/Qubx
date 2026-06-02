from collections import deque
from dataclasses import dataclass, field

import numpy as np

from qubx.core.basics import Balance, Deal, Instrument, Order, OrderStatus, Position


@dataclass
class AccountState:
    exchange: str
    _active_orders: dict[str, Order] = field(default_factory=dict)
    _positions: dict[Instrument, Position] = field(default_factory=dict)
    _balances: dict[str, Balance] = field(default_factory=dict)

    _venue_id_index: dict[str, str] = field(default_factory=dict)
    _inflight_index: set[str] = field(default_factory=set)
    _pending_evict_index: dict[str, np.datetime64] = field(default_factory=dict)
    _terminal_history: deque = field(default_factory=lambda: deque(maxlen=10_000))
    _last_snapshot_as_of: np.datetime64 | None = None

    def get_orders(self) -> dict[str, Order]:
        return dict(self._active_orders)

    def get_order(self, client_order_id: str) -> Order | None:
        if (o := self._active_orders.get(client_order_id)) is not None:
            return o
        for hist in self._terminal_history:
            if hist.client_order_id == client_order_id:
                return hist
        return None

    def get_order_by_venue_id(self, venue_order_id: str) -> Order | None:
        cid = self._venue_id_index.get(venue_order_id)
        return self._active_orders.get(cid) if cid else None

    def get_active_order(self, client_order_id: str) -> Order | None:
        # Active (non-evicted) order only — distinct from get_order, which also searches
        # the terminal-history ring buffer.
        return self._active_orders.get(client_order_id)

    def has_active_order(self, client_order_id: str) -> bool:
        return client_order_id in self._active_orders

    def get_inflight_orders(self) -> list[Order]:
        # Orders still awaiting a venue verdict (SUBMITTED / PENDING_*). Prunes any cid whose
        # order was evicted or is no longer in-flight (lazy index maintenance).
        out: list[Order] = []
        for cid in list(self._inflight_index):
            order = self._active_orders.get(cid)
            if order is None or not (order.status == OrderStatus.SUBMITTED or order.status.is_pending):
                self._inflight_index.discard(cid)
                continue
            out.append(order)
        return out

    def get_positions(self) -> dict[Instrument, Position]:
        return dict(self._positions)

    def get_position(self, instrument: Instrument) -> Position | None:
        return self._positions.get(instrument)

    def get_balance(self, currency: str) -> Balance | None:
        return self._balances.get(currency)

    def get_balances(self) -> list[Balance]:
        return list(self._balances.values())

    def add_order(self, order: Order) -> None:
        self._active_orders[order.client_order_id] = order
        if order.venue_order_id is not None:
            self._venue_id_index[order.venue_order_id] = order.client_order_id
        if not order.status.is_terminal:
            self._inflight_index.add(order.client_order_id)

    def transition_order(self, cid: str, new_status: OrderStatus, now: np.datetime64) -> Order:
        order = self._active_orders[cid]
        order.status = new_status
        order.last_updated_at = now
        if new_status.is_terminal:
            self._inflight_index.discard(cid)
            self._pending_evict_index[cid] = now
        elif new_status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
            self._inflight_index.discard(cid)
        else:
            self._inflight_index.add(cid)
        return order

    def set_venue_id(self, cid: str, venue_order_id: str) -> None:
        order = self._active_orders[cid]
        # Re-key: drop any previous venue id this order was indexed under before re-pointing.
        if order.venue_order_id is not None:
            self._venue_id_index.pop(order.venue_order_id, None)
        order.venue_order_id = venue_order_id
        self._venue_id_index[venue_order_id] = cid

    def apply_fill(self, cid: str, fill: Deal, now: np.datetime64) -> Order:
        order = self._active_orders[cid]
        if fill.trade_id in order.seen_trade_ids:
            return order
        order.seen_trade_ids.add(fill.trade_id)
        order.filled_quantity += abs(fill.amount)
        if order.avg_fill_price is None:
            order.avg_fill_price = fill.price
        else:
            total = order.filled_quantity
            prev = total - abs(fill.amount)
            order.avg_fill_price = (order.avg_fill_price * prev + fill.price * abs(fill.amount)) / total
        order.last_updated_at = now
        return order

    def remove_order(self, cid: str) -> None:
        order = self._active_orders.pop(cid, None)
        if order is not None and order.venue_order_id is not None:
            self._venue_id_index.pop(order.venue_order_id, None)
        self._inflight_index.discard(cid)
        self._pending_evict_index.pop(cid, None)

    def evict_to_history(self, cid: str) -> None:
        order = self._active_orders.pop(cid, None)
        if order is not None:
            self._terminal_history.append(order)
            if order.venue_order_id is not None:
                self._venue_id_index.pop(order.venue_order_id, None)
        self._pending_evict_index.pop(cid, None)

    def set_position(self, instrument: Instrument, position: Position) -> None:
        self._positions[instrument] = position

    def update_balance(self, currency: str, balance: Balance) -> None:
        self._balances[currency] = balance

    def is_snapshot_stale(self, as_of: np.datetime64) -> bool:
        return self._last_snapshot_as_of is not None and as_of <= self._last_snapshot_as_of

    def mark_snapshot_applied(self, as_of: np.datetime64) -> None:
        self._last_snapshot_as_of = as_of

    def get_last_snapshot_as_of(self) -> np.datetime64 | None:
        return self._last_snapshot_as_of

    def evict_due_terminals(self, now: np.datetime64, grace: np.timedelta64, history_size: int) -> None:
        # Move terminal orders past the grace window into the bounded history ring buffer,
        # resizing it first if the configured size changed.
        if self._terminal_history.maxlen != history_size:
            self._terminal_history = deque(self._terminal_history, maxlen=history_size)
        for cid in list(self._pending_evict_index):
            if (now - self._pending_evict_index[cid]) >= grace:
                self.evict_to_history(cid)
