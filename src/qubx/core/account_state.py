from collections import deque
from dataclasses import dataclass, field

import numpy as np

from qubx.core.basics import Balance, Deal, Instrument, Order, OrderStatus, Position


@dataclass
class AccountState:
    exchange: str
    active_orders: dict[str, Order] = field(default_factory=dict)
    positions: dict[Instrument, Position] = field(default_factory=dict)
    balances: dict[str, Balance] = field(default_factory=dict)

    _venue_id_index: dict[str, str] = field(default_factory=dict)
    _inflight_index: set[str] = field(default_factory=set)
    _pending_evict_index: dict[str, np.datetime64] = field(default_factory=dict)
    _terminal_history: deque = field(default_factory=lambda: deque(maxlen=10_000))
    _last_snapshot_as_of: np.datetime64 | None = None

    def get_orders(self) -> dict[str, Order]:
        return dict(self.active_orders)

    def get_order(self, client_order_id: str) -> Order | None:
        if (o := self.active_orders.get(client_order_id)) is not None:
            return o
        for hist in self._terminal_history:
            if hist.client_order_id == client_order_id:
                return hist
        return None

    def get_order_by_venue_id(self, venue_order_id: str) -> Order | None:
        cid = self._venue_id_index.get(venue_order_id)
        return self.active_orders.get(cid) if cid else None

    def get_positions(self) -> dict[Instrument, Position]:
        return dict(self.positions)

    def get_position(self, instrument: Instrument) -> Position | None:
        return self.positions.get(instrument)

    def get_balance(self, currency: str) -> Balance | None:
        return self.balances.get(currency)

    def _add_order(self, order: Order) -> None:
        self.active_orders[order.client_order_id] = order
        if order.venue_order_id is not None:
            self._venue_id_index[order.venue_order_id] = order.client_order_id
        if not order.status.is_terminal():
            self._inflight_index.add(order.client_order_id)

    def _transition_order(self, cid: str, new_status: OrderStatus, now: np.datetime64) -> Order:
        order = self.active_orders[cid]
        order.status = new_status
        order.last_updated_at = now
        if new_status.is_terminal():
            self._inflight_index.discard(cid)
            self._pending_evict_index[cid] = now
        elif new_status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
            self._inflight_index.discard(cid)
        else:
            self._inflight_index.add(cid)
        return order

    def _set_venue_id(self, cid: str, venue_order_id: str) -> None:
        order = self.active_orders[cid]
        order.venue_order_id = venue_order_id
        self._venue_id_index[venue_order_id] = cid

    def _apply_fill(self, cid: str, fill: Deal, now: np.datetime64) -> Order:
        order = self.active_orders[cid]
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

    def _remove_order(self, cid: str) -> None:
        order = self.active_orders.pop(cid, None)
        if order is not None and order.venue_order_id is not None:
            self._venue_id_index.pop(order.venue_order_id, None)
        self._inflight_index.discard(cid)
        self._pending_evict_index.pop(cid, None)

    def _evict_to_history(self, cid: str) -> None:
        order = self.active_orders.pop(cid, None)
        if order is not None:
            self._terminal_history.append(order)
            if order.venue_order_id is not None:
                self._venue_id_index.pop(order.venue_order_id, None)
        self._pending_evict_index.pop(cid, None)

    def _set_position(self, instrument: Instrument, position: Position) -> None:
        self.positions[instrument] = position

    def _update_balance(self, currency: str, balance: Balance) -> None:
        self.balances[currency] = balance
