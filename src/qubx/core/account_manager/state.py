"""Pure per-exchange financial state.

One instance per exchange, held inside AccountManager's dict[str, AccountState].
This object is *data + indices only*: it enforces no state-machine legality,
runs no reconcile rules, fires no callbacks, and depends on no clock. The state
machine and reconcile orchestration live in AccountManager; AccountState just
stores orders/positions/balances and keeps its lookup indices consistent.

Threading: single-thread, mutable. All writes happen on the strategy thread
through AccountManager (the single-mutator invariant). The mutators are
framework-internal — strategies only see the read API. The read methods hand
out live element references; callers that stash an Order/Position for off-thread
use must copy.copy() it first.
"""

from collections import Counter, deque

import numpy as np

from qubx import logger
from qubx.core.basics import Balance, Deal, Instrument, Order, OrderStatus, OrderTransition, Position

_INFLIGHT_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.SUBMITTED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
    }
)
_TERMINAL_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)
_PENDING_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.PENDING_CANCEL,
        OrderStatus.PENDING_UPDATE,
    }
)


class AccountState:
    __slots__ = (
        "exchange",
        "_active_orders",
        "_positions",
        "_balances",
        "_venue_id_index",
        "_inflight_index",
        "_pending_evict_index",
        "_seen_trade_ids",
        "_terminal_history",
        "_retry_count",
        "_pre_pending_status",
        "_last_snapshot_as_of",
        "_transition_counts",
    )

    def __init__(self, exchange: str, *, terminal_history_size: int = 10_000):
        self.exchange: str = exchange

        # ---- primary data ------------------------------------------------
        self._active_orders: dict[str, Order] = {}  # client_order_id -> Order
        self._positions: dict[Instrument, Position] = {}
        self._balances: dict[str, Balance] = {}  # currency -> Balance

        # ---- lookup / bookkeeping indices (mutator-maintained) -----------
        # venue_order_id -> client_order_id
        self._venue_id_index: dict[str, str] = {}
        # cids in SUBMITTED / PENDING_* — lets the in-flight sweep run O(k)
        self._inflight_index: set[str] = set()
        # cid -> terminal-at timestamp; drives O(k) terminal eviction
        self._pending_evict_index: dict[str, np.datetime64] = {}
        # cid -> applied venue trade_ids; fill dedup. Created lazily on first
        # fill (resting/canceled orders never allocate a set), dropped on
        # eviction so the memory dies with the order.
        self._seen_trade_ids: dict[str, set[str]] = {}
        # bounded ring buffer of evicted terminals (FIFO); slow-path lookups
        self._terminal_history: deque[Order] = deque(maxlen=terminal_history_size)
        # cid -> in-flight sweep poll count; reset on any status change
        self._retry_count: dict[str, int] = {}
        # cid -> status captured on entry to PENDING_*; revert target on reject/give-up
        self._pre_pending_status: dict[str, OrderStatus] = {}

        # ratchet for out-of-order snapshot rejection (written by AM reconcile)
        self._last_snapshot_as_of: np.datetime64 | None = None
        # Audit counter: number of status transitions by destination status (status.value ->
        # count). Read via AccountManager.get_metrics(); never reset within a session.
        self._transition_counts: Counter = Counter()

    def __repr__(self) -> str:
        return (
            f"AccountState({self.exchange}: {len(self._active_orders)} active orders, "
            f"{len(self._inflight_index)} in-flight, {len(self._positions)} positions, "
            f"{len(self._balances)} balances, {len(self._terminal_history)} retained)"
        )

    def get_orders(self) -> dict[str, Order]:
        return dict(self._active_orders)

    def get_order(self, client_order_id: str) -> Order | None:
        order = self._active_orders.get(client_order_id)
        if order is not None:
            return order
        # Slow path: recently-evicted terminals (most-recent first).
        for o in reversed(self._terminal_history):
            if o.client_order_id == client_order_id:
                return o
        return None

    def get_active_order(self, client_order_id: str) -> Order | None:
        # Active (non-evicted) order only — distinct from get_order, which also searches
        # the terminal-history ring buffer.
        return self._active_orders.get(client_order_id)

    def has_active_order(self, client_order_id: str) -> bool:
        return client_order_id in self._active_orders

    def get_order_by_venue_id(self, venue_order_id: str) -> Order | None:
        cid = self._venue_id_index.get(venue_order_id)
        return self._active_orders.get(cid) if cid is not None else None

    def get_inflight_orders(self) -> list[Order]:
        return [self._active_orders[cid] for cid in self._inflight_index]

    def get_positions(self) -> dict[Instrument, Position]:
        return dict(self._positions)

    def get_position(self, instrument: Instrument) -> Position | None:
        return self._positions.get(instrument)

    def get_balance(self, currency: str) -> Balance | None:
        return self._balances.get(currency)

    def get_balances(self) -> list[Balance]:
        return list(self._balances.values())

    def get_pre_pending(self, cid: str) -> OrderStatus | None:
        return self._pre_pending_status.get(cid)

    def get_retry(self, cid: str) -> int:
        return self._retry_count.get(cid, 0)

    def get_last_snapshot_as_of(self) -> np.datetime64 | None:
        return self._last_snapshot_as_of

    def get_transition_counts(self) -> dict[str, int]:
        """Audit counter snapshot: status.value -> number of transitions into it."""
        return dict(self._transition_counts)

    # ================================================================== #
    # Mutators — framework-internal; only AccountManager calls these,    #
    # on the strategy thread. No legality checks here: the transition    #
    # table is enforced upstream in AccountManager. These setters' only  #
    # job is to mutate fields and keep every index consistent.           #
    # ================================================================== #

    def add_order(self, order: Order) -> None:
        cid = order.client_order_id
        self._active_orders[cid] = order
        if order.venue_order_id is not None:
            self._venue_id_index[order.venue_order_id] = cid
        if order.status in _INFLIGHT_STATUSES:
            self._inflight_index.add(cid)
        elif order.status in _TERMINAL_STATUSES:
            if order.last_updated_at is None:
                raise ValueError("terminal orders must have last_updated_at set for eviction")
            self._pending_evict_index[cid] = order.last_updated_at

    def transition_order(self, cid: str, new_status: OrderStatus, now: np.datetime64) -> Order:
        """Low-level status setter and the sole maintainer of every status-derived
        structure: the in-flight and pending-evict indices, the retry counter, and
        the pre-pending status capture.
        """
        order = self._active_orders[cid]
        old_status = order.status
        order.transitions.append(OrderTransition(time=now, from_status=old_status, to_status=new_status))
        self._transition_counts[new_status.value] += 1
        order.status = new_status
        order.last_updated_at = now

        if new_status in _INFLIGHT_STATUSES:
            self._inflight_index.add(cid)
        else:
            self._inflight_index.discard(cid)

        if new_status in _TERMINAL_STATUSES:
            self._pending_evict_index[cid] = now
        else:
            self._pending_evict_index.pop(cid, None)

        if new_status in _PENDING_STATUSES:
            # capture only on first entry, so PENDING_UPDATE -> PENDING_CANCEL keeps the original
            if old_status not in _PENDING_STATUSES:
                self._pre_pending_status[cid] = old_status
        else:
            self._pre_pending_status.pop(cid, None)

        self._retry_count.pop(cid, None)
        return order

    def bump_retry(self, cid: str) -> int:
        count = self._retry_count.get(cid, 0) + 1
        self._retry_count[cid] = count
        return count

    def set_venue_id(self, cid: str, venue_order_id: str) -> None:
        order = self._active_orders[cid]
        if order.venue_order_id is not None:
            # drop stale key before re-pointing
            self._venue_id_index.pop(order.venue_order_id, None)
        order.venue_order_id = venue_order_id
        self._venue_id_index[venue_order_id] = cid

    def apply_fill(self, cid: str, fill: Deal, now: np.datetime64) -> bool:
        """Apply a fill, deduped by trade id. Returns True if newly applied, False if duplicate."""
        order = self._active_orders[cid]
        seen = self._seen_trade_ids.setdefault(cid, set())
        if fill.trade_id in seen:
            logger.debug(f"duplicate fill {fill.trade_id} on {cid}; skipping")
            return False
        seen.add(fill.trade_id)
        order.record_fill(fill.amount, fill.price)  # filled_quantity + avg-price math lives on Order
        order.last_updated_at = now
        return True

    def remove_order(self, cid: str) -> None:
        # Drop an order from state entirely (e.g. a submit that raised before reaching the
        # venue) — distinct from evict_to_history, which retains it in the ring buffer.
        order = self._active_orders.pop(cid, None)
        if order is None:
            return
        if order.venue_order_id is not None:
            self._venue_id_index.pop(order.venue_order_id, None)
        self._inflight_index.discard(cid)
        self._pending_evict_index.pop(cid, None)
        self._seen_trade_ids.pop(cid, None)
        self._retry_count.pop(cid, None)
        self._pre_pending_status.pop(cid, None)

    def evict_to_history(self, cid: str) -> None:
        """Evict a terminal order from active state into the history ring buffer,
        dropping its index entries. Called by AM's eviction sweep after the
        terminal grace period elapses.
        """
        order = self._active_orders.pop(cid, None)
        if order is None:
            return
        if order.venue_order_id is not None:
            self._venue_id_index.pop(order.venue_order_id, None)
        self._inflight_index.discard(cid)
        self._pending_evict_index.pop(cid, None)
        self._seen_trade_ids.pop(cid, None)
        self._retry_count.pop(cid, None)
        self._pre_pending_status.pop(cid, None)
        self._terminal_history.append(order)

    def prune_terminal_orders(self, now: np.datetime64, retention: np.timedelta64) -> None:
        due = [cid for cid, terminal_at in self._pending_evict_index.items() if (now - terminal_at) >= retention]
        for cid in due:
            self.evict_to_history(cid)

    def mark_snapshot_applied(self, as_of: np.datetime64) -> None:
        self._last_snapshot_as_of = as_of

    def is_snapshot_stale(self, as_of: np.datetime64) -> bool:
        return self._last_snapshot_as_of is not None and as_of <= self._last_snapshot_as_of

    def set_position(self, instrument: Instrument, position: Position) -> None:
        # Identity-preserving: an existing Position is updated in place (callers across
        # the framework hold references to it), never swapped for a new object.
        existing = self._positions.get(instrument)
        if existing is None:
            self._positions[instrument] = position
        else:
            existing.reset_by_position(position)

    def update_balance(self, currency: str, balance: Balance) -> None:
        # Identity-preserving, like set_position.
        existing = self._balances.get(currency)
        if existing is None:
            self._balances[currency] = balance
        else:
            existing.reset_by_balance(balance)

    def apply_position_snapshot(self, position: Position) -> bool:
        # Reconcile a snapshot position into state; returns True if it was new or its
        # size/price changed (so the caller records it in the reconcile diff).
        existing = self._positions.get(position.instrument)
        changed = (
            existing is None
            or existing.quantity != position.quantity
            or existing.position_avg_price != position.position_avg_price
        )
        self.set_position(position.instrument, position)
        return changed

    def apply_balance_snapshot(self, balance: Balance) -> bool:
        existing = self._balances.get(balance.currency)
        changed = existing is None or (
            existing.total != balance.total or existing.free != balance.free or existing.locked != balance.locked
        )
        self.update_balance(balance.currency, balance)
        return changed
