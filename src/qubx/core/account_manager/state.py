"""Pure per-exchange financial state.

One instance per exchange, held inside AccountManager's dict[str, AccountState].
This object is *data + indices only*: it enforces no state-machine legality,
runs no reconcile rules, fires no callbacks, and depends on no clock. The state
machine and reconcile orchestration live in AccountManager; AccountState just
stores orders/positions/balances and keeps its lookup indices consistent.

Threading: single-thread, mutable. All writes happen on the strategy thread
through AccountManager (the single-mutator invariant). The `_`-prefixed mutators
are framework-internal — strategies only see the read API. The read methods hand
out live element references; callers that stash an Order/Position for off-thread
use must copy.copy() it first.
"""

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from qubx import logger
from qubx.core.basics import AssetBalance, Deal, Instrument, Order, OrderStatus, Position

_INFLIGHT_STATUSES: frozenset[OrderStatus] = frozenset({
    OrderStatus.SUBMITTED,
    OrderStatus.PENDING_CANCEL,
    OrderStatus.PENDING_UPDATE,
})
_TERMINAL_STATUSES: frozenset[OrderStatus] = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
})
_PENDING_STATUSES: frozenset[OrderStatus] = frozenset({
    OrderStatus.PENDING_CANCEL,
    OrderStatus.PENDING_UPDATE,
})


@dataclass
class VenueAccountFigures:
    """Account-level figures reported by the exchange. Preferred over derived
    metrics in live; each is optional since a venue may report only some."""

    as_of: np.datetime64
    equity: float | None = None
    available_margin: float | None = None
    margin_ratio: float | None = None


def _recompute_avg(order: Order, fill: Deal) -> float:
    """Quantity-weighted average fill price after applying `fill`.

    Called *before* order.filled_quantity is advanced, so the old quantity is
    still the pre-fill total. Works in unsigned magnitudes: `Deal.amount` is
    signed (negative for sells), but `filled_quantity` is magnitude + `side`,
    so the fill is taken as `abs(amount)` to stay consistent.
    """
    prev_qty = order.filled_quantity
    prev_avg = order.avg_fill_price or 0.0
    fill_qty = abs(fill.amount)
    new_qty = prev_qty + fill_qty
    if new_qty == 0:
        return prev_avg
    return (prev_avg * prev_qty + fill.price * fill_qty) / new_qty


def _notional(position: Position) -> float:
    # NaN for an unmarked position -> 0.0, so one unmarked position can't poison aggregates
    n = position.notional_value
    return 0.0 if n != n else float(n)


class AccountState:
    __slots__ = (
        "exchange",
        "base_currency",
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
        "_venue_figures",
    )

    def __init__(self, exchange: str, base_currency: str, *, terminal_history_size: int = 10_000):
        self.exchange: str = exchange
        self.base_currency: str = base_currency.upper()

        # ---- primary data ------------------------------------------------
        self._active_orders: dict[str, Order] = {}  # client_id -> Order
        self._positions: dict[Instrument, Position] = {}
        self._balances: dict[str, AssetBalance] = {}  # currency -> AssetBalance

        # ---- lookup / bookkeeping indices (mutator-maintained) -----------
        # venue_id -> client_id
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
        # exchange-reported account figures; None in sim, set from venue events in live
        self._venue_figures: VenueAccountFigures | None = None

    def __repr__(self) -> str:
        return (
            f"AccountState({self.exchange}: {len(self._active_orders)} active orders, "
            f"{len(self._inflight_index)} in-flight, {len(self._positions)} positions, "
            f"{len(self._balances)} balances, {len(self._terminal_history)} retained)"
        )

    def get_orders(self) -> Mapping[str, Order]:
        return MappingProxyType(self._active_orders)

    def get_order(self, client_order_id: str) -> Order | None:
        order = self._active_orders.get(client_order_id)
        if order is not None:
            return order
        # Slow path: recently-evicted terminals (most-recent first).
        for o in reversed(self._terminal_history):
            if o.client_id == client_order_id:
                return o
        return None

    def get_active_order(self, client_order_id: str) -> Order | None:
        return self._active_orders.get(client_order_id)

    def get_order_by_venue_id(self, venue_order_id: str) -> Order | None:
        cid = self._venue_id_index.get(venue_order_id)
        return self._active_orders.get(cid) if cid is not None else None

    def get_inflight_orders(self) -> list[Order]:
        return [self._active_orders[cid] for cid in self._inflight_index]

    def get_positions(self) -> Mapping[Instrument, Position]:
        return MappingProxyType(self._positions)

    def get_position(self, instrument: Instrument) -> Position | None:
        return self._positions.get(instrument)

    def get_balances(self) -> Mapping[str, AssetBalance]:
        return MappingProxyType(self._balances)

    def get_balance(self, currency: str) -> AssetBalance | None:
        return self._balances.get(currency)

    def get_pre_pending(self, cid: str) -> OrderStatus | None:
        return self._pre_pending_status.get(cid)

    def get_retry(self, cid: str) -> int:
        return self._retry_count.get(cid, 0)

    def get_last_snapshot_as_of(self) -> np.datetime64 | None:
        return self._last_snapshot_as_of

    def get_venue_figures(self) -> VenueAccountFigures | None:
        return self._venue_figures

    # ================================================================== #
    # Derived metrics — single exchange. In live, exchange-reported      #
    # figures are preferred over the derived value, per metric.          #
    # ================================================================== #

    def total_capital(self) -> float:
        venue = self._venue_figures
        if venue is not None and venue.equity is not None:
            return venue.equity
        base = self._balances.get(self.base_currency)
        cash = base.total if base is not None else 0.0
        return cash + sum(p.market_value_funds for p in self._positions.values())

    def total_initial_margin(self) -> float:
        return sum(p.initial_margin for p in self._positions.values())

    def total_maint_margin(self) -> float:
        return sum(p.maint_margin for p in self._positions.values())

    def available_margin(self) -> float:
        venue = self._venue_figures
        if venue is not None and venue.available_margin is not None:
            return venue.available_margin
        return self.total_capital() - self.total_initial_margin()

    def margin_ratio(self) -> float:
        venue = self._venue_figures
        if venue is not None and venue.margin_ratio is not None:
            return venue.margin_ratio
        maint = self.total_maint_margin()
        return 100.0 if maint == 0 else min(100.0, self.total_capital() / maint)

    def leverage(self, instrument: Instrument) -> float:
        pos = self._positions.get(instrument)
        if pos is None:
            return 0.0
        capital = self.total_capital()
        return _notional(pos) / capital if capital > 0 else 0.0

    def net_leverage(self) -> float:
        capital = self.total_capital()
        if capital <= 0:
            return 0.0
        return sum(_notional(p) for p in self._positions.values()) / capital

    def gross_leverage(self) -> float:
        capital = self.total_capital()
        if capital <= 0:
            return 0.0
        return sum(abs(_notional(p)) for p in self._positions.values()) / capital

    def conversion_rate(self, instrument: Instrument) -> float:
        del instrument  # TODO(account-mgmt): convert settle/quote -> base_currency via marks
        return 1.0

    # ================================================================== #
    # Mutators — framework-internal; only AccountManager calls these,    #
    # on the strategy thread. No legality checks here: the transition    #
    # table is enforced upstream in AccountManager. These setters' only  #
    # job is to mutate fields and keep every index consistent.           #
    # ================================================================== #

    def _add_order(self, order: Order) -> None:
        cid = order.client_id
        self._active_orders[cid] = order
        if order.venue_id is not None:
            self._venue_id_index[order.venue_id] = cid
        if order.status in _INFLIGHT_STATUSES:
            self._inflight_index.add(cid)
        elif order.status in _TERMINAL_STATUSES:
            if order.last_updated_at is None:
                raise ValueError("terminal orders must have last_updated_at set for eviction")
            self._pending_evict_index[cid] = order.last_updated_at

    def _transition_order(self, cid: str, new_status: OrderStatus, now: np.datetime64) -> Order:
        """Low-level status setter and the sole maintainer of every status-derived
        structure: the in-flight and pending-evict indices, the retry counter, and
        the pre-pending status capture.
        """
        order = self._active_orders[cid]
        old_status = order.status
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

    def _bump_retry(self, cid: str) -> int:
        count = self._retry_count.get(cid, 0) + 1
        self._retry_count[cid] = count
        return count

    def _set_venue_id(self, cid: str, venue_order_id: str) -> None:
        order = self._active_orders[cid]
        if order.venue_id is not None:
            # drop stale key before re-pointing
            self._venue_id_index.pop(order.venue_id, None)
        order.venue_id = venue_order_id
        self._venue_id_index[venue_order_id] = cid

    def _apply_fill(self, cid: str, fill: Deal, now: np.datetime64) -> bool:
        """Apply a fill, deduped by trade id. Returns True if newly applied, False if duplicate."""
        order = self._active_orders[cid]
        seen = self._seen_trade_ids.setdefault(cid, set())
        if fill.id in seen:
            logger.debug(f"duplicate fill {fill.id} on {cid}; skipping")
            return False
        seen.add(fill.id)
        order.avg_fill_price = _recompute_avg(order, fill)
        # filled_quantity is unsigned magnitude (direction lives in order.side),
        # matching Order.quantity; Deal.amount is signed, so take its magnitude.
        order.filled_quantity += abs(fill.amount)
        order.last_updated_at = now
        return True

    def _remove_order(self, cid: str) -> None:
        """Evict a terminal order from active state into the history ring buffer,
        dropping its index entries. Called by AM's eviction sweep after the
        terminal grace period elapses.
        """
        order = self._active_orders.pop(cid, None)
        if order is None:
            return
        if order.venue_id is not None:
            self._venue_id_index.pop(order.venue_id, None)
        self._inflight_index.discard(cid)
        self._pending_evict_index.pop(cid, None)
        self._seen_trade_ids.pop(cid, None)
        self._retry_count.pop(cid, None)
        self._pre_pending_status.pop(cid, None)
        self._terminal_history.append(order)

    def _prune_terminal_orders(self, now: np.datetime64, retention: np.timedelta64) -> None:
        due = [cid for cid, terminal_at in self._pending_evict_index.items() if (now - terminal_at) >= retention]
        for cid in due:
            self._remove_order(cid)

    def _mark_snapshot_applied(self, as_of: np.datetime64) -> None:
        self._last_snapshot_as_of = as_of

    def _set_venue_figures(self, figures: VenueAccountFigures) -> None:
        self._venue_figures = figures

    def _update_balance(self, currency: str, balance: AssetBalance) -> None:
        if currency not in self._balances:
            self._balances[currency] = balance
        else:
            self._balances[currency].set(balance)

    def _update_position(self, position: Position) -> None:
        instrument = position.instrument
        if instrument not in self._positions:
            self._positions[instrument] = position
        else:
            self._positions[instrument].reset_by_position(position)

    def _ensure_position(self, instrument: Instrument) -> Position:
        pos = self._positions.get(instrument)
        if pos is None:
            pos = Position(instrument)
            self._positions[instrument] = pos
        return pos

    def _adjust_balance(self, currency: str, delta: float) -> None:
        bal = self._balances.get(currency)
        if bal is None:
            bal = AssetBalance(exchange=self.exchange, currency=currency)
            self._balances[currency] = bal
        bal.total += delta
        bal.free = bal.total - bal.locked
