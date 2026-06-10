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

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from qubx import logger
from qubx.core.basics import Balance, Deal, Instrument, Order, OrderStatus, OrderTransition, Position

# Bounded per-exchange funding-bucket dedup (insertion order ≈ funding-event time order):
# old buckets evict once the cap is hit so the set can't grow unbounded over long-running
# sessions. A re-delivered funding event only needs RECENT buckets to dedup against.
_FUNDING_BUCKET_CAP: int = 4096


@dataclass
class VenueAccountFigures:
    """Account-level figures reported by the exchange. Preferred over derived
    metrics in live; each is optional since a venue may report only some."""

    as_of: np.datetime64
    equity: float | None = None
    available_margin: float | None = None
    margin_ratio: float | None = None


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
        "_snapshot_fill_as_of",
        "_terminal_history",
        "_retry_count",
        "_pre_pending_status",
        "_last_snapshot_as_of",
        "_transition_counts",
        "_venue_figures",
        "_applied_funding_buckets",
        "_position_push_as_of",
        "_balance_push_as_of",
    )

    def __init__(self, exchange: str, base_currency: str, *, terminal_history_size: int = 10_000):
        self.exchange: str = exchange
        self.base_currency: str = base_currency.upper()

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
        # cid -> as_of of the last snapshot that raised the order's filled_quantity:
        # deals at or before it were already counted by the snapshot (order totals AND
        # position/balance legs) and must not book again. Same lifecycle as
        # _seen_trade_ids: dropped on remove/evict.
        self._snapshot_fill_as_of: dict[str, np.datetime64] = {}
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
        # exchange-reported account figures; None in sim, set from venue snapshots in live
        self._venue_figures: VenueAccountFigures | None = None
        # applied funding buckets ((instrument, bucket-index) keys), LRU-bounded dedup
        # side-table — same family as _seen_trade_ids, but keyed per funding interval
        self._applied_funding_buckets: OrderedDict[tuple, None] = OrderedDict()
        # venue event time of the last applied WS push, per instrument / per currency
        # (venue clock — same domain as Deal.time). Drive the stale-push ratchet and
        # the covered-delta guards. Instrument/currency-keyed and tiny, so retained
        # for the state's lifetime (no eviction).
        self._position_push_as_of: dict[Instrument, np.datetime64] = {}
        self._balance_push_as_of: dict[str, np.datetime64] = {}

    def __repr__(self) -> str:
        return (
            f"AccountState({self.exchange}: {len(self._active_orders)} active orders, "
            f"{len(self._inflight_index)} in-flight, {len(self._positions)} positions, "
            f"{len(self._balances)} balances, {len(self._terminal_history)} retained)"
        )

    def get_orders(self) -> dict[str, Order]:
        return dict(self._active_orders)

    def get_order(self, client_order_id: str | None) -> Order | None:
        if client_order_id is None:  # venue-id-only event; caller falls back to get_order_by_venue_id
            return None
        order = self._active_orders.get(client_order_id)
        if order is not None:
            return order
        # Slow path: recently-evicted terminals (most-recent first).
        for o in reversed(self._terminal_history):
            if o.client_order_id == client_order_id:
                return o
        return None

    def get_active_order(self, client_order_id: str | None) -> Order | None:
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

    def get_snapshot_fill_as_of(self, cid: str) -> np.datetime64 | None:
        return self._snapshot_fill_as_of.get(cid)

    def get_position_push_as_of(self, instrument: Instrument) -> np.datetime64 | None:
        return self._position_push_as_of.get(instrument)

    def get_balance_push_as_of(self, currency: str) -> np.datetime64 | None:
        return self._balance_push_as_of.get(currency)

    def get_transition_counts(self) -> dict[str, int]:
        """Audit counter snapshot: status.value -> number of transitions into it."""
        return dict(self._transition_counts)

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

    def add_order(self, order: Order) -> None:
        cid = order.client_order_id
        if cid in self._active_orders:
            # Silent overwrite orphaned the caller's Order reference and left stale
            # index entries behind; duplicates indicate a framework bug — fail loudly.
            raise ValueError(f"[{self.exchange}] duplicate active order cid {cid}; refusing to overwrite")
        self._active_orders[cid] = order
        if order.venue_order_id is not None:
            self._venue_id_index[order.venue_order_id] = cid
        if order.status.is_inflight:
            self._inflight_index.add(cid)
        elif order.status.is_terminal:
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

        if new_status.is_inflight:
            self._inflight_index.add(cid)
        else:
            self._inflight_index.discard(cid)

        if new_status.is_terminal:
            self._pending_evict_index[cid] = now
        else:
            self._pending_evict_index.pop(cid, None)

        if new_status.is_pending:
            # capture only on first entry, so PENDING_UPDATE -> PENDING_CANCEL keeps the original
            if not old_status.is_pending:
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

    def record_trade_id(self, cid: str, trade_id: str) -> None:
        """Mark a trade id as seen WITHOUT booking it — for executions a snapshot already
        counted, so a later re-delivery (DealEvent or embedded fill) dedups via apply_fill."""
        self._seen_trade_ids.setdefault(cid, set()).add(trade_id)

    def mark_snapshot_fill(self, cid: str, as_of: np.datetime64) -> None:
        self._snapshot_fill_as_of[cid] = as_of

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
        self._snapshot_fill_as_of.pop(cid, None)
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
        self._snapshot_fill_as_of.pop(cid, None)
        self._retry_count.pop(cid, None)
        self._pre_pending_status.pop(cid, None)
        self._terminal_history.append(order)

    def prune_terminal_orders(self, now: np.datetime64, retention: np.timedelta64) -> None:
        due = [cid for cid, terminal_at in self._pending_evict_index.items() if (now - terminal_at) >= retention]
        for cid in due:
            self.evict_to_history(cid)

    def mark_snapshot_applied(self, as_of: np.datetime64) -> None:
        self._last_snapshot_as_of = as_of

    def set_venue_figures(self, figures: VenueAccountFigures) -> None:
        self._venue_figures = figures

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

    def reconcile_position_from_snapshot(self, snapshot: Position) -> bool:
        """Reconcile a venue snapshot position into state. The snapshot is authoritative
        for size/avg-price and venue-reported margin/mark ONLY — locally accumulated
        accounting (r_pnl, commissions, cumulative_funding) always survives, so this
        never routes through reset_by_position. Returns True when the position is new
        or its size/avg-price changed (the caller records it in the reconcile diff);
        the margin/mark refresh is unconditional and never reported as a change.
        """
        existing = self._positions.get(snapshot.instrument)
        if existing is None:
            self._positions[snapshot.instrument] = snapshot
            return True

        changed = existing.quantity != snapshot.quantity or existing.position_avg_price != snapshot.position_avg_price
        if changed:
            existing.reconcile_size(snapshot.quantity, snapshot.position_avg_price)

        # external margins first, so the mark refresh below skips recalculating them
        if snapshot._maint_margin_external:
            existing.set_external_maint_margin(snapshot.maint_margin)
        if snapshot._initial_margin_external:
            existing.set_external_initial_margin(snapshot.initial_margin)
        if not np.isnan(snapshot.last_update_price):
            existing.update_market_price(
                snapshot.last_update_time, snapshot.last_update_price, snapshot.last_update_conversion_rate
            )
        elif changed and not np.isnan(existing.last_update_price):
            # size changed but the snapshot carried no mark: re-derive pnl/value at the held mark
            existing.update_market_price(
                existing.last_update_time, existing.last_update_price, existing.last_update_conversion_rate
            )
        return changed

    def mark_position_push(self, instrument: Instrument, as_of: np.datetime64) -> None:
        """Record the venue event time of an applied position push. Staleness checks
        live in the reducer (this setter is dumb, like mark_snapshot_fill)."""
        self._position_push_as_of[instrument] = as_of

    def apply_balance_push(
        self, currency: str, total: float, as_of: np.datetime64, *, free: float = np.nan, locked: float = np.nan
    ) -> bool:
        """Apply an absolute venue balance push (WS user-data stream).

        Per-currency ratchet: a push strictly older than the last applied one is
        dropped (returns False; same-time pushes apply — absolute overwrite is
        idempotent and a later same-ms push is the fresher figure). Futures pushes
        carry total only (free/locked NaN): total is overwritten and free moves by
        the same delta, preserving locked. When the venue reports a real split
        (free AND locked non-NaN), all three fields are overwritten. ``as_of`` is
        venue event time (Deal.time clock domain).
        """
        last = self._balance_push_as_of.get(currency)
        if last is not None and as_of < last:
            return False
        self._balance_push_as_of[currency] = as_of
        bal = self._balances.get(currency)
        if bal is None:
            bal = Balance(exchange=self.exchange, currency=currency)
            self._balances[currency] = bal
        if not np.isnan(free) and not np.isnan(locked):
            bal.free = free
            bal.locked = locked
            bal.total = total
        else:
            delta = total - bal.total
            bal.total = total
            bal.free += delta
        return True

    def apply_balance_snapshot(self, balance: Balance) -> bool:
        existing = self._balances.get(balance.currency)
        changed = existing is None or (
            existing.total != balance.total or existing.free != balance.free or existing.locked != balance.locked
        )
        self.update_balance(balance.currency, balance)
        return changed

    def ensure_position(self, instrument: Instrument) -> Position:
        pos = self._positions.get(instrument)
        if pos is None:
            pos = Position(instrument=instrument)
            self._positions[instrument] = pos
        return pos

    def adjust_balance(self, currency: str, delta: float) -> None:
        # free moves with total (free == total - locked stays invariant for any locked),
        # mutating the held Balance in place so external references stay live.
        bal = self._balances.get(currency)
        if bal is None:
            bal = Balance(exchange=self.exchange, currency=currency)
            self._balances[currency] = bal
        bal.total += delta
        bal.free += delta

    def is_funding_applied(self, bucket: tuple) -> bool:
        return bucket in self._applied_funding_buckets

    def mark_funding_applied(self, bucket: tuple) -> None:
        self._applied_funding_buckets[bucket] = None
        if len(self._applied_funding_buckets) > _FUNDING_BUCKET_CAP:
            self._applied_funding_buckets.popitem(last=False)  # evict the oldest bucket
