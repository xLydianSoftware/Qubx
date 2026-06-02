"""
``AccountManager`` — applies typed account events to ``AccountState`` through the order
state machine, and sweeps stuck in-flight orders.

It is the sole writer of ``AccountState`` and runs entirely on the dispatch thread
(connectors only *produce* events onto the Channel; see ``ACCOUNT_MANAGEMENT_PLAN.md``).
Behaviour is derived from the ``account-management`` excalidraw flows (submit / cancel /
update / stuck-order recovery); a few rules are judgement calls, flagged inline.
"""

import numpy as np

from qubx import logger
from qubx.core.account_manager.connector import IConnector
from qubx.core.account_manager.events import (
    AccountMessage,
    AccountSnapshotEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.account_manager.scheduling import _ms_to_cron
from qubx.core.account_manager.state import AccountState, ManagedOrder
from qubx.core.account_manager.state_machine import (
    PENDING_STATES,
    OrderState,
    is_pending,
    is_terminal,
    transition,
)
from qubx.core.basics import ITimeProvider, dt_64

_EPS = 1e-12


class AccountManager:
    """Event-driven owner of ``AccountState``."""

    def __init__(
        self,
        state: AccountState | None = None,
        connectors: dict[str, IConnector] | None = None,
        time_provider: ITimeProvider | None = None,
        min_inflight_age_sec: float = 5.0,
        max_inflight_retries: int = 5,
        evict_grace_sec: float = 30.0,
        tombstone_ttl_sec: float = 3600.0,
        inflight_tick_interval_ms: int = 2000,
    ):
        self.state = state if state is not None else AccountState()
        self._connectors = connectors or {}
        self._time = time_provider
        self.min_inflight_age_sec = min_inflight_age_sec
        self.max_inflight_retries = max_inflight_retries
        self.evict_grace_sec = evict_grace_sec
        self.tombstone_ttl_sec = tombstone_ttl_sec
        self.inflight_tick_interval_ms = inflight_tick_interval_ms

        self._dispatch = {
            OrderAcceptedEvent: self._on_accepted,
            OrderRejectedEvent: self._on_rejected,
            OrderFilledEvent: self._on_filled,
            OrderCanceledEvent: self._on_canceled,
            OrderCancelRejectedEvent: self._on_cancel_rejected,
            OrderUpdatedEvent: self._on_updated,
            OrderUpdateRejectedEvent: self._on_update_rejected,
            AccountSnapshotEvent: self._on_snapshot,
        }

    # -- public API ---------------------------------------------------------------------

    def get_order(self, cid: str) -> ManagedOrder | None:
        return self.state.active_orders.get(cid)

    def add_order(self, order: ManagedOrder) -> None:
        """Register a freshly-submitted order (SUBMITTED) in the cache and in-flight index."""
        now = self._now()
        if order.created_at is None:
            order.created_at = now
        order.last_updated_at = now
        order.status = OrderState.SUBMITTED
        prev = self.state.active_orders.get(order.client_id)
        if prev is not None:
            logger.warning(f"[AccountManager] add_order overwriting live order for cid={order.client_id}")
            # drop the replaced order's stale venue-id mapping so the index can't dangle
            if prev.venue_id and self.state._venue_id_index.get(prev.venue_id) == order.client_id:
                del self.state._venue_id_index[prev.venue_id]
        self.state.active_orders[order.client_id] = order
        self.state._inflight_index.add(order.client_id)

    def apply(self, event: AccountMessage) -> None:
        """Dispatch an account event to its handler. Unknown event types are ignored."""
        handler = self._dispatch.get(type(event))
        if handler is None:
            logger.debug(f"[AccountManager] ignoring unhandled event type {type(event).__name__}")
            return
        try:
            handler(event)
        except Exception as e:
            # one malformed event/ordering must not take down the dispatch loop
            cid = getattr(event, "client_id", "?")
            logger.warning(f"[AccountManager] dropping {type(event).__name__} for cid={cid}: {e!r}")

    def transition_order(self, cid: str, target: OrderState) -> None:
        """
        Move an order into a PENDING_* state on a strategy-issued cancel/update, capturing
        ``pre_pending_status`` so a venue rejection can revert it.
        """
        o = self.get_order(cid)
        if o is None:
            return
        if target in PENDING_STATES and not is_pending(o.status):
            o.pre_pending_status = o.status
        self._set_status(o, target)

    @property
    def should_register_inflight_tick(self) -> bool:
        """Whether the wiring layer should arm the stuck-order sweep on the scheduler."""
        return True

    def inflight_tick_cron(self) -> str:
        """Cron expression for the in-flight sweep, for ``scheduler.schedule_event``."""
        return _ms_to_cron(self.inflight_tick_interval_ms)

    def on_inflight_tick(self, ctx=None) -> None:
        """Scheduler callback (registered via ``_ms_to_cron`` in live)."""
        self._sweep_stuck_inflight()

    def evict_terminal(self, now: dt_64 | None = None) -> None:
        """
        Drop terminal orders whose grace window has elapsed: the heavy order object is
        removed from ``active_orders`` (and the venue index), but the cid is kept as a
        tombstone in ``_pending_evict_index`` so late events / snapshots can't resurrect it.
        """
        now = now if now is not None else self._now()
        if now is None:
            return
        ttl = np.timedelta64(int(self.tombstone_ttl_sec * 1e9), "ns")
        for cid, evict_at in list(self.state._pending_evict_index.items()):
            if evict_at is None or evict_at > now:
                continue
            # past the grace window: drop the heavy order object, keep the cid as a tombstone
            o = self.state.active_orders.pop(cid, None)
            if o is not None and o.venue_id and self.state._venue_id_index.get(o.venue_id) == cid:
                del self.state._venue_id_index[o.venue_id]
            self.state._inflight_index.discard(cid)
            # past the tombstone TTL: purge the tombstone too, so the index can't grow unbounded
            if now - evict_at > ttl:
                del self.state._pending_evict_index[cid]

    # -- event handlers -----------------------------------------------------------------

    def _on_accepted(self, ev: OrderAcceptedEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None or is_terminal(o.status):
            return
        if ev.venue_id:
            self._set_venue_id(o, ev.venue_id)
        if o.status in (OrderState.SUBMITTED, OrderState.STALE):
            # a genuine ack of a fresh submit, or a late ack that resurrects a quarantined order
            o.retry_count = 0
            self._set_status(o, OrderState.ACCEPTED)
        # For a PENDING_* order an "accepted/open" status response means the cancel/update
        # hasn't taken effect; do NOT reset retry_count here (that would livelock the sweep),
        # let retries exhaust and the give-up path revert it.

    def _on_rejected(self, ev: OrderRejectedEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None or is_terminal(o.status):
            return
        self._set_status(o, OrderState.REJECTED)
        o.pre_pending_status = None

    def _on_filled(self, ev: OrderFilledEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None or ev.fill is None:
            return
        if is_terminal(o.status):
            # late fill racing a terminal (e.g. fill after CANCELED) — ignore, don't resurrect.
            # NOTE: a fill after CANCELED also signals the cancel was wrong; deeper reconcile is a follow-up.
            logger.debug(f"[AccountManager] ignoring fill {ev.fill.id} on terminal order cid={o.client_id}")
            return
        tid = ev.fill.id
        if tid in o.fill_trade_ids:  # dedup replayed fills by trade id
            return
        o.fill_trade_ids.add(tid)
        o.filled_quantity += abs(ev.fill.amount)
        target = (
            OrderState.FILLED
            if o.filled_quantity >= abs(o.quantity) - _EPS
            else OrderState.PARTIALLY_FILLED
        )
        self._set_status(o, target)
        if is_terminal(target):
            o.pre_pending_status = None

    def _on_canceled(self, ev: OrderCanceledEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None or is_terminal(o.status):
            return
        self._set_status(o, OrderState.CANCELED)
        o.pre_pending_status = None

    def _on_cancel_rejected(self, ev: OrderCancelRejectedEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None or o.status != OrderState.PENDING_CANCEL:
            return
        self._revert_pending(o)

    def _on_updated(self, ev: OrderUpdatedEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None:
            return
        if o.status != OrderState.PENDING_UPDATE:
            # unsolicited / duplicate update confirm — ignore rather than mutate a non-updating order
            logger.debug(f"[AccountManager] ignoring update for cid={o.client_id} in status {o.status.value}")
            return
        if ev.venue_id and ev.venue_id != o.venue_id:  # replace-style modify re-keys venue id
            self._set_venue_id(o, ev.venue_id)
        if ev.price is not None:
            o.price = ev.price
        if ev.quantity is not None:
            o.quantity = ev.quantity
        # re-evaluate completeness: a quantity shrink to the filled size finalizes the order
        if o.filled_quantity > _EPS and o.filled_quantity >= abs(o.quantity) - _EPS:
            target = OrderState.FILLED
        elif o.filled_quantity > _EPS:
            target = OrderState.PARTIALLY_FILLED
        else:
            target = OrderState.ACCEPTED
        self._set_status(o, target)
        o.pre_pending_status = None

    def _on_update_rejected(self, ev: OrderUpdateRejectedEvent) -> None:
        o = self.get_order(ev.client_id)
        if o is None or o.status != OrderState.PENDING_UPDATE:
            return
        self._revert_pending(o)

    def _on_snapshot(self, ev: AccountSnapshotEvent) -> None:
        self._reconcile_snapshot(ev)

    # -- stuck-order sweep --------------------------------------------------------------

    def _sweep_stuck_inflight(self, now: dt_64 | None = None) -> None:
        now = now if now is not None else self._now()
        if now is None:
            return
        for cid in list(self.state._inflight_index):
            o = self.state.active_orders.get(cid)
            if o is None or o.last_updated_at is None:
                continue
            try:
                age = (now - o.last_updated_at) / np.timedelta64(1, "s")
                if age < self.min_inflight_age_sec:
                    continue  # too young to be considered stuck
                if o.retry_count >= self.max_inflight_retries:
                    self._give_up_stuck_order(o)
                    continue
                conn = self._connectors.get(o.instrument.exchange)
                if conn is not None:
                    conn.request_order_status(o.client_id, o.venue_id)
                o.retry_count += 1
            except Exception as e:  # one bad order must not abort the whole sweep
                logger.warning(f"[AccountManager] sweep error for cid={cid}: {e}")

    def _give_up_stuck_order(self, o: ManagedOrder) -> None:
        """No venue ack after N status queries: quarantine a never-confirmed submit; revert a stuck pending."""
        if o.status == OrderState.SUBMITTED:
            # never confirmed — but it may be live at the venue, so DO NOT auto-reject.
            # Quarantine in STALE and wait for an authoritative signal (late event / snapshot).
            self._set_status(o, OrderState.STALE)
            o.pre_pending_status = None
        else:
            # PENDING_CANCEL / PENDING_UPDATE: the cancel/modify never acked, but the underlying
            # order is presumed still live at the venue -> revert to its pre-pending status
            self._revert_pending(o)
            o.retry_count = 0

    # -- snapshot reconcile (minimal; grace + freshness only) ---------------------------

    def _reconcile_snapshot(self, ev: AccountSnapshotEvent) -> None:
        snap_ts = ev.timestamp
        for snap_o in ev.orders:
            if not isinstance(snap_o, ManagedOrder):
                logger.warning(f"[AccountManager] snapshot order of unexpected type {type(snap_o).__name__}; skipping")
                continue
            cid = snap_o.client_id
            existing = self.state.active_orders.get(cid)
            # grace window: do not resurrect an order we just terminalized
            if existing is None and cid in self.state._pending_evict_index:
                continue
            if existing is None:
                self.state.active_orders[cid] = snap_o
                if snap_o.venue_id:
                    self.state._venue_id_index[snap_o.venue_id] = cid
                # an adopted order awaiting a venue verdict must be sweep-eligible
                if snap_o.status == OrderState.SUBMITTED or snap_o.status in PENDING_STATES:
                    self.state._inflight_index.add(cid)
                else:
                    self.state._inflight_index.discard(cid)
                continue
            # freshness: a locally newer update wins over a stale snapshot
            if (
                snap_ts is not None
                and existing.last_updated_at is not None
                and existing.last_updated_at > snap_ts
            ):
                continue
            # (further field-level reconcile is a follow-up; minimal merge keeps venue id)
            if snap_o.venue_id and existing.venue_id is None:
                self._set_venue_id(existing, snap_o.venue_id)

    # -- internals ----------------------------------------------------------------------

    def _now(self) -> dt_64 | None:
        return self._time.time() if self._time is not None else None

    def _set_venue_id(self, o: ManagedOrder, venue_id: str) -> None:
        if o.venue_id and o.venue_id in self.state._venue_id_index:
            del self.state._venue_id_index[o.venue_id]
        o.venue_id = venue_id
        self.state._venue_id_index[venue_id] = o.client_id

    def _revert_pending(self, o: ManagedOrder) -> None:
        target = o.pre_pending_status if o.pre_pending_status is not None else self._previous_status_before_pending(o)
        self._set_status(o, target)
        o.pre_pending_status = None

    def _previous_status_before_pending(self, o: ManagedOrder) -> OrderState:
        """Fallback revert target when ``pre_pending_status`` is unavailable (e.g. post-snapshot)."""
        if o.filled_quantity > _EPS:
            return OrderState.PARTIALLY_FILLED
        if o.venue_id is None:
            return OrderState.SUBMITTED
        return OrderState.ACCEPTED

    def _set_status(self, o: ManagedOrder, new_status: OrderState) -> None:
        if new_status != o.status:
            transition(o.status, new_status)  # validate; raises IllegalOrderTransition
            o.status = new_status
        o.last_updated_at = self._now()
        cid = o.client_id
        if new_status == OrderState.SUBMITTED or new_status in PENDING_STATES:
            self.state._inflight_index.add(cid)
        else:
            self.state._inflight_index.discard(cid)
        if is_terminal(new_status):
            now = self._now()
            self.state._pending_evict_index[cid] = (
                now + np.timedelta64(int(self.evict_grace_sec * 1e9), "ns") if now is not None else now
            )


class SimulationAccountManager(AccountManager):
    """
    Backtest variant of ``AccountManager``.

    The ``SimulatedConnector`` acks/fills deterministically, so there are never genuinely
    stuck in-flight orders — the wall-clock sweep is therefore **opt-in** (off by default)
    to avoid overhead and nondeterminism. The Q3 stuck-order-recovery conformance test sets
    ``register_inflight_tick=True`` and drives the sweep via the ``SimulatedScheduler``.
    See ``ACCOUNT_MANAGEMENT_PLAN.md``.
    """

    def __init__(self, *args, register_inflight_tick: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_inflight_tick = register_inflight_tick

    @property
    def should_register_inflight_tick(self) -> bool:
        return self._register_inflight_tick
