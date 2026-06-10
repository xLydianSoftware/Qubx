"""Reconciliation logic: snapshot reconcile + the periodic-sweep decision rules.

The reconciler splits along the logic-vs-orchestration line: this module owns the
LOGIC — ``reconcile_snapshot`` (the as_of ratchet, grace-window detection of orders
missing from the snapshot, RECOVERED/EXTERNAL materialization, per-order
freshness-guarded updates, position/balance application, venue figures) and the
sweep decision helpers (stuck-inflight selection, retry give-up, snapshot freshness,
liveness). AccountManager drives the ticks and makes every connector call; the
reducer routes the snapshot event here and surfaces the returned ``ReconcileDiff``
on its ApplyResult. Orders missing from the snapshot past grace are REPORTED on the
diff, not terminalized: the order may have FILLED during a WS gap, so the manager
fetches its true status first and terminalizes (``terminalize_missing``) only once
the per-order fetch budget is exhausted.

Import rule: this module must never import the reducer (or the manager) — that is
the cycle-relevant rule; state, state_machine and the leaf event/basics modules are
fine. The validate+transition helper (``transition``) lives HERE for that reason:
the reducer and the manager import it from this module, never the other way around.
"""

from dataclasses import dataclass, field

import numpy as np

from qubx import logger
from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.account_manager.state_machine import can_transition, validate_transition
from qubx.core.basics import (
    EXTERNAL_CID_PREFIX,
    Balance,
    Order,
    OrderOrigin,
    OrderStatus,
    Position,
    classify_origin,
)
from qubx.core.events import (
    AccountSnapshot,
    OrderCancelRejectedEvent,
    OrderEvent,
    OrderRejectedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import InvalidOrderTransition


@dataclass
class ReconcileDiff:
    """What one applied snapshot reconcile actually changed.

    Surfaced on ``ApplyResult.reconcile_diff`` (None when the event wasn't a snapshot
    or the snapshot was rejected as stale). Replaces the old debug-log tally with a
    real record of the affected objects. ``missing`` holds orders absent from the
    snapshot past grace that still have fetch budget left — the manager requests
    their true status; once the budget is exhausted they are terminalized instead
    and moved to ``terminated``.
    """

    missing: list[Order] = field(default_factory=list)
    terminated: list[Order] = field(default_factory=list)
    materialized: list[Order] = field(default_factory=list)
    updated: list[Order] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    balances: list[Balance] = field(default_factory=list)
    venue_figures: VenueAccountFigures | None = None


def transition(state: AccountState, cid: str, new_status: OrderStatus, now: np.datetime64) -> Order:
    """Validate-then-apply for an active order's status — the single legality chokepoint
    (the reducer and the manager delegate here; see the import rule above)."""
    order = state.get_active_order(cid)
    if order is None:
        raise KeyError(f"order {cid} not found in {state.exchange}")
    validate_transition(cid, order.status, new_status)
    return state.transition_order(cid, new_status, now)


def is_snapshot_stale(state: AccountState, as_of: np.datetime64) -> bool:
    """The out-of-order snapshot ratchet: anything at or before the last applied as_of
    is rejected wholesale. The rule lives here (not on AccountState) — state only
    stores the ratchet value."""
    last = state.get_last_snapshot_as_of()
    return last is not None and as_of <= last


def _last_seen_at(order: Order) -> np.datetime64 | None:
    # Age base for grace/staleness decisions: the freshest of last_updated_at and
    # submitted_at (both are dt_64 | None). None means we cannot age the order at all.
    return order.last_updated_at or order.submitted_at


def reconcile_snapshot(
    state: AccountState, snapshot: AccountSnapshot, now: np.datetime64, grace: np.timedelta64
) -> ReconcileDiff | None:
    """Reconcile one venue snapshot into ``state``; returns what changed, or None when
    the snapshot is rejected by the as_of ratchet.

    Reconcile mutates state silently; the strategy is notified once via
    on_account_update (the PM routes the AccountSnapshotEvent there) rather than per
    applied change.
    """
    if is_snapshot_stale(state, snapshot.as_of):
        return None
    state.mark_snapshot_applied(snapshot.as_of)

    diff = ReconcileDiff()

    if snapshot.open_orders is not None:
        snap_by_vid = {o.venue_order_id: o for o in snapshot.open_orders if o.venue_order_id}
        snap_cids = {o.client_order_id for o in snapshot.open_orders if o.client_order_id}
        for cid, cached in state.get_orders().items():
            if cached.status.is_terminal:
                continue
            vid = cached.venue_order_id
            if (vid is not None and vid in snap_by_vid) or cid in snap_cids:
                # Still open at the venue — property drift is reconciled in the
                # open-orders loop below (_update_from_snapshot), not here. The cid
                # fallback covers an unacked framework order (lost create ack, no
                # venue id yet) that the snapshot reports under our own cid.
                continue
            seen_at = _last_seen_at(cached)
            if seen_at is None:
                # No timestamps at all: we cannot age the order, so treat it as
                # just-seen and skip — terminalizing it would risk killing a
                # just-submitted order racing the snapshot.
                continue
            if (snapshot.as_of - seen_at) < grace:
                continue
            # Missing past grace: report, don't terminalize — the order may have FILLED
            # during a WS gap and a blind CANCELED would lose the execution forever. The
            # manager fetches the true status (the fetched FILLED/CANCELED replays through
            # the normal event path) and falls back to terminalize_missing on give-up.
            diff.missing.append(cached)
        for snap_order in snapshot.open_orders:
            existing = state.get_order_by_venue_id(snap_order.venue_order_id) if snap_order.venue_order_id else None
            if existing is None and snap_order.client_order_id:
                # cid fallback: an unacked framework order has venue_order_id=None and
                # can never match by venue id — capture the id the snapshot carries and
                # update in place instead of materializing a RECOVERED twin.
                existing = state.get_active_order(snap_order.client_order_id)
                if existing is not None and snap_order.venue_order_id:
                    if existing.venue_order_id != snap_order.venue_order_id:
                        state.set_venue_id(existing.client_order_id, snap_order.venue_order_id)
            if existing is None:
                diff.materialized.append(_materialize_from_snapshot(state, snap_order, snapshot.as_of))
            elif existing.last_updated_at is None or snapshot.as_of > existing.last_updated_at:
                _update_from_snapshot(state, existing, snap_order, snapshot.as_of, now)
                diff.updated.append(existing)

    # Positions and balances: the snapshot is the venue's authoritative truth for
    # size/amount, and stale snapshots are already rejected wholesale by the as_of
    # ratchet above. Positions reconcile surgically (size/avg-price/margin/mark only —
    # locally accumulated r_pnl/commissions/funding always survive); balances overwrite.
    # No per-record freshness here (unlike orders, where it guards a fresh fill):
    # Position/Balance carry no reliable last-update timestamp yet.
    # TODO(account-mgmt): once WS PositionUpdate/BalanceUpdate events are wired,
    # add per-record freshness backed by a real timestamp so a snapshot older than
    # a recent WS update can't clobber it.
    if snapshot.positions is not None:
        for snap_pos in snapshot.positions:
            if state.reconcile_position_from_snapshot(snap_pos):
                diff.positions.append(snap_pos)

    if snapshot.balances is not None:
        for snap_bal in snapshot.balances:
            if state.apply_balance_snapshot(snap_bal):
                diff.balances.append(snap_bal)

    # Venue-reported account figures: prefer-venue-else-derive happens per metric in
    # AccountState. A snapshot with no figures (sim, or a failed balance leg) keeps the
    # previous capture rather than clearing — absence means "not observed", not "gone".
    if snapshot.equity is not None or snapshot.available_margin is not None or snapshot.margin_ratio is not None:
        figures = VenueAccountFigures(
            as_of=snapshot.as_of,
            equity=snapshot.equity,
            available_margin=snapshot.available_margin,
            margin_ratio=snapshot.margin_ratio,
        )
        state.set_venue_figures(figures)
        diff.venue_figures = figures

    # No summary log here: the manager logs the operator-facing INFO/WARNING line after
    # resolving the missing list (only then is diff.terminated known).
    return diff


def _materialize_from_snapshot(state: AccountState, snap_order: Order, as_of: np.datetime64) -> Order:
    # cid prefix classifies origin: our prefix → a recovered framework order;
    # anything else → external. Keep an already-synthesized ext: cid as-is,
    # otherwise synthesize one from the venue id.
    origin = classify_origin(snap_order.client_order_id)
    if origin is OrderOrigin.RECOVERED or snap_order.client_order_id.startswith(EXTERNAL_CID_PREFIX):
        cid = snap_order.client_order_id
    else:
        cid = f"{EXTERNAL_CID_PREFIX}{snap_order.venue_order_id}"
    order = Order(
        client_order_id=cid,
        venue_order_id=snap_order.venue_order_id,
        origin=origin,
        type=snap_order.type,
        instrument=snap_order.instrument,
        submitted_at=snap_order.submitted_at,
        quantity=snap_order.quantity,
        price=snap_order.price,
        side=snap_order.side,
        status=snap_order.status,
        time_in_force=snap_order.time_in_force,
        filled_quantity=snap_order.filled_quantity,
        avg_fill_price=snap_order.avg_fill_price,
        last_updated_at=as_of,
    )
    state.add_order(order)
    if order.filled_quantity > 0.0:
        # The snapshot's filled_quantity (and its position/balance legs) already count
        # executions up to as_of — a late DealEvent for one of them must not book again.
        state.mark_snapshot_fill(order.client_order_id, as_of)
    return order


def _update_from_snapshot(
    state: AccountState, existing: Order, snap_order: Order, as_of: np.datetime64, now: np.datetime64
) -> None:
    if snap_order.status != existing.status:
        _reconcile_status_from_snapshot(state, existing, snap_order.status, now)
    if snap_order.filled_quantity > existing.filled_quantity:
        # The snapshot counted executions we haven't seen as deals yet (and its
        # position/balance legs incorporate them) — remember as_of so a late DealEvent
        # at or before it isn't booked twice (_handle_deal checks this).
        state.mark_snapshot_fill(existing.client_order_id, as_of)
    existing.filled_quantity = snap_order.filled_quantity
    existing.avg_fill_price = snap_order.avg_fill_price
    existing.price = snap_order.price
    existing.quantity = snap_order.quantity
    existing.last_updated_at = as_of


def _reconcile_status_from_snapshot(
    state: AccountState, existing: Order, venue_status: OrderStatus, now: np.datetime64
) -> None:
    # Status reconciliation goes through state.transition_order — the sole maintainer of
    # the transitions audit, the counters and the inflight/pending-evict indices — never
    # a bare ``existing.status =`` write, which left snapshot-terminalized orders as
    # permanent hidden residents of active state with a stale inflight entry.
    cid = existing.client_order_id
    if existing.status.is_pending and not venue_status.is_terminal:
        # The snapshot is a poll of venue state: our cancel/update request may still be
        # in flight, so a live venue status must not wipe the pending marker (same
        # rationale as the accept-during-PENDING_CANCEL guard). The venue resolves the
        # race itself — and a terminal status IS that resolution, so it falls through.
        return
    if not can_transition(existing.status, venue_status):
        # Venue-authoritative weirdness (e.g. resurrecting a locally-terminal order, or
        # PARTIALLY_FILLED back to ACCEPTED): the snapshot wins, but loudly — and still
        # via transition_order so the audit and indices stay consistent.
        logger.warning(
            f"[{state.exchange}] reconcile: forcing illegal transition {cid}: "
            f"{existing.status} -> {venue_status} (snapshot is authoritative)"
        )
    state.transition_order(cid, venue_status, now)


def terminalize_missing(state: AccountState, order: Order, now: np.datetime64) -> bool:
    """Give-up terminalization for an order missing from snapshots past grace whose
    status-fetch budget is exhausted: REJECTED if the venue never acked it, else
    CANCELED. Returns True when the transition applied."""
    cid = order.client_order_id
    terminal = OrderStatus.REJECTED if order.status == OrderStatus.SUBMITTED else OrderStatus.CANCELED
    order.rejected_reason = "reconcile: missing from snapshot"
    try:
        transition(state, cid, terminal, now)
        return True
    except InvalidOrderTransition:
        logger.warning(f"reconcile: cannot terminate {cid} from {order.status}")
        return False


# ---- sweep decision helpers (the manager's ticks call these, then act) -------- #


def select_overdue_inflight(state: AccountState, now: np.datetime64, threshold: np.timedelta64) -> list[Order]:
    """In-flight orders not heard from for at least ``threshold`` — candidates for a
    status poll or give-up. An order with neither last_updated_at nor submitted_at is
    treated as just-seen and skipped (it cannot be aged)."""
    overdue = []
    for order in state.get_inflight_orders():
        seen_at = _last_seen_at(order)
        if seen_at is None:
            continue
        if (now - seen_at) >= threshold:
            overdue.append(order)
    return overdue


def retries_exhausted(state: AccountState, cid: str, max_retries: int) -> bool:
    return state.get_retry(cid) >= max_retries


def giveup_event(order: Order, retries: int) -> OrderEvent:
    """The synthetic reject for an in-flight order whose retry budget is exhausted —
    the venue never acked, so we synthesize the reject it never sent. The manager
    routes it through pm.process_event (the same path venue events take), so the
    normal handlers do the transition/revert and the PM fires the strategy callback
    error-isolated."""
    reason = f"reconcile: no venue ack after {retries} retries"
    if order.status == OrderStatus.SUBMITTED:
        return OrderRejectedEvent(
            instrument=order.instrument,
            client_order_id=order.client_order_id,
            venue_order_id=order.venue_order_id,
            reason=reason,
        )
    if order.status == OrderStatus.PENDING_CANCEL:
        return OrderCancelRejectedEvent(
            instrument=order.instrument,
            client_order_id=order.client_order_id,
            venue_order_id=order.venue_order_id,
            reason=reason,
        )
    return OrderUpdateRejectedEvent(
        instrument=order.instrument,
        client_order_id=order.client_order_id,
        venue_order_id=order.venue_order_id,
        reason=reason,
    )


def snapshot_due(state: AccountState, now: np.datetime64, interval: np.timedelta64) -> bool:
    last = state.get_last_snapshot_as_of()
    return last is None or (now - last) > interval


def liveness_overdue(unready_since: np.datetime64, now: np.datetime64, threshold: np.timedelta64) -> bool:
    return (now - unready_since) >= threshold
