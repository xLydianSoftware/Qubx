"""Reconciler — stage 2: resolve live-trading discrepancies over time.

Owns the domain logic — the `Differ`, the snapshot due-timer, the diff→task mapping — and a
small task registry (opaque `key -> task`, one per key) it routes inputs through. Its entry
points (`on_tick` / `on_snapshot` / `on_event`) return a `list[Action]` describing the I/O the
driver (AccountManager0) must perform, and may mutate the in-memory `AccountState` they
are handed — but never do I/O themselves. So every live scenario is a mock-free data test.

See docs/account-management/reconciliation-redesign.md (+ .canvas).
"""

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np

from qubx import logger
from qubx.core.account_manager.diffs import (
    BalanceMismatch,
    Differ,
    LocalOrderMissing,
    LocalPositionMissing,
    OrderFieldMismatch,
    OriginalBalanceMissing,
    OriginalPositionMissing,
    PositionFieldMismatch,
    PositionSizeMismatch,
)
from qubx.core.account_manager.state import AccountState
from qubx.core.account_manager.state_machine import can_transition
from qubx.core.basics import Instrument, Order, OrderStatus, Position
from qubx.core.events import (
    AccountSnapshot,
    DealEvent,
    OrderLostEvent,
    OrderPartiallyFilledEvent,
)
from qubx.utils.time import to_timedelta


@dataclass(frozen=True)
class RequestStatus:
    # - fetch one order's true status from the venue (missing/uncertain order)
    cid: str
    venue_id: str | None
    instrument: Instrument


@dataclass(frozen=True)
class RequestSnapshot:
    # - pull a fresh account snapshot for this exchange
    exchange: str


@dataclass(frozen=True)
class RequestHistDeals:
    # - fetch trades since `since` to recover deals missed behind a position diff
    instrument: Instrument
    since: np.datetime64


@dataclass(frozen=True)
class RouteEvent:
    # - a synthesized event AM feeds to pm.process_event (reducer applies + strategy notified +
    #   later WS dup deduped). For decisions with no venue event of their own, e.g. give-up LOST.
    event: object


Action = RequestStatus | RequestSnapshot | RequestHistDeals | RouteEvent | OrderPartiallyFilledEvent


class Tick:
    # - the periodic clock input that drives task timers
    __slots__ = ()


@dataclass(frozen=True)
class OrderIn:
    # - an order-lifecycle event routed to tasks (OrderCanceledEvent, OrderFilledEvent, ...)
    event: object


@dataclass(frozen=True)
class DealIn:
    # - a trade (DealEvent) routed to tasks
    deal: DealEvent


@dataclass(frozen=True)
class SnapshotIn:
    # - a snapshot arrival routed to tasks (e.g. order reappeared)
    snap: AccountSnapshot


def _td(v: str | np.timedelta64) -> np.timedelta64:
    return to_timedelta(v).asm8 if isinstance(v, str) else v


class Task(ABC):
    """A pure FSM step. Carries an opaque ``key`` the Reconciler dedups & routes by; mutates
    the in-memory ``state`` it is handed and returns the actions it wants performed (no I/O).
    """

    key: Hashable

    @abstractmethod
    def handles(self, inp: Any) -> bool: ...

    @abstractmethod
    def step(self, inp: Any, state: AccountState, now: np.datetime64) -> list[Action]: ...

    @abstractmethod
    def done(self) -> bool: ...


class ResolveMissingOrder(Task):
    """A local order absent from the snapshot. Wait, then fetch its status (≤ max_retries),
    until an event resolves it or the budget runs out → LOST. The arriving status is applied
    by the normal event path; this task only nudges the venue and counts retries.
    """

    def __init__(self, order: Order, now: np.datetime64, *, wait: np.timedelta64, max_retries: int):
        self.key = order.client_order_id
        self._cid = order.client_order_id
        self._venue_id = order.venue_order_id
        self._instrument = order.instrument
        self._wait = wait
        self._max_retries = max_retries
        self._retries = 0
        self._next_fetch_at = now + wait
        self._done = False

    def done(self) -> bool:
        return self._done

    def handles(self, inp: Any) -> bool:
        if isinstance(inp, (Tick, SnapshotIn)):
            return True
        if isinstance(inp, OrderIn):
            return self._matches(inp.event)
        return False

    def step(self, inp: Any, state: AccountState, now: np.datetime64) -> list[Action]:
        match inp:
            case OrderIn():
                self._done = True  # - any event for our id resolves it (normal path applies it)
                return []
            case SnapshotIn(snap=snap):
                # - reappeared open, or just applied terminal → no longer missing
                order = state.get_order(self._cid)
                self._done = (order is not None and order.status.is_terminal) or self._present_in(snap)
                return []
            case Tick():
                return self._on_tick(state, now)
            case _:
                return []

    def _on_tick(self, state: AccountState, now: np.datetime64) -> list[Action]:
        if now < self._next_fetch_at:
            return []
        if self._retries < self._max_retries:
            self._retries += 1
            self._next_fetch_at = now + self._wait
            return [RequestStatus(cid=self._cid, venue_id=self._venue_id, instrument=self._instrument)]
        # - budget exhausted → route LOST via the bus (never silent: a silent mutate is invisible
        #   and the later WS dup gets deduped, so the strategy would miss it)
        logger.warning(
            f"[{state.exchange}] reconcile give-up on order <g>{self._cid}</g> (vid=<blue>{self._venue_id}</blue>) "
            f"-> <r>LOST</r> after {self._retries} status fetches with no venue answer"
        )
        self._done = True
        return [
            RouteEvent(
                OrderLostEvent(
                    instrument=self._instrument,
                    client_order_id=self._cid,
                    venue_order_id=self._venue_id,
                    reason=f"reconcile give-up after {self._retries} status fetches",
                )
            )
        ]

    def _matches(self, event: object) -> bool:
        return getattr(event, "client_order_id", None) == self._cid or (
            self._venue_id is not None and getattr(event, "venue_order_id", None) == self._venue_id
        )

    def _present_in(self, snap: AccountSnapshot) -> bool:
        for o in snap.open_orders or []:
            if o.client_order_id == self._cid or (self._venue_id is not None and o.venue_order_id == self._venue_id):
                return True
        return False


class ConfirmPositionBySnapshot(Task):
    """Recover the deals behind a position-size diff (snapshot already fixed the size).

    Wait the window for late WS deals, crediting each against the missed delta. Fully covered →
    drop, no fetch. Window elapses with any remainder → one RequestHistDeals, then drop.
    """

    def __init__(
        self,
        instrument: Instrument,
        since: np.datetime64,
        now: np.datetime64,
        *,
        wait: np.timedelta64,
        expected_delta: float,
    ):
        self.key = instrument.symbol
        self._instrument = instrument
        self._since = since
        self._deadline = now + wait
        self._remaining = expected_delta  # - signed size still to recover (snapshot - prior)
        self._eps = instrument.lot_size * 0.5  # - half-lot, matches the Differ
        self._done = False

    def done(self) -> bool:
        return self._done

    def handles(self, inp: Any) -> bool:
        if isinstance(inp, Tick):
            return True
        if isinstance(inp, DealIn):
            return getattr(inp.deal, "instrument", None) == self._instrument
        return False

    def step(self, inp: Any, state: AccountState, now: np.datetime64) -> list[Action]:
        if isinstance(inp, DealIn):
            # - signed credit, so an opposite-side trade doesn't falsely cover
            self._remaining -= inp.deal.deal.amount
            self._done = abs(self._remaining) <= self._eps
            return []
        if isinstance(inp, Tick) and now >= self._deadline:
            self._done = True  # - hard deadline: exactly one fetch then drop
            return [RequestHistDeals(instrument=self._instrument, since=self._since)]
        return []


class Reconciler:
    _differ: Differ
    _snapshot_interval: np.timedelta64  # - reconciler owns the due-timer
    _missing_wait: np.timedelta64  # - knobs handed to ResolveMissingOrder tasks
    _missing_max_retries: int
    _position_confirm_wait: np.timedelta64  # - ConfirmPositionBySnapshot window
    _tasks: dict[Hashable, Task]  # - opaque key -> task, one per key
    _last_snapshot: dict[str, np.datetime64]  # - per-exchange last snapshot time

    def __init__(
        self,
        differ: Differ,
        *,
        snapshot_interval: str | np.timedelta64 = "30s",
        missing_wait: str | np.timedelta64 = "2s",
        missing_max_retries: int = 3,
        position_confirm_wait: str | np.timedelta64 = "2s",
    ):
        self._differ = differ
        self._snapshot_interval = _td(snapshot_interval)
        self._missing_wait = _td(missing_wait)
        self._missing_max_retries = missing_max_retries
        self._position_confirm_wait = _td(position_confirm_wait)
        self._tasks = {}
        self._last_snapshot = {}

    def active_keys(self) -> set[Hashable]:
        return set(self._tasks)

    def on_tick(self, state: AccountState, now: np.datetime64) -> list[Action]:
        actions: list[Action] = []
        if self._snapshot_due(state.exchange, now):
            self._last_snapshot[state.exchange] = now
            logger.debug(f"[{state.exchange}] reconcile: snapshot due -> RequestSnapshot")
            actions.append(RequestSnapshot(state.exchange))
        return actions + self._dispatch(Tick(), state, now)

    def on_snapshot(self, state: AccountState, snap: AccountSnapshot, now: np.datetime64) -> list[Action]:
        # - applies are idempotent, so repeated field atoms for one order/position are safe.
        #   leaf arms precede their base arm (a leaf must win). unhandled atoms are deferred.
        self._last_snapshot[state.exchange] = now  # - reset the snapshot due-timer
        actions: list[Action] = []
        for difference in self._differ.diff(state, snap):
            logger.debug(difference.describe())
            match difference:
                case LocalOrderMissing(order=order):
                    self._spawn(
                        ResolveMissingOrder(order, now, wait=self._missing_wait, max_retries=self._missing_max_retries)
                    )

                case OrderFieldMismatch(origin=snap_order):
                    if (ev := self._reconcile_order(state, snap_order)) is not None:
                        actions.append(RouteEvent(ev))

                # - size diff = missed deals
                case PositionSizeMismatch(origin=snap_pos) | OriginalPositionMissing(position=snap_pos):
                    self._reconcile_missed_position(state, snap, now, snap_pos)

                # - avg/margin only = figure refresh, no missed deals
                case PositionFieldMismatch(origin=snap_pos):
                    state.reconcile_position_from_snapshot(snap_pos)

                # - local holds it, venue flat = missed the close
                case LocalPositionMissing(position=local_pos):
                    self._flatten_missed_close(state, snap, now, local_pos.instrument)

                case BalanceMismatch(origin=snap_bal) | OriginalBalanceMissing(balance=snap_bal):
                    state.apply_balance_snapshot(snap_bal, snap.as_of)

        return actions + self._dispatch(SnapshotIn(snap), state, now)

    def _reconcile_missed_position(
        self, state: AccountState, snap: AccountSnapshot, now: np.datetime64, snap_pos: Position
    ) -> None:
        # - apply the snapshot size, watermark it (reducer won't re-book deals it already covers),
        #   then confirm-task the missed delta (snapshot - prior, captured before the apply).
        prior = state.get_position(snap_pos.instrument)
        prior_qty = prior.quantity if prior is not None else 0.0
        delta = snap_pos.quantity - prior_qty
        # - flip: the missed deals crossed zero, so realize_only attributes the close leg against the
        #   already-flipped avg → recovered r_pnl is partial (size/avg stay venue-authoritative)
        if prior_qty != 0.0 and snap_pos.quantity != 0.0 and np.sign(prior_qty) != np.sign(snap_pos.quantity):
            logger.warning(
                f"[{state.exchange}] reconcile: <y>{snap_pos.instrument}</y> flipped "
                f"{prior_qty} -> {snap_pos.quantity}; recovered-deal r_pnl may be incomplete"
            )
        state.reconcile_position_from_snapshot(snap_pos)
        since = snap_pos.last_update_time if snap_pos.last_update_time is not None else snap.as_of
        state.mark_position_reconcile(snap_pos.instrument, since)
        self._spawn(
            ConfirmPositionBySnapshot(
                snap_pos.instrument, since, now, wait=self._position_confirm_wait, expected_delta=delta
            )
        )

    def _flatten_missed_close(
        self, state: AccountState, snap: AccountSnapshot, now: np.datetime64, instrument: Instrument
    ) -> None:
        # - flatten (keeps r_pnl/commissions/funding); missed delta is the whole prior size.
        #   no venue position ts (absent from snapshot) → as_of is the watermark / hist `since`.
        prior = state.get_position(instrument)
        delta = -(prior.quantity if prior is not None else 0.0)
        state.settle_position(instrument)
        state.mark_position_reconcile(instrument, snap.as_of)
        self._spawn(
            ConfirmPositionBySnapshot(
                instrument, snap.as_of, now, wait=self._position_confirm_wait, expected_delta=delta
            )
        )

    def _reconcile_order(self, state: AccountState, snap_order: Order) -> Action | None:
        """Apply a snapshot order's status/filled to local state; return the fill event to route.

        Snapshot is authoritative for the order's own state. Guards: skip if locally terminal or
        not venue-newer; never wipe an in-flight pending marker (the venue resolves that race).
        """
        local = state.get_active_order(snap_order.client_order_id)
        if local is None or local.status.is_terminal or not self._venue_newer(snap_order, local):
            return None
        if local.status.is_pending and not snap_order.status.is_terminal:
            return None
        self._apply_order_snapshot(state, local, snap_order)
        return self._fill_event(local)

    @staticmethod
    def _venue_newer(snap: Order, local: Order) -> bool:
        ts = snap.last_update_time
        return ts is not None and (local.last_update_time is None or ts > local.last_update_time)  # type: ignore

    @staticmethod
    def _apply_order_snapshot(state: AccountState, local: Order, snap: Order) -> None:
        # - go through transition_order (sole index/audit writer); venue wins, so force illegal
        #   transitions but warn
        if snap.status != local.status:
            if not can_transition(local.status, snap.status):
                logger.warning(
                    f"[{state.exchange}] reconcile: forcing {local.client_order_id} "
                    f"{local.status} -> {snap.status} (snapshot authoritative)"
                )
            state.transition_order(local.client_order_id, snap.status, snap.last_update_time)
        else:
            local.last_update_time = snap.last_update_time
        local.filled_quantity = snap.filled_quantity
        local.avg_fill_price = snap.avg_fill_price

    @staticmethod
    def _fill_event(order: Order) -> Action | None:
        # - open_orders only lists live orders, so the only progress here is a partial fill
        #   (fill=None — no deal); FILLED arrives via the missing→RequestStatus reply
        if order.status == OrderStatus.PARTIALLY_FILLED:
            return OrderPartiallyFilledEvent(
                instrument=order.instrument,
                client_order_id=order.client_order_id,
                venue_order_id=order.venue_order_id,
                fill=None,
            )
        return None

    def on_event(self, state: AccountState, event: object, now: np.datetime64) -> list[Action]:
        inp = DealIn(event) if isinstance(event, DealEvent) else OrderIn(event)
        return self._dispatch(inp, state, now, only=self._keys_of(event))

    def _spawn(self, task: Task) -> None:
        if task.key in self._tasks:  # one task per key — duplicate ignored
            return
        self._tasks[task.key] = task
        logger.debug(f"[{task.__class__.__name__}] reconcile spawn task <g>{task.key}</g>")

    def _dispatch(
        self, inp: Any, state: AccountState, now: np.datetime64, only: set[Hashable] | None = None
    ) -> list[Action]:
        out: list[Action] = []
        for key, task in list(self._tasks.items()):
            if only is not None and key not in only:
                continue

            if not task.handles(inp):
                continue

            acts = task.step(inp, state, now)
            if acts:
                logger.debug(
                    f"[{state.exchange}] reconcile step on task <g>{key}</g>(<y>{type(inp).__name__}</y>) ==> <r>{acts}</r>"
                )

            out += acts
            if task.done():
                # - "resolved" = finished with no action (e.g. all deals arrived / order event seen);
                #   "completed" = dropped after a terminal action (fetch / LOST)
                reason = "completed" if acts else "resolved"
                logger.debug(f"[{state.exchange}] reconcile {type(task).__name__} <g>{key}</g> {reason} -> dropped")
                del self._tasks[key]
        return out

    def _snapshot_due(self, exchange: str, now: np.datetime64) -> bool:
        last = self._last_snapshot.get(exchange)
        return last is None or ((now - last) >= self._snapshot_interval)  # type: ignore

    @staticmethod
    def _keys_of(event: object) -> set:
        keys: set = set()
        for attr in ("client_order_id", "venue_order_id"):
            if (v := getattr(event, attr, None)) is not None:
                keys.add(v)
        if (instrument := getattr(event, "instrument", None)) is not None:
            keys.add(getattr(instrument, "symbol", instrument))
        return keys
