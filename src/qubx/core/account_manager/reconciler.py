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
from qubx.core.account_manager.diffs import Differ, DiffOrders, LocalOrderMissing
from qubx.core.account_manager.state import AccountState
from qubx.core.account_manager.state_machine import can_transition
from qubx.core.basics import Instrument, Order, OrderStatus
from qubx.core.events import (
    AccountSnapshot,
    DealEvent,
    OrderLostEvent,
    OrderPartiallyFilledEvent,
)
from qubx.utils.time import to_timedelta


# - Actions: the I/O vocabulary AccountManager executes (extensible) ------------------------ #
@dataclass(frozen=True)
class RequestStatus:
    cid: str
    venue_id: str | None
    instrument: Instrument


@dataclass(frozen=True)
class RequestSnapshot:
    exchange: str


@dataclass(frozen=True)
class RequestHistDeals:
    instrument: Instrument
    since: np.datetime64


@dataclass(frozen=True)
class RouteEvent:
    # an account event the Reconciler synthesizes for the normal pipeline: AM feeds it to
    # pm.process_event so the reducer applies it AND the strategy is notified (and a later
    # duplicate WS event is deduped there). For decisions with no venue event of their own —
    # e.g. the give-up LOST.
    event: object


Action = RequestStatus | RequestSnapshot | RequestHistDeals | RouteEvent | OrderPartiallyFilledEvent


# - Inputs fed to tasks --------------------------------------------------------- #
class Tick:
    __slots__ = ()


@dataclass(frozen=True)
class OrderIn:
    event: object  # an order-lifecycle event (OrderCanceledEvent, OrderFilledEvent, ...)


@dataclass(frozen=True)
class DealIn:
    deal: DealEvent


@dataclass(frozen=True)
class SnapshotIn:
    snap: AccountSnapshot


def _td(v: str | np.timedelta64) -> np.timedelta64:
    return to_timedelta(v).asm8 if isinstance(v, str) else v


# - Tasks ----------------------------------------------------------------------- #
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
    """FSM I — a local order absent from the snapshot. WAIT (latency freedom) → fetch its
    true status (≤ ``max_retries``) → resolved by the arriving event, or LOST on give-up.

    The arriving real status (fill/cancel/reject, or the order reappearing) is applied by
    the normal event path, not here; this task only nudges the venue and counts retries.
    Knobs are its own constructor params — different task types configure independently.
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
        if isinstance(inp, OrderIn):
            # any order event for our id means the uncertainty is resolved (the normal path
            # applies it) — drop the task
            self._done = True
            return []
        if isinstance(inp, SnapshotIn):
            # the order reappeared at the venue (still open), or its status was just applied
            # terminal from this very snapshot — either way it is no longer "missing"
            order = state.get_order(self._cid)
            if (order is not None and order.status.is_terminal) or self._present_in(inp.snap):
                self._done = True
            return []
        if not isinstance(inp, Tick):
            return []
        if now < self._next_fetch_at:
            return []  # still waiting (latency freedom)
        if self._retries < self._max_retries:
            self._retries += 1
            self._next_fetch_at = now + self._wait
            return [RequestStatus(cid=self._cid, venue_id=self._venue_id, instrument=self._instrument)]
        # give up: no venue answer after the retry budget. Route a LOST event through the
        # bus (AM -> pm.process_event) so the pipeline terminalizes it AND notifies the
        # strategy — never a silent mutation here (a silent one would be invisible, and the
        # later WS duplicate would be deduped by the venue-clock guard → strategy misses it).
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


# - Reconciler ------------------------------------------------------------------ #
class Reconciler:
    def __init__(
        self,
        differ: Differ,
        *,
        snapshot_interval: str | np.timedelta64 = "30s",
        missing_wait: str | np.timedelta64 = "2s",
        missing_max_retries: int = 3,
    ):
        self._differ = differ
        self._snapshot_interval = _td(snapshot_interval)  # reconciler owns the due-timer
        self._missing_wait = _td(missing_wait)  # knobs handed to ResolveMissingOrder tasks
        self._missing_max_retries = missing_max_retries
        self._tasks: dict[Hashable, Task] = {}  # opaque key -> task, one per key
        self._last_snapshot: dict[str, np.datetime64] = {}

    def active_keys(self) -> set[Hashable]:
        return set(self._tasks)

    # -- entry points (pure: return actions, mutate in-mem state) -- #

    def on_tick(self, state: AccountState, now: np.datetime64) -> list[Action]:
        actions: list[Action] = []
        if self._snapshot_due(state.exchange, now):
            self._last_snapshot[state.exchange] = now
            logger.debug(f"[{state.exchange}] reconcile: snapshot due -> RequestSnapshot")
            actions.append(RequestSnapshot(state.exchange))
        return actions + self._dispatch(Tick(), state, now)

    def on_snapshot(self, state: AccountState, snap: AccountSnapshot, now: np.datetime64) -> list[Action]:
        self._last_snapshot[state.exchange] = now  # the snapshot just arrived → reset due-timer
        order_drifts: dict[str, Order] = {}  # cid -> snapshot order (deduped across field atoms)
        for d in self._differ.diff(state, snap):
            logger.debug(d.describe())
            if isinstance(d, LocalOrderMissing):
                self._spawn(
                    ResolveMissingOrder(d.order, now, wait=self._missing_wait, max_retries=self._missing_max_retries)
                )
            elif isinstance(d, DiffOrders):
                order_drifts[d.local.client_order_id] = d.origin
            # positions (II) / balances (III) handled in later steps
        # Order status/filled reconcile: the snapshot is authoritative for the order's own
        # state (deals drive the ledger, NOT order status), and a missed WS status update may
        # never arrive — so mutate status/filled in-mem (guarded), then route the fill event so
        # the strategy is notified. (Position size diffs spawn the retrieve-deals task — II.)
        actions: list[Action] = []
        for snap_order in order_drifts.values():
            if (ev := self._reconcile_order(state, snap_order)) is not None:
                actions.append(RouteEvent(ev))
        return actions + self._dispatch(SnapshotIn(snap), state, now)

    def _reconcile_order(self, state: AccountState, snap_order: Order) -> Action | None:
        """Reconcile one snapshot order into local state; return the fill event to route.

        The snapshot is authoritative for the order's own status/filled (deals never move
        order status, and a missed WS update may never arrive). Skips unless the snapshot
        leg is strictly newer (monotonic venue clock), never resurrects a locally-terminal
        order, and never wipes an in-flight pending marker (our cancel/update may still be
        racing the venue — same rationale as the live accept-during-PENDING_CANCEL guard).
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
        ts = snap.last_updated_at
        return ts is not None and (local.last_updated_at is None or ts > local.last_updated_at)  # type: ignore

    @staticmethod
    def _apply_order_snapshot(state: AccountState, local: Order, snap: Order) -> None:
        # status through transition_order (the sole indices/audit maintainer); on a snapshot
        # the venue is authoritative, so an illegal transition is forced — but loudly.
        if snap.status != local.status:
            if not can_transition(local.status, snap.status):
                logger.warning(
                    f"[{state.exchange}] reconcile: forcing {local.client_order_id} "
                    f"{local.status} -> {snap.status} (snapshot authoritative)"
                )
            state.transition_order(local.client_order_id, snap.status, snap.last_updated_at)
        else:
            local.last_updated_at = snap.last_updated_at
        local.filled_quantity = snap.filled_quantity
        local.avg_fill_price = snap.avg_fill_price

    @staticmethod
    def _fill_event(order: Order) -> Action | None:
        # An order present in the snapshot's open_orders is, by definition, still open
        # (ACCEPTED / PARTIALLY_FILLED) — a FILLED order is terminal and never listed there
        # (it arrives via the missing -> RequestStatus fetch reply, carrying the real deal). So
        # the only fill-progress event this path produces is a partial fill, fill=None (no real
        # deal), routed so the strategy is notified.
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

    # -- task registry (one per key) + routing -- #

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
                logger.debug(f"[{state.exchange}] reconcile task <g>{key}</g> done -> dropped")
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
