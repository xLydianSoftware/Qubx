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

from qubx import area_logger
from qubx.core.account_manager.diffs import (
    BalanceMismatch,
    Differ,
    LocalOrderMissing,
    LocalPositionMissing,
    OrderFieldMismatch,
    OriginalBalanceMissing,
    OriginalOrderMissing,
    OriginalPositionMissing,
    PositionFieldMismatch,
    PositionSizeMismatch,
)
from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.account_manager.state_machine import can_transition
from qubx.core.basics import EXTERNAL_CID_PREFIX, Instrument, Order, OrderOrigin, OrderStatus, Position
from qubx.core.events import (
    AccountSnapshot,
    DealEvent,
    OrderLostEvent,
    OrderPartiallyFilledEvent,
)
from qubx.utils.time import to_timedelta

# Area-tagged logger: important reconcile outcomes log at INFO (always visible); per-tick / per-task
# noise logs at DEBUG and only surfaces with QUBX_DEBUG_AREAS=reconciler.
_log = area_logger("reconciler")

# Hist-deals fetch reaches slightly before the position watermark: the triggering trade can sit
# at/just-before the snapshot's position update ts, and fetch_my_trades(since=...) would exclude
# it. Kept small (seconds) so it only catches this episode's boundary trade, not old history —
# re-fetched deals are deduped by trade_id and realize-only-guarded anyway.
HIST_DEALS_LOOKBACK = np.timedelta64(2, "s")

# Funding sweep window: reach back far enough from the trigger to cover the settlement that
# caused it plus venue income-record write-lag; overlap dedups in the reducer buckets.
FUNDING_SWEEP_LOOKBACK = np.timedelta64(30, "m")

# Settlements land on hour boundaries on supported venues; the offset covers venue write-lag.
FUNDING_SWEEP_OFFSET = np.timedelta64(10, "m")


@dataclass(frozen=True)
class RequestStatus:
    # - fetch one order's true status from the venue (missing/uncertain order)
    cid: str
    venue_id: str | None
    instrument: Instrument


@dataclass(frozen=True)
class RequestSnapshot:
    # - pull a fresh account snapshot for this exchange. include_orders=True -> orders (regular +
    #   algo/trigger) + positions + balance (startup discovery + periodic sweep).
    #   include_orders=False -> positions + balance only (steady state): cheap, stays off the
    #   shared REST throttle so it can't delay order sends.
    exchange: str
    include_orders: bool = True


@dataclass(frozen=True)
class RequestHistDeals:
    # - fetch trades since `since` to recover deals missed behind a position diff
    instrument: Instrument
    since: np.datetime64


@dataclass(frozen=True)
class RequestFundingPayments:
    # - fetch account funding settlements since `since` to recover any missed on the WS path
    exchange: str
    since: np.datetime64


@dataclass(frozen=True)
class RouteEvent:
    # - a synthesized event AM feeds to pm.process_event (reducer applies + strategy notified +
    #   later WS dup deduped). For decisions with no venue event of their own, e.g. give-up LOST.
    event: object


Action = (
    RequestStatus | RequestSnapshot | RequestHistDeals | RequestFundingPayments | RouteEvent | OrderPartiallyFilledEvent
)


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
        _log.warning(
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


class AwaitOrderConfirm(Task):
    """
    An order we sent that the venue hasn't confirmed. Wait, then fetch its status
    (≤ max_retries) until a venue event confirms/terminalizes it, or give up → LOST.
    Replaces the manager's inflight tick: time-driven, spawned on order-send.
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
        if isinstance(inp, Tick):
            return True

        if isinstance(inp, OrderIn):
            return self._matches(inp.event)

        return False

    def step(self, inp: Any, state: AccountState, now: np.datetime64) -> list[Action]:
        match inp:
            case OrderIn():
                self._done = True  # - a venue event for our id confirms/terminalizes it
                return []

            case Tick():
                return self._on_tick(state, now)

            case _:
                return []

    def _on_tick(self, state: AccountState, now: np.datetime64) -> list[Action]:
        order = state.get_order(self._cid)

        if order is None or not order.status.is_inflight:
            self._done = True  # - confirmed (ACCEPTED), terminal, or gone
            return []

        self._venue_id = order.venue_order_id or self._venue_id

        if now < self._next_fetch_at:
            return []

        if self._retries < self._max_retries:
            self._retries += 1
            self._next_fetch_at = now + self._wait
            return [RequestStatus(cid=self._cid, venue_id=self._venue_id, instrument=self._instrument)]

        # - budget exhausted → route LOST via the bus (same as ResolveMissingOrder give-up)
        _log.warning(
            f"[{state.exchange}] reconcile give-up on unconfirmed order <g>{self._cid}</g> "
            f"(vid=<blue>{self._venue_id}</blue>) -> <r>LOST</r> after {self._retries} status fetches"
        )

        self._done = True

        return [
            RouteEvent(
                OrderLostEvent(
                    instrument=self._instrument,
                    client_order_id=self._cid,
                    venue_order_id=self._venue_id,
                    reason=f"send not confirmed after {self._retries} status fetches",
                )
            )
        ]

    def _matches(self, event: object) -> bool:
        return getattr(event, "client_order_id", None) == self._cid or (
            self._venue_id is not None and getattr(event, "venue_order_id", None) == self._venue_id
        )


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
            # - reach a hair before the watermark so a trade exactly at it isn't excluded; the
            #   watermark itself (realize-only guard) is unchanged, so booking stays correct
            return [RequestHistDeals(instrument=self._instrument, since=self._since - HIST_DEALS_LOOKBACK)]
        return []


class Reconciler:
    _differ: Differ
    _snapshot_interval: np.timedelta64  # - reconciler owns the due-timer
    _full_snapshot_interval: np.timedelta64  # - how often a due snapshot is FULL (else light)
    _missing_wait: np.timedelta64  # - knobs handed to ResolveMissingOrder tasks
    _missing_max_retries: int
    _position_confirm_wait: np.timedelta64  # - ConfirmPositionBySnapshot window
    _order_confirm_wait: np.timedelta64  # - AwaitOrderConfirm window
    _order_confirm_max_retries: int
    _funding_sweep_enabled: bool  # - False disables the funding sweep entirely (kill switch)
    _tasks: dict[Hashable, Task]  # - opaque key -> task, one per key
    _last_snapshot: dict[str, np.datetime64]  # - per-exchange last snapshot time
    _last_full_snapshot: dict[str, np.datetime64]  # - per-exchange last FULL snapshot request time
    _funding_next: dict[str, np.datetime64]  # - per-exchange next sweep time (hour boundary + offset)
    _funding_floor: dict[str, np.datetime64]  # - per-exchange startup anchor; sweep windows never reach below it

    def __init__(
        self,
        differ: Differ,
        *,
        snapshot_interval: str | np.timedelta64 = "30s",
        full_snapshot_interval: str | np.timedelta64 = "5m",
        missing_wait: str | np.timedelta64 = "2s",
        missing_max_retries: int = 3,
        position_confirm_wait: str | np.timedelta64 = "2s",
        order_confirm_wait: str | np.timedelta64 = "5s",
        order_confirm_max_retries: int = 5,
        funding_sweep_enabled: bool = True,
    ):
        self._differ = differ
        self._snapshot_interval = _td(snapshot_interval)
        self._full_snapshot_interval = _td(full_snapshot_interval)
        self._missing_wait = _td(missing_wait)
        self._missing_max_retries = missing_max_retries
        self._position_confirm_wait = _td(position_confirm_wait)
        self._order_confirm_wait = _td(order_confirm_wait)
        self._order_confirm_max_retries = order_confirm_max_retries
        self._funding_sweep_enabled = funding_sweep_enabled
        self._tasks = {}
        self._last_snapshot = {}
        self._last_full_snapshot = {}
        self._funding_next = {}
        self._funding_floor = {}

    def active_keys(self) -> set[Hashable]:
        return set(self._tasks)

    def on_order_sent(self, state: AccountState, order: Order, now: np.datetime64) -> list[Action]:
        # - spawn an AwaitOrderConfirm for a freshly-sent order (idempotent per cid)
        self._spawn(
            AwaitOrderConfirm(order, now, wait=self._order_confirm_wait, max_retries=self._order_confirm_max_retries)
        )
        return []

    def on_tick(self, state: AccountState, now: np.datetime64) -> list[Action]:
        actions: list[Action] = []
        if self._snapshot_due(state.exchange, now):
            self._last_snapshot[state.exchange] = now
            full_sweep = self._full_snapshot_due(state.exchange, now)
            if full_sweep:
                self._last_full_snapshot[state.exchange] = now
            _log.debug(f"[{state.exchange}] reconcile: snapshot due -> RequestSnapshot(include_orders={full_sweep})")
            actions.append(RequestSnapshot(state.exchange, include_orders=full_sweep))
        if (sweep := self._funding_sweep(state, now)) is not None:
            actions.append(sweep)
        return actions + self._dispatch(Tick(), state, now)

    def _funding_sweep(self, state: AccountState, now: np.datetime64) -> RequestFundingPayments | None:
        # - hourly-aligned sweep: books settlements ≤~10min after any boundary (Binance 8h
        #   boundaries are hour-aligned; HPL settles hourly and additionally books instantly via
        #   WS); WS gaps self-heal at the next sweep; the floor guards restored-state double-booking.
        if not self._funding_sweep_enabled:
            return None
        exchange = state.exchange
        if (floor := self._funding_floor.get(exchange)) is None:
            self._funding_floor[exchange] = floor = self._funding_anchor(state, now)
            self._funding_next[exchange] = self._next_sweep_at(now)
            _log.debug(f"[{exchange}] reconcile: startup funding sweep -> RequestFundingPayments(since={floor})")
            return RequestFundingPayments(exchange, since=floor)
        if now < self._funding_next[exchange]:
            return None
        self._funding_next[exchange] = self._next_sweep_at(now)
        since = max(floor, now - FUNDING_SWEEP_LOOKBACK)
        _log.debug(f"[{exchange}] reconcile: funding sweep due -> RequestFundingPayments(since={since})")
        return RequestFundingPayments(exchange, since=since)

    @staticmethod
    def _next_sweep_at(now: np.datetime64) -> np.datetime64:
        return now.astype("datetime64[h]") + np.timedelta64(1, "h") + FUNDING_SWEEP_OFFSET

    @staticmethod
    def _funding_anchor(state: AccountState, now: np.datetime64) -> np.datetime64:
        # - startup watermark: +1ns past the newest restored settlement (EXCLUSIVE — it is already
        #   inside restored cumulative_funding and the bucket table is empty after restart, so
        #   re-fetching it would double-book); no restored anchor -> now (restart gap stays open
        #   until last_funding_time persistence lands).
        times = []
        for pos in state.get_positions().values():
            t = pos.last_funding_time
            if isinstance(t, (int, np.integer)):  # in-session bookings stamp the raw ns epoch
                t = np.datetime64(int(t), "ns")
            if not np.isnat(t):
                times.append(t)
        return max(times) + np.timedelta64(1, "ns") if times else now

    def on_snapshot(
        self,
        state: AccountState,
        snap: AccountSnapshot,
        now: np.datetime64,
        *,
        changed_positions: list[Position] | None = None,
    ) -> list[Action]:
        # - applies are idempotent, so repeated field atoms for one order/position are safe.
        #   leaf arms precede their base arm (a leaf must win). unhandled atoms are deferred.
        #   changed_positions (opt-in) collects every reconciled position so the AM can fire
        #   on_position_change at snapshot time.
        self._last_snapshot[state.exchange] = now  # - reset the snapshot due-timer
        # - as_of ratchet: drop an out-of-order snapshot wholesale (at/before the last applied)
        last_as_of = state.get_last_snapshot_as_of()
        if last_as_of is not None and snap.as_of <= last_as_of:
            _log.debug(f"[{state.exchange}] reconcile: stale snapshot as_of={snap.as_of} <= {last_as_of} — dropped")
            return []
        # - FIRST reconcile after start (no prior snapshot applied, last_as_of is None): adopt the
        #   venue positions but skip hist-deals recovery. Adopting startup state is the position
        #   RESTORER's job; recovering trades on start would double-count against restored r_pnl.
        #   Restorers seed positions/balances, never the snapshot watermark, so last_as_of stays
        #   None across a restart. In-session snapshots (last_as_of set) recover missed deals.
        spawn_confirm = last_as_of is not None
        state.mark_snapshot_applied(snap.as_of)
        changed = changed_positions if changed_positions is not None else []
        actions: list[Action] = []
        for difference in self._differ.diff(state, snap):
            _log.debug(difference.describe())
            # - each atom is applied in isolation: a handler that raises (e.g. a venue/state
            #   surprise) must not sink the rest of the snapshot reconcile (balances/figures/
            #   other positions). Log and move on; the next snapshot re-derives the diff.
            try:
                match difference:
                    case LocalOrderMissing(order=order):
                        self._spawn(
                            ResolveMissingOrder(
                                order, now, wait=self._missing_wait, max_retries=self._missing_max_retries
                            )
                        )

                    # - venue has an order we don't track -> recover it (RECOVERED / EXTERNAL)
                    case OriginalOrderMissing(order=snap_order):
                        self._recover_order(state, snap_order, snap.as_of)

                    case OrderFieldMismatch(origin=snap_order):
                        if (ev := self._reconcile_order(state, snap_order)) is not None:
                            actions.append(RouteEvent(ev))

                    # - size diff = missed deals
                    case PositionSizeMismatch(origin=snap_pos) | OriginalPositionMissing(position=snap_pos):
                        changed.append(self._reconcile_missed_position(state, snap, now, snap_pos, spawn_confirm))

                    # - avg/margin only = figure refresh, no missed deals
                    case PositionFieldMismatch(origin=snap_pos):
                        if state.reconcile_position_from_snapshot(snap_pos):
                            changed.append(state.get_position(snap_pos.instrument))

                    # - local holds it, venue flat = missed the close
                    case LocalPositionMissing(position=local_pos):
                        changed.append(
                            self._flatten_missed_close(state, snap, now, local_pos.instrument, spawn_confirm)
                        )

                    case BalanceMismatch(origin=snap_bal) | OriginalBalanceMissing(balance=snap_bal):
                        # - push-wins: a WS push at/after the snapshot's as_of supersedes it (venue event
                        #   time vs local fetch clock) — skip the whole currency, not just the stamp
                        push_as_of = state.get_balance_push_as_of(snap_bal.currency)
                        if push_as_of is None or push_as_of < snap.as_of:
                            state.apply_balance_snapshot(snap_bal, snap.as_of)
            except Exception:
                _log.exception(f"[{state.exchange}] reconcile: diff atom failed, skipping -> {difference.describe()}")

        # - venue-reported figures (equity/margins): prefer-venue-else-derive per metric in
        #   AccountState. Absence = "not observed" -> keep the previous capture, never clear.
        if any(v is not None for v in (snap.equity, snap.available_margin, snap.margin_ratio, snap.withdrawable)):
            state.set_venue_figures(
                VenueAccountFigures(
                    as_of=snap.as_of,
                    equity=snap.equity,
                    available_margin=snap.available_margin,
                    margin_ratio=snap.margin_ratio,
                    withdrawable=snap.withdrawable,
                )
            )

        return actions + self._dispatch(SnapshotIn(snap), state, now)

    @staticmethod
    def _recover_order(state: AccountState, snap_order: Order, as_of: np.datetime64) -> Order:
        # - venue order absent locally: trust the producer-assigned origin (only the connector
        #   knows its cid prefix). EXTERNAL keeps/derives an ext: cid; anything else is a
        #   framework order seen back -> RECOVERED. No deficit: the position watermark guards
        #   any replayed fills (situation II).
        if snap_order.origin is OrderOrigin.EXTERNAL:
            origin = OrderOrigin.EXTERNAL
            cid = snap_order.client_order_id
            if not cid.startswith(EXTERNAL_CID_PREFIX):
                cid = f"{EXTERNAL_CID_PREFIX}{snap_order.venue_order_id}"
        else:
            origin = OrderOrigin.RECOVERED
            cid = snap_order.client_order_id

        # - recovered order
        state.add_order(
            order := Order(
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
                last_update_time=snap_order.last_update_time if snap_order.last_update_time is not None else as_of,
            )
        )
        _log.info(
            f"[{state.exchange}] reconcile: '{origin.value}' order <y>{cid}</y> recovered from snapshot (status={order.status})"
        )
        return order

    def _reconcile_missed_position(
        self, state: AccountState, snap: AccountSnapshot, now: np.datetime64, snap_pos: Position, spawn_confirm: bool
    ) -> Position:
        # - apply the snapshot size, watermark it (reducer won't re-book deals it already covers),
        #   then confirm-task the missed delta (snapshot - prior, captured before the apply).
        #   Returns the reconciled live position (for on_position_change). On the first reconcile
        #   after start (spawn_confirm=False) the confirm task / hist-deals request is skipped —
        #   the restorer owns startup state; only in-session drift recovers missed deals.
        prior = state.get_position(snap_pos.instrument)
        prior_qty = prior.quantity if prior is not None else 0.0
        delta = snap_pos.quantity - prior_qty
        # - flip: the missed deals crossed zero, so realize_only attributes the close leg against the
        #   already-flipped avg → recovered r_pnl is partial (size/avg stay venue-authoritative)
        if prior_qty != 0.0 and snap_pos.quantity != 0.0 and np.sign(prior_qty) != np.sign(snap_pos.quantity):
            _log.warning(
                f"[{state.exchange}] reconcile: <y>{snap_pos.instrument}</y> flipped "
                f"{prior_qty} -> {snap_pos.quantity}; recovered-deal r_pnl may be incomplete"
            )
        state.reconcile_position_from_snapshot(snap_pos)
        since = snap_pos.last_update_time if snap_pos.last_update_time is not None else snap.as_of
        state.mark_position_reconcile(snap_pos.instrument, since)
        if spawn_confirm:
            self._spawn(
                ConfirmPositionBySnapshot(
                    snap_pos.instrument, since, now, wait=self._position_confirm_wait, expected_delta=delta
                )
            )
        else:
            _log.info(
                f"[{state.exchange}] first reconcile: adopted venue position <y>{snap_pos.instrument}</y> "
                f"size={snap_pos.quantity} without hist-deals (restorer owns startup state)"
            )
        return state.get_position(snap_pos.instrument)

    def _flatten_missed_close(
        self,
        state: AccountState,
        snap: AccountSnapshot,
        now: np.datetime64,
        instrument: Instrument,
        spawn_confirm: bool,
    ) -> Position:
        # - flatten (keeps r_pnl/commissions/funding); missed delta is the whole prior size.
        #   no venue position ts (absent from snapshot) → as_of is the watermark / hist `since`.
        #   Returns the now-flat live position (for on_position_change). First reconcile after start
        #   (spawn_confirm=False) adopts the venue-flat state without a hist-deals request.
        prior = state.get_position(instrument)
        delta = -(prior.quantity if prior is not None else 0.0)
        state.settle_position(instrument)
        state.mark_position_reconcile(instrument, snap.as_of)
        if spawn_confirm:
            self._spawn(
                ConfirmPositionBySnapshot(
                    instrument, snap.as_of, now, wait=self._position_confirm_wait, expected_delta=delta
                )
            )
        return state.get_position(instrument)

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
        # - venue-vs-venue only (the connector stamps last_update_time on snapshot orders); no
        #   local-clock as_of fallback — that would reintroduce the cross-clock skew
        ts = snap.last_update_time
        return ts is not None and (local.last_update_time is None or ts > local.last_update_time)  # type: ignore

    @staticmethod
    def _apply_order_snapshot(state: AccountState, local: Order, snap: Order) -> None:
        # - capture the venue id for an unacked order matched by cid (lost create-ack): the
        #   snapshot carries the id the venue assigned; without this, later cancel/status-by-vid break
        if snap.venue_order_id is not None and local.venue_order_id != snap.venue_order_id:
            state.set_venue_id(local.client_order_id, snap.venue_order_id)
        # - go through transition_order (sole index/audit writer); venue wins, so force illegal
        #   transitions but warn
        if snap.status != local.status:
            if not can_transition(local.status, snap.status):
                _log.warning(
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
        _log.debug(f"[{task.__class__.__name__}] reconcile spawn task <g>{task.key}</g>")

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
                _log.debug(
                    f"[{state.exchange}] reconcile step on task <g>{key}</g>(<y>{type(inp).__name__}</y>) ==> <r>{acts}</r>"
                )

            out += acts
            if task.done():
                # - "resolved" = finished with no action (all deals arrived / order event seen) -> DEBUG noise;
                #   "completed" = dropped after a terminal action (hist-deals fetch / LOST) -> INFO (important)
                if acts:
                    _log.info(f"[{state.exchange}] reconcile {type(task).__name__} <g>{key}</g> completed -> dropped")
                else:
                    _log.debug(f"[{state.exchange}] reconcile {type(task).__name__} <g>{key}</g> resolved -> dropped")
                del self._tasks[key]
        return out

    def _snapshot_due(self, exchange: str, now: np.datetime64) -> bool:
        last = self._last_snapshot.get(exchange)
        return last is None or ((now - last) >= self._snapshot_interval)  # type: ignore

    def _full_snapshot_due(self, exchange: str, now: np.datetime64) -> bool:
        # - first request ever (no local order state yet -> must discover all/algo orders) or the
        #   periodic full sweep is due. Driven by request time, not arrival, so a light snapshot
        #   landing never resets the sweep timer.
        last = self._last_full_snapshot.get(exchange)
        return last is None or ((now - last) >= self._full_snapshot_interval)  # type: ignore

    @staticmethod
    def _keys_of(event: object) -> set:
        keys: set = set()
        for attr in ("client_order_id", "venue_order_id"):
            if (v := getattr(event, attr, None)) is not None:
                keys.add(v)
        if (instrument := getattr(event, "instrument", None)) is not None:
            keys.add(getattr(instrument, "symbol", instrument))
        return keys
