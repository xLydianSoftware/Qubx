# Reconciliation Redesign — the Reconciler (stage 2)

**Status:** design agreed; tests next, then implementation.
**Date:** 2026-06-24
**Scope:** stage 2 — consuming the `Diff` atoms from stage 1 (the `Differ` in `diffs.py`,
see `reconciliator-redesign.md`) and resolving live-trading discrepancies over time.
**Visual companion:** `reconciliation-redesign.canvas` (Obsidian).

## Why

Exchange truth and local (qubx) state drift apart because of: lags (an update not
received yet), missed events (exchange-side or ours), external changes (manual UI /
another bot), and position-size calc differences. Reconciliation periodically pulls a
venue snapshot, diffs it against local state, and resolves the differences — but a
difference is rarely certain at first sight (a "missing" order may just have FILLED in a
WS gap), so resolution is **temporal**: wait, retry, then conclude.

The old approach (`reconcile.py` + `_resolve_missing_orders`/`_on_inflight_tick` in the
manager) fused detection, mutation and orchestration in one pass, smeared the wait/retry
logic across grace windows + retry counters + manager loops, and carried a
covered-quantity *deficit* accounting that is hard to follow and barely tested.

## Core invariant — monotonic venue clock

Every applied object — **Order, Position, Balance** — carries its **own venue update
timestamp**. An incoming update (WS event *or* snapshot leg) is applied **only if its
venue ts is strictly newer** than what we hold; an equal-or-older update is **logged and
dropped, never applied**.

- Order → reuse `Order.last_updated_at`. Position already has it. Balance gets one.
- `Deal.time` is already venue-clock (ms trade time; `ccxt_convert_deal_info`, utils.py:188).
- Snapshot objects must be stamped with **their own per-leg venue ts**, NOT the snapshot
  `as_of` (= request time). The three legs (orders/positions/balances) return at different
  times and in unknown order, so a single request stamp is wrong.

This one rule replaces the old freshness-guard **and** the deficit/suppression accounting:
*last-writer-by-venue-clock wins.* It also covers status safely — a stale event is dropped
whole, so it cannot resurrect a terminal order or force an illegal transition. The guard
lives at the **front of the normal event-application path** (AM0/reducer), because late
updates arrive as ordinary events.

## Architecture

```
            tick / account-event / snapshot-event
                          │
                    ┌─────▼──────┐   pure: returns list[Action], mutates in-mem state, no I/O
                    │ Reconciler │   • owns snapshot due-timer + Differ + diff→task creation
                    │  (engine)  │   • task registry: opaque key -> task, one per key
                    └─────┬──────┘
                  list[Action]  │
                    ┌─────▼──────┐   the ONLY place I/O happens
                    │    AM0     │   executes actions against the connector,
                    │ (thin I/O) │   feeds tick + events + snapshots back in
                    └────────────┘
```

- **`Reconciler` is pure** — no connector/PM calls. Every entry point returns `list[Action]`
  and may mutate the in-memory `AccountState` it is handed. This makes scenario tests
  mock-free.
- **`AM0` does all I/O** — the thin driver. Built in parallel as `AccountManager0`; the
  existing `AccountManager` stays untouched until AM0 is proven, then rename + delete old.
- **`reconcile.py` → `Reconciler` class** in a new `reconciler.py`; the loose module
  functions are dropped once AM0 lands.

## The task engine

The Reconciler is a generic engine of FSM **tasks**:

- Engine is **identity-agnostic**: a flat `dict[key, task]`. The key is opaque (a cid for
  order tasks, a symbol for position tasks); cid/symbol meaning lives only inside tasks.
- **One task per key** — `_spawn` ignores a duplicate key.
- Events carry **candidate keys** — a deal has both a symbol and an order id — so an event
  is routed to any task owning one of its keys.
- One `pm.schedule(tick, on_tick)` drives everything; `_on_inflight_tick` disappears (it
  becomes a task type).

### Task interface

```python
Action = RequestStatus | RequestSnapshot | RequestHistDeals   # extensible

@dataclass(frozen=True)
class RequestStatus:    cid: str; venue_id: str | None; instrument: Instrument
@dataclass(frozen=True)
class RequestSnapshot:  exchange: str
@dataclass(frozen=True)
class RequestHistDeals: instrument: Instrument; since: np.datetime64

class Task(Protocol):
    key: Hashable                                              # cid or symbol (opaque to engine)
    def handles(self, inp: Input) -> bool: ...
    def step(self, inp: Input, state: AccountState, now) -> list[Action]: ...   # mutates state, no I/O
    def done(self) -> bool: ...
# Input ∈ {Tick, DealIn(deal), OrderIn(order_event), SnapshotIn(snap)}
```

`step` is pure-except-in-mem-mutation: it sets position size / transitions an order to
LOST / etc. on the handed `state`, and returns the I/O it wants AM0 to perform. Terminalize
→ LOST is an in-mem mutation, **not** an Action.

### Reconciler entry points

```python
class Reconciler:
    def __init__(self, differ, cfg):
        self._tasks: dict[Hashable, Task] = {}                 # one per key
        self._last_snapshot: dict[str, np.datetime64] = {}

    def on_tick(self, state, now) -> list[Action]:
        # reconciler owns the snapshot due-timer (NOT a task)
        out = [RequestSnapshot(state.exchange)] if self._snapshot_due(state.exchange, now) else []
        return out + self._step_all(Tick(), state, now)

    def on_snapshot(self, state, snap, now) -> list[Action]:
        for d in self._differ.diff(state, snap):               # reconciler owns diff -> task creation
            match d:
                case LocalOrderMissing():           self._spawn(ResolveMissingOrder(d.order))
                case _PositionDiff():               apply_position(state, d); self._spawn(ConfirmPositionBySnapshot(...))
                case _BalanceDiff() | VenueFiguresMismatch():  apply_balance(state, d)     # inline, no task (III)
                case _:                             apply_order_field(state, d)            # inline, venue-ts guarded
        return self._step_all(SnapshotIn(snap), state, now)    # a waiting task may consume it too

    def on_event(self, state, ev, now) -> list[Action]:
        return self._step_all(to_input(ev), state, now, only=keys_of(ev))

    def _step_all(self, inp, state, now, only=None) -> list[Action]:
        out = []
        for key, task in list(self._tasks.items()):
            if (only is None or key in only) and task.handles(inp):
                out += task.step(inp, state, now)
                if task.done(): del self._tasks[key]
        return out

    def _spawn(self, task): self._tasks.setdefault(task.key, task)   # one per key
```

AM0 is the thin driver — feed inputs, execute actions; the one place that touches the
connector.

## Situations & their FSMs

### I. ResolveMissingOrder (local order absent from snapshot)
A cached order is missing from the snapshot's open-orders. It may have filled/cancelled/
rejected (event not received yet, or missed). Don't blind-cancel.
`MISSING → WAIT (latency freedom) → REQUEST_STATUS (fetch by cid, ≤ n) → RESOLVED | LOST`.
- An arriving order event (the real fill/cancel/reject) resolves it — handled by the normal
  event path, not the task. The task only nudges (`RequestStatus`) and counts retries.
- Exhausted with no answer → terminal **`OrderStatus.LOST`** (new enum value).

### II. ConfirmPositionBySnapshot (size diff, deals not yet seen)
Snapshot size ≠ local and no confirming deals booked. Update the position to the snapshot
(authoritative) immediately, stamped with the venue ts. Then watch for the late deals:
- a deal with `ts <= position.last_venue_ts` → **logged but not booked** to the position
  (already counted in the snapshot value — the venue-clock invariant does this for free, no
  deficit math).
- no deals for a period → `REQUEST_HIST_DEALS(symbol, since)`; the connector **pushes the
  fetched deals onto the CtrlChannel as normal `DealEvent`s** — no special logging path.
  The standard pipeline then persists them (`save_deals`), and:
  - already-received trades are **deduped by `trade_id`** (`_apply_execution` /
    `_seen_trade_ids`) → not re-logged, not re-booked.
  - a genuinely-missed trade is new → logged + recorded, but `_book_deal` is **skipped** by
    the venue-ts guard (`deal.time <= position.last_venue_ts`) so the position isn't
    double-applied.
- `OK → drop`.

The only reducer change this needs: split "record/log the deal" from "book it to the
position" in `_handle_deal` — a stale-by-venue-ts deal still returns `result.deal` (logs +
dedup-records) but skips `_book_deal`.

> Deferred edge: `_seen_trade_ids` is dropped on terminal-order eviction and `_handle_deal`
> early-returns for a terminal+evicted order, so a missed deal for an *already-evicted*
> order wouldn't log via this path. Active order/position (the situation-II case) is fine.

### III. Balances / figures
No task. Apply from the snapshot as-is (venue-ts guarded for ordering vs late pushes).

### AwaitOrderConfirm (ex `_inflight_tick`)
On send order/update, push a task keyed by cid: `SENT → WAIT(timeout) → REQUEST_STATUS(≤ n)
→ CONFIRMED | give-up (synth reject / LOST)`. Same engine, replaces the inflight tick.

## Inline vs task (summary)
- **inline, in `on_snapshot`** (venue-ts guarded): order field drifts (price/qty/status on a
  still-open order), positions update, balances, figures.
- **spawns a task**: `LocalOrderMissing` (I), any position diff (II — apply inline *and*
  spawn the confirm task), order send (AwaitOrderConfirm).

## Testing (mock-free)
Because the Reconciler is pure, a live-trading scenario is a plain data test:
build an `AccountState` + a `Reconciler`, call `on_snapshot(state, snap)`, then drive a
sequence of `on_tick` / `on_event(deal|order)` and assert on `(resulting state, returned
actions, live task keys)`. No connector mocks, no getter/setter tests — only semantic
outcomes (e.g. "a late deal under the snapshot ts does not change the position; it is
logged" / "a missing order with no answer after n ticks ends LOST").

## Migration
1. add per-object venue timestamps (Order via `last_updated_at`, Balance new) + stamp
   snapshot legs with their own venue ts.
2. add `OrderStatus.LOST`.
3. build `reconciler.py` (`Reconciler` + tasks + actions) and `AccountManager0`, old AM
   untouched.
4. prove via scenario tests + paper; then rename AM0 → AM, drop old AM + `reconcile.py`.

## Deferred / open
- `RequestHistDeals` time window + dedup against already-booked deals.
- venue-figures presence (snapshot has figures we lack / vice versa) — still stage-2-later.
- exact retry/timeout config knobs (reuse the existing `AccountManagerConfig` fields).
