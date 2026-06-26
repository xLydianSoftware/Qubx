# Reconciliation Redesign — the Reconciler (stage 2)

**Status:** situation I (orders) implemented + tested; situation II (positions/balances)
and the venue-timestamp foundation are next. See **Implementation status & plan** below.
**Date:** 2026-06-25
**Scope:** stage 2 — consuming the `Diff` atoms from stage 1 (the `Differ` in `diffs.py`,
see `reconciliator-redesign.md`) and resolving live-trading discrepancies over time.
**Visual companion:** `reconciliation-redesign.canvas` (Obsidian).
**Code:** `reconciler.py` (the `Reconciler`, `Task`/`ResolveMissingOrder`, actions),
`reconciler_test.py` (mock-free scenario tests).

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

- Each object needs a **dedicated venue-update timestamp field, in the venue clock**,
  stamped on **both** sides from venue data. **`Order.last_updated_at` is NOT it** — locally
  `transition_order` sets it to our `now` (local clock), and `ccxt_convert_order_info` does
  not set it on snapshot orders at all (so it's `None` there). Reusing it gives a
  cross-clock compare *and* a no-op guard in prod. This field + its stamping is unbuilt —
  see **Venue-timestamp foundation** below. It is a prerequisite for *any* guard to work
  live; the current code uses `last_updated_at` as a placeholder (works only in tests that
  set it by hand).
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

### I. ResolveMissingOrder (local order absent from snapshot) — IMPLEMENTED
A cached order is missing from the snapshot's open-orders. It may have filled/cancelled/
rejected (event not received yet, or missed). Don't blind-cancel.
`MISSING → WAIT (latency freedom) → REQUEST_STATUS (fetch by cid, ≤ n) → RESOLVED | LOST`.
- An arriving order event (the real fill/cancel/reject) resolves it — handled by the normal
  event path, not the task. The task only nudges (`RequestStatus`) and counts retries.
- The order **reappearing in a later snapshot** (still open, or now terminal) also resolves
  the task (`SnapshotIn` → `done`) — a snapshot race must not grind a live order to LOST.
- Exhausted with no answer → give up by **routing `OrderLostEvent`** through the bus
  (`RouteEvent`), *not* a silent mutation: the pipeline terminalizes to `OrderStatus.LOST`
  (new enum value) **and** notifies the strategy — and there is no later WS event to rely on.

### I.b Order status/filled reconcile (present order, fill-progress) — IMPLEMENTED
A still-open order the snapshot shows more-filled than local (we missed the WS fill).
Because **deals never move order status** (they drive the ledger) and the missed status
update may never arrive, the snapshot is authoritative for the order's own state:
`on_snapshot` reconciles `status`/`filled_qty` **in-mem** (`_reconcile_order`), then routes
an **`OrderPartiallyFilledEvent`** (`fill=None`) so the strategy is notified. Guards
(adopted from the old `_reconcile_status_from_snapshot`): strictly-newer venue ts, never
resurrect a terminal order, never wipe an in-flight `PENDING_*` marker, `can_transition`
with force-and-warn. A `FILLED` order can't appear in `open_orders` → only the missing
path produces a real fill notification.

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
- **inline, in `on_snapshot`** (venue-ts guarded): order **fill-progress** reconcile (I.b —
  mutate status/filled + route `OrderPartiallyFilledEvent`). Balances/figures (III) will be
  inline too. Pure price/quantity amends are out of scope for now.
- **spawns a task**: `LocalOrderMissing` → `ResolveMissingOrder` (I); a position size diff
  → `ConfirmPositionBySnapshot` (II, not built); order send → `AwaitOrderConfirm` (not built).
- **routes an event** (`RouteEvent` → `pm.process_event`): give-up LOST (`OrderLostEvent`),
  fill-progress notify (`OrderPartiallyFilledEvent`). The Reconciler injects events only
  where there is no venue event to rely on.

## Testing (mock-free)
Because the Reconciler is pure, a live-trading scenario is a plain data test:
build an `AccountState` + a `Reconciler`, call `on_snapshot(state, snap)`, then drive a
sequence of `on_tick` / `on_event(deal|order)` and assert on `(resulting state, returned
actions, live task keys)`. No connector mocks, no getter/setter tests — only semantic
outcomes (e.g. "a late deal under the snapshot ts does not change the position; it is
logged" / "a missing order with no answer after n ticks ends LOST").

## Implementation status & plan

### Done (situation I — orders)
- `diffs.py` — the pure `Differ` (stage 1), 40 tests.
- `reconciler.py` — `Reconciler` (pure; owns Differ + due-timer + diff→task + a small
  one-per-key task registry, inline `_spawn`/`_dispatch`), `Task` ABC, `ResolveMissingOrder`.
- Actions: `RequestStatus`, `RequestSnapshot`, `RequestHistDeals`, `RouteEvent`.
- `OrderStatus.LOST` (new terminal); `OrderLostEvent` (carries the give-up).
- Behaviors: spawn-without-terminalize; wait→fetch→retry; resolve-by-event; resolve-by-
  reappearance; give-up → `RouteEvent(OrderLostEvent)`; order fill-progress reconcile
  (`_reconcile_order`, with pending/terminal/legality/venue-ts guards) → route
  `OrderPartiallyFilledEvent`. 10 mock-free scenario tests; guards falsified.
- Logging (spawn / step / give-up / drop / snapshot-due).

### Next — situation II (positions & balances)
Sketch below. Then `AwaitOrderConfirm` (replaces `_on_inflight_tick`).

### Next — venue-timestamp foundation (PREREQUISITE, blocks prod)
See its own section below. Until it lands, every venue-ts guard is a placeholder that only
works in tests that set the ts by hand. **Order/position/balance reconcile is non-functional
in prod without it.**

### Then — AM0 wiring
- AM0 (`AccountManager0`) executes actions: `RequestStatus`/`RequestSnapshot`/
  `RequestHistDeals` → connector; `RouteEvent` → `pm.process_event`.
- Reducer needs an `OrderLostEvent` handler (→ `LOST` + fire `on_order`).
- `RouteEvent` of `OrderPartiallyFilledEvent` after an in-mem reconcile: the reducer must be
  idempotent / notify-mostly so it doesn't double-apply (the Reconciler already mutated).
- Build AM0 in parallel; prove on paper; rename AM0 → AM; drop old AM + `reconcile.py`.

## Situation II sketch — positions & balances

**Positions (size diff → retrieve-deals task).** A position-size diff (`PositionSizeMismatch`
/ presence atoms) means we missed deals. Per the order/position split: **deals reconcile the
position, never order status.**
1. `on_snapshot` applies the snapshot position surgically (size/avg/margin only, keep local
   r_pnl/commissions/funding) via `state.reconcile_position_from_snapshot`, stamped with the
   position's **venue ts**.
2. spawn `ConfirmPositionBySnapshot(symbol)` to recover the missed deals for the **record**:
   - `WAIT` a window for the late WS deals to arrive on their own.
   - on timeout → emit `RequestHistDeals(instrument, since=position venue ts)`; the connector
     pushes the fetched trades as normal `DealEvent`s → the pipeline logs (`save_deals`) +
     dedups by `trade_id`; the venue-ts guard skips `_book_deal` (already in the snapshot
     size) → logged-not-double-booked.
   - `OK → drop` once covered / hist fetched.
3. Reducer change: split "record/log the deal" from "book it" in `_handle_deal` (a
   stale-by-venue-ts deal returns `result.deal` for logging but skips `_book_deal`).
   - Deferred edge: a missed deal for an *already-evicted* order won't log (the
     `_seen_trade_ids`/terminal-evict early-return). Active position is fine.

**Balances / figures (III).** No task — apply from the snapshot in-mem, venue-ts guarded for
ordering vs late WS balance pushes. (Presence/figure-presence still later.)

Open question for II: does `ConfirmPositionBySnapshot` track "covered" by accumulating
arriving deal quantity vs the applied delta, or just wait-then-fetch-then-drop? (Lean: the
simpler wait→fetch→drop; the deals are for the record, the size is already corrected.)

## Venue-timestamp foundation (the real prerequisite) — researched 2026-06-26

The monotonic-venue-clock guard needs a **per-object venue-update timestamp, in the venue
clock, on both sides**. **`Order.last_updated_at` is NOT it** — locally `transition_order`
sets it to our `now`, *and* it is the **local-clock eviction key** (`_pending_evict_index[cid]
= order.last_updated_at`, state.py:286), so it **cannot be repurposed**; and
`ccxt_convert_order_info` never sets it on snapshot orders (`None`).

### Decisions
- **Field name = `last_update_time` everywhere** (matches `Position`'s existing field). On
  `Order` this is a **new** field *alongside* `last_updated_at` (local, eviction). `Balance`
  gets one. The guard compares `last_update_time` (venue vs venue), strictly-newer.

### Per-object venue-ts availability (Binance UM, ccxt 4.5.50)
| Object | snapshot (REST) | WS event |
|---|---|---|
| **Order** | ccxt order `lastUpdateTimestamp` / `info.updateTime` (Binance `updateTime`) ✓ | order event's venue ts (add to events; `OrderAcceptedEvent.accepted_at` already venue) |
| **Position** | `info["timestamp"]` (venue ms) ✓ — already stamped in `ccxt_convert_position` (relax the `markPrice` guard) | position push venue ts |
| **Balance** | **NONE** — `/fapi/v2/account` account `updateTime` = **0**, `assets[]` carry no per-asset `updateTime`, and ccxt unified `balance["timestamp"]` is set only on the spot branch (`None` for futures) | **YES** — WS push venue event time `E` (already `BalanceUpdateEvent.as_of`, same clock as `Deal.time`) |

### Consequence — balances are the weak leg
Balances **cannot be strictly venue-clock-guarded from the snapshot** (the venue gives
nothing). So (matching situation III): apply snapshot balances **as-is**; order vs a fresher
WS push with the old tie-break — **skip a currency whose WS push `as_of` (venue `E`) ≥ the
snapshot's `as_of`** (request time). `ccxt_convert_balance` stamps **no** `last_update_time`;
only the WS push does. Orders & positions get the real per-object guard.

### Touch points (~7 files, ~18–22 sites)
1. `core/basics.py` — `Order.last_update_time` (new), `Balance.last_update_time` (new); reuse `Position.last_update_time`.
2. `core/events.py` — `last_update_time` on `OrderEvent` base (so the WS path carries venue ts).
3. `connectors/ccxt/utils.py` — `ccxt_convert_order_info` set it from `lastUpdateTimestamp`/`info.updateTime`; `ccxt_convert_position` relax the `markPrice` guard. (`ccxt_convert_balance`: nothing — no venue ts.)
4. `connectors/ccxt/connector.py` — `_handle_ws_order` carry venue ts onto order events; WS balance push already stamps `E`.
5. `core/account_manager/reducer.py` — stamp `order.last_update_time = event.last_update_time` in the apply path (not local `now`).
6. `core/account_manager/state.py` — venue ts on position/balance apply.
7. `core/account_manager/reconciler.py` — `_venue_newer` → `last_update_time`; position/balance guards (II).
\+ tests.

### Build order
models + the `_venue_newer` guard → order stamping (ccxt + reducer) → position → balances last.

## Deferred / open
- `RequestHistDeals` time window + dedup against already-booked deals.
- venue-figures presence (snapshot has figures we lack / vice versa) — still stage-2-later.
- exact retry/timeout config knobs (reuse the existing `AccountManagerConfig` fields).
