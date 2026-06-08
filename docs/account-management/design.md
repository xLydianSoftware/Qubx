# Account Manager — Design & Decisions

Redesign of order submission, lifecycle, and account-state ownership. Replaces the
`IBroker` / `IAccountProcessor` split with a central account-state owner, a typed-event
channel, and an explicit order state machine.

This document records the decisions and their rationale. It is the source of truth for
*why* the code is shaped the way it is.

## Components

```
ProcessingManager  — drains the channel, calls AccountManager.apply(event),
                     fires strategy callbacks from the result (error-isolated)
        │
AccountManager     — routes an event to the right per-exchange AccountState,
                     aggregates metrics across exchanges, owns reconcile ticks
        │
reducer.apply()    — applies one event to one AccountState, drives the state machine,
                     returns ApplyResult (pure state mutation; no callbacks/connectors)
        │
AccountState       — per-exchange data + indices + derived metrics (pure store)
state_machine      — legal order transitions (pure, I/O-free)
events             — typed ChannelMessage hierarchy
```

**Single-mutator invariant:** all `AccountState` writes happen on the strategy thread
(the PM drains the channel and calls `apply` there). The reducer never touches the
strategy or connectors; routing and callback-firing live outside it.

## Order model

- Identity is **`client_id`** (primary) + **`venue_id`** (exchange id). No `id`/`time`/`cost`.
- **Statuses** (`OrderStatus`, `StrEnum`): `INITIALIZED, SUBMITTED, ACCEPTED,
  PARTIALLY_FILLED, PENDING_CANCEL, PENDING_UPDATE, FILLED, CANCELED, REJECTED, EXPIRED`.
- `OrderStatus.is_terminal` (FILLED/CANCELED/REJECTED/EXPIRED) and `is_inflight`
  (SUBMITTED + PENDING_*). **No `is_pending`** — it added a third overlapping concept;
  the revert path keys off `pre_pending_status` instead.
- **`filled_quantity` is unsigned magnitude** (direction lives in `side`), matching
  `Order.quantity` and the OME's positive-amount rule. `Deal.amount` is signed, so fills
  accumulate `abs(amount)` (in both `_apply_fill` and `_recompute_avg`).
- `OrderOrigin`: `FRAMEWORK / RECOVERED / EXTERNAL`.

## State machine

- **Venue-authoritative terminalization:** any live order may move straight to a terminal
  state, so the `TRANSITIONS` table lists only **non-terminal → non-terminal** edges;
  `into terminal` is allowed from any live state by rule, terminals have no outgoing edges.
- Single legality chokepoint: every status change goes through `validate_transition`.

## AccountState

Pure per-exchange store: *data + indices + derived metrics*. Enforces no transition
legality, runs no reconcile rules, fires no callbacks, depends on no clock.

- Indices (mutator-maintained): venue-id, in-flight, pending-evict, `seen_trade_ids`
  (fill dedup), terminal-history ring buffer.
- Side-tables (maintained by `_transition_order`): `retry_count` (in-flight sweep),
  `pre_pending_status` (revert target). Captured on first entry to a PENDING_* state, so
  `PENDING_UPDATE → PENDING_CANCEL` keeps the original; cleared on leaving pending.
  Both dropped on eviction. Chosen over fields-on-`Order` to keep `Order` a clean record
  (same precedent as `seen_trade_ids`).
- Snapshot ratchet: `get_last_snapshot_as_of` / `_mark_snapshot_applied`. The stale-check
  itself is a reconcile *rule* → lives in the reconciler, not here.
- `_prune_terminal_orders(now, retention)` — grace-window eviction into history.
- `base_currency` is **explicit, per exchange** (constructor arg, as the old processor had).
  Never inferred from balances.
- **Derived metrics are methods on `AccountState`** (`total_capital`, `available_margin`,
  `margin_ratio`, `leverage`, `net_leverage`, `gross_leverage`, …). Chosen over a separate
  `accounting` module / helper class to avoid threading `state` and a parallel namespace.
  `AccountManager` aggregates across exchanges (`_sum(AccountState.metric, exchange)`,
  short-circuiting the single-exchange case).

## Capital / margin

- `get_available_margin = total_capital − total_initial_margin` (old parity). The `get_capital`
  alias is **removed** (it only aliased available margin).
- `total_capital = venue.equity ?? base.total + Σ market_value_funds` (old formula, all positions).
- **`get_reserved` removed entirely** — it was deprecated and always returned `0.0`.
- **Capital locking deferred.** The old limit-order lock only moved `base.free ↔ base.locked`,
  which no capital query reads (all use `total`); it was inert. In live, `free`/`locked` come
  from the venue anyway, so emulating it would conflict. Reintroduce **sim-only** if a
  consumer of `balance.free`/`locked` ever appears.
- `conversion_rate = 1.0` (single-base-currency); deferred until the reducer's deal-application
  needs it, where its price-data needs decide the signature.

### Venue-reported figures (option 2)

`AccountState` holds an optional `VenueAccountFigures{equity, available_margin, margin_ratio,
as_of}`, set only via typed `AccountSnapshotEvent` / balance events. Each metric prefers its
venue counterpart when present, else derives. Sim never sets it → always derives.

- **Freshness = WS liveness, not a TTL.** Venue figures arrive on the same user-data stream
  as fills, so they update in lockstep with the events that would change them. The only
  staleness is a dead WS, which the liveness → reconnect → snapshot machinery repairs. We
  never time-out to derived; we keep the last venue figure and let reconcile refresh it.
- Cross-margin only for now (account-level figures; no per-instrument isolated equity).

## Event model (hybrid B)

Order **status** events drive the lifecycle; a **`DealEvent`** drives the ledger.

- `OrderFilledEvent` / `OrderPartiallyFilledEvent` carry an **optional** embedded `fill`;
  `DealEvent` (extends `OrderEvent`, carries the order id) carries a `deal`.
- Combined-stream venues (Binance) deliver status+deal together → the deal rides on the
  order event and is applied immediately. Split-stream venues (OKX/Bitfinex) deliver them
  separately → the deal arrives via `DealEvent`.
- **Dedup by `trade_id`** makes deal application idempotent: whichever stream delivers a
  deal first applies it to `filled_quantity` *and* the ledger; the other is a no-op.
- Connectors stay dumb (forward each stream, or emit two events from one message); the
  **reducer** correlates deals to orders. This removes the fragile connector-side stitching
  that the prior design needed for split streams.

## Strategy callbacks (collapsed)

Three single entrypoints replace the ~9 typed order callbacks (which also eliminates the
dead-callback class of bug, e.g. templates shipping an `on_order_update` that's never called):

- `on_order(ctx, order, change: OrderChange)`
- `on_execution(ctx, deal)`
- `on_position_change(ctx, position)` — fires on **any** position change, deal-driven or not
  (liquidation, ADL, funding, venue push, snapshot), independent of seeing the execution.

No `on_balance_change` — balances are ledger-only, read via `ctx`.

`order.status` already encodes accepted/filled/canceled/rejected/expired (+ `rejected_reason`,
updated price/qty), so most "what happened" is on the order itself. **`OrderChange`** adds the
cases status can't express — `UPDATED`, `CANCEL_REJECTED`, `UPDATE_REJECTED` — and gives
strategies a clean public vocabulary decoupled from the internal event classes.

## `apply` contract

```python
def apply(state, event, now) -> ApplyResult: ...

@dataclass
class ApplyResult:
    order: Order | None = None          # status changed   -> on_order(order, order_change)
    order_change: OrderChange | None    # paired with order
    deal: Deal | None = None            # new deal applied  -> on_execution(deal)
    position: Position | None = None    # position changed  -> on_position_change(position)
```

- The reducer **mutates state and returns the result**; it fires no callbacks (keeps it
  testable without a strategy, and keeps a raising callback away from state mutation).
- The **ProcessingManager** fires each non-`None` field's callback, error-isolated.
- `None` fields are the **suppress** signal — deduped duplicate fills, late events on
  terminal orders, and rejects for unknown orders all return empty results → no callback.
- `order_change` is set by the **reducer** (it knows the from-status, so it can be more
  precise than an event-type lookup — e.g. an accept that confirms a `PENDING_UPDATE`).

## Reconciler (planned)

"The reconciler" splits across three places along the logic-vs-orchestration line:

- **`reconcile.py`** (new module) — the reconciliation *logic*: `reconcile_snapshot(state,
  snapshot, now) -> ReconcileDiff` (terminate orders missing from the snapshot, materialize
  new, update existing, apply position/balance snapshots, with the ratchet + freshness
  guards) plus the sweep *decision* helpers (which inflight orders are stuck, give-up after N
  retries, liveness). Imports `state` + `state_machine` only — never `reducer` — so there is
  no import cycle. Justified as its own file by size + cohesion, not reflex.
- **Reducer** routes the snapshot *event*: a thin `_handle_snapshot` delegates to
  `reconcile.reconcile_snapshot` and returns the diff. This is where `ApplyResult` widens
  with a `reconcile_diff` field -> `on_reconcile_complete`.
- **AccountManager** *drives* the ticks: schedules them via `pm.schedule` and makes the
  connector calls (`request_order_status` / `request_snapshot` / `reconnect` /
  `is_ws_ready`). The ticks are thin — call the `reconcile.py` decision helpers, then fire
  the connector requests; the periodic prune sweep reclaims `seen_trade_ids`.

Built in three sub-steps so each stays testable without the full system:

- **5.3a — snapshot reconcile** (no connectors/scheduler). Extend `AccountSnapshotEvent`
  with optional venue figures (feeds option-2) + a flat `open_orders` list; use the event
  `timestamp` as `as_of`. Add `ReconcileDiff`. `reconcile_snapshot(state, event, now, grace)`:
  ratchet (the stale-check rule, kept out of `AccountState`) → terminate orders missing from
  the snapshot past `grace` → materialize new (`RECOVERED` if the cid has the framework
  prefix, else `EXTERNAL`) → update existing under a freshness guard (`as_of >
  last_updated_at`, so a fresh live fill isn't clobbered) → apply positions/balances → set
  venue figures. Reducer `_handle_snapshot` delegates to it; `ApplyResult` gains
  `reconcile_diff` → `on_reconcile_complete`.
- **5.3b — `IConnector` + tick logic** (fake connectors, no scheduler). `IConnector`
  (`request_order_status` / `request_snapshot` / `reconnect` / `is_ws_ready`),
  `AccountManagerConfig` (intervals/thresholds/retries/retention), and the tick logic:
  in-flight sweep (poll stuck orders, give up after N → `REJECTED` or revert via
  `pre_pending`), snapshot-needed check, liveness/reconnect, prune.
- **5.3c — scheduling wiring** (thin). AM gains `dict[exchange → IConnector]` + config and
  registers the ticks via `set_context`/`pm.schedule`, handling the AM↔PM construction order.

Decisions: `reconcile.py` duplicates the tiny `validate_transition + _transition_order`
helper (cannot import `reducer` — cycle); the framework cid prefix (`qubx_`) is a shared
constant defined now (the TradingManager that emits it comes later).

## Deferred / open

- Wiring venue figures from the connector/reconciler.
- Real `conversion_rate` (multi-currency).
- Bounded channel with **account-event drop protection** (account events must not be
  drop-eligible alongside market data; type-aware drop keyed off the typed hierarchy).
- `set_instrument_leverage` / `set_margin_mode` live on the **connector** (venue commands),
  not the AM; the AM only reflects venue-reported settings.

## Build status

- **Done:** typed events, `OrderStatus` properties, `InvalidOrderTransition`, `state_machine`,
  `AccountState` (data + indices + side-tables + snapshot ratchet + prune + venue figures +
  derived metrics), the **full order-lifecycle reducer** (status, fills/deals with hybrid-B
  trade-id dedup, deal→position/balance ledger, update + cancel/update-rejected, external
  materialization), and the **`AccountManager` core** — route + apply on the AM clock,
  cross-exchange read facade + aggregated metrics, and mark-to-market (`on_market_quote`).
- **Next:** the reconciler (5.3a snapshot reconcile → 5.3b `IConnector` + tick logic → 5.3c
  scheduling); then PM callback wiring, the TradingManager write path, connectors, and the
  `IStrategy` surface change.
