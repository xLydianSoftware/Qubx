# Account Manager — Design & Decisions

Redesign of order submission, lifecycle, and account-state ownership. Replaces the
`IBroker` / `IAccountProcessor` split with a central account-state owner, a typed-event
channel, and an explicit order state machine.

This document records the decisions and their rationale — it is the source of truth for
*why* the code is shaped the way it is. It started from PR #302's design doc and is
adapted to what this branch actually built; deliberate departures are collected in
[Divergences from PR #302](#divergences-from-pr-302).

## Components

```
ProcessingManager   — drains the channel, routes AccountMessages to AccountManager.apply,
                      fires strategy callbacks from the ApplyResult (error-isolated),
                      owns the ops-metric seam (apply errors, reconcile drift counters)
        │
AccountManager      — routes an event to the right per-exchange AccountState, applies it
                      via the reducer, owns the periodic ticks (in-flight sweep, snapshot,
                      liveness) and the apply-path terminal-eviction sweep, makes every
                      connector call (status fetch / snapshot / reconnect).
                      SimulatedAccountManager = the same minus ticks (paper + backtest)
        │
reducer.apply()     — applies one event to one AccountState, drives the state machine,
                      returns ApplyResult (pure mutation; no callbacks/connectors/clock)
        │
reconcile.py        — reconciliation logic: snapshot reconcile, the sweep decision
                      helpers, and the shared validate+transition chokepoint
        │
AccountState        — per-exchange data + indices + derived metrics (pure store)
state_machine       — legal order transitions (pure, I/O-free)
core/events.py      — typed ChannelMessage hierarchy (the AccountMessage side is live;
                      MarketDataMessage is reserved — market data rides tuples today)
```

**Single-mutator invariant:** all `AccountState` writes happen on the strategy's
event-processing thread (the PM drains the channel and calls `apply` there; the AM ticks
are PM-scheduled onto the same thread). The reducer never touches the strategy or
connectors; routing and callback-firing live outside it. To keep the invariant honest,
`on_fit` / `on_warmup_finished` run **inline on the processing thread** (the old
ThreadPool(2) is deleted): strategies call `ctx.trade()` / `get_position` from those
callbacks, and `get_position` is a mutating read. Trade-off, stated in the `IStrategy`
docstrings: a long fit blocks event processing.

Import rule: `reconcile.py` never imports the reducer or the manager (they import *it*).
The shared `transition()` (validate-then-apply, the single legality chokepoint) lives in
`reconcile.py` for exactly that reason.

## Order model

- Identity is **`client_order_id`** (primary) + **`venue_order_id`** (exchange id).
  No `id`/`time`/`cost`. Full names, not `client_id`/`venue_id`: both ids travel together
  through connectors, events and reconcile, where the short forms were ambiguous.
- **Statuses** (`OrderStatus`, `StrEnum`): `INITIALIZED, SUBMITTED, ACCEPTED,
  PARTIALLY_FILLED, PENDING_CANCEL, PENDING_UPDATE, FILLED, CANCELED, REJECTED, EXPIRED`.
- `OrderStatus.is_terminal` (FILLED/CANCELED/REJECTED/EXPIRED), `is_inflight`
  (SUBMITTED + PENDING_*), and `is_pending` (PENDING_* only — pure classification, used
  where "a modify is outstanding" is the question). The revert path never infers from
  these; it keys off the captured `pre_pending_status`.
- **`filled_quantity` is unsigned magnitude** (direction lives in `side`), matching
  `Order.quantity` and the OME's positive-amount rule. `Deal.amount` is signed, so fills
  accumulate `abs(amount)`.
- `OrderOrigin`: `FRAMEWORK / RECOVERED / EXTERNAL`, classified from the cid prefix by
  `classify_origin` in `basics` — the shared constants `FRAMEWORK_CID_PREFIX` (`qubx_`)
  and `EXTERNAL_CID_PREFIX` (`ext:`) live next to it; nothing else spells the prefixes.
- **Transition audit:** every status change appends an `OrderTransition` to
  `Order.transitions` and bumps a per-exchange per-status counter
  (`get_order_history(cid)` / `get_metrics()` read them; counters are operator/debug
  surface, never reset within a session).

## State machine

- **Venue-authoritative terminalization:** any live order may move straight to a terminal
  state, so the `TRANSITIONS` table lists only **non-terminal → non-terminal** edges;
  `into terminal` is allowed from any live state by rule, terminals have no outgoing edges.
- Single legality chokepoint: every status change goes through `validate_transition`
  (via `reconcile.transition`). One sanctioned exception: snapshot reconcile may **force**
  an illegal transition with a WARNING — the venue is authoritative, but it must still go
  through `transition_order` so the audit and indices stay consistent.

## AccountState

Pure per-exchange store: *data + indices + derived metrics*. Enforces no transition
legality, runs no reconcile rules, fires no callbacks, depends on no clock.

- Mutators are **public, unprefixed** methods (`add_order`, `transition_order`,
  `set_position`, …), framework-internal by contract (module docstring): the callers are
  sibling modules (reducer/reconcile/manager), so underscore-privates would be a lie.
  Strategies only ever see the read API.
- `add_order` **refuses a duplicate active cid** (raises): silent overwrite orphaned the
  caller's Order reference and left stale index entries behind.
- Indices (mutator-maintained): venue-id, in-flight, pending-evict, `seen_trade_ids`
  (fill dedup), snapshot-fill deficit/suppressed (covered-execution suppression, see
  Reconciler), terminal-history ring buffer, applied-funding buckets (bounded,
  oldest-evicted).
- Side-tables (maintained by `transition_order`): `retry_count` (sweep/fetch budget),
  `pre_pending_status` (revert target). Captured on first entry to a PENDING_* state, so
  `PENDING_UPDATE → PENDING_CANCEL` keeps the original; cleared on leaving pending.
  Both dropped on eviction. Chosen over fields-on-`Order` to keep `Order` a clean record.
- Snapshot ratchet: `get_last_snapshot_as_of` / `mark_snapshot_applied`. The stale-check
  itself is a reconcile *rule* → lives in the reconciler, not here.
- `prune_terminal_orders(now, retention)` — grace-window eviction into history.
- `base_currency` is **explicit, per exchange** (constructor arg). Never inferred from
  balances.
- **Derived metrics are methods on `AccountState`** (`total_capital`, `available_margin`,
  `margin_ratio`, `leverage`, `net_leverage`, `gross_leverage`, …). `AccountManager`
  aggregates across exchanges; leverage aggregation is capital-weighted.
- Identity-preserving writes: `set_position`/`update_balance` update the existing object
  in place (callers across the framework hold references), never swap it.

## Capital / margin

- `get_available_margin = total_capital − total_initial_margin` (old parity). The
  `get_capital` alias is **removed** (it only aliased available margin).
- `total_capital = venue.equity ?? base.total + Σ market_value_funds` (old formula).
- `get_reserved` is **removed** end-to-end (viewer interface, AM, rebalancer tracker);
  the reservation concept itself is gone.
- **Capital locking deferred.** The old limit-order lock only moved `base.free ↔
  base.locked`, which no capital query reads; it was inert. In live, `free`/`locked` come
  from the venue anyway. Reintroduce **sim-only** if a consumer ever appears.
- `conversion_rate = 1.0` — **single-base-currency limitation**: realized PnL, fees and
  funding are booked 1:1 into the settle/quote currency. Instruments not quoted/settled
  in the portfolio base currency (e.g. COIN-M) are out of scope until the multi-currency
  seam is filled.
- Per-exchange fee schedules: the AM holds `tcc: dict[exchange →
  TransactionCostsCalculator]` (ZERO_COSTS fallback), keyed like connectors.

### Venue-reported figures (option 2)

`AccountState` holds an optional `VenueAccountFigures{equity, available_margin,
margin_ratio, as_of}`. Each metric prefers its venue counterpart when present, else
derives. The figures ride **flat on `AccountSnapshot`** (optional fields next to
`as_of`/orders/positions/balances) and are set only by snapshot reconcile; the connector
extracts them via the `_extract_venue_figures` seam (Binance and OKX wired; Bitfinex
documented derive-only — `fetch_balance` carries no account figures; other venues return
None and always derive — as does sim).

- **Freshness = WS liveness, not a TTL.** Venue figures arrive in lockstep with the
  events that would change them; the only staleness is a dead WS, which the liveness →
  reconnect → snapshot machinery repairs. We never time-out to derived.
- Cross-margin only for now (account-level figures; no per-instrument isolated equity).

## Event model (hybrid B)

Order **status** events drive the lifecycle; a **`DealEvent`** drives the ledger.

- Events live in **`core/events.py`**, not inside the `account_manager` package:
  connectors and the PM are producers/consumers too, and the package must stay
  importable without dragging the manager. `AccountMessage` is the marker base the PM
  routes on (`isinstance`) — market data cannot be misrouted into the state machine.
- `OrderEvent.client_order_id` is **`str | None`**: connectors pervasively see venue-id-
  only messages (deals/rejects before the cid index is seeded, external orders). The
  reducer resolves by cid then venue id, and materializes `ext:<venue_id>` orders only on
  **money-carrying** events (fill/partial/deal/updated) — never for bare rejects/cancels.
- `OrderFilledEvent` / `OrderPartiallyFilledEvent` carry an **optional** embedded `fill`;
  `DealEvent` carries a `deal`. Combined-stream venues (Binance) deliver status+deal
  together → the deal rides embedded. Split-stream venues (OKX/Bitfinex, via the
  two-stream connector base) deliver them separately → the deal arrives as a `DealEvent`.
- **Dedup by `trade_id`** makes deal application idempotent: whichever stream delivers a
  deal first applies it to `filled_quantity` *and* the ledger; the other is a no-op.
- Out-of-order delivery is absorbed by the reducer, not the connector: a FILLED arriving
  before its ACCEPTED leaves the late accept a benign no-op (venue id still captured),
  and an accept racing an outstanding cancel **never wipes `PENDING_CANCEL`** — the sweep
  keeps polling and a later cancel-rejected still reverts.
- A deal for a **terminal-but-retained** order (inside the eviction grace window) still
  books — the split-stream FILLED status can beat its final trade. Past eviction the
  dedup table is gone, so post-eviction deals are suppressed instead of risking a
  double-count.
- `FundingPaymentEvent` is an AccountMessage: the reducer values the payment at the
  position's mark price and dedups per `(instrument, funding-interval bucket)` (bounded
  set), so venue + simulated re-deliveries apply once. No mark yet → skip *without*
  consuming the bucket, so a re-delivery can apply later.
- `PositionUpdateEvent` / `BalanceUpdateEvent` are **absolute venue pushes** (Binance UM
  `ACCOUNT_UPDATE`, riding the existing listenKey user-data WS — other venues stay
  snapshot-only via the `_account_streams()` seam), while the deal path books **deltas**.
  The venue's per-fill ordering of `ORDER_TRADE_UPDATE` vs `ACCOUNT_UPDATE` is not
  reliably documented, so application is **ordering-agnostic**:
  - **Position pushes never write size on the event path** — sizes are owned by the deal
    ledger. A push size-equal to local (within one lot) advances the per-instrument push
    `as_of` ratchet and returns the **local** position in `ApplyResult.position`, so
    `on_position_change` fires off local state; a stale push (at/behind the ratchet)
    returns empty and fires nothing. Size drift sets `ApplyResult.position_drift`
    with **zero mutation**, and the AccountManager fires a rate-limited (~2 s per
    exchange) `request_snapshot()` + WARNING — the race-safe snapshot reconcile corrects
    the size. Venue upnl/mark are not taken from pushes (positions stay locally marked);
    hedge-mode pushes (`ps != BOTH`) are warned+skipped at the connector (qubx `Position`
    is net-only).
  - **Balance pushes apply absolutely** (`apply_balance_push`) under a per-currency
    `as_of` ratchet — strictly-older pushes drop; `as_of` is venue event time, the same
    clock domain as `Deal.time`. Futures pushes carry the wallet total only (free/locked
    = NaN by producer contract): `total` overwrites, `free` moves by the delta, `locked`
    survives; a real split (free AND locked non-NaN) overwrites all three.
  - **Covered-delta guard:** `_book_deal` and `_handle_funding_payment` skip *only their
    `adjust_balance` leg* when the currency's push `as_of` is at/after the deal/funding
    venue time — the absolute push already incorporated that cash change. Correct under
    both `[deal, push]` and `[push, deal]` orderings (the ordering-matrix tests pin
    convergence to identical qty/r_pnl/balances), and it fixes the funding double-count
    (our computed `FundingPaymentEvent` vs the venue's actual `FUNDING_FEE` debit).
    Position size, `r_pnl` and `cumulative_funding` always book.
  - Dispatch: a size-equal push reaches `on_position_change` with the local position
    through the purely field-driven path (no PM special case); stale and drift pushes
    fire no strategy callback; balance pushes fire **no** strategy callback either (see
    Strategy callbacks).

  With this, liquidation/ADL and venue balance changes reach state at WS latency instead
  of the ~30 s snapshot tick — **leveraged live (Binance UM) is ready pending a testnet
  soak**, which also pins the real per-fill event ordering for a later opt-in
  direct-apply upgrade.

## Strategy callbacks

The PR #302 callback collapse is **adopted**: `on_order_update` / `on_account_update`
are hard-removed (a strategy defining either fails at construction with a migration
TypeError — no deprecation period, dual surfaces would double-fire), replaced by three
callbacks:

- `on_order(ctx, order, change: OrderChange)` — once per applied order-lifecycle change;
  `change` is the reducer's vocabulary (ACCEPTED / PARTIALLY_FILLED / FILLED / CANCELED /
  EXPIRED / REJECTED / UPDATED / CANCEL_REJECTED / UPDATE_REJECTED).
- `on_execution(ctx, instrument, deal)` — once per newly applied, deduplicated fill
  (`Deal` carries no instrument field, hence the explicit parameter).
- `on_position_change(ctx, position)` — position changed: every fill, funding payments,
  venue position pushes, and each position corrected by a snapshot reconcile. High fire
  rate by design — keep it cheap.

There is deliberately **no balance callback**: balances are read via
`ctx.get_balances()` / `ctx.get_balance()`, and `BalanceUpdateEvent` applies to account
state silently — for applied pushes (`ApplyResult.balance` set) and suppressed ones
alike (both pinned in the dispatch tests).

The dead-callback class of bug (a stale-arity override dying in a swallowed TypeError on
every dispatch) is addressed at the root: a one-time **signature guard at strategy
construction** fails loudly on incompatible overrides and on the removed pre-collapse
names, and the `qubx init` templates ship the current signatures. Cancel/update reject
reasons stay PM-log-only (the STILL-ALIVE warnings), not stored on the order.
Framework-internal fill consumers (trackers, gatherers, logging, export) are fed off
`result.deal` — the deduped truth — independent of the strategy callback.

## `apply` contract

```python
def apply(state, event, now, *, snapshot_grace) -> ApplyResult: ...

@dataclass
class ApplyResult:
    order: Order | None = None              # status changed -> on_order(order, change)
    order_change: OrderChange | None = None # paired with order
    deal: Deal | None = None                # new deal applied -> on_execution + downstream fill consumers
    position: Position | None = None        # position changed -> on_position_change
    reconcile_diff: ReconcileDiff | None = None  # snapshot reconcile -> on_position_change per corrected position
    position_drift: Position | None = None  # venue push disagrees with local size -> AM snapshot request, no callback
    balance: Balance | None = None          # balance push applied -> internal visibility only, NO strategy callback
```

- The reducer **mutates state and returns the result**; it fires no callbacks (testable
  without a strategy; a raising callback stays away from state mutation). The grace
  window — the one config value the reducer needs — is passed kw-only rather than stored.
- The **ProcessingManager** fires callbacks from the result, error-isolated, and counts
  reconcile drift (`terminated`/`materialized`) off the diff — the metric seam lives on
  the PM; the AM holds no emitter.
- **None-as-suppress:** an empty result means the AM suppressed the event (late/
  duplicate/terminal/unknown/deduped funding/stale snapshot — already logged) and nothing
  fires. Each set field fires its callback: `order` + `order_change` → `on_order`,
  `deal` → `on_execution` (plus downstream fill consumers), `position` →
  `on_position_change`, `reconcile_diff` → `on_position_change` per corrected position.
  There is no event-type special case: a size-equal venue position push carries the
  local position in `position` (the reducer never writes size off a push), so it
  reaches `on_position_change` through the same field-driven path, while a stale
  ratchet-dropped push returns empty and fires nothing. `position_drift` fires
  nothing strategy-side (the AM reacts with the rate-limited snapshot request);
  `BalanceUpdateEvent` fires nothing either way (see Strategy callbacks).

## Reconciler

Split along the logic-vs-orchestration line: `reconcile.py` owns the *logic* (snapshot
reconcile, sweep decision helpers, `transition`), the reducer routes the snapshot *event*
(`_handle_snapshot` delegates and surfaces the `ReconcileDiff`), and the **AccountManager
drives the ticks** and makes every connector call.

### Snapshot reconcile

`reconcile_snapshot(state, snapshot, now, grace) -> ReconcileDiff | None`, in order:

1. **as_of ratchet** — a snapshot at or before the last applied one is rejected wholesale
   (returns None; the PM fires nothing).
2. **Missing detection** — a cached live order is "present" if its venue id **or its cid**
   appears in the snapshot's open orders (the cid fallback covers an unacked framework
   order whose create-ack was lost). Untimestamped orders are skipped (cannot be aged —
   could be a just-submitted order racing the snapshot); the rest must be missing past
   `grace`.
3. **Fetch-before-terminalize** — missing orders are *reported* on `diff.missing`, not
   terminalized: the order may have FILLED during a WS gap and a blind CANCELED would
   lose the execution forever. The manager requests the true status per missing order
   (the answer replays through the normal event path: a real FILLED/CANCELED, or a
   synthesized reject on OrderNotFound), spending the shared per-order retry budget;
   only on exhaustion does `terminalize_missing` apply REJECTED (never acked) /
   CANCELED (was live).
4. **Materialize / update** — snapshot orders we don't hold materialize with
   `classify_origin` (framework prefix → `RECOVERED`, keeping the cid; else
   `ext:<venue_id>` → `EXTERNAL`). Known orders update under a freshness guard
   (`as_of > last_updated_at`, so a fresh live fill isn't clobbered); a pending marker
   is never wiped by a live venue status (the venue resolves the race; a terminal status
   *is* the resolution). When a snapshot raises `filled_quantity`, the raise is held per
   order as a **covered-quantity deficit** — the snapshot's position/balance legs
   already counted those executions. An arriving execution on any path (`DealEvent` or
   embedded fill) is suppressed up to the remaining deficit: fully covered records the
   trade id (dedup stays intact) and books nothing; one larger than the deficit books
   only the excess, fee pro-rated. The deficit is consumed as it suppresses, and the
   quantity suppressed since the last sync is absorbed from the next snapshot's raise
   before re-arming (the venue's filled figure already includes it). The rule is
   deliberately **clock-free** — quantity is the only signal, never a venue-vs-local
   time comparison (the balance tie-break below stays the sole sanctioned cross-domain
   one). Residual ambiguity: a genuinely-new post-snapshot execution that fits the
   deficit window is indistinguishable from a redelivered covered one and is suppressed
   in its place — quantity totals stay exact either way (total suppression is bounded by
   what snapshots counted), only per-deal price/fee attribution can shift.
5. **Positions — surgical** (`reconcile_position_from_snapshot`): the snapshot is
   authoritative for size/avg-price and venue-reported margin/mark **only**; locally
   accumulated accounting (`r_pnl`, commissions, funding) always survives. Never routes
   through `reset_by_position`. Positions deliberately keep **no** per-record freshness
   guard against WS pushes: pushes never write size, so the snapshot must remain the sole
   size-correction authority (a guard would block exactly the correction
   `position_drift` requests). Balances overwrite (identity-preserving) — except a
   currency whose push `as_of` is at/after the snapshot's `as_of`: the absolute push is
   the at-least-as-fresh figure (tie-break favors the push; snapshot `as_of` is the
   local fetch clock, push `as_of` is venue event time).
6. **Venue figures** — set when the snapshot carries any; absence keeps the previous
   capture (means "not observed", not "gone").

The manager logs the operator-facing line per applied snapshot (WARNING when orders were
terminated/materialized/missing, INFO for benign corrections; a figures-only diff stays
silent), and the PM emits the drift counters.

### Ticks & sweeps

`AccountManagerConfig` (defaults): in-flight sweep every 2s — poll an in-flight order
not heard from for 5s, give up after 5 retries; snapshot freshness check every 30s;
liveness check every 5s — WS unready for 30s → `reconnect()`; terminal retention 30s,
history ring 10k. Tunable per deployment via the `live.account_manager` YAML block
(1:1 onto the dataclass).

- Ticks register via `pm.schedule` in `set_processing_manager` — the AM is built before
  the PM in live (the AM↔PM construction cycle), so wiring is late and idempotent. They
  run on the processing thread, preserving the single-mutator invariant.
- **Give-up = synthetic event through the PM.** When the retry budget is exhausted, the
  manager synthesizes the reject the venue never sent (`giveup_event`: REJECTED for
  SUBMITTED, cancel/update-rejected for PENDING_* → reverts via `pre_pending`) and routes
  it through `pm.process_event` — the same path venue events take. The AM never calls
  the strategy directly. One bad order / connector error never aborts the rest of the
  sweep (see the in-code comment referencing this section).
- **Terminal eviction is not tick-dependent:** `AccountManager.apply` runs an
  opportunistic sweep when `terminal_retention` has elapsed since the last one — paper
  and backtest (`SimulatedAccountManager` registers no ticks) and the
  `inflight_check_interval_ms=0` edge all evict; live additionally sweeps from the
  in-flight tick so eviction stays prompt during event silence.
- Liveness: a failed/raising `reconnect()` keeps the unready timestamp, so the next tick
  retries instead of restarting the full threshold.

## Connectors (IConnector)

One protocol (`core/connector.py`): `submit_order`, `cancel_order` / `update_order` /
`request_order_status` (addressed by **either id** — the connector picks the id the venue
accepts), `request_snapshot`, `is_ws_ready` / `reconnect` / `connect` / `disconnect`,
`make_client_id`, `set_instrument_leverage` / `set_margin_mode`. Live
connectors resolve via the `ConnectorRegistry` (`@connector("name")`) — a new venue is
one IConnector + a registry entry. The connector is a **pure adapter**: its only outbound
surface is `send(event)` on the channel; it holds no AM/PM reference. Connectors stay
dumb — the reducer correlates deals to orders, absorbing split-stream stitching.

### Rejection boundary

Framework-side rejections (bad params, quote unavailable, below min-notional, read-only)
**raise synchronously** from submit/cancel/update so the caller sees them immediately.
Venue verdicts (insufficient funds, post-only crossing, rate-limit, auth, generic
exchange error) are **caught and emitted** as OrderRejected/Cancel/UpdateRejected events
on the channel — never raised — even when the venue returns them as a synchronous REST
error.

On the TradingManager side, a synchronous raise after the optimistic local transition is
rolled back, not stranded: `trade()` removes the phantom order and re-raises;
`cancel_order` / `update_order` route a synthetic Cancel/UpdateRejectedEvent through the
PM (same shape as the reconcile give-up — reverts PENDING_* via `pre_pending`, fires the
error-isolated STILL-ALIVE warning) and then re-raise to keep the synchronous-failure
contract.

### connect / reconnect contract

- **Case 1 — `connect()`:** start the WS account-event subscription and emit the initial
  snapshot, which seeds/reconciles AM state against venue truth at startup.
- **Case 2 — exchange recreation:** the old WS stream is dead; restart the subscription
  against the fresh exchange and pull a snapshot to resync the AM after the gap.

Read-only connectors keep the read surface alive (account events + snapshots flow);
write methods raise.

## Runtimes & wiring

- **Live:** runner builds connectors via the registry, the AM (with per-exchange base
  currencies + TCCs), then the context; `set_processing_manager` closes the AM↔PM cycle.
- **Paper + backtest:** `SimulatedAccountManager` — same apply path, no asyncio, no WS,
  no ticks (`set_processing_manager` is a no-op; the simulated connector feeds the OME
  through the `IMarketDataSink` protocol). Venue figures are never set → always derive.
- **Restoration/seeding:** the only sanctioned out-of-AM mutation paths are AM-level
  `seed_position` / `seed_balance` (restored state, initial paper capital; skip-unknown-
  exchange) and `adjust_balance` (simulated transfer legs). `AccountState` mutators are
  never called from outside the AM.
- **Unbounded channel:** `CtrlChannel` reverted to an unbounded queue — account events
  must never be dropped, and with no bound the typed-drop-protection question is moot.

## Divergences from PR #302

Sanctioned departures, detailed in the sections above:

1. `client_order_id` / `venue_order_id` naming (PR: `client_id`/`venue_id`).
2. `OrderStatus.is_pending` exists (classification only; revert still via `pre_pending`).
3. PENDING_CANCEL-on-accept guard (and snapshot-side pending guard).
4. Flat `AccountSnapshot` (`as_of` + flat venue figures on the event).
5. `get_position` materializes an empty Position for known instruments (consumers expect
   an object, not None) — a mutating read, which is what forced D3 (fit/warmup inline).
6. Terminal-but-retained deal booking + post-eviction suppression.
7. Unprefixed (public) `AccountState` mutators, framework-internal by contract.
8. Events in `core/events.py`, not in the `account_manager` package.
9. kw-only `snapshot_grace` on `reducer.apply`.
10. Funding payments as account events with bucketed dedup.
11. Per-order transition audit + per-status counters.
12. `AccountManagerConfig` + the three ticks + the apply-path eviction sweep.
13. `SimulatedAccountManager` for paper/backtest.
14. Restoration/seeding via AM-level `seed_*`/`adjust_balance`.
15. Unbounded channel instead of bounded-with-drop-protection.
16. Cancel/update synchronous-raise contract: synthetic reject through the PM + re-raise.
17. Fetch-before-terminalize for orders missing from snapshots.
18. Surgical position reconcile (local accounting always survives a snapshot).

## Deferred / open

- **`on_reconcile_complete(ctx, diff)`**: snapshot order-corrections stay callback-silent
  beyond the per-position `on_position_change` fan-out. Precondition: `ReconcileDiff`
  lives in `core/account_manager/reconcile.py` while `manager.py` imports
  `core.interfaces`, so declaring the signature on `IStrategy` requires relocating
  `ReconcileDiff` first.
- **Venue figures beyond Binance/OKX**: Binance and OKX snapshot legs extract venue
  figures; Bitfinex is documented derive-only (figures absent from `fetch_balance`);
  other venues derive (sound fallback).
- **Real multi-currency `conversion_rate`**: currently 1.0 — see the single-base-currency
  limitation under Capital/margin (COIN-M and non-base-quoted instruments out of scope).
- **Cancel+recreate update fallback**: `_update_via_cancel_recreate` raises NotSupported
  (fail-safe: rejects before canceling); all current target venues have `editOrder`, so
  it is unreachable until a venue without it is onboarded.
- **`set_instrument_leverage` / `set_margin_mode` delegation**: connector methods exist
  but are unwired from the strategy surface; the AM returns neutral per-instrument
  settings until venue-sourced leverage/margin-mode is held in `AccountState`.
- **`qubx_lighter` plugin port**: the external plugin targets the removed
  `IBroker`/`IAccountProcessor` registry API and must be ported to
  IConnector + ConnectorRegistry before any Lighter deployment.
- **Binance `priceMatch` support**: regressed vs main in the IBroker→IConnector cutover.
  The generic submit path forwards the `priceMatch` option but keeps `price` set, which
  Binance rejects — restoring it needs main's rule that `priceMatch` orders are sent with
  the price cleared.
- **Hyperliquid broker port**: main's Hyperliquid-specific broker behavior —
  slippage-priced market orders and order-response enrichment — was dropped in the
  cutover; HL live trading regresses to the generic ccxt path until reimplemented as a
  `CcxtConnector` subclass.
