# Boot Phase Design

**Date:** 2026-08-27
**Status:** Approved design, pending implementation plan
**Issues:** closes [#363](https://github.com/xLydianSoftware/Qubx/issues/363); documents [#388](https://github.com/xLydianSoftware/Qubx/issues/388)

## Background

Enabling a buffered tracker (quantkit `BufferedPositionsTracker`) in a live strategy broke
restarts: the warmup simulation always starts flat — only balances are seeded
(`src/qubx/backtester/runner.py:620`), never positions — and buffering makes targets
path-dependent, so from a flat start every target within the buffer band of zero collapses
to an explicit size-0 target. The warmup sim's end book is therefore a flat-start
*artifact*, not a prescription. `StateResolver.SYNC_STATE` (and the default `REDUCE_ONLY`)
then drove real live positions to that artifact — closing them at boot.

Investigating the incident surfaced four framework-level weaknesses in the boot path,
all in the same pipeline block (`src/qubx/core/mixins/processing.py:499-527`):

1. **No way to guarantee a live boot fit.** `is_on_fit_called` lives on the shared
   `StrategyState`; a warmup-sim fit sets it, so after warmup the live pipeline never
   forces a fit. Strategies that reconcile through their tracker on fit (the correct
   pattern for buffered strategies) had to hand-roll `ctx.trigger_fit()` in
   `on_warmup_finished` — and on venues where the warmup sim ran no fit (LIGHTER: no sim
   data), that hand-rolled fit double-fires with the `processing.py:521` fallback.
2. **State resolution is silently skipped when warmup is off** (#363). The resolver call
   is gated on `get_warmup_positions() or get_warmup_orders()` (`processing.py:510`), so
   disabling warmup — a plain config change — silently disables position-based state
   seeding (frab's pair-book resolver). The gate also *accidentally* protects
   `SYNC_STATE`/`REDUCE_ONLY` from flattening the live book on empty sim state.
3. **Boot failures are invisible.** `__invoke_on_fit` and `__invoke_on_warmup_finished`
   (`processing.py:746-771`) swallow strategy exceptions and latch their done-flags in
   `finally`. A failed boot fit means the bot silently trades (or holds) having never
   successfully fitted, until the next scheduled fit — possibly never.
4. **The account-sync gate leaks at boot.** `_is_ready()` (`processing.py:918-945`)
   falls through after `ACCOUNT_SYNC_TIMEOUT` with only a warning, so boot resolution and
   the boot fit can run against phantom-zero positions — the live-edge cousin of the
   buffering bug; it can double-open positions.

This design makes boot a first-class phase that fixes all four together.

## Terminology

Two unrelated things are both called "warmup"; this document always qualifies them:

- **Warmup simulation** (`live.warmup` config → `_run_warmup`,
  `src/qubx/utils/runner/runner.py:994`): an in-process backtest of *the same strategy
  object* over the trailing period, run before the live context starts. It warms the
  strategy's internal state (fitted models, indicators held on `self`, tracker state) and
  produces the "intended book" consumed by state resolution.
- **Subscription warmup** (`ISubscriptionManager.set_warmup({DataType.OHLC["1h"]: "30d"})`,
  usually called from `on_init`): historical OHLC backfill into the **live** data cache
  when instruments are subscribed, fetched from the exchange by the connector's warmup
  handlers (`src/qubx/connectors/ccxt/warmup_service.py`). The warmup sim does **not**
  populate the live cache — the sim context has its own; only the strategy object's state
  survives into live.

## Goals and invariants

The boot phase provides three guarantees:

1. **No boot action sees an unsynced account.** State resolution, tracker/gatherer
   restore, `on_warmup_finished`, and the boot fit run only after the initial venue
   account snapshot has applied (`is_synced()`), with no timeout fall-through. Live
   non-paper only; simulation and paper are unaffected.
2. **State resolution always runs at live boot** — warmup sim or not (#363). Empty warmup
   output means "hold + loud warning", never "flatten".
3. **`on_event` is never delivered before a *successful* fit** — and with
   `set_fit_on_start(True)`, that fit is guaranteed to have run in the live context
   against live positions.

Non-goals: changing simulation behavior (the sim pipeline must behave identically);
making resolvers tracker/buffer-aware (see Out of scope).

## Design

### 1. Boot state machine

A small `BootStateMachine` helper owned by the `ProcessingManager`, driven from
`_run_strategy_pipeline` on each pass until terminal. It replaces the flag-combination
checks at `processing.py:502-527` with one question ("is boot done?").

```
WAIT_READY → ON_START → RESOLVE → RESTORE → WARMUP_FINISHED → BOOT_FIT → TRADING
                                                                  ↓ (retries exhausted)
                                                               BLOCKED(reason)
```

- The existing `StrategyState` flags (`is_on_start_called`, `is_on_fit_called`,
  `is_on_warmup_finished_called`, `is_warmup_in_progress`) **stay** — the warmup-sim
  runner shares them — the machine wraps them as transition inputs; it does not replace
  them.
- `WAIT_READY` = data ready **and** strict account sync (section 2).
- `ON_START`, `RESOLVE`, `RESTORE`, `WARMUP_FINISHED` invoke the existing handlers
  (`_handle_start`, `_handle_state_resolution`, `_restore_tracker_and_gatherer_state`,
  `_handle_warmup_finished`) in the current order. Steps whose flag is already set (e.g.
  `on_start` already fired inside the warmup-sim context) are passed through.
- `BOOT_FIT` covers **every** first live fit: the `set_fit_on_start` reset, the
  no-warmup boot, and the warmup-sim-ran-no-fit fallback — one code path, which is where
  retry and health handling attach (section 5).
- `BLOCKED` is sticky for trading (no `on_event` delivery), carries a reason, emits a
  health gauge and periodic error logs, and can self-heal (section 5).
- In simulation the machine degenerates to today's behavior: readiness has no account
  requirement, `RESOLVE` is skipped (resolution is a live-boot concept), and hooks fire
  as they do now.

### 2. Strict account-sync gate

`_is_ready()` splits into two concerns:

- **Data readiness** keeps its current two-phase timeout with partial-data fall-through.
- **Account sync**: the timeout fall-through (`processing.py:937-944`) is **removed**.
  Its only consumers were boot actions, so nothing else changes behavior.
  `ACCOUNT_SYNC_TIMEOUT` is repurposed as the *alert* threshold: on crossing it, the
  machine emits `boot.account_sync_blocked = 1` and a warning, then keeps waiting.
  The moment the snapshot applies, boot proceeds and the gauge clears — self-healing, no
  restart required. The bot simply never trades blind.

### 3. Unconditional state resolution (#363)

- The `processing.py:510` gate is removed; `RESOLVE` always runs at live boot.
- The default resolver install (`REDUCE_ONLY`) moves out of `_run_warmup`
  (`src/qubx/utils/runner/runner.py:1019` — currently unreachable when warmup is off)
  into unconditional context setup.
- **Resolver protocol unchanged:** `(ctx, sim_positions, sim_orders, sim_active_targets)`.
  When no warmup sim ran, all three sim args are empty dicts. This is unambiguous: a
  warmup sim that ran always captures a `Position` for every sim instrument, flat ones
  included (`runner.py` capture filters on `quantity is not None`), so "all empty" ⇔
  "no warmup output". Existing custom resolvers (frab, factors) work unchanged.
- **Stock resolver guards:** `REDUCE_ONLY` and `SYNC_STATE` start with: all sim args
  empty → log a loud warning ("state resolver has no warmup output — holding live book")
  and return. `CLOSE_ALL` keeps its semantics (an explicit instruction independent of sim
  state). `NONE` is trivially unaffected.
- **New stock resolver `StateResolver.HOLD`:** cancels all open live orders, keeps all
  positions, emits no signals. This is the recommended partner of
  `set_fit_on_start(True)` for buffered/tracker-reconciling strategies (the factors
  recipe, generalized). `NONE` remains pure do-nothing.
- Cleanup riders: drop the dead `use_limit_order` computation in `SYNC_STATE` (the flag
  is read nowhere); document that `sim_orders` is unused by all stock resolvers (kept in
  the signature for custom resolvers).

### 4. Opt-in boot fit: `set_fit_on_start`

New `IStrategyInitializer` methods `set_fit_on_start(enabled: bool)` /
`get_fit_on_start() -> bool`, called from `IStrategy.on_init(initializer)`.

- **Semantics:** "the warmup-sim fit does not count — my first `on_fit` must run in the
  live context." Identical behavior on first start and restart (the framework does not
  distinguish them; a restart differs only in restored state existing).
- **Mechanism:** after `WARMUP_FINISHED` completes, if the flag is set and a warmup-sim
  fit had latched `is_on_fit_called`, the machine resets the flag; the existing
  first-fit path then fires exactly one live fit (through `_handle_fit`, threaded or
  inline per the executor mode).
- **No-warmup case:** a no-op — nothing ever set the flag, so the boot fit fires anyway
  (today's behavior, unchanged, for every strategy).
- **Contract:** opting in *replaces* the `ctx.trigger_fit()`-in-`on_warmup_finished`
  boilerplate. Calling both produces a double fit (documented, not defended against).
- Explicitly **opt-in**: strategies that must fit only at deliberately scheduled times
  are untouched.

### 5. Boot failure handling

- **Latch only on success (live only).** Inline path: `__invoke_on_fit` sets
  `is_on_fit_called` only when `on_fit` returned without raising. Threaded path:
  `FitCommitData` gains an `error` field set on the fit thread; `_handle_fit_commit`
  (`processing.py:1288-1318`) latches only when `error is None`. Scheduled (non-boot)
  fits see no behavior change — the flag is already `True` by then. **Simulation keeps
  the latch-in-`finally`** (the retry machinery is live-only; without the latch a raising
  sim fit would retry on every tick and change backtest behavior).
- **Boot fit retry:** on failure, the machine schedules a retry — **3 attempts total
  (initial + 2 retries), 60s apart**, deadline-driven from the pipeline (a stored
  next-retry timestamp checked on each pass; no scheduler dependency). Each attempt
  increments `boot.fit_attempts`.
- **Exhausted → `BLOCKED("boot fit failed")`:** `on_event` is never delivered, gauge
  `boot.fit_failed = 1`, periodic error logs. Rationale: for a scheduled fit,
  latch-and-continue is defensible (trading continues on the previous fit's state); for
  the boot fit there is no previous state — continuing means running a
  never-successfully-fitted strategy.
- **Self-heal:** any later *successful* fit (the recurring schedule, or a manual
  `trigger_fit`) releases `BLOCKED(fit)` and clears the gauge — consistent with the
  sync-gate philosophy: block, alert, recover without a restart.
- **`on_warmup_finished` failure:** latch as today (the hook is not guaranteed
  idempotent — no auto-retry), plus gauge `boot.warmup_finished_failed = 1` and an error
  log. Boot continues to `BOOT_FIT`.

### 6. Observability surface

Emitted via the existing `IHealthMonitor.record_gauge`:

| Gauge | Meaning |
|---|---|
| `boot.state` | current phase as a number (WAIT_READY=0 … TRADING=6, BLOCKED=-1) |
| `boot.account_sync_blocked` | 1 while boot is held waiting for the account snapshot past the alert threshold |
| `boot.fit_attempts` | boot-fit attempt counter |
| `boot.fit_failed` | 1 when boot-fit retries are exhausted (cleared on self-heal) |
| `boot.warmup_finished_failed` | 1 when `on_warmup_finished` raised |

Ops alerting hangs off these; the exporters already ship gauges.

## Boot sequencing (reference)

### Without warmup sim

```
runner main thread                        ProcessorThread (boot state machine)
──────────────────                        ────────────────────────────────────
read RestoredState (--restore:
  positions/signals/targets from the
  previous run's logs)
create ctx (restored positions
  seeded into account manager)
ctx.start()
 ├ connect live connectors
 └ initial subscribe commit
    └ WarmupThread: OHLC backfill
       then swap → live streams on ──►  first live tick per instrument
                                        ┌ WAIT_READY   data ready + account synced (strict)
                                        ├ ON_START     on_start fires (live ctx)
                                        ├ RESOLVE      resolver runs, sim args EMPTY
                                        │              → stock: hold + loud warning
                                        │              → custom: e.g. seed from ctx.get_positions()
                                        ├ RESTORE      tracker/gatherer from RestoredState
                                        ├ WARMUP_FIN   on_warmup_finished fires
                                        ├ BOOT_FIT     on_fit fires — ALWAYS (flag never set)
                                        └ TRADING      on_event starts flowing
```

### With warmup sim

```
runner main thread                        ProcessorThread (boot state machine)
──────────────────
read RestoredState
create ctx (restored pos → account)
_run_warmup (blocking, in-process):
 ├ start = start_time_finder(restored) − warmup period
 ├ seed REAL capital from venue snapshot
 ├ backtest same strategy object on sim ctx:
 │   on_start, on_fit(s), on_event(s) fire HERE
 │   (sim book starts FLAT — only balances seeded)
 └ capture sim end state →
     ctx.set_warmup_positions/orders/active_targets
ctx.start()
 └ initial subscribe → WarmupThread backfill
    → swap → live ticks            ──►  ┌ WAIT_READY   data ready + account synced (strict)
                                        ├ ON_START     SKIPPED — already fired in sim ctx (#388)
                                        ├ RESOLVE      resolver vs sim end state (non-empty:
                                        │              sim captures a Position per instrument)
                                        ├ RESTORE      tracker/gatherer from RestoredState
                                        ├ WARMUP_FIN   on_warmup_finished fires (live ctx)
                                        ├ BOOT_FIT     fires IFF fit_on_start OR sim ran no fit;
                                        │              otherwise skipped
                                        └ TRADING
```

### When `on_fit` runs — all cases

| Scenario | Fit during warmup sim | Live boot fit | First `on_event` sees a live-fitted strategy? |
|---|---|---|---|
| No warmup sim | — | always (flag never set) | yes |
| Warmup sim, knob off, sim fit ran | yes, in sim ctx | no — waits for the next scheduled fit | no — fit state came from the sim |
| Warmup sim, knob off, sim ran no fit (e.g. LIGHTER) | no | yes (existing fallback, now the BOOT_FIT path) | yes |
| Warmup sim + `set_fit_on_start(True)` | yes, in sim ctx | exactly one, after `on_warmup_finished` | yes |

### Subscription warmup — when, and who waits

| When | Trigger | Where the fetch runs | Who waits on it |
|---|---|---|---|
| Live boot (warmup sim on or off — identical) | initial subscribe commit inside `ctx.start()` | WarmupThread (`src/qubx/core/mixins/subscription.py:114-152`); ccxt handlers fetch with bounded concurrency + timeout (`warmup_service.py:117-142`) | Nobody blocks directly; the ordering is transitive: swap applies only after history lands → live streams start only after the swap → data-ready needs one live tick per instrument → the machine sits in `WAIT_READY`. |
| Universe change from a fit | subscription commit during FitCommit replay | WarmupThread (deferred) | Only the added instruments — they go live after their backfill; trading on the existing universe continues. |
| During the warmup sim | n/a | the sim reads history synchronously from the warmup storage (`live.warmup.data`) | the sim itself. |

**Guarantee:** `on_start`, `RESOLVE`, and the boot fit always run after the initial
subscription warmup has completed *or degraded by timeout* — never before it. Degraded
paths: (a) backfill timeout cancels the fetch and applies the swap with partial history
(`warmup_service.py:135-142`); (b) data-ready's own two-phase timeout can fall through
with the subset of instruments that ticked; (c) with no subscription warmup configured
the swap is synchronous and the boot sequence proceeds off the first natural ticks with
an empty OHLC cache.

### RESOLVE vs RESTORE

- **RESOLVE** acts on the **venue** and may trade: the resolver compares the live book
  (`ctx.get_positions()` / `get_orders()`) against the strategy's intended book (warmup
  sim end state, or nothing) and emits `InitializingSignal`s → real orders.
- **RESTORE** re-seeds **in-memory bookkeeping** and never trades: the tracker receives
  the previous run's persisted signals, the gatherer the latest persisted targets (from
  `RestoredState`), and targets are re-persisted so the restore chain survives the next
  restart.

## Out of scope

- **Resolver ↔ tracker integration.** Resolver signals route through the internal
  `_InitializationStageTracker` (`src/qubx/trackers/riskctrl.py:988`), bypassing the
  strategy's tracker — resolvers structurally cannot be band/buffer-aware. The supported
  answer for buffered strategies is `HOLD` + `set_fit_on_start(True)`: let the first live
  fit reconcile *through* the tracker. Deeper integration is a separate design if ever
  needed.
- **Warmup sim position seeding.** Seeding the warmup sim with restored/live positions
  (instead of a flat start) would remove the artifact at its source but changes
  backtester semantics; not attempted here.
- **#388's breaker note** (a per-instrument exception counts toward the global
  ten-failure breaker) — separate issue.

## Documentation deliverable (#388)

A "Strategy boot lifecycle" docs page covering: the phase order and diagrams above; the
`on_start`-fires-in-warmup-context asymmetry and its consequence (per-instrument state
must be built lazily or in `on_warmup_finished`, never only in `on_start`); the resolver
contract (empty sim args ⇔ no warmup output); `set_fit_on_start` usage; and a
recommended-patterns table:

| Strategy style | Resolver | `set_fit_on_start` |
|---|---|---|
| Buffered / tracker-reconciling | `HOLD` | `True` |
| Target-state (sim targets are prescriptive) | `SYNC_STATE` | optional |
| Conservative default | `REDUCE_ONLY` | optional |
| Fully custom boot seeding | custom resolver | as needed |

This closes #388's documentation ask.

## Testing

Unit tests (`tests/qubx/core/`), fake time throughout:

- State-machine transitions, including `BLOCKED` entry and both self-heal paths
  (late account snapshot; later successful fit).
- Resolution runs with warmup off; stock-resolver empty guards (hold + warn, no orders);
  `CLOSE_ALL` unaffected by empty args; `HOLD` cancels orders and keeps positions.
- Default resolver installed without warmup config.
- `set_fit_on_start`: exactly one live fit with warmup; no double fit in the
  sim-ran-no-fit case; no-op without warmup; untouched when not opted in.
- Strict sync gate: boot holds past `ACCOUNT_SYNC_TIMEOUT`, gauge emitted, proceeds on
  late snapshot.
- Boot-fit failure: latch-only-on-success in both executor modes (inline and threaded via
  `FitCommitData.error`); 3×60s retry cadence; `BLOCKED` after exhaustion; no `on_event`
  while blocked; release on later successful fit.
- `on_warmup_finished` failure: latches, gauge emitted, boot continues.
- Simulation parity: sim pipeline behavior unchanged (regression suite).

## Rollout

One Qubx release, no feature flags — the removed behaviors are the bug. The changelog
must call out three behavior changes for existing bots:

1. **No-warmup bots now run boot resolution.** Default (`REDUCE_ONLY` + empty guard) →
   hold + warning, no trading action. Bots with a registered custom resolver: it now
   actually fires at boot — this is the #363 fix.
2. **A failed boot fit now blocks trading** (retry → `BLOCKED` + health event) instead of
   silently trading unfitted.
3. **An unsynced account now blocks boot** (+ health event) instead of proceeding after
   the timeout with phantom-zero positions.

Downstream: factors and frab can later simplify to stock `HOLD` + `set_fit_on_start(True)`.

## Follow-ups after this spec

- Post the design (or a summary) as the "Solution" comment on #363.
- Open issues for: opt-in boot fit (`set_fit_on_start`), boot failure handling +
  observability, strict account-sync gate — linking back to this spec.
- Implementation plan via the standard planning flow.
