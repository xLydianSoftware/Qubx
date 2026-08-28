# Strategy Boot Lifecycle

Every live boot — first start or restart — runs through a fixed sequence before the
strategy is allowed to trade: the account has to sync, `on_start` has to fire, the live
book has to be reconciled against whatever the strategy intended, and the strategy has to
fit at least once. This page describes that sequence, what a strategy author needs to do
to cooperate with it, and how to read its health signals.

It does not apply to backtesting: simulation has no account to sync and no venue book to
reconcile, so most of what follows is live-only (called out where it matters).

## Two Kinds of Warmup

Qubx uses the word "warmup" for two unrelated mechanisms. This page always qualifies
which one it means.

**Warmup simulation** (`initializer.set_warmup("14d")` in `on_init`, or the `live.warmup`
config key): before the live context starts, an in-process backtest runs *the same
strategy object* over the trailing period. It warms the strategy's own state — fitted
models, indicators held on `self`, tracker state — and its final book (positions, open
orders, active targets) becomes the "intended state" that the boot state resolver later
compares the live account against. The warmup sim always starts from a flat, balance-only
book (`src/qubx/backtester/runner.py`); it does not touch the live data cache — only the
strategy object's in-memory state survives into the live context.

**Subscription warmup** (`ISubscriptionManager.set_warmup({DataType.OHLC["1h"]: "30d"})`,
usually called from `on_init`): a historical OHLC backfill into the *live* data cache,
fetched by the connector when instruments are subscribed. It is unrelated to the warmup
simulation and runs on every live boot — warmup sim on or off. It gates when live
streaming starts: the swap from history to live ticks only applies once the backfill
lands (or times out), and boot's `WAIT_READY` phase in turn waits for a live tick before
proceeding — so a slow backfill delays boot, it never skips it.

## Boot Sequence

Boot is driven by a state machine with one question per pass: "is boot done?" Its phases
run in order, live or in the warmup sim:

```
WAIT_READY → ON_START → RESOLVE → RESTORE → WARMUP_FINISHED → BOOT_FIT → TRADING
                                                                  ↓ (retries exhausted)
                                                               BLOCKED(reason)
```

`WAIT_READY` requires market data readiness *and*, live only, a synced account. `BLOCKED`
is a sticky failure state — no `on_event` is delivered while blocked — reached only if the
boot fit exhausts its retries (see [Boot Health](#boot-health)). In simulation the account
requirement drops out, `RESOLVE` is skipped entirely (state resolution is a live-boot
concept — there is no venue book to reconcile against), and the rest fires as it always
has.

The concrete sequence differs depending on whether a warmup simulation ran, because the
warmup sim fires `on_start` (and possibly `on_fit`) itself, before the live context even
exists.

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

Two phases are worth calling out explicitly since the diagrams compress them:

- **RESOLVE** acts on the venue and may trade: the resolver compares the live book
  (`ctx.get_positions()` / `ctx.get_orders()`) against the strategy's intended book and
  can emit signals that become real orders.
- **RESTORE** only re-seeds in-memory bookkeeping and never trades: the tracker replays
  the previous run's persisted signals, the gatherer replays the latest persisted target,
  and that target is re-persisted so a subsequent `--restore` still finds something to
  restore from.

## When `on_fit` Runs

Whether the boot fit (the `BOOT_FIT` phase) actually invokes `on_fit` depends on whether a
warmup sim ran, whether it fitted, and whether the strategy opted into
[`set_fit_on_start`](#set_fit_on_start):

| Scenario | Fit during warmup sim | Live boot fit | First `on_event` sees a live-fitted strategy? |
|---|---|---|---|
| No warmup sim | — | always (flag never set) | yes |
| Warmup sim, knob off, sim fit ran | yes, in sim ctx | no — waits for the next scheduled fit | no — fit state came from the sim |
| Warmup sim, knob off, sim ran no fit (e.g. no warmup data for the venue) | no | yes (fallback) | yes |
| Warmup sim + `set_fit_on_start(True)` | yes, in sim ctx | exactly one, after `on_warmup_finished` | yes |

This table is a consequence of one invariant, and it always holds: **`on_event` is never
delivered before a successful fit.** Boot sits in `BOOT_FIT` — no events, no trading — until
`on_fit` returns without raising for the first time in the live context (or, in row 2,
until the machine recognizes the sim's fit as already having satisfied that requirement).

## `on_start` Fires in the Warmup Context

When a warmup sim runs, `on_start` is **not** re-invoked live — the `ON_START` phase is
skipped because the flag is already set from the sim (see the "With warmup sim" diagram
above). That means the `on_start` a strategy actually observes ran inside the sim, against
the sim's universe, before the live context — and its real venue positions — existed at
all.

This has two concrete consequences for strategy code:

- **Any per-instrument state a strategy initializes in `on_start` will not exist for
  instruments that only become relevant later** (a restored position, an instrument the
  live universe includes that the sim window didn't). Build such state lazily on first
  use, or (re)build it in `on_warmup_finished`, which always fires in the live context
  before `BOOT_FIT`. Never rely on `on_start` alone to have initialized it.
- **`ctx.instruments` at live start can already contain instruments the strategy never
  explicitly selected** — held positions from a previous run flow into the context ahead
  of any strategy universe call. Code that assumes `ctx.instruments` is exactly what
  `on_start` selected can be surprised by extra members.

The safe pattern: treat `on_start` as "runs once, maybe in a sim, maybe before real state
exists" and push anything that must reflect the live account into `on_warmup_finished` or
lazy per-instrument initialization.

## State Resolvers

The state resolver is the mechanism that reconciles the live account with the strategy's
intended book at boot. Its contract:

- **It runs at every live boot** — warmup sim or not. This is unconditional; there is no
  config flag that silently disables it. If no custom resolver is registered, the default
  is `StateResolver.REDUCE_ONLY`.
- **All-empty resolver arguments mean "no warmup output."** The resolver signature is
  `(ctx, sim_positions, sim_orders, sim_active_targets)`. When a warmup sim ran, it always
  captures a `Position` per sim instrument — flat ones included — so all three arguments
  being empty is unambiguous: no warmup sim ran (or it produced nothing to compare
  against). Custom resolvers can rely on this to distinguish "nothing to seed from" from
  "sim ended flat."
- **The stock resolvers that steer toward sim state guard the empty case.**
  `REDUCE_ONLY` and `SYNC_STATE` check for all-empty arguments first; if empty, they log a
  loud warning ("State resolver received no warmup output — holding the live book
  as-is.") and return without touching the account. This is what makes it safe to run
  with warmup disabled: nothing gets silently flattened. `CLOSE_ALL` is unaffected —
  closing everything is an explicit instruction independent of sim state. `NONE` does
  nothing, always.
- **`StateResolver.HOLD`** cancels every open live order, leaves all live positions
  untouched, and emits no signals. It is the recommended partner for strategies whose
  tracker (e.g. a buffered/banded position tracker) reconciles state itself — instead of
  letting the resolver drive positions toward a flat-start sim artifact, `HOLD` leaves the
  book alone and lets the first live fit reconcile through the tracker.

Custom resolvers (frab's pair-book resolver, factors' resolver) receive the same
arguments and are unaffected by any of this — they now run at every live boot with no
empty-guard applied. Before this release, a registered custom resolver was silently
skipped at boot whenever warmup was disabled (#363); that gap is what this release
closes.

### Recommended patterns

| Strategy style | Resolver | `set_fit_on_start` |
|---|---|---|
| Buffered / tracker-reconciling | `HOLD` | `True` |
| Target-state (sim targets are prescriptive) | `SYNC_STATE` | optional |
| Conservative default | `REDUCE_ONLY` | optional |
| Fully custom boot seeding | custom resolver | as needed |

Register a resolver from `on_init`:

```python
def on_init(self, initializer: IStrategyInitializer) -> None:
    initializer.set_state_resolver(StateResolver.HOLD)
    initializer.set_fit_on_start(True)
```

## `set_fit_on_start`

`initializer.set_fit_on_start(True)`, called from `on_init`, tells boot: "the warmup-sim
fit doesn't count — my first `on_fit` must run in the live context, against live
positions."

- **What it guarantees:** when a warmup sim ran and fitted, boot forces exactly one
  additional live fit after `on_warmup_finished` completes, before `TRADING`. Without the
  flag, that fit is skipped (row 2 of the [`on_fit` table](#when-on_fit-runs) above) and
  the strategy trades on state the sim computed against a flat, balance-only book.
- **It replaces `ctx.trigger_fit()` boilerplate.** Strategies that needed a guaranteed
  live fit used to hand-roll a call to `ctx.trigger_fit()` in `on_warmup_finished`.
  Opting into `set_fit_on_start(True)` is the supported way to get the same guarantee;
  calling both produces a double fit — this is documented, not defended against, so pick
  one.
- **No-op without a warmup sim.** If no warmup sim runs, nothing ever sets the fit flag,
  so the boot fit fires unconditionally regardless of this setting (row 1 of the table) —
  today's behavior, unchanged.
- **Opt-in only.** Strategies that must fit exclusively on a deliberate schedule are
  untouched if they never call this.

## Boot Health

Boot emits gauges through the existing health-monitor pipeline (same exporters as other
`stg.*` metrics):

| Gauge | Meaning |
|---|---|
| `boot.state` | current phase as a number: `WAIT_READY`=0, `ON_START`=1, `RESOLVE`=2, `RESTORE`=3, `WARMUP_FINISHED`=4, `BOOT_FIT`=5, `TRADING`=6, `BLOCKED`=-1 |
| `boot.account_sync_blocked` | 1 while boot is held in `WAIT_READY`, past the alert threshold, waiting for the initial account snapshot |
| `boot.fit_attempts` | boot-fit attempt counter, incremented on every attempt |
| `boot.fit_failed` | 1 when boot-fit retries are exhausted (cleared on self-heal) |
| `boot.warmup_finished_failed` | 1 when `on_warmup_finished` raised |

Two failure paths hold boot rather than let it proceed blind, and both recover without a
restart — but they are independent mechanisms, not the same one:

- **Unsynced account.** `WAIT_READY` never falls through on a timeout: if the initial
  venue account snapshot hasn't applied, boot simply waits. `ACCOUNT_SYNC_TIMEOUT` (15s)
  is only an *alert* threshold — crossing it emits `boot.account_sync_blocked = 1` and a
  warning, but boot keeps waiting. The moment the snapshot applies, boot proceeds and the
  gauge clears on its own.
- **Boot fit failure → `BLOCKED`.** If `on_fit` raises during `BOOT_FIT`, boot retries: 3
  attempts total (the initial attempt plus 2 retries), 60 seconds apart, each incrementing
  `boot.fit_attempts`. If all 3 fail, boot enters `BLOCKED("boot fit failed")`:
  `boot.fit_failed = 1`, periodic error logs, and no `on_event` is delivered — the
  strategy holds an unreconciled book rather than trade having never successfully fitted.
  `BLOCKED` self-heals: **any** later successful fit — the normal recurring schedule, or a
  manual `ctx.trigger_fit()` — clears it and releases boot into `TRADING`. This works
  because fit outcomes only reach the boot machine while it is still in `BOOT_FIT` or
  `BLOCKED`; once boot reaches `TRADING` normally, ordinary scheduled-fit failures follow
  their usual latch-and-continue behavior and do not re-enter the boot machine at all.

`on_warmup_finished` failure is handled separately and does not block: the hook is not
guaranteed idempotent, so it is latched as called either way (no auto-retry), but a raise
emits `boot.warmup_finished_failed = 1` and an error log, and boot continues on to
`BOOT_FIT`.
