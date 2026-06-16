# Dynamic-universe delisting resilience + on-demand fit trigger (qubx 1.6.1)

**Date:** 2026-06-16
**Status:** Design approved, pending spec review
**Repos:** Qubx (primary, 3 components) + QuantKit (1 component)
**Branch:** `feat/dynamic-universe-delisting-and-trigger-fit`

## Background

The 1.6.0 delisting fix wired the gone-instrument filter (`_drop_gone`: settle held +
exclude + alert) into `UniverseManager.set_universe`. That covers strategies whose
universe flows through `set_universe` (e.g. factors). But the **okx.aggregator**
(`quantkit.aggregation.strategy.AggregatorStrategy`) builds its universe dynamically:
`on_event` → `_update_universe` → `ctx.add_instruments(...)`. `add_instruments`
(`core/mixins/universe.py:125`) does **not** run `_drop_gone`, so a delisted
instrument (TON) was added unfiltered → it landed in a **bulk orderbook** batch →
ccxt `BadSymbol` → that whole orderbook stream stopped, leaving the batchmates
without quotes (signals stuck "no quote"), and TON sat as a stuck position.

This was observed on a real-money aggregator restart (qubx 1.6.0). It is a
**fix-gap**, not a 1.6.0 regression — the same degraded behavior existed on the
prior version (bulk-orderbook `BadSymbol` is non-fatal and predates 1.6.0).

Two further findings during design:
- `StrategyContext` inherits `IMarketManager` but does **not** delegate
  `is_instrument_listed` to its market manager — so `ctx.is_instrument_listed(...)`
  hits the interface stub and returns `None` (a latent footgun, and a blocker for
  the quantkit-side fix).
- A stale entry in the source Redis stream (the aggregator's `RedisTargetSource`
  reads the last `lookback_messages`, e.g. 100, via `xrevrange`) makes the
  aggregator re-add and re-target a gone instrument every cycle until it ages out.
  So the framework safety net alone leaves the aggregator churning; the aggregator
  must also skip gone instruments at the source.

Separately, there is no way to trigger a strategy `on_fit` on demand (only the cron
schedule + the one-time startup fit), which made post-deploy validation awkward
(issue #307). The user asked to bundle this into the same 1.6.1 update.

## Goals

- A delisted instrument added via `add_instruments` is never subscribed/held —
  it is settled (no trade) and excluded, exactly like the `set_universe` path.
- `ctx.is_instrument_listed(...)` works (delegates to the market manager).
- The aggregator does not add or target instruments whose market is gone, so a
  stale Redis entry can't keep re-driving it.
- A control action can trigger `on_fit` on demand, running safely on the strategy
  thread.
- Backtests and the `set_universe` path are unaffected; no change to live-trading
  behavior beyond the gone-instrument handling.

## Non-goals

- The account-poller live-subscription path ("Fix B": a *mid-run* delisting that
  arrives via exchange-reported positions rather than `add_instruments`). Not what
  bit the aggregator; tracked as a follow-up.
- A `dry_run` preview for `trigger_fit` (v1 is a plain trigger; preview can be a
  later addition).
- Any Redis stream cleanup — the aggregator filter makes stale entries harmless;
  they age out of the lookback window naturally.

## Design

Four components — three in Qubx (1.6.1), one in QuantKit. Components 1–2 are the
framework safety net; 3 is the source-side filter; 4 is the independent fit-trigger
feature.

### 1. Qubx — delegate `is_instrument_listed` on `StrategyContext` (prerequisite)

`StrategyContext` (`core/context.py`) explicitly delegates market-manager methods
(`ohlc`, `quote`, `query_instrument`, `time`, …) to `self._market_data_provider`
but is missing `is_instrument_listed`. Add the delegation next to the others:

```python
def is_instrument_listed(self, instrument: Instrument) -> bool:
    return self._market_data_provider.is_instrument_listed(instrument)
```

Without this, `ctx.is_instrument_listed(...)` returns the `IMarketManager` stub's
`None` (→ `not None` = `True` would mis-classify everything as gone in any caller).
Required for component 3; also fixes the latent stub bug.

### 2. Qubx — `add_instruments` runs the gone-filter (framework safety net)

`UniverseManager.add_instruments` (`core/mixins/universe.py:125`) currently adds
instruments without the gone-filter. Run the existing `_drop_gone(...)` at the top,
mirroring `set_universe` (where `_drop_gone` runs before `filter_delistings`):

```python
def add_instruments(self, instruments: list[Instrument]):
    instruments = self._drop_gone(instruments)   # settle held gone + exclude + alert
    to_add = list(set([instr for instr in instruments if instr not in self._instruments]))
    ...
```

`_drop_gone` already: detects gone via `_is_market_gone` (live `is_instrument_listed`
authoritative), settles a held gone position in place (`settle_position`, no trade)
via the `__do_remove_instruments` gone-branch, alerts, and returns the kept
(tradeable) instruments. Reusing it (DRY) means any strategy that dynamically adds a
delisted instrument is protected identically to `set_universe`.

We deliberately run only `_drop_gone` (already-gone markets) here, **not**
`filter_delistings` (future/scheduled delist dates). A still-listed instrument with
a future `delist_date` that a strategy explicitly adds remains addable/tradeable and
is handled by the existing scheduled-delist path (the 23:30 check / `set_universe`
filtering) — `add_instruments` should not silently refuse to add a still-tradeable
instrument.

### 3. QuantKit — `AggregatorStrategy` skips gone instruments at the source

In `AggregatorStrategy.on_event` (`src/quantkit/aggregation/strategy.py:154`), the
universe and targets are derived from the source targets. Filter out targets whose
instrument is **not listed** before `_update_universe` and `process_targets`:

```python
def on_event(self, ctx, event):
    all_targets = [t for src in self._sources.values() for t in src.get_targets(ctx)]
    # skip instruments whose market is gone (delisted/removed) — fail-open
    live_targets = [t for t in all_targets if ctx.is_instrument_listed(t.instrument)]
    dropped = {t.instrument.symbol for t in all_targets} - {t.instrument.symbol for t in live_targets}
    if dropped:
        logger.warning(f"[aggregator] skipping delisted/gone instruments: {sorted(dropped)}")
    instruments = set(t.instrument for t in live_targets)
    self._update_universe(ctx, instruments)
    signals = self._state_manager.process_targets(ctx, live_targets)
    ...
```

So the aggregator neither adds (`add_instruments`) nor generates a target/signal for
a gone instrument, regardless of stale Redis messages. Depends on component 1
(`ctx.is_instrument_listed`). Fail-open: if markets aren't loaded yet,
`is_instrument_listed` returns `True`, so nothing live is dropped.

### 4. Qubx — `trigger_fit` control action (issue #307)

Add a thin public wrapper on the context/processing mixin that runs `on_fit` once on
the strategy thread, reusing the existing public `delay(duration, callback)` (which
generates a unique event id, so it does **not** disturb the recurring `"fit"` cron
schedule):

```python
# core/mixins/processing.py (context)
def trigger_fit(self) -> None:
    """Run on_fit once, on demand. Marshals onto the strategy thread via the
    scheduler; does not affect the recurring fit schedule."""
    self.delay("1s", lambda c: c._handle_fit(None, "fit", (None, c.time())))
```

(`_handle_fit(None, "fit", (None, time))` is exactly the startup first-fit
invocation — `processing.py:398`.) Declare `trigger_fit` on `IStrategyContext`.

Add a control action in `control/builtin.py`, marked **⚠ dangerous** (a fit
recomputes targets and emits signals → it can trade):

```python
def _trigger_fit(ctx: IStrategyContext, **kwargs) -> ActionResult:
    ctx.trigger_fit()
    return ActionResult(success=True, data={"status": "fit scheduled"})
# registered as ActionDef(name="trigger_fit", category="strategy", read_only=False, dangerous=True)
```

Components 2 and 4 are independent; either could ship alone. They are bundled into
1.6.1 at the user's request.

## Error handling / fail-open

- `is_instrument_listed` is fail-open everywhere (markets empty/unloaded ⇒ `True`),
  so both the qubx filter (component 2) and the aggregator filter (component 3) only
  drop **affirmatively gone** instruments — never a live one.
- `trigger_fit` runs on the strategy thread via the existing scheduler/channel path;
  no concurrency with the data loop.

## Testing

- **Component 1:** `ctx.is_instrument_listed` delegates to the market manager
  (returns its value, not the stub `None`).
- **Component 2:** `UniverseManager.add_instruments([gone])` → gone instrument not
  added/subscribed; a held gone position is settled (qty 0, no trade); a listed
  instrument is added normally. (Mirror the existing `set_universe` gone test.)
- **Component 3 (quantkit):** `AggregatorStrategy.on_event` with a gone instrument
  among the source targets → no `add_instruments` call for it and no
  target/signal produced; a listed instrument passes through. `ctx` mocked with
  `is_instrument_listed` side-effect.
- **Component 4:** `ctx.trigger_fit()` schedules a one-off event that invokes
  `_handle_fit` (assert via the scheduler/`delay` path, on the strategy thread);
  the `trigger_fit` action is registered and dangerous-gated.

## Rollout

1. Qubx: implement components 1, 2, 4 on this branch → PR → `dev` → publish
   **qubx 1.6.1** (PyPI + Docker).
2. QuantKit: implement component 3 → new quantkit tag (e.g. `v2.4.3.devN`). QuantKit
   pins `qubx>=1.5.0`, so it resolves 1.6.1 on rebuild.
3. Rebuild `okx.aggregator` against the new quantkit tag (xrelease config
   `release.source.ref`), which pulls qubx 1.6.1 → upload new version.
4. Redeploy the aggregator bots **smoke-first** (`am-smoke` / `okx-prismatic-smoke`),
   then `okx-am-agg` / `okx-prismatic-agg` one at a time (real money) — keeping the
   proven base `image_tag`.
