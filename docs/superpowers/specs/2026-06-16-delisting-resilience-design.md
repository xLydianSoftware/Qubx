# Delisting resilience: don't crash or hold phantom positions on delisted instruments

**Date:** 2026-06-16
**Status:** Design approved, pending spec review
**Repo:** Qubx
**Branch:** `feat/delisting-resilience`

## Problem

A live `okx.factors` bot crash-looped on restart and could not start. The
warmup OHLC fetch raised:

```
ccxt.base.errors.BadSymbol: okx does not have market symbol TON/USDT:USDT
```

OKX had delisted the `TON-USDT-SWAP` perpetual (confirmed: the OKX public API
returns `code 51001 "Instrument ID ... doesn't exist"` and lists no TON swap at
all). The bot still held an open TON position. On restart:

1. `ctx.start()` force-adds every open position to the initial instrument set
   (`core/context.py:378-380`), so the held-but-delisted TON instrument is
   included in warmup.
2. `OhlcDataHandler.warmup()` (`connectors/ccxt/handlers/ohlc.py:80-126`) loops
   over instruments calling `fetch_ohlcv` with **no per-instrument error
   handling**, and `execute_all_warmups` does `asyncio.gather(...)` then
   re-raises (`connectors/ccxt/warmup_service.py:111-116`). So one delisted
   instrument aborts the entire warmup and the bot never starts.

Two distinct failures: (a) warmup is not resilient to a missing market, and
(b) the framework keeps a phantom position on an instrument that no longer
trades, with no path to reconcile it (the normal removal path trades to flat,
which is impossible on a gone market).

The existing `DelistingDetector` (`core/detectors/delisting.py`) only handles a
**future** `delist_date` within a look-ahead window. It does not cover
"the market is already gone," and nothing consults the live exchange.

## Goals

- A delisted instrument must never abort warmup or block startup.
- An instrument the exchange no longer lists must never be subscribed, warmed
  up, or kept as a tradeable position.
- A held position on a delisted (already cash-settled) instrument is forgotten
  and reconciled to exchange truth, with an operator alert.
- Detection is connector-agnostic at the Qubx core level; connectors supply
  only the raw "is this listed?" fact.
- Backtests are completely unaffected.

## Non-goals

- Future-delist handling — the existing `DelistingDetector` stays as-is for
  scheduled delistings.
- Broker-side listing checks (data side covers the crash and subscriptions).
- Retroactive cleanup of historical logs or PnL re-derivation (we trust the
  live account-balance sync).
- A global config kill-switch (see "Decisions" — rejected as over-engineering;
  fail-open + a conservative forget guard cover the risk).

## Design

Three layers; detection is expressed through a connector-agnostic interface and
orchestrated by the `UniverseManager`, with each connector supplying only the
raw listing fact.

### 1. Connector capability — `IDataProvider.is_instrument_listed`

```python
class IDataProvider:
    def is_instrument_listed(self, instrument: Instrument) -> bool:
        """True if the instrument currently exists on the exchange.
        Base default: True (unknown ⇒ assume listed; never wrongly drop)."""
        return True
```

- **ccxt** (`CcxtDataProvider`, `connectors/ccxt/data.py`): return `True` if
  `self._exchange_manager.exchange.markets` is empty/not loaded (fail-open
  against a load hiccup); otherwise
  `instrument_to_ccxt_symbol(instrument) in exchange.markets`.
- **Simulator / backtester:** inherits the base → always `True` → backtests
  unaffected.
- Lives on `IDataProvider` only. Brokers are out of scope (YAGNI).

### 2. Detection + reconciliation on `UniverseManager`

Two-tier, **ordered metadata → live** (metadata is cheap/offline; the live
check needs the connector). No standalone component, no callables threaded
through signatures — the manager reaches its own dependencies.

```python
def _is_delisted(self, instrument: Instrument) -> bool:
    # tier 1: delist_date metadata (from the InstrumentsLookup), via the
    #         existing detector — reuses its date math, no reimplementation
    if self._delisting_detector.is_delisting(instrument):
        return True
    # tier 2: live market listing, via the per-exchange data provider
    dp = self._data_provider_for(instrument)
    return dp is not None and not dp.is_instrument_listed(instrument)

# mirrors the existing filter_delistings idiom: returns the KEPT (tradeable)
# instruments and handles drops (forget + alert) internally
def _drop_delisted(self, instruments: list[Instrument]) -> list[Instrument]:
    delisted = [i for i in instruments if self._is_delisted(i)]
    for i in delisted:
        self._forget_if_held(i)        # see §3 (with conservative guard)
    if delisted:
        self._notify_delisted(delisted)
    kept = set(delisted)
    return [i for i in instruments if i not in kept]
```

- Add `DelistingDetector.is_delisting(instrument) -> bool` — the
  single-instrument form of its existing batch `detect_delistings` logic — so
  tier 1 reuses the detector rather than duplicating the date comparison.
- `_data_provider_for(instrument)` resolves the per-exchange `IDataProvider`
  the manager already has access to via the context (`ctx._data_providers`).
- `set_universe` calls `_drop_delisted(...)` exactly where it currently calls
  `self._delisting_detector.filter_delistings(...)` (`core/mixins/universe.py:78`),
  at the **top**, before `__do_add_instruments`/subscribe (`:90`, `:215`). This
  single placement covers startup (`ctx.start()` → `set_universe`) and every
  mid-run universe change.

### 3. Forget path — `IAccountProcessor.drop_position`

No public way exists to drop a position without trading; the normal removal
path (`UniverseManager.__do_remove_instruments` → `alter_positions` → trade to
flat) is impossible on a gone market. Add:

```python
def drop_position(self, instrument: Instrument) -> None:
    """Remove an instrument's position from tracking WITHOUT trading.
    For delisted/settled markets the exchange already cash-settled it."""
```

`_forget_if_held(instrument)`:
- if no open position → just ensure it's unsubscribed / out of the universe;
- if a position is held → **conservative guard**: only `drop_position` when we
  have *positive* delisting evidence, namely either (a) an explicit `delist_date`
  (tier 1), or (b) the instrument is absent from a **loaded, non-empty** market
  list (tier 2). The empty/unloaded case is already excluded by fail-open in
  `is_instrument_listed` (§ below), so a wholesale `load_markets` failure can
  never trigger a forget. When the guard is not satisfied we **skip the forget
  and emit a "needs manual review" alert** instead of discarding a position.
  - *Optional hardening (not required for v1):* protect against a *partial*
    market load by comparing the current market count against the connector's
    last-known-good count and skipping the forget on a large unexpected drop.

Capital stays correct because the live account-balance sync is authoritative
(the settled value is already reflected in the quote-currency balance); we only
discard the stale phantom.

### 4. Connector-side warmup guard (the actual crash fix)

The warmup crash happens in the runner's warmup pass, **before** the live
`UniverseManager` exists, so the manager cannot prevent it. The fix is local
and connector-appropriate: in `OhlcDataHandler.warmup()`
(`connectors/ccxt/handlers/ohlc.py`), per instrument:

- skip when `not self._data_provider.is_instrument_listed(instrument)`
  (log a warning, continue);
- wrap the fetch in `try/except` catching `BadSymbol`/`NotSupported` (and
  generally per-instrument failures) → log + continue, so no single instrument
  can abort `asyncio.gather`.

Other connectors mirror the same skip in their own warmup handlers.

### Responsibilities split

- **Warmup robustness** → connector guard (§4).
- **Universe + phantom-position lifecycle** → `UniverseManager` (§2–3),
  two-tier metadata→live detection.
- **No runner changes, no new component, no callables, no global toggle.**

### Fail-safe & alerting

- **Fail-open everywhere:** if a connector can't produce a confident listed-set
  (`is_instrument_listed` returns base `True`, or markets failed to load),
  nothing is dropped. A market-load hiccup must never nuke the universe.
- **Conservative forget (§3):** the only destructive step (forgetting a held
  position) is additionally gated on the market list looking healthy.
- **Alert:** on any drop,
  `ctx.notifier.notify_message("[<bot>] Dropped delisted instruments: TONUSDT (...)", metadata=...)`
  plus a WARN log. Silent drops are not acceptable for a prod trading bot.

## Why no restored-state pre-filter is needed (verified)

`set_universe` is not the only subscription path, so this was checked directly:

1. **Universe path** (`set_universe`): the delisting filter runs at the top
   (`core/mixins/universe.py:78`) before subscribe (`:90`/`:215`); a dropped
   instrument never reaches `to_add` → never subscribed.
2. **Account poller path** (`connectors/ccxt/account.py`): `_update_subscriptions`
   (`:312`) subscribes `_required_instruments`, which is populated **only from
   exchange-reported state** — `_update_balance` (`:367`) and `_update_positions`
   (`:379`). It is *not* populated by `attach_positions` (restore just fills the
   `_positions` dict). For a genuinely delisted instrument the exchange no
   longer reports it, so it never enters `_required_instruments`.

Restored positions (`QUBX_RESTORE`) flow into `ctx.start()` →
`_initial_instruments` (`core/context.py:378-380`) → `set_universe` (path 1,
filtered) and do **not** feed the account's `_required_instruments`. So the
restored phantom is filtered before subscription on the path that handles it,
and the account path never sees it because the exchange doesn't report it.

Residual case — metadata says delisted but the exchange *still* lists it (e.g. a
near-future `delist_date` during settlement overlap): the account poller could
briefly subscribe it, but (a) it isn't actually gone yet so a brief
subscription is harmless, (b) `set_universe` removal unsubscribes it, and (c) the
**live** OHLC watch handler already has per-instrument try/except
(`connectors/ccxt/handlers/ohlc.py:197+`). Only *warmup* lacked the guard, which
§4 fixes.

Conclusion: §2 (filter at top of `set_universe`) + §4 (warmup guard) cover
subscription without a restored-state pre-filter.

## Behavior walkthrough (the TON restart scenario)

1. Bot restarts with a held TON position; OKX no longer lists TON.
2. **Warmup**: §4 guard skips TON (not in `exchange.markets`), warms the other
   instruments, logs a warning, completes — no crash.
3. **Live start**: `ctx.start()` builds `_initial_instruments` including the
   restored TON position → `set_universe(...)`.
4. `_drop_delisted` runs at the top: tier 1 sees the `delist_date` you added
   (or, even without it, tier 2 sees TON missing from `exchange.markets`) → TON
   is delisted. `_forget_if_held(TON)`: market list is healthy → `drop_position`
   + unsubscribe. Notifier alert: "Dropped delisted instruments: TONUSDT".
5. TON is absent from `to_add` → never subscribed. The bot starts with the
   remaining live positions; capital reflects the OKX-settled balance.

## Testing

- **Unit**
  - `DelistingDetector.is_delisting` (past, within-window, future-beyond-window,
    no `delist_date`).
  - `UniverseManager._is_delisted` ordering: metadata hit short-circuits before
    the data-provider call; metadata miss falls through to the listing check;
    no data provider ⇒ fail-open (not delisted).
  - ccxt `is_instrument_listed` against a stubbed `exchange.markets`
    (present / absent / empty-fail-open).
  - `drop_position` removes the position without emitting a trade.
  - `_forget_if_held` conservative guard: empty/too-small market list ⇒ no
    forget, "needs review" alert instead.
- **Integration**
  - Warmup with one delisted instrument in the set → completes, warms the rest,
    logs the skip, no crash (direct regression for the TON incident).
  - Restart with a restored delisted position → instrument dropped, position
    forgotten, alert emitted, bot starts; instrument never subscribed.
- **Backtest guard**
  - A sim run still warms/trades everything (`is_instrument_listed` no-ops).

## Scope

**In:** `IDataProvider.is_instrument_listed` (+ ccxt impl), `UniverseManager`
detection/reconciliation (`_is_delisted`, `_drop_delisted`, `_forget_if_held`,
`_data_provider_for`, `_notify_delisted`), `DelistingDetector.is_delisting`,
`IAccountProcessor.drop_position`, ccxt warmup guard, alerting, tests.

**Out:** future-delist handling (unchanged), broker-side listing checks,
restored-state pre-filter (shown unnecessary), global kill-switch toggle, PnL
re-derivation, historical-log cleanup.

## Rollout

1. Implement + test on `feat/delisting-resilience`; PR to Qubx `dev`.
2. Push to `dev` → CI publishes a new qubx version (conventional commits drive
   the version bump).
3. Bump factors' qubx pin to the new version; rebuild the strategy release.
4. Update the `okx.factors` bot to the new release version and restart — TON is
   skipped in warmup and forgotten at start; the bot comes up clean without a
   manual state reset.
