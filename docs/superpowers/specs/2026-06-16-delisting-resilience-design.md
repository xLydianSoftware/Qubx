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

## Conceptual model: two distinct states (do not conflate)

The decision of what to do with a held position keys on **tradeability — is the
market physically there right now? — not on `delist_date`.**

**A. Delisting *scheduled* (future `delist_date`, market still listed).** The
instrument is still physically tradeable; a position on it is real and
legitimate. The correct exit is a **trade to flat**, which the existing
`DelistingDetector` + `set_universe` removal + the 23:30 scheduled check already
perform (remove from universe, `alter_positions` → close via order). This path
is **unchanged** by this design, and such a position is **never forgotten** —
forgetting it would silently discard a real, still-tradeable holding.

**B. Delisting *already happened* (market gone from the exchange).** You
physically cannot hold a tradeable position anymore — the exchange has
cash-settled it and there is no market to trade against. Only here is "forget"
(`drop_position`, no trade) correct, because a closing order is impossible. This
is the TON case and the new behavior in this design.

| State | Market listed? | `delist_date` | Position action |
|---|---|---|---|
| Scheduled delist | yes | future | existing path: remove from universe, **close via trade** |
| Settlement overlap | yes | past | prefer close via trade; **don't forget** (alert if stuck) |
| Gone | **no** | any / past | skip warmup/subscribe, **forget** (`drop_position`) |

The authoritative signal for state B is the **live market-listing absence**
(`is_instrument_listed() == False`). A past `delist_date` is only a cheap
first-pass hint to skip warmup/subscription; it never, on its own, authorizes
the destructive forget (see §3).

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
- A global config kill-switch — rejected as over-engineering; fail-open plus a
  live-confirmed forget (§3) cover the false-positive risk.

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

### 2. "Market gone" detection + reconciliation on `UniverseManager`

This handles **only state B (market gone)**. State A (future/scheduled
delisting) is left entirely to the existing `filter_delistings` path, which
still runs and still closes positions via trade.

The authoritative "gone" signal is the **live market-listing absence**. A
**past** `delist_date` is a cheap offline hint usable for *exclusion* (skip
warmup/subscription), but a **future** `delist_date` is explicitly *not* gone.
No standalone component, no callables threaded through signatures — the manager
reaches its own dependencies.

```python
def _is_market_gone(self, instrument: Instrument) -> bool:
    """State B only: the market no longer exists / is untradeable.
    A future delist_date (state A) is NOT gone."""
    dp = self._data_provider_for(instrument)
    if dp is not None:
        return not dp.is_instrument_listed(instrument)   # authoritative
    # No live signal (e.g. connector can't tell): fall back to metadata,
    # but ONLY a past delist_date counts as gone — never a future one.
    d = instrument.delist_date
    return d is not None and to_timestamp(d) <= self._time_provider.time()

# Excludes gone instruments and forgets held positions on them (§3 guard).
# Returns the KEPT (still-tradeable) instruments. Runs IN ADDITION to the
# existing filter_delistings (state A), not instead of it.
def _drop_gone(self, instruments: list[Instrument]) -> list[Instrument]:
    gone = [i for i in instruments if self._is_market_gone(i)]
    for i in gone:
        self._forget_if_held(i)        # see §3 (live-confirmed forget only)
    if gone:
        self._notify_gone(gone)
    gone_set = set(gone)
    return [i for i in instruments if i not in gone_set]
```

- `_data_provider_for(instrument)` resolves the per-exchange `IDataProvider`
  the manager already has access to via the context (`ctx._data_providers`).
- `set_universe` calls `_drop_gone(...)` at the **top**, alongside the existing
  `self._delisting_detector.filter_delistings(...)` (`core/mixins/universe.py:78`),
  before `__do_add_instruments`/subscribe (`:90`, `:215`). `filter_delistings`
  keeps handling state A (remove → close via trade); `_drop_gone` handles state
  B (exclude → forget). One placement covers startup (`ctx.start()` →
  `set_universe`) and every mid-run universe change.
- A held gone instrument is forgotten by `_drop_gone` even when it was never in
  the prior universe (the restored-position startup case), since `_forget_if_held`
  acts on the held position directly, not on `prev_set` membership.
- **Removal path also branches on "gone":** `__do_remove_instruments`
  (`core/mixins/universe.py:159`) currently closes positions via
  `alter_positions` → trade. Add a guard: if `_is_market_gone(instr)`, forget
  (`drop_position`) instead of trading — so the scheduled 23:30 delisting check
  and any other removal can never attempt a trade on a vanished market.

The existing `DelistingDetector` is **unchanged** — no `is_delisting` predicate
is added; state B never routes through it.

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
- if a position is held → **forget only on a live-confirmed gone market.**
  Concretely, `drop_position` is allowed only when the per-exchange data
  provider exists and reports the instrument **absent from a loaded, non-empty
  market list** (`is_instrument_listed() == False` with markets loaded). In any
  other case — a past `delist_date` but the market still lists it (settlement
  overlap, where a reduce-only close may still be possible), no live signal
  available, or markets empty/unloaded (fail-open) — we **do not forget**; we
  **emit a "needs manual review" alert** and leave the position for the existing
  close-via-trade path or an operator. We never forget a position whose
  delisting has not actually happened.
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
- **State A (scheduled delist, still listed)** → existing `filter_delistings` +
  removal path: remove from universe, **close via trade**. Unchanged.
- **State B (market gone) + phantom-position lifecycle** → `UniverseManager`
  (§2–3): exclude + forget (live-confirmed).
- **No runner changes, no new component, no callables, no global toggle.**

### Fail-safe & alerting

- **Fail-open everywhere:** if a connector can't produce a confident listed-set
  (`is_instrument_listed` returns base `True`, or markets failed to load),
  nothing is treated as gone. A market-load hiccup must never nuke the universe.
- **Forget only on a live-confirmed gone market (§3):** the only destructive
  step (forgetting a held position) requires the live listing signal; metadata
  alone, settlement overlap, or no signal ⇒ no forget, manual-review alert.
- **Alert:** on any gone-drop,
  `ctx.notifier.notify_message("[<bot>] Dropped delisted (gone) instruments: TONUSDT (...)", metadata=...)`
  plus a WARN log; on a guarded skip, a "needs manual review" alert. Silent
  drops are not acceptable for a prod trading bot.

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
4. `_drop_gone` runs at the top: `_is_market_gone(TON)` is true because the live
   data provider reports TON absent from `exchange.markets` (the `delist_date`
   you added is corroborating but not required, and is not what authorizes the
   forget). `_forget_if_held(TON)`: market is live-confirmed gone and markets are
   loaded/non-empty → `drop_position` + unsubscribe. Notifier alert: "Dropped
   delisted (gone) instruments: TONUSDT".
5. TON is absent from `to_add` → never subscribed. The bot starts with the
   remaining live positions; capital reflects the OKX-settled balance.

Contrast — a *scheduled* (state A) delist, e.g. TON still listed with a
`delist_date` 2 days out: `_is_market_gone` is **false** (still listed), so TON
is never forgotten. The existing `filter_delistings`/removal path closes it via
a normal trade when its window arrives.

## Testing

- **Unit**
  - `UniverseManager._is_market_gone`: live signal authoritative (listed ⇒
    not gone even with a past `delist_date`; not-listed ⇒ gone); no data
    provider ⇒ fall back to metadata, where a **past** `delist_date` ⇒ gone but
    a **future** `delist_date` ⇒ not gone; fail-open (markets empty) ⇒ not gone.
  - **State A is never forgotten:** an instrument still listed with a future
    `delist_date` is not gone and routes through the existing close-via-trade
    path, not `drop_position`.
  - ccxt `is_instrument_listed` against a stubbed `exchange.markets`
    (present / absent / empty-fail-open).
  - `drop_position` removes the position without emitting a trade.
  - `_forget_if_held` guard: live-confirmed gone ⇒ forget; past `delist_date`
    but still listed (settlement overlap) ⇒ no forget, "needs review" alert;
    no data provider / empty markets ⇒ no forget, alert.
  - `__do_remove_instruments` gone-branch: removing a gone instrument forgets
    (no trade); removing a still-listed one trades to flat as before.
- **Integration**
  - Warmup with one gone instrument in the set → completes, warms the rest,
    logs the skip, no crash (direct regression for the TON incident).
  - Restart with a restored gone position → instrument excluded, position
    forgotten, alert emitted, bot starts; instrument never subscribed.
  - Restart with a still-listed instrument carrying a future `delist_date` →
    position retained and closed via the existing trade path; not forgotten.
- **Backtest guard**
  - A sim run still warms/trades everything (`is_instrument_listed` no-ops).

## Scope

**In:** `IDataProvider.is_instrument_listed` (+ ccxt impl), `UniverseManager`
state-B handling (`_is_market_gone`, `_drop_gone`, `_forget_if_held`,
`_data_provider_for`, `_notify_gone`, and the `__do_remove_instruments`
gone-branch), `IAccountProcessor.drop_position`, ccxt warmup guard, alerting,
tests.

**Out:** state-A / future-delist handling (existing `DelistingDetector` +
close-via-trade path, unchanged), broker-side listing checks, restored-state
pre-filter (shown unnecessary), global kill-switch toggle, PnL re-derivation,
historical-log cleanup.

## Rollout

1. Implement + test on `feat/delisting-resilience`; PR to Qubx `dev`.
2. Push to `dev` → CI publishes a new qubx version (conventional commits drive
   the version bump).
3. Bump factors' qubx pin to the new version; rebuild the strategy release.
4. Update the `okx.factors` bot to the new release version and restart — TON is
   skipped in warmup and forgotten at start; the bot comes up clean without a
   manual state reset.
