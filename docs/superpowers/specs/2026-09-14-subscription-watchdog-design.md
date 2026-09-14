# Subscription watchdog: intent ownership, reconciliation, per-exchange status

**Date:** 2026-09-14
**Incident:** Lighter venue outage 2026-09-13 11:01–11:15 UTC — four prod bots blind for 10–22h
**Status:** approved design, pre-implementation

## Problem

On 2026-09-13 Lighter served nginx `503 Service Temporarily Unavailable` on both REST
and the WebSocket upgrade for ~14 minutes. Four prod bots (`lighter.nimble`,
`lighter.reversals`, `lighter.prisma-ctm`, `lighter.prisma-agg`) stopped receiving
market data. Two of them never recovered and needed manual restarts 22 and 20 hours
later; `lighter.nimble` reported `capital=101,388` unchanged for the entire window
while its thread stayed alive, its heartbeats kept firing, and its socket reconnected
seven times.

The socket was never the problem. The connector threw its own subscriptions away and
nothing could rebuild them.

### Root cause

`SubscriptionManager._monitor_subscription_status` recovers stale data by
**unsubscribe → sleep 3s → subscribe** (`subscription.py:487-494`), with no rollback
and no retry. Running that *during* an outage:

1. `unsubscribe()` succeeds — on Lighter it deletes `_handlers[(sub_type, market_id)]`
   synchronously (`qubx_lighter/data.py:229`) and removes the instruments from
   `_subscriptions`.
2. `subscribe()` raises in `_ensure_websocket_connected` **before recording anything**
   (`qubx_lighter/data.py:158-163`), because the venue is still returning 503.

The consequences compound:

- **The watchdog disarms itself.** `_monitor_subscription_status` iterates
  `data_provider.get_subscriptions()` → `list(self._subscriptions.keys())`. The
  orderbook set was emptied, so the key was deleted, so there is nothing left to
  police. Confirmed in Loki: not one `Stale data detected` line in the following 22
  hours. The monitor *thread* survives — its `while True` catches the exception — it
  simply has an empty worklist.
- **The information is gone system-wide.** `_apply_swap` computes
  `_current_sub_instruments = set(self.get_subscribed_instruments(_sub))`
  (`subscription.py:201`), which reads *from the provider*. The provider's
  `_subscriptions` dict is the only record anywhere of what the universe is, so a
  subsequent `set_universe` could not repair it either.
- **Recovery paths silently no-op.** `WebSocketManager._resubscribe_all` replays
  `self._subs` — a second private copy of the truth, which survived because
  `unsubscribe` sends its frame *before* `del self._subs[channel]` and `send()` raises
  when `_ws is None`. So every later reconnect re-established all 21 `order_book/*`
  channels successfully, and the venue streamed orderbooks into a connector whose
  callback does `self._handlers.get(key)` → `None` → **drops the message with no log**
  (`qubx_lighter/data.py:484`).

The same shape exists elsewhere and is one venue hiccup from the same outcome:
`ccxt/data.py:313-316` does the identical unsubscribe→subscribe inside a `try/except`
that swallows the failure, and `qubx_hyperliquid/data.py:179-181` tears down then
rebuilds inside `subscribe(reset=True)`.

### Contributing factor

At 11:16:30 the socket returned and `_resubscribe_all` sent ~23 subscribe frames at
50ms spacing. The venue answered `[30009] Too Many Websocket Messages`, rejecting them
**asynchronously** — after the calls had returned successfully.
`_resubscribe_all` wraps each send in `contextlib.suppress(Exception)`
(`websocket_manager.py:399-400`), so nothing was logged and nothing retried. No
ordering discipline or handshake can catch a rejection that arrives after the call
returns; only checking whether data actually resumed can.

## Design principle

**Subscription intent is owned above the transport, and a loop closes the gap between
intent and reality. No single failure may destroy intent, and no repair is trusted
until data is observed to resume.**

## Decisions (agreed with Yuriy, 2026-09-14)

| # | Decision |
|---|----------|
| D1 | One PR. Scope is `qubx` only — **no `IDataProvider` interface change**, so `qubx-lighter` / `qubx-hyperliquid` need no release. |
| D2 | `SubscriptionManager` owns the desired universe (`_desired`) and stops reading current state from the provider. A failed repair leaves the provider wrong and intent intact; the next tick repairs it. |
| D3 | The watchdog moves into its own class with its own thread, policy and tests. |
| D4 | Repair stays `unsubscribe → sleep → subscribe`. It is required (see "Why not subscribe-only") and is now safe, because the retry loop — not the ordering — provides correctness. |
| D5 | A repair is verified on the following tick. An unverified repair retries with backoff and, once exhausted, escalates to *visibility* — `EXCHANGE_MAINTENANCE` plus ERROR logging. Transport-level escalation (reconnect / recreate) needs an `IDataProvider` method that does not exist and is deferred with D1. |
| D6 | Per-exchange status is published through the **existing** `ContextStatus` / `DegradeReason.EXCHANGE_MAINTENANCE`, not a new enum. |
| D7 | `QubxDegradedState` no longer counts toward `MAX_NUMBER_OF_STRATEGY_FAILURES`. |
| D8 | `synchronized` is replaced with a real shared lock; this is a precondition, not a cleanup. |

## Why not subscribe-only

`subscribe(sub, desired_set, reset=True)` alone would heal two of the three connectors
and is the shape the primary path already uses (`_apply_swap`, `subscription.py:229`).
It cannot be used, because `reset` means three different things:

| connector | `subscribe(set, reset=True)` | heals alone? |
|---|---|---|
| ccxt | `add_subscription(reset=True)` → `execute_subscription` replays the whole stream | yes |
| hyperliquid | internally unsubscribes existing, then subscribes | yes |
| lighter | issues transport only for `set − already_registered`; `reset` rewrites bookkeeping only | **no — a no-op** |

Unifying `reset` semantics is an `IDataProvider` change and is deliberately deferred
(see Follow-ups). Until then the repair sequence must include the unsubscribe, which
all three connectors do heal from:

- **lighter** — `new = desired − remaining` = the stale instruments →
  `_subscribe_instrument` recreates the handler and re-issues the WS subscribe.
- **ccxt** — partial unsubscribe resubscribes the remainder, then `reset=True` replaces
  the stream.
- **hyperliquid** — `reset=True` tears down and rebuilds.

### On the 3s sleep

Retained. It guards against unsubscribe/subscribe reordering at the venue, which is a
real hazard even though it does not currently bite on Lighter: both operations take
`WebSocketManager._subs_lock` (`websocket_manager.py:224,232`) and
`run_coroutine_threadsafe` schedules FIFO from one thread, so send order holds and TCP
preserves it. That is an emergent property of code in another repo, undocumented and
untested, and it would not survive someone adding batching or a second send path.

Its **status** changes: it is a churn-reducer, not the safety net. Correctness comes
from D5. The implementation must carry a comment saying so, or the next reader will
delete it (as the author of this spec first proposed) or come to depend on it.

## Components

### 1. `qubx/utils/misc.py` — fix `synchronized` (precondition)

`synchronized` builds its lock at **decoration** time, one per function
(`misc.py:507-516`). `subscribe`, `unsubscribe` and `commit` therefore hold *different*
locks and do not exclude each other, and today's watchdog calls providers holding none
at all. A reconcile racing a commit would corrupt `_desired` itself.

Replace with a single per-instance `threading.RLock` held across intent mutation and
reconciliation. Re-entrant because `commit()` → `_apply_swap()` → `reconcile()` is one
call chain. Audit every current `@synchronized` user; anything relying on the accidental
cross-instance exclusion must be called out rather than silently changed.

### 2. `qubx/core/mixins/subscription.py` — intent ownership

```python
_desired: dict[str, set[Instrument]]      # subscription key -> instruments
_unsupported: set[tuple[str, str]]        # (exchange, subscription key)
```

`_apply_swap` maintains `_desired` and no longer sources `_current_sub_instruments`
from the provider (`subscription.py:201,209`). The manager-level accessors
`get_subscribed_instruments` / `has_subscription` read `_desired`.

**Blast radius is small.** Those accessors are consumed by `context.py:996,1009` (the
strategy API, `ctx.get_subscribed_instruments` / `ctx.has_subscription`) and internally;
backtester paths call provider-level methods directly (`backtester/data.py:131,140`)
and are untouched. Strategies begin seeing intent rather than transport state — which
is what they always believed they were getting, since the two never diverged before
this bug.

`_desired` is authoritative from the first `_apply_swap`. It is not seeded from the
provider at startup: an empty intent plus an empty provider is consistent, and any
divergence introduced later is what reconciliation exists to fix.

### 3. `reconcile(refresh: set[Instrument] = frozenset())`

Drives each provider toward `_desired`, under the lock from §1.

- **Targeted repair** — for each `(exchange, subscription key)` with instruments in
  `refresh`: `unsubscribe(key, refresh_subset)` → `sleep(3s)` →
  `subscribe(key, desired_subset, reset=True)`.
- **Wholesale re-assertion** — an empty `refresh` re-asserts `_desired` for every
  `(exchange, key)` with `subscribe(key, desired_subset, reset=True)` and no unsubscribe.
  Used after a reconnect and after a provider recreation, where the transport is known
  to have lost state but no instrument is individually suspect.

  Note this cannot detect drift on its own: §2 deliberately stops reading the provider,
  so the manager has no view of transport state to diff against. Staleness is the only
  drift signal, and it arrives through `refresh`. That is intentional — a second source
  of truth to compare against is exactly what this design removes.
- Per-exchange `try/except` so one bad venue cannot abort the rest.
- `NotSupported` records `(exchange, key)` in `_unsupported` and is never retried.
  Without this the watchdog spins forever on a capability the venue does not have —
  `_apply_swap` currently only warns (`subscription.py:231`), which is adequate for a
  one-shot call and not for a loop.

A raised exception leaves `_desired` untouched. That is the entire fix.

### 4. `qubx/core/subscription_watchdog.py` — `SubscriptionWatchdog` (new)

Owns the thread, the interval, and all policy. Constructed with
`(data_providers, health_monitor, status, reconcile_fn, strategy_state, interval=30.0)` —
no dependency on `SubscriptionManager` beyond the callable, so it is unit-testable
against a fake provider whose `subscribe` raises on command. That test does not exist
today because there is no seam to write it against.

Per tick, per exchange:

1. Skip unless `strategy_state.is_on_warmup_finished_called`; skip simulation providers.
2. Read the exchange facts (§5) and classify:
   - **OK** — nothing stale. Clear any held repair state.
   - **DARK** — see §6. **No per-instrument repair.** This is the 11:16 case where 21
     simultaneous subscribes tripped `30009` and cost the recovery. Hold
     `EXCHANGE_MAINTENANCE` (§6), wait, and re-assert intent **once** when data returns
     — `reconcile()` with an empty `refresh`. The connector reconnects on its own; the
     watchdog's job while dark is to stop making it worse.
   - **DEGRADED** — some stale, some live. Call `reconcile(refresh=stale)`.
3. **Verify on the next tick.** Instruments repaired last tick that are still stale are
   retried, with exponential backoff (1 tick → 2 → 4, capped at 8) so a persistently
   unreachable instrument does not churn every 30s. After the cap is reached twice
   without recovery, the watchdog stops repairing that instrument and escalates to
   **visibility**: hold `EXCHANGE_MAINTENANCE` for the exchange, log at ERROR naming the
   instruments, keep exporting the counts. It does not keep retrying silently.

   **Transport escalation is out of reach in this PR.** `IDataProvider` exposes
   `start()`, `close()` and `is_connected()` but no `reconnect()`
   (`interfaces.py:594-745`), so the watchdog cannot re-establish a socket it did not
   create, and `close()`+`start()` has undefined per-connector semantics — using it here
   would be guessing. ccxt's `ExchangeManager` already self-recreates on its own stale
   signal; lighter and hyperliquid have no equivalent reachable from core. Until
   `IDataProvider.reconnect()` exists (see Follow-ups), an unrecoverable wedge is
   surfaced loudly and left to the supervisor — which in practice means a pod restart,
   the same action that ended the 2026-09-13 incident, but taken within minutes on a
   signal rather than within a day on a human noticing.

Not gated on `is_connected()`. The warning at `subscription.py:459-461` holds: a wedged
ccxt provider drops that flag exactly when the watchdog is needed. Connection state
selects the remedy; it never suppresses action.

### 5. `qubx/health/base.py` — per-exchange facts

```python
@dataclass(frozen=True, slots=True)
class ExchangeDataStatus:
    exchange: str
    connected: bool | None       # None = no callback registered
    subscribed: int
    stale: int                   # per-instrument count, NOT max() over the exchange
    last_event_time: dt_64 | None
```

`IHealthMonitor.get_exchange_data_status(exchange, subscribed_instruments)` computes it.
Both inputs already exist (`_last_event_time`, `_is_connected_callbacks`); nothing new is
plumbed.

Two existing pieces are deliberately *not* reused as-is:

- `is_exchange_stale` (`base.py:388`) builds on `get_last_event_time_by_exchange`, a
  **`max()` across all instruments** (`base.py:355-366`). One live instrument makes the
  whole exchange look healthy. This is the same false-clear that resolved
  `lighter.reversals`' platform alert at 00:00 while 29 instruments and all 3 open
  positions stayed dark for another 9.5 hours. `max()` is the right aggregation for
  "is the venue entirely dark" and the wrong one for "is everything healthy" — so it
  informs DARK only, never OK.
- `is_connected` (`base.py:343`) returns `True` when no callback is registered.
  `ExchangeDataStatus.connected` preserves that as `None` so callers can distinguish
  "connected" from "nobody told me"; the fail-open default at the old call site is left
  alone.

### 6. When `EXCHANGE_MAINTENANCE` is held

`DegradeReason.EXCHANGE_MAINTENANCE` exists (`core/status.py:29`) and has never had a
writer. Its **reader is fully built**: `trading.py:127-135` refuses every order for a
degraded exchange with `QubxDegradedState`, including position-reducing ones, and its
docstring already reasons about precisely this case.

Held when, for **2 consecutive ticks**, either:

- **(A)** `connected is False` — the fast path. Catches a refused or dropped socket in
  ~60s. In the incident this would have published at ~11:03:30, eight minutes before
  staleness could say anything.
- **(B)** `stale == subscribed and subscribed >= 2` — the slow path. Catches a provider
  that believes it is connected and delivers nothing. This is the case that mattered:
  from 11:17 onward `is_connected()` was `True` for 22 hours
  (`websocket_manager.py:168` — `state == CONNECTED and _ws is not None`) while every
  message was being dropped. (B) would have held from ~11:22.

`connected is None` is not a trigger for (A) — we cannot tell — but (B) still applies.
`subscribed >= 2` because with one instrument, "all stale" and "one wedged stream" are
the same observation.

Cleared on the first tick where any instrument on the exchange delivers. Asymmetric on
purpose: slow to halt trading, quick to resume.

**Partial staleness never publishes a degradation.** `is_degraded_for` is scoped to the
exchange, not the instrument, so publishing on 2-of-21 wedged instruments would refuse
orders on the 19 healthy ones.

Why 2 ticks: every healthy reconnect in the incident logs completed in 1–4s
(`0.56s`, `0.73s`, `0.80s`…). A single-tick rule would flap the order path on routine
blips.

### 7. `qubx/core/mixins/processing.py` — degradation is not a strategy failure

`QubxDegradedState` raised inside `on_event` currently propagates into the generic
handler at `processing.py:658-668`, which counts 10 **consecutive** failures and raises
`StrategyExceededMaxNumberOfRuntimeFailuresError`, stopping the run.

`deny_trading_when_degraded` defaults to `False` in qubx (`initializer.py:54`), but
**frab defaults it to `True`** (`frab/src/frab/strategies/base.py:249`). Wiring the
`EXCHANGE_MAINTENANCE` writer without this change would convert a 14-minute venue blip
into a dead frab run.

Catch `QubxDegradedState` separately: log at WARNING (throttled), do **not** increment
`_fails_counter`, do not reset it either. It is a framework-generated refusal of an
expected condition, not a strategy bug. frab's own documentation describes the counter
interaction as something to work around by disabling the flag, which means the feature
is not usable as designed; this removes that sharp edge.

## Error handling

| failure | behaviour |
|---|---|
| `subscribe` raises during repair | `_desired` intact; logged; retried next tick |
| `unsubscribe` raises during repair | same; the provider may be left with a live stream the manager will re-assert |
| repair succeeds but data does not resume | caught by verification (§4.3); retried with backoff, then escalated to `EXCHANGE_MAINTENANCE` + ERROR |
| venue rejects the subscribe asynchronously | indistinguishable from the above, and handled identically — this is the `30009` case |
| `NotSupported` | recorded in `_unsupported`, never retried |
| one exchange failing | isolated; other exchanges reconcile normally |
| watchdog tick raises | caught, logged with traceback, thread survives (as today) |

## Testing

Unit, against a fake `IDataProvider` — the seam that does not exist today:

1. `subscribe` raises during repair → `_desired` unchanged; next tick retries; provider converges.
2. Provider registry emptied behind the manager's back → reconcile restores it from `_desired`. **This is the incident, reduced to a test.**
3. Repair "succeeds" but the fake keeps reporting stale → retries back off 1/2/4/8 ticks, then stop; `EXCHANGE_MAINTENANCE` held and one ERROR logged, no further per-instrument churn.
4. DARK exchange → zero `subscribe`/`unsubscribe` calls issued (the `30009` regression).
5. `NotSupported` → recorded once, never retried.
6. (A) and (B) each independently hold `EXCHANGE_MAINTENANCE` after exactly 2 ticks, not 1; cleared after one delivering tick.
7. `connected is None` does not trigger (A); `subscribed == 1` does not trigger (B).
8. `QubxDegradedState` in `on_event` 20 times in a row → run does not stop; any other exception 10 times → it does.
9. Concurrency: `commit()` on one thread and `reconcile()` on the watchdog thread do not interleave (regression for the `synchronized` bug).

Integration: existing live-connector suites must pass unchanged — no `IDataProvider`
signature moves, so `qubx-lighter` and `qubx-hyperliquid` build against this without a
release.

## Observability

- Watchdog logs one line per repair with exchange, subscription key, instrument count,
  and rung; one line per escalation; one per `EXCHANGE_MAINTENANCE` add/clear.
- `ExchangeDataStatus` (`stale`/`subscribed` per exchange) goes into the state snapshot
  the health monitor already writes. This is the signal
  `bot_data_last_event_timestamp_seconds` is missing platform-side: that gauge is
  labelled `(bot_id, exchange, data_type)`, so one live instrument clears it for the
  whole exchange. Exporting the counts lets the platform alert on a *fraction*, which
  is what the `lighter.reversals` false-clear needed. The control-api rule change is
  out of scope here and tracked separately.

## Out of scope / follow-ups

1. **`IDataProvider` unification** — two additions, one coordinated release of both
   connector plugins:
   - `set_subscriptions(type, instruments, refresh=...)` replacing the three-way-ambiguous
     `reset` flag. Removes the unsubscribe from the repair path entirely and closes the
     transient window, retiring the 3s sleep with it.
   - `reconnect()`, which unblocks the transport escalation rungs §4 currently cannot
     reach. Without it an unrecoverable wedge can only be reported, not fixed in-process.
     This is the single highest-value item on this list.
2. **Lighter silent drop** — `qubx_lighter/data.py:484` drops a message with no handler
   and logs nothing. Bounded to one tick by this design, but still undiagnosable. Needs
   a connector fix.
3. **`_resubscribe_all`'s blanket `contextlib.suppress`** (`websocket_manager.py:399`)
   and its 50ms burst that tripped `30009`. Should log, and should pace against the
   venue's inbound budget.
4. **Fold in ccxt's `ExchangeManager._stale_monitor_loop`** once `reconnect()` exists.
   Today it is a second, uncoordinated watchdog acting on the same signal with a
   different remedy (full exchange recreation), invisible to this one — so a ccxt
   exchange can be recreated underneath a watchdog that is mid-repair.
5. **Platform alert** — per-instrument or fraction-based staleness rule in
   `k8s/apps/{dev,prod}/kube-prometheus-stack.yaml`.
