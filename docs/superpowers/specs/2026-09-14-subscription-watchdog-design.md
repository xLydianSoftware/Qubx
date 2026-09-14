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
| D5 | A repair is verified on the following tick **by `last_event_time` advancing past the repair**, never by `is_stale()` — which reads `True` right after a successful repair and would loop-repair sparse feeds. An unverified repair retries indefinitely at a backoff capped at `threshold/10`, reported at ERROR; it is never abandoned. Transport-level escalation (reconnect / recreate) needs an `IDataProvider` method that does not exist and is deferred with D1. |
| D5a | **A partially stale exchange is never marked `EXCHANGE_MAINTENANCE`.** Instruments still delivering prove the venue is up, so the fault is ours; degrading the exchange would be false and would halt trading on the healthy instruments. Only a fully dark exchange (§6) publishes a degradation. |
| D5b | An instrument is eligible for repair, and counts toward the per-exchange facts, only after `subscribed_at + threshold`. Freshly subscribed instruments report stale until their first message, so without this the watchdog repairs new subscriptions on sight and a full universe swap falsely reads as a dark exchange. |
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
1a. **Drop instruments still in their grace window.** An instrument is eligible for
   repair — and counts toward the facts in §5 — only once
   `now >= subscribed_at + STALE_THRESHOLDS[base_type]`. `IHealthMonitor.subscribe` does
   not seed `_last_event_time` (`health/base.py:185-193`) and `is_stale` returns `True`
   for a missing entry (`base.py:383-384`), so **every freshly subscribed instrument
   reports stale until its first message**. Without this gate the watchdog would tear
   down a brand-new subscription on the very next tick, before it ever had a chance to
   deliver — and a mid-session universe swap that replaces the whole universe would make
   `stale == subscribed`, falsely triggering DARK and halting trading (§6). The grace
   window is the threshold itself, because that is already the framework's statement of
   how long this data type may legitimately be silent.
2. Read the exchange facts (§5) and classify:
   - **OK** — nothing stale. Clear any held repair state.
   - **DARK** — see §6. **No per-instrument repair.** This is the 11:16 case where 21
     simultaneous subscribes tripped `30009` and cost the recovery. Hold
     `EXCHANGE_MAINTENANCE` (§6), wait, and re-assert intent **once** when data returns
     — `reconcile()` with an empty `refresh`. The connector reconnects on its own; the
     watchdog's job while dark is to stop making it worse.
   - **DEGRADED** — some stale, some live. Call `reconcile(refresh=stale)`.
3. **Verify on the next tick — by advancement, not by staleness.**

   A repair records `repaired_at`. It is verified when
   `last_event_time > repaired_at` — i.e. *some* message arrived after the repair.

   **Do not verify with `not is_stale()`.** Staleness compares `last_event_time` against
   a 10- or 30-minute threshold, so it reads `True` immediately after a successful repair
   and keeps reading `True` until enough fresh data accumulates. On orderbook that
   resolves in milliseconds and the distinction looks academic; on `trade` (30 min
   threshold, legitimately sparse) a repair can succeed and the feed stay quiet for
   minutes, so a staleness-based check would tear the working subscription down again on
   the next tick, and again, indefinitely — a self-sustaining repair loop on a healthy
   feed. Advancement is unambiguous: a message arrived, therefore the subscription is
   live, whatever the threshold says.

   **The repair must not call `IHealthMonitor.unsubscribe`.** That method pops
   `_last_event_time` (`health/base.py:207`) and discards the instrument from
   `_active_subscriptions`. Popping would erase the very evidence this verification
   depends on. Only `_apply_swap` — which genuinely removes an instrument from the
   universe — may call it; a repair is not a removal. (Today's watchdog already calls
   `data_provider.unsubscribe` directly, `subscription.py:487`, so this preserves
   existing behaviour rather than changing it.)

   **Backoff.** An unverified repair is retried at the next tick, then at doubling
   intervals, capped at `STALE_THRESHOLDS[base_type] / 10` — 1 min for quote/orderbook,
   3 min for trade. Tying the cap to that type's own threshold rather than a flat
   constant makes retries aggressive exactly where feeds tick continuously and patient
   where they do not, and avoids a second tuning knob that can drift out of sync with the
   first. Backoff resets on verification.

   Starting faster than one tick is not worth building: detection is bounded by the
   staleness threshold, so an instrument is already 10+ minutes dark before the first
   repair. Shaving the first retry from 30s to 5s is noise against that.

   Retries continue **forever**. An instrument is never abandoned: the backoff exists to
   bound churn, not to give up, and a wedged instrument that silently stops being
   repaired is the failure mode this whole design exists to remove.

   Escalation here is **reporting only**, never a degradation (D5a). On the first
   unverified retry the log moves from INFO to WARNING; at the backoff cap it moves to
   ERROR and names the instruments, repeating once per capped interval rather than once
   per tick. The exchange stays tradeable throughout, because it demonstrably works —
   other instruments on it are delivering. Protecting the strategy from the stale subset
   is an operator/platform decision on the exported counts, not something this watchdog
   decides unilaterally.

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
    subscribed: int              # eligible only — excludes instruments in grace
    stale: int                   # per-instrument count, NOT max() over the exchange
    in_grace: int                # subscribed too recently to judge
    last_event_time: dt_64 | None
```

`IHealthMonitor.get_exchange_data_status(exchange, subscribed_instruments)` computes it.

**`subscribed` and `stale` count only instruments past their grace window** (§4.1a).
Counting in-grace instruments as stale would make `stale == subscribed` immediately after
a full universe swap and falsely publish `EXCHANGE_MAINTENANCE`, halting trading on a
healthy venue. `in_grace` is carried separately so the condition stays visible rather than
silently hidden.

This requires one new piece of state: `_subscribed_at[(instrument, base_type)]`, recorded
in `IHealthMonitor.subscribe` (`base.py:185-193`, which today records only membership) and
cleared in `unsubscribe` alongside `_last_event_time`. The other two inputs
(`_last_event_time`, `_is_connected_callbacks`) already exist.

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

**Partial staleness never publishes a degradation** (D5a). Two independent reasons:

- **It would be false.** Instruments still delivering are proof the venue is serving us.
  A wedged subset means something is wrong on *our* side — a dropped handler, a rejected
  subscribe, a stream the venue silently closed. `EXCHANGE_MAINTENANCE` would attribute
  our bug to the exchange, and anyone reading the status later would draw the wrong
  conclusion.
- **It would be harmful.** `is_degraded_for` matches on exchange, so degrading on
  2-of-21 wedged instruments refuses orders on the 19 healthy ones — and under
  `deny_trading_when_degraded` that includes position-reducing orders.

**Do not attempt to fix this by scoping a degradation to an instrument.**
`QubxStatusInfo._scopes` is a flat `frozenset` of scope strings and `is_degraded_for`
tests `exchange in self._scopes` (`core/status.py:52-60`). An instrument-scoped entry
would never match any exchange, so it would be accepted, stored, reported in
`degradations` — and have **no effect on the order path at all**. A silent no-op is
worse than an absent feature. Instrument-level protection needs a real reader change
and is a follow-up.

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
| repair succeeds but data does not resume, others on the venue are live | caught by verification (§4.3); retried indefinitely at capped backoff, reported at ERROR. Exchange stays tradeable (D5a) |
| repair succeeds but data does not resume, whole venue dark | no repair attempted at all; `EXCHANGE_MAINTENANCE` held (§6) |
| venue rejects the subscribe asynchronously | indistinguishable from the above, and handled identically — this is the `30009` case |
| `NotSupported` | recorded in `_unsupported`, never retried |
| sparse feed quiet but healthy | verification by advancement (§4.3) prevents the repair loop; backoff caps residual churn at `threshold/10` |
| instrument subscribed seconds ago | in grace (D5b) — not repaired, not counted, cannot trigger DARK |
| one exchange failing | isolated; other exchanges reconcile normally |
| watchdog tick raises | caught, logged with traceback, thread survives (as today) |

## Testing

Unit, against a fake `IDataProvider` — the seam that does not exist today:

1. `subscribe` raises during repair → `_desired` unchanged; next tick retries; provider converges.
2. Provider registry emptied behind the manager's back → reconcile restores it from `_desired`. **This is the incident, reduced to a test.**
3. Repair "succeeds" but 2 of 5 orderbook instruments never deliver → retries double from one tick and **continue indefinitely** at the 1 min cap (`threshold/10`); log level climbs INFO → WARNING → ERROR; `EXCHANGE_MAINTENANCE` is **never** held and the exchange stays tradeable (D5a).
3a. The same instruments recover at tick 20 → retries stop, backoff resets, no degradation was ever published.
4. DARK exchange → zero `subscribe`/`unsubscribe` calls issued (the `30009` regression).
5. `NotSupported` → recorded once, never retried.
6. (A) and (B) each independently hold `EXCHANGE_MAINTENANCE` after exactly 2 ticks, not 1; cleared after one delivering tick.
7. `connected is None` does not trigger (A); `subscribed == 1` does not trigger (B).
8. `QubxDegradedState` in `on_event` 20 times in a row → run does not stop; any other exception 10 times → it does.
9. Concurrency: `commit()` on one thread and `reconcile()` on the watchdog thread do not interleave (regression for the `synchronized` bug).
10. **Sparse-feed repair loop (D5).** A `trade` subscription is repaired, then delivers exactly one message and goes quiet for 20 ticks → verified on the first tick after the message and **never repaired again**, even though `is_stale()` stays `True` throughout. Asserting on `is_stale` instead of advancement must fail this test.
11. **Grace period (D5b).** An instrument subscribed at t=0 reports no data → not repaired and not counted as stale before `t + threshold`; repaired on the first tick after it.
12. **False DARK on universe swap (D5b).** The entire universe is replaced mid-session → every instrument is in grace, `subscribed` is 0, and `EXCHANGE_MAINTENANCE` is not published. Without the grace gate this test halts trading on a healthy venue.
13. A repair never calls `IHealthMonitor.unsubscribe` — assert on a spy, since popping `_last_event_time` would silently destroy the verification signal in test 10.

Integration: existing live-connector suites must pass unchanged — no `IDataProvider`
signature moves, so `qubx-lighter` and `qubx-hyperliquid` build against this without a
release.

## Observability

- Watchdog logs one line per repair with exchange, subscription key and instrument count;
  one line per `EXCHANGE_MAINTENANCE` add/clear. Unverified repairs escalate INFO →
  WARNING → ERROR and then repeat only once per capped backoff interval, not once per
  tick — a permanently wedged instrument must stay visible without flooding the log for
  days, which is the shape the 2026-09-13 logs would have had.
- `ExchangeDataStatus` (`stale` / `subscribed` / `in_grace` per exchange) goes into the
  state snapshot the health monitor already writes. This is the signal
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
6. **Instrument-level trading protection.** D5a leaves a real, accepted gap: a
   persistently wedged subset on a working venue means the strategy keeps trading those
   instruments on stale prices, with nothing but a log line and a metric to stop it.
   Degrading the exchange is the wrong remedy (§6) and instrument-scoping the existing
   `Degradation` is a silent no-op (§6). Closing it properly needs
   `QubxStatusInfo.is_degraded_for` to take an optional instrument and the order path to
   consult it — a reader change with its own design discussion. Until then the exported
   `stale`/`subscribed` counts are the signal an operator acts on.
