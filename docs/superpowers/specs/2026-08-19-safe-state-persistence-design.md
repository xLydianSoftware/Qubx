# Safe state persistence & bounded outbound I/O

**Date:** 2026-08-19
**Issue:** xLydianSoftware/xlydian-platform#375 (epic #374 — incident 2026-08-19)
**Status:** approved design, pre-implementation

## Problem

A hard power-cycle of the prod platform node (2026-08-19) killed TCP peers without
RST and exposed that qubx performs **unbounded blocking network I/O on the event
path**:

1. `RedisStatePersistence` builds its client as bare `redis.from_url(url)` — no
   socket timeouts, no keepalive, no retry policy (`state/redis.py`).
2. The periodic state snapshot (`_handle_state_snapshot`, `processing.py`) runs on
   **ProcessorThread** — the single event loop — and calls `persistence.save()`
   inline. With a half-open socket the synchronous `SET` sat in TCP retransmission
   for ~15 minutes. Four high-frequency bots (two real-money) froze completely:
   event queues grew to 280–430k, positions unmanaged, balance reconciles stopped.
   Two more real-money bots had the same call wedge past self-heal and needed pod
   restarts. Same failure class as the earlier okx-am-agg zombie (18 days
   undetected pre-health-panel).
3. `RedisStreamsExporter` and `QuestDBMetricEmitter` are already off the event
   path (ThreadPoolExecutor) but share two flaws: their clients also lack
   timeouts (a worker task can hang ~15 min, serializing the pool), and the
   executor queues are **unbounded** — outages accumulate tasks in memory and
   flush as a burst on recovery (implicated in the QuestDB O3 merge storm and a
   pod OOM during the incident).

## Design principle

**The event loop never performs network I/O, and every network call anywhere is
bounded in time; every buffer is bounded in space.**

## Decisions (agreed with Yuriy, 2026-08-19)

| # | Decision |
|---|----------|
| D1 | Scope covers all three layers: client timeouts, async state persistence, bounded exporter/emitter queues — one change. |
| D2 | Contract A: *all* persistence calls are async with read-your-writes; no caller ever blocks on the network. |
| D3 | If persistence is enabled and unreachable/unreadable at startup: **exit the bot** (running with potentially lost state is worse than crash-looping). Grace: ~60s exponential backoff, then exit non-zero. Corrupt stored JSON at startup is equally fatal; a missing key is not. |
| D4 | Implementation shape: wrapper layer (`SafeStatePersistence`) + shared `BoundedWorker` primitive; transports stay dumb and synchronous. |

## Components

### 1. `qubx/state/safe.py` — `SafeStatePersistence(IStatePersistence)` (new)

Wraps any real backend. Owns:

- **Pending buffer**: `dict[key -> value | _Tombstone]` under a lock — per-key
  latest-wins. Bounded by construction (one value per key).
- **Writer thread** (daemon, named `StatePersistenceWriter`): waits on an event,
  swaps the pending dict out, writes each entry through the backend
  (`save`/`delete`). On failure, merges failed keys back (newer pending values
  win) and retries with exponential backoff (1s → 30s cap). Tracks
  `last_success_ts` and `consecutive_failures`; failure logging throttled.
- **API semantics**:
  - `save(key, value)`: eager `json.dumps` **dry-run in the caller's thread** so
    `TypeError`/`ValueError` on unserializable values still raise at the call
    site (programming errors stay loud); then buffer + wake writer. Never blocks
    on network; never raises network errors.
  - `load(key, default)`: pending buffer first (read-your-writes, tombstones
    honored), then backend (bounded by client timeouts). Mid-run network failure
    raises to the caller — existing strategy code (frab) already documents
    graceful degradation.
  - `delete(key)` / `exists(key)`: tombstone in pending; `exists` consults
    pending (incl. tombstones) before the backend.
  - `stop()`: bounded flush (default 5s) so clean shutdowns persist final state.
  - `last_success_age() -> float | None`: seconds since last successful write
    (None until first write) — consumed by the health monitor.
- **Startup validation** (called by the factory before the context starts):
  probe the backend (`exists("__qubx_probe__")`) with exponential backoff
  (0.5s, 1, 2, 4, 8, then 10s steps) up to a **60s budget**; on exhaustion raise
  `StatePersistenceUnavailable` — the runner treats this as fatal and the
  process exits non-zero (D3). Corrupt-JSON on any load during startup
  propagates (fatal); `None`/missing key does not.

`DummyStatePersistence` is never wrapped.

### 2. `qubx/state/redis.py` (modified)

Stays synchronous and dumb. Client construction gains bounded-failure defaults,
all overridable through the existing `StatePersistenceConfig.parameters`
passthrough (zero config changes required for deployed bots):

```python
redis.from_url(
    url,
    socket_connect_timeout=2.0,
    socket_timeout=5.0,
    socket_keepalive=True,
    health_check_interval=30,
)
```

No client-level retry (the wrapper owns retry policy).

### 3. `qubx/utils/threading.py` — `BoundedWorker` (new)

The shared layer-3 primitive: single daemon worker thread + `deque(maxlen=N)`
(**drop-oldest**) + dropped-item counter + throttled warning (at most one per
30s: "dropped M items since last report"). `submit(fn, *args)` never blocks.
`stop(flush_timeout)` drains best-effort.

Single worker per instance is deliberate: it also restores FIFO ordering for
redis-stream exports (the current 2-worker pool can reorder `XADD`s — a latent
bug for downstream target consumers).

### 4. `qubx/exporters/redis_streams.py` (modified)

- `ThreadPoolExecutor(max_workers=2)` → one `BoundedWorker(maxlen=1000)`.
- Client gains the same timeout/keepalive kwargs as §2.
- Behavior change to document: under a prolonged outage, oldest queued exports
  are dropped (counted + warned). For target streams this is *safer* than the
  status quo — replaying stale targets to a live executor after recovery is
  worse than skipping them.

### 5. `qubx/emitters/questdb.py` (modified)

- `ThreadPoolExecutor(max_workers=1)` → `BoundedWorker(maxlen=10_000)` (metrics
  are cheap and numerous; dropping oldest under outage is free).
- `Sender` construction gains bounded timeouts (verified against the installed
  client API): `request_timeout=5000`, `retry_timeout=5000` (ms; client default
  retry_timeout is 10000 — halved so a dead server costs ≤10s per flush attempt
  instead of stacking).

### 6. `qubx/utils/runner/factory.py` (modified)

`create_state_persistence()` wraps any real backend:
`SafeStatePersistence(backend)` + startup validation (D3) before returning.

### 7. Health integration (`qubx/core/status.py`, `qubx/health/base.py`)

- New `DegradeReason.STATE_PERSISTENCE_STALE`.
- The existing health-monitor thread (which survived the incident and detected
  the queue overflow) additionally polls `persistence.last_success_age()` when
  the context's persistence is a `SafeStatePersistence`:
  - age > `max(3 × snapshot_interval, 60s)` → set
    `DEGRADED(STATE_PERSISTENCE_STALE, scope="state")`;
  - recovery → clear, log.
- Emit gauge `state_persistence_lag` via the metric emitter (feeds the
  platform-side staleness alert, xlydian-platform#379).

### 8. Unchanged on purpose

`_handle_state_snapshot` keeps building the snapshot dict on ProcessorThread —
that is where the consistent view of positions/orders/balances lives — and its
existing `persistence.save("state", snapshot)` call becomes non-blocking purely
via the wrapper. `IStatePersistence` protocol is unchanged; all consumers
(frab `AssetOverrides`/`RebalancerState`, quantkit drawdown tracker, control-api
readers) keep working with strictly better failure behavior.

## Failure timeline (the incident, replayed against this design)

Node power-cycles; redis peer half-open:

1. Next snapshot tick: build on ProcessorThread (~ms), `save()` buffers, returns.
   **Event loop never blocks.** Quotes keep processing.
2. Writer thread hits `socket_timeout=5s`, logs, backs off, retries. Pending
   holds exactly one latest snapshot + any dirty strategy keys.
3. After 60s of staleness: context → DEGRADED(STATE_PERSISTENCE_STALE); the
   `state_persistence_lag` gauge climbs; platform alert fires (#379).
4. Redis returns (≤ backoff lag ≈ 30s): one write per dirty key — no burst, no
   memory growth, no O3 storm. DEGRADED clears.
5. If instead the *bot* restarts mid-outage: startup probe fails for 60s →
   exit non-zero → crash-loop until redis is back (loud, safe; no amnesia).

## Testing

Unit (fake backend with scripted delays/failures/permanent hangs):
- `save()` returns <1ms while backend hangs forever; loop-liveness invariant.
- Read-your-writes incl. tombstones; latest-wins under repeated failure; failed
  batch re-merge does not clobber newer pending values.
- Backoff schedule; `last_success_age` accounting; throttled logging.
- Startup validation: success inside the budget after N failures; exhaustion
  raises; corrupt-JSON fatal; missing key benign.
- `BoundedWorker`: drop-oldest, counters, warn throttling, FIFO order, bounded
  stop/flush.
- Exporter: ordering preserved (single worker); drops counted under a hung fake.

Integration (docker redis, CI marker): mid-run `SIGSTOP` of redis-server →
assert event processing continues, DEGRADED flips, recovery re-persists, and a
subsequent cold start against the stopped server exits within ~70s.

Manual platform acceptance (per #375): kill redis under a live paper bot; event
loop continues within seconds; state writes resume on reconnect.

## Rollout

Lands on Qubx `dev` (full CI → dev PyPI release). Bots inherit it when their
releases rebuild against the new qubx. No config changes required; timeout
overrides available via `StatePersistenceConfig.parameters` if ever needed.
Deployed old-release bots (factors v0.1.x etc.) remain on the old behavior until
their next release bump — worth prioritizing the paper feeders and real-money
books when bumping.

## Related platform work (out of scope here, tracked in #374)

- xlydian-platform#379: per-bot staleness alert on the redis `state:{bot}:state`
  timestamp (>60s), superseding the metrics-based BotMetricsStale approach of
  PR #365 — the snapshot is *built on the event loop*, so its freshness is a
  direct liveness probe of the exact thread that froze; this design's
  `state_persistence_lag` gauge and DEGRADED reason are the in-bot halves of the
  same signal.
- control-api redis retry + QuestDB circuit breaker (#376), Postgres/QuestDB
  server-side keepalives and caps (#377, #378).
