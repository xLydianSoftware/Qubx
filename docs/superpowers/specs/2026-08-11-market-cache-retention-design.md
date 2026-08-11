# Market-Cache Retention Config Design

**Date:** 2026-08-11
**Status:** Approved for planning
**Driver:** [frab#85](https://github.com/xLydianSoftware/frab/issues/85) — live frab bots hold
2.6–3.3 GB RSS, of which ~1.2 GB is dead history retained by one hardcoded default.
**Repos:** qubx (this spec's implementation) + consumer rollout in frab, xrelease,
xlydian-platform (small coordinated changes, listed in §Rollout).

## Problem

`CachedMarketDataHolder` buffers every event stream 10,000 items deep per instrument
(`max_buffer_size=10_000`, threaded from nowhere — no caller overrides it). Measured on the
dev frab bot (26h uptime, RSS 3,325 MB):

- **1,078,188 `FundingRate` ticks** (118 instruments × full 10k `Indexed` buffers) ≈ 373 MB —
  consumers read only the latest tick;
- **460k `OrderBook` snapshots** on the prod live bot (~46 instruments × 10k) — consumers read
  only the current book;
- (frab-side, same pattern, fixed in rollout: 1,144 one-second spread series × 10k bars ≈ 841 MB
  feeding a 300-bar EMA.)

There is no leak — only retention nobody asked for. Profiling method and full budget:
quantkit `docs/live-bot-memory-profiling.md`.

## Design (qubx)

### Config schema

New model in `utils/runner/configs.py`, wired into `LiveConfig`:

```python
class MarketCacheConfig(StrictBaseModel):
    """Retention caps for the per-instrument market-data buffers."""
    default_length: int = 10_000              # unchanged qubx default
    per_type: dict[str, int] = Field(default_factory=dict)
```

```python
class LiveConfig(StrictBaseModel):
    ...
    market_cache: MarketCacheConfig = Field(default_factory=MarketCacheConfig)
```

YAML shape (values live in deployment configs, not in qubx — qubx ships no opinionated caps;
defaults reproduce today's behavior exactly):

```yaml
live:
  market_cache:
    per_type:
      orderbook: 4
      funding_rate: 64
```

Validation: every value must be `>= 1` (pydantic validator; a zero/negative cap is a config
error, not a disable switch). Keys are **base dtype names**; unknown keys are permitted and
simply never match (forward compatibility), but each configured key is logged once at startup
(`market cache retention: orderbook=4, funding_rate=64`) so typos are visible.

### Threading

`runner.py` passes the config through to the context, which passes it to the market manager:

- `MarketManager.__init__(..., max_buffer_size: int = 10_000, per_type_lengths: dict[str, int] | None = None)`
- `StrategyContext` ctor gains the same two values, sourced from `config.live.market_cache`
  (`default_length` → `max_buffer_size`, `per_type` → `per_type_lengths`).
- Backtester/simulation paths are untouched: they construct their own holders and `LiveConfig`
  does not apply. (Warmup runs inside the live process but through the sim construction path —
  also untouched.)

### Resolution

Applied **at series creation time** in the market manager:

- Generic series (`get_data` — the path that buffers `FundingRate`, `OrderBook`, trades, …):
  resolve the cap by the event type's **base name** via `DataType.from_str(event_type)`
  stripping parameters, so `orderbook(0,1)` matches the `orderbook` key. Cap =
  `per_type_lengths.get(base_name, max_buffer_size)`.
- OHLC series: same lookup under the `ohlc` key (no deployment sets it today; default applies).

Series already created keep their length for the life of the process — caps apply from process
start, which is how bots deploy anyway (restart per release). No runtime mutation API (YAGNI).

### Rider: `FundingRate` slots

`qubx.core.basics.FundingRate` becomes `@dataclass(slots=True)` (~363 B → ~120 B per object).
Grep shows no dynamic attribute assignment on instances anywhere in qubx, quantkit, frab, or
the connector plugins. Risk: third-party pickling of old instances — none exists (objects are
transient stream events).

## Rollout (consumer repos, in order)

1. **qubx**: this spec → PR → dev-push release → note the released version `X`.
2. **xrelease**: bump the qubx pin its config validation uses to `X` **before** any frab config
   mentions `market_cache` — unknown fields fail builds with `extra_forbidden` (the pin-lag trap
   is already documented inline in the dev yaml's `health:` block comment).
3. **frab**:
   - `domain/spread.py`: `MAX_SPREAD_LENGTH` 10_000 → **600** (5-min EMA on 1s bars needs 300;
     2× headroom). One constant; series already thread it.
   - lock bump to qubx `X`; deploy configs (`xrelease dev/binance/v11-dynamic-bhpl.yaml`,
     `prod/binance/frab-olereon-bin-hpl.yaml`) and the paper twin's platform overrides gain
     `live.market_cache.per_type: {orderbook: 4, funding_rate: 64}` (64 ticks ≈ 10 min of rate
     history; the funding monitor's staleness guard compares only against the last tick).
   - tag `v0.11.8`, release train, restart dev + both prod bots.
4. **xlydian-platform**: `MALLOC_ARENA_MAX=2` added to `baseBotEnv()` in
   `control-api/internal/k8s/deployer.go` (same value as the xrust workloads; glibc arena cap —
   near-free for GIL-bound processes, recovers arena fragmentation). Applies per bot at next
   restart. Independent of 1–3; can land any time.

## Testing

- **qubx unit tests**: config parse (defaults = today's values; per_type validation rejects 0);
  resolution (parameterized dtype `orderbook(0,1)` hits the `orderbook` cap; unmatched dtype
  falls back to default; `ohlc` key honored); threading (MarketManager receives the values from
  a constructed context); `FundingRate` slots (no `__dict__`, fields intact, repr unchanged).
- **frab**: existing spread-tracker tests must stay green with the shorter buffer (they use far
  fewer than 600 bars); one assertion added that `MAX_SPREAD_LENGTH == 600` with the EMA-window
  rationale in the test name.
- **Field verification** (dev bot, `debug-prof` census still standing): 24h after deploy expect
  FundingRates ≈ 7.5k (from 1.08M), 1s bars ≈ 690k (from 9.36M), RSS trending toward
  1.3–1.6 GB; prod live additionally sheds the 460k OrderBook snapshots (cap works regardless
  of the still-unidentified subscriber).

## Out of scope

- Opinionated qubx defaults for any dtype (deployment configs own the values — explicit choice).
- Runtime retention mutation, per-instrument caps, prefetch `cache_size_mb` shrink (optional
  belt-and-braces a deployment can set today; not part of this change).
- Naming the prod orderbook subscriber (tracked in frab#85; cap neutralizes it).
- jemalloc or allocator swaps (arena cap only).
