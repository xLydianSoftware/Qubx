# Market-Cache Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-data-type retention caps for qubx's market-data buffers (config-exposed), frab's spread-series cut to 600, `MALLOC_ARENA_MAX=2` for bot pods — PRs prepared, NOT merged; validated by a local A/B paper run measuring RSS.

**Architecture:** `MarketCacheConfig {default_length, per_type}` in the live config schema, threaded runner → `StrategyContext` → `MarketManager` → `CachedMarketDataHolder`, which resolves a cap by base dtype name at series creation. frab and platform changes are one-liners with tests. Validation runs two frab paper bots side by side in tmux from the dev xrelease config — baseline (main + released qubx) vs fixed (worktree source deps + caps configured) — comparing RSS slopes.

**Tech Stack:** Python/pydantic (qubx), Cython-adjacent series (read-only touch), Go (platform deployer), uv path-dependency overrides, tmux.

**Spec:** `docs/superpowers/specs/2026-08-11-market-cache-retention-design.md` (this worktree)

## Global Constraints

- qubx defaults must reproduce today's behavior exactly: `default_length: int = 10_000`, empty `per_type` — no opinionated caps ship in qubx.
- `per_type` values validated `>= 1` (pydantic); a configured key logs once at startup: `market cache retention: <name>=<n>, ...`.
- Resolution by **base dtype name** via `DataType.from_str(...)` — `orderbook(0,1)` must match key `orderbook`; unmatched dtypes use the default; the `ohlc` key applies to OHLC series creation.
- Backtester/sim/warmup construction paths untouched.
- `FundingRate` becomes `@dataclass(slots=True)`; no other field changes.
- frab: `MAX_SPREAD_LENGTH = 600`; nothing else in frab's PR (config/lock bumps happen post-merge per spec §Rollout).
- **PRs are prepared and pushed; NOTHING is merged** in any repo.
- Worktrees: qubx work in `~/devs/Qubx/.worktrees/market-cache-retention` (branch `feat/market-cache-retention`, exists); frab work in `~/projects/frab/.worktrees/mem-retention` (branch `fix/spread-series-retention`, create). Never touch the main checkouts.
- Subagents: sonnet or better. Conventional commits, no Co-Authored-By.
- Gates: qubx `uv run pytest tests/qubx/core tests/qubx/utils -q` minimum + full build import; frab `uv run pytest -q`; platform `cd control-api && go build ./... && go test ./internal/k8s/...`.

## File Map

| Repo/worktree | File | Change |
|---|---|---|
| qubx wt | `src/qubx/utils/runner/configs.py` | `MarketCacheConfig`; `LiveConfig.market_cache` |
| qubx wt | `src/qubx/core/mixins/market.py` | `CachedMarketDataHolder(per_type_lengths=...)` + resolution + log; `MarketManager` forwards |
| qubx wt | `src/qubx/core/context.py:251` | pass market-cache values into `MarketManager` |
| qubx wt | `src/qubx/utils/runner/runner.py:654` | pass `config.live.market_cache` into `StrategyContext` |
| qubx wt | `src/qubx/core/basics.py:79` | `@dataclass(slots=True)` on `FundingRate` |
| qubx wt | `tests/qubx/utils/test_market_cache_config.py` | new |
| qubx wt | `tests/qubx/core/test_market_cache_retention.py` | new |
| frab wt | `src/frab/domain/spread.py:15` | `MAX_SPREAD_LENGTH = 600` |
| frab wt | `tests/test_spread_tracker.py` | length-rationale assertion |
| platform | `control-api/internal/k8s/deployer.go` (`baseBotEnv`) | `MALLOC_ARENA_MAX=2` |

---

### Task 1: qubx — MarketCacheConfig schema

**Files:** Modify `src/qubx/utils/runner/configs.py`; Test `tests/qubx/utils/test_market_cache_config.py` (new). Work in the qubx worktree.

**Interfaces — Produces:** `MarketCacheConfig(StrictBaseModel)` with `default_length: int = 10_000` and `per_type: dict[str, int] = Field(default_factory=dict)`, validator rejecting values `< 1`; `LiveConfig.market_cache: MarketCacheConfig = Field(default_factory=MarketCacheConfig)`.

- [ ] **Step 1: Failing tests** — new file:

```python
import pytest
from pydantic import ValidationError
from qubx.utils.runner.configs import LiveConfig, MarketCacheConfig


def _live(**over):
    base = dict(exchanges={}, logging={"logger": "InMemoryLogsWriter"})
    base.update(over)
    return LiveConfig(**base)


def test_defaults_reproduce_current_behavior():
    cfg = _live()
    assert cfg.market_cache.default_length == 10_000
    assert cfg.market_cache.per_type == {}


def test_per_type_parses():
    cfg = _live(market_cache={"per_type": {"orderbook": 4, "funding_rate": 64}})
    assert cfg.market_cache.per_type == {"orderbook": 4, "funding_rate": 64}
    assert cfg.market_cache.default_length == 10_000


def test_zero_cap_rejected():
    with pytest.raises(ValidationError):
        MarketCacheConfig(per_type={"orderbook": 0})
    with pytest.raises(ValidationError):
        MarketCacheConfig(default_length=0)
```

- [ ] **Step 2: Run to fail** — `uv run pytest tests/qubx/utils/test_market_cache_config.py -q` → import error / missing attr.
- [ ] **Step 3: Implement** in `configs.py` (near the other small config models):

```python
class MarketCacheConfig(StrictBaseModel):
    """Retention caps for per-instrument market-data buffers (spec 2026-08-11)."""

    default_length: int = 10_000
    per_type: dict[str, int] = Field(default_factory=dict)

    @field_validator("default_length")
    @classmethod
    def _default_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("default_length must be >= 1")
        return v

    @field_validator("per_type")
    @classmethod
    def _caps_positive(cls, v: dict[str, int]) -> dict[str, int]:
        for k, n in v.items():
            if n < 1:
                raise ValueError(f"per_type[{k!r}] must be >= 1")
        return v
```

and on `LiveConfig`: `market_cache: MarketCacheConfig = Field(default_factory=MarketCacheConfig)`.

- [ ] **Step 4: Run to pass**, then the utils suite: `uv run pytest tests/qubx/utils -q`.
- [ ] **Step 5: Commit** — `feat(config): MarketCacheConfig — per-dtype retention caps in LiveConfig`

### Task 2: qubx — resolution + threading

**Files:** Modify `src/qubx/core/mixins/market.py` (`CachedMarketDataHolder.__init__` line ~51, `get_data` line ~204, the OHLC series creation path that uses `_max_series_length`; `MarketManager.__init__` line ~408 and its `CachedMarketDataHolder()` call), `src/qubx/core/context.py` (~251), `src/qubx/utils/runner/runner.py` (~654 + StrategyContext signature). Test `tests/qubx/core/test_market_cache_retention.py` (new).

**Interfaces — Consumes:** Task 1's `MarketCacheConfig`. **Produces:** `CachedMarketDataHolder(default_timeframe=None, max_buffer_size=10_000, per_type_lengths: dict[str, int] | None = None)`; `MarketManager(..., max_buffer_size=..., per_type_lengths=...)`; `StrategyContext(..., market_cache_config: MarketCacheConfig | None = None)` (None → defaults; mirrors the `rate_limiting_config` passing pattern visible at runner.py:650-675).

- [ ] **Step 1: Failing tests** — construct a `CachedMarketDataHolder` directly (no full context needed):

```python
from qubx.core.mixins.market import CachedMarketDataHolder


def test_per_type_cap_applies_to_generic_series():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"orderbook": 4, "funding_rate": 64})
    assert h._resolve_series_length("orderbook") == 4
    assert h._resolve_series_length("orderbook(0,1)") == 4      # parameterized form matches base name
    assert h._resolve_series_length("funding_rate") == 64
    assert h._resolve_series_length("quote") == 10_000          # unmatched -> default
    assert h._resolve_series_length("ohlc(1h)") == 10_000       # ohlc key unset -> default


def test_ohlc_key_honored():
    h = CachedMarketDataHolder(max_buffer_size=10_000, per_type_lengths={"ohlc": 500})
    assert h._resolve_series_length("ohlc(1h)") == 500
```

- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implement.** In `CachedMarketDataHolder`: store `self._per_type_lengths = dict(per_type_lengths or {})`; add

```python
def _resolve_series_length(self, event_type: str) -> int:
    try:
        base = str(DataType.from_str(event_type)[0] if isinstance(DataType.from_str(event_type), tuple) else DataType.from_str(event_type))
    except Exception:
        base = event_type
    base = base.split("(")[0]
    n = self._per_type_lengths.get(base, self._max_series_length)
    if base in self._per_type_lengths and base not in self._logged_caps:
        self._logged_caps.add(base)
        logger.info(f"market cache retention: {base}={n}")
    return n
```

(implementer: check `DataType.from_str`'s actual return shape in `core/basics.py` and simplify the base-name extraction accordingly — the test's parameterized case is the contract; a plain `event_type.split("(")[0]` fallback must hold either way; `self._logged_caps: set[str]` initialized in `__init__`). Use `self._resolve_series_length(event_type)` at the `GenericSeries(...)` creation in `get_data` (replacing the bare `self._max_series_length`) and at the OHLC series creation site that currently passes `self._max_series_length`. `MarketManager.__init__` gains and forwards both values into its `CachedMarketDataHolder(...)` call. `StrategyContext` gains `market_cache_config` (default None) and passes the two values to `MarketManager`. `runner.py` passes `market_cache_config=config.live.market_cache`.

- [ ] **Step 4: Run to pass**, then `uv run pytest tests/qubx/core -q` and an import smoke `uv run python -c "import qubx.utils.runner.runner"`.
- [ ] **Step 5: Commit** — `feat(core): per-dtype market-cache retention, threaded from live config`

### Task 3: qubx — FundingRate slots + PR prep

**Files:** Modify `src/qubx/core/basics.py:79`; Test appended to `tests/qubx/core/test_market_cache_retention.py`.

- [ ] **Step 1: Failing test:**

```python
def test_funding_rate_has_slots():
    import numpy as np
    from qubx.core.basics import FundingRate
    fr = FundingRate(time=np.datetime64(0, "ns"), rate=0.0001, interval="1h", next_funding_time=np.datetime64(3600_000_000_000, "ns"))
    assert not hasattr(fr, "__dict__")
    with pytest.raises(AttributeError):
        fr.extra = 1  # type: ignore[attr-defined]
```

- [ ] **Step 2: fail → Step 3:** change decorator to `@dataclass(slots=True)`. Grep the worktree for `FundingRate` attribute writes outside the constructor to confirm none (`grep -rn "\.rate = \|\.next_funding_time = " src/`) — report findings.
- [ ] **Step 4:** targeted tests + `uv run pytest tests/qubx -q` (full unit sweep; skip integration markers per repo convention).
- [ ] **Step 5: Commit** `perf(core): FundingRate slots — 3x smaller per tick`; push branch `feat/market-cache-retention`; open PR to `dev` titled `feat: per-dtype market-cache retention caps + FundingRate slots` referencing frab#85 and the spec, stating defaults are behavior-preserving. **Do not merge.**

### Task 4: frab — spread-series length

**Files:** worktree `~/projects/frab/.worktrees/mem-retention` (create: `cd ~/projects/frab && git worktree add .worktrees/mem-retention -b fix/spread-series-retention origin/main`). Modify `src/frab/domain/spread.py:15`; Test `tests/test_spread_tracker.py`.

- [ ] **Step 1: Failing test** (append):

```python
def test_spread_retention_covers_ema_window_twice():
    # 5-min EMA on 1s bars needs 300 bars; retention is 2x that window, not 10k
    # (frab#85: 1,144 series x 10k bars held ~841MB on the dev bot).
    from frab.domain.spread import MAX_SPREAD_LENGTH
    assert MAX_SPREAD_LENGTH == 600
```

- [ ] **Step 2: fail → Step 3:** `MAX_SPREAD_LENGTH = 600` → **Step 4:** `uv run pytest -q` full suite green (existing tests use « 600 bars).
- [ ] **Step 5: Commit** `perf(spread): cap 1s spread series at 600 bars (frab#85)`; push; PR to main referencing frab#85 + the qubx spec, noting the config/lock rollout lands after the qubx release. **Do not merge.**

### Task 5: platform — MALLOC_ARENA_MAX

**Files:** `~/devs/xlydian-platform` branch `perf/bot-malloc-arena` from origin/main. Modify `control-api/internal/k8s/deployer.go` (`baseBotEnv()`); Test alongside existing deployer tests if a `baseBotEnv` test exists, else assert via the deployment-spec builder test pattern in `internal/k8s`.

- [ ] **Step 1:** locate `baseBotEnv()` (deployer.go ~line 109) and any test constructing the bot Deployment env; write the failing assertion that env contains `MALLOC_ARENA_MAX=2`.
- [ ] **Step 2: fail → Step 3:** append `corev1.EnvVar{Name: "MALLOC_ARENA_MAX", Value: "2"}` to `baseBotEnv()` with a one-line comment: `// glibc arena cap — GIL-bound bots gain nothing from per-core arenas; frab#85`.
- [ ] **Step 4:** `cd control-api && go build ./... && go test ./internal/k8s/...`.
- [ ] **Step 5: Commit** `perf(deployer): MALLOC_ARENA_MAX=2 on bot pods (frab#85)`; push; PR. **Do not merge.**

### Task 6: local A/B paper validation (controller-run, not subagent)

Run two frab paper bots locally in tmux from the dev xrelease config; measure RSS over ≥45 min.

- [ ] **Step 1: Baseline env** — `~/projects/frab` main checkout as-is (`uv sync` current lock = released qubx). **Fixed env** — the `mem-retention` worktree with `[tool.uv.sources] qubx = {path = "~/devs/Qubx/.worktrees/market-cache-retention", editable = true}` added LOCALLY (uncommitted — validation harness only, must not enter the PR) + `uv sync`.
- [ ] **Step 2: Local config** — copy `~/devs/xrelease/dev/binance/v11-dynamic-bhpl.yaml` to scratch twice (baseline/fixed): strip `notifiers`; keep aux `xdata::quantlab` + warmup (resolve reachability the way the frab research kernels do — check `~/.qubx/accounts.toml`/env conventions; if quantlab is unreachable locally, disable warmup AND note that pair discovery still needs aux — in that case port-forward or tailnet is required, not optional); `xstream_url` → `http://localhost:18001` via `kubectl -n data-service port-forward svc/xstream-service 18001:80` (dev cluster), `api_url` → `http://localhost:18002` via control-api port-forward. Fixed config additionally gets `live.market_cache.per_type: {orderbook: 4, funding_rate: 64}`.
- [ ] **Step 3: Run the allocator × retention matrix** — tmux session `mem-ab`, four windows, all `uv run qubx run <cfg> --paper`, launched together so every variant sees identical market data:

| Window | Code | Config caps | Allocator env |
|---|---|---|---|
| A baseline | frab main + released qubx | none | none |
| B caps | worktree deps | orderbook=4, funding_rate=64 | none |
| C caps+arena | worktree deps | same as B | `MALLOC_ARENA_MAX=2` |
| D caps+jemalloc | worktree deps | same as B | `LD_PRELOAD=libjemalloc.so.2` (install `libjemalloc2` if absent; verify preload took via `grep jemalloc /proc/<pid>/maps`) |

  A↔B isolates the retention caps; B↔C isolates the glibc arena cap; B↔D tests whether jemalloc beats capped glibc. If local RAM can't carry four bots (~1–2 GB each plus warmup spikes), run (A,B) concurrently then (C,D) concurrently and compare within pairs only. Confirm B/C/D log `market cache retention: orderbook=4, funding_rate=64` at startup and all variants discover pairs and arm the funding monitor.
- [ ] **Step 4: Measure** — sample every PID every 30s for ≥45 min: `ps -o rss= -p <pid>` appended to per-variant CSVs in the scratchpad. Success criteria: B's RSS slope after warm-up visibly flatter than A with ≥ ~100 MB divergence by minute 45 (spread series stop growing at 600 bars ≈ minute 10; FundingRate buffers cap at 64×~118); C and D quantify allocator effects on top of B — report their deltas as data, no pass/fail threshold (45 min undersells fragmentation effects that accrue over days; note this in the report).
- [ ] **Step 5: Report** — RSS curves per variant (numbers, not adjectives), startup-log proof of caps, allocator verdict (does `MALLOC_ARENA_MAX=2` measurably help; does jemalloc beat it enough to justify a base-image/deployer change — if yes, file a follow-up issue rather than expanding scope here), functional parity checks (pair counts and armed legs must match across variants); attach the summary to the PRs and frab#85. Kill tmux, drop port-forwards, leave worktrees for PR review.
