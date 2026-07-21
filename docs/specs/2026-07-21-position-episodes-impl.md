# Position Episodes — Implementation Spec

- **Date:** 2026-07-21
- **Status:** Draft (design approved: [2026-07-21-position-episodes-design.md](2026-07-21-position-episodes-design.md))
- **Branch:** `feat/position-episodes` off `main`
- **Touched files:** `src/qubx/core/basics.py`, `src/qubx/core/loggers.py`, `src/qubx/core/account_manager/state.py`, `src/qubx/restorers/position.py`, new `tests/qubx/core/test_position_episodes.py`

## 1. `Position` (basics.py)

### 1.1 New state (class attrs, near the funding block at ~line 931)

```python
# episode tracking — baselines stamped at the last flat→open transition
episode_start_time: dt_64 = np.datetime64("NaT")
r_pnl_at_open: float = 0.0
commissions_at_open: float = 0.0
cumulative_funding_at_open: float = 0.0
```

Zero defaults = legacy degradation (episode accessors equal lifetime values).

### 1.2 `__init__` (line ~939)

Add optional kwargs after the existing accumulators:

```python
episode_start_time: dt_64 | None = None,
r_pnl_at_open: float | None = None,
commissions_at_open: float | None = None,
cumulative_funding_at_open: float | None = None,
```

After the existing accumulator assignment and the `quantity != 0` block:
- If all four kwargs provided → assign verbatim (restorer round-trip path).
- Else if the position is constructed **open** (`quantity != 0.0`) → stamp episode-at-init: baselines = the supplied lifetime accumulators, `episode_start_time = episode_start_time or NaT` (restorers pass restore time; NaT means "opening never observed"). A flat construction leaves the zero defaults.

### 1.3 Accessors (next to `get_realized_price_pnl`, ~line 1221)

Exactly the five methods from the design doc: `episode_pnl`, `episode_funding`, `episode_commissions`, `episode_price_pnl`, `episode_net_pnl`. One-line docstrings; `episode_pnl = self.pnl - self.r_pnl_at_open`.

### 1.4 `update_position` ordering refactor (line ~1071)

Current order: compute `qty_closing`/`qty_opening` → deal_pnl → (realize_only early-return) → size/avg mutation → `r_pnl += deal_pnl` (once, line 1135) → `update_market_price` → `commissions += fee` (once, line 1141). Restructure the accumulator application so stamping can sit between the closing and opening halves:

```python
was_open = self.is_open()                       # BEFORE any mutation
...existing qty_closing / qty_opening / deal_pnl computation...
...realize_only early-return UNCHANGED (never stamps)...

# fee attribution: pro-rata by quantity when a deal both closes and opens
#   (sign flip); otherwise the whole fee goes to the single side.
abs_c, abs_o = abs(qty_closing), abs(qty_opening)
fee_closing = fee_amount * abs_c / (abs_c + abs_o) if (abs_c and abs_o) else (fee_amount if abs_c and not abs_o else 0.0)
fee_opening = fee_amount - fee_closing

...existing closing-half size/avg mutation...
self.r_pnl += deal_pnl / conversion_rate        # closing realization → OLD episode
self.commissions += fee_closing / conversion_rate

opens_episode = (not was_open and not np.isclose(qty_opening, 0.0)) \
                or (abs_c and abs_o)            # flip always re-stamps
if opens_episode:
    self.episode_start_time = <deal timestamp as dt_64>   # same normalization as update_market_price
    self.r_pnl_at_open = self.r_pnl
    self.commissions_at_open = self.commissions
    self.cumulative_funding_at_open = self.cumulative_funding

...existing opening-half avg-price mutation...
self.quantity = position; self.position_avg_price_funds = ...
self.commissions += fee_opening / conversion_rate
self.update_market_price(...)
```

Return value `(deal_pnl, comms)` must remain the total fee in funds currency (`(fee_closing + fee_opening) / conversion_rate`) — callers (`AccountState`, sim accounting) treat it as the whole deal's commission; verify against every call site of `update_position`/`change_position_by`/`update_position_by_deal` before changing anything about the return.

Notes:
- Sub-lot-dust reopen (`was_open` False but `quantity != 0`, same sign): no `qty_closing`, plain stamp-then-open — dust's residual value rides into the new episode via avg-price averaging, which is the existing behavior; only the *baselines* are new.
- Pure close (`qty_opening ≈ 0`): no stamp; whole fee = `fee_closing` → old episode.
- `update_market_price` call stays last, as today.

### 1.5 Lifecycle methods

- `reset()` (~line 961): clear the four fields (NaT / 0.0).
- `reset_by_position` (~line 1008): copy the four fields.
- `flatten()` (~line 989): **no change** — comment noting the episode ends via the flat predicate and final values stay readable.
- `reconcile_size` (~line 1032): add keyword-only `timestamp: dt_64 | None = None`. If the position was flat (`not is_open()` pre-call) and the reconciled quantity is open → stamp episode from current accumulators with `episode_start_time = timestamp if timestamp is not None else self.last_update_time`. Caller update in `account_manager/state.py:423`: pass the venue snapshot's timestamp (whatever field the surrounding reconcile code already uses for staleness ordering — read the local code, don't guess the field name).
- `apply_funding_payment`: no change.

## 2. Position log record (loggers.py)

Both position-record emit sites (~line 112 and ~line 152 — the per-position record and the batch/portfolio variant; locate by the existing `funding_pnl_quoted` key) gain:

```python
"episode_start_time": <ISO or None if NaT>,
"realized_pnl_at_open_quoted": position.r_pnl_at_open,
"commissions_at_open_quoted": position.commissions_at_open,
"funding_at_open_quoted": position.cumulative_funding_at_open,
```

Match the surrounding None/NaN handling style exactly (see how `last_funding_time`-adjacent fields serialize, if any; otherwise mirror `funding_pnl_quoted`).

## 3. Restorers (restorers/position.py)

All three (`CsvPositionRestorer`, `MongoDBPositionRestorer`, `PostgresPositionRestorer`): read the four fields from the record with `.get(..., None)` and pass them to `Position(...)`. When absent (legacy rows) pass `None` for the baselines and the **restore time** as `episode_start_time` — `__init__`'s episode-at-init path then stamps baselines from the restored accumulators. When present, round-trip verbatim. Postgres: check whether the position log table schema is column-typed or JSONB — if columns, the write side (postgres logger under `src/qubx/loggers/postgres.py`) and any DDL/migration for the log table must gain the four columns too; investigate before editing.

## 4. Tests (`tests/qubx/core/test_position_episodes.py`, new)

Model fixtures on the existing `basics_test.py` / `test_position_funding.py` style. Cases (one test each, names matching):

1. `test_open_from_flat_stamps_pre_deal` — entry fee lands inside episode; `episode_net_pnl ≈ -fee` right after open.
2. `test_partial_trim_and_add_keep_baselines`.
3. `test_full_close_preserves_final_episode` — accessors still report the closed episode.
4. `test_reopen_restamps`.
5. `test_sign_flip_splits_pnl_and_fee` — closing pnl + pro-rata fee in old episode; new episode starts at ≈ −fee_opening.
6. `test_funding_while_flat_attributed_to_old_episode` — settle after close, then reopen: new `episode_funding() == 0`.
7. `test_realize_only_never_stamps`.
8. `test_reconcile_size_flat_to_open_stamps_at_now`.
9. `test_reset_clears_reset_by_position_copies_flatten_keeps`.
10. `test_legacy_zero_baselines_degrade_to_lifetime`.
11. `test_constructor_open_position_stamps_episode_at_init` (with and without explicit episode kwargs).
12. Restorer round-trip: extend the existing restorer tests (find them under `tests/`) — new fields round-trip; legacy record (fields absent) → episode-at-restore.
13. Simulation integration: scripted mini-backtest (open → funding settle → trim → close → reopen) asserting accessor values at each step — follow the existing sim-test harness patterns (look at how `test_position_funding.py` drives deals/settles).

## 5. Gates

- `uv run pytest tests/qubx/core -q` plus the full suite the repo's CI runs (`just test` if defined, else `uv run pytest`), and the repo linter (`just style-check` / ruff per config). Cython build untouched — no rebuild concerns.
- Grep-verify: every constructor call of `Position(` in src/ still type-checks with the new optional kwargs (they're keyword-optional — no call-site churn expected).

## 6. Subagent plan

| Stage | Agent | Model | Scope |
|---|---|---|---|
| Implement | 1 | opus | Everything above in one pass — the change is small but tightly coupled (ordering refactor + persistence + tests); splitting it invites seam bugs |
| Adversarial review | 1 | opus | Refute correctness of: fee pro-rata vs return-value contract, stamping order vs `realize_only`, dust/flip edges, restorer legacy path, no behavioral change to lifetime accumulators (diff every existing test expectation) |

Main loop: commit, run full gates, author the PR to `dev`.

## 7. Rollout

PR → `main` → CI releases a stable wheel to PyPI automatically. frab repin + consumption is the already-sequenced separate PR (do not start it until the new qubx version exists — per ecosystem rule, wait for CI to finish before pinning downstream).
