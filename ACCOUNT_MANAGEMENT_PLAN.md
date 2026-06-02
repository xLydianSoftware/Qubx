# Account management redesign — Qubx rollout plan

Qubx-side rollout of the account-management redesign. This doc covers only the work
that lands in **this repo** (`qubx`), plus the coordination touchpoints where qubx meets
the exchanges repo and the downstream release. The exchanges-repo PR sequence (E1–E3) and
venue-adapter internals are out of scope here — they live in the exchanges repo.

See `account-management-design` for the full architecture (event hierarchy, `AccountManager`,
state machine, `IConnector` contract, conformance suite, decided defaults).

## Approach

- **No feature flag.** All work happens on a long-lived `feat/*` branch. Intermediate PRs
  against that branch are allowed to leave branch CI red. The final coordinated merge restores
  green. The cost of a flag (`QUBX_USE_NEW_ACCOUNT=1`-style gates threaded through
  `StrategyContext` and dispatch) is not worth paying when nothing on `main` depends on the branch.
- `main` stays green throughout the rollout. Non-redesign work (bug fixes, features, security
  patches) lands on `main` independently. The feature branch periodically rebases / merges `main`
  into itself to stay current.

## Branch layout

| Repo | Branch | Off |
|---|---|---|
| `qubx` | `feat/account-manager-redesign` | `main` |

The branch stays alive for weeks and takes multiple small, independently reviewable PRs.
`main` is untouched until the coordinated merge at the end.

**Branch CI stays red from Q2 until Q4.** This is intentional — the migration window has
half-built state. Q3 greens the backtester but CCXT conformance keeps broad CI red until Q4
brings the live path home. Use the new test suite (Q1 state-machine tests + Q3 backtester
strategy tests) as the green signal during that window rather than broad CI.

## PR sequence (`feat/account-manager-redesign`)

| # | Scope | Branch CI |
|---|---|---|
| **Q1** | Add new modules: event hierarchy (`ChannelMessage` / `AccountMessage` / `MarketDataMessage` + all leaves), `AccountState`, `AccountManager`, `SimulationAccountManager`, `_ms_to_cron`, `IConnector` Protocol. No callers yet. Unit tests for `AccountManager` + state machine (the most bug-prone part — heavy coverage here, in-process, no I/O). | **Green** (additive only). |
| **Q2** | Replace `TradingManager` + `ProcessingManager` + `StrategyContext` wiring with the new path. `ctx.trade` returns `Order(SUBMITTED)`; `ctx.cancel` / `ctx.update` become void. Old `IBroker` / `IAccountProcessor` still exist but unused. `SimulatedBroker` / `SimulatedAccountProcessor` tests break here. | **Red** (backtester + simulated connector tests fail; expected). |
| **Q3** | Migrate `SimulatedConnector` + `SimulationAccountManager` wiring. Backtester runs on the new path. Backtester strategy-level tests for the full lifecycle (submit→fill, cancel happy/rejected, update happy/rejected, stuck-order recovery). | Green for backtester; CCXT connector still red. |
| **Q4** | Migrate CCXT connector to `IConnector`. Run the exchanges-repo conformance suite (via git source) against the backtester in-process for fast feedback, then Binance testnet manually. | **Green**. |
| **Q5** | Strategy-side audit across downstream strategy repos. Run the migration greps from `account-management-design#Migration`. Delete `IBroker`, `IAccountProcessor`, `BasicAccountProcessor`, `CompositeAccountProcessor`, `process_order_request`, `send_order_async` and friends. Tag release candidate. | **Green**. |

Each PR is independently reviewable. Q1 lands without breaking anything. Q2 deliberately
breaks things on the branch and won't recover until Q3. Q4 brings the live path home. Q5 is
the cleanup + audit pass.

## Test layering (qubx-owned layers)

Two qubx-owned test layers (the third, exchanges conformance, lives in the exchanges repo).
No duplication across layers.

- **qubx core** — `tests/qubx/core/account_manager_test.py` (lands in Q1). Pure unit tests
  against `AccountManager` + `AccountState` with a mock connector. State machine transition
  table, illegal-transition raises, `pre_pending_status` capture/clear, fill dedup via
  `trade_id`, snapshot reconcile rules (grace window, freshness check),
  `_inflight_index` / `_pending_evict_index` membership.
  **The state machine is the bug-magnet — test it heavily here, in-process, no I/O.**
  State-machine tests live in qubx core (not exchanges/conformance); they need no connector at all.

- **qubx backtester** — `tests/qubx/backtester/` (lands in Q3). `SimulatedConnector` +
  `SimulationAccountManager` strategy-level tests. End-to-end lifecycle assertions per the 17
  conformance scenarios in `account-management-design#Conformance test suite`. The backtester
  is the de facto in-process conformance host — fast, deterministic, no testnet creds.
  **Stuck-order-recovery scenario wiring:** configure the `SimulatedConnector` to withhold the
  ack for a submitted order (simulate venue silence), arm the in-flight sweep for that test
  (`register_inflight_tick=True`), then advance simulated time past the thresholds (≥5s age, ≥2s
  tick). `SimulationRunner`'s time pump fires the `SimulatedScheduler` → `_sweep_stuck_inflight`
  → `request_order_status` → the connector replies `'open'` (`SUBMITTED→ACCEPTED`) or keeps
  failing (`SUBMITTED→REJECTED` after N retries). Fully in-process and deterministic.

**Bonus:** the exchanges conformance suite is parameterized on `connector_factory`. In E1 a
`--target=simulated|testnet` pytest option lets it point at qubx's `SimulatedConnector` for
in-process smoke before pushing to testnet — near-zero extra CI minutes, faster local iteration.

## Cross-repo dependency (qubx side)

The exchanges repo depends on qubx (not the reverse), so qubx drives the cadence:

- No pre-release publishing to PyPI from the feature branch. Avoids polluting PyPI with N
  `2.0.0.devX` versions that nothing on `main` consumes. The final coordinated merge cuts a
  single clean `2.0.0`.
- The exchanges branch pins qubx via a git source in its `pyproject.toml`, pinned to a
  **specific commit SHA** of `feat/account-manager-redesign` (not the branch name) so unrelated
  commits on the qubx branch don't break exchanges CI unpredictably:

  ```toml
  [tool.uv.sources]
  qubx = { git = "https://github.com/xLydianSoftware/Qubx", rev = "<qubx-branch-commit-sha>" }
  ```

  The SHA is bumped deliberately when fresher qubx code is wanted downstream. Note: the
  exchanges-side E1 PR (which imports the new types) must land **after** qubx Q1 merges to the
  qubx branch.

## Coordinated merge (qubx steps)

1. **Merge qubx branch → `main`.** Triggers the Qubx release pipeline → cuts `2.0.0` (or
   whatever the next major is, per conventional-commit auto-bump).
2. Exchanges branch then bumps `qubx[connectors]>=2.0.0` and removes the `[tool.uv.sources]`
   override (exchanges-repo step, downstream of #1).
3. Merge exchanges branch → `main`; tag connector releases (`qubx-hyperliquid`, `qubx-lighter`)
   — exchanges-repo steps.
4. Re-release affected strategies via `xrelease` pinned to qubx `2.0.0`.

## Pre-decided choices (qubx)

- **No feature flag in `StrategyContext`.** Branch + git-source override gives the safety;
  flag-gated dispatch is dead weight.
- **No pre-release PyPI publishing from the feature branch.** Git source only. One clean
  release at merge.
- **`AccountState` is single-writer — mutated only on the dispatch thread.** Verified against
  current qubx: there is exactly one loop thread (`ProcessorThread` in live, `context.py:402-405`;
  the main thread in sim, no thread spawned). Strategy callbacks run synchronously inside that
  loop, so `ctx.trade` → `TradingManager.trade` mutates the account inline *on that same thread*
  (`trading.py:133-138`), and `apply(event)` runs on it too (both drain from the same
  `channel.receive()` loop). The `[sync] add_order` and `apply(event)` arrows in the diagram are
  therefore serialized, never concurrent — which is why the legacy `AccountProcessor` needs no
  locks and the new `AccountManager` needs none either. **Connectors are event producers only:**
  their background REST/WS asyncio loops never touch `AccountState`; they call `self.send(...)`
  onto the thread-safe `Channel` (`queue.Queue`), and the dispatch thread is the sole consumer
  and sole writer. Make this a stated contract in the design: *"`AccountState` is mutated only on
  the dispatch thread; connectors emit events to the Channel and never mutate state directly."*
  Residual caveat — the invariant is single-*writer*, not single-*accessor*: any reader on
  another thread (external dashboard, health endpoint, off-thread logger) is the only thing that
  could observe torn state. In current qubx this is a non-issue because logging is itself driven
  by scheduled events that fire on the dispatch thread. If the platform runner ever reads
  `ctx.positions` from its own thread, *that read* — not the writes — is what would need guarding.
- **`SimulationAccountManager(AccountManager)` subclass** for backtest. The in-flight stuck-order
  sweep is **opt-in, not hard-skipped**: by default the sim does not auto-arm it (the
  `SimulatedConnector` acks/fills deterministically, so there are never genuinely stuck in-flight
  orders, and an auto-armed sweep would be overhead and a nondeterminism risk). Expose it as a
  parameter (e.g. `register_inflight_tick=False` default, `True` for the recovery test) so the Q3
  stuck-order-recovery scenario can drive it deterministically. The tick is pumped the same way
  every other periodic check is in backtest — by `SimulatedScheduler` (`backtester/utils.py`, a
  `BasicScheduler` subclass with no watcher thread) advanced from `SimulationRunner`'s time loop
  (`runner.py:236-240`): before each data point at `t`, `while t >= scheduler.next_expected_event_time():
  set_time(next); check_and_run_tasks()`. So a 2s cron tick fires deterministically as simulated
  time crosses each boundary — no wall clock, fully reproducible. (Live path: `pm.schedule` →
  `BasicScheduler` watcher thread → `_trigger` only `channel.send(...)` at `helpers.py:347`; the
  sweep itself still runs on the dispatch thread.) Decided in
  `account-management-design#Backtester migration`.
- **State-machine tests live in qubx core**, not exchanges/conformance.
- **Branch CI stays red from Q2 until Q4** (Q3 greens the backtester; CCXT conformance keeps
  broad CI red until Q4). Don't gate progress on broad CI green during the migration window; use
  the Q1 state-machine tests + Q3 backtester strategy tests as the green signal.

## Open questions to align before opening Q1

1. **Branch trigger safety in Qubx CI.** `.github/workflows/` has special handling for `dev`
   and `main` (publishes to PyPI, builds Docker images, deploys docs). Confirm `feat/*` branches
   don't accidentally trigger a release. If they might, prefix the branch or scope the workflow
   triggers explicitly.
2. **Legacy interface tests during Q2.** `test_account_processor_test.py` and
   `test_trading_test.py` test the legacy interfaces and will be the most visible "this is red"
   signals. Options:
   - Delete them in Q2 (the interfaces go away in Q5 anyway).
   - Mark them `@pytest.mark.skip("migrating in Q5")` and delete in Q5.

   **Recommendation: skip-mark.** Gives back the CI green signal earlier and makes the deletion
   an obvious follow-up.
3. **xlydian-platform runner.** The platform's session runner constructs `StrategyContext` from
   a config. The constructor signature changes (now takes `connectors: dict[str, IConnector]`
   and a single `AccountManager` instead of separate broker/account dicts). Either:
   - (a) Land a parallel platform PR coordinated with the Qubx 2.0.0 release, or
   - (b) Pin the platform to qubx 1.x until a platform-side adapter lands separately.
4. **xrelease compatibility.** All strategy ZIPs on the platform are built against the 1.x `ctx`
   API. Once Qubx 2.0.0 ships, running strategies must be re-released or pinned to 1.x:
   - **Lockstep**: re-release everything in the deploy window, or
   - **Strategy-by-strategy**: roll forward per strategy — slower but safer per-blast-radius.

## References

- `account-management-design` — full architecture (event hierarchy, `AccountManager`, state
  machine, `IConnector` contract, conformance suite, decided defaults).
- `excalidraw/account-management` — sequence diagrams (submit / cancel / update / stuck-order recovery).
- `.github/workflows/` — Qubx CI config (`ci.yml`, `integration-tests.yml`, `e2e-tests.yml`).
