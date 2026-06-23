# Reconciliator Redesign — git-style diff/merge

**Status:** design approved, stage 1 (the differ) ready to implement.
**Date:** 2026-06-23
**Scope:** replaces `reconcile.py`'s snapshot-reconcile logic with a two-stage,
git-flavored design. **This document covers stage 1 only** (finding diffs). Stage 2
(applying diffs / merging into local state) is deferred to a follow-up design.

## Motivation

The current `reconcile_snapshot` (in `src/qubx/core/account_manager/reconcile.py`)
folds three concerns into one pass: detecting what differs, deciding what to do about
it, and mutating `AccountState` — all interleaved with grace windows, freshness
guards, and covered-quantity deficit bookkeeping. The result is hard to unit-test:
there is no point at which "what differs between local and the venue" is a plain,
inspectable value, so the many live-trading edge cases cannot be enumerated as tests.

The redesign borrows git's split:

| git | our domain |
|---|---|
| two trees being compared (HEAD vs working tree) | `origin: AccountSnapshot` (venue truth) vs `local: AccountState` |
| `git diff` → list of **deltas** | `Differ.diff(local, origin)` → `list[Diff]` |
| delta status **ADD / DELETE / MODIFY** | `OriginalOrderMissing` / `LocalOrderMissing` / per-field `*Mismatch` |
| per-delta **hunks** (the changed lines) | the fine-grained field atoms of a modified order |
| `git diff` reports, `git apply`/`merge` decides | stage 1 = `Differ`, stage 2 = merge |

Stage 1 produces a **flat list of fine-grained `Diff` atoms** — a deterministic,
clock-free value (modulo one grace gate explained below) that can be asserted on
directly in tests, covering every live-trading scenario.

## Stage 1 — the `Differ`

### API

```python
class Differ:
    def __init__(self, grace: str | np.timedelta64 = "5s") -> None: ...
    def diff(self, local: AccountState, origin: AccountSnapshot) -> list[Diff]: ...
```

- No `ITimeProvider`. The only time-based rule (the grace gate) compares
  `origin.as_of` against each order's own `seen_at`; neither needs the wall clock.
- `diff()` raises `ValueError` if `local.exchange != origin.exchange`.
- `diff()` is **read-only** — it never mutates `local` or `origin`.

### Grace gate (orders only)

`origin.as_of` is the snapshot **request time** — stamped before the venue fetch
round-trip (`_snapshot_async` in `connectors/ccxt/connector.py` sets
`as_of = self._time.time()` *before* `fetch_open_orders/positions/balance`). So an
order's local state is uncertain relative to the snapshot if it changed in the window
`[as_of − grace, now]`:

- changed **after** the request (`seen_at > as_of`) — the snapshot cannot reflect it;
- changed within **grace before** the request (`as_of − grace ≤ seen_at ≤ as_of`) —
  the venue may not have registered it yet (propagation lag).

A single comparison against `as_of` implements that whole window (the fetch always
takes time, so `now > as_of` holds and never needs to enter the formula):

```python
seen_at = order.last_updated_at or order.submitted_at   # last local change
if seen_at is None:                       # untimestamped → cannot age → skip
    continue
if (origin.as_of - seen_at) < grace:      # inside [as_of - grace, now] → skip
    continue
# else: settled before as_of - grace → eligible to diff
```

The grace gate suppresses **all** atoms for an in-window order (both
`LocalOrderMissing` and every field `*Mismatch`) — if we are not sure of the order's
state we say nothing about it. The gate applies only where there is a local order with
a `seen_at`:

- `OriginalOrderMissing` (snapshot-only order, no local `seen_at`) is **never**
  grace-gated — emitted immediately.
- Positions / balances / venue-figures have **no** grace gate (no per-record
  `seen_at`; the snapshot is the size/figure authority, matching the old impl).

### Order matching

1. Index `origin.open_orders` by `venue_order_id` **and** by `client_order_id`.
2. For each **non-terminal** local order (terminal orders are skipped entirely):
   - match in origin by `venue_order_id`, else by `client_order_id`;
   - apply the grace gate;
   - **matched & past grace** → compare fields, emit one atom **per** differing field;
   - **unmatched & past grace** → `LocalOrderMissing`.
3. Each origin order matched to no local order → `OriginalOrderMissing` (no gate).

The cid-fallback match covers an unacked framework order (lost create-ack,
`venue_order_id is None` locally) that the snapshot reports under our own cid; when it
matches by cid and the venue id differs, that surfaces as `OrderVenueIdMismatch`.

### Diff taxonomy

```
Diff                                    # base: __repr__ → describe()
├─ LocalOrderMissing(order)             # DELETE  (grace-gated)
├─ OriginalOrderMissing(order)          # ADD     (no gate)
├─ DiffOrders(local, origin)            # base for order MODIFY atoms (FIELD marker)
│   ├─ OrderStatusMismatch          FIELD="status"
│   ├─ OrderFilledQtyMismatch       FIELD="filled_quantity"
│   ├─ OrderPriceMismatch           FIELD="price"
│   ├─ OrderVenueIdMismatch         FIELD="venue_order_id"
│   ├─ OrderQuantityMismatch        FIELD="quantity"
│   └─ OrderAvgFillPriceMismatch    FIELD="avg_fill_price"
├─ DiffPositions(local, origin)         # base for position MODIFY atoms (FIELD marker)
│   ├─ PositionSizeMismatch         FIELD="quantity"   (rendered as "size")
│   ├─ PositionAvgPriceMismatch     FIELD="position_avg_price"
│   └─ PositionMarginMismatch       FIELD="maint_margin"
├─ BalanceMismatch(local, origin)       # local, origin : Balance
└─ VenueFiguresMismatch(local, origin)  # local, origin : VenueAccountFigures
```

- The generic `OrdersMismatch` from the scaffold is **dropped** — fine-grained
  per-field atoms supersede it.
- `PositionMarkMismatch` is **deferred** (not added in stage 1).
- `BalanceMismatch` / `VenueFiguresMismatch` carry both `local` and `origin` (the
  scaffold had `origin` only, and `VenueFiguresMismatch` was mis-typed `origin: Balance`
  — both fixed) so tests and logs can show the before/after.
- Atoms are frozen + slotted + kw-only via the `@diffatom` decorator (same shape as
  `events.msg`).

### Tolerances (avoid dust atoms)

Field comparisons use a tolerance so float dust does not emit phantom diffs:

- quantity / filled_quantity / size: `instrument.lot_size * 0.5` (fallback `0.0`
  when no instrument). From the old `fill_qty_epsilon`.
- price / avg-price: `instrument.tick_size * 0.5` (fallback `0.0`).
- margin / balances / venue figures (no lot/tick): **relative** tolerance
  `rtol = 1e-9` (`abs(a - b) <= rtol * max(abs(a), abs(b))`).
- `status`, `venue_order_id`: exact (in)equality.

### String representation

`Diff.__repr__` delegates to `describe()`; the subtree bases render one git-flavored
line. `__repr__` (not just `__str__`) is overridden so a **list** of diffs prints
cleanly in logs (Python uses `__repr__` for container elements). Modify-atoms read
their `FIELD` marker via `getattr`, so each leaf stays a 2-line declaration and the
rendering lives once per subtree. Numeric fields append `Δ`.

```
OrderPriceMismatch[BINANCE.UM:SWAP:BTCUSDT cid=qubx_a1b2] price: 100.50 → 101.00 (Δ +0.50)
OrderFilledQtyMismatch[…BTCUSDT cid=qubx_a1b2] filled_quantity: 0.40 → 0.60 (Δ +0.20)
OrderStatusMismatch[…ETHUSDT cid=qubx_c3] status: OPEN → CLOSED
OrderVenueIdMismatch[…BTCUSDT cid=qubx_a1b2] venue_order_id: None → '88f3a1'
LocalOrderMissing[…SOLUSDT cid=qubx_z9 status=OPEN qty=2.0]  present locally, absent from snapshot
OriginalOrderMissing[…XRPUSDT vid=77a2 status=OPEN qty=10.0]  present in snapshot, absent locally
PositionSizeMismatch[…BTCUSDT] size: 1.50 → 1.20 (Δ -0.30)
BalanceMismatch[USDT] free: 1000.00 → 980.00 (Δ -20.00)
VenueFiguresMismatch equity: 50000.00 → 49800.00 (Δ -200.00)
```

### Explicitly OUT of stage 1 (→ stage-2 merge)

`now`, state mutation, fetch-before-terminalize, covered-quantity deficit /
suppression, strategy callbacks, the `as_of` ratchet (stale-snapshot rejection). The
differ is pure except the grace gate.

## Test plan

Deterministic, no clock to mock; `as_of` and `seen_at` are set explicitly per case.
A scenario matrix (`tests/qubx/core/account_manager/snapshot_actions_test.py`):

**Orders**
- in sync → no atom
- price-only drift → `OrderPriceMismatch`
- filled-qty drift → `OrderFilledQtyMismatch`
- status drift → `OrderStatusMismatch`
- quantity (amend) drift → `OrderQuantityMismatch`
- avg-fill-price drift → `OrderAvgFillPriceMismatch`
- multi-field drift → multiple atoms for the same cid
- local-only past grace → `LocalOrderMissing`
- local-only within grace → **no atom**
- local-only changed after `as_of` → **no atom**
- local-only untimestamped → **no atom**
- terminal local order absent from snapshot → **no atom** (skipped)
- snapshot-only → `OriginalOrderMissing`
- cid-match with new venue id → `OrderVenueIdMismatch`
- drift below tolerance (sub-tick / sub-lot) → **no atom**

**Positions** — in sync / size / avg-price / margin / multi-field / sub-tolerance.
**Balances** — in sync / free drift / total drift / sub-tolerance / local-absent / origin-absent.
**Venue figures** — in sync / each figure drift / None-leg handling.
**Guards** — exchange mismatch raises `ValueError`; `__repr__` of every atom renders the expected line.

## Files

- `src/qubx/core/account_manager/diffs.py` — `Differ`, `@diffatom`, atom
  hierarchy, `diff()`. (Replaces the scaffold's `Reconciliator`.)
- `tests/qubx/core/account_manager/snapshot_actions_test.py` — the scenario matrix.
- `reconcile.py` stays untouched in stage 1 (still wired into the live path); it is
  retired when stage 2 lands.
