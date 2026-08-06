# update_order: total-quantity contract + connector-owned amend dialects

**Date:** 2026-08-06
**Status:** Design approved in discussion; supersedes PR #367 (`feat/order-replace`), which will be closed unmerged.
**Repos touched:** Qubx (this spec), exchanges (`qubx-hyperliquid` plugin), qubx-lighter (verification), quantkit (consumer follow-up).

## Problem

`ctx.update_order(price, amount)` is broken for partially-filled orders because nobody owns the
translation between the framework's idea of "amount" and each venue's amend dialect:

| Venue | Native amend | Quantity in the request means | If new qty ≤ filled |
|---|---|---|---|
| Binance UM (`PUT /fapi/v1/order`) | atomic modify, orderId kept | **new TOTAL**, executedQty preserved | **silent cancel** (documented) |
| Gate futures (`PUT /futures/{settle}/orders/{id}`) | atomic modify | **new TOTAL "including filled part"** (documented) | **silent cancel** (documented) |
| Hyperliquid (`modify` action) | cancel+replace at the matching engine, new oid, cloid kept | **size of the replacement order** (= desired remaining; fills restart at 0). High-confidence inference from ccxt's `sz`/`origSz` mapping + schema reuse — **needs one testnet probe before merge** | not documented |
| Lighter (signed `modify_order` tx) | atomic modify (price+size in one tx) | **unverified** | unverified |

Today the trading mixin passes `abs(amount)` verbatim to the connector, and connectors pass it
verbatim to the venue. The one production caller in the entire ecosystem (quantkit
`maker_taker/executor.py:461`, reprice-only) passes `signed_remaining(live)` — which double-shrinks
on Binance/Gate (total dialect) and was venue-correct on HL only by accident. On HL the amend ack
then overwrote `Order.quantity` with the remaining while `filled_quantity` stayed cumulative, so
`remaining = quantity − filled` went negative — the 2026-08-05 ACE incident.

PR #367 fixed this with `amount = desired signed remaining` plus a framework-side cancel+replace
orchestration (ReplaceIntent) for `filled > 0`. Review verdict: over-engineered relative to real
usage — no caller anywhere resizes a working order; every venue we trade has a native atomic
modify; and the orchestration turned Binance's race-free atomic amend into a two-round-trip
cancel-gap-replace with its own failure modes (expiry, suppression, superseded-oid markers) and two
ledgered deferrals.

## Design

One principle: **the framework speaks one dialect — quantity is always the order's new TOTAL,
including everything already filled — and each connector translates to its venue's dialect in both
directions.** No orchestration layer; the venue's own atomic modify is used everywhere.

### 1. Contract (`ITradingManager.update_order` / trading mixin)

```python
def update_order(
    self,
    price: float | None = None,
    quantity: float | None = None,
    order_id: str | None = None,
    client_order_id: str | None = None,
    exchange: str | None = None,
) -> None
```

- At least one of `price` / `quantity` must be set, else `ValueError`.
- `quantity` is **unsigned, positive, the new TOTAL including filled**. The signed-amount
  convention is dropped: the order already knows its side, so sign carries no information.
  (`quantity <= 0` → `ValueError`.)
- **`quantity <= order.filled_quantity` → `ValueError`** (best-effort pre-check). This guards the
  documented Binance/Gate behavior of *silently cancelling* the order in that case — a shrink below
  filled is a cancel in disguise and must be requested as one. `quantity == filled` (remaining 0)
  is included: that's `cancel_order`, not an update.
- Unchanged from main: terminal order → `OrderAlreadyTerminal`; `PENDING_UPDATE` in flight → no-op;
  synchronous connector failure → synthetic `OrderUpdateRejectedEvent` (reverts `PENDING_UPDATE`)
  then re-raise.
- `_adjust_size` / `_adjust_price` run on the total when quantity is given; when `quantity is
  None`, no size adjustment happens (price-only update).

`IConnector.update_order(order, *, price=None, quantity=None)` keeps its signature; `quantity`
now carries total semantics and either field may be `None` (meaning "unchanged").

### 2. Event + reducer

- `OrderUpdatedEvent.new_quantity` means **new TOTAL, or `None` = unchanged**. Connectors echo the
  requested total (or `None`) — never a venue-dialect figure.
- Reducer on update-ack: `quantity = new_quantity if new_quantity is not None else unchanged`.
  Invariant guard: if the resulting `quantity < filled_quantity` (can only happen via a race or a
  buggy connector), log an error and clamp `quantity = filled_quantity` (remaining 0), then let the
  periodic snapshot / status refetch reconcile. Never let `remaining` go negative — that is the
  ACE poison and gets a regression test.
- No splice arithmetic, no ReplaceIntent table, no cancel suppression, no expiry machinery.

### 3. ccxt connector (Binance, Gate — and public-framework HL correctness)

- Resolve `None` fields from the Order before building the venue request (Binance requires both
  `quantity` and `price` on modify): `price or order.price`, `quantity or order.quantity`.
- **Dialect map** in the connector (it already holds the full `Order`, so `filled_quantity` is at
  hand): exchanges with replacement-dialect amend (currently `hyperliquid`) get
  `venue_amount = total − order.filled_quantity`; everything else (Binance, Gate) passes the total
  through verbatim — which ccxt already sends unmodified. We don't trade HL through ccxt, but the
  public framework must not ship an oversize landmine.
- **Silent-cancel detection**: after a successful `editOrder`, inspect the response's order status;
  if the venue reports the order CANCELED (the documented Binance/Gate response to an in-flight
  fill overtaking the new total), emit a truthful cancel event instead of `OrderUpdatedEvent`.
- **Fix the cid-only (pre-ack) edit path**, verified broken today:
  - Gate: `edit_order_request` does no `t-` prefix substitution → empty order-id in the URL path
    plus a stray `clientOrderId` body key. Fix in the qubx Gate subclass (mirror the
    `fetch_order`/`cancel_order` substitution) **or** reject pre-ack updates framework-side the way
    the HL plugin does (`InvalidOrderParameters`). Prefer the fix; the reject is the fallback.
  - ccxt-HL: `parse_to_int('')` crashes locally → every pre-ack update becomes UPDATE_REJECTED.
    Pre-ack reject with a clear message is sufficient here (we don't trade this path).
  - Binance already works via the existing `BinanceQV` override; keep its regression test.

### 4. exchanges repo — `qubx-hyperliquid` plugin (the connector we actually trade HL with)

`_build_modify_action` (hyperliquid/connector.py):

- `new_size = (quantity if quantity is not None else order.quantity) − order.filled_quantity`;
  raise `InvalidOrderParameters` if `new_size <= 0` (mirror of the mixin pre-check, read at build
  time so the window is smaller).
- `_do_update` already emits `new_quantity=quantity` verbatim — under the new contract that is
  exactly right (requested total, or `None`). No change.
- Everything else stays: pre-ack reject, trigger-order reject, new-oid re-keying with the
  WS-inside-REST race handling.
- **Merge gate:** one testnet probe confirming the modify-size = remaining inference (place, partial
  fill or verify sizing behavior, modify, read back `origSz`/`sz`). Lands next to the existing
  conformance/e2e plan (`docs/superpowers/plans/2026-08-05-cancel-fill-reporting-conformance.md`).

### 5. qubx-lighter (verification only, separate repo)

`_modify_order` sends `base_amount` in a signed modify tx; whether Lighter reads it as total or
remaining is unverified. Verify against Lighter docs/testnet and translate if needed. Also fix the
stale `update_order=False` conformance capability flag in the exchanges-monorepo copy.

### 6. Backtester

No change. The sim connector's cancel+recreate uses `quantity or old.quantity` as the new order's
full size, and the OME has no partial-fill matching, so a resting sim order always has
`filled == 0` — "new total" and "replacement size" coincide for every reachable state.

### 7. quantkit follow-up (separate PR, after the qubx release)

`executor.py:461` becomes price-only: `ctx.update_order(price=new_price, client_order_id=...,
exchange=...)`. Deletes the `signed_remaining` computation (the exact derived value the ACE
bookkeeping poison flowed through) and the signed-amount min-notional hack. Tests that encode
remaining-semantics (`test_amends_unfilled_remainder`,
`test_amend_sell_passes_signed_negative_amount`, integration repricing tests) are updated to
assert price-only calls.

## Accepted races (documented, not defended against)

- **Total-dialect venues (Binance/Gate):** an in-flight fill shrinks the effective remaining below
  what the caller computed → venue under-works or (fill overtakes the total) silently cancels. Both
  outcomes are truthful: fills are fills, and the silent cancel is detected (§3) and reported as a
  cancel. No corrupted bookkeeping is possible.
- **Replacement-dialect venues (HL):** an in-flight fill between reading `filled_quantity` and the
  venue processing the modify oversizes the replacement by that δ. Bounded by one round trip;
  identical to today's live behavior; quantkit's settle trim self-corrects the overshoot. The only
  way to eliminate it is a framework cancel+replace that reads the cancel-ack's final fill before
  sizing — exactly the PR #367 orchestration this design rejects as not worth its complexity.
- **Pre-existing, now guarded:** after an HL modify the venue counts the replacement's fills from
  zero, so a snapshot that wins the venue-newer race used to clobber cumulative `filled_quantity`
  with that lower figure. Under total semantics an un-guarded clobber doesn't just lose history —
  it OVER-states `remaining` (`total − filled`), feeding an over-sized next amend. `_apply_order_snapshot`
  now adopts `filled_quantity` monotonically (never decreases it), closing that hole. The residual is
  cosmetic: `OrderQuantityMismatch` diff noise against the venue's own post-modify figures until
  venue-id-aware adoption (`filled_base` offset — separate snapshot-semantics design) lands; still
  deferred.

## Salvaged from PR #367

- The two backtester sim tests (reprice-unfilled-limit through the real OME with book-truth
  assertions and an intermediate no-fill quote), adapted to the new signature.
- Reducer regression tests, adapted: the never-negative-remaining invariant, update-ack semantics.
- The `docs/account-management/design.md` section, rewritten for this design.
- Dropped: ReplaceIntent state + reconciler decisions + `_execute` replace path + expiry recovery +
  suppression/superseded-oid machinery + `OrderCanceledEvent.venue_filled_quantity` plumbing (that
  enrichment only served the orchestration; re-introduce independently if ever needed).

## Testing

- **Unit (qubx):** mixin validation (both-None, non-positive, `<= filled`, terminal,
  PENDING_UPDATE no-op, sync-failure revert); reducer total-ack + clamp invariant; ccxt connector
  dialect map (HL subtraction incl. `None` defaults), silent-cancel detection, Gate cid-only fix
  (wire-level request assertion like the existing BinanceQV test).
- **Sim (qubx):** the two salvaged backtester tests.
- **exchanges:** unit tests on `_build_modify_action` size math (`None` quantity → current
  remaining; shrink-below-filled raises); testnet probe as merge gate.
- **quantkit:** executor tests flip to price-only assertions.

## Rollout order

1. Qubx PR (contract + reducer + ccxt) → release via dev pipeline.
2. exchanges `qubx-hyperliquid` PR (translation + testnet probe) — gated on the HL semantics probe.
3. quantkit PR (price-only executor) pinned to the new qubx version.
4. qubx-lighter verification issue.
5. Close PR #367 with a pointer to this spec once the Qubx PR is open.

**Deployment coupling:** the `amount` → `quantity` parameter rename is a deliberate loud break —
an old quantkit calling `update_order(amount=...)` fails with `TypeError` at call time (surfaced
as the executor's amend-failure warning, maker chase stops repricing) instead of silently sending
the wrong dialect. Strategy deployments must therefore bump qubx and quantkit **together** (frab
pins both; step 3's quantkit release is the pairing partner). No compatibility alias.
