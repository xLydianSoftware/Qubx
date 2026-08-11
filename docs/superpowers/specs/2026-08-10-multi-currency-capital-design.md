# Multi-currency capital: account state, transfers, and reporting

Date: 2026-08-10
Status: approved, ready for planning
Target: single PR against `Qubx:main`

## Problem

A multi-exchange simulation whose venues settle in different currencies reports the wrong
account capital, sizes positions off that wrong number, and then hides the error in the
per-exchange reporting. Two independent defects, both reproducible today on `main`.

### Defect 1 — capital blind to non-base currencies (behavioural)

`SimulatedAccountManager` gives every venue the same base currency
(`backtester/runner.py:611`, `base_currencies={exchange: setup.base_currency}`) and seeds each
venue's balance in it (`runner.py:622-629`). Money, however, books to the *instrument's* settle
currency:

- realized PnL and fees — `core/account_manager/reducer.py:320`,
  `adjust_balance(instrument.settle, realized_pnl - fee)`, which creates the balance on demand
- funding — `core/account_manager/manager.py:683`, same settle currency, but only when that
  balance already exists

`AccountState.total_capital()` (`core/account_manager/state.py:186-195`) reads exactly one
balance: `self._balances.get(self.base_currency)`. On a USDC-settled venue seeded in USDT, every
dollar the venue earns lands in a USDC balance the metric cannot see. The venue's reported
capital stays pinned near its frozen USDT seed plus unrealized `market_value_funds`.

Strategies size off this number. In frab, `FundingPair.amount_for_pair_leverage` calls
`ctx.get_total_capital()`, so the sizing base inherits the blind spot.

Two existing TODOs mark the same seam: `reducer.py:315` ("revisit for instruments whose settle
currency differs from the portfolio base currency") and `state.py:245` ("convert settle/quote ->
base_currency via marks").

### Defect 2 — per-exchange equity reporting (read-side)

`TradingSessionResult.get_equity_per_exchange` (`core/metrics.py:766-837`) is wrong in two ways:

1. **Transfers are silently dropped.** Lines 824 and 829 gate on `if ts in transfer_amounts.index`
   — exact membership in the portfolio-log index. Portfolio logs are sampled on a bar boundary;
   transfers fire on strategy schedules, so their timestamps rarely coincide. The function also
   counts rows regardless of `status`, so a pending or failed transfer would book as settled.
2. **Per-exchange initial capital is wrong after a storage round-trip.** Line 803 reads
   `self.capital[exchange] if isinstance(self.capital, dict) else self.capital`, treating a scalar
   as *each* venue's seed rather than the total.

The simulator itself is correct: `SimulationSetup.__post_init__` (`backtester/utils.py:106-110`)
normalizes a scalar into a per-exchange dict, and `backtester/simulator.py:327` passes that dict
through. The dict is destroyed by persistence:

- `utils/results.py:234` declares the metadata column as `pa.float64()`
- `utils/results.py:656` writes `float(result.get_total_capital())`
- `utils/results.py:918` reads back `capital=float(meta.get("capital", 0.0))`

So an in-memory result plots correctly and a reloaded one does not. `core/metrics.py:713`
flattens the same way when slicing, recomputing `capital` as `float(equity at the cut)`.

## Evidence

Diagnosed on frab backtest `FARB_V4/v11_00_fp1d_to05/20260716_001508` (qubx 1.11.7,
BINANCE.UM ⇄ HYPERLIQUID.F, `capital: 100000`, 2025-11-01 → 2026-07-10).

Sizing base recovered from `executions_log` and fitted against candidate quantities:

| candidate | median error | correlation |
|---|---|---|
| `2 × BINANCE equity` | +2.2% | +0.69 |
| `BINANCE + HYPERLIQUID` (true total) | +25.9% | **−0.76** |
| `2 × HYPERLIQUID equity` | +51.3% | −0.75 |

The base tracked the USDT venue and was *anti-correlated* with true total equity. It fell
114k → 59k while the account grew 100k → 123k. Position count stayed flat (~9.8 legs per venue)
while per-leg notional halved, and reported gross leverage decayed from ~400% to ~170%.

For defect 2 on the same run: 17 completed transfers, every timestamp at `HH:02:00` against an
hourly log — zero matched, so none appeared in the chart. Per-exchange equity summed to
$223,402 against a true equity of $123,402, exactly one scalar capital too much.

## Goals

- `get_total_capital()` reflects all of a venue's money regardless of settle currency
- Simulated venues hold the currency they could actually hold
- Cross-venue transfers model the real withdraw-swap-deposit
- Per-exchange equity reconciles with total equity, transfers visible, for both fresh and
  already-saved results
- Live behaviour unchanged

## Non-goals

- Marks-based currency conversion. `conversion_rate_to_base` answers 1.0 for cash currencies and
  `None` elsewhere; the seam is wired and typed, not filled.
- A live `ITransferManager`. Qubx ships only the simulated one; live is injected by the strategy
  (frab supplies quantkit's `XChangesTransferService`).
- Changing the funding guard at `manager.py:682`. See "Deliberately unchanged".

## Design

### 1. Per-exchange base currency

`SimulationSetup.base_currency` accepts `str | dict[str, str]` and normalizes to a per-exchange
dict in `__post_init__`, beside the existing capital split (`backtester/utils.py:106-110`).
Resolution order per venue:

1. explicit mapping entry, if given
2. otherwise, the shared `settle` of that venue's configured instruments, when they agree *and*
   that currency is a recognized stable
3. otherwise, the scalar `base_currency`

An unrecognized settle currency can't be valued at par, so a BTC-settled venue
(`BINANCE.UM:ETHBTC`) falls through to the scalar instead of being seeded 100,000 BTC counted as
capital.

`runner.py:611` passes the resolved dict to `SimulatedAccountManager` instead of broadcasting one
string; `runner.py:622-629` seeds each venue in its own currency.

Existing scalar configs are unaffected. A BINANCE.UM/HYPERLIQUID.F config resolves to
USDT/USDC from its instruments with no config change. Seeding runs once at init, before any
dynamic universe exists, so resolution reads only the configured instrument list.

YAML form for the override:

```yaml
simulation:
  base_currency:
    BINANCE.UM: USDT
    HYPERLIQUID.F: USDC
```

### 2. Currency-agnostic total capital

`AccountState.total_capital()` (`state.py:186`) iterates **every** balance instead of reading
`self.base_currency` alone, valuing each through a rate lookup that is allowed to answer "unknown":

```python
def conversion_rate_to_base(self, currency: str) -> float | None:
    """Rate from `currency` to this account's base currency, or None when unknown."""

def total_capital(self) -> float:
    venue = self._venue_figures
    if venue is not None and venue.equity is not None:
        return venue.equity
    cash = sum(
        b.total * rate
        for c, b in self._balances.items()
        if (rate := self.conversion_rate_to_base(c)) is not None
    )
    return cash + sum(p.market_value_funds for p in list(self._positions.values()))
```

The venue-reported-equity short-circuit stays ahead of it untouched, so live continues to prefer
`_venue_figures.equity`.

Summing *all* balances at 1.0 would be wrong, not merely imprecise: spot fills credit the base
asset (`reducer.py:325`, `adjust_balance(instrument.base, deal.amount)`), so a flat sum values a
million PEPE at a million dollars. Today's defect understates capital by a bounded amount; that
would overstate it without bound. `tests/qubx/core/account_manager/state_metrics_test.py:61-70`
already pins the correct behaviour and must keep passing unedited.

The rate table therefore starts narrow and widens later without moving a call site: today it
answers 1.0 for the venue's cash currencies and `None` for everything else; when marks-based
conversion lands (the `state.py:245` TODO) it answers for everything. PEPE is excluded because it
is unpriced, not because it is special-cased.

Cash currencies are tracked explicitly: a persisted `_cash_currencies: set[str]` on `AccountState`
(added to `__slots__`, `state.py:49-67`), seeded with `base_currency` and extended wherever a
settle balance is adjusted (`reducer.py:320`, `manager.py:683`), via `mark_cash_currency`, which
only admits recognized stables — the account's own `base_currency` stays exempt since it is
seeded in at construction. Persisting rather than deriving from open positions means a flat
venue's leftover USDC keeps counting.

This also closes the mixed-settle hole that per-exchange seeding cannot: a BINANCE.UM carrying
both USDT-M and USDC-M perps resolves to a single base currency, and the USDC-M side's PnL is only
visible because USDC is a settle currency.

### 3. Converting transfers (simulation)

`SimulationTransferManager.transfer_funds` (`backtester/transfers.py:21-46`) resolves each side's
base currency from the account, debits the source in its currency, credits the destination in its
currency at `conversion_rate`, and records both currencies and both amounts on the `Transfer`.
The `currency` argument becomes the *source* currency. The insufficient-funds check reads the
source venue's balance in that currency.

This models withdraw USDC from HPL → arrive as USDT on Binance. It is a simulation model only:
`runner.py:634` wires this manager for backtests and paper twins, while live uses whatever the
strategy injected (`runner.py:1058`).

### 4. Reporting — transfer alignment

Replace exact membership (`metrics.py:824`, `:829`) with as-of alignment: build signed per-venue
deltas indexed by transfer timestamp, `cumsum()` in transfer-time order, then
`reindex(portfolio_index, method="ffill").fillna(0.0)`.

This resolves every failure mode at once — off-grid stamps land on the following bar, several
transfers inside one bar sum instead of colliding, transfers before the first bar still count,
and transfers after the last bar drop out.

Two corrections ride along: filter to completed transfers, and apply the debit amount to the
source venue and the credit amount to the destination now that the two can differ.

### 5. Reporting — capital round-trip

Persist the resolved split. Keep the `capital` float column as the total so existing DuckDB
`storage.search()` queries keep working, and add a `capital_by_exchange` JSON column
(`utils/results.py:234`, `:656`, `:918`) that the loader prefers.

When the column is absent — every result saved so far — split the scalar evenly across
`exchanges`, matching what `SimulationSetup` has done since March 2026 (`69ed8448`). Results
already in storage then render correctly without a rerun. Multi-exchange results produced
*before* that commit were seeded differently and the even split will misattribute them; they are
old enough not to warrant a version-conditional path.

Slicing (`metrics.py:713`) carries the shape instead of collapsing it: per-venue capital at the
cut, taken from `get_equity_per_exchange`.

### Invariant

`get_equity_per_exchange().sum(axis=1) == get_equity()` at every bar. This catches the
capital-split defect: a scalar capital leaking into a multi-exchange result doubles the
per-exchange sum against total equity (verified; on the run above it reads $223,402 against
$123,402). It does not catch the transfer defect — a transfer's two legs are exact negatives
landing on the same bar, so a value-preserving transfer leaves the sum unchanged whether or not
it is aligned correctly. That defect is caught separately, by asserting the transferred amount
actually lands on the source and destination venues' own equity curves.

### Deliberately unchanged

The funding guard at `manager.py:682` ("only an existing settle balance is adjusted"). With
per-exchange seeding the settle balance exists from init, so the guard is satisfied. It exists so
funding never invents a wallet, and loosening it would mask a genuinely mis-seeded account.

## Testing

Account and currency:

- `SimulationSetup` resolution table: scalar broadcast, explicit override, single-settle
  derivation, mixed-settle fallback
- `total_capital()` counts a non-base *settle* balance, still excludes an unpriced asset balance
  (the existing PEPE pin passes unedited), and the venue-figures short-circuit still wins
  (this is what keeps live unchanged)
- `_cash_currencies` grows when a settle balance is adjusted and keeps counting after the
  position closes
- converting transfers, extending `tests/qubx/backtester/test_transfers.py`: source debited in
  its currency, destination credited in its own, both amounts recorded, insufficient funds
  checked against the source balance

Regression that reproduces the original defect: a two-venue sim, USDT and USDC, one hedged pair,
funding enabled, asserting `ctx.get_total_capital()` tracks total equity. The quantity is
anti-correlated with equity before the fix, so this fails hard on `main`.

Reporting, in `tests/qubx/core/metrics_test.py`:

- the sum-equals-total invariant in memory, after a storage round-trip, after slicing, and with
  transfers present
- transfer alignment against off-grid stamps, two in one bar, before the first bar, after the
  last bar, and non-completed statuses
- legacy metadata with no `capital_by_exchange` splits evenly

## Compatibility

Summing one balance equals reading it, so same-currency venue pairs are unchanged — BINANCE⇄GATEIO
simulations produce identical results. Only mixed-settle pairs move: today BINANCE.UM ⇄
HYPERLIQUID.F, plus Lighter if it is simulated.

Paper twins share `SimulationTransferManager` and therefore pick up converting transfers, which is
intended — paper should mirror simulation.

Live venues that report equity are unaffected: the `_venue_figures` short-circuit runs ahead of
the sum, and no live transfer manager ships in qubx. A live venue with no reported equity falls
back to the derived path, which now counts every balance instead of one — more complete, and the
same correction being made for simulation.

## Rollout

One PR against `Qubx:main`, organized as separate commits (account/currency, transfers,
reporting/persistence, tests) so review stays navigable. Merging `main` cuts a stable release.

1. merge the PR, wait for the release pipeline
2. bump frab's qubx pin to the released version
3. frab PR: `BalanceDifferenceRebalancer` stops hardcoding `target_currency = "USDT"`
   (`components/rebalancer.py:31`) and uses the source venue's base currency
4. rerun `v11_00_fp1d_to05`

Expect a genuinely different result rather than a rescaled one: gross leverage flat near the
configured 2.0× instead of decaying 4.0 → 1.7, with Sharpe and CAGR moving accordingly. That run
also predates frab's `e417def` leverage² fix (merged 2026-07-20), which the rerun picks up.

## Open question (non-blocking)

Whether quantkit's `XChangesTransferService` can bridge USDC → USDT in live. Step 3 changes what
frab asks for in production as well as simulation. Asking to move USDT off a USDC-settled venue is
likely already wrong today, so this is a latent live issue to confirm rather than one this design
introduces.
