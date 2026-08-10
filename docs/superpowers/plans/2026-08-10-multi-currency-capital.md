# Multi-currency capital Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a multi-exchange simulation whose venues settle in different currencies report and size against the correct account capital, and make per-exchange equity reporting reconcile with total equity.

**Architecture:** Each simulated venue gets its own base currency, resolved from its instruments' settle currency with an explicit override. `AccountState.total_capital()` iterates every balance and values each through a rate lookup that returns `None` for unpriced assets, so non-base settle balances count while spot holdings do not. Cross-venue transfers debit the source in its currency and credit the destination in its own. On the reporting side, transfers align as-of onto the portfolio-log index and the per-exchange capital split survives the storage round-trip.

**Tech Stack:** Python 3.12, pandas, pyarrow/parquet, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-08-10-multi-currency-capital-design.md`

## Global Constraints

- Python `>=3.12,<4.0`. Modern types only: `list`, `dict`, `X | None`, `tuple`. Never `from __future__ import annotations`.
- ruff, `line-length = 120`.
- Logging via `from qubx import logger`.
- Conventional commits, no co-authored-by lines.
- Run tests with `uv run pytest <path> -v`. Full suite: `just test`.
- Single PR against `Qubx:main`; each task below is one commit.
- `tests/qubx/core/account_manager/state_metrics_test.py:61-70` (the PEPE pin) must keep passing **unedited**. If a change requires editing it, the change is wrong.
- Comments state only the non-obvious. No section-divider banners, no rationale essays.

---

### Task 1: Resolve a base currency per exchange in SimulationSetup

**Files:**
- Modify: `src/qubx/backtester/utils.py:1-2` (import `field`), `:85-113` (`SimulationSetup`), `:346` (`recognize_simulation_configuration` annotation)
- Modify: `src/qubx/backtester/simulator.py:41` (`simulate` annotation), `src/qubx/utils/runner/configs.py:249` (`SimulationConfig.base_currency` annotation)
- Test: `tests/qubx/backtester/test_base_currency_seeding.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SimulationSetup.base_currencies: dict[str, str]` — resolved currency per exchange, populated in `__post_init__`. `SimulationSetup.base_currency` stays a `str` after init (display/persistence) even when a dict was passed in. Module-level `_derive_settle_currency(instruments: list[Instrument], exchange: str) -> str | None`.

**Why the annotations:** a YAML mapping reaches `SimulationSetup` through
`SimulationConfig.base_currency` → `runner.py:1219` (`sim_params["base_currency"] = sim.base_currency`)
→ `simulate(base_currency=...)` → `recognize_simulation_configuration(basic_currency=...)`. The
values pass through untouched; only the three `str` annotations along that chain reject a dict.
The live path already resolves per-exchange currencies (`runner.py:759-770`), so this brings
simulation in line with it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/backtester/test_base_currency_seeding.py`:

```python
def _multi_setup(base_currency, instruments) -> SimulationSetup:
    return SimulationSetup(
        setup_type=SetupTypes.STRATEGY,
        name="test",
        generator=None,
        tracker=None,
        instruments=instruments,
        exchanges=["BINANCE.UM", "HYPERLIQUID.F"],
        capital=100_000.0,
        base_currency=base_currency,
    )


def _swap(exchange: str, symbol: str, settle: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol[:3],
        quote=settle,
        settle=settle,
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def test_base_currency_derived_per_exchange_from_settle():
    setup = _multi_setup(
        "USDT",
        [_swap("BINANCE.UM", "BTCUSDT", "USDT"), _swap("HYPERLIQUID.F", "BTCUSDC", "USDC")],
    )
    assert setup.base_currencies == {"BINANCE.UM": "USDT", "HYPERLIQUID.F": "USDC"}


def test_explicit_mapping_overrides_derivation():
    setup = _multi_setup(
        {"HYPERLIQUID.F": "usdc"},
        [_swap("BINANCE.UM", "BTCUSDT", "USDT"), _swap("HYPERLIQUID.F", "BTCUSDC", "USDC")],
    )
    assert setup.base_currencies == {"BINANCE.UM": "USDT", "HYPERLIQUID.F": "USDC"}
    assert isinstance(setup.base_currency, str)


def test_mixed_settle_venue_falls_back_to_scalar():
    setup = _multi_setup(
        "USDT",
        [_swap("BINANCE.UM", "BTCUSDT", "USDT"), _swap("BINANCE.UM", "BTCUSDC", "USDC")],
    )
    assert setup.base_currencies["BINANCE.UM"] == "USDT"


def test_no_instruments_keeps_scalar_for_every_exchange():
    setup = _multi_setup("usdt", [])
    assert setup.base_currencies == {"BINANCE.UM": "USDT", "HYPERLIQUID.F": "USDT"}
```

Add `Instrument` and `MarketType` to the existing `from qubx.core.basics import Balance` line.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/backtester/test_base_currency_seeding.py -v`
Expected: FAIL — `AttributeError: 'SimulationSetup' object has no attribute 'base_currencies'`

- [ ] **Step 3: Implement resolution**

In `src/qubx/backtester/utils.py`, change the import on line 2 to `from dataclasses import dataclass, field`, add the helper above the `SimulationSetup` class:

```python
def _derive_settle_currency(instruments: list[Instrument], exchange: str) -> str | None:
    """The one settle currency a venue's instruments agree on, or None when absent or mixed."""
    settles = {i.settle.upper() for i in instruments if i.exchange == exchange and i.settle}
    return settles.pop() if len(settles) == 1 else None
```

Change the field declaration (`:96`) to `base_currency: str | dict[str, str]`, add below `enable_funding` (`:100`):

```python
    base_currencies: dict[str, str] = field(init=False, default_factory=dict)
```

and replace `__post_init__` (`:102-110`) with:

```python
    def __post_init__(self) -> None:
        # Per-exchange base currency: explicit mapping wins, else the venue's shared settle
        # currency, else the scalar. AccountState upper-cases, so normalize here or the seeded
        # Balance lands under a key total_capital never reads.
        explicit = (
            {ex: ccy.upper() for ex, ccy in self.base_currency.items()}
            if isinstance(self.base_currency, dict)
            else {}
        )
        default_ccy = (
            self.base_currency.upper()
            if isinstance(self.base_currency, str)
            else (explicit.get(self.exchanges[0], "USDT") if self.exchanges else "USDT")
        )
        self.base_currencies = {
            ex: explicit.get(ex) or _derive_settle_currency(self.instruments, ex) or default_ccy
            for ex in self.exchanges
        }
        self.base_currency = self.base_currencies[self.exchanges[0]] if self.exchanges else default_ccy
        # Normalize capital to a per-exchange dict: split evenly when a single float is given
        if isinstance(self.capital, (int, float)):
            n = len(self.exchanges)
            per_exchange = float(self.capital) / n if n > 0 else float(self.capital)
            self.capital = {exchange: per_exchange for exchange in self.exchanges}
```

- [ ] **Step 4: Widen the annotations along the config chain**

Three annotation-only edits, no logic changes:

- `src/qubx/utils/runner/configs.py:249`: `base_currency: str | dict[str, str] | None = None`
- `src/qubx/backtester/simulator.py:41`: `base_currency: str | dict[str, str] = "USDT"`
- `src/qubx/backtester/utils.py:346`: `basic_currency: str | dict[str, str],`

Add to `tests/qubx/backtester/test_base_currency_seeding.py`:

```python
def test_simulation_config_accepts_per_exchange_mapping():
    from qubx.utils.runner.configs import SimulationConfig

    cfg = SimulationConfig(
        capital=100_000.0,
        instruments=["BINANCE.UM:SWAP:BTCUSDT", "HYPERLIQUID.F:SWAP:BTCUSDC"],
        start="2026-01-01",
        stop="2026-02-01",
        data={"storage": "qdb::quantlab"},
        base_currency={"HYPERLIQUID.F": "USDC"},
    )
    assert cfg.base_currency == {"HYPERLIQUID.F": "USDC"}
```

If `SimulationConfig` requires fields beyond these, copy them from an existing config fixture in
`tests/qubx/utils/` rather than guessing.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/backtester/test_base_currency_seeding.py -v`
Expected: PASS, including the two pre-existing tests in that file.

- [ ] **Step 6: Commit**

```bash
git add src/qubx/backtester/utils.py src/qubx/backtester/simulator.py src/qubx/utils/runner/configs.py tests/qubx/backtester/test_base_currency_seeding.py
git commit -m "feat(backtester): resolve base currency per exchange in SimulationSetup"
```

---

### Task 2: Seed each venue in its own currency

**Files:**
- Modify: `src/qubx/backtester/utils.py` (add `initial_balances`), `src/qubx/backtester/runner.py:609-630`
- Test: `tests/qubx/backtester/test_base_currency_seeding.py`

**Interfaces:**
- Consumes: `SimulationSetup.base_currencies` from Task 1.
- Produces: `initial_balances(setup: SimulationSetup) -> dict[str, Balance]` in `qubx.backtester.utils` — exchange → the `Balance` to seed. The runner calls it instead of building balances inline, so seeding is testable without constructing a runner.

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/backtester/test_base_currency_seeding.py`:

```python
def test_initial_balances_seed_each_venue_in_its_own_currency():
    setup = _multi_setup(
        "USDT",
        [_swap("BINANCE.UM", "BTCUSDT", "USDT"), _swap("HYPERLIQUID.F", "BTCUSDC", "USDC")],
    )
    balances = initial_balances(setup)
    assert balances["BINANCE.UM"].currency == "USDT"
    assert balances["HYPERLIQUID.F"].currency == "USDC"
    assert balances["BINANCE.UM"].total == 50_000.0
    assert balances["HYPERLIQUID.F"].free == 50_000.0


def test_seeded_capital_visible_per_venue():
    setup = _multi_setup(
        "USDT",
        [_swap("BINANCE.UM", "BTCUSDT", "USDT"), _swap("HYPERLIQUID.F", "BTCUSDC", "USDC")],
    )
    am = SimulatedAccountManager(
        connectors={ex: MagicMock() for ex in setup.exchanges},
        base_currencies=setup.base_currencies,
        time=_Clock(),
    )
    for exchange, balance in initial_balances(setup).items():
        am.seed_balance(exchange, balance)

    assert am.get_total_capital("HYPERLIQUID.F") == 50_000.0
    assert am.get_total_capital() == 100_000.0
```

Add `initial_balances` to the `from qubx.backtester.utils import ...` line.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/backtester/test_base_currency_seeding.py -v -k initial_balances`
Expected: FAIL with `ImportError: cannot import name 'initial_balances'`

- [ ] **Step 3: Implement and wire the runner**

Add to `src/qubx/backtester/utils.py`, below `SimulationSetup`:

```python
def initial_balances(setup: SimulationSetup) -> dict[str, Balance]:
    """Startup balance per exchange, each in that venue's own base currency."""
    assert isinstance(setup.capital, dict)
    return {
        exchange: Balance(
            exchange=exchange,
            currency=setup.base_currencies[exchange],
            total=capital,
            free=capital,
            locked=0.0,
        )
        for exchange, capital in setup.capital.items()
    }
```

Add `Balance` to the `from qubx.core.basics import (...)` block (`:10-18`) if absent.

In `src/qubx/backtester/runner.py`, replace lines 609-630 with:

```python
        am = SimulatedAccountManager(
            connectors=self._connectors,
            base_currencies=self.setup.base_currencies,
            time=time_provider,
            cfg=AccountManagerConfig(),
            account_id=self.account_id,
            tcc=_exchange_to_tcc,
        )

        # - seed initial capital per exchange into the account state
        for exchange, balance in initial_balances(self.setup).items():
            am.seed_balance(exchange, balance)
```

Add `initial_balances` to the existing `from qubx.backtester.utils import ...` in `runner.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/backtester/test_base_currency_seeding.py -v && uv run pytest tests/qubx/backtester -v -x`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/backtester/utils.py src/qubx/backtester/runner.py tests/qubx/backtester/test_base_currency_seeding.py
git commit -m "feat(backtester): seed each simulated venue in its own base currency"
```

---

### Task 3: Value every balance through a rate lookup that may answer "unknown"

**Files:**
- Modify: `src/qubx/core/account_manager/state.py:49-67` (`__slots__`), `:69-80` (`__init__`), `:186-195` (`total_capital`)
- Test: `tests/qubx/core/account_manager/state_metrics_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `AccountState.mark_cash_currency(currency: str) -> None` and `AccountState.conversion_rate_to_base(currency: str) -> float | None`. `total_capital()` keeps its signature and its venue-equity short-circuit.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/core/account_manager/state_metrics_test.py`:

```python
def test_total_capital_counts_marked_settle_currency():
    state = _state(1000.0)  # base USDT
    state.mark_cash_currency("USDC")
    state.update_balance("USDC", Balance(exchange="binance", currency="USDC", free=250.0, locked=0.0, total=250.0))
    assert state.total_capital() == 1250.0


def test_marked_cash_currency_survives_position_close():
    state = _state(1000.0)
    state.mark_cash_currency("USDC")
    state.update_balance("USDC", Balance(exchange="binance", currency="USDC", free=250.0, locked=0.0, total=250.0))
    assert not state.get_positions()
    assert state.total_capital() == 1250.0


def test_conversion_rate_unknown_for_unpriced_asset():
    state = _state(1000.0)
    assert state.conversion_rate_to_base("USDT") == 1.0
    assert state.conversion_rate_to_base("PEPE") is None


def test_venue_equity_still_wins_over_summed_cash():
    state = _state(1000.0)
    state.mark_cash_currency("USDC")
    state.update_balance("USDC", Balance(exchange="binance", currency="USDC", free=250.0, locked=0.0, total=250.0))
    state.set_venue_figures(_venue(equity=5000.0))
    assert state.total_capital() == 5000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/account_manager/state_metrics_test.py -v`
Expected: FAIL — `AttributeError: 'AccountState' object has no attribute 'mark_cash_currency'`

- [ ] **Step 3: Implement**

In `src/qubx/core/account_manager/state.py`, add `"_cash_currencies",` to `__slots__` after `"_balances",`, and in `__init__` after the `_balances` line:

```python
        # Currencies this venue settles cash in: base plus any settle currency actually booked.
        # Persisted rather than derived from open positions so leftovers survive a flat book.
        self._cash_currencies: set[str] = {self.base_currency}
```

Replace `total_capital` (`:186-195`) and add the two helpers next to it:

```python
    def mark_cash_currency(self, currency: str) -> None:
        if currency:
            self._cash_currencies.add(currency.upper())

    def conversion_rate_to_base(self, currency: str) -> float | None:
        """Rate from `currency` to this account's base currency, or None when unknown.

        Cash currencies are stable-to-stable at 1.0; everything else is unpriced until
        marks-based conversion lands, and unpriced balances are excluded rather than
        counted at par — a spot base asset is not capital.
        """
        return 1.0 if currency.upper() in self._cash_currencies else None

    def total_capital(self) -> float:
        venue = self._venue_figures
        if venue is not None and venue.equity is not None:
            return venue.equity
        cash = sum(
            b.total * rate
            for c, b in list(self._balances.items())
            if (rate := self.conversion_rate_to_base(c)) is not None
        )
        return cash + sum(p.market_value_funds for p in list(self._positions.values()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/core/account_manager -v`
Expected: PASS, with `test_base_currency_explicit_not_inferred` passing unedited (PEPE is unpriced, so excluded).

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/account_manager/state.py tests/qubx/core/account_manager/state_metrics_test.py
git commit -m "feat(account): count every priced cash balance in total_capital"
```

---

### Task 4: Book settle currencies as cash on fills and funding

**Files:**
- Modify: `src/qubx/core/account_manager/reducer.py:316-320`, `src/qubx/core/account_manager/manager.py:681-683`
- Test: `tests/qubx/core/account_manager/reducer_test.py`

**Interfaces:**
- Consumes: `AccountState.mark_cash_currency` from Task 3.
- Produces: no new API. After any futures fill or funding settlement, `instrument.settle` is a cash currency on that venue.

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/account_manager/reducer_test.py`. It already has `_fill(trade_id, amount, price) -> Deal` (`:70`) and imports `reducer`, `AccountState`, and `Instrument`; add `MarketType` to the `from qubx.core.basics import (...)` block if absent. The instrument is built literally rather than via `lookup`, which has no USDC-settled fixtures:

```python
USDC_PERP = Instrument(
    symbol="BTCUSDC",
    market_type=MarketType.SWAP,
    exchange="hyperliquid",
    base="BTC",
    quote="USDC",
    settle="USDC",
    exchange_symbol="BTCUSDC",
    tick_size=0.01,
    lot_size=0.001,
    min_size=0.001,
    contract_size=1.0,
)


def test_futures_deal_marks_settle_currency_as_cash():
    # Round trip on a USDC-settled perp inside a USDT-based state: buy 1 @ 100, sell 1 @ 110.
    state = AccountState("hyperliquid", "USDT")
    reducer._book_deal(state, USDC_PERP, _fill(amount=1.0, price=100.0))
    reducer._book_deal(state, USDC_PERP, _fill(trade_id="t2", amount=-1.0, price=110.0))

    assert state.conversion_rate_to_base("USDC") == 1.0
    assert state.get_balance("USDC").total == pytest.approx(10.0)
    assert state.total_capital() == pytest.approx(10.0)  # flat book: cash only, no market value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/account_manager/reducer_test.py -v -k marks_settle`
Expected: FAIL — `total_capital()` returns 0.0 because USDC is unpriced on a USDT-based state.

- [ ] **Step 3: Implement**

In `reducer.py`, inside the `if instrument.is_futures():` branch, immediately before the existing `state.adjust_balance(instrument.settle, realized_pnl - fee)`:

```python
        state.mark_cash_currency(instrument.settle)
```

In `manager.py`, replace the funding balance block (`:681-683`) with:

```python
        # Only an existing settle balance is adjusted — funding never creates one.
        if state.get_balance(instrument.settle) is not None:
            state.mark_cash_currency(instrument.settle)
            state.adjust_balance(instrument.settle, amount)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/core/account_manager -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/account_manager/reducer.py src/qubx/core/account_manager/manager.py tests/qubx/core/account_manager/reducer_test.py
git commit -m "feat(account): mark settle currencies as cash on fills and funding"
```

---

### Task 5: Converting cross-venue transfers

**Files:**
- Modify: `src/qubx/core/basics.py:928-960` (`Transfer` fields and `to_dict`), `src/qubx/backtester/transfers.py:21-48`, `src/qubx/backtester/utils.py:643` and `:661-673` (transfers-log columns)
- Test: `tests/qubx/backtester/test_transfers.py`

**Interfaces:**
- Consumes: `AccountManager.get_base_currency(exchange)`, `AccountState.conversion_rate_to_base` from Task 3.
- Produces: `Transfer.to_currency: str | None` and `Transfer.to_amount: float | None` (both default `None`, meaning "same as source"). The transfers-log frame gains `to_currency` and `to_amount` columns. `transfer_funds(from_exchange, to_exchange, currency, amount)` keeps its signature; `currency` is now explicitly the **source** currency.

- [ ] **Step 1: Write the failing tests**

Append to `tests/qubx/backtester/test_transfers.py`:

```python
def _am_cross_currency():
    am = SimulatedAccountManager(
        connectors={"BINANCE.UM": MagicMock(), "HYPERLIQUID.F": MagicMock()},
        base_currencies={"BINANCE.UM": "USDT", "HYPERLIQUID.F": "USDC"},
        time=_T(),
    )
    am.get_state("BINANCE.UM").update_balance(
        "USDT", Balance(exchange="BINANCE.UM", currency="USDT", total=1000.0, free=1000.0)
    )
    am.get_state("HYPERLIQUID.F").update_balance(
        "USDC", Balance(exchange="HYPERLIQUID.F", currency="USDC", total=1000.0, free=1000.0)
    )
    return am


def test_transfer_debits_source_currency_credits_destination_currency():
    am = _am_cross_currency()
    mgr = SimulationTransferManager(am, _T())
    tid = mgr.transfer_funds("HYPERLIQUID.F", "BINANCE.UM", "USDC", 250.0)

    assert am.get_balance("USDC", "HYPERLIQUID.F").free == 750.0
    assert am.get_balance("USDT", "BINANCE.UM").free == 1250.0
    assert am.get_balance("USDC", "BINANCE.UM").total == 0.0

    t = mgr.get_transfer_status(tid)
    assert (t.currency, t.amount) == ("USDC", 250.0)
    assert (t.to_currency, t.to_amount) == ("USDT", 250.0)


def test_transfer_insufficient_funds_checks_source_currency():
    am = _am_cross_currency()
    mgr = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="Insufficient funds"):
        mgr.transfer_funds("HYPERLIQUID.F", "BINANCE.UM", "USDC", 5000.0)


def test_transfer_rejects_unpriced_currency():
    am = _am_cross_currency()
    am.get_state("HYPERLIQUID.F").update_balance(
        "PEPE", Balance(exchange="HYPERLIQUID.F", currency="PEPE", total=10.0, free=10.0)
    )
    mgr = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="no conversion rate"):
        mgr.transfer_funds("HYPERLIQUID.F", "BINANCE.UM", "PEPE", 1.0)


def test_transfers_log_carries_destination_columns():
    am = _am_cross_currency()
    mgr = SimulationTransferManager(am, _T())
    mgr.transfer_funds("HYPERLIQUID.F", "BINANCE.UM", "USDC", 100.0)
    df = collect_transfers_log(mgr)

    assert list(df.columns) == _TRANSFERS_LOG_COLUMNS
    assert df.iloc[0]["to_currency"] == "USDT"
    assert df.iloc[0]["to_amount"] == 100.0
```

Add `_TRANSFERS_LOG_COLUMNS` to the `from qubx.backtester.utils import ...` line.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/backtester/test_transfers.py -v`
Expected: FAIL — the destination is credited in USDC, and `Transfer` has no `to_currency`.

- [ ] **Step 3: Implement**

In `src/qubx/core/basics.py`, add to the `Transfer` dataclass after `failure_reason`:

```python
    to_currency: str | None = None  # destination currency when it differs from `currency`
    to_amount: float | None = None  # credited amount in `to_currency`
```

and include both in the `to_dict()` mapping alongside the existing fields.

Replace `transfer_funds` in `src/qubx/backtester/transfers.py:21-48`:

```python
    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        if amount <= 0:
            raise ValueError(f"Transfer amount must be positive, got {amount}")
        to_currency = self._account.get_base_currency(to_exchange)
        # Venues settle in their own currency: a withdrawal leaves in `currency` and arrives as
        # `to_currency`. Both legs must be priced — an unpriced asset has no transferable value.
        from_rate = self._account.get_state(from_exchange).conversion_rate_to_base(currency)
        to_rate = self._account.get_state(to_exchange).conversion_rate_to_base(to_currency)
        if from_rate is None or to_rate is None:
            raise ValueError(f"Cannot transfer {currency} to {to_exchange} {to_currency}: no conversion rate")
        to_amount = amount * from_rate / to_rate

        from_balance = self._account.get_balance(currency, exchange=from_exchange)
        if from_balance.free < amount:
            raise ValueError(
                f"Insufficient funds in {from_exchange}: "
                f"{from_balance.free:.8f} {currency} available, {amount:.8f} requested"
            )

        self._account.adjust_balance(from_exchange, currency, -amount)
        self._account.adjust_balance(to_exchange, to_currency, to_amount)

        transaction_id = f"sim_{uuid.uuid4().hex[:12]}"
        self._transfers[transaction_id] = Transfer(
            transaction_id=transaction_id,
            from_exchange=from_exchange,
            to_exchange=to_exchange,
            currency=currency,
            amount=amount,
            status=TransferStatus.COMPLETED,  # transfers are instant in simulation
            timestamp=self._time.time(),
            to_currency=to_currency,
            to_amount=to_amount,
        )
        logger.debug(
            f"[SimTransfer] {amount:.8f} {currency} {from_exchange} -> "
            f"{to_amount:.8f} {to_currency} {to_exchange} ({transaction_id})"
        )
        return transaction_id
```

In `src/qubx/backtester/utils.py`, extend the columns constant (`:643`) and the row dict (`:661-670`):

```python
_TRANSFERS_LOG_COLUMNS = [
    "transaction_id",
    "from_exchange",
    "to_exchange",
    "currency",
    "amount",
    "status",
    "to_currency",
    "to_amount",
]
```

```python
                "status": str(t.status),
                "to_currency": t.to_currency or t.currency,
                "to_amount": t.to_amount if t.to_amount is not None else t.amount,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/backtester/test_transfers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/basics.py src/qubx/backtester/transfers.py src/qubx/backtester/utils.py tests/qubx/backtester/test_transfers.py
git commit -m "feat(backtester): convert cross-venue transfers at the venue boundary"
```

---

### Task 6: Align transfers as-of onto the portfolio-log index

**Files:**
- Modify: `src/qubx/core/metrics.py:766-837` (`get_equity_per_exchange`), plus a module-level helper above it
- Test: `tests/qubx/core/test_session_result_equity.py` (create)

**Interfaces:**
- Consumes: the `to_amount` column from Task 5.
- Produces: `_transfer_offsets(transfers: pd.DataFrame, exchange: str, index: pd.DatetimeIndex) -> pd.Series` in `qubx.core.metrics` — cumulative net transfer effect on one venue, aligned onto `index`.

- [ ] **Step 1: Write the failing tests**

Create `tests/qubx/core/test_session_result_equity.py`:

```python
import pandas as pd

from qubx.core.metrics import _transfer_offsets

INDEX = pd.date_range("2026-01-01", periods=4, freq="1h")


def _transfers(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("timestamp")


def test_off_grid_transfer_lands_on_next_bar():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 100.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 100.0, 100.0]
    assert list(_transfer_offsets(tl, "A", INDEX)) == [0.0, 0.0, -100.0, -100.0]


def test_two_transfers_in_one_bar_sum():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 100.0,
                "status": "completed",
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 01:30"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 50.0,
                "to_amount": 50.0,
                "status": "completed",
            },
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 150.0, 150.0]


def test_transfer_before_first_bar_counts_from_the_start():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2025-12-31 23:00"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 10.0,
                "to_amount": 10.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [10.0, 10.0, 10.0, 10.0]


def test_transfer_after_last_bar_is_ignored():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-02 00:00"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 10.0,
                "to_amount": 10.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 0.0, 0.0]


def test_non_completed_transfers_excluded():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 100.0,
                "status": "pending",
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 7.0,
                "to_amount": 7.0,
                "status": "failed",
            },
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 0.0, 0.0]


def test_converted_transfer_credits_destination_amount():
    tl = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "A",
                "to_exchange": "B",
                "amount": 100.0,
                "to_amount": 99.0,
                "status": "completed",
            }
        ]
    )
    assert list(_transfer_offsets(tl, "B", INDEX)) == [0.0, 0.0, 99.0, 99.0]
    assert list(_transfer_offsets(tl, "A", INDEX)) == [0.0, 0.0, -100.0, -100.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/test_session_result_equity.py -v`
Expected: FAIL with `ImportError: cannot import name '_transfer_offsets'`

- [ ] **Step 3: Implement**

Add above `TradingSessionResult` in `src/qubx/core/metrics.py`:

```python
def _transfer_offsets(transfers: pd.DataFrame, exchange: str, index: pd.DatetimeIndex) -> pd.Series:
    """Cumulative net transfer effect on `exchange`, aligned as-of onto `index`.

    Transfer timestamps come from strategy schedules and rarely coincide with portfolio-log
    bars, so alignment is as-of (first bar at or after the transfer) rather than exact match.
    """
    if not {"from_exchange", "to_exchange", "amount"}.issubset(transfers.columns):
        return pd.Series(0.0, index=index)
    done = transfers
    if "status" in done.columns:
        done = done[done["status"].astype(str).str.lower() == "completed"]
    if done.empty:
        return pd.Series(0.0, index=index)

    credited = done["to_amount"].fillna(done["amount"]) if "to_amount" in done.columns else done["amount"]
    deltas = credited.where(done["to_exchange"] == exchange, 0.0) - done["amount"].where(
        done["from_exchange"] == exchange, 0.0
    )
    cumulative = deltas.groupby(level=0).sum().sort_index().cumsum()
    return cumulative.reindex(index, method="ffill").fillna(0.0)
```

Replace the transfer block inside `get_equity_per_exchange` (`:808-833`) with:

```python
            if with_transfers and self.transfers_log is not None and not self.transfers_log.empty:
                equity = equity + _transfer_offsets(self.transfers_log, exchange, self.portfolio_log.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/core/test_session_result_equity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/metrics.py tests/qubx/core/test_session_result_equity.py
git commit -m "fix(metrics): align transfers as-of onto the portfolio-log index"
```

---

### Task 7: Keep the per-exchange capital split across persistence and slicing

**Files:**
- Modify: `src/qubx/utils/results.py:234` (schema), `:630-660` (writer), `:905-925` (loader), `src/qubx/core/metrics.py:709-715` (slicing)
- Test: `tests/qubx/core/test_session_result_equity.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: metadata column `capital_by_exchange` (JSON string, `pa.string()`), and `_capital_from_meta(meta: dict, exchanges: list[str]) -> float | dict[str, float]` in `qubx.utils.results`. The `capital` float column keeps meaning "total" so existing DuckDB queries are unaffected.

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/test_session_result_equity.py`:

```python
from qubx.utils.results import _capital_from_meta


def test_capital_by_exchange_round_trips():
    meta = {"capital": 100_000.0, "capital_by_exchange": '{"BINANCE.UM": 60000.0, "HYPERLIQUID.F": 40000.0}'}
    assert _capital_from_meta(meta, ["BINANCE.UM", "HYPERLIQUID.F"]) == {
        "BINANCE.UM": 60_000.0,
        "HYPERLIQUID.F": 40_000.0,
    }


def test_legacy_scalar_capital_splits_evenly():
    meta = {"capital": 100_000.0}
    assert _capital_from_meta(meta, ["BINANCE.UM", "HYPERLIQUID.F"]) == {
        "BINANCE.UM": 50_000.0,
        "HYPERLIQUID.F": 50_000.0,
    }


def test_single_exchange_capital_stays_scalar():
    assert _capital_from_meta({"capital": 100_000.0}, ["BINANCE.UM"]) == 100_000.0


def test_malformed_capital_by_exchange_falls_back_to_even_split():
    meta = {"capital": 100_000.0, "capital_by_exchange": "not-json"}
    assert _capital_from_meta(meta, ["A", "B"]) == {"A": 50_000.0, "B": 50_000.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/test_session_result_equity.py -v -k capital`
Expected: FAIL with `ImportError: cannot import name '_capital_from_meta'`

- [ ] **Step 3: Implement**

In `src/qubx/utils/results.py`, add to the metadata schema right after `("capital", pa.float64()),`:

```python
            ("capital_by_exchange", pa.string()),  # - JSON string; absent on legacy results
```

In the metadata writer, beside the existing `"capital": float(_capital),`:

```python
            "capital_by_exchange": _json.dumps(result.capital) if isinstance(result.capital, dict) else "",
```

Add the module-level loader helper:

```python
def _capital_from_meta(meta: dict, exchanges: list[str]) -> float | dict[str, float]:
    """Per-exchange capital from metadata, falling back to an even split for legacy results.

    Legacy rows predate the `capital_by_exchange` column and store only the total; the sim has
    split a scalar evenly across venues since 69ed8448, so an even split reproduces what ran.
    """
    import json as _json

    raw = meta.get("capital_by_exchange") or ""
    if raw:
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                return {str(k): float(v) for k, v in parsed.items()}
        except (ValueError, TypeError):
            logger.warning(f"Ignoring malformed capital_by_exchange metadata: {raw!r}")
    total = float(meta.get("capital", 0.0))
    return {ex: total / len(exchanges) for ex in exchanges} if len(exchanges) > 1 else total
```

In the loader, replace `capital=float(meta.get("capital", 0.0)),` with `capital=_capital_from_meta(meta, _to_list(meta.get("exchanges"))),`.

In `src/qubx/core/metrics.py`, replace the slicing block (`:709-715`) with:

```python
        # Recompute capital at the cut, preserving the per-exchange split so
        # get_equity_per_exchange keeps reconciling with get_equity after a slice.
        if start is not None and not self.portfolio_log.empty:
            if isinstance(self.capital, dict) and len(self.exchanges) > 1:
                prior = self.get_equity_per_exchange().loc[:start]
                capital = {ex: float(prior[ex].iloc[-1]) for ex in prior.columns} if not prior.empty else self.capital
            else:
                prior_equity = self.get_equity().loc[:start]
                capital = float(prior_equity.iloc[-1]) if not prior_equity.empty else self.capital
        else:
            capital = self.capital
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/qubx/core/test_session_result_equity.py -v && uv run pytest tests/qubx/utils -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/utils/results.py src/qubx/core/metrics.py tests/qubx/core/test_session_result_equity.py
git commit -m "fix(results): persist per-exchange capital and preserve it when slicing"
```

---

### Task 8: Pin the sum-equals-total invariant

**Files:**
- Test: `tests/qubx/core/test_session_result_equity.py`

**Interfaces:**
- Consumes: `_transfer_offsets` (Task 6), `_capital_from_meta` (Task 7).
- Produces: nothing consumed downstream. This is the regression gate for both reporting defects.

- [ ] **Step 1: Write the failing-if-regressed tests**

Append to `tests/qubx/core/test_session_result_equity.py`:

```python
from qubx.core.metrics import TradingSessionResult


def _two_venue_result(capital, transfers: pd.DataFrame | None = None) -> TradingSessionResult:
    idx = INDEX
    portfolio = pd.DataFrame(
        {
            "BINANCE.UM:BTCUSDT_PnL": [0.0, 10.0, 10.0, 10.0],
            "BINANCE.UM:BTCUSDT_Commissions": [0.0, 1.0, 0.0, 0.0],
            "BINANCE.UM:BTCUSDT_Value": [1000.0, 1000.0, 1000.0, 1000.0],
            "HYPERLIQUID.F:BTCUSDC_PnL": [0.0, -10.0, -10.0, -10.0],
            "HYPERLIQUID.F:BTCUSDC_Commissions": [0.0, 1.0, 0.0, 0.0],
            "HYPERLIQUID.F:BTCUSDC_Value": [-1000.0, -1000.0, -1000.0, -1000.0],
        },
        index=idx,
    )
    return TradingSessionResult(
        id=0,
        name="t",
        start=idx[0],
        stop=idx[-1],
        exchanges=["BINANCE.UM", "HYPERLIQUID.F"],
        instruments=[],
        capital=capital,
        base_currency="USDT",
        commissions=None,
        portfolio_log=portfolio,
        executions_log=pd.DataFrame(),
        signals_log=pd.DataFrame(),
        targets_log=pd.DataFrame(),
        strategy_class="test",
        transfers_log=transfers,
    )


def _assert_reconciles(result: TradingSessionResult) -> None:
    per_exchange = result.get_equity_per_exchange().sum(axis=1)
    pd.testing.assert_series_equal(per_exchange, result.get_equity(), check_names=False)


def test_per_exchange_equity_reconciles_in_memory():
    _assert_reconciles(_two_venue_result({"BINANCE.UM": 50_000.0, "HYPERLIQUID.F": 50_000.0}))


def test_per_exchange_equity_reconciles_after_legacy_scalar_load():
    capital = _capital_from_meta({"capital": 100_000.0}, ["BINANCE.UM", "HYPERLIQUID.F"])
    _assert_reconciles(_two_venue_result(capital))


def test_per_exchange_equity_reconciles_with_transfers():
    transfers = _transfers(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:02"),
                "from_exchange": "BINANCE.UM",
                "to_exchange": "HYPERLIQUID.F",
                "amount": 5_000.0,
                "to_amount": 5_000.0,
                "status": "completed",
            }
        ]
    )
    result = _two_venue_result({"BINANCE.UM": 50_000.0, "HYPERLIQUID.F": 50_000.0}, transfers)
    _assert_reconciles(result)
    per_exchange = result.get_equity_per_exchange()
    assert per_exchange["HYPERLIQUID.F"].iloc[-1] - per_exchange["HYPERLIQUID.F"].iloc[0] > 4_000.0


def test_per_exchange_equity_reconciles_after_slicing():
    result = _two_venue_result({"BINANCE.UM": 50_000.0, "HYPERLIQUID.F": 50_000.0})
    _assert_reconciles(result[INDEX[1] :])
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/qubx/core/test_session_result_equity.py -v`
Expected: PASS. If `test_per_exchange_equity_reconciles_after_slicing` fails, the slicing branch in Task 7 is wrong — fix it there, not by relaxing this assertion. (`TradingSessionResult.__getitem__` at `metrics.py:685` accepts a slice, so `result[INDEX[1]:]` is the correct call.)

- [ ] **Step 3: Verify the tests actually bite**

Temporarily revert `_transfer_offsets` to exact-membership matching (`if ts in index`) and re-run.
Expected: `test_per_exchange_equity_reconciles_with_transfers` FAILS on the drift assertion. Restore the implementation.

- [ ] **Step 4: Run the full suite**

Run: `just test`
Expected: PASS. Pay attention to `tests/qubx/core/account_manager/state_metrics_test.py` and `tests/qubx/backtester/` — those cover the changed paths.

- [ ] **Step 5: Commit**

```bash
git add tests/qubx/core/test_session_result_equity.py
git commit -m "test(metrics): pin per-exchange equity against total equity"
```

---

## Manual verification before the PR

Not automated — it needs market data and a strategy, so it belongs in the PR description rather than the suite.

1. Run a short BINANCE.UM ⇄ HYPERLIQUID.F frab simulation (`configs/experiments/v11_bhpl/v11_00_baseline.yml`, one month).
2. Confirm `ctx.get_total_capital()` at the first execution equals the configured capital, not double it.
3. Confirm `r.gross_leverage` stays near the configured leverage rather than decaying across the run.
4. Confirm `r.get_equity_per_exchange().sum(axis=1)` matches `r.equity`.
5. Reload a **pre-existing** stored run (`FARB_V4/v11_00_fp1d_to05/20260716_001508`) and confirm the same reconciliation now holds via the legacy even-split path, and that its 17 transfers are visible as steps.

## Follow-ups (not in this PR)

- frab: `BalanceDifferenceRebalancer` stops hardcoding `target_currency = "USDT"` (`components/rebalancer.py:31`) and passes the source venue's base currency.
- quantkit: confirm `XChangesTransferService` can bridge USDC → USDT before live rebalancing depends on it.
- qubx: marks-based `conversion_rate_to_base` for unpriced assets (the `state.py:245` TODO).
