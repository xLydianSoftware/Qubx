# Delisting Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a delisted-and-gone instrument (market removed from the exchange) from crashing warmup or lingering as a phantom position, while leaving scheduled (future) delistings on the existing close-via-trade path untouched.

**Architecture:** A connector-agnostic `IDataProvider.is_instrument_listed` capability (ccxt reads `exchange.markets`) is surfaced through `IMarketManager.is_instrument_listed`. The `UniverseManager` uses it to detect "market gone" (state B), exclude such instruments from the universe, and settle held positions **in place** (quantity → 0, realized PnL kept) without trading. A ccxt warmup guard skips unlisted instruments so startup never crashes. Future/scheduled delistings (state A) keep flowing through the existing `DelistingDetector`/close-via-trade path, unchanged.

**Tech Stack:** Python 3.12, pytest + pytest-mock, ccxt, qubx core (`UniverseManager`, `IAccountProcessor`, `Position`, `IDataProvider`, `IMarketManager`).

**Spec:** `docs/superpowers/specs/2026-06-16-delisting-resilience-design.md`

**Run a single test:** `uv run pytest <path>::<test> -v`
**Run the unit suite:** `just test` (i.e. `uv run pytest -m "not integration and not e2e" --ignore=debug -v -n auto`)

---

### Task 1: `Position.flatten()` — zero quantity without losing history

Settle a position in place: zero quantity and derived market values/margins, but
keep `r_pnl`, average price, commissions, and funding history. Distinct from the
existing `Position.reset()` which also wipes `r_pnl`.

**Files:**
- Modify: `src/qubx/core/basics.py` (class `Position`, near `reset()` at `:800`)
- Test: `tests/qubx/core/basics_test.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/qubx/core/basics_test.py`:

```python
def test_position_flatten_zeroes_quantity_but_keeps_realized(mocker):
    from qubx.core.basics import Instrument, Position

    instr = mocker.Mock(spec=Instrument)
    instr.lot_size = 0.001
    pos = Position(instr, quantity=10.0, pos_average_price=100.0, r_pnl=42.0)
    pos.market_value = 1000.0
    pos.market_value_funds = 1000.0
    pos.initial_margin = 250.0
    pos.maint_margin = 125.0

    pos.flatten()

    assert pos.quantity == 0.0
    assert pos.market_value == 0.0
    assert pos.market_value_funds == 0.0
    assert pos.initial_margin == 0.0
    assert pos.maint_margin == 0.0
    assert pos.pnl == 42.0            # unrealized is 0 at qty 0 → pnl == r_pnl
    assert pos.r_pnl == 42.0          # realized preserved
    assert pos.position_avg_price == 100.0   # entry preserved
    assert pos.is_open() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/basics_test.py::test_position_flatten_zeroes_quantity_but_keeps_realized -v`
Expected: FAIL with `AttributeError: 'Position' object has no attribute 'flatten'`

- [ ] **Step 3: Implement `flatten()`**

In `src/qubx/core/basics.py`, add this method to `Position` immediately after `reset()` (after line ~812, before `reset_by_position`):

```python
    def flatten(self) -> None:
        """
        Mark the position flat WITHOUT trading: zero quantity and the derived
        market values / margins, while KEEPING realized PnL, average price,
        commissions and funding history.

        For an instrument whose market has been delisted/removed from the
        exchange (already cash-settled) we cannot place a closing order, so we
        reconcile the in-memory position to flat. Unlike reset(), this preserves
        r_pnl so the record stays identical to a normally-closed position.
        """
        self.quantity = 0.0
        self.market_value = 0.0
        self.market_value_funds = 0.0
        self.initial_margin = 0.0
        self.maint_margin = 0.0
        self.pnl = self.r_pnl  # unrealized PnL is zero at zero quantity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/basics_test.py::test_position_flatten_zeroes_quantity_but_keeps_realized -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qubx/core/basics.py tests/qubx/core/basics_test.py
git commit -m "feat(core): add Position.flatten() to zero quantity without losing realized PnL"
```

---

### Task 2: `IDataProvider.is_instrument_listed` + ccxt implementation

Connector-agnostic capability: "does this instrument currently exist on the
exchange?" Base default `True` (unknown ⇒ assume listed, never wrongly drop).
ccxt reads `exchange.markets`, fail-open when markets are empty/unloaded.

**Files:**
- Modify: `src/qubx/core/interfaces.py` (class `IDataProvider`, after `get_quote` ~`:906`)
- Modify: `src/qubx/connectors/ccxt/data.py` (class `CcxtDataProvider`; import at `:7`)
- Test: `tests/qubx/connectors/ccxt/test_data_provider.py`

- [ ] **Step 1: Add the base interface method**

In `src/qubx/core/interfaces.py`, inside `class IDataProvider`, add after the `get_quote` method (around line 910):

```python
    def is_instrument_listed(self, instrument: Instrument) -> bool:
        """
        Whether the instrument currently exists / is tradeable on the exchange.

        Default is True: callers must never drop an instrument just because a
        provider cannot determine listing status (fail-open). Connectors that
        can answer authoritatively (e.g. ccxt via exchange.markets) override this.
        """
        return True
```

- [ ] **Step 2: Write the failing ccxt test**

Append to `tests/qubx/connectors/ccxt/test_data_provider.py` (uses the existing
`data_provider` and `btc_instrument` fixtures in that file):

```python
class TestIsInstrumentListed:
    def test_listed_when_symbol_in_markets(self, data_provider, btc_instrument):
        from qubx.connectors.ccxt.utils import instrument_to_ccxt_symbol

        sym = instrument_to_ccxt_symbol(btc_instrument)
        data_provider._exchange_manager.exchange.markets = {sym: {"id": "x"}}
        assert data_provider.is_instrument_listed(btc_instrument) is True

    def test_not_listed_when_symbol_absent(self, data_provider, btc_instrument):
        data_provider._exchange_manager.exchange.markets = {"OTHER/USDT:USDT": {}}
        assert data_provider.is_instrument_listed(btc_instrument) is False

    def test_fail_open_when_markets_empty(self, data_provider, btc_instrument):
        data_provider._exchange_manager.exchange.markets = {}
        assert data_provider.is_instrument_listed(btc_instrument) is True

    def test_fail_open_when_markets_none(self, data_provider, btc_instrument):
        data_provider._exchange_manager.exchange.markets = None
        assert data_provider.is_instrument_listed(btc_instrument) is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_data_provider.py::TestIsInstrumentListed -v`
Expected: FAIL — base returns `True` for the `absent` case (assertion error: expected False).

- [ ] **Step 4: Implement in `CcxtDataProvider`**

In `src/qubx/connectors/ccxt/data.py`, extend the import at line 7:

```python
from qubx.connectors.ccxt.utils import ccxt_convert_timeframe_to_exchange_format, instrument_to_ccxt_symbol
```

Then add this method to `CcxtDataProvider` (place it next to the other public
methods, e.g. right after `__init__` ends / before `subscribe`):

```python
    def is_instrument_listed(self, instrument: Instrument) -> bool:
        markets = getattr(self._exchange_manager.exchange, "markets", None)
        if not markets:  # None or empty ⇒ not loaded / can't tell ⇒ fail-open
            return True
        return instrument_to_ccxt_symbol(instrument) in markets
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_data_provider.py::TestIsInstrumentListed -v`
Expected: PASS (all 4)

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/interfaces.py src/qubx/connectors/ccxt/data.py tests/qubx/connectors/ccxt/test_data_provider.py
git commit -m "feat(data): add IDataProvider.is_instrument_listed (ccxt reads exchange.markets, fail-open)"
```

---

### Task 3: `IMarketManager.is_instrument_listed` — resolve provider by exchange

Surface the capability at the market-manager level so the `UniverseManager`
(which already depends on `IMarketManager`) can ask without reaching into
per-exchange data providers. Fail-open when no provider is found.

**Files:**
- Modify: `src/qubx/core/interfaces.py` (class `IMarketManager` ~`:971`)
- Modify: `src/qubx/core/mixins/market.py` (class `MarketManager`, near `_get_data_provider` `:486`)
- Test: `tests/qubx/core/mixins/market_manager_listing_test.py` (new)

- [ ] **Step 1: Add the interface method**

In `src/qubx/core/interfaces.py`, inside `class IMarketManager`, add:

```python
    def is_instrument_listed(self, instrument: Instrument) -> bool:
        """Whether the instrument is currently listed on its exchange.
        Fail-open: True when no data provider can answer."""
        ...
```

- [ ] **Step 2: Write the failing test**

Create `tests/qubx/core/mixins/market_manager_listing_test.py`:

```python
import pytest
from pytest_mock import MockerFixture

from qubx.core.basics import Instrument
from qubx.core.mixins.market import MarketManager


def _instr(mocker: MockerFixture, exchange: str = "BINANCE.UM"):
    i = mocker.Mock(spec=Instrument)
    i.exchange = exchange
    i.symbol = "TONUSDT"
    return i


def _market_manager_with(provider):
    mm = MarketManager.__new__(MarketManager)  # bypass heavy __init__
    mm._exchange_to_data_provider = {} if provider is None else {"BINANCE.UM": provider}
    return mm


def test_listed_delegates_to_provider_true(mocker: MockerFixture):
    dp = mocker.Mock()
    dp.is_instrument_listed.return_value = True
    mm = _market_manager_with(dp)
    assert mm.is_instrument_listed(_instr(mocker)) is True


def test_not_listed_delegates_to_provider_false(mocker: MockerFixture):
    dp = mocker.Mock()
    dp.is_instrument_listed.return_value = False
    mm = _market_manager_with(dp)
    assert mm.is_instrument_listed(_instr(mocker)) is False


def test_fail_open_when_no_provider(mocker: MockerFixture):
    mm = _market_manager_with(None)
    assert mm.is_instrument_listed(_instr(mocker)) is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/mixins/market_manager_listing_test.py -v`
Expected: FAIL with `AttributeError: 'MarketManager' object has no attribute 'is_instrument_listed'`

- [ ] **Step 4: Implement in `MarketManager`**

In `src/qubx/core/mixins/market.py`, add immediately after `_get_data_provider` (after line 491):

```python
    def is_instrument_listed(self, instrument: Instrument) -> bool:
        try:
            dp = self._get_data_provider(instrument.exchange)
        except ValueError:
            return True  # no provider ⇒ can't tell ⇒ fail-open
        return dp.is_instrument_listed(instrument)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/mixins/market_manager_listing_test.py -v`
Expected: PASS (3)

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/interfaces.py src/qubx/core/mixins/market.py tests/qubx/core/mixins/market_manager_listing_test.py
git commit -m "feat(market): add IMarketManager.is_instrument_listed resolving provider by exchange"
```

---

### Task 4: `IAccountProcessor.settle_position` — flatten in place, no trade

Public way to settle a held position without trading, for a market the exchange
already removed/cash-settled. Implemented on both account processors.

**Files:**
- Modify: `src/qubx/core/interfaces.py` (class `IAccountProcessor` ~`:1450`, near `attach_positions` `:1583`)
- Modify: `src/qubx/core/account.py` (`BasicAccountProcessor` `:27`, `CompositeAccountProcessor` `:615`)
- Test: `tests/qubx/core/account_processor_test.py`

- [ ] **Step 1: Add the interface method**

In `src/qubx/core/interfaces.py`, inside `class IAccountProcessor`, add near `attach_positions`:

```python
    def settle_position(self, instrument: Instrument) -> None:
        """Flatten a held position in place (no trade) for a delisted/removed
        market the exchange has already cash-settled. Realized PnL is kept."""
        ...
```

- [ ] **Step 2: Write the failing test**

Append to `tests/qubx/core/account_processor_test.py`:

```python
def test_settle_position_flattens_without_trade(mocker):
    from qubx.core.account import BasicAccountProcessor
    from qubx.core.basics import Instrument, Position

    acc = BasicAccountProcessor(
        account_id="t",
        time_provider=mocker.Mock(),
        base_currency="USDT",
        health_monitor=mocker.Mock(),
        exchange="OKX.F",
    )
    instr = mocker.Mock(spec=Instrument)
    instr.lot_size = 0.001
    pos = Position(instr, quantity=3175.0, pos_average_price=1.69, r_pnl=-7.0)
    acc.attach_positions(pos)

    acc.settle_position(instr)

    settled = acc.get_positions()[instr]
    assert settled.quantity == 0.0
    assert settled.r_pnl == -7.0          # realized kept
    assert settled.is_open() is False


def test_settle_position_noop_when_not_held(mocker):
    from qubx.core.account import BasicAccountProcessor
    from qubx.core.basics import Instrument

    acc = BasicAccountProcessor(
        account_id="t",
        time_provider=mocker.Mock(),
        base_currency="USDT",
        health_monitor=mocker.Mock(),
        exchange="OKX.F",
    )
    instr = mocker.Mock(spec=Instrument)
    acc.settle_position(instr)  # must not raise
    assert instr not in acc.get_positions()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/qubx/core/account_processor_test.py::test_settle_position_flattens_without_trade tests/qubx/core/account_processor_test.py::test_settle_position_noop_when_not_held -v`
Expected: FAIL with `AttributeError: 'BasicAccountProcessor' object has no attribute 'settle_position'`

- [ ] **Step 4: Implement on both processors**

In `src/qubx/core/account.py`, add to `BasicAccountProcessor` (e.g. right after `attach_positions`, after line ~264):

```python
    def settle_position(self, instrument: Instrument) -> None:
        pos = self._positions.get(instrument)
        if pos is not None:
            pos.flatten()
```

And add to `CompositeAccountProcessor` (mirror its `get_positions` dispatch style, near line 742):

```python
    def settle_position(self, instrument: Instrument) -> None:
        exch = self._get_exchange(instrument=instrument)
        self._account_processors[exch].settle_position(instrument)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/qubx/core/account_processor_test.py::test_settle_position_flattens_without_trade tests/qubx/core/account_processor_test.py::test_settle_position_noop_when_not_held -v`
Expected: PASS (2)

- [ ] **Step 6: Commit**

```bash
git add src/qubx/core/interfaces.py src/qubx/core/account.py tests/qubx/core/account_processor_test.py
git commit -m "feat(account): add settle_position to flatten a delisted position without trading"
```

---

### Task 5: `UniverseManager` — detect "gone", exclude, and settle in place

The orchestration: `_is_market_gone` (live-listing authoritative; past
`delist_date` as fallback, future `delist_date` is NOT gone), `_drop_gone`
(exclude + settle held + alert), `_settle_if_held` (settle only when
live-confirmed gone), `_notify_gone`. Wire `_drop_gone` into `set_universe`
alongside the existing `filter_delistings`, and branch `__do_remove_instruments`
to settle (not trade) gone instruments.

**Files:**
- Modify: `src/qubx/core/mixins/universe.py` (imports `:1`; `set_universe` `:78`; `__do_remove_instruments` close loop `:180-195`; add new methods)
- Test: `tests/qubx/core/universe_manager_test.py`

- [ ] **Step 1: Extend the test fixture with a market-manager listing default**

In `tests/qubx/core/universe_manager_test.py`, the `mock_dependencies` fixture
sets `market_data_manager` as a bare `Mock`. Add a default so listing is
"listed" unless a test overrides it. Edit the `market_data_manager` setup block
(after line 18) to add:

```python
    market_data_manager.is_instrument_listed.return_value = True
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/qubx/core/universe_manager_test.py`:

```python
def _gone_instr(mocker, symbol="TONUSDT"):
    i = mocker.Mock(spec=Instrument, symbol=symbol)
    i.exchange = "OKX.F"
    i.delist_date = None
    i.min_size = 0.001
    return i


def test_set_universe_excludes_gone_instrument(universe_manager, mock_dependencies, mocker):
    mock_dependencies["subscription_manager"].auto_subscribe = True
    live = mocker.Mock(spec=Instrument, symbol="BTCUSDT")
    live.exchange = "OKX.F"
    live.delist_date = None
    live.min_size = 0.001
    gone = _gone_instr(mocker)

    def listed(instr):
        return instr is not gone

    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = listed
    # no positions held
    mock_dependencies["account"].positions = {}

    universe_manager.set_universe([live, gone])

    assert set(universe_manager.instruments) == {live}
    assert gone not in universe_manager.instruments


def test_gone_held_position_is_settled_not_traded(universe_manager, mock_dependencies, mocker):
    gone = _gone_instr(mocker)
    pos = mocker.Mock()
    pos.quantity = 3175.0
    mock_dependencies["account"].positions = {gone: pos}
    mock_dependencies["market_data_manager"].is_instrument_listed.side_effect = lambda i: i is not gone
    mock_dependencies["subscription_manager"].auto_subscribe = True

    universe_manager.set_universe([gone])

    mock_dependencies["account"].settle_position.assert_called_once_with(gone)
    mock_dependencies["position_gathering"].alter_positions.assert_not_called()


def test_future_delist_but_listed_is_not_gone(universe_manager, mock_dependencies, mocker):
    import pandas as pd

    instr = _gone_instr(mocker, symbol="SOLUSDT")
    instr.delist_date = pd.Timestamp("2999-01-01")  # far future
    mock_dependencies["market_data_manager"].is_instrument_listed.return_value = True
    mock_dependencies["account"].positions = {}
    mock_dependencies["subscription_manager"].auto_subscribe = True

    universe_manager.set_universe([instr])

    assert instr in universe_manager.instruments
    mock_dependencies["account"].settle_position.assert_not_called()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/qubx/core/universe_manager_test.py -k "gone or future_delist" -v`
Expected: FAIL — `test_set_universe_excludes_gone_instrument` keeps `gone` in the universe (no `_drop_gone` yet).

- [ ] **Step 4: Add imports and the new methods**

In `src/qubx/core/mixins/universe.py`, extend the imports. Change line 1 and add `to_timestamp`:

```python
from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.utils.time import to_timestamp
```

(Keep the existing `from qubx.core.detectors import DelistingDetector` and the `from qubx.core.interfaces import (...)` block.)

Add these methods to `UniverseManager` (place them right after `_has_position`, after line 63):

```python
    def _is_market_gone(self, instrument: Instrument) -> bool:
        """State B only: the market no longer exists / is untradeable.
        A future delist_date (state A) is NOT gone."""
        if not self._mkt_manager.is_instrument_listed(instrument):
            return True  # authoritative live signal
        d = instrument.delist_date
        if d is None:
            return False
        return to_timestamp(d).replace(tzinfo=None) <= to_timestamp(self._time_provider.time())

    def _settle_if_held(self, instrument: Instrument) -> None:
        if not self._has_position(instrument):
            return
        if not self._mkt_manager.is_instrument_listed(instrument):
            # live-confirmed gone: cannot trade out, exchange already settled
            self._trading_manager.cancel_orders(instrument)
            self._account.settle_position(instrument)
            logger.warning(f"[UniverseManager] Settled delisted position {instrument.symbol} (market gone)")
        else:
            # flagged by past delist_date metadata only, but still listed (settlement
            # overlap) — leave it for the close-via-trade path / manual review
            logger.warning(
                f"[UniverseManager] {instrument.symbol} flagged delisted by metadata but still listed; "
                "leaving position for close-via-trade / manual review"
            )

    def _notify_gone(self, instruments: list[Instrument]) -> None:
        symbols = ", ".join(sorted(i.symbol for i in instruments))
        logger.warning(f"[UniverseManager] Dropping delisted (gone) instruments: {symbols}")
        notifier = getattr(self._context, "notifier", None)
        if notifier is not None and not self._context.is_simulation:
            notifier.notify_message(
                f"[{self._context.strategy_name}] Dropped delisted (gone) instruments: {symbols}",
                metadata={"event": "delisted_gone", "instruments": symbols},
            )

    def _drop_gone(self, instruments: list[Instrument]) -> list[Instrument]:
        gone = [i for i in instruments if self._is_market_gone(i)]
        if not gone:
            return instruments
        for i in gone:
            self._settle_if_held(i)
        self._notify_gone(gone)
        gone_set = set(gone)
        return [i for i in instruments if i not in gone_set]
```

- [ ] **Step 5: Wire `_drop_gone` into `set_universe`**

In `src/qubx/core/mixins/universe.py`, in `set_universe`, replace the existing
delisting filter (line 78) so `_drop_gone` runs **BEFORE** `filter_delistings`.
Order matters: `filter_delistings` strips anything with `delist_date <= now +
check_days` (incl. past dates), so running it first would remove a gone
instrument that also carries a `delist_date` before `_drop_gone` could settle
its held position. Settle state B first, then filter state A:

```python
        # Settle & exclude instruments whose market is already gone (state B) FIRST
        instruments = self._drop_gone(instruments)
        # Then filter scheduled/upcoming delistings (state A: still listed → close via trade)
        instruments = self._delisting_detector.filter_delistings(instruments)
```

- [ ] **Step 6: Branch `__do_remove_instruments` to settle gone instruments**

In `src/qubx/core/mixins/universe.py`, in `__do_remove_instruments`, change the
close loop (lines ~180-195) so a gone instrument is settled instead of traded:

```python
        # - close all open positions
        exit_targets = []
        for instr in instruments:
            if self._has_position(instr):
                if not self._mkt_manager.is_instrument_listed(instr):
                    # market gone — cannot trade; settle in place
                    self._account.settle_position(instr)
                    logger.warning(f"[UniverseManager] Settled delisted position {instr.symbol} on removal")
                    continue

                # - create exit target
                exit_targets.append(instr.target(self._context, 0))

                self._removal_in_progress.add(instr)

                # - emit service signals for instruments that are being removed
                self._context.emit_signal(
                    instr.service_signal(self._context, 0, group="Universe", comment="Universe change")
                )

        # - alter positions
        self._position_gathering.alter_positions(self._context, exit_targets)
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `uv run pytest tests/qubx/core/universe_manager_test.py -k "gone or future_delist" -v`
Expected: PASS (3)

- [ ] **Step 8: Run the full universe-manager suite (no regressions)**

Run: `uv run pytest tests/qubx/core/universe_manager_test.py -v`
Expected: PASS (existing tests still green; `filter_delistings` and `is_instrument_listed` mocks default to no-op/listed)

- [ ] **Step 9: Commit**

```bash
git add src/qubx/core/mixins/universe.py tests/qubx/core/universe_manager_test.py
git commit -m "feat(universe): detect gone markets, exclude from universe, settle held positions in place"
```

---

### Task 6: ccxt warmup guard — skip unlisted, never abort the batch

The startup crash fix: `OhlcDataHandler.warmup()` must skip instruments whose
market is gone and tolerate a per-instrument fetch failure, so one delisted
instrument can't abort `asyncio.gather`.

**Files:**
- Modify: `src/qubx/connectors/ccxt/handlers/ohlc.py` (`warmup`, loop at `:96`)
- Test: `tests/qubx/connectors/ccxt/test_ohlc_warmup_guard.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/qubx/connectors/ccxt/test_ohlc_warmup_guard.py`:

```python
import asyncio

import pytest
from pytest_mock import MockerFixture

from qubx.connectors.ccxt.handlers.ohlc import OhlcDataHandler
from qubx.core.basics import Instrument


def _handler(mocker: MockerFixture, listed_map, fetch_side_effect):
    data_provider = mocker.Mock()
    data_provider.is_instrument_listed.side_effect = lambda i: listed_map[i]
    data_provider._get_exch_timeframe.return_value = "1d"
    data_provider._time_msec_nbars_back.return_value = 0

    exchange = mocker.Mock()
    exchange.fetch_ohlcv = mocker.AsyncMock(side_effect=fetch_side_effect)
    exchange_manager = mocker.Mock()
    exchange_manager.exchange = exchange

    return OhlcDataHandler(data_provider=data_provider, exchange_manager=exchange_manager, exchange_id="okx"), exchange


def _instr(mocker, symbol):
    i = mocker.Mock(spec=Instrument)
    i.symbol = symbol
    return i


def test_warmup_skips_unlisted_instrument(mocker: MockerFixture):
    listed = _instr(mocker, "BTCUSDT")
    gone = _instr(mocker, "TONUSDT")
    channel = mocker.Mock()

    # one bar then stop, for any listed instrument
    async def fetch(symbol, timeframe, since, limit):
        return [[0, 1.0, 2.0, 0.5, 1.5, 100.0]]

    handler, exchange = _handler(mocker, {listed: True, gone: False}, fetch)
    # patch the bar/quote converters to avoid needing real Instrument internals
    mocker.patch.object(handler, "_convert_ohlcv_to_bar", side_effect=lambda oh: oh)
    mocker.patch.object(handler, "_convert_ohlcv_to_quote", return_value=object())

    asyncio.run(handler.warmup({listed, gone}, channel, warmup_period="1d", timeframe="1d"))

    # gone instrument is never fetched; its ccxt symbol is never requested
    called_instruments = {c.kwargs.get("symbol") if c.kwargs else None for c in exchange.fetch_ohlcv.call_args_list}
    # at least the listed instrument was fetched and gone was skipped (fewer than 2 distinct)
    assert exchange.fetch_ohlcv.await_count >= 1


def test_warmup_continues_when_one_instrument_raises(mocker: MockerFixture):
    a = _instr(mocker, "AAAUSDT")
    b = _instr(mocker, "BBBUSDT")
    channel = mocker.Mock()

    from ccxt import BadSymbol

    async def fetch(symbol, timeframe, since, limit):
        # first instrument raises, second returns data
        if fetch.calls == 0:
            fetch.calls += 1
            raise BadSymbol("okx does not have market symbol AAA/USDT:USDT")
        return [[0, 1.0, 2.0, 0.5, 1.5, 100.0]]

    fetch.calls = 0
    handler, exchange = _handler(mocker, {a: True, b: True}, fetch)
    mocker.patch.object(handler, "_convert_ohlcv_to_bar", side_effect=lambda oh: oh)
    mocker.patch.object(handler, "_convert_ohlcv_to_quote", return_value=object())

    # must NOT raise even though one instrument failed
    asyncio.run(handler.warmup({a, b}, channel, warmup_period="1d", timeframe="1d"))
    assert exchange.fetch_ohlcv.await_count >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ohlc_warmup_guard.py -v`
Expected: FAIL — `test_warmup_continues_when_one_instrument_raises` propagates `BadSymbol` (no guard yet).

- [ ] **Step 3: Add the guard in `warmup()`**

In `src/qubx/connectors/ccxt/handlers/ohlc.py`, wrap the per-instrument body of
the `for instrument in instruments:` loop (starts ~line 96). Replace the loop
body so it begins with a listed-check and wraps the fetch/send in try/except:

```python
        for instrument in instruments:
            if not self._data_provider.is_instrument_listed(instrument):
                logger.warning(
                    f"<yellow>{self._exchange_id}</yellow> {instrument} is not listed on the exchange "
                    "(delisted/removed); skipping warmup"
                )
                continue

            try:
                start_since = self._data_provider._time_msec_nbars_back(timeframe, nbarsback)
                ccxt_symbol = instrument_to_ccxt_symbol(instrument)

                # Paginate: exchanges may return fewer bars than requested per call
                ohlcv_map: dict[int, list] = {}
                while len(ohlcv_map) < nbarsback:
                    batch = await self._exchange_manager.exchange.fetch_ohlcv(
                        ccxt_symbol, exch_timeframe, since=start_since,
                        limit=min(nbarsback - len(ohlcv_map), self.MAX_BARS_PER_REQUEST_FOR_PROVIDER) + 1,
                    )
                    if not batch:
                        break
                    prev_count = len(ohlcv_map)
                    for bar in batch:
                        ohlcv_map[bar[0]] = bar
                    if len(ohlcv_map) == prev_count:
                        break
                    start_since = batch[-1][0] + _tf_msec

                ohlcv = list(ohlcv_map.values())
                logger.debug(f"<yellow>{self._exchange_id}</yellow> {instrument}: loaded {len(ohlcv)} {timeframe} bars")

                channel.send(
                    (
                        instrument,
                        DataType.OHLC[timeframe],
                        [self._convert_ohlcv_to_bar(oh) for oh in ohlcv],
                        True,  # historical data
                    )
                )

                if len(ohlcv) > 0:
                    # Send a quote update to the context at the end of warmup
                    channel.send((instrument, DataType.QUOTE, self._convert_ohlcv_to_quote(ohlcv, instrument), False))
            except Exception as e:
                logger.warning(
                    f"<yellow>{self._exchange_id}</yellow> warmup failed for {instrument} "
                    f"({type(e).__name__}: {e}); skipping"
                )
                continue
```

(Keep the lines above the loop — `nbarsback`, `exch_timeframe`, `_tf_msec` — as they are.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/qubx/connectors/ccxt/test_ohlc_warmup_guard.py -v`
Expected: PASS (2)

- [ ] **Step 5: Commit**

```bash
git add src/qubx/connectors/ccxt/handlers/ohlc.py tests/qubx/connectors/ccxt/test_ohlc_warmup_guard.py
git commit -m "fix(ccxt): warmup skips unlisted instruments and tolerates per-instrument failures"
```

---

### Task 7: Full suite + lint

- [ ] **Step 1: Run the unit suite**

Run: `just test`
Expected: all pass (no regressions). If `pytest-xdist` OOMs locally, use `uv run pytest -m "not integration and not e2e" --ignore=debug -n 2 -q`.

- [ ] **Step 2: Lint / format**

Run: `just style-check` (or `uv run ruff check src tests` and `uv run ruff format --check src tests`)
Expected: clean. Fix any line-length (120) issues introduced.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint delisting resilience changes"
```

---

## Self-Review

**Spec coverage:**
- §1 connector capability → Task 2. ✅
- §2 `_is_market_gone`/`_drop_gone`/`_settle_if_held`/`_notify_gone` + `set_universe` wiring + `__do_remove_instruments` gone-branch → Task 5; `IMarketManager` resolution → Task 3. ✅
- §3 `Position.flatten` + `IAccountProcessor.settle_position`, live-confirmed forget guard → Tasks 1, 4, 5 (`_settle_if_held`). ✅
- §4 ccxt warmup guard → Task 6. ✅
- Fail-open (markets empty/no provider) → Tasks 2, 3 tests. ✅
- State A never settled (future delist still listed) → Task 5 `test_future_delist_but_listed_is_not_gone`. ✅
- Backtest unaffected → base `is_instrument_listed` returns True (Task 2); simulator inherits base. (No explicit sim test task — acceptable; covered by existing simulation suite in Task 7.)

**Placeholder scan:** No TBD/TODO; every code step has complete code. The `...`
in interface stubs (Tasks 2–4) are intentional Python interface bodies matching
the existing `IDataProvider`/`IAccountProcessor` style.

**Type/signature consistency:** `is_instrument_listed(instrument) -> bool` used
identically on `IDataProvider` (Task 2), `IMarketManager`/`MarketManager`
(Task 3), and called as `self._mkt_manager.is_instrument_listed(...)` (Task 5).
`settle_position(instrument) -> None` consistent across interface + both
processors (Task 4) and called in Task 5. `Position.flatten()` defined Task 1,
used by `settle_position` Task 4.

**Out of scope (deliberate):** runner-side restored-state pre-filter (spec shows
unnecessary), state-A behavior changes, global kill-switch, optional partial-load
hardening.
