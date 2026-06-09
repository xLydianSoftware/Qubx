"""Per-exchange derived metrics on AccountState (ported from PR #302's state_metrics_test).

Covers the prefer-venue-else-derive rule per metric, the NaN guard on unmarked
positions, signed leverage, and the explicit (never inferred) base currency.
"""

import numpy as np

from qubx.core.account_manager.state import AccountState, VenueAccountFigures, _notional
from qubx.core.basics import Balance, Instrument, MarketType, Position

T0 = np.datetime64("2026-05-28T00:00:00", "ns")


def _state(cash: float = 1000.0, base_currency: str = "USDT") -> AccountState:
    state = AccountState(exchange="binance", base_currency=base_currency)
    state.update_balance(
        base_currency, Balance(exchange="binance", currency=base_currency, free=cash, locked=0.0, total=cash)
    )
    return state


def _instrument(symbol: str = "BTCUSDT") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="binance",
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _venue(**kwargs) -> VenueAccountFigures:
    return VenueAccountFigures(as_of=T0, **kwargs)


def test_total_capital_cash_only():
    assert _state(1000.0).total_capital() == 1000.0


def test_total_capital_adds_position_market_value():
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.market_value_funds = 200.0
    state.set_position(pos.instrument, pos)
    assert state.total_capital() == 1200.0


def test_total_capital_prefers_venue_equity():
    state = _state(1000.0)
    state.set_venue_figures(_venue(equity=5000.0))
    assert state.total_capital() == 5000.0


def test_base_currency_explicit_not_inferred():
    # base_currency is the constructor arg, never the max-total balance: a bigger
    # non-base balance must not redefine which cash leg total_capital reads.
    state = AccountState(exchange="binance", base_currency="usdc")
    assert state.base_currency == "USDC"
    state.update_balance("USDC", Balance(exchange="binance", currency="USDC", free=100.0, locked=0.0, total=100.0))
    state.update_balance(
        "PEPE", Balance(exchange="binance", currency="PEPE", free=1_000_000.0, locked=0.0, total=1_000_000.0)
    )
    assert state.total_capital() == 100.0


def test_total_margins_from_positions():
    state = _state()
    pos = Position(_instrument())
    pos.initial_margin = 10.0
    pos.maint_margin = 4.0
    state.set_position(pos.instrument, pos)
    assert state.total_initial_margin() == 10.0
    assert state.total_maint_margin() == 4.0


def test_available_margin_derived():
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.initial_margin = 10.0
    state.set_position(pos.instrument, pos)
    assert state.available_margin() == 990.0


def test_available_margin_prefers_venue():
    state = _state(1000.0)
    state.set_venue_figures(_venue(available_margin=777.0))
    assert state.available_margin() == 777.0


def test_available_margin_derives_from_venue_equity_when_only_equity_reported():
    # Per-metric preference: a venue reporting only equity still improves the
    # DERIVED available margin (total_capital prefers venue equity inside it).
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.initial_margin = 10.0
    state.set_position(pos.instrument, pos)
    state.set_venue_figures(_venue(equity=5000.0))
    assert state.available_margin() == 4990.0


def test_margin_ratio_no_maint_is_100():
    assert _state(1000.0).margin_ratio() == 100.0


def test_margin_ratio_derived():
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.maint_margin = 100.0
    state.set_position(pos.instrument, pos)
    assert state.margin_ratio() == 10.0


def test_margin_ratio_prefers_venue():
    state = _state(1000.0)
    state.set_venue_figures(_venue(margin_ratio=42.0))
    assert state.margin_ratio() == 42.0


def test_notional_nan_for_unmarked_is_zero():
    assert _notional(Position(_instrument())) == 0.0


def test_leverage_no_position_is_zero():
    assert _state(1000.0).leverage(_instrument()) == 0.0


def test_leverage_marked_position():
    inst = _instrument()
    state = _state(1000.0)
    pos = Position(inst)
    pos.quantity = 0.1
    pos.update_market_price(T0, 50_000.0, 1.0)
    state.set_position(inst, pos)
    capital = 1000.0 + pos.market_value_funds
    assert state.total_capital() == capital
    assert state.leverage(inst) == _notional(pos) / capital


def test_net_and_gross_leverage():
    inst = _instrument()
    state = _state(1_000_000.0)
    pos = Position(inst)
    pos.quantity = 0.1
    pos.update_market_price(T0, 50_000.0, 1.0)
    state.set_position(inst, pos)
    capital = state.total_capital()
    n = _notional(pos)
    assert state.net_leverage() == n / capital
    assert state.gross_leverage() == abs(n) / capital


def test_net_leverage_signed_negative_for_short():
    inst = _instrument()
    state = _state(1_000_000.0)
    pos = Position(inst)
    pos.quantity = -0.1
    pos.update_market_price(T0, 50_000.0, 1.0)
    state.set_position(inst, pos)
    capital = state.total_capital()
    n = _notional(pos)
    assert n < 0
    assert state.net_leverage() == n / capital < 0
    assert state.gross_leverage() == abs(n) / capital > 0
    assert state.leverage(inst) == n / capital < 0


def test_conversion_rate_is_identity():
    # The single multi-currency seam: 1.0 until real settle/quote -> base conversion lands.
    assert _state().conversion_rate(_instrument()) == 1.0


def test_venue_figures_unset_by_default():
    assert _state().get_venue_figures() is None
