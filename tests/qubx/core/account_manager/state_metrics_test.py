import numpy as np

from qubx.core.account_manager.state import AccountState, VenueAccountFigures, _notional
from qubx.core.basics import AssetBalance, Position
from qubx.core.lookups import lookup

T0 = np.datetime64("2026-05-28T00:00:00", "ns")


def _state(cash: float = 1000.0) -> AccountState:
    state = AccountState("binance", "USDT")
    state._update_balance("USDT", AssetBalance(exchange="binance", currency="USDT", free=cash, locked=0.0, total=cash))
    return state


def _instrument(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol("BINANCE.UM", symbol)
    assert inst is not None
    return inst


def _venue(**kwargs) -> VenueAccountFigures:
    return VenueAccountFigures(as_of=T0, **kwargs)


def test_total_capital_cash_only():
    assert _state(1000.0).total_capital() == 1000.0


def test_total_capital_adds_position_market_value():
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.market_value_funds = 200.0
    state._update_position(pos)
    assert state.total_capital() == 1200.0


def test_total_capital_prefers_venue_equity():
    state = _state(1000.0)
    state._set_venue_figures(_venue(equity=5000.0))
    assert state.total_capital() == 5000.0


def test_total_margins_from_positions():
    state = _state()
    pos = Position(_instrument())
    pos.initial_margin = 10.0
    pos.maint_margin = 4.0
    state._update_position(pos)
    assert state.total_initial_margin() == 10.0
    assert state.total_maint_margin() == 4.0


def test_available_margin_derived():
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.initial_margin = 10.0
    state._update_position(pos)
    assert state.available_margin() == 990.0


def test_available_margin_prefers_venue():
    state = _state(1000.0)
    state._set_venue_figures(_venue(available_margin=777.0))
    assert state.available_margin() == 777.0


def test_margin_ratio_no_maint_is_100():
    assert _state(1000.0).margin_ratio() == 100.0


def test_margin_ratio_derived():
    state = _state(1000.0)
    pos = Position(_instrument())
    pos.maint_margin = 100.0
    state._update_position(pos)
    assert state.margin_ratio() == 10.0


def test_margin_ratio_prefers_venue():
    state = _state(1000.0)
    state._set_venue_figures(_venue(margin_ratio=42.0))
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
    state._update_position(pos)
    capital = 1000.0 + pos.market_value_funds
    assert state.total_capital() == capital
    assert state.leverage(inst) == _notional(pos) / capital


def test_net_and_gross_leverage():
    inst = _instrument()
    state = _state(1_000_000.0)
    pos = Position(inst)
    pos.quantity = 0.1
    pos.update_market_price(T0, 50_000.0, 1.0)
    state._update_position(pos)
    capital = state.total_capital()
    n = _notional(pos)
    assert state.net_leverage() == n / capital
    assert state.gross_leverage() == abs(n) / capital
