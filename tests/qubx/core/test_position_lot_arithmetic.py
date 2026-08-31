"""A booked deal moves the position by whole lots, so a float subtraction cannot lose one."""

import pandas as pd
import pytest

from qubx.core.basics import Instrument, MarketType, Position

TIME = lambda x: pd.Timestamp(x, unit="ns").asm8
T0 = TIME("2026-08-28 00:00:00")

# - built here rather than looked up: the arithmetic only depends on lot_size, and reading it from
#   the instrument store makes the test need one. LIGHTER's ETHFIUSDC and AEROUSDC both carry 0.1,
#   which is where the live numbers below come from.
LOT = 0.1


def _instrument(symbol: str, lot_size: float = LOT) -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="LIGHTER",
        base=symbol.replace("USDC", ""),
        quote="USDC",
        settle="USDC",
        exchange_symbol=symbol,
        tick_size=1e-05,
        lot_size=lot_size,
        min_size=lot_size,
        contract_size=1.0,
    )


def _pos(exchange: str, symbol: str, quantity: float, price: float) -> Position:
    p = Position(_instrument(symbol))
    p.update_position(T0, position=quantity, exec_price=price)
    return p


def test_a_subtraction_that_lands_a_few_ulp_short_still_books_a_whole_lot():
    # 26.0 - 25.6 is 0.3999999999999986; flooring the float sum returned 0.3 and lost a lot
    p = _pos("LIGHTER", "ETHFIUSDC", 26.0, 0.58)
    p.change_position_by(T0, -25.6, 0.58)
    assert p.quantity == pytest.approx(0.4, abs=1e-12)


def test_the_last_lot_is_not_swallowed():
    # 28.2 - 28.1 is 0.09999999999999787; the position read 0.0 while the venue still held 0.1
    p = _pos("LIGHTER", "AEROUSDC", 28.2, 0.53)
    p.change_position_by(T0, -28.1, 0.53)
    assert p.quantity == pytest.approx(0.1, abs=1e-12)


def test_a_short_position_books_the_same_way():
    p = _pos("LIGHTER", "AEROUSDC", -28.2, 0.53)
    p.change_position_by(T0, 28.1, 0.53)
    assert p.quantity == pytest.approx(-0.1, abs=1e-12)


def test_a_close_reaches_exactly_flat():
    p = _pos("LIGHTER", "ETHFIUSDC", 51.6, 0.58)
    for amount in (-25.6, -25.6, -0.2, -0.2):
        p.change_position_by(T0, amount, 0.58)
    assert p.quantity == pytest.approx(0.0, abs=1e-12)


def test_a_booked_amount_off_the_lot_grid_is_reported_and_still_booked():
    """Stale instrument metadata or a connector fault — but dropping the deal is worse."""
    p = _pos("LIGHTER", "AEROUSDC", 28.2, 0.53)
    # - 0.05 is half a lot; C rounds half away from zero, so it books as one
    p.change_position_by(T0, -0.05, 0.53)
    assert p.quantity == pytest.approx(28.1, abs=1e-12)

    # - 0.16 is 1.6 lots -> 2
    p.change_position_by(T0, -0.16, 0.53)
    assert p.quantity == pytest.approx(27.9, abs=1e-12)
