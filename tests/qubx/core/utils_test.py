import math
import random
from decimal import Decimal

import pandas as pd
import pytest

from qubx.core.basics import Balance, Deal, Instrument, ITimeProvider, Position, dt_64
from qubx.core.utils import add_in_lots, grid_ceil, grid_floor, is_lot_multiple, prec_ceil, prec_floor


def test_prec_floor():
    a = 608.8135
    precision = 2
    assert prec_floor(a, precision) == 608.81
    assert prec_floor(prec_floor(a, precision), precision) == prec_floor(a, precision)

    assert prec_floor(608.16, 1) == 608.1


def test_prec_ceil():
    a = 608.8135
    precision = 2
    assert prec_ceil(a, precision) == 608.82
    assert prec_ceil(prec_ceil(a, precision), precision) == prec_ceil(a, precision)


def test_precision_zero_is_a_true_floor_and_ceil():
    """Rounding the SCALED value to `precision` decimals made this a round-to-nearest."""
    assert prec_floor(3.7, 0) == 3.0
    assert prec_floor(200.6, 0) == 200.0
    assert prec_floor(0.6, 0) == 0.0
    assert prec_floor(-3.7, 0) == -3.0
    assert prec_ceil(3.4, 0) == 4.0
    assert prec_ceil(200.4, 0) == 201.0
    assert prec_ceil(-3.4, 0) == -4.0


def test_a_boundary_value_is_not_snapped_away():
    assert prec_floor(1.2349999, 3) == 1.234
    assert prec_ceil(2.0000000005, 0) == 3.0
    assert prec_ceil(0.0000000001, 0) == 1.0


def test_binary_noise_still_lands_on_the_tick():
    assert prec_floor(0.29, 2) == 0.29
    assert prec_floor(0.07, 2) == 0.07
    assert prec_floor(0.1 + 0.2, 1) == 0.3
    assert prec_floor(1.1 * 3, 1) == 3.3
    # the tolerance is relative: 0.29 * 1e8 sits 3.7e-9 from the tick
    assert prec_floor(0.29, 8) == 0.29
    assert prec_ceil(0.29, 8) == 0.29


def test_a_notional_derived_minimum_ceils_to_a_tradeable_size():
    """_adjust_size falls back to round_size_up(min_size); min_notional/price is fractional."""
    assert prec_ceil(23.196474135931336, 0) == 24.0
    assert prec_ceil(113.37868480725623, 0) == 114.0


class DummyTimeProvider(ITimeProvider):
    def time(self) -> dt_64:
        return pd.Timestamp("2024-04-07 13:48:37.611000").asm8


class StubAccount:
    """Minimal position/capital bookkeeper for tracker/gathering tests.

    Reproduces the slice of the old BasicAccountProcessor those tests relied on
    (update_balance / attach_positions / process_deals / get_total_capital /
    positions). The central AccountManager replaced BasicAccountProcessor, but it
    is event-driven (apply(OrderFilledEvent)) rather than process_deals(deals); this
    stub keeps the tracker tests self-contained without coupling them to the AM's
    live event model. Single-base-currency, conversion_rate=1 — same as the old default.
    """

    def __init__(self, base_currency: str = "USDT", exchange: str = "TEST"):
        self.base_currency = base_currency.upper()
        self.exchange = exchange
        self._positions: dict[Instrument, Position] = {}
        self._balances: dict[str, Balance] = {}
        self._processed_trades: dict[str, list] = {}

    @property
    def positions(self) -> dict[Instrument, Position]:
        return self._positions

    def update_balance(self, currency: str, total: float, locked: float) -> None:
        self._balances[currency] = Balance(
            exchange=self.exchange, currency=currency, free=total - locked, locked=locked, total=total
        )

    def attach_positions(self, *positions: Position) -> "StubAccount":
        for p in positions:
            self._positions.setdefault(p.instrument, p)
        return self

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        pos = self._positions.get(instrument)
        if pos is None:
            return
        for d in deals:
            seen = self._processed_trades.setdefault(d.order_id, [])
            if d.trade_id in seen:
                continue
            seen.append(d.trade_id)
            pos.update_position_by_deal(d, conversion_rate=1)

    def get_total_capital(self, exchange: str | None = None) -> float:
        cash = self._balances[self.base_currency].total if self.base_currency in self._balances else 0.0
        return cash + sum(p.market_value_funds for p in self._positions.values())


def test_add_in_lots_survives_a_cancelling_subtraction():
    """28.2 - 28.1 is 0.09999999999999787 as floats; in lots it is 282 - 281."""
    assert add_in_lots(26.0, -25.6, 0.1) == 0.4
    assert add_in_lots(28.2, -28.1, 0.1) == 0.1
    assert add_in_lots(21.4, -21.0, 0.1) == 0.4
    assert add_in_lots(51.6, -25.6, 0.1) == 26.0
    assert add_in_lots(-28.2, 28.1, 0.1) == -0.1
    assert add_in_lots(0.1, -0.1, 0.1) == 0.0


def test_is_lot_multiple_admits_the_grid_and_refuses_a_half_lot():
    assert is_lot_multiple(28.2, 0.1)
    assert is_lot_multiple(-28.2, 0.1)
    assert is_lot_multiple(0.0, 0.1)
    assert is_lot_multiple(0.00054, 0.00001)
    assert not is_lot_multiple(-0.05, 0.1)
    assert not is_lot_multiple(23.196474135931336, 1.0)


def test_grid_rounding_reaches_a_grid_no_decimal_precision_can_express():
    """int(abs(log10(step))) folds the sign, so lot 10/100 both round as if the grid were sub-unit."""
    assert grid_floor(157, 10) == 150.0
    assert grid_ceil(157, 10) == 160.0
    assert grid_floor(157, 100) == 100.0
    assert grid_ceil(157, 100) == 200.0
    assert grid_floor(150, 100) == 100.0
    assert grid_floor(37, 10) == 30.0
    # a grid no power of ten can reach at all
    assert grid_floor(13, 5) == 10.0
    assert grid_ceil(13, 5) == 15.0


def test_grid_rounding_on_a_half_decimal_step():
    """Kraken ticks 0.5/0.05 and Bybit 5e-5 all truncate to the next decade under prec_floor."""
    assert grid_floor(101.7, 0.5) == 101.5
    assert grid_ceil(101.7, 0.5) == 102.0
    assert grid_floor(1.234, 0.05) == 1.2
    assert grid_ceil(1.234, 0.05) == 1.25
    assert grid_floor(0.123456, 5e-05) == 0.12345


def test_grid_rounding_brackets_the_value_and_lands_on_the_grid():
    random.seed(11)
    for step in (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0):
        for _ in range(500):
            x = random.uniform(-1e4, 1e4)
            lo, hi = grid_floor(x, step), grid_ceil(x, step)
            assert is_lot_multiple(lo, step)
            assert is_lot_multiple(hi, step)
            assert abs(lo) <= abs(x) + step  # the noise snap may round |x| up by <1 tick
            assert abs(hi) >= abs(x) - step
            assert abs(hi) - abs(lo) <= step * 1.0000001


def test_grid_rounding_rounds_toward_zero_like_prec_rounding():
    assert grid_floor(-157, 10) == -150.0
    assert grid_ceil(-157, 10) == -160.0
    assert grid_floor(0.0, 10) == 0.0


def test_grid_floor_keeps_the_sub_lot_noise_snap():
    """prepare_ccxt_order_payload depends on this: a bare floor would send 0 and raise."""
    assert grid_floor(0.009999999999999998, 0.01) == 0.01
    assert grid_floor(0.29, 0.01) == 0.29
    assert grid_floor(0.1 + 0.2, 0.1) == 0.3
    assert grid_floor(1.1 * 3, 0.1) == 3.3


def test_a_value_a_hair_under_an_integer_step_snaps_up_to_the_grid():
    """An unsnapped return here is a whole tick down: ccxt TRUNCATEs 0.9999999999999999 to 0."""
    assert grid_floor(0.9999999999999999, 1.0) == 1.0
    assert grid_ceil(0.9999999999999999, 1.0) == 1.0
    assert grid_floor(29.999999999999996, 10.0) == 30.0
    assert grid_ceil(29.999999999999996, 10.0) == 30.0
    assert grid_floor(1234.9999999999998, 1.0) == 1235.0
    assert grid_ceil(1234.9999999999998, 1.0) == 1235.0


def test_an_on_grid_value_comes_back_bit_exact_on_a_power_of_ten_step():
    """Sub-unit powers of ten take the exact-reciprocal branch, so the multiply-back never moves them."""
    for step in (0.1, 0.01, 0.001, 1e-05, 1e-08):
        for ticks in (1, 3, 7, 29, 1234):
            x = prec_floor(ticks * step, int(abs(math.log10(step))))
            assert grid_floor(x, step) == x
            assert grid_ceil(x, step) == x


def test_a_decimal_step_with_no_exact_reciprocal_still_lands_on_the_decimal_grid():
    """Kraken ships lot 0.3 (22 pairs), 0.15 and 0.0075. 1/0.3 is not a whole number, so these
    miss the reciprocal branch; n * step would land an ulp low and ccxt truncates 0.3-lot
    0.8999999999999999 to 0.6 — two whole lots down."""
    assert grid_floor(0.9, 0.3) == 0.9
    assert grid_ceil(0.9, 0.3) == 0.9
    assert grid_floor(0.44999999999999996, 0.15) == 0.45
    assert grid_ceil(0.44999999999999996, 0.15) == 0.45
    assert grid_floor(9.254999999999999, 0.0075) == 9.255
    assert grid_ceil(9.254999999999999, 0.0075) == 9.255
    assert grid_floor(2.0999999999999996, 0.7) == 2.1
    assert grid_floor(0.9, 0.4) == 0.8
    assert grid_ceil(0.9, 0.4) == 1.2


@pytest.mark.parametrize("step", [0.3, 0.15, 0.0075, 0.06, 0.7, 0.9, 5.0, 2.5, 1.0, 10.0])
def test_every_returned_value_is_a_whole_number_of_steps(step):
    """The property the decimal decomposition buys: floor/ceil never hand back a fraction of a lot."""
    random.seed(3)
    for _ in range(2000):
        x = random.uniform(-1e4, 1e4)
        for v in (grid_floor(x, step), grid_ceil(x, step)):
            q = Decimal(repr(v)) / Decimal(repr(step))
            assert q == q.to_integral_value(), f"{v!r} is not a whole multiple of {step}"


def test_a_step_no_power_of_ten_resolves_degrades_instead_of_misbehaving():
    """No exact reciprocal and no whole numerator under the 1e12 bound: the divide-by-step path
    takes over, still bracketing rather than hanging or returning nan."""
    step = math.pi / 10
    for x in (0.9, 1.7, 123.456):
        lo, hi = grid_floor(x, step), grid_ceil(x, step)
        assert lo <= x <= hi
        assert hi - lo <= step * 1.0000001


@pytest.mark.parametrize("step", [0.0, -1.0])
def test_a_non_positive_step_raises(step):
    """precision.amount is None on broken venue metadata; sending the unrounded size is worse."""
    with pytest.raises(ValueError):
        grid_floor(1.23, step)
    with pytest.raises(ValueError):
        grid_ceil(1.23, step)


@pytest.mark.parametrize("power", range(9))
def test_grid_rounding_is_bit_identical_to_prec_rounding_on_a_power_of_ten_grid(power):
    """Every venue except Bybit/Kraken is on a 10^-k grid; those must not move by a single ulp.

    Bit-identity, not a tolerance: multiplying by an inexact 1e-3 instead of dividing by an exact
    1000.0 disagrees in the last ulp on ~19% of samples, and a relative tolerance would hide it.
    """
    random.seed(20260902 + power)
    step = 10.0**-power
    precision = int(abs(math.log10(step)))
    for i in range(4000):
        x = random.uniform(-1.0, 1.0) * (10.0 ** (i % 5 - 2))
        assert grid_floor(x, step) == prec_floor(x, precision)
        assert grid_ceil(x, step) == prec_ceil(x, precision)
