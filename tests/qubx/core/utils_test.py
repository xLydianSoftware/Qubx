import pandas as pd

from qubx.core.basics import Balance, Deal, Instrument, ITimeProvider, Position, dt_64
from qubx.core.utils import add_in_lots, is_lot_multiple, prec_ceil, prec_floor


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
