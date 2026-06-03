import pandas as pd

from qubx.core.basics import Balance, Deal, Instrument, ITimeProvider, Position, dt_64
from qubx.core.utils import prec_ceil, prec_floor


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
            if d.id in seen:
                continue
            seen.append(d.id)
            pos.update_position_by_deal(d, conversion_rate=1)

    def get_total_capital(self, exchange: str | None = None) -> float:
        cash = self._balances[self.base_currency].total if self.base_currency in self._balances else 0.0
        return cash + sum(p.market_value_funds for p in self._positions.values())
