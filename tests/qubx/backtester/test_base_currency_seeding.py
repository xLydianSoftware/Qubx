"""Pins for backtester base-currency normalization and capital seeding.

Regression for the ignored-wallet bug: SimulationSetup.base_currency reached the
runner raw, so simulate(..., base_currency="usdt") seeded the Balance under "usdt"
while AccountState.base_currency was "USDT" — total_capital() read 0 and the whole
seeded capital was ignored. SimulationSetup now upper-cases at the config boundary.
"""

from unittest.mock import MagicMock

import numpy as np

from qubx.backtester.utils import SetupTypes, SimulationSetup
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Balance


class _Clock:
    def time(self):
        return np.datetime64("2026-06-01T00:00:00", "ns")


def _setup(base_currency: str) -> SimulationSetup:
    return SimulationSetup(
        setup_type=SetupTypes.STRATEGY,
        name="test",
        generator=None,
        tracker=None,
        instruments=[],
        exchanges=["BINANCE.UM"],
        capital=10_000.0,
        base_currency=base_currency,
    )


def test_simulation_setup_uppercases_base_currency():
    assert _setup("usdt").base_currency == "USDT"


def test_lowercase_base_currency_capital_not_ignored():
    # Mirrors BacktestRunner._create_account_manager: AM base currencies and the seeded
    # Balance both come from setup.base_currency. With a lowercase input the seeded
    # capital must still be visible to total_capital (the derived base-cash leg).
    setup = _setup("usdt")
    am = SimulatedAccountManager(
        connectors={ex: MagicMock() for ex in setup.exchanges},
        base_currencies={ex: setup.base_currency for ex in setup.exchanges},
        time=_Clock(),
    )
    assert isinstance(setup.capital, dict)
    for exchange, capital in setup.capital.items():
        am.seed_balance(
            exchange,
            Balance(exchange=exchange, currency=setup.base_currency, total=capital, free=capital, locked=0.0),
        )

    assert am.get_base_currency("BINANCE.UM") == "USDT"
    assert am.get_balance("USDT", "BINANCE.UM").total == 10_000.0
    assert am.get_total_capital("BINANCE.UM") == 10_000.0
