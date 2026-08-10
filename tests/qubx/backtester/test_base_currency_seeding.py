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
from qubx.core.basics import Balance, Instrument, MarketType


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
