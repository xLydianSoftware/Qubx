"""Pins for the config -> AccountManager base-currency chain.

Covers _resolve_base_currency's priority order, the lower/mixed-case config
contract (AccountState upper-cases at the constructor, every consumer reads it
back from there), and paper-capital seeding on a canonicalized venue
(BINANCE.PM settings -> BINANCE.UM account state).
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.mixins.utils import canonical_exchange
from qubx.utils.runner.accounts import AccountConfigurationManager
from qubx.utils.runner.configs import ExchangeConfig, LiveConfig, LoggingConfig
from qubx.utils.runner.runner import _resolve_base_currency, _seed_paper_capital


class _Clock:
    def time(self):
        return np.datetime64("2026-06-01T00:00:00", "ns")


def _exchange_cfg(base_currency: str | None = None) -> ExchangeConfig:
    return ExchangeConfig(connector="ccxt", universe=["BTCUSDT"], base_currency=base_currency)


def _live_cfg(exchange_cfg: ExchangeConfig, base_currency: str | None = None) -> LiveConfig:
    return LiveConfig(
        exchanges={"BINANCE.UM": exchange_cfg},
        base_currency=base_currency,
        logging=LoggingConfig(logger="InMemoryLogsWriter"),
    )


def _accounts_toml(tmp_path: Path, body: str) -> AccountConfigurationManager:
    path = tmp_path / "accounts.toml"
    path.write_text(body)
    return AccountConfigurationManager(account_config=path)


def test_resolve_base_currency_priority_chain(tmp_path):
    mgr = _accounts_toml(tmp_path, '[[defaults]]\nexchange = "BINANCE.UM"\nbase_currency = "btc"\n')
    per_exchange = _exchange_cfg(base_currency="eth")
    assert _resolve_base_currency("BINANCE.UM", per_exchange, _live_cfg(per_exchange, "sol"), mgr) == "eth"
    no_exchange = _exchange_cfg()
    assert _resolve_base_currency("BINANCE.UM", no_exchange, _live_cfg(no_exchange, "sol"), mgr) == "sol"
    assert _resolve_base_currency("BINANCE.UM", no_exchange, _live_cfg(no_exchange), mgr) == "btc"


def test_resolve_base_currency_defaults_to_usdt_without_any_config(tmp_path):
    # The only sanctioned hardcoded USDT: the ExchangeSettings config-boundary default.
    mgr = AccountConfigurationManager(account_config=tmp_path / "missing.toml")
    cfg = _exchange_cfg()
    assert _resolve_base_currency("BINANCE.UM", cfg, _live_cfg(cfg), mgr) == "USDT"


def test_lowercase_config_value_reads_back_uppercased():
    am = SimulatedAccountManager(
        connectors={"BINANCE.UM": MagicMock()}, base_currencies={"BINANCE.UM": "usdt"}, time=_Clock()
    )
    assert am.get_base_currency("BINANCE.UM") == "USDT"
    assert am.get_base_currency() == "USDT"


def test_seed_paper_capital_binance_pm_lowercase_base(tmp_path):
    # BINANCE.PM is the configured venue (settings/credentials key); the AM state is keyed
    # by the canonical BINANCE.UM. Seeding must read the venue's initial_capital but land
    # on the canonical state under the upper-cased base currency, so total_capital sees it.
    mgr = _accounts_toml(
        tmp_path, '[[defaults]]\nexchange = "BINANCE.PM"\nbase_currency = "usdt"\ninitial_capital = 5000.0\n'
    )
    canonical = canonical_exchange("BINANCE.PM")
    assert canonical == "BINANCE.UM"
    base = _resolve_base_currency("BINANCE.PM", _exchange_cfg(), _live_cfg(_exchange_cfg()), mgr)
    assert base == "usdt"  # raw config value; the AccountState constructor owns normalization
    am = SimulatedAccountManager(connectors={canonical: MagicMock()}, base_currencies={canonical: base}, time=_Clock())

    _seed_paper_capital(am, canonical, "BINANCE.PM", mgr)

    assert am.get_balance("USDT", canonical).total == 5000.0
    assert am.get_total_capital(canonical) == 5000.0
