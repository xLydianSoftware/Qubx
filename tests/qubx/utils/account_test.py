from pathlib import Path

import pytest

from qubx.utils.runner.accounts import AccountConfigurationManager

CONFIGS_DIR = Path(__file__).parent / "configs"


def test_account_config_parsing():
    basic_toml = CONFIGS_DIR / "accounts.toml"
    manager = AccountConfigurationManager(basic_toml)

    # binance pm
    bpm = manager.get_exchange_credentials("BINANCE.PM")
    assert bpm.name == "bnc-pm-test1"
    assert bpm.api_key == "your_binance_pm_api_key"
    assert bpm.commissions is None

    # binance um
    bum = manager.get_exchange_credentials("BINANCE.UM")
    assert bum.name == "bnc-um-test1"
    assert bum.api_key == "your_binance_um_api_key"
    assert bum.base_currency == "USDT"
    assert bum.commissions is None
    assert bum.testnet is True

    # kraken futures
    krf = manager.get_exchange_credentials("KRAKEN.F")
    assert krf.name == "kr-f-test"
    assert krf.api_key == "your_kraken_f_api_key"
    assert krf.base_currency == "USD"
    assert krf.commissions == "K0"


def test_defaults_account_config_parsing():
    defaults_toml = CONFIGS_DIR / "defaults_accounts.toml"
    manager = AccountConfigurationManager(defaults_toml)

    settings = manager.get_exchange_settings("BINANCE.UM")
    assert settings.base_currency == "USDT"


def test_invalid_account_config():
    invalid_toml = CONFIGS_DIR / "invalid_accounts.toml"
    with pytest.raises(ValueError):
        AccountConfigurationManager(invalid_toml)
