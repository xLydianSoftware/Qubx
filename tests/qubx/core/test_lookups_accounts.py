"""Tests for AccountsLookup functionality in LookupsManager."""

from unittest.mock import MagicMock

import pytest

from qubx.core.lookups import LookupsManager, register_accounts
from qubx.utils.runner.accounts import ExchangeCredentials, ExchangeSettings


@pytest.fixture
def manager():
    lm = LookupsManager()
    lm._a_lookup._manager = None  # reset between tests
    return lm


@pytest.fixture
def mock_account_manager():
    am = MagicMock()
    am.get_exchange_credentials.return_value = ExchangeCredentials(
        name="test",
        exchange="LIGHTER",
        api_key="test-key",
        secret="test-secret",
    )
    am.get_exchange_settings.return_value = ExchangeSettings(
        exchange="LIGHTER",
        testnet=True,
    )
    return am


def test_get_credentials_not_registered(manager):
    """Raises RuntimeError when no account manager registered."""
    with pytest.raises(RuntimeError, match="No account manager registered"):
        manager.get_credentials("LIGHTER")


def test_get_settings_not_registered(manager):
    """Raises RuntimeError when no account manager registered."""
    with pytest.raises(RuntimeError, match="No account manager registered"):
        manager.get_settings("LIGHTER")


def test_register_and_get_credentials(manager, mock_account_manager):
    """register_accounts() + lookup.get_credentials('LIGHTER') returns ExchangeCredentials."""
    register_accounts(mock_account_manager)

    creds = manager.get_credentials("LIGHTER")

    assert creds.api_key == "test-key"
    assert creds.secret == "test-secret"
    mock_account_manager.get_exchange_credentials.assert_called_once_with("LIGHTER")


def test_register_and_get_settings(manager, mock_account_manager):
    """register_accounts() + lookup.get_settings('LIGHTER') returns ExchangeSettings."""
    register_accounts(mock_account_manager)

    settings = manager.get_settings("LIGHTER")

    assert settings.testnet is True
    mock_account_manager.get_exchange_settings.assert_called_once_with("LIGHTER")
