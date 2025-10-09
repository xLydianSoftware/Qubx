"""Tests for account configuration with exchange-specific fields"""
import tempfile
from pathlib import Path

from qubx.utils.runner.accounts import (
    AccountConfigurationManager,
    ExchangeCredentials,
    ExchangeSettings,
)


class TestExchangeSettings:
    """Test ExchangeSettings with extra fields"""

    def test_basic_settings(self):
        """Test basic exchange settings"""
        settings = ExchangeSettings(exchange="BINANCE.UM")
        assert settings.exchange == "BINANCE.UM"
        assert settings.testnet is False
        assert settings.base_currency == "USDT"
        assert settings.initial_capital == 100_000

    def test_settings_with_overrides(self):
        """Test settings with custom values"""
        settings = ExchangeSettings(
            exchange="LIGHTER", testnet=True, base_currency="USDC", initial_capital=50_000
        )
        assert settings.exchange == "LIGHTER"
        assert settings.testnet is True
        assert settings.base_currency == "USDC"
        assert settings.initial_capital == 50_000

    def test_settings_with_extra_fields(self):
        """Test that extra fields are accepted"""
        settings = ExchangeSettings(
            exchange="LIGHTER", base_currency="USDC", custom_field="custom_value", another_field=123
        )
        assert settings.exchange == "LIGHTER"
        # Extra fields should be stored
        assert settings.get_extra_field("custom_field") == "custom_value"
        assert settings.get_extra_field("another_field") == 123


class TestExchangeCredentials:
    """Test ExchangeCredentials with exchange-specific fields"""

    def test_basic_credentials(self):
        """Test basic credentials"""
        creds = ExchangeCredentials(
            name="test-account", exchange="BINANCE.UM", api_key="test_key", secret="test_secret"
        )
        assert creds.name == "test-account"
        assert creds.exchange == "BINANCE.UM"
        assert creds.api_key == "test_key"
        assert creds.secret == "test_secret"

    def test_lighter_specific_fields(self):
        """Test Lighter-specific fields (account_index, api_key_index)"""
        creds = ExchangeCredentials(
            name="lighter-account",
            exchange="LIGHTER",
            api_key="0xAddress",
            secret="0xPrivateKey",
            account_index=225671,
            api_key_index=2,
        )
        assert creds.name == "lighter-account"
        assert creds.exchange == "LIGHTER"
        assert creds.api_key == "0xAddress"
        assert creds.secret == "0xPrivateKey"

        # Check extra fields
        assert creds.get_extra_field("account_index") == 225671
        assert creds.get_extra_field("api_key_index") == 2

    def test_get_extra_field_with_default(self):
        """Test getting extra field with default value"""
        creds = ExchangeCredentials(
            name="test", exchange="BINANCE.UM", api_key="key", secret="secret", custom=123
        )
        assert creds.get_extra_field("custom") == 123
        assert creds.get_extra_field("missing_field", "default_value") == "default_value"
        assert creds.get_extra_field("missing_field") is None


class TestAccountConfigurationManager:
    """Test AccountConfigurationManager with TOML files"""

    def test_load_lighter_config(self):
        """Test loading Lighter configuration from TOML"""
        # Create temporary TOML file with Lighter config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[[defaults]]
exchange = "LIGHTER"
base_currency = "USDC"
commissions = "maker=0.0,taker=0.0"

[[accounts]]
name = "xlydian1-lighter"
exchange = "LIGHTER"
api_key = "0xd00BEc5Fd22A898e85481D57AE7b24C98121eEFb"
secret = "0x7a4654fc036ee7d23f3088c03bbda717fbc5282cd36dffb387710f878cdc6548"
account_index = 225671
api_key_index = 2
base_currency = "USDC"
commissions = "maker=0.0,taker=0.0"
""")
            temp_path = Path(f.name)

        try:
            # Load configuration
            manager = AccountConfigurationManager(account_config=temp_path)

            # Check settings
            settings = manager.get_exchange_settings("LIGHTER")
            assert settings.exchange == "LIGHTER"
            assert settings.base_currency == "USDC"

            # Check credentials
            creds = manager.get_exchange_credentials("LIGHTER")
            assert creds.name == "xlydian1-lighter"
            assert creds.api_key == "0xd00BEc5Fd22A898e85481D57AE7b24C98121eEFb"
            assert creds.get_extra_field("account_index") == 225671
            assert creds.get_extra_field("api_key_index") == 2

        finally:
            # Clean up
            temp_path.unlink()

    def test_load_multiple_exchanges(self):
        """Test loading configuration for multiple exchanges"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[[accounts]]
name = "binance-account"
exchange = "BINANCE.UM"
api_key = "binance_key"
secret = "binance_secret"

[[accounts]]
name = "lighter-account"
exchange = "LIGHTER"
api_key = "0xAddress"
secret = "0xPrivateKey"
account_index = 12345
api_key_index = 1
""")
            temp_path = Path(f.name)

        try:
            manager = AccountConfigurationManager(account_config=temp_path)

            # Check Binance
            binance_creds = manager.get_exchange_credentials("BINANCE.UM")
            assert binance_creds.name == "binance-account"
            assert binance_creds.api_key == "binance_key"
            assert binance_creds.get_extra_field("account_index") is None  # No extra fields

            # Check Lighter
            lighter_creds = manager.get_exchange_credentials("LIGHTER")
            assert lighter_creds.name == "lighter-account"
            assert lighter_creds.get_extra_field("account_index") == 12345
            assert lighter_creds.get_extra_field("api_key_index") == 1

        finally:
            temp_path.unlink()

    def test_case_insensitive_exchange_lookup(self):
        """Test that exchange lookup is case-insensitive"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[[accounts]]
name = "test"
exchange = "lighter"
api_key = "key"
secret = "secret"
""")
            temp_path = Path(f.name)

        try:
            manager = AccountConfigurationManager(account_config=temp_path)

            # Should work with any case
            creds_lower = manager.get_exchange_credentials("lighter")
            creds_upper = manager.get_exchange_credentials("LIGHTER")
            creds_mixed = manager.get_exchange_credentials("Lighter")

            assert creds_lower.name == "test"
            assert creds_upper.name == "test"
            assert creds_mixed.name == "test"

        finally:
            temp_path.unlink()
