from pathlib import Path

import toml
from pydantic import BaseModel, ConfigDict, Field


class ExchangeSettings(BaseModel):
    """Base exchange settings with support for custom fields"""

    model_config = ConfigDict(extra="allow")  # Allow extra fields for exchange-specific config

    exchange: str
    testnet: bool = False
    base_currency: str = "USDT"
    commissions: str | None = None
    initial_capital: float = 100_000

    def get_extra_field(self, key: str, default=None):
        """Get an exchange-specific extra field"""
        if hasattr(self, "__pydantic_extra__"):
            return self.__pydantic_extra__.get(key, default)
        return default


class ExchangeCredentials(ExchangeSettings):
    """Exchange credentials with support for exchange-specific fields

    Standard fields:
        - name: Account identifier
        - api_key: API key or address (e.g., Ethereum address for Lighter)
        - secret: Secret key or private key

    Exchange-specific fields (accessed via model_extra):
        - account_index: For Lighter exchange
        - api_key_index: For Lighter exchange
        - private_key: Alternative to secret for some exchanges
    """

    name: str
    api_key: str
    secret: str

    def get_extra_field(self, key: str, default=None):
        """Get an exchange-specific extra field"""
        if hasattr(self, "__pydantic_extra__"):
            return self.__pydantic_extra__.get(key, default)
        return default


class AccountConfiguration(BaseModel):
    defaults: list[ExchangeSettings] = Field(default_factory=list)
    accounts: list[ExchangeCredentials] = Field(default_factory=list)


class AccountConfigurationManager:
    """
    Manages account configurations.
    """

    def __init__(
        self,
        account_config: Path | None = None,
        strategy_dir: Path | None = None,
        search_qubx_dir: bool = False,
    ):
        self._exchange_settings: dict[str, ExchangeSettings] = {}
        self._exchange_credentials: dict[str, ExchangeCredentials] = {}
        self._settings_to_config: dict[str, Path] = {}
        self._credentials_to_config: dict[str, Path] = {}

        self._config_paths = [Path("~/.qubx/accounts.toml").expanduser()] if search_qubx_dir else []
        if strategy_dir:
            self._config_paths.append(strategy_dir / "accounts.toml")
        # QUBX_ACCOUNT_FILE from unified settings (env var or config.json)
        from qubx.config import settings
        if settings.account_file:
            self._config_paths.append(Path(settings.account_file).expanduser())
        if account_config:
            self._config_paths.append(account_config)
        self._config_paths = [config for config in self._config_paths if config.exists()]
        for config in self._config_paths:
            self._load(config)

    def get_exchange_settings(self, exchange: str) -> ExchangeSettings:
        """
        Get the basic settings for an exchange such as the base currency and commission tier.

        Priority: [[accounts]] credentials > [[defaults]] > hardcoded defaults.
        Credentials are more specific (the actual trading account) so they take precedence.
        """
        exchange = exchange.upper()
        # Credentials take priority — they represent the actual account being used
        if exchange in self._exchange_credentials:
            creds = self._exchange_credentials[exchange]
            return ExchangeSettings(
                exchange=creds.exchange,
                testnet=creds.testnet,
                base_currency=creds.base_currency,
                commissions=creds.commissions,
                initial_capital=creds.initial_capital,
            )
        if exchange in self._exchange_settings:
            return self._exchange_settings[exchange].model_copy()
        return ExchangeSettings(exchange=exchange)

    def get_exchange_credentials(self, exchange: str) -> ExchangeCredentials:
        """
        Get the api key and secret for an exchange as well as the base currency and commission tier.
        """
        return self._exchange_credentials[exchange.upper()].model_copy()

    def get_config_path_for_settings(self, exchange: str) -> Path:
        return self._settings_to_config[exchange.upper()]

    def get_config_path_for_credentials(self, exchange: str) -> Path:
        return self._credentials_to_config[exchange.upper()]

    def __repr__(self):
        exchanges = set(self._exchange_credentials.keys()) | set(self._exchange_settings.keys())
        _e_str = "\n".join([f" - {exchange}" for exchange in exchanges])
        return f"AccountManager:\n{_e_str}"

    def _load(self, config: Path):
        config_dict = toml.load(config)
        account_config = AccountConfiguration(**config_dict)
        for exchange_config in account_config.defaults:
            _exchange = exchange_config.exchange.upper()
            self._exchange_settings[_exchange] = exchange_config
            self._settings_to_config[_exchange] = config
        for exchange_config in account_config.accounts:
            _exchange = exchange_config.exchange.upper()
            self._exchange_credentials[_exchange] = exchange_config
            self._credentials_to_config[_exchange] = config
