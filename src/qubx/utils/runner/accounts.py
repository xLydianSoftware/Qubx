from pathlib import Path

import toml
from pydantic import BaseModel, Field


class ExchangeSettings(BaseModel):
    exchange: str
    testnet: bool = False
    base_currency: str = "USDT"
    commissions: str | None = None
    initial_capital: float = 100_000


class ExchangeCredentials(ExchangeSettings):
    name: str
    api_key: str
    secret: str


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
        if account_config:
            self._config_paths.append(account_config)
        self._config_paths = [config for config in self._config_paths if config.exists()]
        for config in self._config_paths:
            self._load(config)

    def get_exchange_settings(self, exchange: str) -> ExchangeSettings:
        """
        Get the basic settings for an exchange such as the base currency and commission tier.
        """
        exchange = exchange.upper()
        if exchange not in self._exchange_settings:
            return ExchangeSettings(exchange=exchange)
        return self._exchange_settings[exchange.upper()].model_copy()

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
