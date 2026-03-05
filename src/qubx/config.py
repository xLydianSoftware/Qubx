"""Unified configuration management for Qubx.

Provides a single source of truth for all Qubx settings using pydantic-settings.
Configuration is loaded from (in priority order):
1. Environment variables (prefix QUBX_)
2. ~/.qubx/config.json

Usage:
    from qubx.config import settings, get_s3_account

    # Access settings
    print(settings.log_level)
    print(settings.s3)

    # Get a named S3 account
    account = get_s3_account("hetzner")
    print(account.endpoint_url)
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class S3Account(BaseModel):
    """S3-compatible storage account credentials."""

    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region: str | None = None


class LookupConfig(BaseModel):
    """Configuration for instrument/fees lookups."""

    type: str = "file"
    mongo_url: str | None = None
    reload_interval: str | None = None
    path: str | None = None


class _QubxJsonConfigSource(PydanticBaseSettingsSource):
    """Load settings from ~/.qubx/config.json."""

    _CONFIG_PATH: ClassVar[Path] = Path.home() / ".qubx" / "config.json"

    def __init__(self, settings_cls: type[BaseSettings], config_path: Path | None = None):
        super().__init__(settings_cls)
        self._path = config_path or self._CONFIG_PATH

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        data = self._load()
        value = data.get(field_name)
        return value, field_name, value is not None

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def __call__(self) -> dict[str, Any]:
        data = self._load()
        return {k: v for k, v in data.items() if v is not None}


class QubxSettings(BaseSettings):
    """Qubx framework settings.

    Resolution order:
    1. Environment variables (prefix QUBX_, nested with __)
    2. ~/.qubx/config.json
    3. Defaults
    """

    s3: dict[str, S3Account] = {}
    default_s3_account: str | None = None
    log_level: str = "WARNING"
    instrument_lookup: LookupConfig = LookupConfig()
    fees_lookup: LookupConfig = LookupConfig()

    model_config = {"env_prefix": "QUBX_", "env_nested_delimiter": "__"}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _QubxJsonConfigSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> QubxSettings:
    """Get the singleton QubxSettings instance."""
    return QubxSettings()


def get_s3_account(name: str) -> S3Account:
    """Look up a named S3 account from settings.

    Args:
        name: Account name (e.g., "hetzner")

    Returns:
        S3Account with credentials

    Raises:
        KeyError: If account not found
    """
    s = get_settings()
    if name not in s.s3:
        available = list(s.s3.keys()) or "(none)"
        raise KeyError(f"S3 account '{name}' not found in config. Available: {available}")
    return s.s3[name]


def reset_settings() -> None:
    """Clear the cached settings (useful for testing)."""
    get_settings.cache_clear()


# Module-level singleton
settings = get_settings()
