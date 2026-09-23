import json
import os

import pytest

from qubx.config import QubxSettings, _QubxJsonConfigSource, get_settings


@pytest.fixture
def qubx_config(tmp_path, monkeypatch):
    """An isolated `~/.qubx/config.json` and no QUBX_* env; the settings cache is cleared around the test."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(_QubxJsonConfigSource, "_CONFIG_PATH", path)
    for name in list(os.environ):
        if name.startswith("QUBX_"):
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


def test_settings_load_a_config_file_with_an_iceberg_block(qubx_config):
    qubx_config.write_text(json.dumps({"iceberg": {"r2": {"uri": "u", "warehouse": "w", "token": "t"}}}))
    account = QubxSettings().iceberg["r2"]
    assert (account.uri, account.warehouse, account.token, account.namespace_prefix) == ("u", "w", "t", "")


def test_iceberg_account_fields_come_from_env(qubx_config, monkeypatch):
    monkeypatch.setenv("QUBX_ICEBERG__R2__TOKEN", "from-env")
    monkeypatch.setenv("QUBX_ICEBERG__R2__NAMESPACE_PREFIX", "scratch__")
    account = QubxSettings().iceberg["r2"]
    assert (account.token, account.namespace_prefix, account.uri) == ("from-env", "scratch__", None)
