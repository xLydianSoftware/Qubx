import pytest

from qubx.core.account_manager_config import AccountManagerConfig, _ms_to_cron


def test_default_config():
    cfg = AccountManagerConfig()
    assert cfg.inflight_check_interval_ms == 2_000
    assert cfg.snapshot_check_interval_ms == 30_000
    assert cfg.liveness_check_interval_ms == 5_000
    assert cfg.terminal_order_retention_ms == 30_000


def test_ms_to_cron_sub_minute():
    assert _ms_to_cron(2_000) == "* * * * * */2"
    assert _ms_to_cron(30_000) == "* * * * * */30"


def test_ms_to_cron_whole_minute():
    assert _ms_to_cron(60_000) == "*/1 * * * *"
    assert _ms_to_cron(300_000) == "*/5 * * * *"


def test_ms_to_cron_rejects_unrepresentable():
    with pytest.raises(ValueError):
        _ms_to_cron(1_500)
    with pytest.raises(ValueError):
        _ms_to_cron(90_000)
