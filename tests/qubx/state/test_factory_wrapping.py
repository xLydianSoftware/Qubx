from unittest.mock import patch

from qubx.state.dummy import DummyStatePersistence
from qubx.state.safe import SafeStatePersistence
from qubx.utils.runner.configs import StatePersistenceConfig
from qubx.utils.runner.factory import create_state_persistence


class InMemoryBackend:
    def __init__(self, strategy_name: str = "", **kwargs):
        self.store = {}
        self.probed = False

    def save(self, key, value):
        self.store[key] = value

    def load(self, key, default=None):
        return self.store.get(key, default)

    def delete(self, key):
        return self.store.pop(key, None) is not None

    def exists(self, key):
        self.probed = True
        return key in self.store


def test_real_backend_is_wrapped_and_validated():
    cfg = StatePersistenceConfig(type="RedisStatePersistence", parameters={}, snapshot_interval="5s")
    with patch("qubx.utils.runner.factory.class_import", return_value=InMemoryBackend):
        sp = create_state_persistence(cfg, "strat")
    assert isinstance(sp, SafeStatePersistence)
    assert sp.staleness_threshold_s == 60.0  # max(3*5s, 60s)
    sp.stop()


def test_threshold_scales_with_long_snapshot_interval():
    cfg = StatePersistenceConfig(type="RedisStatePersistence", parameters={}, snapshot_interval="1m")
    with patch("qubx.utils.runner.factory.class_import", return_value=InMemoryBackend):
        sp = create_state_persistence(cfg, "strat")
    assert sp.staleness_threshold_s == 180.0  # max(3*60s, 60s)
    sp.stop()


def test_dummy_backend_is_not_wrapped():
    cfg = StatePersistenceConfig(type="DummyStatePersistence", parameters={})
    with patch("qubx.utils.runner.factory.class_import", return_value=DummyStatePersistence):
        sp = create_state_persistence(cfg, "strat")
    assert isinstance(sp, DummyStatePersistence)


def test_none_config_returns_none():
    assert create_state_persistence(None, "strat") is None
