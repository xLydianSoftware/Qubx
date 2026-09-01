"""The runtime-info blob is written exactly once, under the same gate as the snapshot."""

from unittest.mock import Mock

from qubx.core.mixins.processing import ProcessingManager
from qubx.state import DummyStatePersistence


def _manager(persistence) -> ProcessingManager:
    # ProcessingManager.__init__ wires the whole trading context; the hook only
    # touches these attributes, so build a bare instance for a focused test.
    m = object.__new__(ProcessingManager)
    m._scheduler = Mock()
    m._is_simulation = False
    m._context = Mock()
    m._context.persistence = persistence
    m._exporter = None
    m._strategy = Mock()
    m._time_provider = Mock()
    m._time_provider.time = Mock(return_value="2026-09-01T09:00:00")
    m._runtime_info_saved = False
    return m


def test_writes_info_once_when_snapshot_enabled():
    persistence = Mock()
    m = _manager(persistence)

    m.configure_state_snapshot("5s")
    m.configure_state_snapshot("5s")  # reconfigure must not rewrite

    info_calls = [c for c in persistence.save.call_args_list if c.args[0] == "info"]
    assert len(info_calls) == 1
    blob = info_calls[0].args[1]
    assert blob["schema_version"] == 1
    assert "qubx" in blob


def test_skips_in_simulation_and_with_dummy_persistence():
    m = _manager(Mock())
    m._is_simulation = True
    m.configure_state_snapshot("5s")
    m._context.persistence.save.assert_not_called()

    m2 = _manager(DummyStatePersistence())
    m2.configure_state_snapshot("5s")
    assert m2._runtime_info_saved is False


def test_failed_write_does_not_raise_and_allows_retry():
    persistence = Mock()
    persistence.save.side_effect = RuntimeError("redis down")
    m = _manager(persistence)
    m.configure_state_snapshot("5s")  # must not raise
    assert m._runtime_info_saved is False
