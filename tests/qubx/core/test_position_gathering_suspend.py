from unittest.mock import Mock

from qubx.core.interfaces import IPositionGathering


def test_default_gatherer_is_not_suspended():
    g = IPositionGathering()
    assert g.is_suspended is False


def test_default_suspend_resume_are_noops():
    g = IPositionGathering()
    ctx = Mock()
    g.suspend(ctx)  # must not raise
    g.resume(ctx)  # must not raise
    assert g.is_suspended is False
