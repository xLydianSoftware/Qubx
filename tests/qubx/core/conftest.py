"""Shared factories for core tests (also importable from subpackages, e.g. mixins)."""

from unittest.mock import MagicMock

from qubx.core.mixins.processing import ProcessingManager


def make_pm(**overrides) -> ProcessingManager:
    """ProcessingManager half-object for dispatch-path tests: real methods, mocked
    collaborators. Keyword overrides replace any attribute (e.g. ``_account_manager=real_am``).
    """
    pm = ProcessingManager.__new__(ProcessingManager)
    pm._is_simulation = True  # not paper: keeps _feed_simulated_connector a no-op
    pm._strategy = MagicMock()
    pm._account_manager = MagicMock()
    pm._context = MagicMock()
    pm._context.emitter = None
    pm._position_gathering = MagicMock()
    pm._exporter = None
    pm._universe_manager = MagicMock()
    pm._logging = MagicMock()
    pm._market_data = MagicMock()
    pm._position_tracker = MagicMock()
    pm._instruments_in_init_stage = set()
    pm._init_stage_position_tracker = MagicMock()
    pm._active_targets = {}
    for name, value in overrides.items():
        setattr(pm, name, value)
    return pm
