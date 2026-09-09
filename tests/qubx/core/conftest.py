"""Shared factories for core tests (also importable from subpackages, e.g. mixins)."""

from unittest.mock import MagicMock

from qubx.core.boot import BootStateMachine
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
    pm._context.initializer.get_fit_on_start.return_value = False  # a MagicMock would read as the knob being on
    pm._boot = BootStateMachine(MagicMock())  # __new__ skips __init__, which builds it
    # _event_handlers is annotation-only on the class (no mutable class default), so __new__
    # leaves it UNSET and every dispatch-path test would AttributeError in
    # _process_custom_event. Seed it here, per-instance, like _boot above.
    pm._event_handlers = {}
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


def real_handler_map() -> dict:
    """The ``_handlers`` map exactly as ``ProcessingManager.__init__`` builds it.

    Tests that pin the register_handler shadow guard use this rather than a synthetic dict,
    so renaming/removing a ``_handle_*`` method breaks the test instead of silently leaving
    a name unguarded.
    """
    return {
        n.split("_handle_")[1]: f
        for n, f in ProcessingManager.__dict__.items()
        if callable(f) and n.startswith("_handle_")
    }
