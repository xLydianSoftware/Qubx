from unittest.mock import MagicMock

import pytest

from qubx.core.mixins.processing import ProcessingManager
from tests.qubx.core.conftest import make_pm


def _pm_with_handlers() -> ProcessingManager:
    """make_pm plus everything the tuple path (__process_data -> _process_custom_event ->
    _run_strategy_pipeline) touches, mirroring _pm_for_tuple_path in test_processing_dispatch.py
    so process_data can drive a registered handler exactly the way post_event (Task 2) will.

    _custom_scheduled_methods carries a mutable class-level default (`{}` on the class body);
    make_pm's __new__ bypass of __init__ never shadows it per-instance, so leaving it alone
    (isinstance(..., dict) is already True) would mutate the SAME dict across every test/pm
    that doesn't set it explicitly -> duplicate-registration bleed between tests. Assign a
    fresh dict unconditionally instead.
    """
    pm = make_pm()
    pm._time_provider = MagicMock()
    pm._health_monitor = MagicMock()
    pm._subscription_manager = MagicMock()
    pm._cache = MagicMock()
    pm._strategy_name = "RegisterHandlerTest"
    pm._emitted_signals = []
    pm._data_throttler = None
    pm._handlers = {}
    pm._custom_scheduled_methods = {}
    pm._pending_no_quote_signals = {}
    pm._fit_is_running = False
    pm._warmup_finished_is_running = False
    pm._position_tracker.update.return_value = []
    pm._context.instruments = []
    pm._context._strategy_state.is_on_start_called = True
    pm._context._strategy_state.is_warmup_in_progress = False
    pm._context._strategy_state.is_on_warmup_finished_called = True
    pm._context._strategy_state.is_on_fit_called = True
    pm._strategy.on_market_data.return_value = None
    return pm


def test_register_handler_runs_method_on_process_data():
    pm = _pm_with_handlers()
    calls = []
    pm.register_handler("agg.sources", lambda ctx: calls.append(ctx))

    pm.process_data(None, "agg.sources", None, False)  # what post_event enqueues

    assert calls == [pm._context]
    pm._strategy.on_event.assert_not_called()
    pm._strategy.on_market_data.assert_not_called()


def test_register_handler_rejects_duplicate_and_empty_name():
    pm = _pm_with_handlers()
    pm.register_handler("agg.sources", lambda ctx: None)
    with pytest.raises(ValueError):
        pm.register_handler("agg.sources", lambda ctx: None)
    with pytest.raises(ValueError):
        pm.register_handler("", lambda ctx: None)


def test_registered_handler_exception_is_logged_not_raised():
    pm = _pm_with_handlers()

    def boom(ctx):
        raise RuntimeError("x")

    pm.register_handler("agg.sources", boom)
    pm.process_data(None, "agg.sources", None, False)  # must not raise (same as scheduled methods)
