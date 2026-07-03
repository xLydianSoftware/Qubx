"""Regression test for the on-demand fit chain.

trigger_fit() schedules a one-off delayed method that must run the fit on the
strategy thread. The scheduler invokes that method with the *context* (see
ProcessingManager._process_custom_event -> method(self._context)), and the
composition-based StrategyContext has no _handle_fit (it is a ProcessingManager
method, not a context method). The delayed method must therefore bind
_handle_fit on the processing manager itself, never route through the context.
"""

from unittest.mock import MagicMock

import numpy as np

from tests.qubx.core.conftest import make_pm


def test_trigger_fit_schedules_one_off_delay():
    pm = make_pm(_scheduler=MagicMock())
    captured: dict[str, object] = {}

    def fake_delay(duration, method):
        captured["duration"] = duration
        captured["method"] = method
        return "delay_x"

    pm.delay = fake_delay  # type: ignore[method-assign]

    pm.trigger_fit()

    assert "method" in captured, "trigger_fit must schedule a delayed method"
    assert captured["duration"]  # non-zero delay defers onto the strategy thread


def test_delayed_fit_method_invokes_handle_fit_on_manager_not_context():
    # The crux: the delayed method receives the context (which lacks _handle_fit),
    # yet must drive _handle_fit on the processing manager.
    pm = make_pm(_scheduler=MagicMock(), _time_provider=MagicMock())
    pm._time_provider.time.return_value = np.datetime64("2025-01-01T00:00:00")
    pm._handle_fit = MagicMock()

    captured: dict[str, object] = {}
    pm.delay = lambda duration, method: captured.setdefault("method", method)  # type: ignore[method-assign]

    pm.trigger_fit()
    delayed_method = captured["method"]

    # Scheduler passes the context (no _handle_fit attribute) — mirror that exactly.
    context_without_handle_fit = object()
    delayed_method(context_without_handle_fit)

    pm._handle_fit.assert_called_once()
    args = pm._handle_fit.call_args.args
    assert args[0] is None  # instrument
    assert args[1] == "fit"  # event type
    assert args[2] == (None, pm._time_provider.time.return_value)


def test_process_custom_event_runs_delayed_fit_end_to_end():
    # Full path: delay() registers the method, _process_custom_event invokes it
    # with self._context, _handle_fit fires on the manager.
    pm = make_pm(_scheduler=MagicMock(), _time_provider=MagicMock())
    pm._time_provider.time.return_value = np.datetime64("2025-01-01T00:00:00")
    pm._handle_fit = MagicMock()
    pm._custom_scheduled_methods = {}

    event_id = pm.delay("1s", lambda c: pm._handle_fit(None, "fit", (None, pm._time_provider.time())))
    assert event_id in pm._custom_scheduled_methods

    pm._process_custom_event(None, event_id, None)

    pm._handle_fit.assert_called_once_with(None, "fit", (None, pm._time_provider.time.return_value))
